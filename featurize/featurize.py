import os
import sys
import numpy as np
import copy
from myutils import get_AAtype_properties, read_pdb, read_mol2, findAAindex, find_atype, sasa_from_xyz, atype2elem
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

def one_hot_encode(data):
    unique_atoms = ['C', 'H', 'E']
    atom_to_index = {atom: index for index, atom in enumerate(unique_atoms)}
    num_atoms = len(unique_atoms)
    one_hot_encodings = []

    for atom in data:
        encoding = np.zeros(num_atoms)
        encoding[atom_to_index[atom]] = 1
        one_hot_encodings.append(encoding)

    return np.array(one_hot_encodings)


def find_neighbors(xyz, ligname, reschains, resnames):
    ligrc = reschains[resnames.index(ligname)]
    
    xyz_lig = xyz[ligrc]

    neighbors = []
    for rc in reschains:
        if rc == ligrc:
            neighbors.append(rc)
        else:
            xyz_ca = xyz[rc]['CA']
            dx = np.array(list(xyz_lig.values())) - xyz_ca
            d2 = np.sum(dx*dx,axis=1)
            if (d2<400.0).sum() > 0: neighbors.append(rc) # any ligatm < 20.0 Ang
    return neighbors

def report_pdb(outf, reschains, xyz, atmnames, residx):
    out = open(outf,'w')
    iatm = 0
    #ATOM      1  N   LYS A 208      43.052  17.088  47.886  1.00  0.00           N
    form = 'ATOM  %-5d  %-3s%4s %1s %3d    %8.3f%8.3f%8.3f  1.00  0.00\n'
    for rc,x,at,ri in zip(reschains,xyz,atmnames,residx):
        iatm += 1
        rs,c = rc[1:],rc[0]
        out.write(form%(iatm,at,rs,c,ri+1,x[0],x[1],x[2]))
    out.close()

def featurize_properties(pdb, ligname, inputpath,
                         outf, ligand_feat,
                         keep_H=False,
                         store_npz=True, verbose=False): # -> save *.prop.npz

    qs_aa, atypes_aa, atms_aa, bnds_aa, _ = get_AAtype_properties()
    
    # Parsing PDB file
    resnames_, reschains_, xyz_, _ = read_pdb('%s/%s'%(inputpath, pdb), read_ligand=True)

    if len(xyz_) == 0:
        return False

    # read in only heavy + hpol atms as lists
    atypes, xyz, atmres, aas, residue_idx, reschains, qs = [],[],[],[],[],[],[]
    atmnames, resnames_read = [], []

    # find neighboring residues only
    neighbors = find_neighbors(xyz_, ligname, reschains_, resnames_)
    
    # length: residue number
    bnds = []
    nresatm = {}
    for i, (resname, reschain) in enumerate(zip(resnames_, reschains_)):
        resi, resnum = reschain.split(".")
        if reschain not in neighbors: continue
        
        if resname == ligname:
            qs_, atypes_ = ligand_feat['qs'], ligand_feat['atypes']
            atms_, bnds_ = ligand_feat['atms'], ligand_feat['bnds']
            iaa = 0 # ligand AA type
        else:
            iaa = findAAindex(resname)
            if iaa == -1:
                print("unknown residue: %s, skip"%resname)
                continue
            qs_, atypes_, atms_, bnds_ = qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa]

        natm = len(xyz)
        atms_r = []

        for iatm, atm in enumerate(atms_):
            if atm not in xyz_[reschain]:
                continue
            if not keep_H and atypes_[iatm] in [23,24]: continue

            atms_r.append(atm)
            atypes.append(atypes_[iatm])
            qs.append(qs_[atm])
            aas.append(iaa)
            xyz.append(xyz_[reschain][atm])
            
            atmres.append((reschain,atm))
            reschains.append(reschain.replace('.',''))
            residue_idx.append(i)

        if len(bnds_) > 0:
            bnds_ = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]
        
        # make sure all bonds are right
        #print(resname, len(bnds_), len(xyz_[reschain]))
        for (i1,i2) in copy.copy(bnds_):
            dv = np.array(xyz[i1+natm]) - np.array(xyz[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.3:
                print("Warning, abnormal bond distance: ", inputpath, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)

        bnds_ = np.array(bnds_,dtype=int)
        atmnames.append(atms_r)
        resnames_read.append(resname)
        nresatm[reschain] = len(atms_r)
        
        if i == 0:
            bnds = bnds_
        elif bnds_ != []:
            bnds_ += natm
            bnds = np.concatenate([bnds,bnds_])

    xyz = np.array(xyz)

    elems = [atype2elem(at) for at in atypes]
    atypes_rec = [find_atype(a) for a in atypes] #TODO
    sasa, nsasa, _ = sasa_from_xyz( xyz, elems )
    
    atmnames = np.concatenate(atmnames)

    #report_pdb(outf+'.pdb',reschains,xyz,atmnames,residue_idx)
    
    if store_npz:
        np.savez(outf,
                 # per-atm
                 aas=aas, #int
                 xyz=xyz, #np.array
                 atypes=atypes, #int
                 bnds=bnds, #list of [(i,j), ...]
                 sasa=nsasa, #np.array; normalized SASA
                 qs=qs,
                 
                 # auxiliary -- lists
                 residue_idx=residue_idx,
                 reschains=reschains,
                 atmnames=atmnames,
                 resnames=resnames_read,
        )
        return True
    else:
        return aas_rec, xyz_rec, atypes_rec, reschains, atmnames #unused with few exceptions

def main(input,
         verbose=False,  # tag = 'T01'
         out=sys.stdout,
         inputpath = './', #/ml/MotifLead/raw/PDBentropy/pdbs/',
         outpath = None,
         outprefix = None):

    pdb,mol2,ligname = input
    if inputpath[-1] != '/': inputpath+='/'
    if outprefix == None:
        outprefix = pdb.replace('/','.')#.replace('.pdb','')

    if outpath == None: outpath = '/'.join(pdb.split('/')[:-1])
    if outpath == '': outpath = './'
    if not os.path.exists(outpath): os.mkdir(outpath)

    outpath = './'
    outf = './%s.feat.npz'%outprefix #(outpath,outprefix)
    if os.path.exists(outf):
        print("exist and pass:", outf)
        return
    if not os.path.exists('%s/%s'%(inputpath, pdb)): return
    
    if verbose:
        print(f'save {pdb} prop at {outf}')

    ligand_feat = read_mol2(mol2) #atomwise info

    status = featurize_properties(pdb, ligname,
                                  inputpath,
                                  outf,
                                  ligand_feat,
                                  verbose=verbose)

    
    
    if not status:
        print("skip ", pdb)

def main_multi(nproc):
    a = mp.Pool(processes=nproc)
    pdbs  = [l[:-1] for l in open(sys.argv[1])]
    mol2s = [l[:-1] for l in open(sys.argv[2])]
    lignames = [l.split('.')[0] for l in open(sys.argv[2])] # make sure mol2 names match ligand name
    inputs = [(p,m,l) for p,m in zip(pdbs,mol2s,lignames)]

    a.map(main, inputs)

if __name__ == "__main__":
    pdb = sys.argv[1]
    mol2 = sys.argv[2]
    ligname = sys.argv[3]
    main((pdb, mol2, ligname), outpath='', outprefix=pdb.split('/')[-1][:-4])
