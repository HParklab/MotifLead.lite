import os
import sys
import numpy as np
import copy
from myutils import get_AAtype_properties, read_pdb, ALL_AAS, findAAindex, find_gentype2num, get_chain_SS3, sasa_from_xyz
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

def featurize_target_properties(pdb, inputpath, outf, store_npz=True, extra={}, verbose=False): # -> save *.prop.npz

    # (AA: 20, NA: 5, Metal: 7)
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = get_AAtype_properties()
    
    # Parsing PDB file
    resnames, reschains, xyz, _ = read_pdb('%s/%s'%(inputpath, pdb), read_ligand=True)

    if len(xyz) == 0:
        return False

    # read in only heavy + hpol atms as lists
    atypes_rec, xyz_rec, atmres_rec, aas_rec, residue_idx, reschains_rec, qs_rec = [],[],[],[],[],[],[]
    atmnames, resnames_read, is_repsatm = [], [], []
    
    # length: residue number
    bnds_rec = []
    nresatm = {}
    for i, (resname, reschain) in enumerate(zip(resnames, reschains)):
        resi, resnum = reschain.split(".")
        if resname == ligname: #TODO
            iaa = 0
            qs, atypes, atms, bnds_, repsatm = ligand_feature()
        elif resname in ALL_AAS:
            iaa = findAAindex(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            print("unknown residue: %s, skip"%resname)
            continue

        natm = len(xyz_rec)
        atms_r = []

        for iatm, atm in enumerate(atms):
            if atm not in xyz[reschain]:
                continue
    
            atms_r.append(atm)
            atypes_rec.append(atypes[iatm])
            qs_rec.append(qs[atm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain,atm))
            is_repsatm.append((iatm == repsatm))
            reschains_rec.append(reschain.replace('.',''))
            residue_idx.append(i)
            
        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]
        
        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                print("Warning, abnormal bond distance: ", inputpath, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)

        bnds = np.array(bnds,dtype=int)
        atmnames.append(atms_r)
        resnames_read.append(resname)
        nresatm[reschain] = len(atms_r)
        
        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])

    xyz = np.array(xyz)

    atypes_rec = [find_gentype2num(a) for a in atypes_rec] #TODO
    sasa, nsasa, _ = sasa_from_xyz( xyz_rec, elems )
    
    atmnames = np.concatenate(atmnames)
    
    if store_npz:
        np.savez(outf,
                 # per-atm
                 aas_rec=aas_rec, #int
                 xyz=xyz, #np.array
                 atypes_rec=atypes_rec, #int
                 bnds_rec=bnds_rec, #list of [(i,j), ...]
                 sasa_rec=nsasa, #np.array
                 qs_rec=qs_rec,
                 SS3=SS3,
                 bb_dihe=bb_dihe,
                 is_repsatm=is_repsatm,
                 
                 # auxiliary -- lists
                 residue_idx=residue_idx,
                 reschains=reschains_rec,
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

    pdb,mol2 = input
    if inputpath[-1] != '/': inputpath+='/'
    if outprefix == None:
        outprefix = pdb.replace('/','.')

    if outpath == None: outpath = '/'.join(pdb.split('/')[:-1])
    if outpath == '': outpath = './'
    if not os.path.exists(outpath): os.mkdir(outpath)

    outpath = './'
    outf = './%s.prop.npz'%outprefix #(outpath,outprefix)
    if os.path.exists(outf):
        print("exist and pass:", outf)
        return
    if not os.path.exists('%s/%s'%(inputpath, pdb)): return
    
    if verbose:
        print(f'save {pdb} prop at {outf}')

    ligand_feat = read_ligand_mol2(mol2) #atomwise info
    status = featurize_properties(pdb, ligand_feat, inputpath,
                                  outf,
                                  verbose=verbose)
    if not status:
        print("skip ", pdb)

def main_multi(nproc):
    a = mp.Pool(processes=nproc)
    if sys.argv[1].endswith('.pdb'):
        inputs = [(sys.argv[1], sys.argv[2])]
    else:
         pdbs  = [l[:-1] for l in open(sys.argv[1])]
         mol2s = [l[:-1] for l in open(sys.argv[2])]
         inputs = [(p,m) for p,m in zip(pdbs,mol2s)]

    a.map(main, inputs)

if __name__ == "__main__":
    NP=20
    main_multi(NP)
