import numpy as np
import os
import sys
import glob
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdchem
from MyClasses import BondInfo, EdgeFeat, NodeOneHot, NodeFeat, GenPotFeat

ELEMS_TYPE = ['C','O','N','P','S','Br','I','F','Cl']

NP = 40
SKIP_EXIST= True

'''
def CountMatches(mol,patt,unique=True):
    return mol.GetSubstructMatches(patt,uniquify=unique)
    
def LoadPatterns(fileName=None):
    fns=[]
    check_list: list =['fr_lactone','fr_amide','fr_ester','fr_C=S','fr_phos_acid','fr_alkyl_carbamate','fr_urea','fr_sulfone','fr_sulfone2','fr_ketone','fr_ether','fr_Al_OH','fr_Ar_OH']
    defaultPatternFileName: str =('/home/kathy531/Caesar-lig/code/rdkit/FragmentsDescriptors.csv')
    if fileName is None:
        fileName = defaultPatternFileName
    try:
        inF = open(fileName,'r')
    except IOError:
        raise IOError
    else:
        for line in inF.readlines():
            if len(line) and line[0] != '#':
                splitL = line.split('\t')
                if len(splitL)>=3:
                    name = splitL[0]
                    if name not in check_list: continue
                    descr = splitL[1]
                    sma = splitL[2]
                    descr=descr.replace('"','')
                    ok=1
                    try:
                        patt = Chem.MolFromSmarts(sma) #smart -> mol
                    except:
                        ok=0
                    else:
                        if not patt or patt.GetNumAtoms()==0: ok=0
                    if not ok: raise ImportError#'Smarts %s could not be parsed'%(repr(sma))
                    fn = lambda mol,countUnique=True,pattern=patt:CountMatches(mol,pattern,unique=countUnique)
                    fn.__doc__ = descr
                    name = name.replace('=','_')
                    name = name.replace('-','_')
                    fns.append((name,fn))
    return fns
    
def FuncG_rev(fns,mol):
    func_dict={}
    known_atoms=[]
    if fns is not None:
        for name, fn in fns:
            if name not in func_dict:
                func_dict[name]=[]
            for i in fn(mol):
                for j in i:
                    func_dict[name].append(j)
                    known_atoms.append(j)
    total_atoms = [idx for idx in range(mol.GetNumAtoms())]
    unknown_atoms = [idx for idx in total_atoms if idx not in known_atoms]
    func_dict['unknown'] = unknown_atoms 
    return func_dict

# Use smiles function(Using .sdf file)
def func_features(sdf, debug=False):   
    # Load SDF
    mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
    
    if mol == None: return False

    nodefeaturizer  = NodeFeat(mol, mode='polH') #
    edgefeaturizer = EdgeFeat(mol, idx=nodefeaturizer.idx) # borrow node index

    # supplier to canonical SMILES
    #smiles = Chem.MolToSmiles(mol, canonical=True)
    #nodefeaturizer = NodeFeat(smiles) # w/o hydrogen
    
    one_hot_node   = NodeOneHot()
    one_hot_edge   = BondInfo()
    
    # Node
    features = {}
    features['Aromatic'] = np.array(nodefeaturizer.Aromatic(one_hot_node))
    features['nCH3']     = np.array(nodefeaturizer.NumCH3())
    features['Ring']     = np.array(nodefeaturizer.Ring(one_hot_node))
    features['Hybrid']   = np.array(nodefeaturizer.Hybrid(one_hot_node))
    features['elems']    = np.array(nodefeaturizer.Elems())

    features['xyz']      = np.array(nodefeaturizer.Coordinates())
    features['numH']     = np.array(nodefeaturizer.NumH())
    features['atypes']   = np.array(nodefeaturizer.GenAtomType())

    fps = Chem.AllChem.MMFFGetMoleculeProperties(mol)
    if fps == None: return False
    features['qs_lig'] = [fps.GetMMFFPartialCharge(x) for x,atom in enumerate(mol.GetAtoms())]

    #collect atoms up to polarHs

    #for key in ['xyz','numH','atypes']:
    # functional groups
    fns = LoadPatterns(fileName='/home/kathy531/Caesar-lig/code/rdkit/FragmentsDescriptors.csv')
    mol = Chem.RemoveHs(mol)
    func_dict= FuncG_rev(fns, mol)
    features['FuncG'] = nodefeaturizer.OneHotFuncG(func_dict,one_hot_node)
            
    #Edge
    features['Bond_idx'] = edgefeaturizer.BndIdx()
    features['Is_conju'] = edgefeaturizer.IsConjugated(one_hot_edge)
    features['Bond_info'] = edgefeaturizer.BndInfo(one_hot_edge)
    features['Num_rot'] = edgefeaturizer.NumRot()

    if debug:
        for key in ['Aromatic','nCH3','Ring','Hybrid','elems','xyz','numH','atypes','qs_lig', 'Bond_idx']:
            print(key,len(features[key]),features[key])
    
    return features 

def features_from_sdf(sdf, debug=False):
    features = func_features(sdf, debug=debug)
    if not features: return False

    # To add: gen_atypes_lig (for vdw)
    element_to_index = {element: index for index, element in enumerate(ELEMS_TYPE) if element!='H'}
    indices = []
    for element in features['elems']:
        if element in element_to_index:
            indices.append(element_to_index[element])
        elif element == 'H': continue
        else:
            sys.exit(f"KeyError: '{element}' is not found")
                    
    one_hot_encoded = np.zeros((len(indices), len(ELEMS_TYPE)), dtype=int)
    one_hot_encoded[np.arange(len(indices)), indices] = 1
    features['elems'] = one_hot_encoded
    return features

def multiP_crest(j):
    npzfile = '/ml/crest-zinc/features/subset_%d.prop.npz'%(j)
    
    outpath = '/ml/MotifLead/Ligand/subset_%d/'%j
    if not os.path.exists(outpath): os.mkdir(outpath)
    
    featurize_with_npz(sdf, npzfile, outpath, j)

def multiP_single_sdf(sdf):
    outpath = './'
    tag = sdf.split('/')[-1][:-4]
    
    outf = outpath+tag+'.lig.npz'
    #if os.path.exists(outf): return
    if not os.path.exists(outpath): os.mkdir(outpath)
    
    #try:
    features = features_from_sdf(sdf, debug=('-debug' in sys.argv))
    if features:
        np.savez(outf, **features)
    #except:
    #    pass

def featurize_with_npz(npzfile, outprefix, j):
    data = np.load(npzfile, allow_pickle=True)
    
    for i,tag in enumerate(data):
        if i%100 == 0:
            print("processing", tag, "from", npzfile, i, "/", len(data))
        sdf = '/ml/crest-zinc/subset_%d/%s.sdf'%(j,tag)

        datum = data[tag].item()
        features = featurize_from_sdf(sdf)

        features['Hlig'] = datum['energy'] #label
        features['Slig'] = datum['ent'] #label
        features['xyz']  = datum['xyz'] #override

        outf = outprefix+tag+'.lig.npz'
        np.savez(outf, **features)
'''

# In case sdf+RDkit fails...
## hold until needed...

def features_from_mol2(mol2, debug=False, include_H=False):
    featurizer = GenPotFeat(mol2, include_H=include_H)
    features = featurizer.extract_features_using_genpot()
    
    if debug:
        for key in ['Aromatic','nCH3','Ring','Hybrid','elems','xyz','numH','atypes','q']:
            print(key,len(features[key]),features[key])
    
    return features
        
def features_from_pdbs_and_params(pdbs, receptor_prop, mol2, debug=False, include_H=False):
    # grab common ligand features first
    features = features_from_mol2(mol2, include_H=include_H)
    #print(features['anames'])
    
    if not features or features == None:
        print("failed to parse mol2: ", mol2)
        return False
    
    # additional per-decoy info
    features['name'] = []
    features['xyz_rec'] = []
    features['xyz'] = [] # reset

    # make sure run this once receptor info is prepared
    for pdb in pdbs:
        name = pdb.split('/')[-1]
        xyz_lig, xyz_rec = read_pdb(pdb, 'LG1', receptor_prop, features, index_by_order=include_H)
        if len(xyz_lig) == 0:
            continue
        features['name'].append(name)
        features['xyz'].append(xyz_lig)
        features['xyz_rec'].append(xyz_rec)

    features['xyz'] = np.array(features['xyz'])
    features['xyz_rec'] = np.array(features['xyz_rec'])

    if len(features['xyz']) <= 5:
        print(f"skip {mol2} due to too small number of passed pdbs {len(features['xyz'])}/{len(pdbs)}")
        return False
    return features
    
def read_pdb(pdb, ligname, receptor_prop, features, index_by_order=False):
    anames_lig = features['anames']
    if len(anames_lig) == 0:
        return [],[]
        
    reschains = receptor_prop['reschains']
    anames_rec = receptor_prop['atmnames']

    rca2id = {rc+'.'+aname:i for i,(rc,aname) in enumerate(zip(reschains,anames_rec))}

    if index_by_order:
        for i,aname in enumerate(anames_lig):
            rca2id['X1.'+aname] = i
    
    xyz_rec = np.zeros_like(receptor_prop['xyz_rec'])
    xyz_lig = np.zeros((len(features['anames']),3))

    nlig,nrec,nfail = 0,0,0
    
    # make sure atomic index is same as in the prop
    for l in open(pdb):
        if not l.startswith('ATOM') and not l.startswith('HETATM'): continue
        chain = l[21]
        res = l[22:26].strip()
        resname = l[16:20].strip()
        aname = l[12:16].strip()
        
        if aname == '1H': aname = 'H'
        elif aname in ['2H','3H']: continue
            
        rca = chain+str(res)+'.'+aname
        xyz = np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
            
        if resname == ligname:
            if index_by_order:
                i = nlig
            else:
                if rca not in rca2id:
                    nfail += 1
                    continue
                i = rca2id[rca]
            xyz_lig[i] = xyz
            nlig += 1
        else:
            # ignore hydrogens for receptor part
            if rca not in rca2id:
                #aname = remapH(aname)
                #print("?",pdb, rca)
                nfail += 1
                continue
            i = rca2id[rca]
            xyz_rec[i] = xyz
            nrec += 1

    #print(len(xyz_rec), nrec, len(xyz_lig), nlig)
    if len(xyz_rec) != nrec or len(xyz_lig) != nlig:
        print(f"Warning: {pdb} rec {len(xyz_rec)} != {nrec} or lig {len(xyz_lig)} != {nlig}")
        return [],[]
            
    return xyz_lig, xyz_rec

def read_xyzf(xyzf,amap,include_H=False):
    if xyzf.endswith('.gz'):
        import gzip
        out = gzip.open(xyzf, 'rt')
    else:
        out = open(xyzf)

    i = 0
    Es = [] # unused here
    xyzs = []
    names = []
    idx = 0 #atomidx
    for l in out:
        words = l[:-1].split()
        if len(words) < 4:
            i += 1
        else:
            i = 0
            
        if i == 2:
            Es.append(float(words[0]))
            xyzs.append(np.zeros((len(amap),3)))
            names.append('conf%04d'%len(names))
            idx = -1
            
        elif i == 0:
            elem = words[0][0]
            if (not include_H) and (elem == 'H'): continue
            idx += 1
            aname = l[:-1].split()
            iatm = amap[idx]
            xyzs[-1][iatm] = np.array([float(words[1]),float(words[2]),float(words[3])])
    return np.array(Es), np.array(xyzs), names

def get_atommap(refxyz,sdf,include_H=False): #xyz2mol2
    anames_xyz = [l[:-1].split()[0] for i,l in enumerate(open(refxyz)) if i >= 2]
    anames_sdf = [l[:-1].split()[3] for i,l in enumerate(open(sdf)) if i >= 4 and len(l) > 50]

    if not include_H:
        anames_xyz = [a for a in anames_xyz if not a.startswith('H')]
        anames_sdf = [a for a in anames_sdf if not a.startswith('H')]

    amap = [anames_sdf.index(a) for a in anames_xyz] # xyz idx -> sdf idx
    return amap

def multiP_single_mol2(args):
    mol2,xyzf,outpath,include_H = args

    if '/' in mol2:
        setno = mol2.split('/')[-2]
    else:
        setno = os.getcwd().split('/')[-1]
        
    if outpath == '': outpath = './'
    tag = mol2.split('/')[-1][:-5]
    outf = outpath+'/'+tag+'.lig.npz'

    #if os.path.exists(outf): return
    
    if not os.path.exists(outpath): os.mkdir(outpath)

    refnpz = '/ml/MotifLead/Oct2024/Ligand/%s/%s.lig.npz'%(setno,mol2[:-5])
    if not os.path.exists(refnpz):
        refnpz = '/ml/MotifLead/Oct2024/Ligand/%s/%s.feat.npz'%(setno,mol2[:-5])

    skip_Slig = False
    if not os.path.exists(refnpz):
        skip_Slig = True
    #    print("file not exist; pass", mol2)
    #    return
        
    #try:
    features = features_from_mol2(mol2, include_H=include_H)
    features['names'] = 'conf0000'
    
    if not skip_Slig:
        refxyzf = mol2[:-5]+'.reindex.xyz'
        if os.path.exists(refxyzf) and os.path.exists(xyzf):
            amap = get_atommap(refxyzf,mol2[:-5]+'.reindex.sdf',include_H)
            Hs,xyzs,names = read_xyzf(xyzf,amap,include_H)
            features['Hlig'] = Hs
            features['xyz']  = xyzs
            features['name'] = names
        else:
            features['Hlig'] = np.array([0.0])
        
        Slig = np.load(refnpz,allow_pickle=True)['Slig']
        features['Slig'] = Slig
    
    np.savez(outf, **features)
    print("save at", outf)

def multiP_pdbpath(path_a, outpath=None):
    if outpath == None: outpath = './' #outpath = path_a
    tag = path_a.split('/')[-1]
    
    outf = outpath+'/'+tag+'.lig.npz'
    if SKIP_EXIST and os.path.exists(outf): return 
    if not os.path.exists(outpath): os.mkdir(outpath)

    mol2 = 'mol2/'+tag+'.ligand.mol2'
    if not os.path.exists(tag+'.prop.npz'): return
    
    prop = np.load(tag+'.prop.npz',allow_pickle=True)

    if not os.path.exists(mol2):
        print(f'no mol2f {mol2} exist!')
        return
        
    pdbs = glob.glob(path_a+'/*00[0-9]?.pdb')
    try:
        features = features_from_pdbs_and_params(pdbs, prop, mol2, debug=('-debug' in sys.argv), include_H=include_H)
        if features:
            np.savez(outf, **features)
        else:
            print("failed to featurize ", path_a)
    except:
        print("error to featurize ", path_a)
        pass
    
if __name__ == "__main__":
    # generate using sdf
    opt = sys.argv[1]
    outpath = sys.argv[3]
    
    if not os.path.exists(outpath): os.mkdir(outpath)
    if len(sys.argv) <= 3:
        sys.exit('USAGE: python featurize_ligand.py [path/mol2/mol2extra/sdf] [input] [outpath]')

    include_H = True # since May2025


    if opt == 'sdf':
        print("disabled due to consistency")
        pass 
        #if sys.argv[2].endswith('.sdf'):
        #    sdfs = [sys.argv[2]]
        #else:
        #    sdfs = [l[:-1] for l in open(sys.argv[2])]
        #    
        #with mp.Pool(processes=min(len(sdfs),NP)) as pool:
        #    pool.map(multiP_single_sdf, sdfs)
            
    elif opt.startswith('mol2'):
        if sys.argv[2].endswith('.mol2'):
            fs = [sys.argv[2]]
        else:
            fs = [l[:-1] for l in open(sys.argv[2])]

        if opt == 'mol2':
            args = [(f,None,outpath+'/', include_H) for f in fs]
        elif opt == 'mol2extra':
            args = [(f,f[:-5]+'-conf.xyz.gz',outpath+'/', include_H) for f in fs]

        with mp.Pool(processes=min(len(args),NP)) as pool:
            pool.map(multiP_single_mol2, args)
        #for arg in args: multiP_single_mol2(arg)

    # in case ligands are at docked poses
    elif opt == 'path':
        # make sure mol2 or params file exist
        pdbpath = [l[:-1] for l in open(sys.argv[2])]
        
        #multiP_pdbpath(pdbpath[0])
        with mp.Pool(processes=min(len(pdbpath),NP)) as pool:
            pool.map(multiP_pdbpath, pdbpath)
    else:
        print("unknown option ", opt, ", should be [path/sdf]")

    

