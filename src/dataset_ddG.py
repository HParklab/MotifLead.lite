import sys
import os
import numpy as np
import scipy.spatial
import torch
import dgl
import time
# from SE3_nvidia.utilsXG import *
from torch.utils import data
from copy import deepcopy
import src.myutils as myutils
from typing import Tuple

class DataSet(torch.utils.data.Dataset):
    def __init__(self,
                 scaffold_groups, #
                 args,
                 is_train
    ):
        self.edgek = args.edgek
        self.randomize = args.randomize
        self.ball_radius = args.ball_radius
        self.datadir = args.datapath
        
        self.verbose = args.verbose
        self.debug = args.debug
        self.is_train = is_train
        
        self.read_atomic_params()
        self.vnode_at_lig = (args.ligand_args['mode'] == 'vnode')
        self.ligand_edge = args.ligand_args['edge'] #bond or topk
        self.ligemb_type = args.ligand_args['emb']

        t0 = time.time()
        ## preload pairs
        self.ligemb = {}
        self.inputs = scaffold_groups

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        grp = self.inputs[index]


        fname_npz = os.path.join(self.datadir, grp)
        data = np.load(fname_npz, allow_pickle=True)

        ligids = self.sample_ligands(data)

        Glig1 = self.make_complex_graph( data, ligids[0] )
        Glig2 = self.make_complex_graph( data, ligids[1] )

        info = { 'grp':grp, 'ligid':ligids }
        info['label'] = data['dG'][ligids[0]] - data['dG'][ligids[1]]
        
        return Glig1, Glig2, info
    
    def _skip_getitem(self, info):
        if self.verbose:
            print("SKIP:", info)
        info['valid'] = False
        return None
    
    def sample_ligand( self, data ):
        # ensure min-max diff is big enough
        dGs = data['dG']
        diffs = np.abs(dGs[None,:] - dGs[:,None])
        diffs = np.exp(-diffs/diffs.max())-0.999 #0.001 for identical

        idxs = [idx for idx,P in np.denumerate(diffs)]
        Ps = np.array([P for idx,P in np.denumerate(diffs)])
        Ps /= sum(P)

        return idxs[np.random.choice([i for _ in P], p=P)] # difference-weighted random index

    def adjust_xyz(self, G, origin):
        xyz = G.ndata['x'][:,:] - origin
        
        #if self.randomize > 1e-3:
        #    randxyz = self.randomize*np.random.randn( xyz.shape[0], xyz.shape[1], 3 ).astype(np.float32)
        #    xyz = xyz + (2.0*randxyz-1.0)
        G.ndata['x'] = xyz

    def make_complex_graph(self, data, idx, topk=8):
        ## Node features
        # 0-30: aatype
        # 31-50: SYBYL atom types
        # 51: q
        # 52: sasa
        
        obt = [data[key][idx] for key in ['aas','atypes','qs','sasa']]
        obt = np.concatenate(obt,axis=1) #N x 52?
        obt = np.concatenate([np.zeros((obt.shape[0],64)),obt],axis=1) #make empty slot at first 64 channels

        xyz = data['xyz'][idx]
            
        ## Redefine edge
        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-8)
        top_k_var = min(X.shape[1],topk+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)

        # define chemical bond index
        N = recxyz.shape[0]
        bnds_bin = torch.zeros((N,N)).float()
        a = data['bnds'][:,0]
        b = data['bnds'][:,1]
        bnds_bin[a,b] = 1.0

        xyz = torch.tensor(data['xyz']).float()
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)
        
        # normalize
        w = 1.0/(1.0+torch.exp(-2.0*(w-0.5))) #normalized
        ebt = torch.zeros((u.shape[0],2)).float()
        ebt[:,0] = bnds_bin[u,v] #chemical bonds
        ebt[:,1] = w

        # Concatenate coord & centralize xyz to ca.
        G = dgl.graph((u,v))
        G.ndata['0'] = torch.tensor(obt).float()
        G.ndata['x'] = xyz[:,None,:]
        G.edata['rel_pos'] = xyz[v] - xyz[u]
        G.edata['0'] = ebt

        return G
    
    def report_graph_with_aa(self, G, residx, resnames):
        out = open('tmp.pdb','w')
        form = "HETATM %5d%-4s %3s X %3s   %8.3f %8.3f %8.3f 1.00  0.00\n"
        #idx_at_pred = {i:True for i in np.concatenate(residx)}
        #idx_at_env = [i for i in range(G.number_of_nodes()) if not idx_at_pred[i]]
        
        for z,(idx,resnf) in enumerate(zip(residx,resnames)):
            resn = resnf.split('_')[-1][1:]
            iaas = torch.argmax(G.ndata['0'][idx,:21],dim=-1)
            for i,iaa in zip(idx,iaas):
                x = G.ndata['x'][i]
                aa = ["UNK","ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                      "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"][iaa]
                out.write(form%(i,' C  ',aa,resn,x[0],x[1],x[2]))

        b = i+1
        for i,x in enumerate(G.ndata['x']):
            out.write(form%(b+i,' H  ','ENV',0,x[0],x[1],x[2]))

        out.close()
        
def collate(samples):
    valid = [v for v in samples if v != None]
    if len(valid) == 0:
        print("no valid", samples)
        return

    info = {}
    Gempty = dgl.graph(([],[]))

    bG1 = []
    bG2 = []
    for i,s in enumerate(valid): 
        G1,G2,_info = s
        if i == 0:
            info = {key:[] for key in _info}
            
        for key in _info:
            info[key].append(_info[key])
            
        if G1 == None: G1 = Gempty.clone()
        if G2 == None: G2 = Gempty.clone()
        bG1.append(G1)
        bG2.append(G2)

    bG1 = dgl.batch(bG1)
    bG2 = dgl.batch(bG2)
    
    return bG1, bG2, info
    
