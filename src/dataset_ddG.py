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
                 args,
                 inputs, #
                 is_inference=False
    ):
        self.mode = args.mode #pair or single
        
        self.edgek = args.edgek
        self.randomize = args.randomize
        self.ball_radius = args.ball_radius
        self.datadir = args.datapath
        
        self.verbose = args.verbose
        self.debug = args.debug
        
        t0 = time.time()
        ## preload pairs
        self.ligemb = {}
        self.inputs = inputs
        self.inference = is_inference

        if not is_inference:
            self.labels, self.grps = self.read_label(args.label_f)
            print(f'load {len(self.labels)} labels for {len(self.grps)} groups')
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        grp = self.inputs[index]

        if self.mode == 'single':
            fname_npz = os.path.join(self.datadir, grp+'.feat.npz')
            
            data = np.load(fname_npz, allow_pickle=True)
            Glig, ligmask = self.make_complex_graph( data, topk=self.edgek )
            info = { 'grp': grp, 'ligname': [grp] }
            if not self.inference:
                info['label'] = (self.labels[grp],0.0)
            info['ligmask'] = (ligmask, [])

            return Glig, dgl.graph(([],[])), info
        
        elif self.mode == 'pair':
            ligids, lignames = self.sample_ligands(grp, option.sampling_mode)
            
            fname_npz1 = os.path.join(self.datadir, lignames[0]+'.feat.npz')
            fname_npz2 = os.path.join(self.datadir, lignames[1]+'.feat.npz')
            if not os.path.exists(fname_npz1) or not os.path.exists(fname_npz2):
                print("no ", fname_npz1, "or", fname_npz2)
                return None
            
            data1 = np.load(fname_npz1, allow_pickle=True)
            data2 = np.load(fname_npz2, allow_pickle=True)

            Glig1,ligmask1 = self.make_complex_graph( data1, topk=self.edgek ) 
            Glig2,ligmask2 = self.make_complex_graph( data2, topk=self.edgek )

            info = { 'grp': grp, 'ligname':lignames }
            if not self.inference:
                info['label'] = (self.labels[lignames[0]], self.labels[lignames[1]])
            info['ligmask'] = (ligmask1, ligmask2)

            return Glig1, Glig2, info
    
    def _skip_getitem(self, info):
        if self.verbose:
            print("SKIP:", info)
        info['valid'] = False
        return None

    def read_label(self, f):
        labels = {}
        grps = {}
        for l in open(f):
            words = l[:-1].split()
            ligname = words[0]
            dG = float(words[1])
            labels[ligname] = dG
            if len(words) > 2:
                grp = words[2]
                if grp not in grps: grps[grp] = []
                grps[grp].append(ligname)
        return labels, grps
    
    def sample_ligands( self, grp, sampling_mode ):
        # ensure min-max diff is big enough
        ligs = np.array(self.grps[grp])
        dGs = np.array([self.labels[lig] for lig in ligs])
        diffs = np.abs(dGs[None,:] - dGs[:,None])
        if sampling_mode == 'weighted':
            diffs = np.exp(-diffs/(diffs.max()+0.001))+0.001 #0.001 for identical

            idxs = [idx for idx,P in np.ndenumerate(diffs)]
            Ps = np.array([P+1.0e-6 for idx,P in np.ndenumerate(diffs)])
            Ps /= sum(Ps)

            idx_sel = idxs[np.random.choice(len(Ps), p=Ps)]
        else:
            idx_sel = np.where(diffs)[0]

        return idx_sel, (ligs[idx_sel[0]], ligs[idx_sel[1]]) # difference-weighted random index

    def adjust_xyz(self, G, origin):
        xyz = G.ndata['x'][:,:] - origin
        
        #if self.randomize > 1e-3:
        #    randxyz = self.randomize*np.random.randn( xyz.shape[0], xyz.shape[1], 3 ).astype(np.float32)
        #    xyz = xyz + (2.0*randxyz-1.0)
        G.ndata['x'] = xyz

    def make_complex_graph(self, data, topk=8):
        ## Node features
        # 0-30: aatype
        # 31-71: SYBYL atom types
        # 72: q
        # 73: sasa
        
        t0 = time.time()

        '''
        if self.ignore_hydrogen:
            atmidx = np.intersect1d(np.where(data['atypes']!=23), np.where(data['atypes']!=24))
            atmmap = {i:j for i,j in enumerate(atmidx)}
        else:
            atmidx = np.arange(len(data['atypes']))
            atmmap = {i:i for i,_ in enumerate(atmidx)}
        '''
            
        obt = []
        obt.append(np.eye(31)[data['aas']])
        obt.append(np.eye(41)[data['atypes']])
        obt.append(data['qs'][:,None])
        obt.append(data['sasa'][:,None])
                   
        '''
        for key in ['aas','atypes','qs','sasa']: # 31 + 41 + 1 + 1 = 74
            print(key, data[key].shape, data[key][:10])
            if len(data[key].shape) == 1: obt.append(data[key][:,None])
            else: obt.append(data[key])
        '''

        obt = np.concatenate(obt,axis=1)

        xyz = data['xyz']
            
        t1 = time.time()
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

        t2 = time.time()
        # define chemical bond index
        N = xyz.shape[0]
        bnds_bin = torch.zeros((N,N)).float()
        bonds = data['bnds']
        
        a = data['bnds'][:,0]
        b = data['bnds'][:,1]
        bnds_bin[a,b] = 1.0

        xyz = torch.tensor(xyz).float()
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)
        
        t3 = time.time()
        # normalize
        w = 1.0/(1.0+torch.exp(-2.0*(w-0.5))) #normalized
        ebt = torch.zeros((u.shape[0],2)).float()
        ebt[:,0] = bnds_bin[u,v] # bool: chemical bonds
        ebt[:,1] = w # normalized distance

        # Concatenate coord & centralize xyz to ca.
        G = dgl.graph((u,v))
        G.ndata['0'] = torch.tensor(obt).float()
        G.ndata['x'] = xyz[:,None,:]
        G.edata['rel_pos'] = xyz[v] - xyz[u]
        G.edata['0'] = ebt

        ligmask = torch.zeros(G.number_of_nodes())
        ilig = torch.where(G.ndata['0'][:,0]==1)[0]
        ligmask[ilig] = 1
        t4 = time.time()

        #print("time %8.5f %8.5f %8.5f %8.5f, num edge %d"%(t1-t0, t2-t1, t3-t2, t4-t3, G.number_of_edges()))

        return G, ligmask
    
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
    #n1, n2 = [], []
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
        #n1.append(G1.number_of_nodes())
        #n2.append(G2.number_of_nodes())

    bG1 = dgl.batch(bG1)
    bG2 = dgl.batch(bG2)

    # re-index
    b = len(valid)
    ligmask1 = torch.zeros(b,bG1.number_of_nodes())
    ligmask2 = torch.zeros(b,bG2.number_of_nodes())

    n1, n2 = 0,0
    for i,(mask1,mask2) in enumerate(info['ligmask']):
        ligmask1[i,n1:n1+len(mask1)] = mask1
        if len(mask2) > 0: # for mode == "single"
            ligmask2[i,n2:n2+len(mask2)] = mask2
        n1 += len(mask1)
        n2 += len(mask2)

    info['ligmask'] = (ligmask1, ligmask2)

    if 'label' in info:
        dG = torch.zeros(b,2)
        for i,(dG1, dG2) in enumerate(info['label']):
            dG[i,0] = dG1
            dG[i,1] = dG2
        info['label'] = dG
    
    return bG1, bG2, info
    
