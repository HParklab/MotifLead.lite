import numpy as np
import torch
import dgl
import myutils
import scipy
import time
#from torch.utils.data.sampler import Sampler
#from args import args_default #nbatch 변수 가져옴
#import random
def distogram(D, mind=3.0, maxd=8.0, dbin=0.5):
    nbin = int((maxd-mind)/dbin)+1
    D = torch.clamp( ((D-mind)/dbin).int(), min=0, max=nbin-1).long()
    return torch.eye(nbin)[D]

class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataf, args):
        self.targets = [l[:-1] for l in open(dataf)]
       #966c.A132
        self.datapath = args.datapath
        self.topk = args.topk

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,index):
        #이 부분 다시 보기
        t0 = time.time()
        # adding RESNAME
        # [PDBID].[RESNAME RESNO]
        target = self.targets[index] # should be named as [PDBID].[RESNO]
        protein,resno = target.split('.')
        resname,resno=resno.split(' ')
        
        npzf = self.datapath + '/' + protein + '_rev.prop.npz'
        labelf = self.datapath + '/' + protein + '.label.npz'
        t1 = time.time()
        data = np.load(npzf,allow_pickle=True)
        #print(data)
        
        #print(label_value['label'])
        #print()
        
        #try:
        if True:
        
            G,mask = self.make_unbound_graph(data, resno)
            t2 = time.time()
            label = np.load(labelf,allow_pickle=True)['label'].item() #[resno]
            label = torch.tensor([label[resname+' '+resno]])
            label_int=label #also using in batch_sampler, float=>int will be necessary
            #print(label)
            #print()
            
            if label_int.item() > 2.55:
                label_int.item()==2.55

            binning=np.linspace(0,2.5,26)
            digitized_label=np.digitize(label.item(), binning)-1
            digitized_label=max(0,min(25,digitized_label))

            label=torch.tensor(digitized_label)
            #print('label_db', label)
        #except:
        else:
        
            return None, None, None, None, None

        
        info = {'target':target}
        t9 = time.time()
        #print(f"? {G.number_of_nodes():5d} {t1-t0:6.4f} {t2-t1:6.4f} {t9-t2:6.4f}")

        return G, label, mask, info, label_int
        
        
    def make_unbound_graph(self,data,resno,ball_radius=12.0):

        xyz = data['xyz_rec']
        reschain = data['reschains']
        t0 = time.time()

        idx_res = np.where(reschain==resno)[0]
        
        xyz_res = xyz[idx_res]
        t1a = time.time()
        
        # 1. extract coordinates around resno within < 12Ang
        kd      = scipy.spatial.cKDTree(xyz)
        kd_ca   = scipy.spatial.cKDTree(xyz_res)
        indices = np.concatenate(kd_ca.query_ball_tree(kd, ball_radius))
        idx_ord = list(np.array(np.unique(indices),dtype=np.int16))
        xyz     = xyz[idx_ord] # trimmed coordinates
        
        newidx = {idx:i for i,idx in enumerate(idx_ord)}
        #print(newidx)
        mask = [newidx[i] for i in idx_res]
        #print(mask)
        mask = torch.eye(len(xyz))[mask].T
        #print(mask.size())
        #print(idx_res)
        #print()
        t1 = time.time()
        
        # 2. make edges b/w topk
        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-8)#6이 default
        top_k_var = min(X.shape[1],self.topk+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        # exclude self-connection
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        #print("D_neighbor size", D_neighbors.shape)
        #print("E_idx.shape",E_idx.shape)
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)
        G = dgl.graph((u,v)) # build graph
        #print(G.edges())
        # 3. assign node features
        aas = data['aas_rec'][idx_ord]
        aas = np.eye(myutils.N_AATYPE)[aas] #28
        atypes = data['atypes_rec'][idx_ord]
        atypes = np.eye(max(myutils.gentype2num.values())+1)[atypes] #65
        t2 = time.time()
        '''
        sasa = data['sasa_rec'][:,None][idx_ord]
        C_one = data['C_narray'][:,None][idx_ord]
        H_one = data['H_narray'][:,None][idx_ord]
        E_one = data['E_narray'][:,None][idx_ord]
        phi_cos= data['phi_cos_narray'][:,None][idx_ord]
        phi_sin= data['phi_sin_narray'][:,None][idx_ord]
        psi_cos= data['psi_cos_narray'][:,None][idx_ord]
        psi_sin= data['psi_sin_narray'][:,None][idx_ord]
        '''
        sasa = data['sasa_rec'][:,None]
        C_one = data['C_narray'][:,None]
        H_one = data['H_narray'][:,None]
        E_one = data['E_narray'][:,None]
        phi_cos= data['phi_cos_narray'][:,None]
        phi_sin= data['phi_sin_narray'][:,None]
        psi_cos= data['psi_cos_narray'][:,None]
        psi_sin= data['psi_sin_narray'][:,None]
        t2a = time.time()
        sasa    = sasa   [idx_ord]
        C_one   = C_one  [idx_ord]
        H_one   = H_one  [idx_ord]
        E_one   = E_one  [idx_ord]
        phi_cos = phi_cos[idx_ord]
        phi_sin = phi_sin[idx_ord]
        psi_cos = psi_cos[idx_ord]
        psi_sin = psi_sin[idx_ord]
        nodefeats = [aas,atypes,sasa,C_one,H_one,E_one,phi_cos,phi_sin,psi_cos,psi_sin]
        '''
        print("aas")
        print(aas)
        print(aas.shape)
        print("atypes")
        print(atypes)
        print(atypes.shape)
        print("sasa")
        print(sasa.shape)
        print(data['sasa_rec'][:,None])
        print(data['sasa_rec'][:,None].shape)
        print("E_one")
        print(E_one.shape)
        print()
        '''
        #nodefeats = [aas,atypes,sasa]
        nodefeats = np.concatenate(nodefeats,axis=1)# N x (33+64+1)
        nodefeats = nodefeats.astype(float)
        t3 = time.time()
        
        # 4. assign edge features
        bnds = data['bnds_rec']
        bnds_bin = np.zeros((len(xyz),len(xyz)))

        # re-index atms due to trimming
        # new way -- just iter through bonds
        for i,j in bnds:
            if i not in newidx or j not in newidx: continue # excluded node by kd
            k,l = newidx[i], newidx[j] 
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        bonds = torch.tensor(bnds_bin[u,v]).float()[:,None]
        
        disto = distogram(D[:,u,v]).squeeze()
        edgefeats = torch.cat([bonds, disto],axis=1) # E x 3
        #print("edgefeats shape:", edgefeats.shape)
        G.ndata['xyz'] = X.squeeze().float()
        G.ndata['attr'] = torch.from_numpy(nodefeats).float()
        #print(G.ndata['attr'].shape)
        G.edata['attr'] = edgefeats
         
        t9 = time.time()
        print(f"{G.number_of_nodes():5d}  {G.number_of_edges():5d} {t1-t0:6.4f} {t2-t1:6.4f} {t3-t2:6.4f} {t9-t3:6.4f}")
        
        #print(mask)
        #print(mask.shape)
        return G, mask
    
def collate(samples):
    Gs = []
    labels = []
    masks = []
    njs = []
    label_int=[]
    info = {'target':[]}

    nfull = 0
    for g,l,m,i,l_i in samples:
        if g == None: continue
        Gs.append(g)
        labels.append(l)
        masks.append(m)
        njs.append(m.shape[1])
        nfull += m.shape[0]
        label_int.append(l_i)

        for key in info: info[key].append(i[key])
    

    if len(Gs) == 0:
        return [],[],[],[],[]

    labels = torch.tensor(labels)
    label_int=torch.tensor(label_int)
    bG = dgl.batch(Gs)

    ## should revisit in case b > 1
    mask = torch.zeros((len(Gs),nfull,max(njs))).to(bG.device)
    bi = 0
    for b,m in enumerate(masks):
        ni, nj = m.shape
        mask[b,bi:bi+ni,:nj] = m
        bi += ni

    return bG, labels, mask, info, label_int

