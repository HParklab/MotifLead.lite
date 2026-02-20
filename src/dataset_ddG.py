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

        info = {'grp':grp}

        fname_npz = os.path.join(self.datadir, grp)

        ligids = self.sample_ligands(fname_npz)

        
        
        return Glig1, Glig2, info
    
    def load_single_complex(self, complex_prop, config):
        G_complex = self.make_complex_graph(complex_prop, topk=self.edgek[2])
        #origin = self.sample_origin(xyz_ball,axis=1)

        origin = torch.mean(xyz_ball,axis=0).float()
        self.adjust_xyz(G_complex, origin)

        if mode == 0: #Hinter mode
            fname_complex = os.path.join(self.datadir, cmplx+".label.npz")
            if not os.path.exists(fname_complex): return self._skip_getitem(info)
            
            data_complex = np.load(fname_complex,allow_pickle=True)
            inter_energy, do_eval = self.complex_features(data_complex, ligname)
            if not inter_energy: return self._skip_getitem(info)
                
            info['label_Hinter'] = inter_energy
            info['eval_Hinter'] = do_eval
                
        elif mode == 3: #dG mode
            ## to do: mode == 3 also get dG? or unify mode based on file existence
            tag = cmplx.split('/')[-1]
            if tag not in self.dGvals:
                info['reason'] = 'no expt dG value'
                return self._skip_getitem(info)
                
            info['label_dG'] = myutils.find_bin(self.dGvals[tag],self.dGrange)
            info['eval_dG'] = True

        return G_complex, info
            
    def load_single_ligand(self, data_lig, conf_idx=-1):
        if not self.is_train: conf_idx = 0 # fix to the first conformer
        args  = self.sample_ligand(data_lig,
                                   conf_idx=conf_idx,
                                   near_native=(mode==3) )
            
        if len(args) < 4 :
            info['reason'] = 'ligand sampling'
            return self._skip_getitem(info)
        else:
            ligemb = []
            if self.ligemb_type != None:
                ligemb = self.load_ligemb(ligand)
                if ligemb == []: 
                    info['reason'] = 'Ligand Embedding not found'
                    return self._skip_getitem(info)
            try:
                lig_xyz, ligname, lig_label, conf_idx = args
                G_lig = self.make_ligand_graph(data_lig, lig_xyz, topk=self.edgek[1], ligemb=ligemb)
                if not G_lig:
                    info['reason'] = 'ligand graph construction failure'
                    return self._skip_getitem(info)
                        
            except:
                info['reason'] = 'ligand graph construction error'
                return self._skip_getitem(info)
                
        info['label_Hlig']   = lig_label['Hlig']
        info['label_Slig']   = lig_label['Slig']
        if ligname != 'unk':
            info['eval_Hlig']    = 1
            info['eval_Slig']    = 1

        return G_lig, info

    def load_single_receptor(self, data_rec, xyz_ball):
        if xyz_ball == None or len(xyz_ball) == 0: # undefined case
            contact_res, residx, xyz_ball = self.random_sample_res(data_rec)
        else:
            contact_res, residx = self.contact_to_ball(data_rec, xyz_ball)

        
        t1 = time.time()
        # Receptor features & graph (rec_prop)
        # receptor xyz should be [nconf,Natm,3]

        if len(residx) < 1:
            info['reason'] = 'Contact residue'
            return self._skip_getitem(info)
                
        fname_rec_label = os.path.join(self.datadir, receptor+".label.npz")
        if mode == 1 and os.path.exists(fname_rec_label):
            rec_label, resname_Srec = self.read_rec_label( np.load(fname_rec_label, allow_pickle=True),
                                                           contact_res,
                                                           maxbin=self.Srec_max_bin )
                
            if len(rec_label) != len(residx):
                info['reason'] = 'Srec label size'
                return self._skip_getitem(info)
                    
            info['label_Srec'] = rec_label
            info['aas_Srec'] = [data_rec['aas_rec'][idx[0]] for idx in residx]
            info['resname_Srec'] = [receptor.split('/')[-1]+'_'+rn for rn in resname_Srec]
            info['eval_Srec'] = 1
        t1b = time.time()

        if self.debug:
            args  = self.make_receptor_graph(data_rec, xyz_ball, residx, topk=self.edgek[0])
                
        try:
            args  = self.make_receptor_graph(data_rec, xyz_ball, residx, topk=self.edgek[0])
            if len(args) < 4:
                info['reason'] = 'Receptor graph construction'
                return self._skip_getitem(info)
            else:
                G_rec, residx, Grec_idx, rec_bnds = args
        except:
            info['reason'] = 'Receptor graph construction'
            return self._skip_getitem(info)

        info['Srec_index'] = residx #N x var size indicating atomic idx for each selected re
        
        return G_rec, info
            
                    
    def load_ligemb(self, ligand):
        ligemb = []
        ligname = ligand.split('/')[-1] # make sure ligname is a unique identifier...
        fname_ligemb = os.path.join(self.datadir,os.path.dirname(ligand),self.ligemb_type+'.npz')

        if fname_ligemb not in self.ligemb:
            ## on-the-fly loading
            if not os.path.exists(fname_ligemb):
                fname_ligemb = os.path.join(self.datadir, ligand+".ligemb.npz")
                            
            if os.path.exists(fname_ligemb):
                ligemb = np.load(fname_ligemb,allow_pickle=True)['emb'].item()
                if ligname in ligemb:
                    ligemb = ligemb[ligname]
        else:
            if ligname in self.ligemb[fname_ligemb]:
                ligemb = self.ligemb[fname_ligemb][ligname]
            
        return ligemb

    def _skip_getitem(self, info):
        if self.verbose:
            print("SKIP:", info)
        info['valid'] = False
        return None

    def contact_to_ball( self, data, xyz_ball ):
        xyz = data['xyz_rec']
        reschains = data['reschains']
        kd_ball  = scipy.spatial.cKDTree(xyz_ball)
        kd      = scipy.spatial.cKDTree(xyz)
        idx = np.unique(np.concatenate(kd_ball.query_ball_tree(kd, self.ball_radius))).astype(np.int16)

        xyz_res = xyz[idx]
        
        res_included = np.unique(reschains[idx])

        idx_in_res = []
        for res in res_included:
            idx_in_res.append(np.where(reschains==res)[0])
        return res_included, idx_in_res
    
    def random_sample_res( self, data ):
        xyz = data['xyz_rec']
        iatm = np.random.choice(len(xyz))
        reschains = data['reschains']

        # make a group of residues for label evaluation 
        #ires = np.where(reschains == reschains[iatm])[0]
        #xyz_res = xyz[ires]

        # select a seed point
        xyz_com = xyz[np.random.choice(len(xyz))][None,:]
        kd_com  = scipy.spatial.cKDTree(xyz_com)
        kd      = scipy.spatial.cKDTree(xyz)
        idx = np.unique(np.concatenate(kd_com.query_ball_tree(kd, self.ball_radius)))

        xyz_res = xyz[idx]

        res_included = np.unique(reschains[idx])

        idx_in_res = []
        for res in res_included:
            idx_in_res.append(np.where(reschains==res)[0])
        return res_included, idx_in_res, xyz_res

    def read_rec_label( self, data, ballres, maxbin=20 ):
        label = data['label'].item()
        aas = [res for res in ballres if res in label]
        label = [label[res] for res in ballres if res in label]
        return label, aas
    
    def adjust_xyz(self, G, origin):
        xyz = G.ndata['x'][:,:] - origin
        
        #if self.randomize > 1e-3:
        #    randxyz = self.randomize*np.random.randn( xyz.shape[0], xyz.shape[1], 3 ).astype(np.float32)
        #    xyz = xyz + (2.0*randxyz-1.0)
        G.ndata['x'] = xyz

    def concatenate_prop(self, rec_prop, lig_prop, rec_idx, bnds_rec, conf_idx):
        complex_prop = {}

        '''
        rec
        ['aas_rec', 'xyz_rec', 'atypes_rec', 'bnds_rec', 'sasa_rec', 'qs_rec', 'SS3', 'bb_dihe', 'is_repsatm', 'residue_idx', 'reschains', 'atmnames', 'resnames']
        lig
        ['Aromatic', 'nCH3', 'Ring', 'Hybrid', 'elems', 'xyz', 'atypes', 'anames', 'qs_lig', 'FuncG', 'Bond_idx', 'Is_conju', 'Bond_info', 'Num_rot', 'numH', 'name', 'xyz_rec']
        '''
        
        # Concatenate receptor & ligand: receptor comes first
        Nrec = len(rec_idx)

        ## Common features
        # aatype
        aas = np.concatenate([rec_prop['aas_rec'][rec_idx],np.zeros_like(lig_prop['qs_lig'])]).astype(int)
        complex_prop['aas'] = np.eye(myutils.N_AATYPE)[aas]
        
        # Bond properties
        bnds_lig = lig_prop['Bond_idx'] + Nrec
        complex_prop['bnds'] = np.zeros((bnds_rec[0].shape[0]+bnds_lig.shape[0], 2))
        complex_prop['bnds'][:,0] = np.concatenate([bnds_rec[0], bnds_lig[:,0]])
        complex_prop['bnds'][:,1] = np.concatenate([bnds_rec[1], bnds_lig[:,1]])

        charges = np.concatenate([rec_prop['qs_rec'][rec_idx], lig_prop['qs_lig']])[:,None]
        complex_prop['qs'] = charges #1.0/(1.0+np.exp(-2.0*charges))

        islig = np.zeros_like(complex_prop['qs'],dtype=int)
        islig[Nrec+1:] = 1
        complex_prop['islig'] = islig

        atypes = np.concatenate([rec_prop['atypes_rec'][rec_idx], lig_prop['atypes']])
        atypes_int = np.array([myutils.find_gentype2num(at) for at in atypes]) # string to integers
        complex_prop['atypes'] = np.eye(max(myutils.gentype2num.values())+1)[atypes_int]     
        
        sasa_lig = np.zeros_like(lig_prop['qs_lig'])+0.5 #neutral value
        complex_prop['sasa'] = np.concatenate([rec_prop['sasa_rec'][rec_idx], sasa_lig])[:,None]
        ligxyz = lig_prop['xyz']
        if conf_idx > -1:
            ligxyz = ligxyz[conf_idx]
        complex_prop['xyz'] = np.concatenate([rec_prop['xyz_rec'][rec_idx],ligxyz],axis=0)
       
        ## Features specific for energy
        complex_prop['w_vdw'] = torch.tensor([self.atomic_params['W'][atype] for atype in list(atypes)]) # vdw depth
        complex_prop['s_vdw'] = torch.tensor([self.atomic_params['S'][atype] for atype in list(atypes)]) # vdw radius
        
        return complex_prop
    
    def make_complex_graph(self, prop, topk=8):
        ## Node features
        obt = [prop[key] for key in ['aas','atypes','qs','islig','sasa']]
        obt = np.concatenate(obt,axis=1) #N x102
        obt = np.concatenate([np.zeros((obt.shape[0],64)),obt],axis=1) #make empty slot at first 64 channels
            
        ## Redefine edge
        X = torch.tensor(prop['xyz'][None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-8)
        top_k_var = min(X.shape[1],topk+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)

        # define chemical bond index
        N = prop['xyz'].shape[0]
        bnds_bin = torch.zeros((N,N)).float()
        a = prop['bnds'][:,0]
        b = prop['bnds'][:,1]
        bnds_bin[a,b] = 1.0

        xyz = torch.tensor(prop['xyz']).float()
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
        
        G.ndata['vdw_sig'] = prop['s_vdw'].float()
        G.ndata['vdw_dep'] = prop['w_vdw'].float()
        G.ndata['qs'] = torch.tensor(prop['qs']).float()
        G.edata['rel_pos'] = xyz[v] - xyz[u]
        G.edata['0'] = ebt

        return G
    
    def make_receptor_graph(self, data, xyz_ball, residx, graph_radius=12.0, topk=8):
        xyz = data['xyz_rec']
        if self.randomize > 1e-3:
            randxyz = self.randomize*np.random.randn( xyz.shape[0], 3 ).astype(np.float32)
            xyz = xyz + (2.0*randxyz-1.0)

        contact_idx = np.concatenate(residx)
        # 1. extract coordinates around pocket within < 12Ang
        kd      = scipy.spatial.cKDTree(xyz)
        kd_ca   = scipy.spatial.cKDTree(xyz_ball)
        contact_idx = np.concatenate([np.concatenate(kd_ca.query_ball_tree(kd, graph_radius)), contact_idx])
        idx_ord = np.unique(contact_idx).astype(np.int16)
        xyz     = xyz[idx_ord] # trimmed coordinates

        ### sanity check
        kd2 = scipy.spatial.cKDTree(xyz)
        idx = np.concatenate(kd2.query_ball_tree(kd2, 0.5)) #any atom < 0.5
        clash = len(idx)-len(xyz)
        if self.randomize < 1e-3 and clash: # skip if randomized
            return [False]
        
        idxmap = {idx:i for i,idx in enumerate(idx_ord)}
        for i,idx in enumerate(residx):
            residx[i] = np.array([idxmap[j] for j in idx])

        # 2. make edges b/w topk
        X = torch.tensor(xyz[None,]) #expand dimension
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-8)
        top_k_var = min(X.shape[1],topk+1) # consider tiny ones
        D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        u = u.detach().numpy()
        v = E_idx[0,].reshape(-1)
        G = dgl.graph((u,v)) # build graph
        
        # 3. assign node features
        aas = data['aas_rec'][idx_ord]
        aas = np.eye(myutils.N_AATYPE)[aas] #28
        atypes = data['atypes_rec'][idx_ord]
        atypes = np.eye(max(myutils.gentype2num.values())+1)[atypes] #65

        sasa = data['sasa_rec'][:,None][idx_ord]
        SS3  = data['SS3'][idx_ord]
        dih4 = data['bb_dihe'][idx_ord]
        nodefeats = np.concatenate([aas,atypes,sasa,SS3,dih4],axis=1).astype(float)
        
        # 4. assign edge features
        bnds = data['bnds_rec']
        # re-index atms due to trimming
        bnds_bin = np.zeros((len(xyz),len(xyz)))
        newidx = {idx:i for i,idx in enumerate(idx_ord)}
        
        for i,j in bnds:
            if i not in newidx or j not in newidx: continue # excluded node by kd
            k,l = newidx[i], newidx[j] 
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        bonds = torch.tensor(bnds_bin[u,v]).float()[:,None]

        def distogram(D, mind=3.0, maxd=8.0, dbin=0.5):
            nbin = int((maxd-mind)/dbin)+1
            D = torch.clamp( ((D-mind)/dbin).int(), min=0, max=nbin-1).long()
            return torch.eye(nbin)[D]

        disto = distogram(D[:,u,v]).squeeze()
        edgefeats = torch.cat([bonds, disto],axis=1) # E x 3

        G.ndata['x'] = X.squeeze().float()
        G.ndata['0'] = torch.from_numpy(nodefeats).float()
        G.edata['0'] = edgefeats.float()
         
        return G, residx, idx_ord,(u,v)

    def sample_ligand( self, data, conf_idx=-1, near_native=False ): # -> tuple(dict):
        label = {'Hlig':-1.0, 'Slig':-1.0}

        # ddG data
        '''
        if 'v2' in data:
            pdbs = data['names']
            if conf_idx == -1:
                conf_idx = np.random.choice(np.arange(n),p=p)
            label['label_dG'] = data['dG'][name]
        
        # conformer-dependent data
        if len(data['xyz'].shape) == 3:
            n = len(data['xyz'])
            if conf_idx == -1:
                if near_native:
                    # first is exact native; rest are docked
                    p = np.zeros(n)
                    p[0] = 1.0
                    if n > 1: p[1:] = 1.0/(n-1)
                else:
                    p = np.ones(n)
                    
                p /= np.sum(p)
                conf_idx = np.random.choice(np.arange(n),p=p)
                if near_native: print(near_native, p, conf_idx)
                
            if 'Hlig' in data and len(data['Hlig']) >= conf_idx-1:
                Hligs = (data['Hlig'] - data['Hlig'].min())*627.5 # Hartree-to-kcal/mol
                label['Hlig'] = Hligs[conf_idx]
            lig_xyz = data['xyz'][conf_idx]
            
            if 'name' in data:
                ligname = data['name'][conf_idx]
            elif 'names' in data:
                ligname = data['names'][conf_idx]
            else:
                ligname = 'conf%04d'%conf_idx
            
        elif len(data['xyz'].shape) == 2:
            lig_xyz = data['xyz']
            ligname = 'unk'
            label['Hlig'] = -1.0
            
        else:
            return [False]
                
        if self.randomize > 1e-3:
            randxyz = self.randomize*np.random.randn( lig_xyz.shape[0], 3 ).astype(np.float32)
            lig_xyz = lig_xyz + (2.0*randxyz-1.0)
            
        if 'Slig' in data:
            label['Slig'] = data['Slig']

        return lig_xyz, ligname, label, conf_idx

    ## TODO: add feats for Slig
    def make_ligand_graph( self, data, xyz, topk=-1, ligemb=[] ) -> dgl.graph: #topk: undefined yet
        obt = []
        natom = len(data['elems'])

        #numH   = np.eyes(4)[data['numH']] #4
        elems  = np.eye(11)[data['elems']] #11

        # clean old format
        hybrid = data['Hybrid']
        hybrid[hybrid >= 5] = 2 #amide&ring -> sp2
        hybrid = np.eye(5)[hybrid] #5
        funcG = data['FuncG'] #already 1-hot; 16
        
        obt = [data['numH'][:,None], #data['numH'][:,1][:,None],
               elems,
               data['Aromatic'][:,None],
               funcG,
               hybrid] #18

        if len(ligemb) > 1:
            if (len(ligemb) < len(data['numH'])):
                if not self.skip_inconsistent_natom:
                    nH = len(data['numH']) - len(ligemb)
                    ligemb = np.pad(ligemb, [(0,nH),(0,0)]) # pad 0 for hydrogens
                else:
                    print("skip ligemb reading due to incorrect atom size: ", len(ligemb), len(data['numH']))
                    return False
            obt.append(ligemb)
            
        obt = np.concatenate(obt,axis=1) # n x 18 or n x 1
        
        ebt = []
        xyz = torch.tensor(xyz)
        
        if self.ligand_edge == 'bond':
            u = np.concatenate([data['Bond_idx'][:,0],data['Bond_idx'][:,1]])
            v = np.concatenate([data['Bond_idx'][:,1],data['Bond_idx'][:,0]])
            b_type = np.concatenate([data['Bond_info'],data['Bond_info']])

            u = torch.tensor(u)
            v = torch.tensor(v)
            b_type = torch.tensor(b_type)
            
        elif self.ligand_edge == 'topk':
            bonds = np.zeros((len(xyz),len(xyz),data['Bond_info'].shape[-1]))
            for (i,j),b_type in zip(data['Bond_idx'],data['Bond_info']):
                bonds[i,j] = b_type
            bonds = torch.tensor(bonds)

            dX = torch.unsqueeze(xyz[None,],1) - torch.unsqueeze(xyz[None,],2)
            D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-8)
            D_neighbors, E_idx = torch.topk(D, min(topk,len(xyz)-1), dim=-1, largest=False)
            D_neighbor =  D_neighbors[:,:,1:]
            E_idx = E_idx[:,:,1:]
            u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
            v = E_idx[0,].reshape(-1)
            b_type = bonds[u,v]
        
        ebt = torch.cat([b_type, b_type],axis=1) #???

        if self.vnode_at_lig:
            # assign vnode at the COM
            xyz = torch.cat([xyz, torch.mean(xyz,axis=0)[None,:]])
            
            # make node feature 0 except the last dim "is_vnode=1"
            obt = np.pad(obt, (0,1)) #(n+1) x 19
            obt[-1,-1] = 1.0 # is_vnode

            # add edges all-to-vnode & assign btype=0
            iv = len(xyz)-1
            u = torch.cat([u,torch.arange(iv),torch.full((iv,1),iv).squeeze(-1)])
            v = torch.cat([v,torch.full((iv,1),iv).squeeze(-1),torch.arange(iv)])
            bt_add = torch.zeros(iv*2,ebt.shape[-1])
            bt_add[:,0] = 1.0 # 1-hot at btype=0
            ebt = torch.cat([ebt,bt_add])

        G = dgl.graph((u,v))
        G.ndata['0'] = torch.from_numpy(obt).float()
        G.edata['0'] = ebt.float()
        G.ndata['x'] = xyz.float()

        return G

    def complex_features(self, data, ligand):
        ligands = data['pdb']
        index = np.where(ligands)[0]
        # match by "name" (e.g. native_dock_000X.pdb)
        if index == []: return [0,0,0,0], False

        vdw = np.min([5,data['vdw'][index][0]]) # cap at +5
        rosetta_e = [0.0,
                     vdw, 
                     data['Coulomb'][index][0]+data['Hbond'][index][0],
                     data['solvation'][index][0]]
        rosetta_e[0] = np.min([sum(rosetta_e),5.0])
        return rosetta_e, True # 1-st: placeholder for features, if needed

    def report_graph(self, G, outpdb):
        form = "HETATM %5d%-4s UNK X %3d   %8.3f %8.3f %8.3f 1.00  0.00\n"
        out = open(outpdb,'w')
        for i,x in enumerate(G.ndata['x']):
            out.write(form%(i+1,'C',i+1,x[0],x[1],x[2]))
        out.close()

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
        
# unused....
def load_dataset(set_params, generator_params,
                 setsuffix="Clean_MT2",shuffle=True): 
    data_path = "/home/bbh9955/DfAF/data/split_data"
    # Datasets
    train_set = EnergyDataset(np.load( os.path.join(data_path, "train%s.npy"%setsuffix)), 
                              training=True,
                              **set_params)
    
    valid_params = deepcopy(set_params)
    valid_params.randomize = 0.0
    val_set = EnergyDataset(np.load(os.path.join(data_path,"valid%s.npy"%setsuffix)), 
                            training=True,
                            **valid_params)
    # DataLoaders
    train_generator = data.DataLoader(train_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      shuffle=shuffle, 
                                      **generator_params)

    valid_generator = data.DataLoader(val_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      shuffle=False, 
                                      **generator_params)
    
    return train_generator, valid_generator

def collate(samples):
    valid = [v for v in samples if v != None]
    valid = [v for v in valid if v[-1]['valid']]
    if len(valid) == 0:
        print("no valid", samples)
        return

    info = {}
    bGall,bGrec,bGlig = [],[],[]
    Gempty = dgl.graph(([],[]))
    
    for i,s in enumerate(valid): #Gcomplex, Grec, Glig, info
        Gall,Grec,Glig,_info = s
        if i == 0:
            info = {key:[] for key in _info}
            
        for key in _info:
            if (key == 'label_Srec' and Grec == None) or \
               (key == 'label_Slig' and Glig == None) or \
               (key == 'label_Hlig' and Glig == None) or \
               (key == 'label_Hinter' and Gall == None):
                continue
            if key not in info: continue
            info[key].append(_info[key])
            
        if Gall == None: Gall = Gempty.clone()
        if Grec == None: Grec = Gempty.clone()
        if Glig == None: Glig = Gempty.clone()
        bGall.append(Gall)
        bGrec.append(Grec)
        bGlig.append(Glig)

    # do this later on loss evaluation steps
    for key in ['label_Slig','label_Srec','label_Hinter','label_Hlig',
                'eval_Slig','eval_Srec','eval_Hinter','eval_Hlig',
                'label_dG', 'eval_dG']:
        try:
            if key in info and info[key] != None:
                if key == 'label_Srec' and len(info[key]) > 0:
                    info['label_Srec'] = torch.tensor(np.concatenate(info[key])).float()
                elif key == 'label_Slig' and len(info[key]) > 0:
                    info[key] = torch.tensor(np.array(info[key])).float()
                elif key == 'label_dG' and len(info[key]) > 0:
                    info[key] = torch.tensor(info[key]).long()
                else: # hack for now
                    info[key] = torch.tensor(info[key]).float()
        except:
            print("collate fail:", info[key])
            return 

    bGall = dgl.batch(bGall)
    bGrec = dgl.batch(bGrec)
    bGlig = dgl.batch(bGlig)
    
    return bGall, bGrec, bGlig, info
    
