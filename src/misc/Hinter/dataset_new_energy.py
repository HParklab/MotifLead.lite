import sys
import os
from os.path import join
import numpy as np
import torch
import dgl
# from SE3_nvidia.utilsXG import *
from torch.utils import data
from scipy.spatial import cKDTree
from copy import deepcopy
from src.new_utils.chem_utils import N_AATYPE, findAAindex, gentype2num, find_gentype2num, cosine_angle
from typing import Tuple
from src.dataset_global import Dataset, DfAFSampler


class EnergyDataset(Dataset):
    def __init__(self,
                 targets,
                 root_dir        = "/projects/ml/ligands/v5", #featurized with extended aa defs! UNK goes to 0
                 verbose         = False,
                 ball_radius     = 10,
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 sasa_method     = "none",
                 bndgraph_type   = 'bonded',
                 edgemode        = 'topk',
                 edgek           = (12,12),
                 edgedist        = (8.0,4.5),
                 ballmode        = 'com',
                 distance_feat   = 'std',
                 normalize_q     = False,
                 aa_as_het       = False,
                 nsamples_per_p  = 1,
                 sample_mode     = 'random',
                 use_AF          = False,
                 subset_size      = 1,
                 training        = False,
                 distance_map_dist = 5.0,
                 occlusion_path  = None,
                 AF_plddt_path = None,
                 extra_edgefeat  = False,
                 ros_dir = "/home/jyun/data/ros_e.npz/"):
        super().__init__(targets, root_dir, verbose, ball_radius, randomize, tag_substr, upsample,
                         num_channels, sasa_method, bndgraph_type, edgemode, edgek, edgedist,
                         ballmode, distance_feat, normalize_q, aa_as_het, nsamples_per_p, sample_mode,
                         use_AF, subset_size, training, distance_map_dist, occlusion_path, AF_plddt_path,
                         extra_edgefeat)
        self.rosdata_dir=ros_dir
        self.W, self.S = self.vdw_radius_depth

    def __getitem__(self, index):
        # Select a sample decoy
        info = {}
        info['sname'] = 'none'
        if self.sample_mode == "serial":
            ip = int(index/self.nsamples_per_p)
            pname = self.proteins[ip]
        else:
            pname = self.proteins[index]
        info['pname'] = pname

        fname = join(self.datadir, pname+".lig.npz")
        if not os.path.exists(fname):
            return Dataset._skip_getitem(info)
        try:
            samples_ros = self.get_all_energy(pname)
            samples, pindex = self.get_a_sample(pname, index, samples_ros)
            prop = np.load(join(self.datadir, pname+".prop.npz"))
        except:
            print("Files does not exist!", join(self.datadir, pname))
            return Dataset._skip_getitem(info)
        
        sname = samples['name'][pindex].reshape(self.subset_size, -1)
        info['pindex'] = pindex
        info['sname'] = sname
        
        # Receptor features (prop)
        charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschain = self.receptor_features(prop)
        
        # Ligand features (lig)
        xyz_ligs, xyz_recs, lddt, fnat, atypes_lig, \
            bnds_lig, charges_lig, aas_lig, repsatm_lig = self.per_ligand_features(samples, pindex)
        
        # Rosetta_energy
        rosetta_e = self.select_energy(samples_ros, sname)
        if rosetta_e == None:
            return Dataset._skip_getitem(info)

        # aa type should directly from feature instead
        aas = np.concatenate([aas_lig, aas_rec]).astype(int)
        aas1hot = np.eye(N_AATYPE)[aas]

        # Representative atoms
        r2a = np.concatenate([np.array([0 for _ in range(xyz_ligs.shape[1])]), r2a])
        r2a1hot = np.eye(max(r2a)+1)[r2a]
        repsatm_idx = np.concatenate([np.array(repsatm_lig), np.array(repsatm_idx, dtype=int)+xyz_ligs.shape[1]]) # shape: (subset_size, num_atom_rep)
    
        # Bond properties
        bnds_rec = prop['bnds_rec'] + xyz_ligs.shape[1] #shift index;
        
        # Concatenate receptor & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]), axis=1)
        if self.normalize_q:
            charges = 1.0/(1.0+np.exp(-2.0*charges))

        islig = np.array([1 for _ in range(xyz_ligs.shape[1])]+[0 for _ in range(xyz_recs.shape[1])])
        islig = np.expand_dims(islig, axis=1) 

        xyz = np.concatenate([xyz_ligs, xyz_recs], axis=1)
        atypes = np.concatenate([atypes_lig, atypes_rec])
        w_vdw = torch.tensor([self.W[atype] for atype in list(atypes)]) # vdw depth
        s_vdw = torch.tensor([self.S[atype] for atype in list(atypes)]) # vdw radius     
 
        atypes_int = np.array([find_gentype2num(at) for at in atypes]) # string to integers
        atypes = np.eye(max(gentype2num.values())+1)[atypes_int]     
        
        if samples["xyz_rec"].shape[1] != prop["xyz_rec"].shape[0]:
            return Dataset._skip_getitem(info)

        sasa = []
        sasa_lig = np.array([0.5 for _ in range(xyz_ligs.shape[1])]) #neutral value
        
        sasa = np.concatenate([sasa_lig,sasa_rec])
        sasa = np.expand_dims(sasa, axis=1)
        bnds = np.concatenate([bnds_lig, bnds_rec])

        # orient around ligand-COM
        center_xyz = np.mean(xyz_ligs, axis=1) 
        center_xyz = np.expand_dims(center_xyz, axis=1)
        xyz = xyz - center_xyz
        xyz_ligs = xyz_ligs - center_xyz
        center_xyz[:,:,:] = 0.0
        
        ball_xyzs = []
        if self.ballmode == 'com':
            ball_xyzs = [center_xyz]
        elif self.ballmode == 'all':
            ball_xyzs = [[a[None,:] for a in xyz_lig] for xyz_lig in xyz_ligs]

        # randomize coordinate
        if self.randomize > 1e-3:
            # randxyz = 2.0*self.randomize*(0.5 - np.random.rand(self.subset_size, xyz.shape[1],3)) # -0.2 ~ 0.2
            randxyz = self.randomize*np.random.randn(self.subset_size, xyz.shape[1],3)
            xyz = xyz + randxyz

        resfeat = [islig, aas1hot, sasa]
        resfeat_extra = [] #from res-index

        if self.use_AF:
            # Dims: islig(1) + aas1hot(21) + atypes(65) + charges(1) -> 88
            input_features = [islig, aas1hot, atypes, charges]

            # AF features (dim=1) -> 89
            input_features  = self.add_extra_features(input_features, pname, sname, prop, samples, self.training)
            if input_features is None:
                return Dataset._skip_getitem(info)

            G_atm_list, idx_ord_list = [], []
            for idx in range(self.subset_size):
                G_atm, idx_ord = self.make_atm_graphs(xyz[idx], ball_xyzs[idx], input_features, bnds, 
                                                      xyz_ligs.shape[1], atypes_int, charges, s_vdw, w_vdw)
                G_atm_list.append(G_atm)
                idx_ord_list.append(idx_ord)
        else:
            G_atm_list, idx_ord_list = [], []
            for idx in range(self.subset_size):
                G_atm, idx_ord = self.make_atm_graphs(xyz[idx], ball_xyzs[idx], [islig,aas1hot,atypes,charges], # dims: islig(1) + aas1hot(33) + atypes(65) + charges(1) -> 100
                                                     bnds, xyz_ligs.shape[1], atypes_int, charges, s_vdw, w_vdw)
                G_atm_list.append(G_atm)
                idx_ord_list.append(idx_ord)

        info['islig'] = torch.tensor(islig).float()

        # Reorder by where atm idx in G_atm are
        rsds_ord_list = [r2a[idx_ord] for idx_ord in idx_ord_list]
        G_res_list, r2amap_list = [], []
        for idx in range(self.subset_size):
            G_res, r2amap = self.make_res_graph(xyz[idx], center_xyz[idx], resfeat,
                                                repsatm_idx, rsds_ord_list[idx], resfeat_extra)
            G_res_list.append(G_res)
            r2amap_list.append(r2amap)

        # store which indices go to ligand atms
        ligidx_list = []
        for idx in range(self.subset_size):
            ligidx = np.zeros((len(idx_ord_list[idx]), xyz_ligs.shape[1]))
            for i in range(xyz_ligs.shape[1]): 
                ligidx[i,i] = 1.0
            ligidx_list.append(torch.tensor(ligidx).float())
        info['ligidx'] = ligidx_list
            
        info['fnat']  = torch.tensor(fnat).float()
        info['rosetta_e'] = torch.tensor(np.array(rosetta_e)).float()
        info['lddt']  = torch.tensor(lddt).float()
        info['r2amap'] = [torch.tensor(r2amap).float() for r2amap in r2amap_list]
        info['r2a']   = torch.tensor(r2a1hot).float()
        info['repsatm_idx'] = torch.tensor(repsatm_idx).float()
        return G_atm_list, G_res_list, info
    
    def make_atm_graphs(self, xyz, ball_xyzs, obt_fs, bnds, nlig, atypes_int, 
                        charges, s_vdw, w_vdw):
        kd      = cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca   = cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
        indices = np.unique(indices)

        # Make sure ligand atms are ALL INCLUDED
        idx_ord = [i for i in indices if i < nlig]
        idx_ord += [i for i in indices if i >= nlig]
        old_idx_map = {new_i: old_i for new_i, old_i in enumerate(idx_ord)}

        # Concatenate all one-body-features
        obt  = []
        for f in obt_fs:
            if len(f) > 0: 
                obt.append(f[idx_ord])
        obt = np.concatenate(obt, axis=-1)
        
        xyz_old = xyz
        bnds_old = bnds
        xyz     = xyz[idx_ord]
        bnds    = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]
        bnds_bin = np.zeros((len(xyz),len(xyz)))
        for i,j in bnds:
            k,l = idx_ord.index(i),idx_ord.index(j)
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        
        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()
        
        # for G_atm
        u,v = self.dist_fn_atm(xyz[None,]) # Num_edges
            
        # Edge feature: distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
        w1hot = self.distance_feature(self.distance_feat,w,0.5,5.0)
        bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:,0] = bnds_bin # Shape: (Num_edges, 2)

        # Other edge features
        if self.extra_edgefeat:
            uv = np.array(torch.concatenate([u[None,], v[None,]]).T)
            edge_hbond = torch.tensor(self.is_hbond(uv, bnds_old, atypes_int, xyz_old, old_idx_map), dtype=torch.int)
            edge_pol_apol = torch.tensor(self.is_polar_apolar(uv, bnds_old, charges, atypes_int, xyz_old, old_idx_map), dtype=torch.int)
            w1hot = torch.concatenate([w1hot, edge_hbond[:,None], edge_pol_apol], dim=-1)
        
        # Other node features
        S_vdw=[s_vdw[i]for i in idx_ord]
        W_vdw=[w_vdw[i]for i in idx_ord]

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        G_atm.ndata['0'] = torch.tensor(obt).float()
        G_atm.ndata['x'] = xyz[:,None,:]
        G_atm.ndata['vdw_sig'] = torch.tensor(S_vdw).float()
        G_atm.ndata['vdw_dep'] = torch.tensor(W_vdw).float()
        G_atm.edata['rel_pos'] = xyz[v] - xyz[u]
        G_atm.edata['0'] = w1hot

        return G_atm, idx_ord
    
    def get_all_energy(self, pname):
        samples_ros = np.load(join(self.rosdata_dir,pname+".npz"), allow_pickle=True)
        return samples_ros
    
    def select_energy(self, samples_ros, sname):
        rosetta_e = []
        for i in range(len(sname)):
            name = str(sname[i][0] + '.pdb')
            index = np.where(samples_ros['pdb'] == name)[0]
            if len(index) == 0:
                return None
            rosetta_e.append([samples_ros['vdw'][index][0], samples_ros['solvation'][index][0], 
                              samples_ros['Coulomb'][index][0],samples_ros['Hbond'][index][0]])
        return rosetta_e

    def get_a_sample(self, pname, index, samples_ros):
        samples = np.load(join(self.datadir, pname+".lig.npz"), allow_pickle=True)
        pindices = list(range(len(samples["name"])))
        names = samples["name"]
        e_names = samples_ros["pdb"]
        xsorted = np.argsort(e_names)
        re_idx = xsorted[np.searchsorted(e_names[xsorted], names)]
        vdw = samples_ros["vdw"][re_idx]
        # Filtering repulsion
        if self.sample_mode == 'random':
            pindex  = np.random.choice(pindices, size=self.subset_size, replace=False, p=self.upsample(vdw))
        elif self.sample_mode == 'serial':
            pindex = [index%len(pindices)]
        return samples, pindex
    
    @property
    def vdw_radius_depth(self):
        path='/home/jyun/projects/Discriminator_for_Model_Docking/src/dH/atom_properties_f.txt'
        with open(path) as file:
            W, S = {}, {}
            for line in file:
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('NAME'):
                    continue
                fields = line.split()
                try:
                    atom_name = fields[0]
                    if fields[3] != '':
                        w = float(fields[3])
                    if fields[2] != '':
                        s = float(fields[2])
                    W[atom_name] = w
                    S[atom_name] = s
                except ValueError: # 숫자 변환에 실패한 경우
                    print(f"파싱 에러: {line}")
        return W, S
    

def collate(samples):
    graphs_atm, graphs_res, info = samples[0]
    
    binfo = {'ligidx':[],'r2a':[],'fnat':[], 'pname':[],'sname':[],'nligatms':[],'rosetta_e':[]}

    try:
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)

        asum, bsum, lsum = 0,0,0
        # Reformat r2a, ligidx
        binfo['r2a'] = torch.zeros((bgraph_atm.number_of_nodes(), bgraph_res.number_of_nodes()))

        nligsum = sum([s.shape[1] for s in info['ligidx']])
        binfo['ligidx'] = torch.zeros((bgraph_atm.number_of_nodes(), nligsum))
        binfo['sname'] = [s[0] for s in info['sname']]
        binfo['pname'] = [info['pname']] * len(info['sname'])
        binfo['rosetta_e']= info['rosetta_e']

        for idx, (a, b) in enumerate(zip(bgraph_atm.batch_num_nodes(), bgraph_res.batch_num_nodes())):
            l = info['ligidx'][idx].shape[1]
            binfo['ligidx'][asum:asum+a,lsum:lsum+l] = info['ligidx'][idx]
            
            binfo['r2a'][asum:asum+a,bsum:bsum+b] = info['r2amap'][idx]
            binfo['fnat'].append(info['fnat'][idx])
            binfo['nligatms'].append(l)
            
            asum += a
            bsum += b
            lsum += l

        binfo['fnat'] = torch.tensor(binfo['fnat'])
        binfo['nligatms'] = torch.tensor(binfo['nligatms'])
        
    except:
        bgraph_atm,bgraph_res = False,False

    return bgraph_atm, bgraph_res, binfo


def load_dataset(set_params, generator_params,
                 setsuffix="Clean_MT2"): 
    data_path = "/home/bbh9955/DfAF/data/split_data"
    # Datasets
    train_set = EnergyDataset(np.load(join(data_path, "train%s.npy"%setsuffix)), 
                            training=True,
                            **set_params)
    
    valid_params = deepcopy(set_params)
    valid_params.randomize = 0.0
    val_set = EnergyDataset(np.load(join(data_path,"valid%s.npy"%setsuffix)), 
                            training=True,
                            **valid_params)
    # DataLoaders
    train_generator = data.DataLoader(train_set,
                                    worker_init_fn=lambda _: np.random.seed(),
                                    shuffle = False, 
                                    sampler = DfAFSampler(np.load(join(data_path, f"train_weight{setsuffix}.npy"))),
                                    **generator_params)

    valid_generator = data.DataLoader(val_set,
                                      worker_init_fn=lambda _: np.random.seed(),
                                      shuffle=True, 
                                      **generator_params)
    
    return train_generator, valid_generator