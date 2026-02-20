import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, TorsionFingerprints
from multiprocessing import Pool
import numpy as np
import math
import rdkit
import pickle
import time

class LigandFeaturizer:
    def __init__(
        self,
        file_paths: Path,
        node_featurizer: BaseNodeFeaturizer,
        edge_featurizer: BaseEdgeFeaturizer,
        #cache_dir: Path = Path('/home/hpark/for/Youngwoo/graph_cache')
        cache_dir: Path = Path('/home/nahmyw/projects/requimol/data/graph_cache')
    ):
        self.file_paths = file_paths
        self.conformer_paths = []
        self.conformer_idx = []
        self.mol_list = []
        self.num_conformers_list = [] # For sdf file
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.cache_dir = cache_dir
        self.graphs = []
        self.metadata = []
        
    
    # TODO : One-hot encoding will be processed at featurizer
    def convert_bond_type(self, bond_type):
        if bond_type == "SINGLE":
            return 1
        if bond_type == "AROMATIC":
            return 2
        if bond_type == "DOUBLE":
            return 3
        if bond_type == "TRIPLE":
            return 4

    def convert_hybrid_type(self, hybrid_type, data_path):
        if hybrid_type == "SP":
            return np.eye(3)[0]
        elif hybrid_type == "SP2":
            return np.eye(3)[1]
        elif hybrid_type == "SP3":
            return np.eye(3)[2]
        else:
            print(data_path)
        
    def find_vdw_relation(self, mol, atom_coor):
        # Check for existing bonds
        bond_list = []
        for bond in mol.GetBonds():
            bond_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        # print(atom_coor[0]) # coor check
        
        # Exception for both atoms in same ring
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        vdw_bond_list = []
        for atom_one in mol.GetAtoms():
            atom_one_idx = atom_one.GetIdx()
            for atom_two in mol.GetAtoms():
                atom_two_idx = atom_two.GetIdx()
                if atom_one_idx != atom_two_idx and [atom_one_idx, atom_two_idx] not in bond_list:
                    atom_one_coor = atom_coor[atom_one_idx]
                    atom_two_coor = atom_coor[atom_two_idx]
                    x_dist = (atom_one_coor[0] - atom_two_coor[0])**2
                    y_dist = (atom_one_coor[1] - atom_two_coor[1])**2
                    z_dist = (atom_one_coor[2] - atom_two_coor[2])**2
                    dist = (x_dist + y_dist + z_dist)**(0.5)
                    
                    if dist < 4:
                        for ring in atom_rings:
                            if atom_one_idx in ring and atom_two_idx in ring:
                                break
                        else:
                            vdw_bond_list.append([atom_one_idx, atom_two_idx, dist])
        return vdw_bond_list

    # input : 0th conformer
    # output : [각 torsion별 dihedral angle],[],... * conformer 수
    def cal_torsion_angle(self, mol, atom_coor):
        non_ring_tor, ring_tor = rdkit.Chem.TorsionFingerprints.CalculateTorsionLists(mol, maxDev='equal', symmRadius=2, ignoreColinearBonds=True)
        non_ring_torsion = []
        for i in range(len(non_ring_tor)):
            for j in range(len(non_ring_tor[i][0])):
                non_ring_torsion.append(list(non_ring_tor[i][0][j]))
    
        torsion_angle = {}
        for at1_idx, at2_idx, at3_idx, at4_idx in non_ring_torsion:
            at1_coor = atom_coor[at1_idx]
            at2_coor = atom_coor[at2_idx]
            at3_coor = atom_coor[at3_idx]
            at4_coor = atom_coor[at4_idx]
            b0, b1, b2 = [0,0,0], [0,0,0], [0,0,0]
            for i in range(3):
                b0[i] = at1_coor[i] - at2_coor[i]
                b1[i] = at3_coor[i] - at2_coor[i]
                b2[i] = at4_coor[i] - at3_coor[i]
            unit_b1 = b1 / np.linalg.norm(b1)
        
            v = b0 - np.dot(b0,unit_b1)*unit_b1
            w = b2 - np.dot(b2,unit_b1)*unit_b1
            x = np.dot(v,w)
            y = np.dot(np.cross(unit_b1,v),w)
        
            if (at2_idx, at3_idx) in torsion_angle.keys():
                torsion_angle[at2_idx, at3_idx].append(math.cos(np.arctan2(y,x)))
            else:
                torsion_angle[at2_idx, at3_idx] = [math.cos(np.arctan2(y,x))]
        return torsion_angle
    
    def :
        
        data_path = self.conformer_paths[idx]
        data_idx = self.conformer_idx[idx]

        item = np.load(data_path, allow_pickle=True)
        atom_coor = item["xyz"][data_idx]  # each conformer
        temp = list(item["elems"])
        for _ in range(temp.count("H")):
            temp.remove("H")
        info['lig_atoms'] = np.array(temp)
        info['lig_numH'] = item["numH"]
        info['energy'] = item["energy"][data_idx]  # each conformer
        
        mol = self.mol_list[idx]
        try:
            info['mol_atom_degree']   = np.array([atom.GetDegree() for atom in mol.GetAtoms()]) # Verified
            info['mol_atom_hybrid']   = np.array([self.convert_hybrid_type(atom.GetHybridization().name, data_path) for atom in mol.GetAtoms()]) # String is converted
            info['mol_atom_aroma']    = np.array([atom.GetIsAromatic() for atom in mol.GetAtoms()]) # Bool
            info['mol_lig_bond']      = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), self.convert_bond_type(bond.GetBondType().name)] for bond in mol.GetBonds()])
            info['mol_vdw_relation']  = np.array([[atom_one, atom_two, atom_dist] for atom_one, atom_two, atom_dist in self.find_vdw_relation(mol, atom_coor)])
            info['torsion_idx_angle'] = self.cal_torsion_angle(mol, atom_coor)
        except (IndexError):
            print(data_path)

        return info
