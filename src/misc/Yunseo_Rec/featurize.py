import torch
from torch import Tensor
import numpy as np
from rdkit import Chem
from abc import ABC, abstractmethod
from typing import Tuple


class BaseNodeFeaturizer(ABC):
    @abstractmethod
    def featurize(self) -> Tensor:
        pass


class BaseEdgeFeaturizer(ABC):
    @abstractmethod
    def featurize(self) -> Tensor:
        pass


class DefaultNodeFeaturizer(BaseNodeFeaturizer):
    def featurize(
        self,
        lig_numH: np.ndarray,
        lig_atoms: np.ndarray,
        entropy: np.ndarray,
        mol_atom_degree: np.ndarray,
        mol_atom_aroma: np.ndarray,
        mol_atom_hybrid: np.ndarray,
    ) -> Tensor:
        atom_node_feats = []
        for atom_num in range(len(lig_atoms)):
            node_numH = lig_numH[atom_num][1]
            node_atom = Chem.GetPeriodicTable().GetAtomicNumber(lig_atoms[atom_num])
            node_degree = mol_atom_degree[atom_num]
            node_aroma = mol_atom_aroma[atom_num]
            node_hybrid = mol_atom_hybrid[atom_num]
            # entropy: vibent, rotent, confent, transent
            # node_hybrid: one-hot encoded, concatenate with others
            atom_node_feats.append(list(node_hybrid)+[node_numH, node_atom, node_degree, node_aroma])
        return torch.tensor(atom_node_feats, dtype=torch.float32)


class DefaultEdgeFeaturizer(BaseEdgeFeaturizer):
    def featurize(self, mol_lig_bond: np.ndarray, mol_vander_relation: np.ndarray, torsion_idx_angle) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            - atom_edge_idx: edge indices
            - atom_bond_feats: edge features
        """
        atom_edge_idx, atom_bond_feats = [], []
        for b_start, b_end, b_type in mol_lig_bond:
            atom_edge_idx.append([b_start, b_end])  # Modify 1 to 0
            atom_edge_idx.append([b_end, b_start])
            
            if b_type == 1:
                b_type = list(np.eye(11)[0])
            if b_type == 2:
                b_type = list(np.eye(11)[1])
            if b_type == 3:
                b_type = list(np.eye(11)[2])
            if b_type == 4:
                b_type = list(np.eye(11)[3])            
            
            # torsion_idx_angle.keys() is 2nd, 3rd atom in 1 torsion group.
            if (b_start, b_end) in torsion_idx_angle.keys():
                for i in range(len(torsion_idx_angle[b_start, b_end])):
                    if len(torsion_idx_angle[b_start, b_end]) < 7:
                        b_type[i+5] = torsion_idx_angle[b_start, b_end][i]
                    else:
                        raise Exception("Torsion is more than 6")
            if (b_end, b_start) in torsion_idx_angle.keys():
                for i in range(len(torsion_idx_angle[b_end, b_start])):
                    if len(torsion_idx_angle[b_end, b_start]) < 7:
                        b_type[i+5] = torsion_idx_angle[b_end, b_start][i]
                    else:
                        raise Exception("Torsion is more than 6")
                
            atom_bond_feats.append(b_type)
            atom_bond_feats.append(b_type)
            
        # mol_vander_relation 내 순서 바꾼 것도 이미 들어가있음.
        for v_start, v_end, dist in mol_vander_relation:
            atom_edge_idx.append([int(v_start), int(v_end)])
            # atom_edge_idx.append([v_end, v_start])
            atom_bond_feats.append(np.eye(11)[4] * 10 * (dist**-7))
            # atom_bond_feats.append(list(np.eye(9)[4]))
        atom_edge_idx = np.array(atom_edge_idx, dtype=np.int64)
        atom_bond_feats = np.array(atom_bond_feats, dtype=np.float32)

        # numpy 배열을 torch.Tensor로 변환
        atom_edge_idx = torch.from_numpy(atom_edge_idx)
        atom_bond_feats = torch.from_numpy(atom_bond_feats)
        
        return atom_edge_idx, atom_bond_feats
