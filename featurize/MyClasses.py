from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import rdchem
from genpot import Molecule, Types, BasicClasses, AtomTypeClassifier

#Edge Feature
@dataclass
class BondInfo:
    is_conjugated: List[bool] = field(default_factory=lambda: [True]) #가변 데이터 타입의 필드에 기본값을 지정(list, set, dict)
    bond_type: dict = field(default_factory=lambda: {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1, 0],
        Chem.rdchem.BondType.UNSPECIFIED: [0, 0, 0, 0, 1]
    })
    
@dataclass
class EdgeFeat:
    smiles : str
    mol : Chem.Mol = field(init=False) #나중에 따로 입력을 받아야함. __post_init__사용
    idx : List[int]
    
    def __post_init__(self):
        if isinstance(self.smiles, str):
            self.mol = Chem.MolFromSmiles(self.smiles)
        elif isinstance(self.smiles, Chem.rdchem.Mol):
            self.mol = self.smiles
            
        self.bonds = []
        for b in self.mol.GetBonds():
            i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
            if i in self.idx and j in self.idx:
                self.bonds.append(b)
            
    def BndIdx(self) -> List[Tuple[int,int]]:
        return [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in self.bonds]
    
    def NumRot(self) -> List[int]:
        rot_idx= Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        rot_num = self.mol.GetSubstructMatches(rot_idx)
        rot_list=[0]*len(self.BndIdx())
        for i in range(len(self.BndIdx())):
            if self.BndIdx()[i] in rot_num:
                rot_list[i]=1
        return rot_list 
     
    def IsConjugated(self, bond_info: BondInfo) -> List[bool]:
        return [b.GetIsConjugated() == bond_info.is_conjugated[0] for b in self.bonds]
    
    def BndInfo(self, bond_type: BondInfo) -> List[List[int]]:
        return [bond_type.bond_type[b.GetBondType()] for b in self.bonds]

#Node Feature  
@dataclass
class NodeOneHot:
    OneHotHybrid: dict = field(default_factory= lambda:{
        rdchem.HybridizationType.S : [1,0,0,0,0],
        rdchem.HybridizationType.SP: [0,1,0,0,0],
        rdchem.HybridizationType.SP2: [0,0,1,0,0],
        rdchem.HybridizationType.SP3: [0,0,0,1,0],
        rdchem.HybridizationType.UNSPECIFIED: [0,0,0,0,1],
        rdchem.HybridizationType.OTHER: [0,0,0,0,1],
        rdchem.HybridizationType.SP2D: [0,0,0,0,1],
        rdchem.HybridizationType.SP3D: [0,0,0,0,1],
        rdchem.HybridizationType.SP3D2: [0,0,0,0,1]
    })
    
    func_listing: dict = field(default_factory= lambda: {'fr_Al_OH':0, #아마이드가 합성이 쉽다.
               'fr_Ar_OH':1,
               'fr_ester':2,
               'fr_ketone':3,
               'fr_ether':4,
               'fr_amide':5
               ,'fr_C_S':6
               ,'fr_sulfone':7
               ,'fr_sulfone2':8
               ,'fr_urea':9
               ,'fr_lactone':10
               ,'fr_phos_acid':11
               ,'fr_alkyl_carbamate':12
               ,'unknown':13})
    
    is_aromatic: List[bool] = field(default_factory= lambda: [True])
    
    is_ring: List[bool] = field(default_factory= lambda: [True])
    

@dataclass
class NodeFeat:
    # Aromatic, NumTermCH3, NumRing, GetHybrid, FuncG
    smiles: str
    mol: Chem.Mol = field(init=False)
    mode : str

    def __post_init__(self):
        if isinstance(self.smiles, str):
            self.mol = Chem.MolFromSmiles(self.smiles)
            #self.mol.UpdatePropertyCache(strict=False)
            #Chem.SanitizeMol(self.mol,sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
        elif isinstance(self.smiles, Chem.rdchem.Mol):
            self.mol = self.smiles

        #default setup
        self.idx = [i for i in range(self.mol.GetNumAtoms())]
        nump = self.NumPolEdge()
        
        if self.mode == 'polH':
            self.idx = [i for i,atom in enumerate(self.mol.GetAtoms()) if atom.GetAtomicNum() != 1 or nump[i] > 0]
        else: #heavy only
            self.idx = [i for i,atom in enumerate(self.mol.GetAtoms()) if atom.GetAtomicNum() != 1]
    
    def Aromatic(self, aroma: NodeOneHot) -> List[bool]:
        return np.array([ring.GetIsAromatic()==aroma.is_aromatic[0] for ring in self.mol.GetAtoms()])[self.idx]

    def NumCH3(self) -> List[bool]:
        terminalCH3=Chem.MolFromSmarts("[CX4H3]")
        calCH3=[False]*self.mol.GetNumAtoms()
        match = self.mol.GetSubstructMatches(terminalCH3)
        for i in match:
            calCH3[i[0]]=True
        return np.array(calCH3)[self.idx]

    def NumH(self) -> List[int]:
        numH = [0 for _ in range(self.mol.GetNumAtoms())]
        for b in self.mol.GetBonds():
            i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
            if self.mol.GetAtoms()[i].GetAtomicNum() == 1: numH[j] +=1
            if self.mol.GetAtoms()[j].GetAtomicNum() == 1: numH[i] +=1
        return np.array(numH)[self.idx]
    
    def NumPolEdge(self) -> List[int]:
        numP = [0 for _ in range(self.mol.GetNumAtoms())]
        for b in self.mol.GetBonds():
            i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
            if self.mol.GetAtoms()[i].GetAtomicNum() in [7,8]: numP[j] +=1
            if self.mol.GetAtoms()[j].GetAtomicNum() in [7,8]: numP[i] +=1
        return np.array(numP)[self.idx]
    
    def Is_Aryl(self) -> List[int]:
        aryl = [0 for _ in range(self.mol.GetNumAtoms())]
        aro  = self.Aromatic()
        for b in self.mol.GetBonds():
            i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
            if aro[i] and not aro[j]: aryl[j] = 1
            if aro[j] and not aro[i]: aryl[i] = 1
        return np.array(aryl)[self.idx]

    def Elems(self) -> List[str]:
        return np.array([atom.GetSymbol() for atom in self.mol.GetAtoms()])[self.idx]
    
    def Ring(self, ring: NodeOneHot) -> List[bool]:
        return np.array([atom.IsInRing()==ring.is_ring[0] for atom in self.mol.GetAtoms()])[self.idx]

    def Hybrid(self, hiv: NodeOneHot) -> List[List[int]]:
        return np.array([hiv.OneHotHybrid[atom.GetHybridization()] for atom in self.mol.GetAtoms()])[self.idx]
    
    def OneHotFuncG(self, func_dict: dict,funclist:NodeOneHot) -> List[List[int]]:
        func_list=func_dict
        one_hot=[[0]*len(funclist.func_listing) for _ in range(self.mol.GetNumAtoms())]
        for func, indices in func_list.items():
            func_idx = funclist.func_listing[func]
            for idx in indices:
                one_hot[idx][func_idx]=1
        return np.array(one_hot)[self.idx]

    def Coordinates(self):
        xyz = [self.mol.GetConformer().GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())]
        xyz = [[a.x,a.y,a.z] for a in xyz]
        return np.array(xyz)[self.idx]
    
    def GenAtomType(self) -> List[int]:
        # conjugate as double?
        bonds = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),int(b.GetBondTypeAsDouble()+0.6)) for b in self.mol.GetBonds()]
        elems = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        hybs  = [atom.GetHybridization() for atom in self.mol.GetAtoms()]

        genmol = Molecule.MoleculeClass(None,mol2fileobj="") #empty initialization
        genmol.initialize_by_info(elems, hybs, bonds)
        atypes = [atm.aclass for atm in genmol.atms]

        return np.array(atypes)[self.idx]
        
class GenPotFeat:
    def __init__(self, mol2, include_H=False):
        option = BasicClasses.OptionClass(['','-s',mol2])
        self.mol2name = mol2
        try:
            self.mol = Molecule.MoleculeClass(mol2, option)
        except:
            self.mol = None

        self.include_H = include_H

    def extract_features_using_genpot(self):
        if self.mol == None:
            return False
        
        features = {}
        mol = self.mol
        if self.include_H:
            natms = len(mol.atms)
        else:
            natms = mol.nheavyatm
            
        atype = np.array([atm.atype for atm in mol.atms[:natms]])

        features['Aromatic'] = np.zeros(natms)
        if len(mol.atms_aro) > 0:
            features['Aromatic'][np.array(mol.atms_aro)] = 1
            
        features['nCH3']     = (atype==4)
        features['Ring']     = np.zeros(natms)
        if len(mol.atms_ring) > 0:
            features['Ring'][np.array(mol.atms_ring)] = 1

        # check 9, 1-hot
        hyb = np.array([self.hybrid_shift(atm.hyb) for atm in mol.atms[:natms]])
        features['Hybrid']   = hyb
        features['elems']    = np.array([atm.atype for atm in mol.atms[:natms]])
        features['xyz']      = np.array(mol.xyz[:natms])
        features['atypes']   = np.array([atm.aclass for atm in mol.atms[:natms]])
        features['anames']    = [atm.name for atm in mol.atms[:natms]] #bookkeeping
        features['qs_lig']    = [atm.charge for atm in mol.atms[:natms]] #bookkeeping
        
        ## functional groups
        features['FuncG'], grps = self.assign_functional_group(mol,self.include_H)

        ## report for debugging purpose
        '''
        print(self.mol2name)
        for i in range(natms):
            print(i,features['anames'][i],
                  features['Hybrid'][i],features['elems'][i],
                  Types.ACLASS_ID[features['atypes'][i]],features['Hybrid'][i],
                  features['qs_lig'][i], ','.join(grps[i]))
        '''
        
        nH = np.zeros(natms)
        features['Bond_idx'] = [] #
        features['Is_conju']  = [] #conjugation or not
        features['Bond_info'] = [] #
        features['Num_rot'] = [] #rotable bond or not
        for bnd in mol.bonds:
            i,j,o = bnd.atm1, bnd.atm2, bnd.order
            if j >= natms: nH[i] += 1
            else:
                features['Bond_idx'].append([i,j])
                features['Is_conju'].append(bnd.is_conjugated)
                border_1hot = np.zeros(5)
                border_1hot[self.bond_order_shift(o)] = 1
                features['Bond_info'].append(border_1hot)
                
        features['numH']     = nH

        return features

    def bond_order_shift(self,o):
        shifted = {1:0,2:1,
                   3:2, #triple
                   4:3, #amide or aromatic
                   9:4} #unused
        return shifted[o]

    def hybrid_shift(self,o):
        shifted = {0:0,1:1,
                   2:2, #sp2
                   3:3, #sp3
                   5:4, #d-orbital
                   8:2,
                   9:2} #amide
        return shifted[o]
    
    def assign_functional_group(self,mol,include_H=False):
        Func = AtomTypeClassifier.FunctionalGroupClassifier()
        Func.apply_to_molecule(mol)

        key2type = {'alcohol':0, #0: aliphatic alcohol
                    'aryl-hydroxyl':1, #1: aromatic alcohol
                    'ester':2, #2: ester
                    'aryl-ester':3, #3: aryl-ether 
                    'ether':4, #4: ether
                    'aryl-ether':5, #7: aryl-ester
                    'amide':6, #amide
                    'aryl-amide': 7, #7: aryl-amide
                    'ketone/aldehyde':8, #8: ketone/aldehyde
                    'carboxylicacid':9, #9: carboxylate
                    'amine3':10, #10: primary amine
                    'azo':11,
                    'diazo':11, #11: azo
                    'phosphate':12, #12: phosphate
                    'sulfonate':13, #13: sulfonate
                    'Ring-aro':14,
                    'Ring-pucker':15,
        }

        if include_H:
            natm = len(mol.atms)
        else:
            natm = mol.nheavyatm
            
        funcgrp1hot = np.zeros((natm,len(key2type)))
        names = [atm.name for atm in mol.atms[:natm]]
        grps = {i:[] for i in range(natm)}

        for grp in mol.funcgrps:
            #print(len(grp.atms), grp.grptype, grp.grptype in key2type)
            if grp.grptype in key2type:
                atms = np.array([names.index(atm.name) for atm in grp.atms if not atm.is_H])
                grptypes = key2type[grp.grptype]
                for i in atms: grps[i].append(grp.grptype)
                funcgrp1hot[atms] = np.eye(len(key2type))[grptypes]
        return funcgrp1hot, grps
    
        
