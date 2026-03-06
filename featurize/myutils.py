import numpy as np
import os
import scipy

# 31 AA types
AMINOACID = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
             "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

ALL_AAS = ["UNK"] + AMINOACID + ["NA","K","CA","FE","MG","MN","ZN","CO","CU","CD"]

# 41 atomtypes
atype2num = { "Null": 0,
              "CNH2": 1, "COO": 1,"CH0": 2,"CH1": 3,"CH2": 4,"CH3": 5,"aroC": 6,
              "Ntrp": 7, "Nhis": 8, "NtrR": 9, "NH2O": 10, "Nlys": 11, "Narg": 12, "Npro": 13,
              "OH" : 15, "ONH2": 16, "OOC": 17,
              "S" : 19, "SH1": 20,
              "Nbb":10, "CAbb":3, "CObb":1, "OCbb":1,
              "Hpol":23, "HS":24, "Hapo":24, "Haro":24, "HNbb":23, "H":23,
              
              #SYBYL type from here
              "C.3": 5, "C.2" : 4, "C.1": 3, "C.ar": 6, "C.cat": 2,
              "N.3": 11, "N.2": 10, "N.1":14, "N.ar":7, "N.am":10, "N.pl3":13,
              "O.3": 15, "O.2": 16, "O.co2": 17, "O.ar":18,
              "S.3": 20, "S.2": 19, "S.O": 21, "S.O2": 22,
              "P.3": 25,
              "I": 26, "F": 27, "Cl": 28, "Br":29, "B":30,
              # Metals
              "Na":31,
              "K": 32,
              "Ca": 33, "Ca2p": 33,
              "Fe": 34, "Fe2p": 34,"Fe3p": 34,
              "Mg": 35, "Mg2p": 35,"Mn": 36,
              "Zn": 37, "Zn2p": 37,
              "Co2p": 38,"Cu2p": 39,"Cd": 40}

def get_AAtype_properties(ignore_hisH=True):
    """
    Get properties of atypes

    Return: qs_aa(dict), atypes_aa(dict), atms_aa(dict), bnds_aa(dict), repsatm_aa(dict)
        each dictionary has 32 number keys (AMINOACID+NUCLEICACID+METAL) with starting number 1.
        "0" key means "UNK".
        atypes_aa[i]->dict: dictionary of more specific atom types (e.g. Nbb, CObb etc) (atom: atype)
        atms_aa[i]->list: atom list
        bnds_aa[i]->list: list of atom set tuples that have connection
        repsatm_aa[i]->int: index of representative atom
    """
    qs_aa = {}
    atypes_aa = {}
    atms_aa = {}
    bnds_aa = {}
    repsatm_aa = {}

    iaa = 0  # "UNK"
    for aa in AMINOACID:
        iaa += 1
        p = defaultparams(aa)
        atms, q, atypes, bnds, repsatm, _ = read_params(p)
        atypes_aa[iaa] = [find_atype(atypes[atm]) for atm in atms]
        qs_aa[iaa] = q
        atms_aa[iaa] = atms
        bnds_aa[iaa] = bnds
        if aa in AMINOACID:
            repsatm_aa[iaa] = atms.index("CA")
        else:
            repsatm_aa[iaa] = repsatm

    return qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa

def defaultparams( aa ):
    """
    Get params path of aa

    Args:
        aa: element for getting params
    Return:
        path of params file
    """
    # First search through Rosetta database
    datapath = os.path.dirname(os.path.abspath(__file__))+'/params/'
    
    if aa in AMINOACID:
        p = "%s/%s.params" % (datapath, aa)
        return p

    p = "%s/%s.params" % (extrapath, aa)
    if not os.path.exists(p):
        sys.exit(
            "Failed to found relevant params file for aa:"
            + aa
        )
    return p

def read_params(
    p: str,
    as_list: bool = False,
    ignore_hisH: bool = True,
    aaname=None,
    read_mode: str = "polarH",
):
    """
    Parsing the params file

    Args:
        p: path of the params file
        as_list: if True, return list type of qs and atypes
    Return:
        atms(list), qs(dict), atypes(dict), bnds(list), repsatm(int), nchi(int)
        atms: atom list
        qs: partial charge (in this research, we used MMFF94)
        atypes: more specific atom types (e.g. Nbb, CObb etc)
        bnds: list of atom set tuples that have connection
        repsatm: NBR_ATOM index of atms
    """
    atms = []
    qs = {}
    atypes = {}
    bnds = []

    is_his = False
    repsatm = 0
    nchi = 0
    for l in open(p):
        words = l[:-1].split()
        if l.startswith("AA"):
            if "HIS" in l:
                is_his = True
        elif l.startswith("NAME"):
            aaname_read = l[:-1].split()[-1]
            if aaname is not None and aaname_read != aaname:
                return False

        if l.startswith("ATOM") and len(words) > 3:
            atm = words[1]
            atype = words[2]
            if atype[0] == "H":
                if read_mode == "heavy":
                    continue
                elif atype not in ["Hpol", "HNbb", "HO", "HS", "HN"]:
                    continue
                elif is_his and (atm in ["HE2", "HD1"]) and ignore_hisH:
                    continue

            if atype == "VIRT":
                continue
            atms.append(atm)
            atypes[atm] = atype
            qs[atm] = float(words[4])

        elif l.startswith("BOND"):
            a1, a2 = words[1:3]
            if a1 not in atms or a2 not in atms:
                continue
            border = 1
            if len(words) >= 4:
                border = {
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "CARBOXY": 2,
                    "DELOCALIZED": 2,
                    "ARO": 4,
                    "4": 4,
                    "3": 3,
                }[words[3]]

            bnds.append((a1, a2))  # ,border))

        elif l.startswith("NBR_ATOM"):
            repsatm = atms.index(l[:-1].split()[-1])
        elif l.startswith("CHI"):
            nchi += 1
        elif l.startswith("PROTON_CHI"):
            nchi -= 1

    if as_list:
        qs = [qs[atm] for atm in atms]
        atypes = [atypes[atm] for atm in atms]
    return atms, qs, atypes, bnds, repsatm, nchi

def read_pdb(
    pdb,
    read_ligand: bool = False,
    aas_allowed: list = [],
    aas_disallowed: list = [],
    ignore_insertion: bool = True,
):
    """
    Parsing PDB file (read only target and ligand).

    Args:
        pdb: path of PDB file for parsing
    Return:
        resnames(list), reschains(list), xyz(dict), atms(dict)
        resnames: list of residue name (e.g. ['SER', 'ILE', ..])
        reschains: list of residue chain (e.g. [['A.1', 'A.2', ..])
        xyz: coordinate (e.g. {'A.1': {'N': [59.419, 26.851, 14.79], 'CA': [...], ...})
        atms: residue chain's atom list (e.g. {'A.1': ['N', 'CA', ...], 'A.2': [...], ...})
    """
    resnames = []
    reschains = []
    xyz = {}
    atms = {}

    for l in open(pdb):
        if not (l.startswith("ATOM") or l.startswith("HETATM")):
            continue
        atm = l[12:17].strip()
        aa3 = l[17:20].strip()

        if aas_allowed != [] and aa3 not in aas_allowed:
            continue
        if aa3 in aas_disallowed:
            continue

        reschain = l[21] + "." + l[22:27].strip()
        if ignore_insertion and l[26] != " ":
            continue

        if aa3 in AMINOACID:
            if atm == "CA":
                resnames.append(aa3)
                reschains.append(reschain)
        elif read_ligand and aa3 != "LG1":
            continue
        elif (
            read_ligand and reschain not in reschains
        ):  # "reschain not in reschains:" -> append only once
            resnames.append(aa3)  # LG1
            reschains.append(reschain)  # X.1

        if reschain not in xyz:
            xyz[reschain] = {}
            atms[reschain] = []
        xyz[reschain][atm] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        atms[reschain].append(atm)

    return resnames, reschains, xyz, atms

def read_mol2(mol2,drop_H=False):
    read_cont = 0

    qs = []
    atypes = []
    bonds = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            atm = words[1]
            atype = find_atype(words[5])
            
            atms.append(atm)
            atypes.append(atype)
            qs.append(float(words[-1]))
                
        elif read_cont == 2:
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            #bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0} 
            #borders.append(bondtypes[words[3]]) #unused

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(atms) if a[0] != 'H']
    else:
        nonHid = [i for i,a in enumerate(atms)]


    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

    feats = {}
    feats['qs'] = {atm:qs[i] for i,atm in enumerate(atms) if i in nonHid}
    feats['atypes'] = np.array(atypes)[nonHid]
    feats['atms'] = np.array(atms)[nonHid]
    feats['bnds'] = [(feats['atms'][i], feats['atms'][j]) for i,j in bonds]
    
    return feats
    

def find_atype(at):
    if at in atype2num:
        return atype2num[at]
    else:
        return 0

def atype2elem(atype):
    if atype == 0:
        return "X"
    elif atype <= 6:
        return "C"
    elif atype <= 14:
        return "N"
    elif atype <= 18:
        return "O"
    elif atype <= 22:
        return "S"
    elif atype <= 24:
        return "H"
    elif atype == 25:
        return "P"
    elif atype == 26:
        return "I"
    elif atype == 27:
        return "F"
    elif atype == 28:
        return "Cl"
    elif atype == 29:
        return "Br"
    elif atype <= 40:
        return "M"
    else:
        return 0
    
def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return -1  # UNK

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8,
                    "M": 2.3, #Mg or Mn
                    "X": 0,
    }
    areas = []
    normareas = []
    centers = xyz
    radii = np.array([atomic_radii[e] for e in elems])
    n_atoms = len(elems)

    inc = np.pi * (3 - np.sqrt(5)) # increment
    off = 2.0/n_samples

    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y*y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)

    kd = scipy.spatial.cKDTree(xyz)
    neighs = kd.query_ball_tree(kd, 8.0)

    occls = []
    for i,(neigh, center, radius) in enumerate(zip(neighs, centers, radii)):
        neigh.remove(i)
        n_neigh = len(neigh)
        d2cen = np.sum((center[None,:].repeat(n_neigh,axis=0) - xyz[neigh]) ** 2, axis=1)
        occls.append(d2cen)

        pts = pts0*(radius+probe_radius) + center
        n_neigh = len(neigh)

        x_neigh = xyz[neigh][None,:,:].repeat(n_samples,axis=0)
        pts = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)

        d2 = np.sum((pts - x_neigh) ** 2, axis=2) # Here. time-consuming line
        r2 = (radii[neigh] + probe_radius) ** 2
        r2 = np.stack([r2] * n_samples)

        # If probe overlaps with just one atom around it, it becomes an insider
        n_outsiders = np.sum(np.all(d2 >= (r2 * 0.99), axis=1))  # the 0.99 factor to account for numerical errors in the calculation of d2
        # The surface area of   the sphere that is not occluded
        area = 4 * np.pi * ((radius + probe_radius) ** 2) * n_outsiders / n_samples
        areas.append(area)

        norm = 4 * np.pi * (radius + probe_radius)
        normareas.append(min(1.0,area/norm))

    occls = np.array([np.sum(np.exp(-occl/6.0),axis=-1) for occl in occls])
    occls = (occls-6.0)/3.0 #rerange 3.0~9.0 -> -1.0~1.0
    return areas, np.array(normareas), occls

