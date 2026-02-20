import numpy as np
from pathlib import Path
import math
import scipy
import glob

AMINOACID = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
             "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

ALL_AAS = ["UNK"] + AMINOACID

gentype2num = { "CS": 0,"CS1": 1,"CS2": 2,"CS3": 3,"CD": 4,"CD1": 5,"CD2": 6,"CR": 7,"CT": 8,
                "CSp": 9,"CDp": 10,"CRp": 11,"CTp": 12,"CST": 13,"CSQ": 14,
                "HO": 15,"HN": 16,"HS": 17,
                # Nitrogen
                "Nam": 18,"Nam2": 19,"Nad": 20,"Nad3": 21,"Nin": 22,"Nim": 23,"Ngu1": 24,"Ngu2": 25,
                "NG3": 26,"NG2": 27,"NG21": 28,"NG22": 29,"NG1": 30,
                # Oxygen
                "Ohx": 31,"Oet": 32,"Oal": 33,"Oad": 34,"Oat": 35,"Ofu": 36,"Ont": 37,"OG2": 38,"OG3": 39,"OG31": 40,
                # S/P
                "Sth": 41,"Ssl": 42,"SR": 43,"SG2": 44,"SG3": 45,"SG5": 46,"PG3": 47,"PG5": 48,
                # Halogens
                "Br": 49,"I": 50,"F": 51,"Cl": 52,"BrR": 53,"IR": 54,"FR": 55,"ClR": 56,
                # Metals
                "Ca2p": 57,"Mg2p": 58,"Mn": 59,"Fe2p": 60,"Fe3p": 60,"Zn2p": 61,"Co2p": 62,"Cu2p": 63,"Cd": 64,
}

def fa2gentype(fats):
    """
    Mapping atypes to "gentype2num"

    Parameters:
        fats (iterable): atypes list
    """
    gts = {
        "Nbb": "Nad",
        "Npro": "Nad3",
        "NH2O": "Nad",
        "Ntrp": "Nin",
        "Nhis": "Nim",
        "NtrR": "Ngu2",
        "Narg": "Ngu1",
        "Nlys": "Nam",
        "CAbb": "CS1",
        "CObb": "CDp",
        "CH1": "CS1",
        "CH2": "CS2",
        "CH3": "CS3",
        "COO": "CDp",
        "CH0": "CR",
        "aroC": "CR",
        "CNH2": "CDp",
        "OCbb": "Oad",
        "OOC": "Oat",
        "OH": "Ohx",
        "ONH2": "Oad",
        "S": "Ssl",
        "SH1": "Sth",
        "HNbb": "HN",
        "HS": "HS",
        "Hpol": "HO",
        "Phos": "PG5",
        "Oet2": "OG3",
        "Oet3": "OG3",  # Nucleic acids
    }

    gents = []
    # if element not in gentype2num, then mapping to gentype2num using gts
    for at in fats:
        if at in gentype2num:
            gents.append(at)
        else:
            gents.append(gts[at])
    return gents

def get_AAtype_properties(ignore_hisH=True, extrapath="", extrainfo={}):
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
        atypes_aa[iaa] = fa2gentype([atypes[atm] for atm in atms])
        qs_aa[iaa] = q
        atms_aa[iaa] = atms
        bnds_aa[iaa] = bnds
        if aa in AMINOACID:
            repsatm_aa[iaa] = atms.index("CA")
        else:
            repsatm_aa[iaa] = repsatm

    if extrapath != "":
        params = glob.glob(extrapath + "/*params")
        for p in params:
            aaname = p.split("/")[-1].replace(".params", "")
            args = read_params(p, aaname=aaname)
            if not args:
                print("Failed to read extra params %s, ignore." % p)
                continue
            else:
                # print("Read %s for the extra res params for %s"%(p,aaname))
                pass
            atms, q, atypes, bnds, repsatm, nchi = args
            atypes = [atypes[atm] for atm in atms]  # same atypes
            extrainfo[aaname] = (q, atypes, atms, bnds, repsatm)
    if extrainfo != {}:
        print("Extra residues read from %s: " % extrapath, list(extrainfo.keys()))
    return qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa

def defaultparams(
    aa, datapath="/home/bbh9955/programs/Rosetta/residue_types", extrapath=""
):
    """
    Get params path of aa

    Args:
        aa: element for getting params
    Return:
        path of params file
    """
    # First search through Rosetta database
    if aa in AMINOACID:
        p = "%s/l-caa/%s.params" % (datapath, aa)
        return p

    p = "%s/%s.params" % (extrapath, aa)
    if not os.path.exists(p):
        p = "%s/LG.params" % (extrapath)
    if not os.path.exists(p):
        sys.exit(
            "Failed to found relevant params file for aa:"
            + aa
            + ",  -- check if LG.params exits"
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
    pdb: Path,
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

def find_gentype2num(at):
    if at in gentype2num:
        return gentype2num[at]
    else:
        return 0

def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return 0  # UNK

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8,
                    "M": 2.3, #Mg or Mn
                    "Z": 2.3  #Zn
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

def dihedral(A, B, C, D):
    AB = B - A
    AC = C - A
    BC = C - B
    BD = D - B
    v1 = np.cross(AB, AC) #np.cross : 벡터의 외적
    v2 = np.cross(BC, BD)
    v1v2 = np.dot(v1, v2) #np.dot : 벡터 점의 곱
    len_v1 = np.sqrt(np.sum(v1*v1)) #np.sqrt : 제곱근, np.sum : 합
    len_v2 = np.sqrt(np.sum(v2*v2))
    radi = math.acos(v1v2 / (len_v1*len_v2)) #math.acos : radian구하기
    x = math.degrees(radi) #math.degrees : 라디안을 각도로 변환
    k = np.dot(np.cross(v1,v2), BC)
    sign = 1
    if k < 0:
        sign = -1
    return (sign*x)

def phi_cal_angle(xyz, num, num_p): #좌표찾고 phi 계산
    if 'C' in xyz[num_p] and 'N' in xyz[num] and 'CA' in xyz[num] and 'C' in xyz[num]:
        phi = dihedral(xyz[num_p]['C'], xyz[num]['N'], xyz[num]['CA'], xyz[num]['C'])
    else:
        phi = None
    return phi

def psi_cal_angle(xyz, num, num_n): #각각의 좌표 찾아서 psi계산
    if 'N' in xyz[num] and 'CA' in xyz[num] and 'C' in xyz[num] and 'N' in xyz[num_n]:
        psi = dihedral(xyz[num]['N'], xyz[num]['CA'], xyz[num]['C'], xyz[num_n]['N'])
    else:
        psi = None
    return psi

def cal_len(A, B): #길이 계산
    P = A - B
    length = np.sqrt(np.sum(P*P))
    return length

def correct_H(ss_dict,Hbonds,phi_dict,psi_dict):
    for a,b in Hbonds:
        if b-a != 4: continue
        for i in range(a,b+1):
            if i not in phi_dict and i not in psi_dict: continue
            
            if ((i not in phi_dict) and (-90<psi_dict[i]<30)):
                ss_dict[i] = "H"
            elif ((i not in psi_dict) and (-150<phi_dict[i]<-30)):
                ss_dict[i] = "H"
            elif ((-150<=phi_dict[i]<=-30) and (-90<=psi_dict[i]<=30)):
                ss_dict[i] = "H"

    return ss_dict

def correct_E(ss_dict, Hbonds, phi, psi):
    for x in ss_dict:
        if ss_dict[x] == 'H': continue
        if phi[x] > -20 or psi[x] < 45: ss_dict[x] = "C"

    for i,j in Hbonds:
        if ss_dict[i] == 'H' or ss_dict[j] == 'H': continue
        if (i-2, j-2) in Hbonds:
            for x in range(i-2,i+1):
                if x in ss_dict: ss_dict[x] = "E"
                
        elif (i+2, j-2) in Hbonds:
            for x in range(i,i+3):
                if x in ss_dict: ss_dict[x] = "E"
    
    for j,i in Hbonds:
        if ss_dict[i] == 'H' or ss_dict[j] == 'H': continue
        if (j-2, i-2) in Hbonds:
            for x in range(i-2,i+1):
                if x in ss_dict: ss_dict[x] = "E"
                
        elif (j+2, i-2) in Hbonds:
            for x in range(i-2,i+1):
                if x in ss_dict: ss_dict[x] = "E"
                    
    return ss_dict

def get_chain_SS3(xyz, reslist):
    # neutral values
    phi_dict = {res:120.0 for res in reslist}
    psi_dict = {res:-120.0 for res in reslist}
    Hbonds = []

    for i,res in enumerate(reslist):
        if i > 0 and res-1 in reslist:
            phi_dict[res] = phi_cal_angle(xyz, i, reslist.index(res-1))
            
        if i < len(reslist)-1 and res+1 in reslist:
            psi_dict[res] = psi_cal_angle(xyz, i, reslist.index(res+1))

    xyz_N = np.array([x['N'] for x in xyz])[None,:,:]
    xyz_O = np.array([x['O'] for x in xyz])[:,None,:]
    d = xyz_N - xyz_O
    Hbonds = np.where(np.sqrt(np.einsum('ijk,ijk->ij',d,d)) < 3.5) # N->O
    Hbonds = [(reslist[i],reslist[j]) for i,j in zip(Hbonds[0],Hbonds[1]) if abs(i-j) > 2]
 
    ss_dict = {res:'C' for res in reslist} #default
    ss_dict = correct_H(ss_dict,Hbonds,phi_dict,psi_dict)
    ss_dict = correct_E(ss_dict,Hbonds,phi_dict,psi_dict)

    SS3 = [ss_dict[res] for res in reslist]
    phi = [phi_dict[res] for res in reslist]
    psi = [psi_dict[res] for res in reslist]

    return SS3, phi, psi
