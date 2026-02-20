import numpy as np
import torch

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

# Tip atom definitions
AA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
             "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
             "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
             "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# Residue number definition
AMINOACID = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
             'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
             'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(AMINOACID[i], i) for i in range(len(AMINOACID))])
NUCLEICACID = ['ADE','CYT','GUA','THY','URA'] #nucleic acids

METAL = ['CA','ZN','MN','MG','FE','CD','CO','CU']
ALL_AAS = ['UNK'] + AMINOACID + NUCLEICACID + METAL
NMETALS = len(METAL)

N_AATYPE = len(ALL_AAS)

gentype2num = {'CS':0, 'CS1':1, 'CS2':2,'CS3':3,
               'CD':4, 'CD1':5, 'CD2':6,'CR':7, 'CT':8,
               'CSp':9,'CDp':10,'CRp':11,'CTp':12,'CST':13,'CSQ':14,
               'HO':15,'HN':16,'HS':17,
               # Nitrogen
               'Nam':18, 'Nam2':19, 'Nad':20, 'Nad3':21, 'Nin':22, 'Nim':23,
               'Ngu1':24, 'Ngu2':25, 'NG3':26, 'NG2':27, 'NG21':28,'NG22':29, 'NG1':30, 
               'Ohx':31, 'Oet':32, 'Oal':33, 'Oad':34, 'Oat':35, 'Ofu':36, 'Ont':37, 'OG2':38, 'OG3':39, 'OG31':40,
               #S/P
               'Sth':41, 'Ssl':42, 'SR':43,  'SG2':44, 'SG3':45, 'SG5':46, 'PG3':47, 'PG5':48, 
               # Halogens
               'Br':49, 'I':50, 'F':51, 'Cl':52, 'BrR':53, 'IR':54, 'FR':55, 'ClR':56,
               # Metals
               'Ca2p':57, 'Mg2p':58, 'Mn':59, 'Fe2p':60, 'Fe3p':60, 'Zn2p':61, 'Co2p':62, 'Cu2p':63, 'Cd':64}


def find_bin(x, bins):
    for i,b in enumerate(bins):
        if x < b: return i
    return len(bins)-1

def bin_it(x, bin_min=0.0, bin_max=5.0, num_classes=20, onehot=False):
    bin_size = (bin_max - bin_min)/num_classes
    x_bin_index = torch.div(x - bin_min, bin_size, rounding_mode='floor').long()
    if onehot:
        return torch.nn.functional.one_hot(x_bin_index, num_classes=num_classes)
    else:
        return x_bin_index

def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return 0 #UNK

def find_gentype2num(at):
    if at in gentype2num:
        return gentype2num[at]
    else:
        return 0 # is this okay?

def atype2hyb(atypes):
    hyb = [0 for _ in atypes]
    for i,a in enumerate(atypes):
        if a in ['C.3','N.3','N.4','O.3','S.3','P.3','N.p13']:
            hyb[i] = 3
        elif a in ['C.2','N.2','O.2','O.co2','S.2','N.am']:
            hyb[i] = 2
        elif a in ['C.1','N.1']:
            hyb[i] = 1
        elif a in ['C.ar','N.ar']:
            hyb[i] = 9
    return hyb

def dat2distribution(datlist,minval,maxval,binsize,normalize=True):
    n_bin = int((maxval-minval)/binsize)
    distr= np.zeros(n_bin,dtype=int)
    
    for i_dat in datlist:
        try:
            scale_dat=int((i_dat-minval)/binsize)
        except:
            continue
        if scale_dat >= n_bin:
            distr[-1]+=1
        elif scale_dat < 0:
            distr[0]+=1
        else:
            distr[scale_dat]+=1

    if normalize:
        distr = distr/np.sum(distr)
    return distr

def calc_AUC(Pt, Pf):
    if len(Pt) == 0 or len(Pf) == 0:
        return -1.0
    
    Tdstr = dat2distribution(Pt, 0.0, 1.0, 0.02)
    Fdstr = dat2distribution(Pf, 0.0, 1.0, 0.02)
    #for i,y in enumerate(Tdstr):
    #    x = 0.02*(0.5+i)
    #    print("%3d %4.2f %8.5f %8.4f"%(t,x,y,Fdstr[i]))
    P = np.array(Tdstr)
    Q = np.array(Fdstr)+0.00001
    #KLdiv = np.sum(P*np.log(P/Q+0.00001))

    label = [1.0 for _ in Pt] + [0.0 for _ in Pf]
    score = Pt + Pf
    auc = roc_auc_score(label, score)
    #fpr, tpr, thrs = roc_curve(label, score)

    return auc

def to_cuda(x, device):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device)

def rmsd(Y,Yp): # Yp: require_grads
    import torch
    device = Y.device
    Y = Y - Y.mean(axis=0)
    Yp = Yp - Yp.mean(axis=0)

    # Computation of the covariance matrix
    # put a little bit of noise to Y
    C = torch.mm(Y.T, Yp)
    
    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3]).to(device)
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))
    
    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    rY = torch.einsum('ij,jk->ik', Y, U)
    dY = torch.sum( torch.square(Yp-rY), axis=1 )

    rms = torch.sqrt( torch.sum(dY) / Yp.shape[0] )
    return rms, U

def make_pdb(atms,xyz,outf,header=""):
    out = open(outf,'w')
    if header != "":
        out.write(header)
        
    #ATOM      1  N   VAL A  33     -15.268  78.177  37.050  1.00 92.09      A    N
    form = "HETATM %5d%-4s UNK X %3d   %8.3f %8.3f %8.3f 1.00  0.00\n"
    for i,(atm,x) in enumerate(zip(atms,xyz)):
        #out.write("%-3s  %8.3f %8.3f %8.3f\n"%(atm,x[0],x[1],x[2]))
        if len(atm) < 4:
            atm = ' '+atm
        else:
            atm = atm[3]+atm[:3]
        out.write(form%(i,atm,1,x[0],x[1],x[2]))

    if header != "":
        out.write("ENDMDL\n")
    out.close()

def to_batched_tensor(arrays):
    maxN = max([len(a) for a in arrays])
    batched_array = torch.zeros(len(arrays),maxN)
    mask = torch.zeros_like(batched_array)
    for b,a in enumerate(arrays):
        batched_array[b,:len(a)] = torch.tensor(a).float()
        mask[b,:len(a)] = 1.0
    return batched_array, mask

def generate_pose(Y, keyidx, xyzfull, atms=[], prefix=None):
    import torch
    Yp = xyzfull[keyidx]
    # find rotation matrix mapping Y to Yp
    T = torch.mean(Yp - Y, dim=0)

    com = torch.mean(Yp,dim=0)
    rms,U = rmsd(Y,Yp)
    
    Z = xyzfull - com
    T = torch.mean(Y - Yp, dim=0) + com
    
    Z = torch.einsum( 'ij,jk -> ik', Z, U.T) + T # aligned xyz

    if prefix != 'None':
        make_pdb(atms,xyzfull,"%s.input.pdb"%prefix)
        if isinstance(keyidx,torch.Tensor):
            keyidx = keyidx.cpu().detach().numpy()
            
        make_pdb(atms[keyidx],Y,'%s.predkey.pdb'%prefix)
        make_pdb(atms, Z, "%s.al.pdb"%prefix) #''',header="MODEL %d\n"%epoch''')
        
    return rms, atms[keyidx]

def report_attention(grids, A, epoch, modelname):
    K=A.shape[1]
    print(A.shape)
    print(K)
    form = "HETATM %5d %-3s UNK X %3d    %8.3f%8.3f%8.3f 1.00%6.2f\n"
    #ATOM      1  N   VAL A  33     -15.268  78.177  37.050  1.00 92.09      A    N
    #HETATM     0 O   UNK X   1       0.000   64.000  71.000 1.00  0.00
    for k in range(K):
        out = open("pdbs/attention%d.epoch%02d.pdb"%(k,epoch),'w')
        out.write("MODEL %d\n"%epoch)
        for i,(x,p) in enumerate(zip(grids,A[k])):
            # print("x:",x,'\n','p:',p)
            out.write(form%(i,"O",1,x[0],x[1],x[2],p))
        out.write("ENDMDL\n")
        out.close()
        
    out = open("pdbs//attentionAll.epoch%02d.pdb"%(epoch),'w')
    out.write("MODEL %d\n"%epoch)

    print(grids, '\n', A)
    # print(max(A[:,i]))
    for i,x in enumerate(grids):
        # print(i,x)
        # print((i,"O",1,x[0],x[1],x[2],max(A[:,i])))
        out.write(form%(i,"O",1,x[0],x[1],x[2],max(A[:,i])))
    out.write("ENDMDL\n")
    out.close()

def show_how_attn_moves(Z, epoch):

    # Z = np.load(npyfile)
    Z = Z[:int(len(Z)/2)]
    nrows, ncols = Z.shape
    X = np.linspace(1, ncols, ncols)
    Y = np.linspace(1, nrows, nrows)
    X,Y = np.meshgrid(X,Y)

    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    
    # Creating plot
    surf = ax.plot_surface(X, Y, Z , cmap='viridis')
    ax.set_zticks([0, 0.0005, 0.05])

    fig.colorbar(surf, shrink=0.6, aspect=8)

    plt.tight_layout()
    # plt.show()
    plt.savefig('../plotpngs/epoch_%d.png'%epoch)
    print('plotpngs/epoch_%d.png with %d points saved'%(epoch,int(len(Z))))

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    import scipy
    
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8}
    
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
        center_exp = center[None,:].repeat(n_neigh,axis=0)
        d2cen = np.sum( (center_exp - xyz[neigh]) ** 2, axis=1)
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

def read_mol2(mol2,drop_H=False):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
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
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS: elem = 'Null'
            
            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            # if words[3] == 'du' or 'un': rint(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs,dtype=float)[nonHid], np.array(atms)[nonHid]

def read_mol2_batch(mol2,tags_read=None,drop_H=True,tag_only=False,):
    read_cont = 0
    qs_s = {}
    elems_s = {}
    xyzs_s = {}
    bonds_s = {}
    borders_s = {}
    atms_s = {}
    nneighs_s = {}
    atypes_s = {}
    tags = []
    elems,tag = None, None

    cont = open(mol2).readlines()
    il = [i for i,l in enumerate(cont) if l.startswith('@<TRIPOS>MOLECULE')]#+[len(cont)]
    ihead = np.zeros(len(cont),dtype=bool)
    ihead[il] = True

    for il,l in enumerate(cont):
        if l.startswith('#'): continue

        if ihead[il] or il == len(cont)-1: 
            # summarize
            add_this_tag = ((tags_read == None) or (tag in tags_read))
            if read_cont > 0 and add_this_tag: #skip at the beginning
                if not tag_only:
                    nneighs = [[0,0,0,0] for _ in qs]
                    for i,j in bonds:
                        if elems[i] in ['H','C','N','O']:
                            k = ['H','C','N','O'].index(elems[i])
                            nneighs[j][k] += 1.0
                        if elems[j] in ['H','C','N','O']:
                            l = ['H','C','N','O'].index(elems[j])
                            nneighs[i][l] += 1.0

                    # drop hydrogens
                    if drop_H:
                        nonHid = [i for i,a in enumerate(elems) if a != 'H']
                    else:
                        nonHid = [i for i,a in enumerate(elems)]

                    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
                    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

                    # override 
                    elems_s[tag] = np.array(elems)[nonHid]
                    qs_s[tag] = np.array(qs)[nonHid]
                    bonds_s[tag] = bonds
                    borders_s[tag] = borders
                    xyzs_s[tag] = np.array(xyzs)[nonHid]
                    nneighs_s[tag] = np.array(nneighs,dtype=float)[nonHid]
                    atms_s[tag] = list(np.array(atms)[nonHid])
                    atypes_s[tag] = np.array(atypes)[nonHid]

            if il != len(cont)-1:
                read_cont = 3
                tag = cont[il+1][:-1]
                if tag not in tags: tags.append(tag)
                
            # reset
            qs,elems,xyzs,bonds,borders,atms,atypes = [],[],[],[],[],[],[]
            continue

        if (not ihead[il+1] and len(l) <=1) or tag == None: continue
 
        if read_cont == 3:
            if tags_read == None or tag in tags_read:
                read_cont = 4
            else:
                read_cont = -1
            
        if read_cont < 0 or tag_only: continue
        
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        elif l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        elif l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            read_cont = 0
            continue
        elif l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if read_cont == 1:
            words = l[:-1].split()
            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS: elem = 'Null'
            
            atms.append(words[1])
            atypes.append(words[5])
            elems.append(elem)
            if len(words) >= 9:
                qs.append(float(words[-1]))
            else:
                qs.append(0.0)
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            words = l[:-1].split()
            if len(words) < 3: continue
            # if words[3] == 'du' or 'un': print(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    if tags_read != None:
        tags_order = [tag for tag in tags_read if tag in tags] # reorder following input
    else:
        tags_order = tags

    if not tag_only:
        elems_s   = [elems_s[tag] for tag in tags_order if tag in tags]
        qs_s      = [qs_s     [tag] for tag in tags_order if tag in tags]
        bonds_s   = [bonds_s  [tag] for tag in tags_order if tag in tags]
        borders_s = [borders_s[tag] for tag in tags_order if tag in tags]
        xyzs_s    = [xyzs_s   [tag] for tag in tags_order if tag in tags]
        nneighs_s = [nneighs_s[tag] for tag in tags_order if tag in tags]
        atms_s    = [atms_s   [tag] for tag in tags_order if tag in tags]
        atypes_s  = [atypes_s [tag] for tag in tags_order if tag in tags]

    return elems_s, qs_s, bonds_s, borders_s, xyzs_s, nneighs_s, atms_s, atypes_s, tags_order

def read_mol2s_xyzonly(mol2):
    read_cont = 0
    xyzs = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            xyzs.append([])
            atms.append([])
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue
        
        if l.startswith('@<TRIPOS>BOND'): 
            read_cont = 2
            continue

        words = l[:-1].split()
        if read_cont == 1:
            is_H = (words[1][0] == 'H')
            if not is_H:
                atms[-1].append(words[1])
                xyzs[-1].append([float(words[2]),float(words[3]),float(words[4])]) 

    return np.array(xyzs), atms

def cosine_angle(v1, v2) -> float:
    t = v1.dot(v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(t)


