import torch,sys,os
#import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]

def read_key(checkpoint,key,n):
    aV = checkpoint['train_loss'][key]
    bV = checkpoint['valid_loss'][key]
    if isinstance(aV,list):
        E = np.array([np.mean(a) for a in aV])
        F = np.array([np.mean(a) for a in bV])
    else:
        E = np.mean(aV,axis=1)
        F = np.mean(bV,axis=1)
    if len(E) == 2*n: E = E[::2]
    if len(F) == 2*n: F = F[::2]
    if len(E) == 4*n: E = E[::4]
    if len(F) == 4*n: F = F[::4]
    return E,F

def refit_if_short(A,I):
    if len(I) < len(A):
        I = np.concatenate([[0 for _ in range(len(A)-len(I))],I])

def read_f(f):
    checkpoint = torch.load(f,map_location=torch.device('cpu'))
    aG = checkpoint['train_loss']['total']
    bG = checkpoint['valid_loss']['total']

    if isinstance(aG,list):
        X  = np.array([np.mean(a) for a in aG])
        Y  = np.array([np.mean(a) for a in bG])
    else:
        X  = np.mean(aG,axis=1)
        Y  = np.mean(bG,axis=1)

    A,B,E,F,G,H,I,J,K,L,O,P = [],[],[],[],[],[],[],[],[],[],[],[]
    M,N = [],[]
    
    #additional
    if 'Srec' in checkpoint['train_loss'] and len(checkpoint['train_loss']['Srec'])>0:
        A,B = read_key(checkpoint,'Srec',len(X))
        
    if 'Hrec' in checkpoint['train_loss'] and len(checkpoint['train_loss']['Hrec'])>0:
       E,F = read_key(checkpoint,'Hrec',len(X))
        
    if 'Slig' in checkpoint['train_loss'] and len(checkpoint['train_loss']['Slig'])>0:
       G,H = read_key(checkpoint,'Slig',len(X))
       
    if 'Hlig' in checkpoint['train_loss'] and len(checkpoint['train_loss']['Hlig'])>0:
       I,J = read_key(checkpoint,'Hlig',len(X))

    if 'Hinter' in checkpoint['train_loss'] and len(checkpoint['train_loss']['Hinter'])>0:
       K,L = read_key(checkpoint,'Hinter',len(X))
           
    if 'dG' in checkpoint['train_loss'] and len(checkpoint['train_loss']['dG'])>0:
       M,N = read_key(checkpoint,'dG',len(X))

    if 'KL' in checkpoint['train_loss'] and len(checkpoint['train_loss']['KL'])>0:
       O,P = read_key(checkpoint,'KL',len(X))

    for a in [I,J,K,L,M,N,O,P]:
        refit_if_short(A,a)
       
    R = []
    if 'reg' in checkpoint['train_loss']:
        R = checkpoint['train_loss']['reg']
        if isinstance(R,list):
            R  = [np.mean(a) for a in R]
        else:
            R  = np.mean(R,axis=1)

    return A,B,X,Y,E,F,G,H,I,J,K,L,M,N,O,P,R

if os.path.exists('%s/model.pkl'%path):
    args = read_f('%s/model.pkl'%path)
elif os.path.exists('models/%s/model.pkl'%path):
    args = read_f('models/%s/model.pkl'%path)
else:
    sys.exit("no pkl found")
A,B,X,Y,E,F,G,H,I,J,K,L,M,N,O,P,R = args

imin = np.argmin(Y)
cont = []
for i in range(len(X)):
    extra = ''
    if i == imin: extra = '*'
    form = "%3d %8.4f %8.4f %8.4f "
    Y2 = [y for y in Y if y == y]
    l = form%(i,X[i],Y[i],min(Y[:i+1]))
 
    if len(A) == len(X) and len(A) > 0: l += ' : %8.4f %8.4f'%(A[i],B[i]) #Srec
    if len(E) == len(X) and len(E) > 0: l += ' : %8.4f %8.4f'%(E[i],F[i]) #unused
    if len(G) == len(X) and len(G) > 0: l += ' : %8.4f %8.4f'%(G[i],H[i]) #Slig
    if len(I) == len(X) and len(G) > 0: l += ' : %8.4f %8.4f'%(I[i],J[i]) #Hlig
    if len(K) == len(X) and len(K) > 0: l += ' - %8.4f %8.4f'%(K[i],L[i]) #Hinter
    if len(M) == len(X) and len(M) > 0: l += ' - %8.4f %8.4f'%(M[i],N[i]) #dG
    if len(O) == len(X) and len(O) > 0: l += ' : %8.4f %8.4f'%(O[i],P[i]) #
        
    if len(R) == len(X) and len(R) > 0: l += ' : %8.4f '%R[i]
    l += extra
    cont.append(l)
    print(l)

path2 = path.split('/')[-1]
out = open('traces/%s.trace'%path2,'w')
out.writelines('\n'.join(cont)+'\n')
out.close()

    
#plt.plot(A)
#plt.plot(B)
#plt.savefig("trainingcurve.png")
#plt.close()
