import numpy as np
import torch

def report_Hlig(pred, label,prefices):
    for P,L,prefix in zip(pred, label,prefices):
        print("Hlig (P/L) %-30s | "%prefix+" %5.3f %5.3f"%(P,L))
                
def report_Slig(pred, label, prefices):
    for P,L,prefix in zip(pred, label, prefices):
        if len(P) > 0 and len(L) > 0:
            print("Slig (P/L) %-30s | "%prefix+" %5.3f"*len(P)%tuple(P) + " | " + " %5.3f"*len(L)%tuple(L))

def report_Srec(pred, label, aas, resnames):
    resnames = np.concatenate(resnames)
    aas = np.concatenate(aas)
    for P,L,aa,rn in zip(pred,label,aas,resnames):
        L = int(L)
        print("Srec (P/L) %-12s %2d %2d %5.3f %2d | "%(rn,aa,L,P[L],torch.argmax(P))+" %5.3f"*len(P)%tuple(P))
    return 
    
def report_Hinter(pred, label,loss):
    for P,L,l in zip(pred,label,loss):
        print("Hinter (Loss,P/L) %7.2f %7.2f %7.2f %7.2f : %7.2f %7.2f %7.2f %7.2f"%(P[0],P[1],P[2],P[3],L[0],L[1],L[2],L[3])+" [ %7.2f %7.2f %7.2f %7.2f ]"%tuple(l))
    return 

def report_dG(pred, label, names):
    for P,L,n in zip(pred,label,names):
        L = int(L)
        print("dG %s (P/L) %2d %5.3f %2d | "%(n,L,P[L],torch.argmax(P))+" %5.3f"*len(P)%tuple(P))
    return 

