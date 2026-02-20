import torch
import torch.nn.functional as F
import numpy as np

aaweight = np.zeros((20,20))
for iaa,l in enumerate(open('data/aaweight.txt')):
    words = l[1:-1].split()
    for j,word in enumerate(words):
        aaweight[iaa,j] = float(word)
    aaweight[iaa,j+1:] = aaweight[iaa,j]

def KL_div(mu, logvar): #regularizer
    # 1/2 sum[ sig^2 + mu^2 - log(sig^2) - 1) ]
    kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
    return kl_loss

def ReconstructionLoss(pos_out, neg_out):
    sigmoid = torch.nn.Sigmoid() 
    pos_loss = F.binary_cross_entropy(sigmoid(pos_out), torch.ones_like(pos_out), reduction='mean')
    neg_loss = F.binary_cross_entropy(sigmoid(neg_out), torch.zeros_like(neg_out),reduction='mean')
    return pos_loss+neg_loss

def MySigmoid(pred, label):
    loss = 1.0 - 1.0/(1+((pred-label)*(pred-label)).mean())
    return loss

def MyWeightedCCE(pred, label, aatype):
    aas = np.concatenate(aatype)-1
    aaw = torch.tensor(aaweight[aas]).to(pred.device)

    eps = 1.0e-9
    loss = torch.tensor(0.0).to(pred.device)
    for i,(aa,l,p) in enumerate(zip(aas,label,pred)):
        w = aaw[aa,l]
        loss += -w*torch.log(p[l]+eps)

    loss /= label.shape[0]
    return loss

def MaskedMSE(pred, label, mask):
    d = mask*(pred-label)*(pred-label)
    return d.mean()
    
def MyCappedMSE(pred, label, term_mask=torch.tensor([1.0,1.0,1.0,1.0])):
    d = pred-label
    sig = torch.nn.Sigmoid()
    sigd = 1 - sig(torch.clamp(label,max=3))*sig(d)
    #dampens if l > 0 and d >0, otherwise quadratic
    loss = sigd*d*d # decomposed
    
    term_mask = term_mask.to(pred.device)
    loss[:,:] = loss[:,:] * term_mask # hack for test
    
    return loss
    
def MyCappedMSE2(pred, label, term_mask=torch.tensor([1.0,1.0,1.0,1.0])):
    d = torch.nn.functional.smooth_l1_loss(pred,label,reduction='none')
    loss = torch.log(4.0) # too modest at > 100
    #loss = torch.sqrt(d+1.0) # sqrt backward grad issue
    #20/(1+exp(-0.025*x))-10 
    #loss = 50.0/(1.0+torch.exp(-0.005*d)) - 25.0
    #print(d[:10],loss[:10])
    term_mask = term_mask.to(pred.device)
    loss[:,:] = loss[:,:] * term_mask # hack for test
    return loss
    
    
    
