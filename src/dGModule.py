import torch
import torch.nn as nn
import dgl

class dGModule(nn.Module):
    def __init__(self, args, debug):
        super().__init__()

        d = args['mid_dim']
        self.dropoutrate = args['dropout_rate']
        if self.dropoutrate > 1.0e-6:
            self.dropout = nn.Dropout(self.dropoutrate)
            
        self.debug = debug
        self.FFcomb = nn.Sequential(nn.Linear(args['in_channels'], d),
                                    nn.LayerNorm(d),
                                    nn.GELU())
        self.FFrec  = nn.Sequential(nn.Linear(args['in_channels'], d),
                                    nn.LayerNorm(d),
                                    nn.GELU())
        self.FFlig  = nn.Sequential(nn.Linear(args['in_channels'], d),
                                    nn.LayerNorm(d),
                                    nn.GELU())
        self.FFdG = nn.Linear(d, args['out_dim'])
        self.mode = args['mode']
        
        self.FFcross  = nn.Sequential(nn.Linear(d, d),
                                    nn.LayerNorm(d),
                                    nn.GELU())
        
    def forward(self, Gall, Grec, Glig, h_comb, h_rec, h_lig, do_dropout=False):
        #torch.autograd.set_detect_anomaly(True)
        if Gall.number_of_nodes() == 0:
            return []

        if do_dropout and self.dropoutrate > 1.0e-6:
            h_comb = self.dropout(h_comb)
            h_rec  = self.dropout(h_rec)
            h_lig  = self.dropout(h_lig)
        
        # N x c -> N x d
        h_comb = self.FFcomb( h_comb ) 
        h_rec  = self.FFrec( h_rec )   
        h_lig  = self.FFrec( h_lig )

        dGs = []
        nall, nrec, nlig = 0,0,0
        for b,(nr,nl,na) in enumerate(zip(Grec.batch_num_nodes(),Glig.batch_num_nodes(),Gall.batch_num_nodes())):
            # mean over nodes
            c_rec  = h_rec [nrec:nrec+nr].mean(axis=0)
            c_lig  = h_lig [nlig:nlig+nl].mean(axis=0)
            if self.mode == 'simple':
                c_comb = h_comb[nall:nall+na].mean(axis=0)
            elif self.mode == 'cross1':
                h_comb_r = h_comb[nall:nall+nr]
                h_comb_l = h_comb[nall+nr:nall+na]
                h_comb_x = torch.einsum('ic,jc->ijc',h_comb_r,h_comb_l)
                c_comb = self.FFcross(h_comb_x).mean(axis=(0,1))
                
            nall += na
            nrec += nr
            nlig += nl
            c_sum = c_comb + c_rec + c_lig
            dGs.append( self.FFdG(c_sum) )

        dGs = torch.stack(dGs).to(Gall.device)
        dGs = dGs.softmax(dim=-1)
        return dGs
