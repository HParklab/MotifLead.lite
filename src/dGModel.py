import torch
import torch.nn as nn
import sys
from src.egnn.egnn import EGNN

class dGModel(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.dGpred = EGNNpredictor( num_channels=args['num_channels'],
                                     num_layers=args['num_layers'],
                                     input_dim=args['num_node_features'],
                                     num_edge_features=args['num_edge_features'],
                                     out_dim=args['out_dim'])
        
    def forward(self, G1, b_ligmask, do_dropout=True):
        h = self.dGpred(G1, G1.ndata['0']).squeeze(-1)

        # b_ligmask: b x N; h: N x outdim; dGs: b x outdim
        dGs = torch.einsum("bi,i->b", b_ligmask, h)
        #dGs = dGs/b_ligmask.sum(axis=-1) # normalize by num ligand atoms

        #print(dGs.shape, h.shape, b_ligmask.shape, b_ligmask.sum(axis=-1).shape)
        '''
        g1s = dgl.unbatch(G1)

        dG = torch.zeros(len(g1s)).to(G1.device())
        
        for i,g1 in enumerate(g1):
            dG[i] = self.dGpred( g1, do_dropout=do_dropout )
        '''
            
        return dGs

class EGNNpredictor(nn.Module):
    def __init__(self,
                 num_channels,
                 num_layers,
                 input_dim,
                 num_edge_features,
                 out_dim,
                 ):
        super().__init__()

        #self.input_layer = nn.Linear( input_dim, num_channels ) 
        
        self.egnn = EGNN(
            in_node_nf=input_dim,
            hidden_nf=num_channels,
            out_node_nf=num_channels,
            n_layers=num_layers,
            in_edge_nf=num_edge_features)

        self.out_layer = nn.Linear(num_channels, out_dim )

    def forward(self, G, h):
        h, _ = self.egnn(h, G.ndata['x'].squeeze(1), G.edges(), G.edata["0"].float())
        h = self.out_layer(h)
        
        return h
