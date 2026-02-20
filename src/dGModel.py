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
        self.dGpred = EGNNpredictor( num_channels=args.num_channels,
                                     num_layers=args.num_layers,
                                     input_dim=args.num_node_features,
                                     num_edge_features=args.num_edge_features,
                                     outdim=args.out_dim )
        
    def forward(self, G1, G2, do_dropout=True):
        g1s = dgl.unbatch(G1)
        g2s = dgl.unbatch(G2)

        dG = torch.zeros(len(g1s),0).to(G1.device())
        
        for i,(g1,g2) in enumerate(zip(g1s,g2s)):
            dG[i,0] = self.dGpred( g1, do_dropout=do_dropout )
            dG[i,1] = self.dGpred( g2, do_dropout=do_dropout )
            
        return dG

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

        self.out_layer = nn.Linear(num_channels, outdim )

    def forward(self, G, h):
        h, _ = self.egnn(h, G.ndata['x'].squeeze(1), G.edges(), G.edata["0"].float())
        h = self.out_layer(h)
        
        return h
