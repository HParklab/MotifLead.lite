from typing import Union, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch.conv import EGNNConv

class EquivalentTransformer(nn.Module):
    def __init__(self,
                 channels: int,
                 nout: int,
                 edge_feat_dim: Optional[int] = 0,
                 num_layers: Optional[int] =2,
                 ) -> None:
        super().__init__()
        
        self.channels = channels
        self.num_layers = num_layers

        # EGNN should get <=2 layers because of node squashing probrom
        self.gcn_layers = nn.ModuleList([EGNNConv(in_size=channels,
                                                  hidden_size=channels,
                                                  out_size=channels,
                                                  edge_feat_size=0) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(channels)
        self.act_fn = nn.GELU()
        self.egnn_out_layer = nn.Linear(channels, nout)
       #nout이 기존 1에서 11로 바뀌어야함 
    def forward(self,
                G: dgl.DGLGraph,
                node_feat: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        coord_feat = G.ndata['xyz']
        
        for gcn in self.gcn_layers:
            node_feat, coord_feat = gcn(graph=G,
                                        node_feat=node_feat,
                                        coord_feat=coord_feat,
                                        edge_feat=edge_feat)
            node_feat = self.act_fn(node_feat)
            node_feat = self.layer_norm(node_feat)
            
        node_feat = self.egnn_out_layer(node_feat)
    
        #pred_soft = F.softmax(node_feat,dim=1) #softmax로 바꿔
        pred=node_feat
        
       #print(pred)
        return pred

class GraphConv(nn.Module):
    def __init__(self,
                 channels: int,
                 num_layers: int = 1,
                 edge_feat_dim: int = 0,
                 out_size: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(channels)

        self.gcn_layers = nn.ModuleList([EGNNConv(in_size=channels,
                                                 hidden_size=channels,
                                                 out_size=channels,
                                                 edge_feat_size=edge_feat_dim) for _ in range(num_layers)])
        
    def forward(self,
                graph: dgl.DGLGraph,
                node_feat: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None,
                coord_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for i in range(self.num_layers):
            gcn_layer = self.gcn_layers[i]
            node_feat, _ = gcn_layer(graph=graph,
                                     node_feat=node_feat,
                                     #edge_feat=edge_feat,
                                     coord_feat=coord_feat)
            #node_feat = self.layer_norm(node_feat)
        return node_feat

class ResidueAtomEncoding(nn.Module):
    def __init__(self,
                 encoder_args: Dict[str, Union[str, int]],
                 n_input_feats: int,
                 channels: int,
                 dropout_rate: float,
                 ) -> None:
        super().__init__()

        self.initial_node_embedding = nn.Sequential(nn.Linear(n_input_feats, channels),
                                                    nn.LayerNorm(channels),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout_rate),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(channels),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout_rate))
        self.graph_conv = GraphConv(**encoder_args)
        self.dropout = nn.Dropout(0.3) #dropout setting

    def forward(self, G: dgl.DGLGraph, do_dropout):
        node_feat = G.ndata['attr']
        edge_feat = G.edata['attr']
        xyz = G.ndata['xyz']

        node_feat = self.initial_node_embedding( node_feat ).squeeze()
        if do_dropout: node_feat = self.dropout(node_feat) #dropout setting
        node_feat = self.graph_conv(graph=G,
                                    node_feat=node_feat,
                                    #edge_feat=edge_feat,
                                    coord_feat=xyz)
        
        return node_feat, edge_feat
    
class MyModel(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        self.device = device

        self.encoder = ResidueAtomEncoding( encoder_args = args.encoder_args,
                                            n_input_feats = args.n_input_feats,
                                            channels = args.channels,
                                            dropout_rate = args.dropout_rate)
        
        self.decoder = EquivalentTransformer(**args.decoder_args)
        self.dropout = nn.Dropout(0.3) #dropout setting

    def forward(self, G, do_dropout):
        #torch.autograd.set_detect_anomaly(True)
        node_feat, edge_feat = self.encoder(G, do_dropout)
        if do_dropout: node_feat = self.dropout(node_feat) #dropout setting
       
        pred = self.decoder(G, node_feat, edge_feat) # only node feat is updated
    
        """
        if torch.isnan(pred).sum() !=0:
             print("pred is nan")
             print()
            
        else:
        """
        return pred

        
    
