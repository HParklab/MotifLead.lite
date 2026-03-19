import copy
import numpy as np

class Argument:
    def __init__(self, modelname):
        self.silent = False
        self.mode = 'pair' # loss calc option; pair-ddG or single-dG
        
        self.modelname = modelname
        self.nbatch = 20
        self.max_epoch = 200
        self.datapath = '/ml/MotifLead/current/ddG/'
        self.dataf_train = 'data/train.txt'
        self.dataf_valid = 'data/valid.txt'
        self.label_f = 'data/dGlabels.txt'
        
        self.LR = 1.0e-4
        self.debug = False
        
        self.ball_radius = 4.0
        self.edgek = 16
        
        self.num_channels = 64 #shared across module
        self.randomize = 0.0
        
        #self.dGrange = np.arange(5.0,12.1,0.5) #in pK value; from 10 um ~ 1 pm
        #self.n_dGbins = len(self.dGrange)
        #self.dGw = np.array([0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 4.0, 6.0, 10.0, 10.0, 10.0])
        
        self.gradient_checkpoint = True
        self.sampling_mode = 'weighted'

        self.w = {'dG':0.3, 'ddG':1.0, 'reg':1.0e-8}
        
        self.model_args = {'num_node_features':74,
                           'num_layers': 4,
                           'num_edge_features':2,
                           'num_channels': self.num_channels,
                           'out_dim': 1,
                           'dropout_rate': 0.2}
        
        self.verbose = True

args_base = Argument( "base" )
