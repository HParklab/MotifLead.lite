import copy
import numpy as np

class Argument:
    def __init__(self, modelname, dropout_rate=0.3):
        self.silent = False
        
        self.modelname = modelname
        self.nbatch = 10
        self.max_epoch = 100
        self.datapath = '/ml/MotifLead/current/'
        self.dataf_train = 'data/train_tmp.txt'
        self.dataf_valid = 'data/valid_tmp.txt'
        self.LR = 1.0e-5
        self.debug = False
        self.ball_radius = 4.0
        self.edgek = [16,8,16]
        self.nchannels = 64 #shared across module
        self.randomize = 0.0
        self.w = {'dG':1.0,
                  'Srec':1.0, #CCE
                  'Hrec':0.0, 'Slig':1.0, 'Hlig':1.0, 'Hinter':5.0,
                  'KL':1.0, 'recon': 1.0, 'reg':1.0e-8}

        self.dGrange = np.arange(5.0,12.1,0.5) #in pK value; from 10 um ~ 1 pm
        self.n_dGbins = len(self.dGrange)
        self.dGw = np.array([0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 4.0, 6.0, 10.0, 10.0, 10.0])

        
        self.gradient_checkpoint = True
        
        self.ligand_args = {'channels': self.nchannels,
                            'num_layers': 2,
                            'n_input_feats': 32, #99 (# of nodefeat:3)
                            'edge_feat_dim': 10, #input
                            'dropout_rate': 0.2,
                            'latent_embedding_size': 32,
                            'output_dim':1, #hard-coded
                            'num_layers_encoder': 1,
                            'num_layers_decoder': 1,
                            'out_size': 64,
                            'num_layers_energy':2,
                            'num_layers_entropy':2,
                            'mode': 'simple',
                            'edge': 'bond',
                            'emb' : None,
                            'encoder': 'vae'
                            #'Slig_bins':20, #unused
        }

        self.receptor_args = {'n_input_feats':107, ##check #107 for pre-May2025; funcgrps added
                              'n_input_edge': 12, ##check
                              'channels': self.nchannels,
                              'dropout_rate':0.2,
                              'nout':1,
                              'edge_feat_dim':8, #edge_channel
                              'num_egnn_layers':5,
                              'num_graphconv_layers':1,
                              'Srec_bins':20,
                              'Srec_max':5.0,
        }

        self.interaction_args = {'n_input_feats':102+self.nchannels,
                                 'num_layers': 4,
                                 'edge_feat_dim':2,
                                 'channels': self.nchannels,
                                 'dropout_rate': 0.2,
                                 'elec_expansion': 0,
                                 'calc_solv_pair' : False
        }

        self.dG_args = {'in_channels': self.nchannels,
                        'mid_dim'    : 16,
                        'out_dim'    : self.n_dGbins,
                        'dropout_rate': 0.0,
                        'mode': 'simple'
        } # dGbins
        
        self.verbose = True

    def ligand_mode(self, mode):
        self.ligand_args['mode'] = mode
        if mode == 'vnode':
            self.ligand_args['n_input_feats'] += 1 #is_vnode

args_default = Argument("default")

args_devel = Argument("devel")
args_devel.dataf_train = "data/train_rec.txt"
args_devel.dataf_valid = "data/valid_rec.short.txt"

args_Hinter = Argument("inter")
args_Hinter.dataf_train = "data/train_inter.noerr.txt"
args_Hinter.dataf_valid = "data/valid_inter.noerr.txt"

args_mix = Argument("mix")
args_mix.dataf_train = "data/train_shuffle500dupl.txt"
args_mix.dataf_valid = "data/valid_tmp.txt"
args_mix.LR = 5.0e-6
args_mix.w['Hinter'] = 0.5 #10.0
args_mix.w['Hlig'] = 0.5 # 0.001
args_mix.w['Slig'] = 0.5 # 0.001
args_mix.w['Srec'] = 2.0 # 0.1

args_inter = Argument("mix_interonly")
args_inter.dataf_train = "data/train_inter.noerr.txt"
args_inter.dataf_valid = "data/valid_inter.noerr.txt"
args_inter.w['Hlig'] = 0.0
args_inter.w['Slig'] = 0.0
args_inter.w['Srec'] = 0.0

args_dG = copy.copy(args_inter)
args_dG.modelname = "dG"
args_dG.w['dG'] = 0.2
args_dG.w['Hinter'] = 5.0
args_dG.w['Hlig'] = 0.1
args_dG.w['Slig'] = 0.1
args_dG.w['Srec'] = 1.0
#args_dG.dataf_train = "data/train_dG.txt"
#args_dG.dataf_train = "data/train_shuffle500.txt"
args_dG.dataf_train = "data/train_shuffle500dG.txt"
args_dG.dataf_valid = "data/valid_alldG.txt"

args_dGsmooth = copy.copy(args_dG)
args_dGsmooth.LR = 5.0e-6
args_dGsmooth.modelname = "dGsmooth"
args_dGsmooth.nbatch = 6
args_dGsmooth.randomize = 0.0
args_dGsmooth.dG_args['dropout_rate'] = 0.1
args_dGsmooth.dG_args['out_dim'] = 17
args_dGsmooth.dGrange = np.arange(4.0,12.1,0.5) #in pK value; from 100 um ~ 1 pm
args_dGsmooth.n_dGbins = 17
args_dGsmooth.dGw = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 4.0, 6.0, 10.0, 10.0, 10.0])

# mode = 'vnode'/'cross1', ligand edge using topk (prv bond only)
args_dGmode = copy.copy(args_dGsmooth) 
args_dGmode.modelname = 'dGmode'
args_dGmode.ligand_args['num_layers_encoder'] = 3
args_dGmode.ligand_args['num_layers_decoder'] = 3
args_dGmode.ligand_mode('vnode')
args_dGmode.dG_args['mode'] = 'cross1'
args_dGmode.ligand_args['edge'] = 'topk'
args_dGmode.LR = 1.0e-5
## pretraining related
#args_dGmode.dataf_train = "data/train_shuffle500dupl.txt"
#args_dGmode.dataf_valid = "data/valid_tmp.txt"
# dG related
args_dGmode.dataf_train = "data/train_shuffle500dG.txt"
args_dGmode.dataf_valid = "data/valid_alldG.txt"
args_dGmode.w['dG'] = 5.0 
args_dGmode.w['Hinter'] = 5.0
args_dGmode.w['Hlig'] = 0.5 #0.1 -> up
args_dGmode.w['Slig'] = 0.1
args_dGmode.w['Srec'] = 2.0 #1.0 -> up

args_glem = copy.copy(args_dGmode) # mode = 'vnode'/'cross1'
args_glem.ligand_args['emb'] = 'GLem1'
args_glem.ligand_args['n_input_feats'] += 128 #GLem dim
args_glem.modelname = 'GLem1'
args_glem.w['Hlig'] = 0.5 #0.1 -> up
args_glem.w['Slig'] = 0.1
#args_glem.dataf_train = "data/train_shuffle500dupl.txt"
#args_glem.dataf_valid = "data/valid_tmp.txt"
args_glem.dataf_train = "data/train_shuffle500dG.txt"
args_glem.dataf_valid = "data/valid_alldG.txt"

#args_glem.dataf_train = "data/train_ligandonly.txt"
#args_glem.dataf_valid = "data/valid_ligandonly.txt"
