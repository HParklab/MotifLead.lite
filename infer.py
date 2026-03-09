import os,sys
import copy
import numpy as np
import torch
import dgl
import glob

# My libs
from src.dGModel import dGModel
from src.dataset_ddG import collate, DataSet
from src.logger import *
from src.args import args_base0 as args_base

from optparse import OptionParser

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")
torch.set_printoptions(sci_mode=False,precision=4)

ddp = ("CUDA_VISIBLE_DEVICES" in os.environ)# and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def parse_args(args_in):
    args_in.mode = 'single'
    args_out = copy.copy(args_in)
    parser = OptionParser(usage="python infer.py [-p parent-path-to-npzfiles]")
    parser.add_option("-p",
                      type="string")
    parser.add_option("-o","--output",
                      default="dG.npz",
                      type="string")

    
    args_add, args = parser.parse_args()
    if args_add.__dict__['p'] == None:
        parser.error("-p not found")

    npzs = glob.glob(args_add.p+"/*feat.npz")
    if len(npzs) == 0:
        sys.exit(f"no npz file found in {args_add.p}")

    # override default
    args_out.datapath = os.path.abspath(args_add.p)
    args_out.output = args_add.output
    
    return args_out, [l.split('/')[-1].replace('.feat.npz','') for l in npzs]

def load_params():
    model = dGModel(args.model_args)
    model.to(device)

    if not os.path.exists("models/%s/model.pkl"%args.modelname):
        sys.exit(f"no model file at models/{args.modelname}/infer.pkl")
        
    checkpoint = torch.load("models/"+args.modelname+"/infer.pkl",map_location=device)

    trained_dict = {}
    for key in checkpoint["model_state_dict"]:
        key2 = key
        if key.startswith('module.'): key2 = key[7:]
        wts = checkpoint["model_state_dict"][key]
        trained_dict[key2] = wts
        
    model.load_state_dict(trained_dict, strict=True)
        
    return model

def infer(args, npzs):
    model = load_params()
    model.eval()

    params_loader={
        'shuffle': False,
        'num_workers':5,
        'pin_memory':True,
        'collate_fn':collate,
        'batch_size':args.nbatch}

    data_set = DataSet( args, npzs ) 
    loader = torch.utils.data.DataLoader(data_set, **params_loader)

    e_count = 0
    tags, values = [],[]

    for i, inputs in enumerate(loader):
        if inputs == None: continue
        if i >= 1000: break

        G1, _, info = inputs
        if G1 == None:
            e_count += 1
            continue

        G1 = G1.to(device)
        
        preds  = model(G1, info['ligmask'][0], do_dropout=False)
        preds = preds.cpu().detach()
        
        for i,ligs in enumerate(info['ligname']):
            tags.append(ligs[0])
            values.append(float(preds[i]))
        
            print(f"{ligs[0]:20s} {preds[i]:8.3f}")

    print(f"saving results for {len(tags)} entries at {args.output}")
    np.savez(args.output,
             tags=tags,
             dGs=values)

## main
if __name__=="__main__":
    args,npzs = parse_args(args_base)
    print(f"running on {len(npzs)} npzs at {args.datapath}")
    infer(args, npzs)
