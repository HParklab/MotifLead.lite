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
from src.args import args_base

## DDP related modules
#import torch.multiprocessing as mp
#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

from optparse import OptionParser

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")
torch.set_printoptions(sci_mode=False,precision=4)

ddp = ("CUDA_VISIBLE_DEVICES" in os.environ) and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1)
#device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
device = torch.device("cpu")

def parse_args(args_in):
    args_in.mode = 'single'
    args_out = copy.copy(args_in)
    parser = OptionParser(usage="python infer.py [-p parent-path-to-npzfiles]")
    parser.add_option("-p",
                      type="string")

    
    args_add, args = parser.parse_args()
    if "p" not in args_add.__dict__:
        parser.error("-p not found")

    npzs = glob.glob(args_add.p+"/*feat.npz")
    if len(npzs) == 0:
        sys.exit(f"no npz file found in {args_add.p}")

    # override default
    args_out.datapath = os.path.abspath(args_add.p)
    
    return args_out, [l.split('/')[-1].replace('.feat.npz','') for l in npzs]

def load_params():
    model = dGModel(args.model_args)
    model.to(device)

    if not os.path.exists("models/%s/model.pkl"%args.modelname):
        sys.exit(f"no model file at models/{args.modelname}/infer.pkl")
        
    checkpoint = torch.load("models/"+args.modelname+"/infer.pkl",map_location=device)
        
    model.load_state_dict(model.state_dict(), strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model

def infer(args, npzs):
    model = load_params()

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

    #np.save(z)

## main
if __name__=="__main__":
    args,npzs = parse_args(args_base)
    print(f"running on {len(npzs)} npzs at {args.datapath}")
    infer(args, npzs)
