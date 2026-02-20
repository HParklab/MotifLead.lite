import os,sys
import copy
import numpy as np
import torch
import dgl

# My libs
from src.dGModel import ComboModel
from src.dataset_devel import collate, DataSet
from src.myutils import bin_it
from src.logger import *
from src.loss import ReconstructionLoss, KL_div, MySigmoid, MyCappedMSE, MyCappedMSE2, MaskedMSE

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.args import args_dGmode as args

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

torch.set_printoptions(sci_mode=False,precision=4)

args.modelname = sys.argv[1]
if len(sys.argv) > 2:
    args.dataf_valid = sys.argv[2]
else:
    args.dataf_valid = 'data/check.txt'
#args.nbatch = 1
#args.debug = True

ddp = True #("CUDA_VISIBLE_DEVICES" in os.environ) and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1)
params_loader={
    'shuffle': (not ddp), 
    'num_workers':5 if not args.debug else 0,
    'pin_memory':True,
    'collate_fn':collate,
    'batch_size':1 if args.debug else args.nbatch}

NullLoss = {'total':[],
            'Srec':[], #BCE
            'Hrec':[], #MSE
            'Slig':[], #MSE
            'Hlig':[], #MSE
            'Hinter':[], #MSE
            'recon':[],
            'KL':[],
            'reg':[] }

def load_params(rank):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    model = ComboModel(args)
    model.to(device)

    epoch = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=args.LR)

    modelf = "models/%s/best.pkl"%args.modelname
    if os.path.exists(modelf):
        if not args.silent: print("Loading a checkpoint")
        checkpoint = torch.load(modelf,map_location=device)

        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        
        for key in checkpoint["model_state_dict"]:
            key2 = key
            if key.startswith('module.'): key2 = key[7:]
            
            if key2 in model_keys:
                wts = checkpoint["model_state_dict"][key]
                if wts.shape == model_dict[key2].shape: # load only if has the same shape
                    trained_dict[key2] = wts
                else:
                    print("skip", key)
            else:
                print("?", key)

        nnew, nexist = 0,0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
                print("new", key)
            else:
                nexist += 1
        if rank == 0: print("params", nnew, nexist)
        
        model.load_state_dict(trained_dict, strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1 
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        for key in NullLoss:
            if key not in train_loss: train_loss[key] = []
        for key in NullLoss:
            if key not in valid_loss: valid_loss[key] = []
            
        if not args.silent: print("Restarting at epoch", epoch)
        
    else:
        sys.exit('no such model exist '+modelf)

    if rank == 0:
        print("Nparams:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss

def parse_set( txt, world_size, rank ):
    # Type Receptor                   Ligand                               Complex
    # 0 RosettaEnergy/features/7abp   RosettaEnergy/features/7abp          RosettaEnergy/features/7abp
    # 1 RecEntropy/5ghv               None                                 None
    # 2 None                          Ligand/subset_1/mol_1                None

    recf_s, ligf_s, mode_s, cmplxf_s = [],[],[],[]
    for l in open(txt):
        if l.startswith('#'): continue
        words = l[:-1].split()
        mode_s.append( int(words[0]) )
        recf_s.append( words[1] )
        ligf_s.append( words[2] )
        cmplxf_s.append( words[3] )

    data_set = DataSet(recf_s, ligf_s, cmplxf_s, mode_s, args, is_train=True) #allow shuffling

    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data_set,num_replicas=world_size, rank=rank
                                                                  ,shuffle=params_loader['shuffle']) #commenting out will allow shuffle
        data_loader = torch.utils.data.DataLoader(data_set,sampler=sampler, **params_loader)
    else:
        data_loader = torch.utils.data.DataLoader(data_set, **params_loader)
    return data_loader

def inference(rank,world_size,dumm):
    gpu = rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank)

    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    ## load_params
    model = load_params(rank)[0]

    if ddp:
        model = DDP(model,device_ids=[gpu],find_unused_parameters=False)

    ## data loader
    loader = parse_set(args.dataf_valid, world_size, rank)
    e_count = 0
    
    with torch.no_grad():
        model.eval()
        for i, inputs in enumerate(loader):
            if inputs == None:
                e_count += 1
                continue

            Gall, Grec, Glig, info = inputs
            if Grec == None:
                e_count += 1
                continue

            if Glig != None: Glig = Glig.to(device)
            if Grec != None: Grec = Grec.to(device)
            if Gall != None: Gall = Gall.to(device)

            loss = infer1( model, Gall, Grec, Glig, info )
            

def split_pred_into_batch(pred, Grec, info):
    pred_b = {}
    bGrec = dgl.unbatch(Grec)
    if 'Srec_index' in info:
        shift = 0
        pred_sum = []
        for b,(G,rec_idx,resname) in enumerate(zip(bGrec,info['Srec_index'],info['aas_Srec'])):
            for z,(atmidx,resn) in enumerate(zip(rec_idx,resname)):
                atmidx = atmidx + shift #index shift from prv graphs
                
                # pool-sum per residue
                a = pred['Srec'][atmidx,:].mean(axis=0)
                iaas = torch.argmax(Grec.ndata['0'][atmidx,:21],dim=-1)
                #aa = ["UNK","ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                #      "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
                #print("loss", resn, z, iaas, atmidx)
                pred_sum.append(pred['Srec'][atmidx,:].mean(axis=0).softmax(axis=-1))
            shift += G.number_of_nodes()

        #into a sparse list
        if len(pred_sum) > 0:
            pred_b['Srec'] = torch.stack(pred_sum)
        
    return pred_b

def infer1(model, Gall, Grec, Glig, info ):
    pred, AEvars = model(Gall, Grec, Glig,
                         do_dropout=False)

    #if has_nan:
    #    print("has nan!", info['ligand'])
    #    return False

    device = Gall.device

    # split prediction into batch
    pred_b = split_pred_into_batch(pred, Grec, info)
    
    lossSrec, lossHrec, lossSlig, lossHlig, lossHinter = \
        torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), \
        torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)


    #w = Srec_per_bin_weight().to(device)
    
    func_mse = torch.nn.MSELoss()
    func_cce = torch.nn.CrossEntropyLoss()
    func_KL = torch.nn.KLDivLoss()
    func_maskedMSE = MaskedMSE
    func_Huber = torch.nn.HuberLoss(reduction='none')

    info['label_Srec'] = info['label_Srec'].flatten()

    #print(pred_b['Srec'].shape, info['label_Srec'].shape, [len(a) for a in info['Srec_index']])
    if ('Srec' in pred_b) and pred_b['Srec'].shape[0] == info['label_Srec'].shape[0]:
        Srec_label_binned = bin_it(info['label_Srec'],
                                   bin_max=args.receptor_args['Srec_max'],
                                   num_classes=args.receptor_args['Srec_bins'])
        lossSrec   = func_cce(pred_b['Srec'], Srec_label_binned.to(device))
        report_Srec(pred_b['Srec'], Srec_label_binned, info['aas_Srec'], info['resname_Srec'] )

    if (len(pred['Slig']) > 0) and (len(pred['Slig'].shape) == len(info['label_Slig'].shape)) and (len(pred['Slig']) == len(info['label_Slig'])):
        info['label_Slig'] = info['label_Slig'].to(device)
        lossSlig   = func_maskedMSE(pred['Slig']  ,info['label_Slig'], (info['label_Slig']> -1e-6).float())
        report_Slig(pred['Slig'], info['label_Slig'], info['ligand'])
        
    if (len(pred['Hlig']) > 0) and (len(pred['Hlig'].shape) == len(info['label_Hlig'].shape)) and (len(pred['Hlig']) == len(info['label_Hlig'])):
        info['label_Hlig'] = info['label_Hlig'].to(device)
        lossHlig   = func_maskedMSE(pred['Hlig'], info['label_Hlig'], (info['label_Hlig']> -1e-6).float())
        report_Hlig(pred['Hlig'], info['label_Hlig'], info['ligand'])
        
    if 'label_Hinter' in info and (len(info['label_Hinter'].shape) == len(pred['Hinter'].shape)) and len(pred['Hinter']) == len(info['label_Hinter']):
        lossHinter = func_Huber(pred['Hinter'],info['label_Hinter'].to(device))
                                                  
        report_Hinter(pred['Hinter'],info['label_Hinter'],lossHinter)
        lossHinter = lossHinter.mean()

    if 'label_dG' in info and len(info['label_dG']) > 0 and len(info['label_dG']) == len(pred['dG']):
        report_dG(pred['dG'], info['label_dG'], info['complex'])
        
    
## main
if __name__=="__main__":
    print("dgl version", dgl.__version__)
    torch.cuda.empty_cache()
    mp.freeze_support()
    world_size=torch.cuda.device_count()
    print("Using %d GPUs.."%world_size)
    
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12348'

    os.system("touch GPU %d"%world_size)

    if ddp:
        mp.spawn(inference,args=(world_size,0),nprocs=world_size,join=True)
    else:
        inference(0, 1, None)
