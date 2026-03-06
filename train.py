import os,sys
import copy
import numpy as np
import torch
import dgl

# My libs
from src.dGModel import dGModel
from src.dataset_ddG import collate, DataSet
from src.logger import *
from src.args import args_base

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

torch.set_printoptions(sci_mode=False,precision=4)

ddp = ("CUDA_VISIBLE_DEVICES" in os.environ) and (len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1)

params_loader={
    'shuffle': (not ddp), 
    'num_workers':5 if not args.debug else 0,
    'pin_memory':True,
    'collate_fn':collate,
    'batch_size':1 if args.debug else args.nbatch}

def bin_it(x, bin_min=0.0, bin_max=5.0, num_classes=20, onehot=False):
    bin_size = (bin_max - bin_min)/num_classes
    x_bin_index = torch.div(x - bin_min, bin_size, rounding_mode='floor').long()
    if onehot:
        return torch.nn.functional.one_hot(x_bin_index, num_classes=num_classes)
    else:
        return x_bin_index

def load_params(rank):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    model = dGModel(args.model_args)
    model.to(device)

    epoch = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=args.LR)

    NullLoss = {'dG':[], 'ddG':[], 'reg':[], 'total':[]}

    if os.path.exists("models/%s/model.pkl"%args.modelname):
        if not args.silent: print("Loading a checkpoint")
        checkpoint = torch.load("models/"+args.modelname+"/model.pkl",map_location=device)

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

        model.load_state_dict(trained_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1 
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        for key in NullLoss:
            if key not in train_loss: train_loss[key] = []
        for key in NullLoss:
            if key not in valid_loss: valid_loss[key] = []
            
        if not args.silent: print("Restarting at epoch", epoch)
        
    else:
        if not args.silent: print("Training a new model")
        train_loss = copy.deepcopy(NullLoss)
        valid_loss = copy.deepcopy(NullLoss)
    
        if not os.path.isdir( os.path.join("models", args.modelname)):
            if not args.silent: print("Creating a new dir at", os.path.join("models", args.modelname))
            os.makedirs( os.path.join("models", args.modelname), exist_ok=True )

    if rank == 0:
        print("Nparams:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss

## TODO
def parse_set( txt, world_size, rank ):
    inputs = [l[:-1] for l in open(txt)] #pair:grp or single: ligid
    data_set = DataSet( args, inputs ) 

    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data_set,num_replicas=world_size, rank=rank
                                                                  ,shuffle=params_loader['shuffle']) #commenting out will allow shuffle
        data_loader = torch.utils.data.DataLoader(data_set,sampler=sampler, **params_loader)
    else:
        data_loader = torch.utils.data.DataLoader(data_set, **params_loader)
    return data_loader

### train_model
def train_model(rank,world_size,dumm):
    gpu = rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank)

    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    ## load_params
    model,optimizer,start_epoch,train_loss,valid_loss = load_params(rank)

    if ddp:
        model = DDP(model,device_ids=[gpu],find_unused_parameters=False)

    ## data loader
    train_loader = parse_set(args.dataf_train, world_size, rank)
    valid_loader = parse_set(args.dataf_valid, world_size, rank)

    ## iteration
    for epoch in range(start_epoch,args.max_epoch):
        ## train
        model.train()
        temp_loss = train_one_epoch( model, optimizer, train_loader, epoch, True, rank, device )
            
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))
            
        #validate
        optimizer.zero_grad()
        with torch.no_grad():
            model.eval()
            temp_loss = train_one_epoch( model, optimizer, valid_loader, epoch, False, rank, device )
        
            for k in valid_loss:
                valid_loss[k].append(np.array(temp_loss[k]))

        print("***SUM***")
        print("[%d] Train loss %3d | %7.4f | Valid loss | %7.4f"%(rank,epoch,np.mean(train_loss['total'][-1]),np.mean(valid_loss['total'][-1])))

        ## update the best model
        if rank==0:
            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                }, os.path.join("models", args.modelname, "best.pkl"))
   
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, os.path.join("models", args.modelname, "model.pkl"))
            
### train_one_epoch
def train_one_epoch( model, optimizer, loader, epoch, is_train, rank, device ):
    temp_loss = {'dG':[], 'ddG':[], 'reg':[], 'total':[]}
    b_count,e_count=0,0
    accum=1

    for i, inputs in enumerate(loader):
        if inputs == None:
            e_count += 1
            continue

        G1, G2, info = inputs
        if args.mode == 'pair' and (G1 == None or G2 == None):
            e_count += 1
            continue

        G1 = G1.to(device)
        G2 = G2.to(device)

        if ddp:
            with torch.cuda.amp.autocast(enabled=False):
                with model.no_sync(): #should be commented if
                    loss = train1( model, G1, G2, info, temp_loss, args.mode, is_train,
                                   epoch, verbose=args.verbose )
        else:
            loss = train1( model, G1, G2, info, temp_loss, args.mode, is_train,
                           epoch, verbose=args.verbose )
            
        if not loss:
            if is_train:
                sys.exit("skip: "+str(i))
            else:
                print("skip: "+str(i))
                continue
            
        # Only update after certain number of accululations.
        b_count += 1
        if (b_count+1)%accum == 0:
            if is_train:
                loss.requires_grad_(True)
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                optimizer.step()    
                optimizer.zero_grad()
                
          
    return temp_loss

def train1(model, G1, G2, info, temp_loss, mode, is_train, epoch, verbose=False ):
    device = G1.device
    
    pred1  = model(G1, info['ligmask'][0], do_dropout=is_train)
    pred2  = model(G2, info['ligmask'][1], do_dropout=is_train)

    label = info['label'].to(device)

    func_mse = torch.nn.MSELoss()
    loss_dG = func_mse(pred1, label[:,0]) + func_mse(pred2, label[:,1])
    loss_ddG = func_mse(pred1 - pred2, label[:,0] - label[:,1])   

    for i,ligs in enumerate(info['ligname']):
        if is_train and verbose:
            print(f"Train, epoch {epoch:4d}, {ligs[0]:20s} {float(pred1[i].cpu().detach()):8.3f} {float(label[i,0].cpu().detach()):8.3f}"+\
                  f", {ligs[1]:20s} {float(pred2[i].cpu().detach()):8.3f} {float(label[i,1].cpu().detach()):8.3f}"+\
                  f" | ddG {float((pred1[i]-pred2[i]).cpu().detach()):8.3f} {float((label[i,0] - label[i,1]).cpu().detach()):8.3f}")
        elif verbose:
            print(f"Valid, epoch {epoch:4d}, {ligs[0]:20s} {float(pred1[i].cpu().detach()):8.3f} {float(label[i,0].cpu().detach()):8.3f}"+\
                  f", {ligs[1]:20s} {float(pred2[i].cpu().detach()):8.3f} {float(label[i,1].cpu().detach()):8.3f}"+\
                  f" | ddG {float((pred1[i]-pred2[i]).cpu().detach()):8.3f} {float((label[i,0] - label[i,1]).cpu().detach()):8.3f}")

    
    l2_reg = torch.tensor(0.).to(device)
    if is_train:
        for param in model.parameters(): l2_reg += torch.norm(param)
                
    ## final loss
    loss = args.w['dG']*loss_dG + args.w['ddG']*loss_ddG + args.w['reg']*l2_reg

    if torch.isnan(loss).any():
        sys.exit(f'ERROR: Nan found in loss! '+info['grp'][0])

    #store as per-sample loss
    temp_loss["dG"].append(loss_dG.cpu().detach().numpy())
    temp_loss["ddG"].append(loss_ddG.cpu().detach().numpy())
    temp_loss["total"].append(loss.cpu().detach().numpy())
    
    return loss

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
        os.environ['MASTER_PORT'] = '12346'

    os.system("touch GPU %d"%world_size)

    if ddp:
        mp.spawn(train_model,args=(world_size,0),nprocs=world_size,join=True)
    else:
        train_model(0, 1, None)
