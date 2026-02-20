import os,sys
import copy
import numpy as np
import torch
import dgl

# My libs
from src.ComboModel import ComboModel
from src.dataset import collate, DataSet
from src.myutils import bin_it
from src.logger import *
from src.args import args_default as args
from src.loss import ReconstructionLoss, KL_div, MySigmoid

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

        nnew, nexist = 0,0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
                #print("new", key)
            else:
                nexist += 1
        if rank == 0: print("params", nnew, nexist)
        
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

    data_set = DataSet(recf_s, ligf_s, cmplxf_s, mode_s, args)

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
    temp_loss = copy.deepcopy(NullLoss)
    b_count,e_count=0,0
    accum=1

    for i, inputs in enumerate(loader):
        if inputs == None:
            e_count += 1
            continue

        Gall, Grec, Glig, info = inputs
        if Grec == None:
            e_count += 1
            continue

        #print("epoch/i", epoch, i, info)
        
        if Glig != None: Glig = Glig.to(device)
        if Grec != None: Grec = Grec.to(device)
        if Gall != None: Gall = Gall.to(device)

        if ddp:
            with torch.cuda.amp.autocast(enabled=False):
                with model.no_sync(): #should be commented if
                    loss = train1( model, Gall, Grec, Glig, info, temp_loss, is_train )
        else:
            loss = train1( model, Gall, Grec, Glig, info, temp_loss, is_train )

        if not loss:
            if is_train:
                sys.exit("skip: "+str(i))
            else:
                print("skip: "+str(i))
                continue
            
        # Only update after certain number of accululations.
        b_count += 1
        if (b_count+1)%accum == 0:
            target = info['receptor'][0]
            mode   = info['mode'][0]
            if is_train:
                loss.requires_grad_(True)
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                optimizer.step()    
                optimizer.zero_grad()
                if args.verbose: report1("%s%d%-25s %2d (%3d)"%("[", rank, "] Train "+target,mode,epoch),
                                         i,len(loader),temp_loss)
                
            else:
                if args.verbose: report1("%s%d%-25s %2d (%3d)"%("[", rank, "] VALID "+target,mode,epoch),
                                         i,len(loader),temp_loss)
          
    return temp_loss

def report1(prefix, i, n, loss_dict):
    print("%-30s"%prefix, "%10s"%("%d/%d"%(i,n)),
          "Total| Srec/Hrec | Slig/Hlig : Hinter ! recon/KL $ reg",
          "%8.3f |"%float(loss_dict['total'][-1]), 
          "%6.3f"%float(loss_dict['Srec'][-1]),
          "%6.3f |"%float(loss_dict['Hrec'][-1]),
          "%8.3f/"%float(loss_dict['Slig'][-1]),
          "%8.3f :"%float(loss_dict['Hlig'][-1]),
          "%8.3f !"%float(loss_dict['Hinter'][-1]),
          "%6.3f"%float(loss_dict['recon'][-1]),
          "%6.3f $"%float(loss_dict['KL'][-1]),
          "%8.3f"%float(loss_dict['reg'][-1]),
          )

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
                
def train1(model, Gall, Grec, Glig, info, temp_loss, is_train ):
    pred, AEvars, has_nan = model(Gall, Grec, Glig,
                                  do_dropout=is_train)

    if has_nan:
        print("has nan!", info['ligand'])
        return False

    device = Gall.device

    # split prediction into batch
    pred_b = split_pred_into_batch(pred, Grec, info)
    
    lossSrec, lossHrec, lossSlig, lossHlig, lossHinter = \
        torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), \
        torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    func_mse = torch.nn.MSELoss()
    func_cce = torch.nn.CrossEntropyLoss()
    func_KL = torch.nn.KLDivLoss()
    func_Huber = torch.nn.HuberLoss(delta=1.0)

    info['label_Srec'] = info['label_Srec'].flatten()

    #print(pred_b['Srec'].shape, info['label_Srec'].shape, [len(a) for a in info['Srec_index']])
    if ('Srec' in pred_b) and pred_b['Srec'].shape[0] == info['label_Srec'].shape[0]:
        Srec_label_binned = bin_it(info['label_Srec'],
                                   bin_max=args.receptor_args['Srec_max'],
                                   num_classes=args.receptor_args['Srec_bins'])
        lossSrec   = func_cce(pred_b['Srec'], Srec_label_binned.to(device))
        if not is_train:
            report_Srec(pred_b['Srec'], Srec_label_binned, info['aas_Srec'] )

    if (len(pred['Slig']) == len(info['label_Slig'].shape)) and (len(pred['Slig']) == len(info['label_Slig'])):
        lossSlig   = func_mse(pred['Slig']  ,info['label_Slig'].to(device))
    if (len(pred['Hlig']) == len(info['label_Hlig'].shape)) and (len(pred['Hlig']) == len(info['label_Hlig'])):
        lossHlig   = func_mse(pred['Hlig'], info['label_Hlig'].to(device))
    if len(pred['Hinter']) == len(info['label_Hinter']):
        lossHinter = func_Huber(pred['Hinter'],info['label_Hinter'].to(device))
        #lossHinter = func_MySigmoid(pred['Hinter'],info['label_Hinter'].to(device))
        
        if not is_train:
            report_Hinter(pred['Hinter'],info['label_Hinter'])

    # VAE related losses
    ## TODO    loss_recon, loss_KL = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
    loss_recon, loss_KL = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    ## todo: need to fix not-in- 0<x<1 issue
    #if AEvars != None:
    #    loss_recon = ReconstructionLoss(AEvars['posout'],AEvars['negout'])
    #    loss_KL    = KL_div(AEvars['mu'], AEvars['logvar'])
    
    ## 3. Full regularizer
    l2_reg = torch.tensor(0.).to(device)
    if is_train:
        for param in model.parameters(): l2_reg += torch.norm(param)
                
    ## final loss
    loss = args.w['Srec']*lossSrec + args.w['Hrec']*lossHrec + \
        args.w['Slig']*lossSlig + args.w['Hlig']*lossHlig + \
        args.w['Hinter']*lossHinter + \
        args.w['recon']*loss_recon + args.w['KL']*loss_KL + \
        args.w['reg']*l2_reg

    if torch.isnan(loss).any():
        sys.exit(f'ERROR: Nan found in loss! '+info['receptor'][0])

    #store as per-sample loss
    temp_loss["total"].append(loss.cpu().detach().numpy())
    
    if lossSrec > 0.0: temp_loss["Srec"].append(lossSrec.cpu().detach().numpy())
    else: temp_loss["Srec"].append(0)
                    
    if lossHrec > 0.0: temp_loss["Hrec"].append(lossHrec.cpu().detach().numpy())
    else: temp_loss["Hrec"].append(0)
    
    if lossSlig > 0.0: temp_loss["Slig"].append(lossSlig.cpu().detach().numpy()) 
    else: temp_loss["Slig"].append(0)
    
    if lossHlig > 0.0: temp_loss["Hlig"].append(lossHlig.cpu().detach().numpy()) 
    else: temp_loss["Hlig"].append(0)
    
    if lossHinter > 0.0: temp_loss["Hinter"].append(lossHinter.cpu().detach().numpy())
    else: temp_loss["Hinter"].append(0)

    if loss_recon > 0.0: temp_loss["recon"].append(loss_recon.cpu().detach().numpy())
    else: temp_loss["recon"].append(0)
    
    if loss_KL > 0.0: temp_loss["KL"].append(loss_KL.cpu().detach().numpy())
    else: temp_loss["KL"].append(0)
    
    if l2_reg > 0.0: temp_loss["reg"].append(l2_reg.cpu().detach().numpy())
    else: temp_loss["reg"].append(0)
    
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
