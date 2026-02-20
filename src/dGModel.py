import torch
import torch.nn as nn
import sys

from src.LigandModule import MaskGAE, LigandEntropyModule, LigandEnergyModule, SimpleLigandEncoder
from src.SrecModule import SrecModule
from src.InteractionModule import InteractionModule
from src.dGModule import dGModule

class ComboModel(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.gradient_checkpoint = args.gradient_checkpoint
        
        # Ligand Part
        self.LigandVAE = MaskGAE(args=args.ligand_args)
        
        self.LigandEntropyPredictor = LigandEntropyModule(args=args.ligand_args)
        self.LigandEnergyPredictor = LigandEnergyModule(args=args.ligand_args)

        # Receptor part
        self.SrecPredictor = SrecModule(args=args.receptor_args)
        
        # Complex part
        self.InteractionPredictor = InteractionModule( args=args.interaction_args,
                                                       debug=args.debug )
        
        self.dGPredictor = dGModule( args=args.dG_args,
                                     debug=args.debug )
        self.ligand_pool = args.ligand_args['mode']

    def forward(self, Gall, Grec, Glig, do_dropout=True):
        # 1. Ligand encoder: Predict Ligand entropy & energy
        pred = {'Slig':[],'Hlig':[],'Srec':[],'Hinter':[], 'dG':[]}

        h_lig, AEvars, z_lig = None, None, None
        if Glig.number_of_nodes() == 0:
            pass
        else:
            h_lig, z_lig, AEvars = self.LigandVAE(Glig, do_dropout)

            # split z_lig into batch
            nnodes = Glig.batch_num_nodes()
            batch_mask = torch.zeros((nnodes.shape[0],h_lig.shape[0]),device=h_lig.device)
            nsum = 0

            for ib,n in enumerate(nnodes):
                if self.ligand_pool == 'simple':
                    batch_mask[ib,nsum:nsum+n] = 1.0
                if self.ligand_pool == 'vnode':
                    batch_mask[ib,nsum+n-1] = 1.0 # open the last node only
                nsum += n
                
            pred['Slig'] = self.LigandEntropyPredictor(z_lig, batch_mask)
            pred['Hlig'] = self.LigandEnergyPredictor(z_lig, batch_mask)

        # 2. Receptor-encoder; Predict Receptor entropy 
        h_rec, pred['Srec'] = self.SrecPredictor( Grec, do_dropout )

        # Feed embedding into Gall
        if Grec.number_of_nodes() > 0 and Glig.number_of_nodes() > 0:
            brec,blig,ball = 0,0,0
            all_chainidx = torch.zeros(Gall.number_of_nodes()).to(Gall.device)
            for b,(nl,nr,na) in enumerate(zip(Glig.batch_num_nodes(),Grec.batch_num_nodes(),Gall.batch_num_nodes())):
                # in case vnode exists
                ## this part is very complicated -- try no touch...
                #if nl != na-nr: #sanity check; last one should be 1
                #    print(b, Glig.ndata['0'][blig+nl-3:blig+nl,-1]) 

                try:
                    if int(nr) == 0 or (int(na) == int(nr)):
                        pass
                    elif ball+nr >= Gall.ndata['0'].shape[0] or brec+nr >= h_rec.shape[0] or blig+na-nr >= h_lig.shape[0]: #why this happen
                        pass
                    else:
                        Gall.ndata['0'][ball:ball+nr,      :h_rec.shape[-1]] = h_rec[brec:brec+nr]
                        Gall.ndata['0'][ball+nr:ball+na,:h_lig.shape[-1]] = h_lig[blig:blig+na-nr] #take non-vnodes (na-nr != nl)
                        
                    all_chainidx[ball+nr:ball+na] = 1.0
                    brec += nr
                    blig += nl
                    ball += na
                    
                except:
                    print("############################die!!!", h_lig.shape, h_rec.shape, brec+nr, blig+na-nr, ball, brec, blig, na, nr, nl)
                    #sys.exit()
                    pass

            
            # Predict complex energy
            pred['Hinter'], h_comb = self.InteractionPredictor( Gall, all_chainidx,
                                                                do_dropout=do_dropout )

            pred['dG'] = self.dGPredictor( Gall, Grec, Glig, h_comb, h_rec, h_lig, do_dropout=do_dropout )
        
        return pred, AEvars

    
