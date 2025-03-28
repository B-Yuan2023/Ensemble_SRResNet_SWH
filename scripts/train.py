"""
Super-resolution

"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet
from mod_srcnn import SRCNN,FSRCNN,SRCNN1,FSRCNN1
from mod_srres import SRRes,SRResA
from mod_srres_ms import SRRes_MS,SRRes_MS1,SRRes_MS2
from datasets import myDataset,my_loss
from funs_prepost import plt_sub,nc_load_depth,plot_line_list

import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name=sys.argv[1]   # 'par01_rd0' # sys.argv[1]
# print(os.getcwd())
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)  
# from mod_para import * 

if mod_para.krand == 0:     # no randomness
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    suf0 = mod_para.suf0
    files_lr = mod_para.files_lr
    files_hr = mod_para.files_hr
    indt_lr = mod_para.indt_lr # 
    indt_hr = mod_para.indt_hr # 
    
    rtra = mod_para.rtra
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    
    if hasattr(mod_para, 'rep'):  # if input has list rep
        rep = mod_para.rep
    else:
        nrep = mod_para.nrep
        rep = list(range(0,nrep))
    # rep = [3,4]
    
    # create nested list of files and indt
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]

    if hasattr(mod_para, 'll_lr'):
        ll_lr = mod_para.ll_lr # user domain latitude longitude
    else:
        ll_lr = [None]*2
    if hasattr(mod_para, 'll_hr'):
        ll_hr = mod_para.ll_hr # user domain latitude longitude
    else:
        ll_hr = [None]*2
    if hasattr(mod_para, 'kintp'):
        kintp = mod_para.kintp # 1, griddata, 2 RBFInterpolator
    else:
        kintp = [0,0] # no interpolation for lr & hr
        
    if hasattr(mod_para, 'kauxhr'): # hr auxiliary field within CNN model
        kauxhr = mod_para.kauxhr # auxiliary fields, 0, none; 1-hr bathymetry;
    else:
        kauxhr = 0
    if hasattr(mod_para, 'auxhr_chls'): # hr auxiliary field within CNN model
        auxhr_chls = mod_para.auxhr_chls # auxiliary fields, 0, none; 1-hr bathymetry;
    else:
        auxhr_chls = 1
    if hasattr(mod_para, 'file_auxhr'): # hr auxiliary field 
        file_auxhr = mod_para.file_auxhr 
        depth_dm= nc_load_depth(file_auxhr,ll_hr[0],ll_hr[1])[4]  # (W,H)
        depth_aux = np.repeat(depth_dm[np.newaxis,np.newaxis,:,:], opt.batch_size, axis=0)
        depth_aux = torch.from_numpy(depth_aux)
    else:
        file_auxhr = None
        depth_aux = None
    if kauxhr > 0:
        assert depth_aux is not None    
        
    if hasattr(mod_para, 'knn'): # nn models, 0-1 srres; 4-5 SRRes_MS, 7-8 FSRCNN, 9-10 SRCNN,
        knn = mod_para.knn
    else:
        knn = 0
    # if knn>0:
    #     suf = suf + '_nn'+str(knn)
    if hasattr(mod_para, 'ker_size'): # kernel size, only work for SRRes,SRRes_MS,SRRes_MS1
        ker_size = mod_para.ker_size # 
    else:
        ker_size = 3
    if hasattr(mod_para, 'kernel_no'): # kernel size, only work for SRRes,SRRes_MS,SRRes_MS1
        kernel_no = mod_para.kernel_no # 
    else:
        kernel_no = 64
    if hasattr(mod_para, 'dsm'): # kernel numbers d,s and repeated layers m in FSRCNN,FSRCNN1
        dsm = mod_para.dsm # 
    else:
        dsm = [56,12,4]
    if hasattr(mod_para, 'mod_att'): # attenion for SRResA, 1-2 SE, 3 GCT, 6 CBAM
        mod_att = mod_para.mod_att # 
    else:
        mod_att = 0
    if hasattr(mod_para, 'katt'): # attenion after conv, 1: add to feature extraction; 2 add to upsampling
        katt = mod_para.katt # 
    else:
        katt = 0
        
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='train',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    valid_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='valid',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    
    for irep in rep:
        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        data_train = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,  # True by default
            num_workers=opt.n_cpu,
        )
        Nbatch = len(data_train)
        data_valid = DataLoader(
            valid_set,
            batch_size=opt.batch_size, 
            num_workers=opt.n_cpu,
        )        
        Nbatch_v = len(data_valid)
    
        out_path = path_par+'results_training/SRF_' + str(opt.up_factor) + suf + '_re'+str(irep)+'/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = path_par+'nn_mod_' + str(opt.up_factor) + suf0 +'/' # 
            
        opath_nn = path_par+'nn_mod_' + str(opt.up_factor) + suf +'/' # 
        if not os.path.exists(opath_nn):
            os.makedirs(opath_nn)
    
        opath_st = path_par+'stat' + suf +'/'
        if not os.path.exists(opath_st):
            os.makedirs(opath_st)
    
        cuda = torch.cuda.is_available()
        
        
        # Initialize generator 
        if knn ==0:
            generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                        n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor)
        # discriminator = Discriminator(input_shape=(nchl_o, *hr_shape))
        elif knn == 1:
            generator = SRRes(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=opt.residual_blocks,
                                        up_factor=opt.up_factor,kernel_size=ker_size,kernel_no=kernel_no)
        elif knn == 2:
            generator = SRResA(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=opt.residual_blocks,
                                        up_factor=opt.up_factor,kernel_size=ker_size,kernel_no=kernel_no,
                                        mod_att=mod_att,katt=katt)
        elif knn == 4:
            generator = SRRes_MS(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor,kernel_size=ker_size)
        elif knn == 5:
            generator = SRRes_MS1(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor,kernel_size=ker_size)
        elif knn == 6:
            generator = SRRes_MS2(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor,kernel_size=ker_size)
        elif knn == 7:
            generator = FSRCNN(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor,d=dsm[0],s=dsm[1],m=dsm[2])
        elif knn == 8:
            generator = FSRCNN1(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor,d=dsm[0],s=dsm[1],m=dsm[2])
        elif knn == 9:
            generator = SRCNN(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor)
        elif knn == 10:
            generator = SRCNN1(in_channels=nchl_i, out_channels=nchl_o,up_factor=opt.up_factor)
        
        if cuda:
            generator = generator.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
        if opt.n0_epoch != 0:
            # Load pretrained models
            if torch.cuda.is_available():
                checkpointG = torch.load(ipath_nn+'netG_epoch_%d.pth' % (opt.n0_epoch))
            else:
                checkpointG = torch.load(ipath_nn+'netG_epoch_%d.pth' % (opt.n0_epoch), map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            optimizer_G.load_state_dict(checkpointG['optimizer_state_dict'])
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        
        # ----------
        #  Training
        # ----------
        results = {'loss_G': [], 'psnr': [], 'ssim': [], 'mse':[],'loss_G_v': [], 'psnr_v': [], 'ssim_v': [], 'mse_v':[]}
    
        for epoch in range(opt.n0_epoch, opt.N_epochs+1):
            eva_bch  = {'loss_G': 0, 'psnr': 0, 'ssim': 0, 'mse':0}
            generator.train()
            # discriminator.train()
            for i, dat in enumerate(data_train):
        
                # Configure model input
                dat_lr = Variable(dat["lr"].type(Tensor))
                dat_hr = Variable(dat["hr"].type(Tensor))
                
                if depth_aux is not None:
                    ibcsize = len(dat_lr) # this is to make sure batchsize of dat_aux and dat_lr match
                    dat_aux = Variable(depth_aux[0:ibcsize,...].type(Tensor))
                else:
                    dat_aux = None

                # ------------------
                #  Train Generators
                # ------------------
        
                optimizer_G.zero_grad()
        
                # Generate a high resolution image from low resolution input
                if knn==0:
                    gen_hr = generator(dat_lr,dat_aux)
                else:
                    gen_hr = generator(dat_lr)
        
                loss_content_pxl = my_loss(gen_hr,dat_hr,opt.nlm) # use pixel loss of data instead of feature maps

                # Total loss
                loss_G = loss_content_pxl #*opt.rlpxl + loss_GAN*opt.rladv + loss_content_per*opt.rlper
        
                loss_G.backward()
                optimizer_G.step()
        
                # --------------
                #  Log Progress
                # --------------
                
                # loss for current batch before optimization 
                eva_bch['loss_G'] += loss_G.item()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                    % (epoch, opt.N_epochs, i, Nbatch, loss_G.item())
                )
                
                batch_mse = ((gen_hr - dat_hr) ** 2).data.mean()
                eva_bch['mse'] += batch_mse.item()
                batch_ssim = ssim_torch(gen_hr, dat_hr,data_range=1.0).item()
                eva_bch['ssim'] += batch_ssim
                # batch_psnr = 10 * log10((dat_hr.max()**2) / batch_mse)
                batch_psnr = 10 * log10(1.0 / batch_mse)
                eva_bch['psnr'] += batch_psnr
        
                if epoch % opt.sample_epoch == 0 and i % opt.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    dat_lr = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor)
                    if nchl_i == nchl_o ==1: # same vars or 1 var to 1 var
                        img_grid = torch.cat((dat_lr, dat_hr,gen_hr), -1)
                    else:
                        img_grid = torch.cat((dat_hr,gen_hr), -1)
                    img_grid = img_grid.cpu()
                    img_grid = img_grid.detach().numpy()
    
                    nsubpfig = 6 # subfigure per figure
                    nfig = int(-(-len(img_grid) // nsubpfig))
                    for j in np.arange(nfig):
                        ne = min((j+1)*nsubpfig,len(img_grid))
                        ind = np.arange(j*nsubpfig,ne)
                        image = img_grid[ind,...]
                        ncol = 2
                        for k in range(nchl_o):
                            figname = out_path+"c%d_epoch%d_batch%d_id%d.png" % (ivar_hr[k],epoch,i,j)
                            plt_sub(image,ncol,figname,k)

            # evaluation using validation set.
            if Nbatch_v>0: 
                generator.eval()
                with torch.no_grad():
                    val_bch  = {'loss_G': 0, 'psnr': 0, 'ssim': 0, 'mse':0}
                    for i, dat in enumerate(data_valid):
                        dat_lr = Variable(dat["lr"].type(Tensor))
                        dat_hr = Variable(dat["hr"].type(Tensor))
                        
                        if depth_aux is not None:
                            ibcsize = len(dat_lr) # this is to make sure batchsize of dat_aux and dat_lr match
                            dat_aux = Variable(depth_aux[0:ibcsize,...].type(Tensor))
                        else:
                            dat_aux = None
                        if knn==0:
                            gen_hr = generator(dat_lr,dat_aux)
                        else:
                            gen_hr = generator(dat_lr)
                    
                        loss_G = my_loss(gen_hr,dat_hr,opt.nlm) 
                        val_bch['loss_G'] += loss_G.item()
                        batch_mse = ((gen_hr - dat_hr) ** 2).data.mean()
                        val_bch['mse'] += batch_mse.item()
                        batch_ssim = ssim_torch(gen_hr, dat_hr,data_range=1.0).item()
                        val_bch['ssim'] += batch_ssim
                        # batch_psnr = 10 * log10((dat_hr.max()**2) / batch_mse)
                        batch_psnr = 10 * log10(1.0 / batch_mse)
                        val_bch['psnr'] += batch_psnr
                
                        if epoch % opt.sample_epoch == 0 and i % opt.sample_interval == 0:
                            # Save image grid with upsampled inputs and SRGAN outputs
                            dat_lr = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor)
                            if nchl_i == nchl_o ==1: # same vars or 1 var to 1 var
                                img_grid = torch.cat((dat_lr, dat_hr,gen_hr), -1)
                            else:
                                img_grid = torch.cat((dat_hr,gen_hr), -1)
                            img_grid = img_grid.cpu()
                            img_grid = img_grid.detach().numpy()
            
                            nsubpfig = 6 # subfigure per figure
                            nfig = int(-(-len(img_grid) // nsubpfig))
                            for j in np.arange(nfig):
                                ne = min((j+1)*nsubpfig,len(img_grid))
                                ind = np.arange(j*nsubpfig,ne)
                                image = img_grid[ind,...]
                                ncol = 2
                                for k in range(nchl_o):
                                    figname = out_path+"c%d_epoch%d_batch%d_id%d_v.png" % (ivar_hr[k],epoch,i,j)
                                    plt_sub(image,ncol,figname,k)
                
            # save model parameters
            if (opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and 
                epoch != opt.n0_epoch and epoch>=40): # do not save the first epoch, skip first 20 epochs
                torch.save({
                    'model_state_dict':generator.state_dict(),
                    'optimizer_state_dict':optimizer_G.state_dict()}, opath_nn+'netG_epoch_%d_re%d.pth' % (epoch,irep))

            # save loss\scores\psnr\ssim
            results['loss_G'].append(eva_bch['loss_G'] / Nbatch)
            results['psnr'].append(eva_bch['psnr'] / Nbatch) # use trainset instead of test, replace Nbatch_t
            results['ssim'].append(eva_bch['ssim'] / Nbatch) # 
            results['mse'].append(eva_bch['mse'] / Nbatch) # 
            if Nbatch_v>0: 
                results['loss_G_v'].append(val_bch['loss_G'] / Nbatch_v)
                results['psnr_v'].append(val_bch['psnr'] / Nbatch_v) # 
                results['ssim_v'].append(val_bch['ssim'] / Nbatch_v) # 
                results['mse_v'].append(val_bch['mse'] / Nbatch_v) 
            
        
            # if epoch % opt.sample_interval == 0 and epoch != 0:
            if Nbatch_v==0: 
                data_frame = pd.DataFrame(
                    data={'loss_G': results['loss_G'], 'psnr': results['psnr'], 'ssim': results['ssim'], 
                          'mse':results['mse']},
                    index=range(opt.n0_epoch+1, epoch+2))
            else:
                data_frame = pd.DataFrame(
                    data={'loss_G': results['loss_G'], 'psnr': results['psnr'], 'ssim': results['ssim'], 'mse':results['mse'],
                          'loss_G_v': results['loss_G_v'], 'psnr_v': results['psnr_v'], 'ssim_v': results['ssim_v'], 'mse_v':results['mse_v']},
                    index=range(opt.n0_epoch+1, epoch+2))
            data_frame.to_csv(opath_st + 'srf_%d_re%d' % (opt.up_factor,irep)+ '_train_results.csv', index_label='Epoch')

        figname = opath_st+'train_loss'+'_re'+str(irep)+'.png'
        epo_lst = np.arange(opt.n0_epoch+1, epoch+2).tolist()
        if Nbatch_v==0: 
            x_lst = [epo_lst]
            dat_lst = [results['loss_G']]
            leg = ['train',]
        else:
            x_lst = [epo_lst,epo_lst]
            dat_lst = [results['loss_G'],results['loss_G_v']]
            leg = ['train','validation']
        plot_line_list(x_lst,dat_lst,figname=figname,axlab=['epoch','loss'],leg=leg,
                       leg_col=1, legloc=None,line_sty=None,capt='')
