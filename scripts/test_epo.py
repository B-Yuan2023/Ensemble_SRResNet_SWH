"""
Super-resolution 

"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet
from mod_srres import SRRes #,SRResA
from datasets import myDataset
from funs_prepost import plt_sub,var_denormalize,nc_load_vars,plt_pcolor_list,plot_line_list,nc_load_depth

import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name= sys.argv[1]         #'par01_rd0' # sys.argv[1]
sys.path.append("../")  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

def cal_metrics(pr,hr,pr_norm0,hr_norm0): # pr,hr are tensors [N,C,H,W], norm0 are arrays,mask
    # hr_norm0[:,:,mask] = np.nan
    nchl = hr.shape[1]
    mse = np.nanmean((pr_norm0 - hr_norm0) ** 2,axis=(0,2,3)) 
    rmse = (mse)**(0.5)
    mae = np.nanmean(abs(pr_norm0 - hr_norm0),axis=(0,2,3))
    
    # to calculate ssim, there should be no nan
    ssim_tor = ssim_torch(pr, hr,data_range=1.0,size_average=False) #.item()  # ,win_size=11
    ssim = np.array([ssim_tor[0,i].item() for i in range(nchl)])
    
    # mask_ud = np.flipud(mask) # dimensionless data flipped
    # hr[:,:,mask_ud.copy()] = np.nan  # for tensor copy is needed. why hr is modified after call
    mse_norm = torch.nanmean(((pr - hr) ** 2).data,axis=(0,2,3)) #.item()
    psnr = np.array([10 * log10(1/mse_norm[i]) for i in range(nchl)]) # for data range in [0,1]
    return rmse, mae, mse, ssim, psnr 


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
    
    if isinstance(rtra, (int, float)): # if only input one number, no validation
        rtra = [rtra,0]
    
    # create nested list of files and indt
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]

    if hasattr(mod_para, 'tshift'):
        tshift = mod_para.tshift # time shift in hour of low resolution data
    else:
        tshift = 0
    if hasattr(mod_para, 'll_lr'):
        ll_lr = mod_para.ll_lr # user domain latitude
    else:
        ll_lr = [None]*2
    if hasattr(mod_para, 'll_hr'):
        ll_hr = mod_para.ll_hr # user domain longitude
    else:
        ll_hr = [None]*2
    if hasattr(mod_para, 'kintp'):
        kintp = mod_para.kintp # 1, griddata, 2 RBFInterpolator
    else:
        kintp = [0,0] # no interpolation for lr and hr
        
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

    if hasattr(mod_para, 'knn'): # types of nn models, default 0,srres; 1 SRCNN, 2 FSRCNN
        knn = mod_para.knn
    else:
        knn = 0    
    if hasattr(mod_para, 'ker_size'): # kernel size, only work for SRRes_MS,SRRes_MS1
        ker_size = mod_para.ker_size # 
    else:
        ker_size = 3
    if hasattr(mod_para, 'kernel_no'): # kernel size, only work for SRRes_MS,SRRes_MS1
        kernel_no = mod_para.kernel_no # 
    else:
        kernel_no = 64
    if hasattr(mod_para, 'dsm'): # kernel numbers d,s and repeated layers m in FSRCNN,FSRCNN1
        dsm = mod_para.dsm # 
    else:
        dsm = [56,12,4]
    if hasattr(mod_para, 'mod_att'): # attenion for SRResA, 10-11 SE, 12 GCT, 20 CBAM
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
    
    # epoc_num = [50,100]
    epoc_num = np.arange(40,opt.N_epochs+1)

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    Nt = len(test_set)
    
    opath_st = '../stat' + suf +'_mk'+str(kmask)+'/'
    if not os.path.exists(opath_st):
        os.makedirs(opath_st)
        
    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    )        
    Nbatch_t = len(data_test)        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    # do not read reconstruction results 
    # metrics_re_bt_chl = {}
    # out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+'_mask'+str(kmask)+'/'
    # # load metrics along batch for direct interpolation 
    # filename = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
    # metrics_re_bt = np.load(filename,allow_pickle='TRUE').item()
    
    # filename99 = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
    # datald = np.load(filename99) # load
    # hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']

    # filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
    # datald = np.load(filename99m) # load
    # rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']

    # sr, GT, diff
    clim = [[[1.3,3.3],[1.3,3.3],[-0.2,0.2]],  # ssh
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # u
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # v
            [[12,15],[12,15],[-1.0,1.0]],  # uw
            [[12,15],[12,15],[-1.0,1.0]],  # vw
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],  # swh
            [[5.0,15],[5.0,15],[-2.0,2.0]],  # pwp
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],]  # swh_ww
    # # nearest,bicubit, sr, GT, diff,diff,diff
    # clim = [[[1.3,3.3],[1.3,3.3],[1.3,3.3],[1.3,3.3],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2]],  # ssh
    #         [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # u
    #         [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # v
    #         [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # uw
    #         [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # vw
    #         [[2.0,5.0],[2.0,5.0],[2.0,5.0],[2.0,5.0],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]],  # swh
    #         [[5.0,15.],[5.0,15.],[5.0,15.],[5.0,15.],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0]],  # pwp
    #         ]
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']

    # layers: repeat/epoch/batch/channel 
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')

        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = '../nn_mod_' + str(opt.up_factor) + suf +'/' # 
    
        out_path = '../results_test/'+'SRF_'+str(opt.up_factor)+suf+'_ep'+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        os.makedirs(out_path, exist_ok=True)
    
        opath_st_rp = opath_st+'re'+ str(irep)+'/'  # 'statistics' + suf +
        os.makedirs(opath_st_rp, exist_ok=True)
        
        # Initialize generator 
        if knn ==0:
            generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                        n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor)
        # discriminator = Discriminator(input_shape=(nchl_o, *hr_shape))
        elif knn == 1:
            generator = SRRes(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=opt.residual_blocks,
                                        up_factor=opt.up_factor,kernel_size=ker_size,kernel_no=kernel_no)

        metrics = {'ep':[],'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], 'mae_99': [],'rmse_99': [],
                   'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],'mae_t': [],'rmse_t': [],} # eopch mean
        metrics_chl = {}
            
        for epoch in epoc_num:
            
            metrics_bt = {'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], } # batch mean
            metrics_bt_chl = {'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], } # each channel
            
            # # check if 99th file is saved 
            # filename99 = out_path + "sr_99th_epoch%d" % (epoch)+'.npz'
            # filename01 = out_path + "sr_01th_epoch%d" % (epoch)+'.npz'
            # filename_m = out_path + "sr_mean_epoch%d" % (epoch)+'.npz'
        
            model_name = 'netG_epoch_%d_re%d.pth' % (epoch,irep)
            if cuda:
                generator = generator.cuda()
                checkpointG = torch.load(ipath_nn + model_name)
            else:
                checkpointG = torch.load(ipath_nn + model_name, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            generator.eval()
            
            sr_all = []
            hr_all = []

            for i, dat in enumerate(data_test):                
                
                dat_lr = Variable(dat["lr"].type(Tensor))
                dat_hr = Variable(dat["hr"].type(Tensor))
                if depth_aux is not None:
                    ibcsize = len(dat_lr) # this is to make sure batchsize of dat_aux and dat_lr match
                    dat_aux = Variable(depth_aux[0:ibcsize,...].type(Tensor))
                else:
                    dat_aux = None
                # Generate a high resolution image from low resolution input
                if knn==0:
                    gen_hr = generator(dat_lr,dat_aux)
                else:
                    gen_hr = generator(dat_lr)
                sr_norm0 = var_denormalize(gen_hr.detach().cpu().numpy(),varm_hr) # (N,C,H,W), flipud height back
                hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
                
                # get mask for time step
                mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
                for ib in range(opt.batch_size):  # use mask for each sample/time
                    it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                    if it>=len(test_set):  # for case the last batch has samples less than batch_size
                        break
                    for ichl in range(nchl):
                        nc_f = test_set.files_hr[it][ichl]
                        indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                        mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                    
                if kmask == 1: 
                    sr_norm0[mask] = np.nan
                    hr_norm0[mask] = np.nan
                    # sr_norm0[:,:,mask] = np.nan
                sr_all.append(sr_norm0)
                hr_all.append(hr_norm0)
                
                rmse, mae, mse, ssim, psnr = cal_metrics(gen_hr,dat_hr,sr_norm0,hr_norm0) # ,mask
                metrics_bt['mse'].append(mse)
                metrics_bt['mae'].append(mae)
                metrics_bt['rmse'].append(rmse)
                metrics_bt['ssim'].append(ssim)
                metrics_bt['psnr'].append(psnr)
        
                if epoch % opt.sample_epoch == 0 and i % opt.sample_interval == 1:
                    # Save image grid with upsampled inputs and SR outputs
                    dat_lr_nr = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor)# default nearest;bicubic; input 4D/5D
                    dat_lr_cu = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='bicubic') 
                    if nchl_i == nchl_o ==1: # 1 var to 1 var
                        img_grid = torch.cat((dat_lr_nr,dat_lr_cu,gen_hr, dat_hr), -1)
                    else:
                        img_grid = torch.cat((dat_hr,gen_hr), -1)
                    img_grid = img_grid.cpu()
                    img_grid = img_grid.detach().numpy()
    
                    nsubpfig = 8 # subfigure per figure
                    nfig = int(-(-len(img_grid) // nsubpfig))
                    for j in np.arange(nfig):
                        ne = min((j+1)*nsubpfig,len(img_grid))
                        ind = np.arange(j*nsubpfig,ne)
                        image = img_grid[ind,...]
                        ncol = 2
                        for k in range(nchl_o):
                            figname = out_path+"c%d_epoch%d_batch%d_id%d.png" % (ivar_hr[k],epoch,i,j)
                            plt_sub(image,ncol,figname,k)
            
            hr_all = np.concatenate(hr_all, axis=0)
            # hr_all = np.array(hr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
            sr_all = np.concatenate(sr_all, axis=0)
            # sr_all = np.array(sr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
            
            sr_99per = np.nanpercentile(sr_all, 99, axis = (0,))
            sr_01per = np.nanpercentile(sr_all, 1, axis = (0,))
            sr_mean = np.nanmean(sr_all, axis = (0,))
            
            filename99 = out_path + 'hr_99per'+'_train%4.2f'%(rtra[0])+'.npz'
            if not os.path.isfile(filename99):
                hr_99per = np.nanpercentile(hr_all, 99, axis = (0,))
                hr_01per = np.nanpercentile(hr_all, 1, axis = (0,))
                hr_mean = np.nanmean(hr_all, axis = (0,))
                np.savez(filename99,v0=hr_99per,v1=hr_01per,v2=hr_mean) 
            else:
                datald = np.load(filename99) # load
                hr_99per= datald['v0']

            # rmse_99,mae_99 = np.zeros((nchl)),np.zeros((nchl))            
            # for i in range(nchl_o): # note: when hr_99per is loaded, 3 channels for s,u,v respectively
            #     ichl = ivar_hr[i]-3
            #     rmse_99[i] = np.nanmean((sr_99per[i,:,:] - hr_99per[ichl,:,:]) ** 2)**(0.5)
            #     mae_99[i] = np.nanmean(abs(sr_99per[i,:,:] - hr_99per[ichl,:,:]))
            
            # estimate and save sr_99per in this script
            rmse_99 = np.nanmean((sr_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
            mae_99 = np.nanmean(abs(sr_99per - hr_99per),axis=(1,2))
            # save 99th data for each epoch 
            filename99 = out_path + "sr_99th_epoch%d" % (epoch)+'.npz'
            np.savez(filename99,v0=sr_99per,v1=hr_99per,v2=rmse_99,v3=mae_99) 
            
            # estimate and save sr_01per in this script
            rmse_01 = np.nanmean((sr_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
            mae_01 = np.nanmean(abs(sr_01per - hr_01per),axis=(1,2))
            # save 99th data for each epoch 
            filename01 = out_path + "sr_01th_epoch%d" % (epoch)+'.npz'
            np.savez(filename01,v0=sr_01per,v1=hr_01per,v2=rmse_01,v3=mae_01) 

            # estimate and save sr_mean in this script
            rmse_m = np.nanmean((sr_mean - hr_mean) ** 2,axis=(1,2))**(0.5) # spatial rmse of temporal mean 
            mae_m = np.nanmean(abs(sr_mean - hr_mean),axis=(1,2))
            # save 99th data for each epoch 
            filename_m = out_path + "sr_mean_epoch%d" % (epoch)+'.npz'
            np.savez(filename_m,v0=sr_mean,v1=hr_mean,v2=rmse_m,v3=mae_m) 
            
            # estimate and save sr_rmse sr_mae 
            sr_rmse = np.nanmean((sr_all - hr_all) ** 2,axis=(0))**(0.5) # temporal rmse per point per channel (C,H,W)
            sr_mae = np.nanmean(abs(sr_all - hr_all),axis=(0))
            rmse_t = np.nanmean((sr_rmse),axis=(1,2))  # spatial average of rmse (C)
            mae_t = np.nanmean(sr_mae,axis=(1,2))
            # save 99th data for each epoch 
            filename_m = out_path + "sr_tave_epoch%d" % (epoch)+'.npz'
            np.savez(filename_m,v0=sr_rmse,v1=sr_mae,v2=rmse_t,v3=mae_t) 
            
            
            # if epoch % opt.sample_epoch == 0:
            for i in range(nchl_o):
                ichl = ivar_hr[i]
                clim_chl = clim[ichl]                
                sample  = [hr_99per[i,:,:],sr_99per[i,:,:],sr_99per[i,:,:]-hr_99per[i,:,:]]
                # sample  = [hr_re1_99per[ichl,:,:],
                #            hr_re1_99per[ichl,:,:],
                #            sr_99per[i,:,:],
                #            hr_99per[ichl,:,:],
                #            hr_re1_99per[ichl,:,:]-hr_99per[ichl,:,:],
                #            hr_re1_99per[ichl,:,:]-hr_99per[ichl,:,:],
                #            sr_99per[i,:,:]-hr_99per[ichl,:,:],
                #            ]
                unit = [unit_suv[ichl]]*len(sample)
                title = ['hr_99','sr_99','sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')']
                # title = ['nearest_99','bilinear_99','sr_99','hr_99',
                #          'nearest-hr'+'(%5.3f'%mae_99_re3[ichl]+',%5.3f'%rmse_99_re3[ichl]+')',
                #          'bilinear-hr'+'(%5.3f'%mae_99_re2[ichl]+',%5.3f'%rmse_99_re2[ichl]+')',
                #          'sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')',]
                figname = out_path+"99th_c%d_epoch%d_ax0.png" % (ivar_hr[i],epoch)
                plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2,axoff=1) 
                # figname = out_path+"99th_c%d_epoch%d_.png" % (ivar_hr[i],epoch)
                # plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2)

            # save metrics epoch average
            for key, value in metrics_bt.items():
                metrics[key].append(sum(metrics_bt[key])/len(metrics_bt[key])) # / Nbatch_t) # 
            for i in range(len(metrics['rmse'])):  # rmse should first average next sqrt
                metrics['rmse'][i] = metrics['mse'][i]**0.5
            metrics['mae_99'].append(mae_99)
            metrics['rmse_99'].append(rmse_99)
            metrics['mae_01'].append(mae_01)
            metrics['rmse_01'].append(rmse_01)
            metrics['mae_m'].append(mae_m)
            metrics['rmse_m'].append(rmse_m)
            metrics['mae_t'].append(mae_t)
            metrics['rmse_t'].append(rmse_t)
            metrics['ep'].append([epoch]*nchl)
            
            # output batch average 
            for i in range(nchl):
                for key, value in metrics_bt.items():
                    metrics_bt_chl[key] = [value[j][i] for j in range(0,len(value))]
                # for key, value in metrics_re_bt.items():
                #     metrics_re_bt_chl[key] = [value[j][ichl] for j in range(0,len(value))]
                data_frame = pd.DataFrame.from_dict(metrics_bt_chl, orient='index').transpose() # |metrics_re_bt_chl
                ofname = "srf_%d_re%d_ep%d_c%d" % (opt.up_factor,irep,epoch,ivar_hr[i]) + '_test_metrics.csv'
                data_frame.to_csv(opath_st_rp + os.sep + ofname, index_label='batch')
                
                # if epoch % opt.sample_epoch == 0:
                #     # plot batch average 
                #     legloc = (0.3,0.40)
                    
                #     leg = ['sr','bicubic','bilinear','nearest']
                #     axlab = [['Batch','rmse(ssh) (m)'],['Batch','rmse(u) (m/s)'],['Batch','rmse(v) (m/s)']]
                #     var = np.array(metrics_bt['rmse'])[:,i]
                #     var_res1 = np.array(metrics_re_bt['rmse_re1'])[:,ichl]
                #     var_res2 = np.array(metrics_re_bt['rmse_re2'])[:,ichl]
                #     var_res3 = np.array(metrics_re_bt['rmse_re3'])[:,ichl]
                #     time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                #     data_lst = [var,var_res1,var_res2,var_res3]
                #     figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_rmse.png'
                #     # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab[ichl],leg=leg,leg_col=2)
                #     plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab[ichl],
                #                     leg=leg,leg_col=2,legloc=legloc,capt='(a)') #,style=style
    
                #     axlab = [['Batch','mae(ssh) (m)'],['Batch','mae(u) (m/s)'],['Batch','mae(v) (m/s)']]
                #     var = np.array(metrics_bt['mae'])[:,i]
                #     var_res1 = np.array(metrics_re_bt['mae_re1'])[:,ichl]
                #     var_res2 = np.array(metrics_re_bt['mae_re2'])[:,ichl]
                #     var_res3 = np.array(metrics_re_bt['mae_re3'])[:,ichl]
                #     time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                #     data_lst = [var,var_res1,var_res2,var_res3]
                #     figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_mae.png'
                #     # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab[ichl],leg=leg,leg_col=2)
                #     plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab[ichl],
                #                     leg=leg,leg_col=2,legloc=legloc,capt='(b)') #,style=style
    
                    # axlab = ['Batch','psnr']
                    # var = np.array(metrics_bt['psnr'])[:,i]
                    # var_res1 = np.array(metrics_re_bt['psnr_re1'])[:,ichl]
                    # var_res2 = np.array(metrics_re_bt['psnr_re2'])[:,ichl]
                    # var_res3 = np.array(metrics_re_bt['psnr_re3'])[:,ichl]
                    # time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                    # data_lst = [var,var_res1,var_res2,var_res3]
                    # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_psnr.png'
                    # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab,leg=leg,leg_col=2)
                    
                    # axlab = ['Batch','ssim']
                    # var = np.array(metrics_bt['ssim'])[:,i]
                    # var_res1 = np.array(metrics_re_bt['ssim_re1'])[:,ichl]
                    # var_res2 = np.array(metrics_re_bt['ssim_re2'])[:,ichl]
                    # var_res3 = np.array(metrics_re_bt['ssim_re3'])[:,ichl]
                    # time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                    # data_lst = [var,var_res1,var_res2,var_res3]
                    # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_ssim.png'
                    # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab,leg=leg,leg_col=2)

        
        # output metrics for all epochs to csv
        for i in range(nchl):
            for key, value in metrics.items():
                metrics_chl[key] = [value[j][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl, orient='index').transpose()
            ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_test_metrics.csv'
            data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        np.save(opath_st + os.sep + ofname, metrics) 
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics_bt.npy'
        np.save(opath_st + os.sep + ofname, metrics_bt) 
        
        # output sorted metrics based on rmse99 for all epochs to csv
        ind_sort = [[]]*nchl
        metrics_chl_sort = {}
        # list rmse from small to large
        for i in range(nchl):
            var = [metrics['rmse_99'][j][i] for j in range(len(metrics['rmse_99']))]
            ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
            for key, value in metrics.items():
                metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
            ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_metrics_sort.csv'
            data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
            
        # output epoc oder for channels
        ep_sort = [[]]*nchl
        for i in range(nchl):
            ep_sort[i] = [metrics['ep'][ind_sort[i][j]][i] for j in range(len(metrics['ep']))]
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_epoc_sort.csv'
        np.savetxt(opath_st + ofname, np.array(ep_sort).transpose(), fmt='%d',delimiter=",")
        # ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")
        
        # output epochs based on rmse instead of rmse99 
        ind_sort = [[]]*nchl
        metrics_chl_sort = {}
        # list rmse from small to large
        for i in range(nchl):
            var = [metrics['rmse'][j][i] for j in range(len(metrics['rmse']))]
            ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
            for key, value in metrics.items():
                metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
            data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
            ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_metrics_sort_rmse.csv'
            data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
            
        ep_sort = [[]]*nchl
        for i in range(nchl):
            ep_sort[i] = [metrics['ep'][ind_sort[i][j]][i] for j in range(len(metrics['ep']))]
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_epoc_sort_rmse.csv'
        np.savetxt(opath_st + ofname, np.array(ep_sort).transpose(), fmt='%d',delimiter=",")



    


