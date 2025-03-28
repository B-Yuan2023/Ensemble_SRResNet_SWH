"""
Super-resolution 

"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorResNet
from datasets import myDataset
from funs_prepost import var_denormalize,nc_load_vars,plt_pcolor_list,plot_line_list,plt_sub,nc_load_depth

import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name= 'par55'          #'par04' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 
kmask = 1

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
        kintp = [0,0] # no interpolation
        
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
    
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    rep = [1] # par01(4,97),s4(4,98),s32(3,100); par11(2,98),s4(2,100),s32(0,95)

    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    epoc_num = [78]
    # epoc_num = np.arange(40,opt.N_epochs+1)
    key_ep_sort = 0 # 0 to use epoc here or 1 load sorted epoc no. 
    nepoc = 1 # no. of sorted epochs for analysis

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    
    
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    if not os.path.exists(opath_st):
        os.makedirs(opath_st)
    
    opath_st_hr = 'statistics_hr'+'_%d_%d'%(opt.hr_height, opt.hr_width)+'/' 
        
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
    lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    metrics_re_bt_chl = {}
    out_path = 'results_test/'+'SRF_%d_mk%d'%(opt.up_factor,kmask)+'/'+ var_hr[0]+'/'
    # load metrics along batch for direct interpolation 
    filename = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
    metrics_re_bt = np.load(filename,allow_pickle='TRUE').item()
    
    filename99 = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
    filename01 = out_path + 'hr_01per_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
    filename_m = out_path + 'hr_mean_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
    datald = np.load(filename99) # load
    hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    datald = np.load(filename01) # load
    hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    datald = np.load(filename_m) # load
    hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']

    filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
    filename01m = out_path + 'hr_01per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
    filename_mm = out_path + 'hr_mean_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
    datald = np.load(filename99m) # load
    rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    datald = np.load(filename01m) # load
    rmse_01_re1,rmse_01_re2,rmse_01_re3,mae_01_re1,mae_01_re2,mae_01_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    datald = np.load(filename_mm) # load
    rmse_m_re1,rmse_m_re2,rmse_m_re3,mae_m_re1,mae_m_re2,mae_m_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
    dir_hr = opt.dir_hr
    ichlo = ivar_hr[0] # only work for one variable
    # load maximum var in test set for showing time evolution
    nbt = 3
    filename = opath_st_hr+'var%d'%ichlo+'_sorted_test'+'_rt%4.2f'%(rtra)+'.npz'
    datald = np.load(filename) # load
    nfl = datald['v1'].size
    var_sort = datald['v1'][0][0:nbt]
    ind_sort = datald['v2'][0][0:nbt]  # index of maximum var in the test set
    ind_bt = (ind_sort/opt.batch_size).astype(int)  # id of sorted batchsize in test set

    # sr, GT, diff
    clim = [[[1.3,3.3],[1.3,3.3],[-0.2,0.2]],  # ssh
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # u
            [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # v
            [[12,15],[12,15],[-1.0,1.0]],  # uw
            [[12,15],[12,15],[-1.0,1.0]],  # vw
            [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],  # swh
            [[5.0,15],[5.0,15],[-2.0,2.0]],  # pwp
            ]
    # nearest,bicubit, sr, GT, diff,diff,diff
    clim = [[[1.3,3.3],[1.3,3.3],[1.3,3.3],[1.3,3.3],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2]],  # ssh
            [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # u
            [[0.2,1.8],[0.2,1.8],[0.2,1.8],[0.2,1.8],[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]],  # v
            [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # uw
            [[12,15],[12,15],[12,15],[12,15],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],  # vw
            [[2.0,5.0],[2.0,5.0],[2.0,5.0],[2.0,5.0],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]],  # swh
            [[5.0,15.],[5.0,15.],[5.0,15.],[5.0,15.],[-2.0,2.0],[-2.0,2.0],[-2.0,2.0]],  # pwp
            ]
    # diff,diff,diff, hr
    clim_99 = [[[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[1.8,3.3]],  # ssh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.2,1.8]],  # u
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.2,1.8]],  # v
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.2,1.8]],  # uw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.2,1.8]],  # vw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[1.0,3.5]],  # swh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.2,1.8]],  # pwp
            ]
    clim_01 = [[[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-2.6,0.4]],  # ssh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-1.7,-0.0]],  # u
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-1.7,-0.0]],  # v
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-1.7,-0.0]],  # uw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-1.7,-0.0]],  # vw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.0,0.4]],  # swh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.0,6.0]],  # pwp
            ]
    clim_m = [[[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[0.0,0.8]],  # ssh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-0.3,0.4]],  # u
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-0.2,0.3]],  # v
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-0.3,0.4]],  # uw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[-0.2,0.3]],  # vw
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[0.0,1.5]],  # swh
            [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3],[5,10]],  # pwp
            ]
    clim_a = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,4.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp

    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']

#  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    capt_all = ['('+alpha[i]+')' for i in range(len(alpha))]
    
    # layers: repeat/epoch/batch/channel 
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')

        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = 'nn_models_' + str(opt.up_factor) + suf +'/' # 
    
        out_path = 'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_ep'+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        os.makedirs(out_path, exist_ok=True)
    
        opath_st_rp = opath_st+'re'+ str(irep)+'/'  # 'statistics' + suf +
        os.makedirs(opath_st_rp, exist_ok=True)
        
        # Initialize generator and discriminator
        generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                    n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor).eval()

        # metrics = {'ep':[],'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], 'mae_99': [],'rmse_99': [],} # eopch mean
        # metrics_chl = {}
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_test_metrics.npy'
        # np.save(opath_st + os.sep + ofname, metrics) 
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_test_metrics_bt.npy'
        # np.save(opath_st + os.sep + ofname, metrics_bt) 
        # metrics_bt = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        
        if key_ep_sort:
            # choose sorted epoc number that gives the smallest rmse_99 and rmse
            ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort.csv' #  rank based on rmse99
            # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort'+tstr+'.csv' # rank based on rt_use highest ssh/
            ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
            ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort_rmse.csv' #  rank based on rmse
            ep_sort1 = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
            epoc_num = np.concatenate([ep_sort.flatten()[0:nepoc*nchl_o],ep_sort1.flatten()[0:nepoc*nchl_o]])
            epoc_num = list(set(epoc_num.tolist())) # remove duplicates,order not reseved
    
        for epoch in epoc_num:
            
            # metrics_bt = {'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], } # batch mean
            # metrics_bt_chl = {'mse': [], 'mae': [], 'rmse': [],'ssim': [],'psnr': [], } # each channel
        
            # check if 99th file is saved 
            filename99 = out_path + "sr_99th_epoch%d" % (epoch)+'.npz'
            filename01 = out_path + "sr_01th_epoch%d" % (epoch)+'.npz'
            filename_m = out_path + "sr_mean_epoch%d" % (epoch)+'.npz'
            # if not os.path.isfile(filename99) or not os.path.isfile(filename01) or not os.path.isfile(filename_m):
                
            model_name = 'netG_epoch_%d_re%d.pth' % (epoch,irep)
            if cuda:
                generator = generator.cuda()
                checkpointG = torch.load(ipath_nn + model_name)
            else:
                checkpointG = torch.load(ipath_nn + model_name, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            
            sr_all = []

            for i, dat in enumerate(data_test):                
                
                dat_lr = Variable(dat["lr"].type(Tensor))
                dat_hr = Variable(dat["hr"].type(Tensor))
                if depth_aux is not None:
                    ibcsize = len(dat_lr) # this is to make sure batchsize of dat_aux and dat_lr match
                    dat_aux = Variable(depth_aux[0:ibcsize,...].type(Tensor))
                else:
                    dat_aux = None
                gen_hr = generator(dat_lr,dat_aux)
                sr_norm0 = var_denormalize(gen_hr.detach().cpu().numpy(),varm_hr) # (N,C,H,W), flipud height back
                hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)

                dat_lr_nr = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor)# default nearest;bicubic; input 4D/5D
                # in case lr_nr_norm0 has 1 channel, les than hr, replicate the array
                if nchl_i==1 and nchl_i<nchl_o:
                    dat_lr_nr = torch.repeat_interleave(dat_lr_nr,nchl_o, dim=1)
                lr_nr_norm0 = var_denormalize(dat_lr_nr.detach().cpu().numpy(),varm_hr)
                # lr_nr_norm0 = np.repeat(lr_nr_norm0,nchl_o, axis=1)

                
                if kmask == 1:                     
                    # get mask for time step
                    mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
                    for ib in range(opt.batch_size):  # use mask for each sample/time
                        it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                        if it>=len(test_set):  # for case the last batch has samples less than batch_size
                            break
                        nc_f = test_set.files_hr[it]
                        indt = indt_hr[it]  # the time index in a ncfile
                        for ichl in range(nchl):
                            mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                        
                    sr_norm0[mask] = np.nan 
                    hr_norm0[mask] = np.nan
                    lr_nr_norm0[mask] = np.nan
                    # sr_norm0[:,:,mask] = np.nan 
                sr_all.append(sr_norm0)
                
                if i in ind_bt: # i % opt.sample_interval == 1:
                    # Save image grid with upsampled inputs and SR outputs
                    cat_dim = 2 # concatenate for dimension H:2, W:-1 or 3. 
                    if nchl_i == nchl_o ==1 or cat_dim == 2: # same vars or 1 var to 1 var
                        img_grid = torch.cat((dat_lr_nr, dat_hr,gen_hr), cat_dim)
                        img_grid = img_grid.cpu().detach().numpy()
                        img_grid_nm0 = np.concatenate((np.flip(lr_nr_norm0,2),
                                                       np.flip(hr_norm0,2), np.flip(sr_norm0,2)), cat_dim) 
                    else:
                        img_grid = torch.cat((dat_hr,gen_hr), cat_dim)
                        img_grid = img_grid.cpu().detach().numpy()
                        img_grid_nm0 = np.concatenate((np.flip(hr_norm0,2), np.flip(sr_norm0,2)), cat_dim)
    
                    nsubpfig = 6 # subfigure per figure
                    nfig = int(-(-len(img_grid) // nsubpfig))
                    for j in np.arange(nfig):
                        ne = min((j+1)*nsubpfig,len(img_grid)) # index of the last sample in a plot
                        ind = np.arange(j*nsubpfig,ne) # index of the samples in a plot
                        image = img_grid[ind,...]
                        image_nm0 = img_grid_nm0[ind,...] #*(N,C,H,W)
                        ncol = 2
                        if cat_dim == 2: # if cat hr, sr in vertical direction, cat samples in W direction
                            temp,temp_nm0 = image[0,:] ,image_nm0[0,:]
                            for ij in range(1,len(image)):
                                temp = np.concatenate((temp,image[ij,...]), -1)
                                temp_nm0 = np.concatenate((temp_nm0,image_nm0[ij,...]), -1)
                            image = temp.reshape(1,temp.shape[0],temp.shape[1],temp.shape[2])
                            image_nm0 = temp_nm0.reshape(1,temp.shape[0],temp.shape[1],temp.shape[2])
                            ncol=1
                        for k in range(nchl_o):
                            ichl = ivar_hr[k]
                            figname = out_path+"c%d_epoch%d_batch%d_id%d.png" % (ichl,epoch,i,j)
                            contl = [-0.01-varm_hr[k][1]/(varm_hr[k][0]-varm_hr[k][1])]
                            plt_sub(image,ncol,figname,cmp='bwr',contl=contl) # 'bwr','coolwarm'
                            figname = out_path+"c%d_epoch%d_batch%d_id%d_nm0.png" % (ichl,epoch,i,j)
                            clim_v = clim_a[ichl]
                            plt_sub(image_nm0,ncol,figname,k,clim=clim_v,cmp='bwr',contl=[-0.05,]) # 'bwr','coolwarm'
            
            sr_all = np.concatenate(sr_all, axis=0)
            # sr_all = np.array(sr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
            sr_99per = np.nanpercentile(sr_all, 99, axis = (0,))
            sr_01per = np.nanpercentile(sr_all, 1, axis = (0,))
            sr_mean = np.nanmean(sr_all, axis = (0,))
            
            if not os.path.isfile(filename99):
                rmse_99, mae_99 = np.zeros((nchl)),np.zeros((nchl))
                for i in range(nchl_o): # note: hr_99per is loaded
                    ichl = ivar_hr[i]
                    rmse_99[i] = np.nanmean((sr_99per[i,:,:] - hr_99per[i,:,:]) ** 2)**(0.5)
                    mae_99[i] = np.nanmean(abs(sr_99per[i,:,:] - hr_99per[i,:,:]))
                # rmse_99 = np.nanmean((sr_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
                # mae_99 = np.nanmean(abs(sr_99per - hr_99per),axis=(1,2))
                np.savez(filename99,v0=sr_99per,v1=hr_99per,v2=rmse_99,v3=mae_99) 
            else: 
                datald = np.load(filename99) # load
                sr_99per,rmse_99,mae_99 = datald['v0'],datald['v2'],datald['v3']

            if not os.path.isfile(filename01):
                rmse_01, mae_01 = np.zeros((nchl)),np.zeros((nchl))
                for i in range(nchl_o): # note: hr_01per is loaded
                    ichl = ivar_hr[i]
                    rmse_01[i] = np.nanmean((sr_01per[i,:,:] - hr_01per[i,:,:]) ** 2)**(0.5)
                    mae_01[i] = np.nanmean(abs(sr_01per[i,:,:] - hr_01per[i,:,:]))
                np.savez(filename01,v0=sr_01per,v1=hr_01per,v2=rmse_01,v3=mae_01) 
            else: 
                datald = np.load(filename01) # load
                sr_01per,rmse_01,mae_01 = datald['v0'],datald['v2'],datald['v3'] 
                
            if not os.path.isfile(filename_m):
                rmse_m, mae_m = np.zeros((nchl)),np.zeros((nchl))
                for i in range(nchl_o): # note: hr_mean is loaded
                    ichl = ivar_hr[i]
                    rmse_m[i] = np.nanmean((sr_mean[i,:,:] - hr_mean[i,:,:]) ** 2)**(0.5)
                    mae_m[i] = np.nanmean(abs(sr_mean[i,:,:] - hr_mean[i,:,:]))
                np.savez(filename_m,v0=sr_mean,v1=hr_mean,v2=rmse_m,v3=mae_m) 
            else: 
                datald = np.load(filename_m) # load
                sr_mean,rmse_m,mae_m = datald['v0'],datald['v2'],datald['v3'] 

            # else: 
            #     datald = np.load(filename99) # load
            #     sr_99per,rmse_99,mae_99 = datald['v0'],datald['v2'],datald['v3']
            #     datald = np.load(filename01) # load
            #     sr_01per,rmse_01,mae_01 = datald['v0'],datald['v2'],datald['v3']
            #     datald = np.load(filename_m) # load
            #     sr_mean,rmse_m,mae_m = datald['v0'],datald['v2'],datald['v3']
            
            # if epoch % opt.sample_epoch == 0:
            for i in range(nchl_o):
                ichl = i
                ichl_v = ivar_hr[i]
                
                clim_chl = clim_m[ichl_v]+clim_99[ichl_v]+clim_01[ichl_v]
                # plot diff in 99per,01per and mean together
                sample  = [
                           hr_re3_mean[ichl,:,:]-hr_mean[ichl,:,:],
                           hr_re2_mean[ichl,:,:]-hr_mean[ichl,:,:],
                           sr_mean[i,:,:]-hr_mean[ichl,:,:],
                           hr_mean[ichl,:,:],
                           hr_re3_99per[ichl,:,:]-hr_99per[ichl,:,:],
                           hr_re2_99per[ichl,:,:]-hr_99per[ichl,:,:],
                           sr_99per[i,:,:]-hr_99per[ichl,:,:],
                           hr_99per[ichl,:,:],
                           hr_re3_01per[ichl,:,:]-hr_01per[ichl,:,:],
                           hr_re2_01per[ichl,:,:]-hr_01per[ichl,:,:],
                           sr_01per[i,:,:]-hr_01per[ichl,:,:],
                           hr_01per[ichl,:,:],
                           ]
                unit = [unit_suv[ichl_v]]*len(sample)
                # title = ['hr_99','sr_99','sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')']
                title = ['nearest_m-hr_m','bilinear_m-hr_m','sr_m-hr_m','hr_m',
                         'nearest_99-hr_99','bilinear_99-hr_99','sr_99-hr_99','hr_99',
                         'nearest_01-hr_01','bilinear_01-hr_01','sr_01-hr_01','hr_01',
                         ]
                figname = out_path+"mean_99_01_c%d_epoch%d_ax0.png" % (ichl_v,epoch)
                txt = ['MAE=%5.3f'%mae_m_re3[ichl]+'\nRMSE=%5.3f'%rmse_m_re3[ichl],
                       'MAE=%5.3f'%mae_m_re2[ichl]+'\nRMSE=%5.3f'%rmse_m_re2[ichl],
                       'MAE=%5.3f'%mae_m[i]+'\nRMSE=%5.3f'%rmse_m[i],'',
                       'MAE=%5.3f'%mae_99_re3[ichl]+'\nRMSE=%5.3f'%rmse_99_re3[ichl],
                       'MAE=%5.3f'%mae_99_re2[ichl]+'\nRMSE=%5.3f'%rmse_99_re2[ichl],
                       'MAE=%5.3f'%mae_99[i]+'\nRMSE=%5.3f'%rmse_99[i],'',
                       'MAE=%5.3f'%mae_01_re3[ichl]+'\nRMSE=%5.3f'%rmse_01_re3[ichl],
                       'MAE=%5.3f'%mae_01_re2[ichl]+'\nRMSE=%5.3f'%rmse_01_re2[ichl],
                       'MAE=%5.3f'%mae_01[i]+'\nRMSE=%5.3f'%rmse_01[i],'',
                       ]
                loc_txt = [0.52,0.40] # location of text
                plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,
                                unit=unit,title=title,nrow=3,axoff=1,capt=capt_all,txt=txt,loc_txt=loc_txt)                


                clim_chl = clim[ichl_v]                
                # sample  = [hr_99per[ichl,:,:],sr_99per[i,:,:],sr_99per[i,:,:]-hr_99per[ichl,:,:]]
                # unit = [unit_suv[ichl]]*len(sample)
                # title = ['hr_99','sr_99','sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')']
                sample  = [hr_re3_99per[ichl,:,:],
                           hr_re2_99per[ichl,:,:],
                           sr_99per[i,:,:],
                           hr_99per[ichl,:,:],
                           hr_re3_99per[ichl,:,:]-hr_99per[ichl,:,:],
                           hr_re2_99per[ichl,:,:]-hr_99per[ichl,:,:],
                           sr_99per[i,:,:]-hr_99per[ichl,:,:],
                           ]
                unit = [unit_suv[ichl]]*len(sample)
                # title = ['hr_99','sr_99','sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')']
                title = ['nearest_99','bilinear_99','sr_99','hr_99',
                         'nearest-hr'+'(%5.3f'%mae_99_re3[ichl]+',%5.3f'%rmse_99_re3[ichl]+')',
                         'bilinear-hr'+'(%5.3f'%mae_99_re2[ichl]+',%5.3f'%rmse_99_re2[ichl]+')',
                         'sr-hr'+'(%5.3f'%mae_99[i]+',%5.3f'%rmse_99[i]+')',]
                figname = out_path+"99th_c%d_epoch%d_ax0.png" % (ichl_v,epoch)
                plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2,axoff=1)                
                # figname = out_path+"99th_c%d_epoch%d_.png" % (ichl,epoch)
                # plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title,nrow=2)

            # plot batch average metrics
            for i in range(nchl):
                ichl = ivar_hr[i]
                # for key, value in metrics_bt.items():
                #     metrics_bt_chl[key] = [value[j][i] for j in range(0,len(value))]
                # for key, value in metrics_re_bt.items():
                #     metrics_re_bt_chl[key] = [value[j][ichl] for j in range(0,len(value))]
                # data_frame = pd.DataFrame.from_dict(metrics_bt_chl|metrics_re_bt_chl, orient='index').transpose()
                # ofname = "srf_%d_re%d_ep%d_c%d" % (opt.up_factor,irep,epoch,ichl) + '_test_metrics.csv'
                # data_frame.to_csv(opath_st_rp + os.sep + ofname, index_label='batch')
                
                ofname = "srf_%d_re%d_ep%d_c%d" % (opt.up_factor,irep,epoch,ichl) + '_test_metrics.csv'
                df = pd.read_csv(opath_st_rp + ofname) # note this is a single channel
                metrics_bt = df.to_dict(orient='list') 
                
                # style = 'seaborn-deep'
                legloc = (0.3,0.40)
                # if epoch % opt.sample_epoch == 0:
                    # plot batch average 
                leg = ['sr','bicubic','bilinear','nearest']
                axlab = [['Batch','rmse(ssh) (m)'],['Batch','rmse(u) (m/s)'],['Batch','rmse(v) (m/s)'],
                         ['Batch','rmse(uw) (m/s)'],['Batch','rmse(vw) (m/s)'],
                         ['Batch','rmse(swh) (m)'],['Batch','rmse(pwp) (s)']]
                var = np.array(metrics_bt['rmse']) #[:,i]
                var_res1 = np.array(metrics_re_bt['rmse_re1'])[:,i]
                var_res2 = np.array(metrics_re_bt['rmse_re2'])[:,i]
                var_res3 = np.array(metrics_re_bt['rmse_re3'])[:,i]
                time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                data_lst = [var,var_res1,var_res2,var_res3]
                figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_rmse.png'
                plot_line_list(time_lst,data_lst,figname=figname,axlab=axlab[ichl],
                                leg=leg,leg_col=2,legloc=legloc,capt='(a)') #,style=style

                axlab = [['Batch','mae(ssh) (m)'],['Batch','mae(u) (m/s)'],['Batch','mae(v) (m/s)'],
                         ['Batch','mae(uw) (m/s)'],['Batch','mae(uw) (m/s)'],
                         ['Batch','mae(swh) (m)'],['Batch','mae(pwp) (s)']]
                var = np.array(metrics_bt['mae']) #[:,i]
                var_res1 = np.array(metrics_re_bt['mae_re1'])[:,i]
                var_res2 = np.array(metrics_re_bt['mae_re2'])[:,i]
                var_res3 = np.array(metrics_re_bt['mae_re3'])[:,i]
                time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                data_lst = [var,var_res1,var_res2,var_res3]
                figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_mae.png'
                plot_line_list(time_lst,data_lst,figname=figname,axlab=axlab[ichl],
                                leg=leg,leg_col=2,legloc=legloc,capt='(b)') #,style=style

                    # axlab = ['Batch','psnr']
                    # var = np.array(metrics_bt['psnr']) # [:,i]
                    # var_res1 = np.array(metrics_re_bt['psnr_re1'])[:,ichl]
                    # var_res2 = np.array(metrics_re_bt['psnr_re2'])[:,ichl]
                    # var_res3 = np.array(metrics_re_bt['psnr_re3'])[:,ichl]
                    # time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                    # data_lst = [var,var_res1,var_res2,var_res3]
                    # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_psnr.png'
                    # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab,leg=leg,leg_col=2)
                    
                    # axlab = ['Batch','ssim']
                    # var = np.array(metrics_bt['ssim']) # [:,i]
                    # var_res1 = np.array(metrics_re_bt['ssim_re1'])[:,ichl]
                    # var_res2 = np.array(metrics_re_bt['ssim_re2'])[:,ichl]
                    # var_res3 = np.array(metrics_re_bt['ssim_re3'])[:,ichl]
                    # time_lst = [np.arange(0,Nbatch_t)] * 4  # repeat n times of the element 
                    # data_lst = [var,var_res1,var_res2,var_res3]
                    # figname = out_path+"c%d_epoch%d" % (ichl,epoch) +'_ssim.png'
                    # plot_sites_cmpn(time_lst,data_lst,figname=figname,axlab=axlab,leg=leg,leg_col=2)

        #     # save metrics epoch average
        #     for key, value in metrics_bt.items():
        #         metrics[key].append(sum(metrics_bt[key])/len(metrics_bt[key])) # / Nbatch_t) # 
        #     metrics['mae_99'].append(mae_99)
        #     metrics['rmse_99'].append(rmse_99)
        #     metrics['ep'].append([epoch]*nchl)
        
        # # output metrics for all epochs to csv
        # for i in range(nchl):
        #     for key, value in metrics.items():
        #         metrics_chl[key] = [value[j][i] for j in range(0,len(value))]
        #     data_frame = pd.DataFrame.from_dict(metrics_chl, orient='index').transpose()
        #     ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_test_metrics.csv'
        #     data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
        
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        # np.save(opath_st + os.sep + ofname, metrics) 
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics_bt.npy'
        # np.save(opath_st + os.sep + ofname, metrics_bt) 
        
        # # output sorted metrics for all epochs to csv
        # ind_sort = [[]]*nchl
        # metrics_chl_sort = {}
        # # list rmse from small to large
        # for i in range(nchl):
        #     var = [metrics['rmse_99'][j][i] for j in range(len(metrics['rmse_99']))]
        #     ind_sort[i] = sorted(range(len(var)), key=lambda k: var[k]) # , reverse=True
        #     for key, value in metrics.items():
        #         metrics_chl_sort[key] = [value[ind_sort[i][j]][i] for j in range(0,len(value))]
        #     data_frame = pd.DataFrame.from_dict(metrics_chl_sort, orient='index').transpose()
        #     ofname = "srf_%d_re%d_c%d_ep%d_%d_mask" % (opt.up_factor,irep,ivar_hr[i]-3,epoc_num[0],epoc_num[-1]) + '_metrics_sort.csv'
        #     data_frame.to_csv(opath_st + os.sep + ofname, index_label='Epoch')
            
        # # output epoc oder for channels
        # ep_sort = [[]]*nchl
        # for i in range(nchl):
        #     ep_sort[i] = [metrics['ep'][ind_sort[i][j]][i] for j in range(len(metrics['ep']))]
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_epoc_sort.csv'
        # np.savetxt(opath_st + ofname, np.array(ep_sort).transpose(), fmt='%d',delimiter=",")
        # # ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")



    


