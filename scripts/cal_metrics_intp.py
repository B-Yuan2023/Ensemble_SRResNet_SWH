#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:39:22 2024
calculate metrics for direct interpolation 
@author: g260218
"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import myDataset
from funs_prepost import (nc_load_vars,var_denormalize,plt_pcolor_list,
                          interpolate_tensor,interp4d_tensor_nearest_neighbor)
import torch
from pytorch_msssim import ssim as ssim_torch
from math import log10
import pandas as pd

import sys
import importlib
mod_name= 'par55e'           # use 1 output channel
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1
# ich = 5  # for plot channel limit: 0 ssh,1-2 uv, 3-4 uw vw, 5 swh, 6 pwp

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
    ivar_lr = mod_para.ivar_lr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    tuse = mod_para.t_hr

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
        
    # nrep = mod_para.nrep
    # nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    # epoc_num = [50,100]
    # epoc_num = np.arange(40,opt.N_epochs+1)

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)
    
    # opath_st = 'statistics' + suf +'/'
    # if not os.path.exists(opath_st):
    #     os.makedirs(opath_st)
        
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
    
    rtra = rtra[0]
    out_path = path_par+'results_test/'+'S%d_mk%d'%(opt.up_factor,kmask)+'/'+ var_hr[0]+'/'
    os.makedirs(out_path, exist_ok=True)

    # interpolaton only if input and output channels are the same 
    if ivar_hr==ivar_lr:
        filename99 = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        filename01 = out_path + 'hr_01per_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        filename_m = out_path + 'hr_mean_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
        filename_t = out_path + 'hr_tave_interp'+'_train%4.2f'%(rtra)+'.npz' # file for temporal rmse/mae of points

        # for direct interpolation
        metrics_re = {'mse_re1': [], 'mae_re1': [], 'rmse_re1': [], 'ssim_re1': [], 'psnr_re1': [],
                      'mse_re2': [], 'mae_re2': [], 'rmse_re2': [], 'ssim_re2': [], 'psnr_re2': [],
                      'mse_re3': [], 'mae_re3': [], 'rmse_re3': [], 'ssim_re3': [], 'psnr_re3': [],
                      }
        metrics_re_bt = {}
        metrics_re_bt_chl = {}
        for key in metrics_re:  # loop only for keys
            metrics_re_bt[key] = []
            metrics_re_bt_chl[key] = []
    
        hr_all = []
        hr_restore1_all = []
        hr_restore2_all = []
        hr_restore3_all = []
        for i, dat in enumerate(data_test):
            dat_lr = Variable(dat["lr"].type(Tensor))
            dat_hr = Variable(dat["hr"].type(Tensor))
            hr_norm0 = var_denormalize(dat_hr.cpu().numpy(),varm_hr)
            lr_norm0 = var_denormalize(dat_lr.cpu().numpy(),varm_lr)
            
            # get mask for time step
            mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
            mask_lr = lr_norm0==lr_norm0 # initialize the boolean array 
            for ib in range(opt.batch_size):  # use mask for each sample/time
                it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                if it>=len(test_set):
                    break
                # nc_f = test_set.files_hr[it]
                # indt = indt_hr[it]  # the time index in a ncfile
                for ichl in range(nchl):
                    nc_f = test_set.files_hr[it][ichl]
                    indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                    mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                    
                    nc_f = test_set.files_lr[it][ichl]
                    indt = test_set.indt_lr[it][ichl]  # the time index in a ncfile
                    mask_lr[ib,ichl,:,:] = nc_load_vars(nc_f,var_lr[ichl],[indt],ll_lr[0],ll_lr[1])[4] 
                    mask_lr[ib,ichl,:,:] = np.flipud(mask_lr[ib,ichl,:,:]) # dimensionless data flipped
                # dat_hr[:,:,mask_ud.copy()] = np.nan
    
            # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
            # hr_restore1 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
            hr_restore2 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
            # hr_restore3 = torch.nn.functional.interpolate(dat_lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
    
            # using other interpolation method instead of torch with extrapolation
            # RBFInterpolator kernal: {'quintic', 'linear', 'inverse_quadratic', 
            # 'cubic', 'thin_plate_spline'(default), 'multiquadric', 'gaussian', 'inverse_multiquadric'}
            dat_lr[torch.tensor(mask_lr)] = np.nan
            hr_restore1 = interpolate_tensor(dat_lr, opt.up_factor, kintp=2, method='linear')

            # use custum nearest neighbor interpolation
            hr_restore3 = interp4d_tensor_nearest_neighbor(dat_lr, opt.up_factor)

            hr_restore1_norm0  = var_denormalize(hr_restore1.cpu().numpy(),varm_hr)
            hr_restore2_norm0  = var_denormalize(hr_restore2.cpu().numpy(),varm_hr)
            hr_restore3_norm0  = var_denormalize(hr_restore3.cpu().numpy(),varm_hr)
    
            if kmask == 1: 
                hr_norm0[mask] = np.nan            
                hr_restore1_norm0[mask] = np.nan
                hr_restore2_norm0[mask] = np.nan
                hr_restore3_norm0[mask] = np.nan
                # hr_norm0[:,:,mask] = np.nan            
                # hr_restore1_norm0[:,:,mask] = np.nan
                # hr_restore2_norm0[:,:,mask] = np.nan
                # hr_restore3_norm0[:,:,mask] = np.nan
            hr_all.append(hr_norm0)
            hr_restore1_all.append(hr_restore1_norm0)
            hr_restore2_all.append(hr_restore2_norm0)
            hr_restore3_all.append(hr_restore3_norm0)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore1,dat_hr,hr_restore1_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re1'].append(mse)
            metrics_re_bt['mae_re1'].append(mae)
            metrics_re_bt['rmse_re1'].append(rmse)
            metrics_re_bt['ssim_re1'].append(ssim)
            metrics_re_bt['psnr_re1'].append(psnr)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore2,dat_hr,hr_restore2_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re2'].append(mse)
            metrics_re_bt['mae_re2'].append(mae)
            metrics_re_bt['rmse_re2'].append(rmse)
            metrics_re_bt['ssim_re2'].append(ssim)
            metrics_re_bt['psnr_re2'].append(psnr)
            
            rmse, mae, mse, ssim, psnr = cal_metrics(hr_restore3,dat_hr,hr_restore3_norm0,hr_norm0) # ,mask
            metrics_re_bt['mse_re3'].append(mse)
            metrics_re_bt['mae_re3'].append(mae)
            metrics_re_bt['rmse_re3'].append(rmse)
            metrics_re_bt['ssim_re3'].append(ssim)
            metrics_re_bt['psnr_re3'].append(psnr)
        for key, value in metrics_re_bt.items():
            metrics_re[key] = sum(metrics_re_bt[key])/len(metrics_re_bt[key]) # / Nbatch_t
        ofname = 'metrics_interp_gb'+'_train%4.2f'%(rtra)+'.npy'
        np.save(out_path + os.sep + ofname, metrics_re) 
    
        # hr_all = np.array(hr_all).reshape(-1,nchl,hr_shape[0],hr_shape[1]) # [Nt,c,H,W]
        # hr_restore1_all = np.array(hr_restore1_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        # hr_restore2_all = np.array(hr_restore2_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        # hr_restore3_all = np.array(hr_restore3_all).reshape(-1,nchl,hr_shape[0],hr_shape[1])
        hr_all = np.concatenate(hr_all, axis=0) # [Nt,c,H,W]
        hr_restore1_all = np.concatenate(hr_restore1_all, axis=0)
        hr_restore2_all = np.concatenate(hr_restore2_all, axis=0)
        hr_restore3_all = np.concatenate(hr_restore3_all, axis=0)    
        
        # nchl_o, save for all time steps. 
        for k in range(nchl_o): 
            filename = out_path + 'c%d_'%(ivar_hr[k])+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
            if not os.path.isfile(filename): 
                var_all  = hr_all[:,k,:,:]
                np.savez(filename,v0=var_all,lat=lat,lon=lon,tuse=tuse) 
            filename = out_path + 'c%d_'%(ivar_hr[k])+'hr_restore3_all'+'_train%4.2f'%(rtra)+'.npz'
            if not os.path.isfile(filename):
                var_all  = hr_restore3_all[:,k,:,:]
                np.savez(filename,v0=var_all)
            filename = out_path + 'c%d_'%(ivar_hr[k])+'hr_restore2_all'+'_train%4.2f'%(rtra)+'.npz'
            if not os.path.isfile(filename):
                var_all  = hr_restore2_all[:,k,:,:]
                np.savez(filename,v0=var_all)
            filename = out_path + 'c%d_'%(ivar_hr[k])+'hr_restore1_all'+'_train%4.2f'%(rtra)+'.npz'
            if not os.path.isfile(filename):
                var_all  = hr_restore1_all[:,k,:,:]
                np.savez(filename,v0=var_all)             
            # filename = out_path + 'c%s_'%(var_hr[k])+'hr_restore_all'+'_train%4.2f'%(rtra)+'.npz'
            # if not os.path.isfile(filename):
            #     var_all  = [hr_restore1_all[:,k,:,:],hr_restore2_all[:,k,:,:],hr_restore3_all[:,k,:,:]]
            #     np.savez(filename,v0=var_all)
        
        # save 99 percentile 
        if not os.path.isfile(filename99): 
            hr_99per = np.nanpercentile(hr_all, 99, axis = (0,))
            hr_re1_99per = np.nanpercentile(hr_restore1_all, 99, axis = (0,))
            hr_re2_99per = np.nanpercentile(hr_restore2_all, 99, axis = (0,))
            hr_re3_99per = np.nanpercentile(hr_restore3_all, 99, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename99,v0=hr_99per,v1=hr_re1_99per,v2=hr_re2_99per,v3=hr_re3_99per,lat=lat,lon=lon) 
        else:
            datald = np.load(filename99) # load
            hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
        if not os.path.isfile(filename01): 
            hr_01per = np.nanpercentile(hr_all, 1, axis = (0,))
            hr_re1_01per = np.nanpercentile(hr_restore1_all, 1, axis = (0,))
            hr_re2_01per = np.nanpercentile(hr_restore2_all, 1, axis = (0,))
            hr_re3_01per = np.nanpercentile(hr_restore3_all, 1, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename01,v0=hr_01per,v1=hr_re1_01per,v2=hr_re2_01per,v3=hr_re3_01per,lat=lat,lon=lon) 
        else:
            datald = np.load(filename01) # load
            hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
        if not os.path.isfile(filename_m): 
            hr_mean = np.nanmean(hr_all, axis = (0,))        
            hr_re1_mean = np.nanmean(hr_restore1_all, axis = (0,))
            hr_re2_mean = np.nanmean(hr_restore2_all, axis = (0,))
            hr_re3_mean = np.nanmean(hr_restore3_all, axis = (0,))
            # save and Load 99per
            # filename = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz'
            np.savez(filename_m,v0=hr_mean,v1=hr_re1_mean,v2=hr_re2_mean,v3=hr_re3_mean,lat=lat,lon=lon)             
        else:
            datald = np.load(filename_m) # load
            hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        
        if not os.path.isfile(filename_t):     
            hr_re1_rmse = np.nanmean((hr_restore1_all - hr_all) ** 2,axis=(0))**(0.5) # temporal rmse per point per channel (C,H,W)
            hr_re1_mae = np.nanmean(abs(hr_restore1_all - hr_all),axis=(0))
            hr_re2_rmse = np.nanmean((hr_restore2_all - hr_all) ** 2,axis=(0))**(0.5) # temporal rmse per point per channel (C,H,W)
            hr_re2_mae = np.nanmean(abs(hr_restore2_all - hr_all),axis=(0))
            hr_re3_rmse = np.nanmean((hr_restore3_all - hr_all) ** 2,axis=(0))**(0.5) # temporal rmse per point per channel (C,H,W)
            hr_re3_mae = np.nanmean(abs(hr_restore3_all - hr_all),axis=(0))
            np.savez(filename_t,v0=hr_re1_rmse,v1=hr_re1_mae,v2=hr_re2_rmse,v3=hr_re2_mae,v4=hr_re3_rmse,v5=hr_re3_mae,lat=lat,lon=lon)             
        else:
            datald = np.load(filename_t) # load
            hr_re1_rmse,hr_re1_mae,hr_re2_rmse,hr_re2_mae,hr_re3_rmse,hr_re3_mae = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        
        # save and Load dict
        file_metric_intp = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
        if not os.path.isfile(file_metric_intp):
            np.save(file_metric_intp, metrics_re_bt) 
        # metrics_re_bt = np.load(file_metric_intp,allow_pickle='TRUE').item()
        # print(metrics_re_bt['rmse_re3']) # displays 
        # else:
        #     datald = np.load(filename99) # load
        #     hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        #     # hr_re1_99per = datald['v1']
        #     # hr_re2_99per = datald['v2']
        #     # hr_re3_99per = datald['v3']
        #     file_metric_intp = out_path + 'metrics_interp'+'_train%4.2f'%(rtra)+'.npy'
        #     metrics_re_bt = np.load(file_metric_intp,allow_pickle='TRUE').item()
        #     datald = np.load(filename01) # load
        #     hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        #     datald = np.load(filename_m) # load
        #     hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        
        # save global metrics rmse/mae
        filename = out_path + 'hr_gb_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # global avraged rmse/mae of points
        if not os.path.isfile(filename): 
            rmse_re1 = np.nanmean((hr_restore1_all - hr_all) ** 2,axis=(0,2,3))**(0.5) # temporal rmse per point per channel (C,H,W)
            mae_re1 = np.nanmean(abs(hr_restore1_all - hr_all),axis=(0,2,3))
            rmse_re2 = np.nanmean((hr_restore2_all - hr_all) ** 2,axis=(0,2,3))**(0.5) # temporal rmse per point per channel (C,H,W)
            mae_re2 = np.nanmean(abs(hr_restore2_all - hr_all),axis=(0,2,3))
            rmse_re3 = np.nanmean((hr_restore3_all - hr_all) ** 2,axis=(0,2,3))**(0.5) # temporal rmse per point per channel (C,H,W)
            mae_re3 = np.nanmean(abs(hr_restore3_all - hr_all),axis=(0,2,3))
            np.savez(filename,v0=rmse_re1,v1=rmse_re2,v2=rmse_re3,
                     v3=mae_re1,v4=mae_re2,v5=mae_re3) 
        else:
            datald = np.load(filename) # load
            rmse_re1,rmse_re2,rmse_re3,mae_re1,mae_re2,mae_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
            

        filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        if not os.path.isfile(filename99m): 
            rmse_99_re1 = np.nanmean((hr_re1_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
            rmse_99_re2 = np.nanmean((hr_re2_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
            rmse_99_re3 = np.nanmean((hr_re3_99per - hr_99per) ** 2,axis=(1,2))**(0.5)
            mae_99_re1 = np.nanmean(abs(hr_re1_99per - hr_99per),axis=(1,2))
            mae_99_re2 = np.nanmean(abs(hr_re2_99per - hr_99per),axis=(1,2))
            mae_99_re3 = np.nanmean(abs(hr_re3_99per - hr_99per),axis=(1,2))
            np.savez(filename99m,v0=rmse_99_re1,v1=rmse_99_re2,v2=rmse_99_re3,
                     v3=mae_99_re1,v4=mae_99_re2,v5=mae_99_re3) 
        else:
            datald = np.load(filename99m) # load
            rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
            
        
        filename01m = out_path + 'hr_01per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        if not os.path.isfile(filename01m): 
            rmse_01_re1 = np.nanmean((hr_re1_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
            rmse_01_re2 = np.nanmean((hr_re2_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
            rmse_01_re3 = np.nanmean((hr_re3_01per - hr_01per) ** 2,axis=(1,2))**(0.5)
            mae_01_re1 = np.nanmean(abs(hr_re1_01per - hr_01per),axis=(1,2))
            mae_01_re2 = np.nanmean(abs(hr_re2_01per - hr_01per),axis=(1,2))
            mae_01_re3 = np.nanmean(abs(hr_re3_01per - hr_01per),axis=(1,2))
            np.savez(filename01m,v0=rmse_01_re1,v1=rmse_01_re2,v2=rmse_01_re3,
                     v3=mae_01_re1,v4=mae_01_re2,v5=mae_01_re3) 
        else:
            datald = np.load(filename01m) # load
            rmse_01_re1,rmse_01_re2,rmse_01_re3,mae_01_re1,mae_01_re2,mae_01_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
            
    
        filename_mm = out_path + 'hr_mean_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
        if not os.path.isfile(filename_mm): 
            rmse_m_re1 = np.nanmean((hr_re1_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
            rmse_m_re2 = np.nanmean((hr_re2_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
            rmse_m_re3 = np.nanmean((hr_re3_mean - hr_mean) ** 2,axis=(1,2))**(0.5)
            mae_m_re1 = np.nanmean(abs(hr_re1_mean - hr_mean),axis=(1,2))
            mae_m_re2 = np.nanmean(abs(hr_re2_mean - hr_mean),axis=(1,2))
            mae_m_re3 = np.nanmean(abs(hr_re3_mean - hr_mean),axis=(1,2))
            np.savez(filename_mm,v0=rmse_m_re1,v1=rmse_m_re2,v2=rmse_m_re3,
                     v3=mae_m_re1,v4=mae_m_re2,v5=mae_m_re3) 
        else:
            datald = np.load(filename_mm) # load
            rmse_m_re1,rmse_m_re2,rmse_m_re3,mae_m_re1,mae_m_re2,mae_m_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
    
        filename_tm = out_path + 'hr_tave_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # global avraged rmse/mae of points
        if not os.path.isfile(filename_tm): 
            rmse_t_re1 = np.nanmean(hr_re1_rmse,axis=(1,2))
            rmse_t_re2 = np.nanmean(hr_re2_rmse,axis=(1,2))
            rmse_t_re3 = np.nanmean(hr_re3_rmse,axis=(1,2))
            mae_t_re1 = np.nanmean(hr_re1_mae,axis=(1,2))
            mae_t_re2 = np.nanmean(hr_re2_mae,axis=(1,2))
            mae_t_re3 = np.nanmean(hr_re3_mae,axis=(1,2))
            np.savez(filename_tm,v0=rmse_t_re1,v1=rmse_t_re2,v2=rmse_t_re3,
                     v3=mae_t_re1,v4=mae_t_re2,v5=mae_t_re3) 
        else:
            datald = np.load(filename_tm) # load
            rmse_t_re1,rmse_t_re2,rmse_t_re3,mae_t_re1,mae_t_re2,mae_t_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
            
        
        # save global metrics to csv 
        for i in range(nchl_o):
            ichl = ivar_hr[i]
            filename = out_path+'c%d_mean_99_01_interp'%ichl+'_train%4.2f'%(rtra)+'.csv'    
            metrics_mod = {'errors':['mae','rmse','mae_99','rmse_99','mae_01','rmse_01','mae_m','rmse_m','mae_t','rmse_t'],
                               'Bicubic':[mae_re1[i],rmse_re1[i],mae_99_re1[i],rmse_99_re1[i],mae_01_re1[i],rmse_01_re1[i],
                                          mae_m_re1[i],rmse_m_re1[i],mae_t_re1[i],rmse_t_re1[i]],
                               'Bilnear':[mae_re2[i],rmse_re2[i],mae_99_re2[i],rmse_99_re2[i],mae_01_re2[i],rmse_01_re2[i],
                                          mae_m_re2[i],rmse_m_re2[i],mae_t_re2[i],rmse_t_re2[i]],
                               'Nearest':[mae_re3[i],rmse_re3[i],mae_99_re3[i],rmse_99_re3[i],mae_01_re3[i],rmse_01_re3[i],
                                          mae_m_re3[i],rmse_m_re3[i],mae_t_re3[i],rmse_t_re3[i]]
                               }
            df = pd.DataFrame(metrics_mod)
            df.to_csv(filename, index=False)
        
        txtname = out_path+'mean_99_01_interp.txt'
        if not os.path.isfile(txtname):
            outfile = open(txtname, 'w')
            outfile.write('# rmse_m_re1, rmse_m_re2,rmse_m_re3\n')
            np.savetxt(outfile, np.vstack((rmse_m_re1,rmse_m_re2,rmse_m_re3)), fmt='%-7.4f,')
            outfile.write('# mae_m_re1, mae_m_re2,mae_m_re3\n')
            np.savetxt(outfile, np.vstack((mae_m_re1,mae_m_re2,mae_m_re3)), fmt='%-7.4f,')
            outfile.write('# max hr_mean, hr_re1_mean, hr_re2_mean, hr_re3_mean\n')
            np.savetxt(outfile, np.vstack((np.nanmax(hr_mean,axis=(1,2)),
                                           np.nanmax(hr_re1_mean,axis=(1,2)),
                                           np.nanmax(hr_re2_mean,axis=(1,2)),
                                           np.nanmax(hr_re3_mean,axis=(1,2)))), fmt='%-7.4f,')
            outfile.write('# min hr_mean, hr_re1_mean, hr_re2_mean, hr_re3_mean\n')
            np.savetxt(outfile, np.vstack((np.nanmin(hr_mean,axis=(1,2)),
                                           np.nanmin(hr_re1_mean,axis=(1,2)),
                                           np.nanmin(hr_re2_mean,axis=(1,2)),
                                           np.nanmin(hr_re3_mean,axis=(1,2)))), fmt='%-7.4f,') 
            outfile.write('# rmse_99_re1, rmse_99_re2,rmse_99_re3\n')
            np.savetxt(outfile, np.vstack((rmse_99_re1,rmse_99_re2,rmse_99_re3)), fmt='%-7.4f,')
            outfile.write('# mae_99_re1, mae_99_re2,mae_99_re3\n')
            np.savetxt(outfile, np.vstack((mae_99_re1,mae_99_re2,mae_99_re3)), fmt='%-7.4f,')
            outfile.write('# max hr_99per, hr_re1_99per, hr_re2_99per, hr_re3_99per\n')
            np.savetxt(outfile, np.vstack((np.nanmax(hr_99per,axis=(1,2)),
                                           np.nanmax(hr_re1_99per,axis=(1,2)),
                                           np.nanmax(hr_re2_99per,axis=(1,2)),
                                           np.nanmax(hr_re3_99per,axis=(1,2)))), fmt='%-7.4f,')
            outfile.write('# min hr_99per, hr_re1_99per, hr_re2_99per, hr_re3_99per\n')
            np.savetxt(outfile, np.vstack((np.nanmin(hr_99per,axis=(1,2)),
                                           np.nanmin(hr_re1_99per,axis=(1,2)),
                                           np.nanmin(hr_re2_99per,axis=(1,2)),
                                           np.nanmin(hr_re3_99per,axis=(1,2)))), fmt='%-7.4f,') 
            outfile.write('# rmse_01_re1, rmse_01_re2,rmse_01_re3\n')
            np.savetxt(outfile, np.vstack((rmse_01_re1,rmse_01_re2,rmse_01_re3)), fmt='%-7.4f,')
            outfile.write('# mae_01_re1, mae_01_re2,mae_01_re3\n')
            np.savetxt(outfile, np.vstack((mae_01_re1,mae_01_re2,mae_01_re3)), fmt='%-7.4f,')
            outfile.write('# max hr_01per, hr_re1_01per, hr_re2_01per, hr_re3_01per\n')
            np.savetxt(outfile, np.vstack((np.nanmax(hr_01per,axis=(1,2)),
                                           np.nanmax(hr_re1_01per,axis=(1,2)),
                                           np.nanmax(hr_re2_01per,axis=(1,2)),
                                           np.nanmax(hr_re3_01per,axis=(1,2)))), fmt='%-7.4f,')
            outfile.write('# min hr_01per, hr_re1_01per, hr_re2_01per, hr_re3_01per\n')
            np.savetxt(outfile, np.vstack((np.nanmin(hr_01per,axis=(1,2)),
                                           np.nanmin(hr_re1_01per,axis=(1,2)),
                                           np.nanmin(hr_re2_01per,axis=(1,2)),
                                           np.nanmin(hr_re3_01per,axis=(1,2)))), fmt='%-7.4f,') 
            outfile.close()
            
        clim = [[[1.3,3.3],[1.3,3.3],[-0.2,0.2]],  # ssh
                [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # u
                [[0.2,1.8],[0.2,1.8],[-0.3,0.3]],  # v
                [[12,15],[12,15],[-1.0,1.0]],  # uw
                [[12,15],[12,15],[-1.0,1.0]],  # vw
                [[2.0,5.0],[2.0,5.0],[-0.5,0.5]],  # swh
                [[5.0,15],[5.0,15],[-2.0,2.0]],  # pwp
                ]
        unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)']
    
        for k in range(nchl_o):
            ich = ivar_hr[k]
            clim_chl = clim[ich]
            sample  = [hr_99per[k,:,:],hr_re1_99per[k,:,:],hr_re1_99per[k,:,:]-hr_99per[k,:,:]]
            unit = [unit_suv[ich]]*len(sample)
            title = ['hr_99','bicubic_99','interp-hr' +'(%5.3f'%mae_99_re1[k]+',%5.3f'%rmse_99_re1[k]+')']
            figname = out_path+"99th_c%d_int1.png" % (ivar_hr[k])
            plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)
    
            sample  = [hr_99per[k,:,:],hr_re2_99per[k,:,:],hr_re2_99per[k,:,:]-hr_99per[k,:,:]]
            title = ['hr_99','bilinear_99','interp-hr'+'(%5.3f'%mae_99_re2[k]+',%5.3f'%rmse_99_re2[k]+')']
            figname = out_path+"99th_c%d_int2.png" % (ivar_hr[k])
            plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)
            
            sample  = [hr_99per[k,:,:],hr_re3_99per[k,:,:],hr_re3_99per[k,:,:]-hr_99per[k,:,:]]
            title = ['hr_99','nearest_99','interp-hr'+'(%5.3f'%mae_99_re3[k]+',%5.3f'%rmse_99_re3[k]+')']
            figname = out_path+"99th_c%d_int3.png" % (ivar_hr[k])
            plt_pcolor_list(lon,lat,sample,figname,cmap = 'coolwarm',clim=clim_chl,unit=unit,title=title)
    else:
    # only estimate 99/01 percentile, mean for hr 
        filename99 = out_path + 'hr_99per'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        filename01 = out_path + 'hr_01per'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        filename_m = out_path + 'hr_mean'+'_train%4.2f'%(rtra)+'.npz' # file for mean

        hr_all = []
        for i, dat in enumerate(data_test):
            dat_lr = Variable(dat["lr"].type(Tensor))
            dat_hr = Variable(dat["hr"].type(Tensor))
            hr_norm0 = var_denormalize(dat_hr.cpu().numpy(),varm_hr)
            
            # get mask for time step
            mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
            for ib in range(opt.batch_size):  # use mask for each sample/time
                it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
                if it>=len(test_set):
                    break
                # nc_f = test_set.files_hr[it]
                # indt = indt_hr[it]  # the time index in a ncfile
                for ichl in range(nchl):
                    nc_f = test_set.files_hr[it][ichl]
                    indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                    mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                
            if kmask == 1: 
                hr_norm0[mask] = np.nan            
            hr_all.append(hr_norm0)
            
        hr_all = np.concatenate(hr_all, axis=0) # [Nt,c,H,W]
        
        # nchl_o, save for all time steps. 
        for k in range(nchl_o): 
            filename = out_path + 'c%d_'%(ivar_hr[k])+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
            if not os.path.isfile(filename): 
                var_all  = hr_all[:,k,:,:]
                np.savez(filename,v0=var_all,lat=lat,lon=lon,tuse=tuse) 
        
        # save 99 percentile 
        if not os.path.isfile(filename99): 
            hr_99per = np.nanpercentile(hr_all, 99, axis = (0,))
            np.savez(filename99,v0=hr_99per,lat=lat,lon=lon) 
        else:
            datald = np.load(filename99) # load
            hr_99per,lat,lon = datald['v0'],datald['lat'],datald['lon']
    
        if not os.path.isfile(filename01): 
            hr_01per = np.nanpercentile(hr_all, 1, axis = (0,))
            np.savez(filename01,v0=hr_01per,lat=lat,lon=lon) 
        else:
            datald = np.load(filename01) # load
            hr_01per,lat,lon = datald['v0'],datald['lat'],datald['lon']
    
        if not os.path.isfile(filename_m): 
            hr_mean = np.nanmean(hr_all, axis = (0,))        
            np.savez(filename_m,v0=hr_mean,lat=lat,lon=lon)             
        else:
            datald = np.load(filename_m) # load
            hr_mean,lat,lon = datald['v0'],datald['lat'],datald['lon']
