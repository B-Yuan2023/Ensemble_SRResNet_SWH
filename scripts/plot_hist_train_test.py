#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:05:35 2024
plot distribution of training and testing data
@author: g260218
"""

import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import myDataset
from funs_prepost import nc_load_vars,var_denormalize,plot_distri

import torch
import pandas as pd

import sys
import importlib
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= 'par55e'          #'par04' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
# from mod_para import * 

import matplotlib.pyplot as plt

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
    
    if isinstance(rtra, (int, float)): # if only input one number, no validation
        rtra = [rtra,0]
    
    # create nested list of files and indt
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
        
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
    # rep = list(range(0,nrep))
    # rep = [0]
    # # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    # print(f'parname: {mod_name}')
    # print('--------------------------------')
    
    # epoc_num = [50,100]
    # # epoc_num = np.arange(40,opt.N_epochs+1)
    # key_ep_sort = 1 # to use epoc here or load sorted epoc no. 
    # nepoc = 2 # no. of sorted epochs for analysis

    nchl = nchl_o
    
    hr_shape = (opt.hr_height, opt.hr_width)

    train_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='train',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)

    test_set = myDataset(files_lr,files_hr,indt_lr,indt_hr,hr_shape, opt.up_factor,
                          mode='test',rtra = rtra,var_lr=var_lr,var_hr=var_hr,
                          varm_lr=varm_lr,varm_hr=varm_hr,ll_lr=ll_lr,ll_hr=ll_hr,kintp=kintp)

    data_train = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )
    data_test = DataLoader(
        test_set,
        batch_size=opt.batch_size, 
        num_workers=opt.n_cpu,
    )        
    
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
    # # get logitude and latitude of data 
    nc_f = test_set.files_hr[0]
    lon = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f[0],var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]
    # nc_f = test_set.files_hr[0]
    # lon = nc_load_all(nc_f,0)[1]
    # lat = nc_load_all(nc_f,0)[2]
    # mask = nc_load_all(nc_f,0)[10] # original data
    
    in_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+'/' # hr_all, etc
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)',]
    
    opath_st = path_par+'stat' + suf +'/'
    os.makedirs(opath_st, exist_ok=True)
        
    # get all training data 
    lr_all_train = []
    hr_all_train = []
    for i, dat in enumerate(data_train):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_hr)
        
        # get mask for time step
        mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
        mask_lr = lr_norm0==lr_norm0 # initialize the boolean array with the shape of lr_norm
        for ib in range(opt.batch_size):  # use mask for each sample/time
            it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
            if it>=len(train_set):  # for case the last batch has samples less than batch_size
                break
            for ichl in range(nchl):
                nc_f = train_set.files_hr[it][ichl]
                indt = train_set.indt_hr[it][ichl]  # the time index in a ncfile
                mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                nc_f = train_set.files_lr[it][ichl]
                indt = train_set.indt_lr[it][ichl]  # the time index in a ncfile
                mask_lr[ib,ichl,:,:] = nc_load_vars(nc_f,var_lr[ichl],[indt],ll_lr[0],ll_lr[1])[4] # mask at 1 time in a batch
        
        hr_norm0[mask] = np.nan
        hr_all_train.append(hr_norm0)
        
        lr_norm0[mask_lr] = np.nan
        lr_all_train.append(lr_norm0)

    hr_all_train = np.concatenate(hr_all_train, axis=0)
    lr_all_train = np.concatenate(lr_all_train, axis=0)
    
    # get all test data 
    lr_all = []
    hr_all = []
    for i, dat in enumerate(data_test):                
        dat_lr = Variable(dat["lr"].type(Tensor))
        dat_hr = Variable(dat["hr"].type(Tensor))
        hr_norm0 = var_denormalize(dat_hr.detach().cpu().numpy(),varm_hr)
        lr_norm0 = var_denormalize(dat_lr.detach().cpu().numpy(),varm_hr) 
        
        mask = hr_norm0==hr_norm0 # initialize the boolean array with the shape of hr_norm0
        mask_lr = lr_norm0==lr_norm0 # initialize the boolean array with the shape of lr_norm
        for ib in range(opt.batch_size):  # use mask for each sample/time
            it = i*opt.batch_size + ib  # this it is no. of time steps in dataset, not true time
            if it>=len(test_set):  # for case the last batch has samples less than batch_size
                break
            for ichl in range(nchl):
                nc_f = test_set.files_hr[it][ichl]
                indt = test_set.indt_hr[it][ichl]  # the time index in a ncfile
                mask[ib,ichl,:,:] = nc_load_vars(nc_f,var_hr[ichl],[indt],ll_hr[0],ll_hr[1])[4] # mask at 1 time in a batch
                nc_f = test_set.files_lr[it][ichl]
                indt = test_set.indt_lr[it][ichl]  # the time index in a ncfile
                mask_lr[ib,ichl,:,:] = nc_load_vars(nc_f,var_lr[ichl],[indt],ll_lr[0],ll_lr[1])[4] # mask at 1 time in a batch
        
        
        hr_norm0[mask] = np.nan
        hr_all.append(hr_norm0)
        
        lr_norm0[mask_lr] = np.nan
        lr_all.append(lr_norm0)

    hr_all = np.concatenate(hr_all, axis=0)
    lr_all = np.concatenate(lr_all, axis=0)
    
    index = np.array([[104, 22],[76, 6],[83, 20],[88, 239]])
    nsta = len(index)  # number of stations 
    sta_user = ['P'+str(ip+1) for ip in range(nsta)]
    ave_sta_train,ave_sta_test = np.zeros((nchl,nsta)),np.zeros((nchl,nsta))
    ave_sta_all = np.zeros((nchl,nsta))
    
    for i in range(nchl):
        mean_hr_sta = np.zeros(nsta)
        for ip in range(nsta):
            # train
            hr_sta_train = hr_all_train[:,i,index[ip,0],index[ip,1]]
            ave_sta_train[i][ip] = np.nanmean(hr_sta_train)
            
            # test
            hr_sta = hr_all[:,i,index[ip,0],index[ip,1]]
            ave_sta_test[i][ip] = np.nanmean(hr_sta)
            
            hr_sta_all = np.concatenate((hr_sta_train,hr_sta), axis=0)
            ave_sta_all[i][ip] = np.nanmean(hr_sta_all)
        ichl = ivar_hr[i]
        ofname = 'c%d'%ichl+'_ave_sta'+'.csv'
        combined_ave_sta= np.concatenate((ave_sta_all,ave_sta_train,ave_sta_test))
        np.savetxt(opath_st + ofname, combined_ave_sta,fmt='%f,') # ,delimiter=","
    
    # show distribution of training/testing dataset
    for i in range(nchl):
        ichl = ivar_hr[i]
        unit_var = unit_suv[ichl]
    
        # plot distribution of reconstructed vs target, all data, histogram
        axlab = (unit_var,'Frequency','')
        leg = ['hr_train','lr_train','hr_test','lr_test'] #,'nearest'
        var1 = hr_all_train[:,i,:,:].flatten()
        var2 = lr_all_train[:,i,:,:].flatten()
        var3 = hr_all[:,i,:,:].flatten()
        var4 = lr_all[:,i,:,:].flatten()
        
        max_hr_train,min_hr_train = np.nanmax(var1), np.nanmin(var1)
        max_lr_train,min_lr_train = np.nanmax(var2), np.nanmin(var2)
        max_hr_test,min_hr_test = np.nanmax(var3), np.nanmin(var3)
        max_lr_test,min_lr_test = np.nanmax(var4), np.nanmin(var4)
        
        ave_hr_train,ave_lr_train = np.nanmean(var1), np.nanmean(var2)
        ave_hr_test,ave_lr_test = np.nanmean(var3), np.nanmean(var4)
        
        ofname = 'c%d'%ichl+'_maxmin_ave'+'.csv'
        combined_ind= np.array([[max_hr_train,min_hr_train,ave_hr_train],
                                [max_hr_test,min_hr_test,ave_hr_test],
                                [max_lr_train,min_lr_train,ave_lr_train],
                                [max_lr_test,min_lr_test,ave_lr_test]])
        np.savetxt(opath_st + ofname, combined_ind,fmt='%f,') # ,delimiter=","
        
        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
               var3[~np.isnan(var3)],var4[~np.isnan(var4)],
               ]
        figname = opath_st+"c%d" % (ichl) +'_dist_train_test'+'.png'
        plot_distri(var,figname,bins=20,axlab=axlab,leg=leg,
                       figsize=(10, 5), fontsize=16,capt='')
    
    