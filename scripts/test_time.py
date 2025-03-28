#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 27, 2024
test time 
@author: g260218
"""

import os
import numpy as np
import time

import torch

from models import GeneratorResNet
from mod_srres import SRRes,SRResA
from funs_prepost import (make_list_file_t,nc_load_vars,nc_normalize_vars,
                          var_denormalize,plot_line_list,plt_pcolorbar_list)

from datetime import datetime, timedelta # , date
from funs_sites import select_sta

import sys
import importlib
mod_name= 'par55e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

if __name__ == '__main__':
    
    start = time.time() 
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    suf0 = mod_para.suf0
    dir_lr = opt.dir_lr
    dir_hr = opt.dir_hr
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
    
    if hasattr(mod_para, 'rep'):  # if input has list rep
        rep = mod_para.rep
    else:
        nrep = mod_para.nrep
        rep = list(range(0,nrep))
    
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
    
    # nrep = mod_para.nrep
    # # rep = list(range(0,nrep))
    rep = [0]
    
    # epoc_num =[100] #
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-20
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average
    nep_skip = 10  # no. of skipped epochs for saving 
    
    # select a range of data for testing 
    # tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    # tlim = [datetime(2021,11,29),datetime(2021,12,2)]
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    tlim = [datetime(2021,4,1),datetime(2021,4,16)]

    dt = 3

    # tlim = [datetime(2021,11,29,3),datetime(2021,11,30,3)] # just for 2d time series
    # dt = 6
    
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d') + '_t%d'%dt
    Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    tuser0 = [(tlim[0] + timedelta(hours=x*dt)) for x in range(0,Nt)]
    tshift = 0 # in hour
    tuser = [(tlim[0] + timedelta(hours=x*dt)) for x in range(tshift,Nt+tshift)] # time shift for numerical model
    # iday0 = (tlim[0] - datetime(2017,1,2)).days+1 # schism out2d_interp_001.nc corresponds to 2017.1.2
    # iday1 = (tlim[1] - datetime(2017,1,2)).days+1
    # id_test = np.arange(iday0,iday1)
    # files_lr = [dir_lr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]  # schism output
    # files_hr = [dir_hr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]

    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    # Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
    
    # # get logitude and latitude of data 
    # nc_f = files_hr[0][0]
    # lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    # lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    # # get all hr data. Note: only for a short period, otherwise could be too large
    # hr_all_test = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    # for i in range(Nt):
    #     for ichl in range(nchl_o): 
    #         nc_f = files_hr[i][ichl]
    #         indt = indt_hr[i][ichl]
    #         dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
    #         # mask =  nc_load_vars(files_hr[i],var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[4]
    #         # dat_hr[mask] = np.nan
    #         hr_all_test[i,ichl,:,:] = dat_hr

    # # load hr_all using the way for NN model 
    # hr_all = []  # check if after normalize/denormalize, the same as hr_all_test
    # for i in range(Nt):
    #     nc_f = files_hr[i]
    #     indt = indt_hr[i]
    #     data = nc_normalize_vars(nc_f,var_hr,indt,varm_hr,
    #                                      ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
    #     x = np.transpose(data,(2,0,1)) #(C,H,W)
    #     hr = torch.from_numpy(x)
    #     mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
    #     mask = np.squeeze(mask)

    #     hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
    #     hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
        
    #     if kmask == 1: 
    #         hr_norm0[:,:,mask] = np.nan
    #     hr_all.append(hr_norm0)
    # hr_all = np.concatenate(hr_all, axis=0)
    # np.allclose(hr_all, hr_all_test, equal_nan=True)


    cuda = torch.cuda.is_available()
            
    # Initialize generator 
    if knn ==0:
        generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                    n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor)
    # discriminator = Discriminator(input_shape=(nchl_o, *hr_shape))
    elif knn == 1:
        generator = SRRes(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=opt.residual_blocks,
                                    up_factor=opt.up_factor,kernel_size=ker_size,kernel_no=kernel_no)

    # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
    ipath_nn = path_par+'nn_mod_' + str(opt.up_factor) + suf +'/' # dir of saved model 
    
    for irep in rep:
        print(f'Repeat {irep}')
        print('--------------------------------')
    
        # out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens/'
        # os.makedirs(out_path, exist_ok=True)
        
        sr_all_ep = np.zeros((Nt,nchl_o,opt.hr_height, opt.hr_width))  # epoch averaged sr_all(Nt,C,H,W)
        iepo = 0
        
        for epoch in epoc_num:

            model_name = 'netG_epoch_%d_re%d.pth' % (epoch,irep)
            if cuda:
                generator = generator.cuda()
                checkpointG = torch.load(ipath_nn + model_name)
            else:
                checkpointG = torch.load(ipath_nn + model_name, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            generator.eval()

            # sr_varm = np.zeros(shape=(nchl_o,2,Nt))  # max/min of sr at Nt for all channels
            # hr_varm = np.zeros(shape=(nchl_o,2,Nt))  # max/min of hr at Nt for all channels
            # dif_varm = np.zeros(shape=(nchl_o,2,Nt))
            
            sr_all = []
            # hr_all = []
            
            for it in range(0,Nt):

                nc_f = files_hr[it] 
                indt = indt_hr[it] # time index in nc_f
                # data = nc_normalize_vars(nc_f,var_hr,indt,varm_hr,
                #                                  ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
                # x = np.transpose(data,(2,0,1)) #(C,H,W)
                # hr = torch.from_numpy(x)
                mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
                mask = np.squeeze(mask)

                nc_f = files_lr[it]
                indt = indt_lr[it] # time index in nc_f
                data = nc_normalize_vars(nc_f,var_lr,indt,varm_lr,
                                         ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
                x = np.transpose(data,(2,0,1)) #(C,H,W)
                lr = torch.from_numpy(x)
                                
                lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
                # hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
                
                # start = time.time()
                sr = generator(lr.float())
                # end = time.time()
                # elapsed = (end - start)
                # print('cost ' + str(elapsed) + 's')
                
                sr_norm0 = var_denormalize(sr.detach().numpy(),varm_hr) # (N,C,H,W), flipud height back
                # hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
                
                if kmask == 1: 
                    # hr_norm0[:,:,mask] = np.nan
                    sr_norm0[:,:,mask] = np.nan
                
                sr_all.append(sr_norm0)
                # hr_all.append(hr_norm0)
        
                # sr_varm[:,0,it] = np.nanmax(sr_norm0,axis=(0,2,3)) # max for channel
                # sr_varm[:,1,it] = np.nanmin(sr_norm0,axis=(0,2,3)) # min for channel
                # hr_varm[:,0,it] = np.nanmax(hr_norm0,axis=(0,2,3)) # max for channel
                # hr_varm[:,1,it] = np.nanmin(hr_norm0,axis=(0,2,3)) # min for channel
                # dif = sr_norm0 - hr_norm0
                # dif_varm[:,0,it] = np.nanmax(dif,axis=(0,2,3)) # max for channel
                # dif_varm[:,1,it] = np.nanmin(dif,axis=(0,2,3)) # min for channel
        
            sr_all = np.concatenate(sr_all, axis=0)
            # hr_all = np.concatenate(hr_all, axis=0)
            # ensemble average: use selected epochs 
            iepo = iepo + 1
            sr_all_ep = (sr_all_ep*(iepo-1) + sr_all)/iepo  # epoch averaged sr [Nt,c,H,W]

    end = time.time()
    elapsed = (end - start)
    print('cost ' + str(elapsed) + 's')
    print('test time')
            
