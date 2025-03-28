#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:49:14 2025
compare input data distribution 
@author: g260218
"""


import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import myDataset
from funs_prepost import nc_load_vars,var_denormalize,plot_distri,make_list_file_t

import torch
import pandas as pd
from datetime import datetime, timedelta # , date

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
    dir_lr = opt.dir_lr
    dir_hr = opt.dir_hr
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
    
    if hasattr(mod_para, 'll_lr'):
        ll_lr = mod_para.ll_lr # user domain lr [latitude,longitude]
    else:
        ll_lr = [None]*2
    if hasattr(mod_para, 'll_hr'):
        ll_hr = mod_para.ll_hr # user domain hr [latitude,longitude]
    else:
        ll_hr = [None]*2
    
    tlim = [datetime(2018,1,1),datetime(2020,12,31)]
    dt = 3
    
    # Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    # tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d') + '_t%d'%dt
    # tuser0 = [(tlim[0] + timedelta(hours=x*dt)) for x in range(0,Nt)]
    # tshift = 0 # in hour
    # tuser = [(tlim[0] + timedelta(hours=x*dt)) for x in range(tshift,Nt+tshift)] # time shift for numerical model

    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]
    
    # get logitude and latitude of data 
    nc_f = files_hr[0][0]
    lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    # get all hr data. Note: only for a short period, otherwise could be too large
    hr_all_train = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    for i in range(Nt):
        for ichl in range(nchl_o): 
            nc_f = files_hr[i][ichl]
            indt = indt_hr[i][ichl]
            dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
            mask =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[4]
            dat_hr[mask] = np.nan
            hr_all_train[i,ichl,:,:] = dat_hr
            
            
    # for a different time step
    dt = 6
    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]

    # get all hr data. Note: only for a short period, otherwise could be too large
    hr_all_train1 = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    for i in range(Nt):
        for ichl in range(nchl_o): 
            nc_f = files_hr[i][ichl]
            indt = indt_hr[i][ichl]
            dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
            mask =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[4]
            dat_hr[mask] = np.nan
            hr_all_train1[i,ichl,:,:] = dat_hr
            
    # for a different time step
    dt = 12
    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]

    # get all hr data. Note: only for a short period, otherwise could be too large
    hr_all_train2 = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    for i in range(Nt):
        for ichl in range(nchl_o): 
            nc_f = files_hr[i][ichl]
            indt = indt_hr[i][ichl]
            dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
            mask =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[4]
            dat_hr[mask] = np.nan
            hr_all_train2[i,ichl,:,:] = dat_hr
            
    # for a different time step
    dt = 48
    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    Nt = len(files_lr)
    
    # create nested list of files and indt, 
    # no. of inner list == no. of channels, no. outer list == no. of samples
    if len(files_hr[0])!=nchl_o:
        files_hr = [[ele for _ in range(nchl_o)] for ele in files_hr]
        indt_hr = [[ele for _ in range(nchl_o)] for ele in indt_hr]
    if len(files_lr[0])!=nchl_i:
        files_lr = [[ele for _ in range(nchl_i)] for ele in files_lr]
        indt_lr = [[ele for _ in range(nchl_i)] for ele in indt_lr]

    # get all hr data. Note: only for a short period, otherwise could be too large
    hr_all_train3 = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    for i in range(Nt):
        for ichl in range(nchl_o): 
            nc_f = files_hr[i][ichl]
            indt = indt_hr[i][ichl]
            dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
            mask =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[4]
            dat_hr[mask] = np.nan
            hr_all_train3[i,ichl,:,:] = dat_hr
            
# plot the distribution of input data together 
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']    
    opath_st = path_par+'stat' + suf +'/'
    os.makedirs(opath_st, exist_ok=True)
    
    for k in range(nchl_o): # nchl_o, save for all time steps. 
        ichl = ivar_hr[k]
        unit_var = unit_suv[ichl]
        # plot distribution of reconstructed vs target, all data, histogram
        axlab = (unit_var,'Frequency','')
        lim = [0,6]
        nbin= 20
        xlim = [0,5]
        ylim = [0,0.30]
        var0 = hr_all_train[:,k,:,:].flatten()
        var1 = hr_all_train1[:,k,:,:].flatten()
        var2 = hr_all_train2[:,k,:,:].flatten()
        var3 = hr_all_train3[:,k,:,:].flatten()
        var = [var0[~np.isnan(var0)],var1[~np.isnan(var1)],var2[~np.isnan(var2)],
               var3[~np.isnan(var3)],
               ]
        leg = ['dt3','dt6', 'dt12','dt48'] 
        figname = opath_st+"dist_c%d" % (ichl)+'_input%d'%(len(var))+'.png'
        plot_distri(var,figname,nbin=nbin,lim=lim,axlab=axlab,leg=leg,
                       figsize=(7.4, 2.5), fontsize=12,capt='',
                       xlim=xlim,ylim=ylim)