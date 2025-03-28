#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:24:04 2024
plot ensemble average resutls 
@author: g260218
"""

import os
import numpy as np

import pandas as pd
from funs_prepost import plot_line_list,plotsubs_line_list
import sys
import importlib

path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= 'par55e_lr5e5'          #'par01' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
kmask = 1

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    suf0 = mod_para.suf0
    rtra = mod_para.rtra
    # rtra = 0.95
    ivar_lr = mod_para.ivar_lr
    ivar_hr = mod_para.ivar_hr
    varm_hr = mod_para.varm_hr
    varm_lr = mod_para.varm_lr
    nchl_i = len(ivar_lr)
    nchl_o = len(ivar_hr)
    
    nrep = mod_para.nrep
    rep = list(range(0,nrep))
    # rep = list(range(0,9))
    # suf = '_res' + str(opt.residual_blocks) + '_max_suv' # + '_nb' + str(opt.batch_size)
    print(f'parname: {mod_name}')
    print('--------------------------------')
    
    # epoc_num = [50,100]

    nchl = nchl_o

    epoc_num = np.arange(40,opt.N_epochs+1)    
    opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'/'
    metrics_rp = {'ep':[],'mse': [], 'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],} # # old output
    metrics_rp = {'ep':[], 'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],
                  'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],
                  'mae_t': [],'rmse_t': [],} # 'mse': [],'ssim': [],'psnr': [],

    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        # np.save(opath_st + ofname, metrics) 
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics_bt.npy'
        # # np.save(opath_st + ofname, metrics_bt) 
        # metrics_bt = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        
        for key in metrics_rp.keys():
            temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
            metrics_rp[key].append(temp)  # list of rep arrays
    
    for i in range(nchl):
        for key in metrics_rp.keys():
            if key not in ['ep']: # exclude ep itself
                xlst = metrics_rp['ep']
                dat_lst = metrics_rp[key]
                figname = opath_st + "srf_%d_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_test_'+key+'.png'
                leg = ['run'+str(k) for k in rep]
                plot_line_list(xlst,dat_lst,tlim=None,figname=figname,axlab=None,leg=None,
                           leg_col=1, legloc=None,line_sty=None,style='default',capt='')
    
    # ensemble averaged 
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-30
    epoc_num = np.arange(epoch0,epoch1,-1)  # use the last 30 epochs for average
    
    opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'_ave/'
    metrics_rp_ave = {'ep':[], 'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],
                      'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],
                      'mae_t': [],'rmse_t': [],} # 'mse': [], 
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'

        # np.save(opath_st + ofname, metrics) 
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics_bt.npy'
        # # np.save(opath_st + ofname, metrics_bt) 
        # metrics_bt = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        
        for key in metrics_rp_ave.keys():
            temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
            metrics_rp_ave[key].append(temp)  # list of rep arrays
            
    for i in range(nchl):  # i not used below
        for key in metrics_rp_ave.keys():
            if key not in ['ep']: # exclude ep itself
                xlst = metrics_rp_ave['ep']
                dat_lst = metrics_rp_ave[key]
                figname = opath_st + "srf_%d_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_testen_'+key+'.png'
                leg = ['run'+str(k) for k in rep]
                plot_line_list(xlst,dat_lst,tlim=None,figname=figname,axlab=None,leg=None,
                           leg_col=1, legloc=None,line_sty=None,style='default',capt='')
                
    
    # plot comparison between original and ensemble metrics 
    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]
    
    # plt.rcParams['axes.prop_cycle'].by_key()['color']
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_sty=['--','--','--','-','-','-'] 
    line_col=['#1f77b4', '#ff7f0e', '#2ca02c','#1f77b4', '#ff7f0e', '#2ca02c',] 
    # line_sty=['b--','r--','g--','b-','r-','g-'] 

    key_use = ['rmse','mae','rmse_m','mae_m',
                'rmse_99','mae_99','rmse_01','mae_01']
    nrow=4
    # key_use = ['rmse','rmse_m','rmse_99','rmse_01']
    # nrow=2
    nkey_u = len(key_use)
    axlab = [['Epoch',key_use[j]+' (m)'] for j in range(nkey_u)]
    leg = ['SR_run'+str(k) for k in rep]+['SR_en_run'+str(k) for k in rep]

    for i in range(nchl):  # i not used below except for name 
        time_lst = [metrics_rp['ep']+metrics_rp_ave['ep']]*nkey_u
        data_lst_np = []
        for ikey in key_use:            
            data_lst_np.append(metrics_rp[ikey]+metrics_rp_ave[ikey])
        figname = opath_st+"/c%d_ep%d_%d" % (ivar_hr[i],epoc_num[0],epoc_num[-1]) +'_cmp_ens_v%d'%nkey_u+'.png'
        tlim = [70, 100]
        subsize = [3.7,2.0]  # A4 8.3*11.7   
        lloc = 9  # legend location, 9-upper center
        legloc = [0.5,1.05] 
        plotsubs_line_list(time_lst,data_lst_np,figname,tlim,subsize = subsize,
                                    fontsize=12,nrow=nrow,title=None,axlab=axlab,
                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                    line_sty=line_sty,line_col=line_col,capt=capt_all)
        