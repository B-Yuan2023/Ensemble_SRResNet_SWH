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
from funs_prepost import plot_errbar_list,plot_line_list,plot_distri
import sys
import importlib

path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_name= 'par55e'          #'par01' # sys.argv[1]
mod_para=importlib.import_module(mod_name)
kmask = 1

# load "sr_99th_epoch%d_%d" % (epoch0,epoch)+'.npz', 01 per, mean
# np.savez(filename99,v0=sr_99per,v1=hr_99per,v2=rmse_99,v3=mae_99) 
def load_per2D(filename,figname,axlab,ivar_hr=[0],unit_var=['ssh (m)']):
    datald = np.load(filename) # load
    sr,hr,rmse,mae = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    error = sr - hr
    err_max = np.nanmax((error),axis=(1,2))
    err_min = np.nanmin((error),axis=(1,2))
    err_std = np.nanstd((error),axis=(1,2))
    nchl = len(mae)
    for i in range(nchl):
        var1 = error[i,:,:].flatten()
        data = [var1[~np.isnan(var1)]]
        figname1 = figname + "_c%d" % (i)+'.png'
        axlab[0] = axlab[0]+unit_var[i]
        # plot_distri(data,figname1,bins=50, axlab=axlab,leg=('', ), 
        #                    figsize=(8, 6), fontsize=12,capt='',style='default')
    return mae,err_max,err_min,err_std,rmse

# load "sr_tave_epoch%d_%d" % (epoch0,epoch)+'.npz'
# np.savez(filename_m,v0=sr_rmse,v1=sr_mae,v2=rmse_t,v3=mae_t) 
def load_tave2D(filename,outpath,epoch,ivar_hr=[0],unit_var=['ssh (m)']):
    datald = np.load(filename) # load
    sr_rmse,sr_mae,rmse_sm,mae_sm = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    mae_max = np.nanmax(sr_mae,axis=(1,2))
    mae_s95 = np.nanpercentile(sr_mae, 95, axis = (1,2))
    mae_s05 = np.nanpercentile(sr_mae, 5, axis = (1,2))
    mae_std = np.nanstd(sr_mae,axis=(1,2))
    nchl = len(sr_mae)
    for i in range(nchl):
        var1 = sr_mae[i,:,:].flatten()
        data = [var1[~np.isnan(var1)]]
        figname = out_path + "sr_tave_epoch%d_dist_mae_c%d" % (epoch,ivar_hr[i])+'.png'
        axlab = ('mae '+unit_var[i], 'Percentage','')
        # plot_distri(data,figname,bins=50, axlab=axlab,leg=('1', ), 
        #                    figsize=(8, 6), fontsize=12,capt='',style='default')
    
    rmse_max = np.nanmax(sr_rmse,axis=(1,2))
    rmse_s95 = np.nanpercentile(sr_rmse, 95, axis = (1,2))
    rmse_s05 = np.nanpercentile(sr_rmse, 5, axis = (1,2))
    rmse_std = np.nanstd(sr_rmse,axis=(1,2))
    for i in range(nchl):
        var1 = sr_rmse[i,:,:].flatten()
        data = [var1[~np.isnan(var1)]]
        figname = out_path + "sr_tave_epoch%d_dist_rmse_c%d" % (epoch,ivar_hr[i])+'.png'
        axlab = ('rmse '+unit_var[i], 'Percentage','')
        # plot_distri(data,figname,bins=50, axlab=axlab,leg=('1', ), 
        #                    figsize=(8, 6), fontsize=12,capt='',style='default')
    return mae_sm,mae_max,mae_s95,mae_s05,mae_std,rmse_sm,rmse_max,rmse_s05,rmse_s95,rmse_std

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
    unit_var = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']

    # read output from test_epo_ave, ensemble averaged 
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-30
    epoc_num = np.arange(epoch0,epoch1,-1)  # use the last 30 epochs for average
    
    opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'_ave/'
    
    # for rmse of time series for 2D grids (spatial distribution of rmse)
    for irep in rep:        
        print(f'Repeat {irep}')
        print('--------------------------------')
        out_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ave/'
        
        metrics_rp_ave = {'ep':[],'mae_sm': [],'mae_max': [],'mae_s95': [],'mae_s05': [],'mae_std': [],
                   'rmse_sm': [],'rmse_max': [],'rmse_s95': [],'rmse_s05': [],'rmse_std': [],
                   }
        metrics = {'ep':[],'mae_sm': [],'mae_max': [],'mae_s95': [],'mae_s05': [],'mae_std': [],
                   'rmse_sm': [],'rmse_max': [],'rmse_s95': [],'rmse_s05': [],'rmse_std': [],
                   }
        for epoch in epoc_num:
            filename = out_path + "sr_tave_epoch%d_%d" % (epoch0,epoch)+'.npz'
            figname = out_path + "sr_tave_epoch%d_%d_dist_err" % (epoch0,epoch)+'.png'
            mae_sm,mae_max,mae_s95,mae_s05,mae_std,rmse_sm,rmse_max,rmse_s05,rmse_s95,rmse_std = load_tave2D(filename,out_path,epoch,ivar_hr,unit_var)
            metrics['ep'].append([epoch])
            metrics['mae_sm'].append(mae_sm)
            metrics['mae_s95'].append(mae_s95)
            metrics['mae_s05'].append(mae_s05)
            metrics['mae_max'].append(mae_max)
            metrics['mae_std'].append(mae_std)
            metrics['rmse_sm'].append(rmse_sm)
            metrics['rmse_s95'].append(rmse_s95)
            metrics['rmse_s05'].append(rmse_s05)
            metrics['rmse_max'].append(rmse_max)
            metrics['rmse_std'].append(rmse_std)
            
        for key in metrics_rp_ave.keys():
            temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
            metrics_rp_ave[key].append(temp)  # list of rep arrays
            
        for i in range(nchl):
            xlst = metrics_rp_ave['ep']
            leg = ['run'+str(k) for k in rep]
            # use std for error bar
            dat_lst = metrics_rp_ave['mae_sm']
            err_lst = metrics_rp_ave['mae_std']
            figname = opath_st + "srf_%d_c%d_ep%d_%d_r%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1],irep) + '_ens_'+'mae_std'+'.png'
            plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname=figname,axlab=['epoch','mae'],leg=None,
                       leg_col=1, legloc=None,line_sty=None,style='default',capt='')
            # use percentile for error bar
            err_lst = [[dat_lst[0]-metrics_rp_ave['mae_s05'][0],metrics_rp_ave['mae_s95'][0]-dat_lst[0]]]
            figname = opath_st + "srf_%d_c%d_ep%d_%d_r%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1],irep) + '_ens_'+'mae_per'+'.png'
            plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname=figname,axlab=['epoch','mae'],leg=None,
                       leg_col=1, legloc=None,line_sty=None,style='default',capt='')
    
            dat_lst = metrics_rp_ave['rmse_sm']
            err_lst = metrics_rp_ave['rmse_std']
            figname = opath_st + "srf_%d_c%d_ep%d_%d_r%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1],irep) + '_ens_'+'rmse_std'+'.png'
            plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname=figname,axlab=['epoch','rmse'],leg=None,
                       leg_col=1, legloc=None,line_sty=None,style='default',capt='')
            # use percentile for error bar
            err_lst = [[dat_lst[0]-metrics_rp_ave['rmse_s05'][0],metrics_rp_ave['rmse_s95'][0]-dat_lst[0]]]
            figname = opath_st + "srf_%d_c%d_ep%d_%d_r%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1],irep) + '_ens_'+'rmse_per'+'.png'
            plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname=figname,axlab=['epoch','rmse'],leg=None,
                       leg_col=1, legloc=None,line_sty=None,style='default',capt='')
        
    # for 99th, 01st percnetile, and mean of rmse
    metrics_rp_ave = {'ep':[], 'err_99_max': [],'err_99_min': [],'err_99_std': [], #'mae_99': [],'rmse_99': [], 
                'err_01_max': [],'err_01_min': [],'err_01_std': [], #'mae_01': [], 'rmse_01': [],
               'err_m_max': [],'err_m_min': [],'err_m_std': [], # 'mae_m': [], 'rmse_m': [],
               }
    filename_std = opath_st+"srf_%d_ep%d_%d_mask" % (opt.up_factor,epoc_num[0],epoc_num[-1]) + '_testen_std.npy'
    if not os.path.isfile(filename_std):
        for irep in rep:        
            print(f'Repeat {irep}')
            print('--------------------------------')
            metrics = {'ep':[], 'err_99_max': [],'err_99_min': [],'err_99_std': [], #'mae_99': [],'rmse_99': [], 
                        'err_01_max': [],'err_01_min': [],'err_01_std': [], #'mae_01': [], 'rmse_01': [],
                       'err_m_max': [],'err_m_min': [],'err_m_std': [], # 'mae_m': [], 'rmse_m': [],
                       }
            out_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ave/'
            for epoch in epoc_num:
                filename = out_path + "sr_99th_epoch%d_%d" % (epoch0,epoch)+'.npz'
                figname = out_path + "sr_99th_epoch%d_dist_err" % (epoch)
                axlab = ['(sr_99-hr_99) ', 'Percentage','']
                mae_99,err_99_max,err_99_min,err_99_std,rmse_99 = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
                
                filename = out_path + "sr_01th_epoch%d_%d" % (epoch0,epoch)+'.npz'
                figname = out_path + "sr_01th_epoch%d_dist_err" % (epoch)
                axlab = ['(sr_01-hr_01) ', 'Percentage','']
                mae_01,err_01_max,err_01_min,err_01_std,rmse_01 = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
                
                filename = out_path + "sr_mean_epoch%d_%d" % (epoch0,epoch)+'.npz'
                figname = out_path + "sr_mean_epoch%d_dist_err" % (epoch)
                axlab = ['(sr_m -hr_m) ', 'Percentage','']
                mae_m,err_m_max,err_m_min,err_m_std,rmse_m = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
            
                metrics['ep'].append([epoch])
                # metrics['mae_99'].append(mae_99)
                # metrics['rmse_99'].append(rmse_99)
                metrics['err_99_min'].append(err_99_min)
                metrics['err_99_max'].append(err_99_max)
                metrics['err_99_std'].append(err_99_std)
                
                # metrics['mae_01'].append(mae_01)
                # metrics['rmse_01'].append(rmse_01)
                metrics['err_01_min'].append(err_01_min)
                metrics['err_01_max'].append(err_01_max)
                metrics['err_01_std'].append(err_01_std)
                
                # metrics['mae_m'].append(mae_m)
                # metrics['rmse_m'].append(rmse_m)
                metrics['err_m_min'].append(err_m_min)
                metrics['err_m_max'].append(err_m_max)
                metrics['err_m_std'].append(err_m_std)
                
            for key in metrics_rp_ave.keys():
                temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
                metrics_rp_ave[key].append(temp)  # list of rep arrays
        np.save(filename_std, metrics_rp_ave) 
    else:
        metrics_rp_ave = np.load(filename_std,allow_pickle='TRUE').item()
        
    for i in range(nchl):
        for key in metrics_rp_ave.keys():
            if key not in ['ep']: # exclude ep itself
                xlst = metrics_rp_ave['ep']
                dat_lst = metrics_rp_ave[key]
                figname = opath_st + "srf_%d_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_testen_'+key+'.png'
                leg = ['run'+str(k) for k in rep]
                plot_line_list(xlst,dat_lst,tlim=None,figname=figname,axlab=['epoch',key],leg='',
                           leg_col=1, legloc=None,line_sty=None,style='default',capt='')
                
    # =========================================================================
    # read output from test_epo
    epoc_num = np.arange(71,opt.N_epochs+1)    
    opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'/'
    metrics_rp = {'ep':[], 'err_99_max': [],'err_99_min': [],'err_99_std': [], #'mae_99': [],'rmse_99': [], 
                'err_01_max': [],'err_01_min': [],'err_01_std': [], #'mae_01': [], 'rmse_01': [],
               'err_m_max': [],'err_m_min': [],'err_m_std': [], # 'mae_m': [], 'rmse_m': [],
               }
    filename_std = opath_st+"srf_%d_ep%d_%d_mask" % (opt.up_factor,epoc_num[0],epoc_num[-1]) + '_testen_std.npy'
    if not os.path.isfile(filename_std):
        for irep in rep:        
            print(f'Repeat {irep}')
            print('--------------------------------')
            metrics = {'ep':[], 'err_99_max': [],'err_99_min': [],'err_99_std': [], #'mae_99': [],'rmse_99': [], 
                        'err_01_max': [],'err_01_min': [],'err_01_std': [], #'mae_01': [], 'rmse_01': [],
                       'err_m_max': [],'err_m_min': [],'err_m_std': [], # 'mae_m': [], 'rmse_m': [],
                       }
            out_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
            for epoch in epoc_num:
                filename = out_path + "sr_99th_epoch%d" % (epoch)+'.npz'
                figname = out_path + "sr_99th_epoch%d_dist_err" % (epoch)
                axlab = ['sr_99-hr_99 ', 'Percentage','']
                mae_99,err_99_max,err_99_min,err_99_std,rmse_99 = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
                
                filename = out_path + "sr_01th_epoch%d" % (epoch)+'.npz'
                figname = out_path + "sr_01th_epoch%d_dist_err" % (epoch)
                axlab = ['sr_01-hr_01 ', 'Percentage','']
                mae_01,err_01_max,err_01_min,err_01_std,rmse_01 = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
                
                filename = out_path + "sr_mean_epoch%d" % (epoch)+'.npz'
                figname = out_path + "sr_mean_epoch%d_dist_err" % (epoch)
                axlab = ['sr_m -hr_m ', 'Percentage','']
                mae_m,err_m_max,err_m_min,err_m_std,rmse_m = load_per2D(filename,figname,axlab,ivar_hr,unit_var)
    
                metrics['ep'].append([epoch])
                # metrics['mae_99'].append(mae_99)
                # metrics['rmse_99'].append(rmse_99)
                metrics['err_99_min'].append(err_99_min)
                metrics['err_99_max'].append(err_99_max)
                metrics['err_99_std'].append(err_99_std)
                
                # metrics['mae_01'].append(mae_01)
                # metrics['rmse_01'].append(rmse_01)
                metrics['err_01_min'].append(err_01_min)
                metrics['err_01_max'].append(err_01_max)
                metrics['err_01_std'].append(err_01_std)
                
                # metrics['mae_m'].append(mae_m)
                # metrics['rmse_m'].append(rmse_m)
                metrics['err_m_min'].append(err_m_min)
                metrics['err_m_max'].append(err_m_max)
                metrics['err_m_std'].append(err_m_std)
            
            for key in metrics_rp.keys():
                temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
                metrics_rp[key].append(temp)  # list of rep arrays
        np.save(filename_std, metrics_rp) 
    else:
        metrics_rp = np.load(filename_std,allow_pickle='TRUE').item()
    
    for i in range(nchl):
        for key in metrics_rp.keys():
            if key not in ['ep']: # exclude ep itself
                xlst = metrics_rp['ep']
                dat_lst = metrics_rp[key]
                figname = opath_st + "srf_%d_c%d_ep%d_%d_mask" % (opt.up_factor,ivar_hr[i],epoc_num[0],epoc_num[-1]) + '_test_'+key+'.png'
                leg = ['run'+str(k) for k in rep]
                plot_line_list(xlst,dat_lst,tlim=None,figname=figname,axlab=['epoch',key],leg='',
                           leg_col=1, legloc=None,line_sty=None,style='default',capt='')