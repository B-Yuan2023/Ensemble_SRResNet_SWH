#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023
compare rmse/mae of 99,01 percentile and mean var between DL and ML models
@author: g260218
"""
import os
import sys
import numpy as np
import pandas as pd
# from datetime import datetime, timedelta # , date

from funs_prepost import plt_pcolorbar_list,var_normalize,ssim_tor

import importlib
mod_name= 'par55e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    files_lr = mod_para.files_lr
    files_hr = mod_para.files_hr
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    ivar_lr = mod_para.ivar_lr
    varm_hr = mod_para.varm_hr

    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    rtra = mod_para.rtra
    
    nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    rep = [0]
    epoc_num =[100] #
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-30 
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average

    kp_2D = 1   # key to plot comparison for 2D map, mean/01/99 percentile 
    kp_2D_ord = 1  # for 2d plot dim0 order: 1 first time (in a row) next model; 2 first model next time
    nep_skip = 10  # no. of skipped epochs for plotting 
        
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_ens/'
    os.makedirs(out_path0, exist_ok=True)    
    
    # ========================================================================
    # load 99/01 percentile sr from linear regression (test.py)
    ml_suf = '_md0'
    ml_mod_name= 'par55e_md0'  # should match with srresnet mod_name
    ml_path_par = '/work/gg0028/g260218/GB_output_interp/wave_cmems_blacksea/ml_traditional/'
    irep = 0
    ml_path0 = ml_path_par+'results_test/'+'S'+str(opt.up_factor)+'_'+ml_mod_name+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
    filename99 = ml_path0 + "sr_99th_" +'re'+ str(irep)+'.npz'
    if not os.path.isfile(filename99): 
        sys.exit('ml 99 file not saved!')
    else:
        datald = np.load(filename99) # load
        sr_99per_ml,hr_99per,rmse_99_ml,mae_99_ml = datald['v0'],datald['v1'],datald['v2'],datald['v3']

    filename01 = ml_path0 + "sr_01th_" +'re'+ str(irep)+'.npz'
    if not os.path.isfile(filename01): 
        sys.exit('ml 01 file not saved!')
    else:
        datald = np.load(filename01) # load
        sr_01per_ml,hr_01per,rmse_01_ml,mae_01_ml = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
    filename_m = ml_path0 + "sr_mean_" +'re'+ str(irep)+'.npz'
    if not os.path.isfile(filename_m): 
        sys.exit('ml mean file not saved!')
    else:
        datald = np.load(filename_m) # load
        sr_mean_ml,hr_mean,rmse_m_ml,mae_m_ml = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
    filename_t = ml_path0 + "sr_tave_" +'re'+ str(irep)+'.npz'
    if not os.path.isfile(filename_t): 
        sys.exit('ml tave file not saved!')
    else:
        datald = np.load(filename_t) # load
        sr_rmse_ml,sr_mae_ml,rmse_t_ml,mae_t_ml = datald['v0'],datald['v1'],datald['v2'],datald['v3']

    # load global mae/rmse for ml models
    opath_st_ml = ml_path_par+'stat_' + ml_mod_name +'/'
    mae_ml, rmse_ml = [],[]
    for i in range(nchl_o):
        ofname = "srf_%d_c%d_mask" % (opt.up_factor,ivar_hr[i]) + '_test_metrics.csv'
        metrics_ml = np.loadtxt(opath_st_ml + ofname, delimiter=",",skiprows=1)
        mae_ml.append(metrics_ml[2])
        rmse_ml.append(metrics_ml[3])
    # ========================================================================

    # load 99/01 percentile of original hr and interpolated hr (cal_metrics_intp.py)
    # note: out_path has no mode_name, only up_factor and var_hr[0]
    # for cases with >1 input channels, interpolation not needed
    out_path = path_par+'results_test/'+'S%d_mk%d'%(opt.up_factor,kmask)+'/'+ var_hr[0]+'/'
    if ivar_hr==ivar_lr:
        filename99 = out_path + 'hr_99per_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        filename01 = out_path + 'hr_01per_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        filename_m = out_path + 'hr_mean_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
        filename_t = out_path + 'hr_tave_interp'+'_train%4.2f'%(rtra)+'.npz' # file for temporal rmse/mae of points
        if not os.path.isfile(filename99):
            sys.exit('hr file from interpolation not saved!')
        else:
            datald = np.load(filename99) # load
            hr_99per,hr_re1_99per,hr_re2_99per,hr_re3_99per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
            lat,lon = datald['lat'],datald['lon']
        
        if not os.path.isfile(filename01):
            sys.exit('hr file from interpolation not saved!')
        else:
            datald = np.load(filename01) # load
            hr_01per,hr_re1_01per,hr_re2_01per,hr_re3_01per = datald['v0'],datald['v1'],datald['v2'],datald['v3']
    
        if not os.path.isfile(filename_m):
            sys.exit('hr file from interpolation not saved!')
        else:
            datald = np.load(filename_m) # load
            hr_mean,hr_re1_mean,hr_re2_mean,hr_re3_mean = datald['v0'],datald['v1'],datald['v2'],datald['v3']
        
        if not os.path.isfile(filename_t):
            sys.exit('hr file from interpolation not saved!')
        else:
            datald = np.load(filename_t) # load
            hr_re1_rmse,hr_re1_mae,hr_re2_rmse,hr_re2_mae,hr_re3_rmse,hr_re3_mae = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
        # load the corresponding rmse/mae of 99/01 percentile, mean, tave, global
        filename99m = out_path + 'hr_99per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        datald = np.load(filename99m) # load
        rmse_99_re1,rmse_99_re2,rmse_99_re3,mae_99_re1,mae_99_re2,mae_99_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
        filename01m = out_path + 'hr_01per_rmse_interp'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        datald = np.load(filename01m) # load
        rmse_01_re1,rmse_01_re2,rmse_01_re3,mae_01_re1,mae_01_re2,mae_01_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        
        filename_mm = out_path + 'hr_mean_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # file for mean
        datald = np.load(filename_mm) # load
        rmse_m_re1,rmse_m_re2,rmse_m_re3,mae_m_re1,mae_m_re2,mae_m_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
        filename_tm = out_path + 'hr_tave_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # global avraged rmse/mae of points
        datald = np.load(filename_tm) # load
        rmse_t_re1,rmse_t_re2,rmse_t_re3,mae_t_re1,mae_t_re2,mae_t_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
    
        filename = out_path + 'hr_gb_rmse_interp'+'_train%4.2f'%(rtra)+'.npz' # global avraged rmse/mae of points
        datald = np.load(filename) # load
        rmse_re1,rmse_re2,rmse_re3,mae_re1,mae_re2,mae_re3 = datald['v0'],datald['v1'],datald['v2'],datald['v3'],datald['v4'],datald['v5']
        
    else:
    # only estimate 99/01 percentile, mean for hr 
        filename99 = out_path + 'hr_99per'+'_train%4.2f'%(rtra)+'.npz' # file for 99th percentile
        filename01 = out_path + 'hr_01per'+'_train%4.2f'%(rtra)+'.npz'# file for 01st percentile
        filename_m = out_path + 'hr_mean'+'_train%4.2f'%(rtra)+'.npz' # file for mean

        if not os.path.isfile(filename99):
            sys.exit('hr 99 file not saved!')
        else:
            datald = np.load(filename99) # load 
            hr_99per,lat,lon = datald['v0'],datald['lat'],datald['lon']
        
        if not os.path.isfile(filename01):
            sys.exit('hr 01 file not saved!')
        else:
            datald = np.load(filename01) # load
            hr_01per = datald['v0']
           
        if not os.path.isfile(filename_m):
            sys.exit('hr mean file not saved!')
        else:
            datald = np.load(filename_m) # load
            hr_mean = datald['v0']
    
    # normalization based on range of 99/01/mean seperately
    if ivar_hr==ivar_lr:
        
        var99 = hr_99per
        # var99 = np.concatenate([hr_99per,sr_99per_ml,hr_re1_99per,hr_re2_99per,hr_re3_99per],axis=1)
        varm_99 = np.stack((np.nanmax(var99,axis=(1,2)),np.nanmin(var99,axis=(1,2))),axis=1)

        var01 = hr_01per
        # var01 = np.concatenate([hr_01per,sr_01per_ml,hr_re1_01per,hr_re2_01per,hr_re3_01per],axis=1)
        varm_01 = np.stack((np.nanmax(var01,axis=(1,2)),np.nanmin(var01,axis=(1,2))),axis=1)

        varmean = hr_mean        
        # varmean = np.concatenate([hr_mean,sr_mean_ml,hr_re1_mean,hr_re2_mean,hr_re3_mean],axis=1)
        varm_m = np.stack((np.nanmax(varmean,axis=(1,2)),np.nanmin(varmean,axis=(1,2))),axis=1)
        
        # varm_99 = np.array([[4,0]])
        # varm_01 = varm_99
        # varm_m = varm_99
        
        hr_re1_99per_nm = var_normalize(hr_re1_99per,varmaxmin=varm_99)
        hr_re2_99per_nm = var_normalize(hr_re2_99per,varmaxmin=varm_99)
        hr_re3_99per_nm = var_normalize(hr_re3_99per,varmaxmin=varm_99)
        
        hr_re1_01per_nm = var_normalize(hr_re1_01per,varmaxmin=varm_01)
        hr_re2_01per_nm = var_normalize(hr_re2_01per,varmaxmin=varm_01)
        hr_re3_01per_nm = var_normalize(hr_re3_01per,varmaxmin=varm_01)
        
        hr_re1_mean_nm = var_normalize(hr_re1_mean,varmaxmin=varm_m)
        hr_re2_mean_nm = var_normalize(hr_re2_mean,varmaxmin=varm_m)
        hr_re3_mean_nm = var_normalize(hr_re3_mean,varmaxmin=varm_m)
        
        sr_99per_ml_nm = var_normalize(sr_99per_ml,varmaxmin=varm_99)
        hr_99per_nm = var_normalize(hr_99per,varmaxmin=varm_99)
        sr_01per_ml_nm = var_normalize(sr_01per_ml,varmaxmin=varm_01)
        hr_01per_nm = var_normalize(hr_01per,varmaxmin=varm_01)
        sr_mean_ml_nm = var_normalize(sr_mean_ml,varmaxmin=varm_m)
        hr_mean_nm = var_normalize(hr_mean,varmaxmin=varm_m)
    else:
        var99 = hr_99per
        # var99 = np.concatenate([hr_99per,sr_99per_ml],axis=1)
        varm_99 = np.stack((np.nanmax(var99,axis=(1,2)),np.nanmin(var99,axis=(1,2))),axis=1)

        var01 = hr_01per
        # var01 = np.concatenate([hr_01per,sr_01per_ml],axis=1)
        varm_01 = np.stack((np.nanmax(var01,axis=(1,2)),np.nanmin(var01,axis=(1,2))),axis=1)
        
        varmean = hr_mean        
        # varmean = np.concatenate([hr_mean,sr_mean_ml],axis=1)
        varm_m = np.stack((np.nanmax(varmean,axis=(1,2)),np.nanmin(varmean,axis=(1,2))),axis=1)
        
        sr_99per_ml_nm = var_normalize(sr_99per_ml,varmaxmin=varm_99)
        hr_99per_nm = var_normalize(hr_99per,varmaxmin=varm_99)
        sr_01per_ml_nm = var_normalize(sr_01per_ml,varmaxmin=varm_01)
        hr_01per_nm = var_normalize(hr_01per,varmaxmin=varm_01)
        sr_mean_ml_nm = var_normalize(sr_mean_ml,varmaxmin=varm_m)
        hr_mean_nm = var_normalize(hr_mean,varmaxmin=varm_m)
    
    
    # ========================================================================
    
    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]

    clim = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,5.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    clim_dif = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[-1,1],[-1,1],[-0.2,0.2],[-1,1.]]  # diff in ssh,u,v,uw,vw,swh,pwp
    clim_m = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,1.5],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    clim_99 = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,4],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    clim_01 = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,0.4],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp

    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']    
    for irep in rep:
        print(f'Repeat {irep}')
        print('--------------------------------')
    
        out_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens'+ml_suf+'/'
        os.makedirs(out_path, exist_ok=True)

        # to load the global mae/rmse (all epochs) from ensemble (test_epo_ave.py)
        opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'_ave/'
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
            
        out_path0 = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ave/'
        iepo = 0
        for epoch in epoc_num:
            # load 99/01 percentile from srresnet ensemble (test_epo_ave.py)
            filename99 = out_path0 + "sr_99th_epoch%d_%d" % (epoch0,epoch)+'.npz'
            if not os.path.isfile(filename99): 
                sys.exit('sr file not saved!')
            else:
                datald = np.load(filename99) # load
                sr_99per,hr_99per,rmse_99,mae_99 = datald['v0'],datald['v1'],datald['v2'],datald['v3']

            filename01 = out_path0 + "sr_01th_epoch%d_%d" % (epoch0,epoch)+'.npz'
            datald = np.load(filename01) # load
            sr_01per,hr_01per,rmse_01,mae_01 = datald['v0'],datald['v1'],datald['v2'],datald['v3']

            filename_m = out_path0 + "sr_mean_epoch%d_%d" % (epoch0,epoch)+'.npz'
            datald = np.load(filename_m) # load
            sr_mean,hr_mean,rmse_m,mae_m = datald['v0'],datald['v1'],datald['v2'],datald['v3']

            filename_t = out_path0 + "sr_tave_epoch%d_%d" % (epoch0,epoch)+'.npz'
            datald = np.load(filename_t) # load
            sr_rmse,sr_mae,rmse_t,mae_t = datald['v0'],datald['v1'],datald['v2'],datald['v3']
                        
            sr_99per_nm = var_normalize(sr_99per,varmaxmin=varm_99)
            sr_01per_nm = var_normalize(sr_01per,varmaxmin=varm_01)
            sr_mean_nm = var_normalize(sr_mean,varmaxmin=varm_m)
            
            ssim_99_ep = ssim_tor(sr_99per_nm, hr_99per_nm)
            ssim_99_ml = ssim_tor(sr_99per_ml_nm, hr_99per_nm)
            ssim_01_ep = ssim_tor(sr_01per_nm, hr_01per_nm)
            ssim_01_ml = ssim_tor(sr_01per_ml_nm, hr_01per_nm)
            ssim_m_ep = ssim_tor(sr_mean_nm, hr_mean_nm)
            ssim_m_ml = ssim_tor(sr_mean_ml_nm, hr_mean_nm)
            
            if kp_2D == 1 and (iepo+1)%nep_skip == 0:
                # save error info to csv 
                for i in range(nchl_o):
                    ichl = ivar_hr[i]
                    filename = out_path+"c%d_re%d_ep%d_%d"% (ichl,irep,epoch,epoch0)+"_err.csv"
                    # metric_mds = np.concatenate([mae_sr_ep[:,i],rmse_sr_ep[:,i],
                    #                              mae_sr_mlr[:,i],rmse_sr_mlr[:,i],
                    #                              mae_re2[:,i],rmse_re2[:,i],
                    #                              mae_re3[:,i],rmse_re3[:,i]],axis=1)
                    # header = 'mae_sr_ep,rmse_sr_ep,mae_sr_mlr,rmse_sr_mlr,mae_re2,rmse_re2,mae_re3,rmse_re3,'
                    # np.savetxt(filename,metric_mds, delimiter=',', header=header, comments="")
                    if ivar_hr==ivar_lr:
                        ssim_99_re1 = ssim_tor(hr_re1_99per_nm, hr_99per_nm)
                        ssim_99_re2 = ssim_tor(hr_re2_99per_nm, hr_99per_nm)
                        ssim_99_re3 = ssim_tor(hr_re3_99per_nm, hr_99per_nm)
                        
                        ssim_01_re1 = ssim_tor(hr_re1_01per_nm, hr_01per_nm)
                        ssim_01_re2 = ssim_tor(hr_re2_01per_nm, hr_01per_nm)
                        ssim_01_re3 = ssim_tor(hr_re3_01per_nm, hr_01per_nm)
                        
                        ssim_m_re1 = ssim_tor(hr_re1_mean_nm, hr_mean_nm)
                        ssim_m_re2 = ssim_tor(hr_re2_mean_nm, hr_mean_nm)
                        ssim_m_re3 = ssim_tor(hr_re3_mean_nm, hr_mean_nm)

                        metrics_mod = {'Errors':['RMSE (m)','MAE (m)','RMSE_m (m)','MAE_m (m)','RMSE_99 (m)','MAE_99 (m)',
                                                 'RMSE_01 (m)','MAE_01 (m)','RMSE_t (m)','MAE_t (m)',
                                                 'SSIM_m','SSIM_99','SSIM_01'],
                                           'SR_en':[metrics['rmse'][iepo][i],metrics['mae'][iepo][i],
                                                    metrics['rmse_m'][iepo][i],metrics['mae_m'][iepo][i],
                                                    metrics['rmse_99'][iepo][i],metrics['mae_99'][iepo][i],
                                                    metrics['rmse_01'][iepo][i],metrics['mae_01'][iepo][i],
                                                    metrics['rmse_t'][iepo][i],metrics['mae_t'][iepo][i],
                                                    ssim_m_ep[i],ssim_99_ep[i],ssim_01_ep[i]],
                                           'MLR':[rmse_ml[i],mae_ml[i],rmse_m_ml[i],mae_m_ml[i],rmse_99_ml[i],mae_99_ml[i],
                                                  rmse_01_ml[i],mae_01_ml[i],rmse_t_ml[i],mae_t_ml[i],
                                                  ssim_m_ml[i],ssim_99_ml[i],ssim_01_ml[i]],
                                           'RBFlinear':[rmse_re1[i],mae_re1[i],rmse_m_re1[i],mae_m_re1[i],rmse_99_re1[i],mae_99_re1[i],
                                                      rmse_01_re1[i],mae_01_re1[i],rmse_t_re1[i],mae_t_re1[i],
                                                      ssim_m_re1[i],ssim_99_re1[i],ssim_01_re1[i]],
                                           'Bilnear':[rmse_re2[i],mae_re2[i],rmse_m_re2[i],mae_m_re2[i],rmse_99_re2[i],mae_99_re2[i],
                                                      rmse_01_re2[i],mae_01_re2[i],rmse_t_re2[i],mae_t_re2[i],
                                                      ssim_m_re2[i],ssim_99_re2[i],ssim_01_re2[i]],
                                           'Nearest':[rmse_re3[i],mae_re3[i],rmse_m_re3[i],mae_m_re3[i],rmse_99_re3[i],mae_99_re3[i],
                                                      rmse_01_re3[i],mae_01_re3[i],rmse_t_re3[i],mae_t_re3[i],
                                                      ssim_m_re3[i],ssim_99_re3[i],ssim_01_re3[i]]
                                           }
                    else:
                        metrics_mod = {'Errors':['RMSE (m)','MAE (m)','RMSE_m (m)','MAE_m (m)','RMSE_99 (m)','MAE_99 (m)',
                                                 'RMSE_01 (m)','MAE_01 (m)','RMSE_t (m)','MAE_t (m)',
                                                 'SSIM_m','SSIM_99','SSIM_01'],
                                           'SR_en':[metrics['rmse'][iepo][i],metrics['mae'][iepo][i],
                                                    metrics['rmse_m'][iepo][i],metrics['mae_m'][iepo][i],
                                                    metrics['rmse_99'][iepo][i],metrics['mae_99'][iepo][i],
                                                    metrics['rmse_01'][iepo][i],metrics['mae_01'][iepo][i],
                                                    metrics['rmse_t'][iepo][i],metrics['mae_t'][iepo][i],
                                                    ssim_m_ep[i],ssim_99_ep[i],ssim_01_ep[i]],
                                           'MLR':[rmse_ml[i],mae_ml[i],rmse_m_ml[i],mae_m_ml[i],rmse_99_ml[i],mae_99_ml[i],
                                                  rmse_01_ml[i],mae_01_ml[i],rmse_t_ml[i],mae_t_ml[i],
                                                  ssim_m_ml[i],ssim_99_ml[i],ssim_01_ml[i]],
                                           }
                    df = pd.DataFrame(metrics_mod)
                    # if not os.path.isfile(filename): 
                    df.to_csv(filename, index=False)
                
                    # plot comparison for 2D field, 99/01/mean
                    loc_txt = [0.01,0.90] # location of text
                    # nt_sub = 5  # plot nt_sub times in one row 
                    kax = 1   # turn ax off or not, 1 off. 

                    if kp_2D_ord==0:   # show data first for time (in row) next for model
                        ncol = 3    
                        if ivar_hr==ivar_lr:
                            sample  = [np.stack((hr_mean[i,:,:],hr_99per[i,:,:],hr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean[i,:,:],sr_99per[i,:,:],sr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean_ml[i,:,:],sr_99per_ml[i,:,:],sr_01per_ml[i,:,:]),axis=0),
                                           np.stack((hr_re1_mean[i,:,:],hr_re1_99per[i,:,:],hr_re1_01per[i,:,:]),axis=0),
                                           np.stack((hr_re3_mean[i,:,:],hr_re3_99per[i,:,:],hr_re3_01per[i,:,:]),axis=0),]
                            sample_dif  = [np.stack((hr_mean[i,:,:],hr_99per[i,:,:],hr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean[i,:,:]-hr_mean[i,:,:],
                                                     sr_99per[i,:,:]-hr_99per[i,:,:],
                                                     sr_01per[i,:,:]-hr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean_ml[i,:,:]-hr_mean[i,:,:],
                                                     sr_99per_ml[i,:,:]-hr_99per[i,:,:],
                                                     sr_01per_ml[i,:,:]-hr_01per[i,:,:]),axis=0),
                                           np.stack((hr_re1_mean[i,:,:]-hr_mean[i,:,:],
                                                     hr_re1_99per[i,:,:]-hr_99per[i,:,:],
                                                     hr_re1_01per[i,:,:]-hr_01per[i,:,:]),axis=0),
                                           np.stack((hr_re3_mean[i,:,:]-hr_mean[i,:,:],
                                                     hr_re3_99per[i,:,:]-hr_99per[i,:,:],
                                                     hr_re3_01per[i,:,:]-hr_01per[i,:,:]),axis=0),]
                            title = ['Reference ' for it in range(ncol)] + \
                                ['SR_en ' for it in range(ncol)] + \
                                ['MLR ' for it in range(ncol)] + \
                                ['RBFlinear ' for it in range(ncol)]+ \
                                ['Nearest ' for it in range(ncol)]
                            title_dif = ['Reference ' for it in range(ncol)] + \
                                ['SR_en-Ref ' for it in range(ncol)] + \
                                ['MLR-Ref ' for it in range(ncol)] + \
                                ['RBFlinear-Ref ' for it in range(ncol)]+ \
                                ['Nearest-Ref ' for it in range(ncol)]
                            # txt = ['' for it in range(ncol)] + \
                            #     [''+'ssim%5.3f'%ssim_m_ep[i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ep[i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ep[i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_re2[i]+'\nmae%5.3f'%mae_m_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_re2[i]+'\nmae%5.3f'%mae_99_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_re2[i]+'\nmae%5.3f'%mae_01_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_re3[i]+'\nmae%5.3f'%mae_m_re3[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_re3[i]+'\nmae%5.3f'%mae_99_re3[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_re3[i]+'\nmae%5.3f'%mae_01_re3[i]]
                            txt = ['' for it in range(ncol)] + \
                                [''+'rmse%5.3f'%metrics['rmse_m'][iepo][i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                                [''+'rmse%5.3f'%metrics['rmse_99'][iepo][i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                                [''+'rmse%5.3f'%metrics['rmse_01'][iepo][i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_m_re1[i]+'\nmae%5.3f'%mae_m_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_99_re1[i]+'\nmae%5.3f'%mae_99_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_01_re1[i]+'\nmae%5.3f'%mae_01_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_m_re3[i]+'\nmae%5.3f'%mae_m_re3[i]] + \
                                [''+'rmse%5.3f'%rmse_99_re3[i]+'\nmae%5.3f'%mae_99_re3[i]] + \
                                [''+'rmse%5.3f'%rmse_01_re3[i]+'\nmae%5.3f'%mae_01_re3[i]]
                        else:
                            sample  = [np.stack((hr_mean[i,:,:],hr_99per[i,:,:],hr_01per[i,:,:]),axis=0),
                                       np.stack((sr_mean[i,:,:],sr_99per[i,:,:],sr_01per[i,:,:]),axis=0),
                                       np.stack((sr_mean_ml[i,:,:],sr_99per_ml[i,:,:],sr_01per_ml[i,:,:]),axis=0),]
                            sample_dif  = [np.stack((hr_mean[i,:,:],hr_99per[i,:,:],hr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean[i,:,:]-hr_mean[i,:,:],
                                                     sr_99per[i,:,:]-hr_99per[i,:,:],
                                                     sr_01per[i,:,:]-hr_01per[i,:,:]),axis=0),
                                           np.stack((sr_mean_ml[i,:,:]-hr_mean[i,:,:],
                                                     sr_99per_ml[i,:,:]-hr_99per[i,:,:],
                                                     sr_01per_ml[i,:,:]-hr_01per[i,:,:]),axis=0)]
                            title = ['Reference ' for it in range(ncol)] + \
                                ['SR_en ' for it in range(ncol)] + \
                                ['MLR ' for it in range(ncol)]
                            title_dif = ['Reference ' for it in range(ncol)] + \
                                ['SR_en-Ref ' for it in range(ncol)] + \
                                ['MLR-Ref ' for it in range(ncol)]
                            # txt = ['' for it in range(ncol)] + \
                            #     [''+'ssim%5.3f'%ssim_m_ep[i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ep[i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ep[i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]] 
                            txt = ['' for it in range(ncol)] + \
                                [''+'rmse%5.3f'%metrics['rmse_m'][iepo][i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                                [''+'rmse%5.3f'%metrics['rmse_99'][iepo][i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                                [''+'rmse%5.3f'%metrics['rmse_01'][iepo][i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]]
                        nrow = len(sample)
                        sample = np.concatenate(sample, axis=0)
                        sample_dif = np.concatenate(sample_dif, axis=0)
               
                    else:   # show data first for model (in row) next for time
                    # dim0 order first model next time, [md0[0],md1[0]...], [md0[1],md1[1]...]...
                        if ivar_hr==ivar_lr:
                            ncol = 5
                            sample  = [np.stack((hr_mean[i,:,:],sr_mean[i,:,:],sr_mean_ml[i,:,:],hr_re1_mean[i,:,:],hr_re3_mean[i,:,:]),axis=0),
                                       np.stack((hr_99per[i,:,:],sr_99per[i,:,:],sr_99per_ml[i,:,:],hr_re1_99per[i,:,:],hr_re3_99per[i,:,:]),axis=0),
                                       np.stack((hr_01per[i,:,:],sr_01per[i,:,:],sr_01per_ml[i,:,:],hr_re1_01per[i,:,:],hr_re3_01per[i,:,:]),axis=0)
                                       ]
                            sample_dif  = [np.stack((hr_mean[i,:,:],
                                                     sr_mean[i,:,:]-hr_mean[i,:,:],
                                                     sr_mean_ml[i,:,:]-hr_mean[i,:,:],
                                                     hr_re1_mean[i,:,:]-hr_mean[i,:,:],
                                                     hr_re3_mean[i,:,:]-hr_mean[i,:,:]),axis=0),
                                           np.stack((hr_99per[i,:,:],
                                                     sr_99per[i,:,:]-hr_99per[i,:,:],
                                                     sr_99per_ml[i,:,:]-hr_99per[i,:,:],
                                                     hr_re1_99per[i,:,:]-hr_99per[i,:,:],
                                                     hr_re3_99per[i,:,:]-hr_99per[i,:,:]),axis=0),
                                           np.stack((hr_01per[i,:,:],
                                                     sr_01per[i,:,:]-hr_01per[i,:,:],
                                                     sr_01per_ml[i,:,:]-hr_01per[i,:,:],
                                                     hr_re1_01per[i,:,:]-hr_01per[i,:,:],
                                                     hr_re3_01per[i,:,:]-hr_01per[i,:,:]),axis=0)
                                       ]
                            title = ['Reference','SR_en','MLR','RBFlinear','Nearest'] + \
                                ['' for it in range((len(sample)-1)*5)]
                            title_dif = ['Reference','SR_en-Ref','MLR-Ref','RBFlinear-Ref','Nearest-Ref'] + \
                                ['' for it in range((len(sample)-1)*5)]
                            # txt = ['Mean'] + \
                            #     [''+'ssim%5.3f'%ssim_m_ep[i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_re2[i]+'\nmae%5.3f'%mae_m_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_re3[i]+'\nmae%5.3f'%mae_m_re3[i]] + \
                            #     ['99per'] + \
                            #     [''+'ssim%5.3f'%ssim_99_ep[i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_re2[i]+'\nmae%5.3f'%mae_99_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_re3[i]+'\nmae%5.3f'%mae_99_re3[i]] + \
                            #     ['01per'] + \
                            #     [''+'ssim%5.3f'%ssim_01_ep[i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_re2[i]+'\nmae%5.3f'%mae_01_re2[i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_re3[i]+'\nmae%5.3f'%mae_01_re3[i]] 
                            txt = ['Mean'] + \
                                [''+'rmse%5.3f'%metrics['rmse_m'][iepo][i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_m_re1[i]+'\nmae%5.3f'%mae_m_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_m_re3[i]+'\nmae%5.3f'%mae_m_re3[i]] + \
                                ['99per'] + \
                                [''+'rmse%5.3f'%metrics['rmse_99'][iepo][i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_99_re1[i]+'\nmae%5.3f'%mae_99_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_99_re3[i]+'\nmae%5.3f'%mae_99_re3[i]] + \
                                ['01per'] + \
                                [''+'rmse%5.3f'%metrics['rmse_01'][iepo][i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]] + \
                                [''+'rmse%5.3f'%rmse_01_re1[i]+'\nmae%5.3f'%mae_01_re1[i]] + \
                                [''+'rmse%5.3f'%rmse_01_re3[i]+'\nmae%5.3f'%mae_01_re3[i]]                         
                        else:
                            ncol = 3
                            sample  = [np.stack((hr_mean[i,:,:],sr_mean[i,:,:],sr_mean_ml[i,:,:]),axis=0),
                                       np.stack((hr_99per[i,:,:],sr_99per[i,:,:],sr_99per_ml[i,:,:]),axis=0),
                                       np.stack((hr_01per[i,:,:],sr_01per[i,:,:],sr_01per_ml[i,:,:]),axis=0)
                                       ]
                            sample_dif  = [np.stack((hr_mean[i,:,:],
                                                     sr_mean[i,:,:]-hr_mean[i,:,:],
                                                     sr_mean_ml[i,:,:]-hr_mean[i,:,:]),axis=0),
                                           np.stack((hr_99per[i,:,:],
                                                     sr_99per[i,:,:]-hr_99per[i,:,:],
                                                     sr_99per_ml[i,:,:]-hr_99per[i,:,:]),axis=0),
                                           np.stack((hr_01per[i,:,:],
                                                     sr_01per[i,:,:]-hr_01per[i,:,:],
                                                     sr_01per_ml[i,:,:]-hr_01per[i,:,:]),axis=0)
                                       ]
                            title = ['Reference','SR_en','MLR'] + \
                                ['' for it in range((len(sample)-1)*3)]
                            title_dif = ['Reference','SR_en-Ref','MLR-Ref'] + \
                                ['' for it in range((len(sample)-1)*3)]
                            # txt = ['Mean'] + \
                            #     [''+'ssim%5.3f'%ssim_m_ep[i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                            #     ['99per'] + \
                            #     [''+'ssim%5.3f'%ssim_99_ep[i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                            #     ['01per'] + \
                            #     [''+'ssim%5.3f'%ssim_01_ep[i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                            #     [''+'ssim%5.3f'%ssim_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]]
                            txt = ['Mean'] + \
                                [''+'rmse%5.3f'%metrics['rmse_m'][iepo][i]+'\nmae%5.3f'%metrics['mae_m'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_m_ml[i]+'\nmae%5.3f'%mae_m_ml[i]] + \
                                ['99per'] + \
                                [''+'rmse%5.3f'%metrics['rmse_99'][iepo][i]+'\nmae%5.3f'%metrics['mae_99'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_99_ml[i]+'\nmae%5.3f'%mae_99_ml[i]] + \
                                ['01per'] + \
                                [''+'rmse%5.3f'%metrics['rmse_01'][iepo][i]+'\nmae%5.3f'%metrics['mae_01'][iepo][i]] + \
                                [''+'rmse%5.3f'%rmse_01_ml[i]+'\nmae%5.3f'%mae_01_ml[i]]
                                
                        nrow = len(sample)
                        sample = np.concatenate(sample, axis=0)
                        sample_dif = np.concatenate(sample_dif, axis=0)
                        # title = ['Refernce:mean','SR_en:mean ','MLR:mean','Bilinear:mean','Nearest:mean',
                        #          'Reference:99per','SR_en:99per ','MLR:99per','Bilinear:99per','Nearest:99per',
                        #          'Reference:01per','SR_en:01per ','MLR:01per','Bilinear:01per','Nearest:01per',]
                        # txt = ['Mean'] + \
                        #     [''+'MAE=%5.3f'%metrics['mae_m'][iepo][i]+'\nRMSE=%5.3f'%metrics['rmse_m'][iepo][i]] + \
                        #     [''+'MAE=%5.3f'%mae_m_ml[i]+'\nRMSE=%5.3f'%rmse_m_ml[i]] + \
                        #     [''+'MAE=%5.3f'%mae_m_re2[i]+'\nRMSE=%5.3f'%rmse_m_re2[i]] + \
                        #     [''+'MAE=%5.3f'%mae_m_re3[i]+'\nRMSE=%5.3f'%rmse_m_re3[i]] + \
                        #     ['99per'] + \
                        #     [''+'MAE=%5.3f'%metrics['mae_99'][iepo][i]+'\nRMSE=%5.3f'%metrics['rmse_99'][iepo][i]] + \
                        #     [''+'MAE=%5.3f'%mae_99_ml[i]+'\nRMSE=%5.3f'%rmse_99_ml[i]] + \
                        #     [''+'MAE=%5.3f'%mae_99_re2[i]+'\nRMSE=%5.3f'%rmse_99_re2[i]] + \
                        #     [''+'MAE=%5.3f'%mae_99_re3[i]+'\nRMSE=%5.3f'%rmse_99_re3[i]] + \
                        #     ['01per'] + \
                        #     [''+'MAE=%5.3f'%metrics['mae_01'][iepo][i]+'\nRMSE=%5.3f'%metrics['rmse_01'][iepo][i]] + \
                        #     [''+'MAE=%5.3f'%mae_01_ml[i]+'\nRMSE=%5.3f'%rmse_01_ml[i]] + \
                        #     [''+'MAE=%5.3f'%mae_01_re2[i]+'\nRMSE=%5.3f'%rmse_01_re2[i]] + \
                        #     [''+'MAE=%5.3f'%mae_01_re3[i]+'\nRMSE=%5.3f'%rmse_01_re3[i]]

                    clim_chl = [clim_m[ichl]]*ncol+[clim_99[ichl]]*ncol+[clim_01[ichl]]*ncol
                    unit = [unit_suv[ichl]]*len(sample)
                    subsize = [2.0,1.6]
                    kbar = 1  # type of colorbar
                    figname = out_path+"99per_c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+"_ax%d_kb%d.png"%(kax,kbar)
                    plt_pcolorbar_list(lon,lat,sample,figname,subsize = subsize,cmap = 'coolwarm',
                                       clim=clim_chl,kbar=kbar,unit=unit,title=title,
                                       nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 
                    
                    ncol = int(len(sample)/nrow+0.5)
                    clim_chl = [clim_m[ichl]]+[clim_dif[ichl]]*(ncol-1)+\
                            [clim_99[ichl]]+[clim_dif[ichl]]*(ncol-1)+\
                            [clim_01[ichl]]+[clim_dif[ichl]]*(ncol-1)
                    subsize = [1.7,1.6]
                    kbar = 7  # type of colorbar
                    figname = out_path+"99per_c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+"_ax%d_kb%d_dif.png"%(kax,kbar)
                    plt_pcolorbar_list(lon,lat,sample_dif,figname,subsize = subsize,cmap = 'coolwarm',
                                       clim=clim_chl,kbar=kbar,unit=unit,title=title_dif,
                                       nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 
                                        

                    # clim_chl0 = [clim[ichl]]+[clim_dif[ichl]]*(ncol-1) # first ref, rest for differences
                    # clim_chl = []
                    # for _ in range(nrow):
                    #     clim_chl.extend(clim_chl0)
                    # unit = [unit_suv[ichl]]*len(sample)
                    # subsize = [1.7,1.6]
                    # kbar = 2  # type of colorbar
                    # figname = out_path+"99per_c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+"_ax%d_kb%d_dif.png"%(kax,kbar)
                    # plt_pcolorbar_list(lon,lat,sample_dif,figname,subsize = subsize,cmap = 'coolwarm',
                    #                    clim=clim_chl,kbar=kbar,unit=unit,title=title_dif,
                    #                    nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 
                                        
                    
            iepo = iepo + 1