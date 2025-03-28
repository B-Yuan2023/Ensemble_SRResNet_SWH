#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta # , date

from funs_prepost import (plot_line_list,plt_pcolorbar_list,var_normalize,ssim_tor,
                          ssim_skimg,plotsubs_line_list)
from funs_sites import select_sta

import importlib
mod_name= 'par534e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    ivar_lr = mod_para.ivar_lr
    varm_hr = mod_para.varm_hr

    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
 
    if hasattr(mod_para, 'rep'):  # if input has list rep
        rep = mod_para.rep
    else:
        nrep = mod_para.nrep
        rep = list(range(0,nrep))
    
    # nrep = mod_para.nrep
    # # rep = list(range(0,nrep))
    rep = [0]
    
    # epoc_num =[100] #
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-20  # 31
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average

    kp_pnt = 1  # key to plot comparison for points at selected period 
    kp_2D = 0   # key to plot comparison for 2D map at selected period 
    kp_2D_ord = 1  # for 2d plot dim0 order: 0 first time (in a row) next model; 1 first model next time
    nep_skip = 20  # no. of skipped epochs for plotting 
    
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    
    # select a range of data for testing 
    # tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    tlim = [datetime(2021,11,29),datetime(2021,12,2)] # for manuscript stations
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    dt = 3

    # for 2d time series in manuscript
    # tlim = [datetime(2021,11,29,3),datetime(2021,11,30,3)] 
    # dt = 6
    
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d') + '_t%d'%dt
    Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    tuser0 = [(tlim[0] + timedelta(hours=x*dt)) for x in range(0,Nt)]
    tshift = 0 # in hour
    tuser = [(tlim[0] + timedelta(hours=x*dt)) for x in range(tshift,Nt+tshift)] # time shift for numerical model
    
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_ens/'
    os.makedirs(out_path0, exist_ok=True)   

    # ========================================================================
    
    # load sr from linear regression for the selected time range
    # saved data for selected period using test_tuse.py
    ml_suf = '_md0'
    ml_mod_name= 'par534e_md0'  # should match with srresnet mod_name
    ml_path_par = '/work/gg0028/g260218/GB_output_interp/wave_cmems_blacksea/ml_traditional/'
    ml_path0 = ml_path_par+'results_pnt/'+'S'+str(opt.up_factor)+ '_'+ ml_mod_name +'/'
    filename = ml_path0 + 'sr'+tstr+'_re%d' % (0) +'.npz'
    if not os.path.isfile(filename): 
        sys.exit('hr file not saved!')
    else:
        datald = np.load(filename) # load
        sr_all_mlr = datald['sr_all']
    sr_all_mlr_nm = var_normalize(sr_all_mlr,varmaxmin=varm_hr)

    # ========================================================================
    
    # load original hr and interpolated hr for the selected time range
    # saved data for selected period using test_epo_ave_tuse.py
    filename = out_path0 + 'hr'+tstr+'.npz'
    if not os.path.isfile(filename): 
        # np.savez(filename,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
        print('hr file not saved!',file=sys.stderr)
    else:
        datald = np.load(filename) # load
        sorted(datald.files)
        hr_all = datald['hr_all']
        lat = datald['lat']
        lon = datald['lon']
        # tuser0 = datald['t']
    hr_all_nm = var_normalize(hr_all,varmaxmin=varm_hr)

    # interpolation only applied to cases input/output channels are the same
    if ivar_hr==ivar_lr:  
        filename = out_path0 + 'hr'+tstr+'_interp'+'.npz'
        if not os.path.isfile(filename):
            # np.savez(filename,hr_re1_all=hr_re1_all,hr_re2_all=hr_re2_all,hr_re3_all=hr_re3_all,lat=lat,lon=lon,t=tuser0)
            print('hr file from interpolation not saved!',file=sys.stderr)
        else:
            datald = np.load(filename) # load
            hr_re1_all = datald['hr_re1_all']
            hr_re2_all = datald['hr_re2_all']
            hr_re3_all = datald['hr_re3_all']
        hr_re1_all_nm = var_normalize(hr_re1_all,varmaxmin=varm_hr)
        hr_re2_all_nm = var_normalize(hr_re2_all,varmaxmin=varm_hr)
        hr_re3_all_nm = var_normalize(hr_re3_all,varmaxmin=varm_hr)
        
    # ========================================================================

    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]

    clim = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,5.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    clim_dif = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[-1,1],[-1,1],[-0.5,0.5],[-1,1.]]  # diff in ssh,u,v,uw,vw,swh,pwp
    
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']    
    for irep in rep:
        print(f'Repeat {irep}')
        print('--------------------------------')
    
        out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens'+ml_suf+'/'
        os.makedirs(out_path, exist_ok=True)
    
        iepo = 0
        for epoch in epoc_num:
            iepo = iepo + 1
            if kp_2D == 1 and iepo%nep_skip == 0:
                
                # load super-resolution hr (single & ensemble) for the selected time range
                # saved using test_epo_ave_tuser.py
                filename = out_path0+'sr'+tstr+'_re%d_ep%d_%d' % (irep,epoch,epoc_num[0]) +'.npz'
                if not os.path.isfile(filename): 
                    # np.savez(filename,sr_all=sr_all,sr_all_ep=sr_all_ep,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
                    sys.exit('hr file not saved!')
                else:
                    datald = np.load(filename) # load
                    sr_all = datald['sr_all']
                    sr_all_ep = datald['sr_all_ep']
                    # hr_all1 = datald['hr_all']
                    # np.allclose(hr_all, hr_all1, equal_nan=True) # checked
                sr_all_ep_nm = var_normalize(sr_all_ep,varmaxmin=varm_hr)
                
                # plot comparison for 2D field
                # rmse_sr = np.nanmean((sr_all - hr_all) ** 2,axis=(2,3))**(0.5)
                # mae_sr = np.nanmean(abs(sr_all - hr_all),axis=(2,3))
                rmse_sr_mlr = np.nanmean((sr_all_mlr - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_sr_mlr = np.nanmean(abs(sr_all_mlr - hr_all),axis=(2,3))
                rmse_sr_ep = np.nanmean((sr_all_ep - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_sr_ep = np.nanmean(abs(sr_all_ep - hr_all),axis=(2,3))

                # ssim_sr_ep_skimg = ssim_skimg(sr_all_ep_nm, hr_all_nm) # checked, the same upto 7 digits
                ssim_sr_ep = ssim_tor(sr_all_ep_nm, hr_all_nm)
                ssim_sr_mlr = ssim_tor(sr_all_mlr_nm, hr_all_nm)
                
                if ivar_hr==ivar_lr:  
                    rmse_re1 = np.nanmean((hr_re1_all - hr_all) ** 2,axis=(2,3))**(0.5)
                    mae_re1 = np.nanmean(abs(hr_re1_all - hr_all),axis=(2,3))
                    rmse_re2 = np.nanmean((hr_re2_all - hr_all) ** 2,axis=(2,3))**(0.5)
                    mae_re2 = np.nanmean(abs(hr_re2_all - hr_all),axis=(2,3))
                    rmse_re3 = np.nanmean((hr_re3_all - hr_all) ** 2,axis=(2,3))**(0.5)
                    mae_re3 = np.nanmean(abs(hr_re3_all - hr_all),axis=(2,3))
                    ssim_re1 = ssim_tor(hr_re1_all_nm, hr_all_nm)
                    ssim_re2 = ssim_tor(hr_re2_all_nm, hr_all_nm)
                    ssim_re3 = ssim_tor(hr_re3_all_nm, hr_all_nm)
                    
                # save error info to csv 
                for i in range(nchl_o):
                    ichl = ivar_hr[i]
                    filename = out_path+"c%d_re%d_ep%d_%d"% (ichl,irep,epoch,epoch0)+tstr+"_err.csv"
                    # metric_mds = np.concatenate([mae_sr_ep[:,i],rmse_sr_ep[:,i],
                    #                              mae_sr_mlr[:,i],rmse_sr_mlr[:,i],
                    #                              mae_re2[:,i],rmse_re2[:,i],
                    #                              mae_re3[:,i],rmse_re3[:,i]],axis=1)
                    # header = 'mae_sr_ep,rmse_sr_ep,mae_sr_mlr,rmse_sr_mlr,mae_re2,rmse_re2,mae_re3,rmse_re3,'
                    # np.savetxt(filename,metric_mds, delimiter=',', header=header, comments="")
                    if ivar_hr==ivar_lr:  
                        df = pd.DataFrame({'Time':tuser0,
                                           'rmse_sr_ep':rmse_sr_ep[:,i],'rmse_sr_mlr':rmse_sr_mlr[:,i],
                                           'rmse_re1':rmse_re1[:,i],'rmse_re2':rmse_re2[:,i],'rmse_re3':rmse_re3[:,i],
                                           'mae_sr_ep':mae_sr_ep[:,i],'mae_sr_mlr':mae_sr_mlr[:,i],
                                           'mae_re1':mae_re1[:,i],'mae_re2':mae_re2[:,i],'mae_re3':mae_re3[:,i],
                                           'ssim_sr_ep':ssim_sr_ep[:,i],'ssim_sr_mlr':ssim_sr_mlr[:,i],
                                           'ssim_re1':ssim_re1[:,i],'ssim_re2':ssim_re2[:,i],'ssim_re3':ssim_re3[:,i],})
                    else:
                        df = pd.DataFrame({'Time':tuser0,
                                           'rmse_sr_ep':rmse_sr_ep[:,i],'rmse_sr_mlr':rmse_sr_mlr[:,i],
                                           'mae_sr_ep':mae_sr_ep[:,i],'mae_sr_mlr':mae_sr_mlr[:,i],
                                           'ssim_sr_ep':ssim_sr_ep[:,i],'ssim_sr_mlr':ssim_sr_mlr[:,i],})
                    df.to_csv(filename, index=False)
                    
                loc_txt = [0.01,0.90] # location of text
                nt_sub = 5  # plot nt_sub times in one row 
                nfig = -(-Nt//nt_sub) # ceiling
                kax = 1   # turn ax off or not, 1 off. 
                for ifig in range(nfig):
                    if nt_sub*(ifig+1)<Nt:
                        ind = np.arange(nt_sub*ifig,nt_sub*(ifig+1), 1).tolist()
                    else:
                        ind = np.arange(nt_sub*ifig,Nt, 1).tolist()
                    
                    for i in range(nchl_o):
                        ichl = ivar_hr[i]
                        if kp_2D_ord==0:   # show data first for time (in row) next for model
                            if ivar_hr==ivar_lr:  
                                sample  = [hr_all[ind,i,:,:],sr_all_ep[ind,i,:,:],
                                           sr_all_mlr[ind,i,:,:],
                                           hr_re1_all[ind,i,:,:],
                                           # hr_re2_all[ind,i,:,:],  # use RBF instead
                                           hr_re3_all[ind,i,:,:]]
                                sample_dif  = [hr_all[ind,i,:,:],
                                               sr_all_ep[ind,i,:,:]-hr_all[ind,i,:,:],
                                               sr_all_mlr[ind,i,:,:]-hr_all[ind,i,:,:],
                                               hr_re1_all[ind,i,:,:]-hr_all[ind,i,:,:],
                                               # hr_re2_all[ind,i,:,:],  # use RBF instead
                                               hr_re3_all[ind,i,:,:]-hr_all[ind,i,:,:]]
                                title = ['Reference ' for it in ind] + \
                                    ['SR_en ' for it in ind] + \
                                    ['MLR ' for it in ind] + \
                                    ['RBFlinear ' for it in ind]+ \
                                    ['Nearest ' for it in ind]
                                title_dif = ['Reference ' for it in ind] + \
                                    ['SR_en-Ref ' for it in ind] + \
                                    ['MLR-Ref ' for it in ind] + \
                                    ['RBFlinear-Ref ' for it in ind]+ \
                                    ['Nearest-Ref ' for it in ind]
                                txt = [''+ tuser0[it].strftime('%Y%m%d %H') for it in ind] + \
                                    [''+'MAE=%5.3f'%mae_sr_ep[it,i]+'\nRMSE=%5.3f'%rmse_sr_ep[it,i] for it in ind] + \
                                    [''+'MAE=%5.3f'%mae_sr_mlr[it,i]+'\nRMSE=%5.3f'%rmse_sr_mlr[it,i] for it in ind] + \
                                    [''+'MAE=%5.3f'%mae_re1[it,i]+'\nRMSE=%5.3f'%rmse_re1[it,i]  for it in ind]+ \
                                    [''+'MAE=%5.3f'%mae_re3[it,i]+'\nRMSE=%5.3f'%rmse_re3[it,i] for it in ind]
                            else:
                                sample  = [hr_all[ind,i,:,:],sr_all_ep[ind,i,:,:],sr_all_mlr[ind,i,:,:]]
                                sample_dif  = [hr_all[ind,i,:,:],
                                               sr_all_ep[ind,i,:,:]-hr_all[ind,i,:,:],
                                               sr_all_mlr[ind,i,:,:]-hr_all[ind,i,:,:]]
                                title = ['Reference ' for it in ind] + \
                                    ['SR_en ' for it in ind] + \
                                    ['MLR ' for it in ind] 
                                title_dif = ['Reference ' for it in ind] + \
                                    ['SR_en-Ref ' for it in ind] + \
                                    ['MLR-Ref ' for it in ind] 
                                txt = [''+ tuser0[it].strftime('%Y%m%d %H') for it in ind] + \
                                    [''+'MAE=%5.3f'%mae_sr_ep[it,i]+'\nRMSE=%5.3f'%rmse_sr_ep[it,i] for it in ind] + \
                                    [''+'MAE=%5.3f'%mae_sr_mlr[it,i]+'\nRMSE=%5.3f'%rmse_sr_mlr[it,i] for it in ind]
                            nrow = len(sample)
                            sample = np.concatenate(sample, axis=0)
                            sample_dif = np.concatenate(sample_dif, axis=0)

                        else:   # show data first for model (in row) next for time
                            if ivar_hr==ivar_lr:  
                                # dim0 order first model next time, [md0[0],md1[0]...], [md0[1],md1[1]...]...
                                sample  = [np.stack([hr_all[it,i,:,:],sr_all_ep[it,i,:,:],
                                                     sr_all_mlr[it,i,:,:],
                                                     hr_re1_all[it,i,:,:],
                                                     hr_re3_all[it,i,:,:]],axis=0) for it in ind]
                                sample_dif  = [np.stack([hr_all[it,i,:,:],
                                                         sr_all_ep[it,i,:,:]-hr_all[it,i,:,:],
                                                         sr_all_mlr[it,i,:,:]-hr_all[it,i,:,:],
                                                         hr_re1_all[it,i,:,:]-hr_all[it,i,:,:],
                                                         hr_re3_all[it,i,:,:]-hr_all[it,i,:,:]],axis=0) for it in ind]
                                title = ['Reference','SR_en','MLR','RBFlinear','Nearest'] + \
                                    ['' for it in range((len(ind)-1)*5)]
                                title_dif = ['Reference','SR_en-Ref','MLR-Ref','RBFlinear-Ref','Nearest-Ref'] + \
                                    ['' for it in range((len(ind)-1)*5)]
                                # txt = [[''+ tuser0[it].strftime('%Y%m%d %H')] + \
                                #         [''+'ssim%5.3f'%ssim_sr_ep[it,i]+'\nmae%5.3f'%mae_sr_ep[it,i]] + \
                                #         [''+'ssim%5.3f'%ssim_sr_mlr[it,i]+'\nmae%5.3f'%mae_sr_mlr[it,i]] + \
                                #         [''+'ssim%5.3f'%ssim_re2[it,i]+'\nmae%5.3f'%mae_re2[it,i]]+ \
                                #         [''+'ssim%5.3f'%ssim_re3[it,i]+'\nmae%5.3f'%mae_re3[it,i]]
                                #         for it in ind]
                                txt = [[''+ tuser0[it].strftime('%Y%m%d %H')] + \
                                        [''+'rmse%5.3f'%rmse_sr_ep[it,i]+'\nmae%5.3f'%mae_sr_ep[it,i]] + \
                                        [''+'rmse%5.3f'%rmse_sr_mlr[it,i]+'\nmae%5.3f'%mae_sr_mlr[it,i]] + \
                                        [''+'rmse%5.3f'%rmse_re1[it,i]+'\nmae%5.3f'%mae_re1[it,i]]+ \
                                        [''+'rmse%5.3f'%rmse_re3[it,i]+'\nmae%5.3f'%mae_re3[it,i]]
                                        for it in ind]
                            else:
                                # dim0 order first model next time, [md0[0],md1[0]...], [md0[1],md1[1]...]...
                                sample  = [np.stack([hr_all[it,i,:,:],sr_all_ep[it,i,:,:],
                                                     sr_all_mlr[it,i,:,:]],axis=0) for it in ind] 
                                sample_dif  = [np.stack([hr_all[it,i,:,:],
                                                         sr_all_ep[it,i,:,:]-hr_all[it,i,:,:],
                                                         sr_all_mlr[it,i,:,:]-hr_all[it,i,:,:]],axis=0) for it in ind] 
                                title = ['Reference','SR_en','MLR'] + \
                                    ['' for it in range((len(ind)-1)*3)]
                                title_dif = ['Reference','SR_en-Ref','MLR-Ref'] + \
                                    ['' for it in range((len(ind)-1)*3)]
                                # txt = [[''+ tuser0[it].strftime('%Y%m%d %H')] + \
                                #        [''+'ssim%5.3f'%ssim_sr_ep[it,i]+'\nmae%5.3f'%mae_sr_ep[it,i]] + \
                                #        [''+'ssim%5.3f'%ssim_sr_mlr[it,i]+'\nmae%5.3f'%mae_sr_mlr[it,i]]
                                #        for it in ind]
                                txt = [[''+ tuser0[it].strftime('%Y%m%d %H')] + \
                                        [''+'rmse%5.3f'%rmse_sr_ep[it,i]+'\nmae%5.3f'%mae_sr_ep[it,i]] + \
                                        [''+'rmse%5.3f'%rmse_sr_mlr[it,i]+'\nmae%5.3f'%mae_sr_mlr[it,i]]
                                        for it in ind]
                            txt = sum(txt, [])  # merge lists in list
                            nrow = len(sample)
                            sample = np.concatenate(sample, axis=0)
                            sample_dif = np.concatenate(sample_dif, axis=0)
                            # title = [['Reference '] + \
                            #     ['SR_en '] + \
                            #     ['MLR '] + \
                            #     ['Bilinear ']+ \
                            #     ['Nearest ']
                            #      for it in ind]
                            # title = sum(title, [])  # merge lists in list
                            # txt = [[''+ tuser0[it].strftime('%Y%m%d %H')] + \
                            #     [''+'MAE=%5.3f'%mae_sr_ep[it,i]+'\nRMSE=%5.3f'%rmse_sr_ep[it,i]] + \
                            #     [''+'MAE=%5.3f'%mae_sr_mlr[it,i]+'\nRMSE=%5.3f'%rmse_sr_mlr[it,i]] + \
                            #     [''+'MAE=%5.3f'%mae_re2[it,i]+'\nRMSE=%5.3f'%rmse_re2[it,i]]+ \
                            #     [''+'MAE=%5.3f'%mae_re3[it,i]+'\nRMSE=%5.3f'%rmse_re3[it,i]]
                            #     for it in ind]

                        clim_chl = [clim[ichl]]*len(sample)
                        unit = [unit_suv[ichl]]*len(sample)
                        # subsize = [2.5,2]
                        subsize = [2.0,1.6]
                        kbar = 5  # type of colorbar
                        tstr_u =  '_'+tuser0[ind[0]].strftime('%Y%m%d%H')+'_'+tuser0[ind[-1]].strftime('%Y%m%d%H')+ '_t%d'%dt
                        figname = out_path+"c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+tstr_u+"_ax%d_kb%d.png"%(kax,kbar)
                        # figname = out_path+"c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+tstr+"_ax%d_kb%d_f%d.png"%(kax,kbar,ifig)
                        plt_pcolorbar_list(lon,lat,sample,figname,subsize = subsize,cmap = 'coolwarm',
                                           clim=clim_chl,kbar=kbar,unit=unit,title=title,
                                           nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 

                        ncol = int(len(sample)/nrow+0.5)
                        clim_chl0 = [clim[ichl]]+[clim_dif[ichl]]*(ncol-1) # first ref, rest for differences
                        clim_chl = []
                        for _ in range(nrow):
                            clim_chl.extend(clim_chl0)
                        subsize = [1.7,1.6]
                        kbar = 2  # type of colorbar
                        figname = out_path+"c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+tstr_u+"_ax%d_kb%d_dif.png"%(kax,kbar)
                        plt_pcolorbar_list(lon,lat,sample_dif,figname,subsize = subsize,cmap = 'coolwarm',
                                           clim=clim_chl,kbar=kbar,unit=unit,title=title_dif,
                                           nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 

    # use selected stations, 3 near buoys at 10, 20, 40 m, 1 at maximam SWH
    index = np.array([[104, 22],[76, 6],[83, 20],[88, 239]])
    # ll_sta = np.array([[27.950,43.200],[27.550,42.500],[27.900,42.675],[33.375,42.800]])
    ll_sta = np.array([[43.200,27.950],[42.500,27.550],[42.675,27.900],[42.800,33.375]]) # 1st lat 2nd lon in select_sta
    sta_user = ['P'+str(ip+1) for ip in range(len(index))]
    
    #  if plot comparions for points 
    if kp_pnt == 1:
        # select several locations for comparison
        # nskp = (80,80)  # skipping grid points 
        # index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all,ivar_hr,lon,lat,nskp)
        nsta = len(index)  # number of stations 
        
        axlab = [['','ssh (m)'],['','u (m/s)'],['','v (m/s)'],
                 ['','uw (m/s)'],['','vw (m/s)'],['','swh (m)'],
                 ['','pwp (s)'],['','swh_ww (m)'],]
        # axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)'],
        #          ['Time','uw (m/s)'],['Time','vw (m/s)'],['Time','swh (m)'],
        #          ['Time','pwp (s)'],['Time','swh_ww (m)'],]
        # line_sty=['k.','b','r-','m-','g-','c']
        # line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
        # line_sty=['ko','-','-','-','-','-'] # 'kv',
        line_sty=['None','-','-','-','-','-'] # 'kv',
        # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # default color cycle
        line_col=['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        marker = ['.']+['None']*5  # 'o', '.'
        
        for irep in rep:
            print(f'Repeat {irep}')
            print('--------------------------------')
            
            out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens'+ml_suf+'/'
            os.makedirs(out_path, exist_ok=True)
        
            iepo = 0
            for epoch in epoc_num:
                iepo = iepo + 1
                if iepo%nep_skip == 0:
                    # load super-resolution hr (single & ensemble) for the selected time range
                    filename = out_path0+'sr'+tstr+'_re%d_ep%d_%d' % (irep,epoch,epoc_num[0]) +'.npz'
                    if not os.path.isfile(filename): 
                        # np.savez(filename,sr_all=sr_all,sr_all_ep=sr_all_ep,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
                        print('sr file not saved!', file=sys.stderr)
                    else:
                        datald = np.load(filename) # load
                        sr_all = datald['sr_all']
                        sr_all_ep = datald['sr_all_ep']
                    
                    sr_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                    sr_sta_ep = np.zeros(shape=(nsta,nchl_o,Nt))
                    sr_sta_mlr = np.zeros(shape=(nsta,nchl_o,Nt))
                    hr_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                    
                    if ivar_hr==ivar_lr: 
                        hr_res1_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                        hr_res2_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                        hr_res3_sta = np.zeros(shape=(nsta,nchl_o,Nt))
        
                    for it in range(0,Nt):
                        for ip in range(nsta):
                            sr_sta[ip,:,it]=sr_all[it,:,index[ip,0],index[ip,1]]
                            sr_sta_ep[ip,:,it]= sr_all_ep[it,:,index[ip,0],index[ip,1]]# epoch averaged sr at stations
                            sr_sta_mlr[ip,:,it]= sr_all_mlr[it,:,index[ip,0],index[ip,1]]# linear regression sr at stations
                            hr_sta[ip,:,it]=hr_all[it,:,index[ip,0],index[ip,1]]
                            if ivar_hr==ivar_lr: 
                                hr_res1_sta[ip,:,it]=hr_re1_all[it,:,index[ip,0],index[ip,1]]
                                hr_res2_sta[ip,:,it]=hr_re2_all[it,:,index[ip,0],index[ip,1]]
                                hr_res3_sta[ip,:,it]=hr_re3_all[it,:,index[ip,0],index[ip,1]]
                    
                    # estimate the error for each point for the selected period
                    rmse_sr_mlr = np.nanmean((sr_sta_mlr - hr_sta) ** 2,axis=(2))**(0.5)
                    mae_sr_mlr = np.nanmean(abs(sr_sta_mlr - hr_sta),axis=(2))
                    rmse_sr_ep = np.nanmean((sr_sta_ep - hr_sta) ** 2,axis=(2))**(0.5)
                    mae_sr_ep = np.nanmean(abs(sr_sta_ep - hr_sta),axis=(2))
                    if ivar_hr==ivar_lr:  
                        rmse_re1 = np.nanmean((hr_res1_sta - hr_sta) ** 2,axis=(2))**(0.5)
                        mae_re1 = np.nanmean(abs(hr_res1_sta - hr_sta),axis=(2))
                        rmse_re2 = np.nanmean((hr_res2_sta - hr_sta) ** 2,axis=(2))**(0.5)
                        mae_re2 = np.nanmean(abs(hr_res2_sta - hr_sta),axis=(2))
                        rmse_re3 = np.nanmean((hr_res3_sta - hr_sta) ** 2,axis=(2))**(0.5)
                        mae_re3 = np.nanmean(abs(hr_res3_sta - hr_sta),axis=(2))

                    # save error info to csv 
                    for i in range(nchl_o):
                        ichl = ivar_hr[i]
                        filename = out_path+"c%d_re%d_ep%d_%d"% (ichl,irep,epoch,epoch0)+tstr+"_err_sta.csv"
                        if ivar_hr==ivar_lr:  
                            df = pd.DataFrame({'Time':sta_user,
                                               'rmse_sr_ep':rmse_sr_ep[:,i],'rmse_sr_mlr':rmse_sr_mlr[:,i],
                                               'rmse_re1':rmse_re1[:,i],'rmse_re2':rmse_re2[:,i],'rmse_re3':rmse_re3[:,i],
                                               'mae_sr_ep':mae_sr_ep[:,i],'mae_sr_mlr':mae_sr_mlr[:,i],
                                               'mae_re1':mae_re1[:,i],'mae_re2':mae_re2[:,i],'mae_re3':mae_re3[:,i],
                                               })
                        else:
                            df = pd.DataFrame({'Time':sta_user,
                                               'rmse_sr_ep':rmse_sr_ep[:,i],'rmse_sr_mlr':rmse_sr_mlr[:,i],
                                               'mae_sr_ep':mae_sr_ep[:,i],'mae_sr_mlr':mae_sr_mlr[:,i],
                                               })
                        df.to_csv(filename, index=False)


                    # plot comparison for locations
                    for i in range(len(ivar_hr)):
                        data_lst_np = []  # store data for all stations to plot in 1 figure
                        ich = ivar_hr[i]
                        line_sty=['ko','-','-','-','-','-'] # 'kv',
                        for ip in range(nsta):
                            var_sta = hr_sta[ip,i,:]
                            if not np.isnan(var_sta).all():  # check if not all nan
                                var = sr_sta[ip,i,:]
                                var_ep = sr_sta_ep[ip,i,:]
                                var_mlr = sr_sta_mlr[ip,i,:]
                                if ivar_hr==ivar_lr: 
                                    var_res1 = hr_res1_sta[ip,i,:] # bicubic,now RBFlinear
                                    var_res2 = hr_res2_sta[ip,i,:] # binear 
                                    var_res3 = hr_res3_sta[ip,i,:] # nearest
                                    data_lst = [var_sta,var_ep,var_mlr,var_res1,var_res3]  # var_res1,
                                    leg = ['Reference','SR_en','MLR','RBFlinear','Nearest']  # 'Bicubic',
                                else:
                                    data_lst = [var_sta,var_ep,var_mlr]  
                                    leg = ['Reference','SR_en','MLR']  

                                data_lst_np.append(data_lst)
                                time_lst = [tuser0] * len(data_lst)
                                # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+ sta_user[ip]+'.png'
                                # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoch) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                                figname = out_path+"/c%d_re%d_ep%d_%d" % (ich,irep,epoch,epoc_num[0]) +tstr+ sta_user[ip]+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                                figsize = [3.7,3.0]  # A4 8.3*11.7
                                lloc = 9  # legend location, 9-upper center
                                legloc = [0.5,1.3]
                                plot_line_list(time_lst,data_lst,tlim,figname,
                                               figsize,axlab=axlab[ich],leg=leg,
                                               leg_col=3,lloc=lloc,legloc=legloc,
                                               line_sty=line_sty,capt=capt_all[ip])

                        line_sty=['None','-','-','-','-','-'] # 'kv',
                        # time_lst = [tuser0] * len(data_lst_np)
                        time_lst = [[tuser0] * len(data_lst_np[0])] * len(data_lst_np)  # nested list, dim as data_lst_np
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ich,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        subsize = [3.7,2.5]  # A4 8.3*11.7
                        lloc = 9  # legend location, 9-upper center
                        legloc = [0.5,1.03]
                        ylim = [[0,3.2],[0,3.2],[0,3.2],[0,6.4]] # ylimit for points
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=2,title=sta_user,axlab=axlab[ich],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all)