#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import os
import sys
import numpy as np

from funs_prepost import plot_distri,plot_mod_vs_obs,plotsubs_mod_vs_obs,plotsubs_line_list
from funs_sites import select_sta
from datetime import datetime

import importlib
mod_name= 'par534e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

# rearrange the time steps in original order for arrays
def rearrange_1d(var,ntpf_use):
    # var: 1d array, with first dim being time
    # ntpf_use: # no. of used time steps in 1 nc file
    Nt_test = len(var)
    nfile_test = int(Nt_test/ntpf_use)
    nt_err = Nt_test%int(ntpf_use)  # mismatched no. of testing samples
    var1 = var.copy()
    for i in range(nfile_test):
        idt0 = int(nt_err + i*ntpf_use)
        idt1 = int(nt_err*2 + i*ntpf_use)
        idt2 = int(nt_err + (i+1)*ntpf_use)
        var1[idt0:idt1] = var[(idt2-nt_err):idt2]
        var1[idt1:idt2] = var[idt0:(idt2-nt_err)]
    return var1

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    ivar_lr = mod_para.ivar_lr
    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    rtra = mod_para.rtra
    tuse0 = mod_para.t_hr
    
    # nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    rep = [0]
    # epoc_num =[100] #
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-20  # 30
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average

    kp_pnt = 0  # key to plot comparison for points in test data
    kp_pnt_t = 1 # key to plot time series comparison for points in test data

    nep_skip = 20  # no. of skipped epochs for plotting 
    
    # for traditional ml models
    ml_suf = '_md0'
    ml_mod_name= 'par534e_md0'  # should match with srresnet mod_name: par55e & par55e_md0, par534e & par534e_md0
    ml_path_par = '/work/gg0028/g260218/GB_output_interp/wave_cmems_blacksea/ml_traditional/'

    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]

    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']    

    for k in range(nchl_o): # nchl_o, save for all time steps. 
        ichl = ivar_hr[k]
        
        # load sr from linear regression for whole testset, saved using test.py
        irep = 0
        ml_path0 = ml_path_par+'results_test/'+'S'+str(opt.up_factor)+'_'+ml_mod_name+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        filename = ml_path0 + "c%d_sr_all" % (ichl)+'.npz'
        if not os.path.isfile(filename): 
            sys.exit('ml sr file not saved!')
        else:
            datald = np.load(filename) # load
            sr_all_mlr = datald['v0']    
    
        # load original hr for whole testset, saved using cal_metrics_intp.py
        out_path0 = path_par+'results_test/'+'S%d_mk%d'%(opt.up_factor,kmask)+'/'+ var_hr[0]+'/'
        filename = out_path0 + 'c%d_'%(ichl)+'hr_all'+'_train%4.2f'%(rtra)+'.npz'
        if not os.path.isfile(filename): 
            print('hr file not saved!',file=sys.stderr)
        else:
            datald = np.load(filename) # load
            hr_all = datald['v0']  # NT,H,W
            lat = datald['lat']
            lon = datald['lon']
            Nt = len(hr_all)
        
        #  if plot comparions for points 
        if kp_pnt == 1 or kp_pnt_t==1:
            # use selected stations, 3 near buoys at 10, 20, 40 m, 1 at maximam SWH
            index = np.array([[104, 22],[76, 6],[83, 20],[88, 239]])
            # ll_sta = np.array([[27.950,43.200],[27.550,42.500],[27.900,42.675],[33.375,42.800]])
            ll_sta = np.array([[43.200,27.950],[42.500,27.550],[42.675,27.900],[42.800,33.375]]) # 1st lat 2nd lon in select_sta
            sta_user = ['P'+str(ip+1) for ip in range(len(index))]
            
            # # select several locations for comparison
            # nskp = (80,80)  # skipping grid points 
            # hr_all4d = np.array(hr_all).reshape(-1,1,hr_all.shape[1],hr_all.shape[2]) # [Nt,c,H,W]
            # index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all4d,ivar_hr,lon,lat,nskp)
            nsta = len(index)  # number of stations 
        
        # interpolation only applied to cases input/output channels are the same
        if ivar_hr==ivar_lr:  
            # nearest interpolation
            filename = out_path0 + 'c%d_'%(ichl)+'hr_restore3_all'+'_train%4.2f'%(rtra)+'.npz'
            datald = np.load(filename) # load
            hr_re3_all = datald['v0']
    
            # # bilinear interpolation
            # filename = out_path0 + 'c%d_'%(ichl)+'hr_restore2_all'+'_train%4.2f'%(rtra)+'.npz'
            # datald = np.load(filename) # load
            # hr_re2_all = datald['v0']

            # RBFlinear interpolation
            filename = out_path0 + 'c%d_'%(ichl)+'hr_restore1_all'+'_train%4.2f'%(rtra)+'.npz'
            datald = np.load(filename) # load
            hr_re1_all = datald['v0']

        # load srresnet ensemble results (only saved for 71-100, 81-100 and 91-100)
        for irep in rep:
            print(f'Repeat {irep}')
            print('--------------------------------')
        
            out_path = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens'+ml_suf+'/'
            os.makedirs(out_path, exist_ok=True)
            
            iepo = 0
            for epoch in epoc_num:
                iepo = iepo + 1
                if iepo%nep_skip == 0:
                    # load super-resolution hr (ensemble) for whole testset
                    # saved using test_epo_ave.py
                    out_path0 = path_par+'results_test/'+'SRF_'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ave/'
                    filename = out_path0+"c%d_sr_all_ep_epoch%d_%d" % (ichl,epoch0,epoch)+'.npz'
                    if not os.path.isfile(filename): 
                        sys.exit('sr file not saved!')
                    else:
                        datald = np.load(filename) # load
                        sr_all_ep = datald['v0']
                    
                    # plot the distribution for all data
                    var1 = hr_all.flatten()
                    var2 = sr_all_ep.flatten()
                    var3 = sr_all_mlr.flatten()
                    
                    max_v1,min_v1 = np.nanmax(var1), np.nanmin(var1)
                    max_v2,min_v2 = np.nanmax(var2), np.nanmin(var2)
                    max_v3,min_v3 = np.nanmax(var3), np.nanmin(var3)
                    
                    if ivar_hr==ivar_lr:
                        var4 = hr_re1_all.flatten()  # use RBFlinear instead bilinear
                        var5 = hr_re3_all.flatten()
                        max_v4,min_v4 = np.nanmax(var4), np.nanmin(var4)
                        max_v5,min_v5 = np.nanmax(var5), np.nanmin(var5)
                        
                        combined_ind= np.array([[max_v1,min_v1],[max_v2,min_v2],
                                       [max_v3,min_v3],[max_v4,min_v4],[max_v5,min_v5]])
                        leg = ['Reference','SR_en', 'MLR','RBFlinear','Nearest'] 
                        # leg = ['GT'+'('+'%4.2f,%4.2f'%(min_v1,max_v1)+')',
                        #        'SRResNet'+'('+'%4.2f,%4.2f'%(min_v2,max_v2)+')',
                        #        'MLR'+'('+'%4.2f,%4.2f'%(min_v3,max_v3)+')',
                        #        'Bilinear'+'('+'%4.2f,%4.2f'%(min_v4,max_v4)+')', 
                        #        'Nearest'+'('+'%4.2f,%4.2f'%(min_v5,max_v5)+')'] 
                        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
                               var3[~np.isnan(var3)],var4[~np.isnan(var4)],
                               ]
                    else:
                        combined_ind= np.array([[max_v1,min_v1],[max_v2,min_v2],
                                       [max_v3,min_v3]])
                        leg = ['Reference','SR_en', 'MLR'] 
                        var = [var1[~np.isnan(var1)],var2[~np.isnan(var2)],
                               var3[~np.isnan(var3)]
                               ]
                        
                    ofname = "dist_c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0])+'_maxmin_mods_ml'+'.csv'
                    np.savetxt(out_path + ofname, combined_ind,fmt='%f,') # ,delimiter=","
                    
                    unit_var = unit_suv[ichl]
                    # plot distribution of reconstructed vs target, all data, histogram
                    axlab = (unit_var,'Frequency','')
                    lim = [0,6]
                    nbin= 20
                    xlim = [0,5]
                    ylim = [0,0.30]
                    figname = out_path+"dist_c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0])+'_all'+'.png'
                    plot_distri(var,figname,nbin=nbin,lim=lim,axlab=axlab,leg=leg,
                                   figsize=(7.4, 2.5), fontsize=12,capt='(a)',
                                   xlim=xlim,ylim=ylim)


                    #  if plot comparions for points 
                    if kp_pnt == 1: 
                        data_lst_np,data_ref_np = [],[]
                        txt_lst_np = []
                        ip_count = 0
                        for ip in range(nsta):
                            hr_sta = hr_all[:,index[ip,0],index[ip,1]]
                            if not np.isnan(hr_sta).all():  # check if not all nan
                                sr_sta_ep = sr_all_ep[:,index[ip,0],index[ip,1]]# epoch averaged sr at stations
                                sr_sta_mlr = sr_all_mlr[:,index[ip,0],index[ip,1]]# linear regression sr at stations
                                # hr_res2_sta = hr_re2_all[:,index[ip,0],index[ip,1]]
                                # hr_res3_sta = hr_re3_all[:,index[ip,0],index[ip,1]]
                        
                                ip_count=ip_count+1
                                target = hr_sta
                                mae_sta_sr_ep = np.nanmean(abs(sr_sta_ep-hr_sta))
                                rmse_sta_sr_ep = np.nanmean((sr_sta_ep-hr_sta)**2)**0.5
                                mae_sta_mlr = np.nanmean(abs(sr_sta_mlr-hr_sta))
                                rmse_sta_mlr = np.nanmean((sr_sta_mlr-hr_sta)**2)**0.5

                                var = [sr_sta_ep,sr_sta_mlr] # ,hr_res3_sta
                                leg = ['SR_en'+'('+'%5.3f,%5.3f'%(mae_sta_sr_ep,rmse_sta_sr_ep)+')',
                                       'MLR'+'('+'%5.3f,%5.3f'%(mae_sta_mlr,rmse_sta_mlr)+')']  # ,'Nearest'
                                figsize = (3.7,3.7)
                                figname = out_path+"dist_c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +'_'+ sta_user[ip]+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                                plot_mod_vs_obs(var,target,figname,axlab=axlab,leg=leg,alpha=0.3,
                                                figsize=figsize, fontsize=12,capt=capt_all[ip_count])
                                
                                txt = sta_user[ip]+'\n'+'mae,rmse'+'\n'+ \
                                      '%5.3f,%5.3f'%(mae_sta_sr_ep,rmse_sta_sr_ep)+'\n'+ \
                                      '%5.3f,%5.3f'%(mae_sta_mlr,rmse_sta_mlr)
                                data_lst_np.append(var)
                                txt_lst_np.append(txt)
                                data_ref_np.append([target,target])
                                
                        subsize = (2.5,2.5)
                        axlab = ('Reference '+unit_var,'Reconstructed '+unit_var,'')
                        leg = ['SR_en','MLR']
                        lloc = 9  # legend location, 9-upper center
                        legloc = [0.5,0.98]
                        loc_txt = [0.99,0.4]
                        lim = [[0,2],[0,2],[0,3],[0,6]]
                        figname = out_path+"dist_c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +'_'+ 'nsta_use'+'.png'
                        plotsubs_mod_vs_obs(data_lst_np,data_ref_np,figname,
                                            subsize=subsize,fontsize=12,nrow=1,
                                            axlab=axlab,leg=leg,lloc=lloc,legloc=legloc,
                                            alpha=0.3,marker='o',capt=capt_all[1:],
                                            txt=txt_lst_np,loc_txt=loc_txt,lim=lim)
                        
                        
                    #  if plot comparions for points 
                    if kp_pnt_t == 1: 
                        
                        ntpf = mod_para.ntpf  # no. of times per nc file
                        dt = mod_para.dt  # time steps between samples
                        ntpf_use = ntpf/dt  # no. of used time steps in 1 nc file

                        data_lst_np = []
                        for ip in range(nsta):
                            hr_sta = hr_all[:,index[ip,0],index[ip,1]]
                            if not np.isnan(hr_sta).all():  # check if not all nan
                                sr_sta_ep = sr_all_ep[:,index[ip,0],index[ip,1]]# epoch averaged sr at stations
                                sr_sta_mlr = sr_all_mlr[:,index[ip,0],index[ip,1]]# linear regression sr at stations
                                
                                hr_sta = rearrange_1d(hr_sta,ntpf_use)
                                sr_sta_ep = rearrange_1d(sr_sta_ep,ntpf_use)
                                sr_sta_mlr = rearrange_1d(sr_sta_mlr,ntpf_use)
                                
                                # target = hr_sta
                                # mae_sta_sr_ep = np.nanmean(abs(sr_sta_ep-hr_sta))
                                # rmse_sta_sr_ep = np.nanmean((sr_sta_ep-hr_sta)**2)**0.5
                                # mae_sta_mlr = np.nanmean(abs(sr_sta_mlr-hr_sta))
                                # rmse_sta_mlr = np.nanmean((sr_sta_mlr-hr_sta)**2)**0.5
                                
                                if ivar_hr==ivar_lr: 
                                    hr_res1_sta = hr_re1_all[:,index[ip,0],index[ip,1]] # RBFlinear instead bilinear
                                    hr_res3_sta = hr_re3_all[:,index[ip,0],index[ip,1]] # nearest
                                    hr_res1_sta = rearrange_1d(hr_res1_sta,ntpf_use)
                                    hr_res3_sta = rearrange_1d(hr_res3_sta,ntpf_use)
                                    
                                    var = [hr_sta,sr_sta_ep,sr_sta_mlr,hr_res1_sta,hr_res3_sta] # 
                                    leg = ['Reference','SR_en', 'MLR','RBFlinear','Nearest'] 
                                else:
                                    var = [hr_sta,sr_sta_ep,sr_sta_mlr]
                                    leg = ['Reference','SR_en', 'MLR'] 
                            data_lst_np.append(var)

                        Nt = len(sr_sta_ep)
                        tuse = tuse0[-Nt:]
                        time_lst = [[tuse] * len(data_lst_np[0])] * len(data_lst_np)  # nested list, dim as data_lst_np
                        nvar = len(data_lst_np[ip])-1

                        ylim = [[0,3.2],[0,3.2],[0,3.2],[0,6.4]] # ylimit for points
                        axlab = [['','ssh (m)'],['','u (m/s)'],['','v (m/s)'],
                                  ['','uw (m/s)'],['','vw (m/s)'],['','swh (m)'],
                                  ['','pwp (s)'],['','swh_ww (m)'],]
                        legloc = [0.5,1.03]
                        line_col=['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                        lloc = 9  # legend location, 9-upper center
                        
                        line_sty=['None','-','-','-','-','-'] 
                        marker = ['.']+['None']*5
                        subsize = [3.7,2.5]  # A4 8.3*11.7
                        loc_txt = [0.10,0.98]

                        tlim = [datetime(2021,11,29),datetime(2021,12,2)]
                        id_t0, id_t1 = tuse.index(tlim[0]), tuse.index(tlim[1])
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(len(data_lst_np[ip])-1):
                                var_mod = data_lst_np[ip][ivar+1][id_t0:id_t1]
                                var_ref = data_lst_np[ip][0][id_t0:id_t1]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))

                        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=2,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 


                        line_sty=['None','-','-','-','-','-'] # 'kv',
                        marker = ['.']+['None']*5
                        loc_txt = [0.60,0.98]
                        
                        subsize = [7.4,2.5]  # A4 8.3*11.7
                        
                        tlim = None
                        # if tlim is not None:
                        #     tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(nvar):
                                var_mod = data_lst_np[ip][ivar+1][:]
                                var_ref = data_lst_np[ip][0][:]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))
                        
                        tstr = '_'+tuse[0].strftime('%Y%m%d')+'_'+tuse[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=4,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 
                        
                        ofname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.csv'
                        err_sta= np.concatenate((rmse_sta_var,mae_sta_var),axis=1)
                        np.savetxt(ofname, err_sta,fmt='%f,') # ,delimiter=","

                        line_sty=['None','-','-','-','-','-'] 
                        marker = ['.']+['None']*5
                        
                        tlim = [datetime(2021,1,1),datetime(2021,3,31)]
                        id_t0, id_t1 = tuse.index(tlim[0]), tuse.index(tlim[1])
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(len(data_lst_np[ip])-1):
                                var_mod = data_lst_np[ip][ivar+1][id_t0:id_t1]
                                var_ref = data_lst_np[ip][0][id_t0:id_t1]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))

                        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=4,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 
                        ofname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.csv'
                        err_sta= np.concatenate((rmse_sta_var,mae_sta_var),axis=1)
                        np.savetxt(ofname, err_sta,fmt='%f,') # ,delimiter=","
                        
                        
                        tlim = [datetime(2021,4,1),datetime(2021,6,30)]
                        id_t0, id_t1 = tuse.index(tlim[0]), tuse.index(tlim[1])
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))                        
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(len(data_lst_np[ip])-1):
                                var_mod = data_lst_np[ip][ivar+1][id_t0:id_t1]
                                var_ref = data_lst_np[ip][0][id_t0:id_t1]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))

                        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=4,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 
                        ofname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.csv'
                        err_sta= np.concatenate((rmse_sta_var,mae_sta_var),axis=1)
                        np.savetxt(ofname, err_sta,fmt='%f,') # ,delimiter=","
                        
                        
                        tlim = [datetime(2021,7,1),datetime(2021,9,30)]
                        id_t0, id_t1 = tuse.index(tlim[0]), tuse.index(tlim[1])
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(len(data_lst_np[ip])-1):
                                var_mod = data_lst_np[ip][ivar+1][id_t0:id_t1]
                                var_ref = data_lst_np[ip][0][id_t0:id_t1]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))

                        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=4,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 
                        ofname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.csv'
                        err_sta= np.concatenate((rmse_sta_var,mae_sta_var),axis=1)
                        np.savetxt(ofname, err_sta,fmt='%f,') # ,delimiter=","
                        
                        
                        tlim = [datetime(2021,10,1),datetime(2021,12,31)]
                        id_t0, id_t1 = tuse.index(tlim[0]), tuse.index(tlim[1])
                        rmse_lst_np = []
                        txt_lst_np = []
                        mae_sta_var,rmse_sta_var = np.zeros((nsta,nvar)),np.zeros((nsta,nvar))
                        for ip in range(nsta):
                            txt_mae,txt_rmse = 'mae : ','rmse: '
                            for ivar in range(len(data_lst_np[ip])-1):
                                var_mod = data_lst_np[ip][ivar+1][id_t0:id_t1]
                                var_ref = data_lst_np[ip][0][id_t0:id_t1]
                                mae_sta_var[ip][ivar] = np.nanmean(abs(var_mod-var_ref))
                                rmse_sta_var[ip][ivar] = np.nanmean((var_mod-var_ref)**2)**0.5
                                txt_mae = txt_mae+'%5.3f,'%(mae_sta_var[ip][ivar])
                                txt_rmse = txt_rmse+'%5.3f,'%(rmse_sta_var[ip][ivar])
                            txt_lst_np.append(txt_mae + '\n'+ txt_rmse + '\n' +'mean:%5.3f'%(np.nanmean(var_ref)))

                        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[-1].strftime('%Y%m%d') 
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.png'
                        plotsubs_line_list(time_lst,data_lst_np,figname,tlim=tlim,ylim=ylim,subsize = subsize,
                                                    fontsize=12,nrow=4,title=sta_user,axlab=axlab[ichl],
                                                    leg=leg,leg_col=len(leg),lloc=9,legloc=legloc,
                                                    line_sty=line_sty,line_col=line_col,marker=marker,
                                                    capt=capt_all,txt=txt_lst_np,loc_txt=loc_txt) 
                        ofname = out_path+"/c%d_re%d_ep%d_%d" % (ichl,irep,epoch,epoc_num[0]) +tstr+ '_nsta_use'+'.csv'
                        err_sta= np.concatenate((rmse_sta_var,mae_sta_var),axis=1)
                        np.savetxt(ofname, err_sta,fmt='%f,') # ,delimiter=","

                        