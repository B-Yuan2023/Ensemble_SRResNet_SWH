#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import os
import sys
import numpy as np
from datetime import datetime, timedelta # , date

from funs_prepost import plot_line_list,plt_pcolorbar_list
from funs_sites import index_stations

import importlib
mod_name= 'par55e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

def lst_flatten(xss):
    return [x for xs in xss for x in xs]

def select_sta(var,ivar_hr,lon,lat,nskp = (40,40),kpshare=1):
    # var 4d array
    # estimate max min value for the selected period
    nchl = len(ivar_hr)
    sta_max = lst_flatten([['v%d_max'%ivar_hr[i], 'v%d_min'%ivar_hr[i]] for i in range(nchl)])
    varm_hr = np.ones((nchl,2))
    ind_varm = np.ones((nchl,2),dtype= np.int64)
    for i in range(nchl):
        varm_hr[i,0] = var[:,i,:,:].max()
        ind_varm[i,0] = np.argmax(var[:,i,:,:])
        varm_hr[i,1] = var[:,i,:,:].min()
        ind_varm[i,1] = np.argmin(var[:,i,:,:])
    temp = np.unravel_index(ind_varm.flatten(), (len(var),len(lat),len(lon)))
    index = np.array([temp[1],temp[2]]).transpose()
    
    # select several observation locations for comparison 
    sta_user0 = ['WAVEB04', 'WAVEB05', 'WAVEB06']
    sta_user1 = [sta_user0[i] + str(j) for i in range(len(sta_user0)) for j in range(4)]
    ll_stas = np.array([[28.611600,43.539200],[27.906700,42.696400],[28.343800,43.371700]])
    ll_shift = np.array([[0,0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    ll_stas = ll_stas+ll_shift
    ind_sta = index_stations(ll_stas[:,0],ll_stas[:,1],lon,lat)
    index = ind_sta
    
    # add points in the domain for testing
    nx_skp = nskp[0]
    ny_skp = nskp[1]
    if kpshare ==1: #  shared poins by lr & hr
        ix = np.arange(0, len(lat)-1, nx_skp) #  shared poins by lr & hr
        iy = np.arange(0, len(lon)-1, nx_skp)
    else:
        ix = np.arange(int(nx_skp/2), len(lat)-1, nx_skp) #  non-shared poins by lr & hr, initial index links to scale
        iy = np.arange(int(nx_skp/2), len(lon)-1, ny_skp)
    xv, yv = np.meshgrid(ix, iy)
    ind_add = np.vstack((np.int_(xv.flatten()), np.int_(yv.flatten()))).T
    sta_add = ['p'+ str(i).zfill(2) for i in range(len(ind_add))]
    
    # add gauging stations 
    index = np.vstack((index, ind_add, ind_sta))
    sta_user = sta_max + sta_add + sta_user1
    # no gauging station
    # index = np.vstack((index, ind_add))
    # sta_user = sta_max + sta_add 
    
    ll_sta = np.array([lat[index[:,0]],lon[index[:,1]]]).transpose() # should corresponds to (H,W), H[0]-lowest lat
    return index,sta_user,ll_sta,varm_hr,ind_varm

if __name__ == '__main__':
    
    opt = mod_para.opt
    suf = mod_para.suf+mod_name
    var_lr = mod_para.var_lr
    var_hr = mod_para.var_hr
    ivar_hr = mod_para.ivar_hr
    ivar_lr = mod_para.ivar_lr
    nchl_i = len(var_lr)
    nchl_o = len(var_hr)
    
    nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    rep = [0]
    epoc_num =[100] #
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-31
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average
    
    key_ep_sort = 0 # to use epoc here or load sorted epoc no. 
    nepoc = 3 # no. of sorted epochs for analysis

    kp_pnt = 1  # key to plot comparison for points at selected period 
    kp_2D = 1   # key to plot comparison for 2D map at selected period 
    nep_skip = 10  # no. of skipped epochs for plotting 
    
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    
    # select a range of data for testing 
    tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    dt = 3
    
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d') + '_t%d'%dt
    Nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    tuser0 = [(tlim[0] + timedelta(hours=x*dt)) for x in range(0,Nt)]
    tshift = 0 # in hour
    tuser = [(tlim[0] + timedelta(hours=x*dt)) for x in range(tshift,Nt+tshift)] # time shift for numerical model
    
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_ens/'
    os.makedirs(out_path0, exist_ok=True)    
    
    # load original hr and interpolated hr for the selected time range
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
    filename = out_path0 + 'hr'+tstr+'_interp'+'.npz'
    if not os.path.isfile(filename):
        # np.savez(filename,hr_re1_all=hr_re1_all,hr_re2_all=hr_re2_all,hr_re3_all=hr_re3_all,lat=lat,lon=lon,t=tuser0)
        print('hr file from interpolation not saved!',file=sys.stderr)
    else:
        datald = np.load(filename) # load
        hr_re1_all = datald['hr_re1_all']
        hr_re2_all = datald['hr_re2_all']
        hr_re3_all = datald['hr_re3_all']
    
    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]

    clim = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,4.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
    unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']    
    for irep in rep:
        print(f'Repeat {irep}')
        print('--------------------------------')
    
        out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens/'
        os.makedirs(out_path, exist_ok=True)
    
        iepo = 0
        for epoch in epoc_num:
            # load super-resolution hr (single & ensemble) for the selected time range
            filename = out_path0+'sr'+tstr+'_re%d_ep%d_%d' % (irep,epoch,epoc_num[0]) +'.npz'
            if not os.path.isfile(filename): 
                # np.savez(filename,sr_all=sr_all,sr_all_ep=sr_all_ep,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
                print('hr file not saved!', file=sys.stderr)
            else:
                datald = np.load(filename) # load
                sr_all = datald['sr_all']
                sr_all_ep = datald['sr_all_ep']
                # hr_all1 = datald['hr_all']
                # np.allclose(hr_all, hr_all1, equal_nan=True) # checked
            
            iepo = iepo + 1
            if kp_2D == 1 and (iepo-1)%nep_skip == 0:
                # plot comparison for 2D field
                rmse_sr = np.nanmean((sr_all - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_sr = np.nanmean(abs(sr_all - hr_all),axis=(2,3))
                rmse_sr_ep = np.nanmean((sr_all_ep - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_sr_ep = np.nanmean(abs(sr_all_ep - hr_all),axis=(2,3))
                rmse_re2 = np.nanmean((hr_re2_all - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_re2 = np.nanmean(abs(hr_re2_all - hr_all),axis=(2,3))
                rmse_re3 = np.nanmean((hr_re3_all - hr_all) ** 2,axis=(2,3))**(0.5)
                mae_re3 = np.nanmean(abs(hr_re3_all - hr_all),axis=(2,3))
                
                loc_txt = [0.01,0.90] # location of text
                nt_sub = 6  # plot nt_sub times in one row 
                nfig = int(Nt/nt_sub)
                kbar = 5  # type of colorbar
                kax = 1   # turn ax off or not, 1 off. 
                for ifig in range(nfig):
                    ind = np.arange(nt_sub*ifig,nt_sub*(ifig+1), 1).tolist()
                    
                    for i in range(nchl_o):
                        ichl = ivar_hr[i]
                        sample  = [hr_all[ind,i,:,:],sr_all_ep[ind,i,:,:],sr_all[ind,i,:,:],hr_re2_all[ind,i,:,:],
                                   hr_re3_all[ind,i,:,:]]
                        nrow = len(sample)
                        sample = np.concatenate(sample, axis=0)
                        clim_chl = [clim[ichl]]*len(sample)
                        unit = [unit_suv[ichl]]*len(sample)                
                        # title = ['hr' for _ in range(nt_sub)] + \
                        #     ['sr_ens'+'(%5.3f'%mae_sr[it,i]+',%5.3f'%rmse_sr[it,i]+')' for it in ind] + \
                        #     ['nearest'+'(%5.3f'%mae_re3[it,i]+',%5.3f'%rmse_re3[it,i]+')'  for it in ind]+ \
                        #     ['bilinear'+'(%5.3f'%mae_re2[it,i]+',%5.3f'%rmse_re2[it,i]+')' for it in ind]
                        title = ['hr'+ tuser0[it].strftime('%Y%m%d %H') for it in ind] + \
                            ['sr_ens'+ tuser0[it].strftime('%Y%m%d %H') for it in ind] + \
                            ['sr'+ tuser0[it].strftime('%Y%m%d %H') for it in ind] + \
                            ['nearest'+ tuser0[it].strftime('%Y%m%d %H') for it in ind]+ \
                            ['bilinear'+ tuser0[it].strftime('%Y%m%d %H') for it in ind]
                        txt = ['hr' for _ in range(nt_sub)] + \
                            ['sr_ens\n'+'MAE=%5.3f'%mae_sr_ep[it,i]+'\nRMSE=%5.3f'%rmse_sr_ep[it,i] for it in ind] + \
                            ['sr\n'+'MAE=%5.3f'%mae_sr[it,i]+'\nRMSE=%5.3f'%rmse_sr[it,i] for it in ind] + \
                            ['nearest\n'+'MAE=%5.3f'%mae_re3[it,i]+'\nRMSE=%5.3f'%rmse_re3[it,i]  for it in ind]+ \
                            ['bilinear\n'+'MAE=%5.3f'%mae_re2[it,i]+'\nRMSE=%5.3f'%rmse_re2[it,i] for it in ind]
                        figname = out_path+"c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoch,epoch0)+tstr+"_ax%d_kb%d_f%d.png"%(kax,kbar,ifig)
                        plt_pcolorbar_list(lon,lat,sample,figname,cmap = 'coolwarm',
                                           clim=clim_chl,kbar=kbar,unit=unit,title=title,
                                           nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 

    #  if plot comparions for points 
    if kp_pnt == 1:
        # select several locations for comparison
        nskp = (80,80)  # skipping grid points 
        index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all,ivar_hr,lon,lat,nskp)
        nsta = len(index)  # number of stations 
        
        axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)'],
                 ['Time','uw (m/s)'],['Time','vw (m/s)'],['Time','swh (m)'],
                 ['Time','pwp (s)'],['Time','swh_ww (m)'],]
        leg = ['hr','sr','sr_ens','bicubic','bilinear','nearest']
        # line_sty=['k.','b','r-','m-','g-','c']
        line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
    
        for irep in rep:
            print(f'Repeat {irep}')
            print('--------------------------------')
        
            out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens/'
            os.makedirs(out_path, exist_ok=True)
        
            iepo = 0
            for epoch in epoc_num:
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
                hr_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                hr_res1_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                hr_res2_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                hr_res3_sta = np.zeros(shape=(nsta,nchl_o,Nt))
    
                for it in range(0,Nt):
                    for ip in range(nsta):
                        sr_sta[ip,:,it]=sr_all[it,:,index[ip,0],index[ip,1]]
                        sr_sta_ep[ip,:,it]= sr_all_ep[it,:,index[ip,0],index[ip,1]]# epoch averaged sr at stations
                        hr_sta[ip,:,it]=hr_all[it,:,index[ip,0],index[ip,1]]
                        hr_res1_sta[ip,:,it]=hr_re1_all[it,:,index[ip,0],index[ip,1]]
                        hr_res2_sta[ip,:,it]=hr_re2_all[it,:,index[ip,0],index[ip,1]]
                        hr_res3_sta[ip,:,it]=hr_re3_all[it,:,index[ip,0],index[ip,1]]
                
                iepo = iepo + 1
                if (iepo-1)%nep_skip == 0:
                    # plot comparison for locations
                    for ip in range(nsta):
                        for i in range(len(ivar_hr)):
                            var_sta = hr_sta[ip,i,:]
                            var = sr_sta[ip,i,:]
                            var_ep = sr_sta_ep[ip,i,:]
                            var_res1 = hr_res1_sta[ip,i,:]
                            var_res2 = hr_res2_sta[ip,i,:]
                            var_res3 = hr_res3_sta[ip,i,:]
                            data_lst = [var_sta,var,var_ep,var_res1,var_res2,var_res3]
                            time_lst = [tuser0] * len(data_lst)
                            ich = ivar_hr[i]
                            # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+ sta_user[ip]+'.png'
                            # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoch) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                            figname = out_path+"/c%d_re%d_ep%d_%d" % (ich,irep,epoch,epoc_num[0]) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                            plot_line_list(time_lst,data_lst,tlim,figname,axlab[ich],leg=leg,leg_col=2,line_sty=line_sty)
