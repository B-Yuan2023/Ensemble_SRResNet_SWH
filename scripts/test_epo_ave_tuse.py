#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import os
import numpy as np
import time

import torch

from models import GeneratorResNet
from mod_srres import SRRes,SRResA
from funs_prepost import (make_list_file_t,nc_load_vars,nc_normalize_vars,
                          var_denormalize,plot_line_list,plt_pcolorbar_list,
                          interpolate_tensor,interp4d_tensor_nearest_neighbor,interp4d_tensor_bilinear)

from datetime import datetime, timedelta # , date
from funs_sites import select_sta

# from scipy.interpolate import griddata, interp2d, RBFInterpolator
# griddata no extrapolation, RBFInterpolator has extrapolation

import sys
import importlib
mod_name= 'par55e'         #'par55e' # sys.argv[1]
path_par = "../"  # the path of parameter files, also used for output path
sys.path.append(path_par)  # add path of parameter files to system temporarily for module import
mod_para=importlib.import_module(mod_name)

kmask = 1

if __name__ == '__main__':
    
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
    # rep=[0]
    
    if hasattr(mod_para, 'tshift'):
        tshift = mod_para.tshift # time shift in hour of low resolution data
    else:
        tshift = 0
    if hasattr(mod_para, 'll_lr'):
        ll_lr = mod_para.ll_lr # user domain lr [latitude,longitude]
    else:
        ll_lr = [None]*2
    if hasattr(mod_para, 'll_hr'):
        ll_hr = mod_para.ll_hr # user domain hr [latitude,longitude]
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
    epoch0,epoch1 = opt.N_epochs, opt.N_epochs-31   # 31
    epoc_num = np.arange(epoch0,epoch1,-1)  # use a range of epochs for average
    nep_skip = 10  # no. of skipped epochs for saving 

    keyplot = 1
    # opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    
    # select a range of data for testing 
    # tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    tlim = [datetime(2021,11,29),datetime(2021,12,2)]
    # tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
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
    
    # get logitude and latitude of data 
    nc_f = files_hr[0][0]
    lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    # get all hr data. Note: only for a short period, otherwise could be too large
    hr_all_test = np.zeros((Nt,nchl_o,len(lat),len(lon))) 
    for i in range(Nt):
        for ichl in range(nchl_o): 
            nc_f = files_hr[i][ichl]
            indt = indt_hr[i][ichl]
            dat_hr =  nc_load_vars(nc_f,var_hr[0],[indt],lats=ll_hr[0],lons=ll_hr[1])[3]
            # mask =  nc_load_vars(files_hr[i],var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[4]
            # dat_hr[mask] = np.nan
            hr_all_test[i,ichl,:,:] = dat_hr

    # load hr_all using the way for NN model 
    hr_all = []  # check if after normalize/denormalize, the same as hr_all_test
    for i in range(Nt):
        nc_f = files_hr[i]
        indt = indt_hr[i]
        data = nc_normalize_vars(nc_f,var_hr,indt,varm_hr,
                                         ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
        x = np.transpose(data,(2,0,1)) #(C,H,W)
        hr = torch.from_numpy(x)
        mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
        mask = np.squeeze(mask)

        hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
        hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
        
        if kmask == 1: 
            hr_norm0[:,:,mask] = np.nan
        hr_all.append(hr_norm0)
    hr_all = np.concatenate(hr_all, axis=0)
    np.allclose(hr_all, hr_all_test, equal_nan=True)
    
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_ens/'
    os.makedirs(out_path0, exist_ok=True)    
    
    # save original hr for the selected time range
    filename = out_path0 + 'hr'+tstr+'.npz'
    # if not os.path.isfile(filename): 
    np.savez(filename,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)

    # reconstuct use direct interpolation 
    # interpolation for cases if low and high variable are the same
    if ivar_hr==ivar_lr:
        hr_re1_all = []
        hr_re2_all = []
        hr_re3_all = []
        for i in range(Nt):
            nc_f = files_hr[i] 
            indt = indt_hr[i] # time index in nc_f
            mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
            mask = np.squeeze(mask)
    
            nc_f = files_lr[i]
            indt = indt_lr[i] # time index in nc_f
            data = nc_normalize_vars(nc_f,var_lr,indt,varm_lr,
                                     ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
            mask_lr = nc_load_vars(nc_f[0],var_lr[0],indt,lats=ll_lr[0],lons=ll_lr[1])[4] #(1,H,W)
            mask_lr = np.squeeze(mask_lr)
            mask_lr_ud = np.flipud(mask_lr) # dimensionless data flipped

            x = np.transpose(data,(2,0,1)) #(C,H,W)
            lr = torch.from_numpy(x)
            lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
            
            # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
            # hr_restore1 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
            # hr_restore2 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
            # hr_restore3 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
            
            # using other interpolation method instead of torch with extrapolation
            # RBFInterpolator kernal: {'quintic', 'linear', 'inverse_quadratic', 
            # 'cubic', 'thin_plate_spline'(default), 'multiquadric', 'gaussian', 'inverse_multiquadric'}
            lr[:,:,mask_lr_ud.copy()] = np.nan
            # interpolate use scale factor, same coordinates of bottom left and top right points
            hr_restore1 = interpolate_tensor(lr, opt.up_factor, kintp=2, method='linear') 
            # based on the present specified coordinates in par file, should use scale factor
            # hr_restore1_ = interpolate_tensor(lr, x_in=ll_lr[1],y_in=ll_lr[0],
            #                                   x_out=ll_hr[1],y_out=ll_hr[0],kintp=2, method='linear') # use xy

            # use custum nearest neighbor interpolation
            hr_restore2 = interp4d_tensor_bilinear(lr, opt.up_factor)
            
            # use custum nearest neighbor interpolation
            hr_restore3 = interp4d_tensor_nearest_neighbor(lr, opt.up_factor)
            
            hr_restore1_norm0  = var_denormalize(hr_restore1.detach().numpy(),varm_hr)
            hr_restore2_norm0  = var_denormalize(hr_restore2.detach().numpy(),varm_hr)
            hr_restore3_norm0  = var_denormalize(hr_restore3.detach().numpy(),varm_hr)
            
            if kmask == 1: 
                hr_norm0[:,:,mask] = np.nan
                hr_restore1_norm0[:,:,mask] = np.nan
                hr_restore2_norm0[:,:,mask] = np.nan
                hr_restore3_norm0[:,:,mask] = np.nan
            
            hr_re1_all.append(hr_restore1_norm0)
            hr_re2_all.append(hr_restore2_norm0)
            hr_re3_all.append(hr_restore3_norm0)
        
        hr_re1_all = np.concatenate(hr_re1_all, axis=0)
        hr_re2_all = np.concatenate(hr_re2_all, axis=0)
        hr_re3_all = np.concatenate(hr_re3_all, axis=0)
        
        filename = out_path0 + 'hr'+tstr+'_interp'+'.npz'
        # if not os.path.isfile(filename):
        np.savez(filename,hr_re1_all=hr_re1_all,hr_re2_all=hr_re2_all,hr_re3_all=hr_re3_all,lat=lat,lon=lon,t=tuser0)

    # nskp = (80,80)  # skipping grid points 
    # index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all_test,ivar_hr,lon,lat,nskp)
    
    # use selected stations, 3 near buoys at 10, 20, 40 m, 1 at maximam SWH
    index = np.array([[104, 22],[76, 6],[83, 20],[88, 239]])
    # ll_sta = np.array([[27.950,43.200],[27.550,42.500],[27.900,42.675],[33.375,42.800]])
    ll_sta = np.array([[43.200,27.950],[42.500,27.550],[42.675,27.900],[42.800,33.375]]) # 1st lat 2nd lon in select_sta
    sta_user = ['P'+str(ip+1) for ip in range(len(index))]

    nsta = len(index)  # number of stations 

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
    
    #  make a list for figure captions
    alpha = list(map(chr, range(ord('a'), ord('z')+1)))
    alpha_l = alpha + ['a'+i for i in alpha]
    capt_all = ['('+alpha_l[i]+')' for i in range(len(alpha_l))]
    
    for irep in rep:
        print(f'Repeat {irep}')
        print('--------------------------------')
    
        out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_re'+ str(irep)+'_mk'+str(kmask)+'_ens/'
        os.makedirs(out_path, exist_ok=True)

        txtname = out_path0+"/para_cmp"+'_rp%d_ep%d_%d'%(irep,epoc_num[0],epoc_num[-1])+tstr+'.txt'
        outfile = open(txtname, 'w')
        # outfile.write('# varm_hr_test, varm_lr_test\n')
        # np.savetxt(outfile, np.hstack((varm_hr_test,varm_lr_test)), fmt='%-7.4f,')
        # outfile.write('# irep: {0}\n'.format(irep))
        
        sr_all_ep = np.zeros((Nt,nchl_o,opt.hr_height, opt.hr_width))  # epoch averaged sr_all(Nt,C,H,W)
        sr_sta_ep = np.zeros(shape=(nsta,nchl_o,Nt))  # epoch averaged sr at stations
        iepo = 0
        
        for epoch in epoc_num:

            outfile.write('# epo {0}\n'.format(epoch))
            # suf = '_res' + str(opt.residual_blocks) + '_max_var1para_r8' # + '_eps_'+str(0)

            model_name = 'netG_epoch_%d_re%d.pth' % (epoch,irep)
            if cuda:
                generator = generator.cuda()
                checkpointG = torch.load(ipath_nn + model_name)
            else:
                checkpointG = torch.load(ipath_nn + model_name, map_location=lambda storage, loc: storage)
            generator.load_state_dict(checkpointG['model_state_dict'])
            generator.eval()
            
            sr_sta = np.zeros(shape=(nsta,nchl_o,Nt))
            hr_sta = np.zeros(shape=(nsta,nchl_o,Nt))
            if ivar_hr==ivar_lr:
                hr_res1_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                hr_res2_sta = np.zeros(shape=(nsta,nchl_o,Nt))
                hr_res3_sta = np.zeros(shape=(nsta,nchl_o,Nt))
            
            sr_varm = np.zeros(shape=(nchl_o,2,Nt))  # max/min of sr at Nt for all channels
            hr_varm = np.zeros(shape=(nchl_o,2,Nt))  # max/min of hr at Nt for all channels
            dif_varm = np.zeros(shape=(nchl_o,2,Nt))
            
            sr_all = []
            hr_all = []
            
            for it in range(0,Nt):

                nc_f = files_hr[it] 
                indt = indt_hr[it] # time index in nc_f
                data = nc_normalize_vars(nc_f,var_hr,indt,varm_hr,
                                                 ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
                x = np.transpose(data,(2,0,1)) #(C,H,W)
                hr = torch.from_numpy(x)
                mask = nc_load_vars(nc_f[0],var_hr[0],indt,lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
                mask = np.squeeze(mask)

                nc_f = files_lr[it]
                indt = indt_lr[it] # time index in nc_f
                data = nc_normalize_vars(nc_f,var_lr,indt,varm_lr,
                                         ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
                x = np.transpose(data,(2,0,1)) #(C,H,W)
                lr = torch.from_numpy(x)
                                
                lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
                hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
                
                start = time.time()
                sr = generator(lr.float())
                end = time.time()
                elapsed = (end - start)
                print('cost ' + str(elapsed) + 's')
                
                sr_norm0 = var_denormalize(sr.detach().numpy(),varm_hr) # (N,C,H,W), flipud height back
                hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
                
                if kmask == 1: 
                    hr_norm0[:,:,mask] = np.nan
                    sr_norm0[:,:,mask] = np.nan
                
                sr_all.append(sr_norm0)
                hr_all.append(hr_norm0)
        
                sr_varm[:,0,it] = np.nanmax(sr_norm0,axis=(0,2,3)) # max for channel
                sr_varm[:,1,it] = np.nanmin(sr_norm0,axis=(0,2,3)) # min for channel
                hr_varm[:,0,it] = np.nanmax(hr_norm0,axis=(0,2,3)) # max for channel
                hr_varm[:,1,it] = np.nanmin(hr_norm0,axis=(0,2,3)) # min for channel
                dif = sr_norm0 - hr_norm0
                dif_varm[:,0,it] = np.nanmax(dif,axis=(0,2,3)) # max for channel
                dif_varm[:,1,it] = np.nanmin(dif,axis=(0,2,3)) # min for channel
                
                for ip in range(nsta):
                    sr_sta[ip,:,it]=sr_norm0[:,:,index[ip,0],index[ip,1]]
                    sr_sta_ep[ip,:,it]=(sr_sta_ep[ip,:,it]*iepo + sr_sta[ip,:,it])/(iepo+1) # epoch averaged sr at stations
                    hr_sta[ip,:,it]=hr_norm0[:,:,index[ip,0],index[ip,1]]
                    if ivar_hr==ivar_lr:
                        hr_res1_sta[ip,:,it]=hr_re1_all[it,:,index[ip,0],index[ip,1]]
                        hr_res2_sta[ip,:,it]=hr_re2_all[it,:,index[ip,0],index[ip,1]]
                        hr_res3_sta[ip,:,it]=hr_re3_all[it,:,index[ip,0],index[ip,1]]
        
            sr_all = np.concatenate(sr_all, axis=0)
            hr_all = np.concatenate(hr_all, axis=0)
            # ensemble average: use selected epochs 
            iepo = iepo + 1
            sr_all_ep = (sr_all_ep*(iepo-1) + sr_all)/iepo  # epoch averaged sr [Nt,c,H,W]
            
            # save super-resolution hr (single & ensemble) for the selected time range
            if iepo%nep_skip == 0:
                filename = out_path0+'sr'+tstr+'_re%d_ep%d_%d' % (irep,epoch,epoc_num[0]) +'.npz'
                # if not os.path.isfile(filename): 
                np.savez(filename,sr_all=sr_all,sr_all_ep=sr_all_ep,hr_all=hr_all,lat=lat,lon=lon,t=tuser0)
            
            sr_varmt = np.array([sr_varm[:,0,:].max(axis=1), sr_varm[:,1,:].min(axis=1)]).transpose()
            hr_varmt = np.array([hr_varm[:,0,:].max(axis=1), hr_varm[:,1,:].min(axis=1)]).transpose()
            dif_varmt = np.array([dif_varm[:,0,:].max(axis=1), dif_varm[:,1,:].min(axis=1)]).transpose()
            
            outfile.write('# stations sr_varmt, hr_varmt, dif_varmt\n')
            np.savetxt(outfile, np.hstack((sr_varmt,hr_varmt,dif_varmt)), fmt='%-7.4f,')
            
            index_max = np.argmax(hr_sta,axis=2)
            index_min = np.argmin(hr_sta,axis=2)
            
            if keyplot == 1 and iepo%nep_skip == 0:
                # plot comparison for locations
                axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)'],
                         ['Time','uw (m/s)'],['Time','vw (m/s)'],['Time','swh (m)'],
                         ['Time','pwp (s)'],['Time','swh_ww (m)'],]
                # line_sty=['k.','b','r-','m-','g-','c']
                line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
                for ip in range(nsta):
                    for i in range(len(ivar_hr)):
                        var_sta = hr_sta[ip,i,:]
                        var = sr_sta[ip,i,:]
                        var_ep = sr_sta_ep[ip,i,:]
                        if ivar_hr==ivar_lr:
                            var_res1 = hr_res1_sta[ip,i,:]
                            var_res2 = hr_res2_sta[ip,i,:]
                            var_res3 = hr_res3_sta[ip,i,:]
                            time_lst = [tuser0,tuser0,tuser0,tuser0,tuser0,tuser0]
                            data_lst = [var_sta,var,var_ep,var_res1,var_res2,var_res3]
                            leg = ['hr','sr','sr_ens','intp1','bilinear','nearest']
                        else:
                            time_lst = [tuser0,tuser0,tuser0]
                            data_lst = [var_sta,var,var_ep] 
                            leg = ['hr','sr','sr_ens']
                        ich = ivar_hr[i]
                        # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+ sta_user[ip]+'.png'
                        # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoch) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                        figname = out_path+"/c%d_re%d_ep%d_%d" % (ich,irep,epoch,epoc_num[0]) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                        plot_line_list(time_lst,data_lst,tlim,figname,axlab=axlab[ich],leg=leg,leg_col=2,line_sty=line_sty)
                    
        outfile.close()
        
        # plot last epoch averaged result for line and 2D. Now use compare_ml_2D_pnt.py
        kp_end = 1  
        if kp_end==1: 
            # plot the epoch averaged sr vs hr
            # plot comparison for locations
            axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)'],
                     ['Time','uw (m/s)'],['Time','vw (m/s)'],['Time','swh (m)'],
                     ['Time','pwp (s)'],['Time','swh_ww (m)'],]
            leg = ['hr','sr','sr_ens','intp1','bilinear','nearest']
            # line_sty=['k.','b','r-','m-','g-','c']
            line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
            for ip in range(nsta):
                for i in range(len(ivar_hr)):
                    var_sta = hr_sta[ip,i,:]
                    var = sr_sta[ip,i,:]
                    var_ep = sr_sta_ep[ip,i,:]
                    var_res1 = hr_res1_sta[ip,i,:]
                    var_res2 = hr_res2_sta[ip,i,:]
                    var_res3 = hr_res3_sta[ip,i,:]
                    time_lst = [tuser0,tuser0,tuser0,tuser0,tuser0,tuser0]
                    date_lst = [var_sta,var,var_ep,var_res1,var_res2,var_res3]
                    ich = ivar_hr[i]
                    figname = out_path+"/c%d_re%d_ep%d_%d" % (ich,irep,epoch,epoc_num[0]) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                    plot_line_list(time_lst,date_lst,tlim,figname,axlab=axlab[ich],leg=leg,leg_col=2,line_sty=line_sty)
                
            # plot comparison for 2D field
            clim = [[-3.0,3.0],[-1.2,1.2],[-1.5,1.5],[12,15],[12,15],[0.0,4.0],[0.0,15.]]  # ssh,u,v,uw,vw,swh,pwp
            unit_suv = ['ssh (m)','u (m/s)','v (m/s)','uw (m/s)','vw (m/s)','swh (m)','pwp (s)','swh_ww (m)']
            rmse_sr = np.nanmean((sr_all - hr_all) ** 2,axis=(2,3))**(0.5)
            mae_sr = np.nanmean(abs(sr_all - hr_all),axis=(2,3))
            rmse_sr_ep = np.nanmean((sr_all_ep - hr_all) ** 2,axis=(2,3))**(0.5)
            mae_sr_ep = np.nanmean(abs(sr_all_ep - hr_all),axis=(2,3))
            rmse_re1 = np.nanmean((hr_re1_all - hr_all) ** 2,axis=(2,3))**(0.5)
            mae_re1 = np.nanmean(abs(hr_re1_all - hr_all),axis=(2,3))
            rmse_re2 = np.nanmean((hr_re2_all - hr_all) ** 2,axis=(2,3))**(0.5)
            mae_re2 = np.nanmean(abs(hr_re2_all - hr_all),axis=(2,3))
            rmse_re3 = np.nanmean((hr_re3_all - hr_all) ** 2,axis=(2,3))**(0.5)
            mae_re3 = np.nanmean(abs(hr_re3_all - hr_all),axis=(2,3))
            
            loc_txt = [0.01,0.90] # location of text
            nt_sub = 4  # plot nt_sub times in one row 
            nfig = int(Nt/nt_sub)
            kbar = 5  # type of colorbar
            kax = 1   # turn ax off or not, 1 off. 
            for ifig in range(nfig):
                ind = np.arange(nt_sub*ifig,nt_sub*(ifig+1), 1).tolist()
                
                for i in range(nchl_o):
                    ichl = ivar_hr[i]
                    sample  = [hr_all[ind,i,:,:],sr_all_ep[ind,i,:,:],sr_all[ind,i,:,:],hr_re3_all[ind,i,:,:],
                               hr_re2_all[ind,i,:,:],hr_re1_all[ind,i,:,:]]
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
                        ['bilinear'+ tuser0[it].strftime('%Y%m%d %H') for it in ind]+ \
                        ['intp1'+ tuser0[it].strftime('%Y%m%d %H') for it in ind]
                    txt = ['hr' for _ in range(nt_sub)] + \
                        ['sr_ens\n'+'MAE=%5.3f'%mae_sr_ep[it,i]+'\nRMSE=%5.3f'%rmse_sr_ep[it,i] for it in ind] + \
                        ['sr\n'+'MAE=%5.3f'%mae_sr[it,i]+'\nRMSE=%5.3f'%rmse_sr[it,i] for it in ind] + \
                        ['nearest\n'+'MAE=%5.3f'%mae_re3[it,i]+'\nRMSE=%5.3f'%rmse_re3[it,i]  for it in ind]+ \
                        ['bilinear\n'+'MAE=%5.3f'%mae_re2[it,i]+'\nRMSE=%5.3f'%rmse_re2[it,i] for it in ind]+ \
                        ['intp1\n'+'MAE=%5.3f'%mae_re1[it,i]+'\nRMSE=%5.3f'%rmse_re1[it,i] for it in ind]
                    figname = out_path+"c%d_re%d_ep%d_%d"% (ivar_hr[i],irep,epoc_num[-1],epoch0)+tstr+"_ax%d_kb%d_f%d.png"%(kax,kbar,ifig)
                    plt_pcolorbar_list(lon,lat,sample,figname,cmap = 'coolwarm',
                                       clim=clim_chl,kbar=kbar,unit=unit,title=title,
                                       nrow=nrow,axoff=kax,capt=capt_all,txt=txt,loc_txt=loc_txt) 

