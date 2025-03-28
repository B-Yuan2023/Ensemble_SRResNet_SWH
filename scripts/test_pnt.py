#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:54:41 2023

@author: g260218
"""
import numpy as np
import os
import time

import torch
from PIL import Image

from models import GeneratorResNet
from mod_srres import SRRes,SRResA
from funs_prepost import (make_list_file_t,nc_load_vars,nc_normalize_vars,var_denormalize,plot_line_list)

from datetime import datetime, timedelta # , date
from funs_sites import index_stations

import sys
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

    # ll_shift = np.array([[-1.0/128*1.0,-0.50/128*1.0],[0,0],[0,0],[0,0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    # ll_stas = ll_stas+ll_shift
    
    # # select several observation locations for comparison 
    # sta_user0 = ['HelgolandTG','AlteWeserTG','WangeroogeTG']
    # sta_user1 = [sta_user0[i] + str(j) for i in range(len(sta_user0)) for j in range(4)]
    # ll_stas = np.array([[7.890000,54.178900],[8.127500,53.863300],[7.929000,53.806000]])
    # ll_shift = np.array([[0,0.51/128*2.0],[0,0],[0,0]]) # shift the station to the water region, lon,lat
    # ll_stas = ll_stas+ll_shift
    # ind_sta = index_stations(ll_stas[:,0],ll_stas[:,1],lon,lat)
    # index = ind_sta
    
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
    # index = np.vstack((index, ind_add, ind_sta))
    # sta_user = sta_max + sta_add + sta_user1
    index = np.vstack((index, ind_add)) # no gauging station
    sta_user = sta_max + sta_add 
    ll_sta = np.array([lat[index[:,0]],lon[index[:,1]]]).transpose() # should corresponds to (H,W), H[0]-lowest lat
    return index,sta_user,ll_sta,varm_hr,ind_varm

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
    
    nrep = mod_para.nrep
    # rep = list(range(0,nrep))
    rep = [0]
    epoc_num =[100] #
    key_ep_sort = 0 # to use epoc here or load sorted epoc no. 
    nepoc = 3 # no. of sorted epochs for analysis

    keyplot = 1
    opath_st = 'statistics' + suf +'_mk'+str(kmask)+'/'
    
    # select a range of data for testing 
    tlim = [datetime(2021,11,29),datetime(2021,12,1)]
    tlim = [datetime(2021,1,26),datetime(2021,1,28)]
    # tlim = [datetime(2021,1,16),datetime(2021,1,18)]
    dt = 1

    files_hr, indt_hr = make_list_file_t(dir_hr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    files_lr, indt_lr = make_list_file_t(dir_lr,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24)
    nfile = len(files_lr)
    
    # get logitude and latitude of data 
    nc_f = files_hr[0]
    lon = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[1]
    lat = nc_load_vars(nc_f,var_hr[0],[0],lats=ll_hr[0],lons=ll_hr[1])[2]

    # get hr test data 
    hr_all_test = np.zeros((nfile,nchl_o,len(lat),len(lon)))
    for i in range(nfile):
        for ichl in range(nchl_o): 
            dat_hr =  nc_load_vars(files_hr[i],var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[3]
            # mask =  nc_load_vars(files_hr[i],var_hr[0],[indt_hr[i]],lats=ll_hr[0],lons=ll_hr[1])[4]
            # dat_hr[mask] = np.nan
            hr_all_test[i,ichl,:,:] = dat_hr
    
    tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d')
    nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps
    tuser0 = [(tlim[0] + timedelta(hours=x)) for x in range(0,nt)]
    tshift = 0 # in hour
    tuser = [(tlim[0] + timedelta(hours=x)) for x in range(tshift,nt+tshift)] # time shift for numerical model
    
    # iday0 = (tlim[0] - datetime(2017,1,2)).days+1 # schism out2d_interp_001.nc corresponds to 2017.1.2
    # iday1 = (tlim[1] - datetime(2017,1,2)).days+1
    # id_test = np.arange(iday0,iday1)
    # files_lr = [dir_lr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]  # schism output
    # files_hr = [dir_hr + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]

    ntpd = int(24)
    index,sta_user,ll_sta,varm_hr_test,ind_varm = select_sta(hr_all_test,ivar_hr,lon,lat)
    
    out_path0 = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf
    os.makedirs(out_path0, exist_ok=True)    

    cuda = torch.cuda.is_available()
    
    for irep in rep:

        print(f'Repeat {irep}')
        print('--------------------------------')
    
        # suf0 = '_res' + str(opt.residual_blocks) + '_max_var1'
        ipath_nn = path_par+'nn_mod_' + str(opt.up_factor) + suf +'/' # 
    
        out_path = path_par+'results_pnt/'+'S'+str(opt.up_factor)+suf+'_ep'+'_re'+ str(irep)+'_mk'+str(kmask)+'/'
        os.makedirs(out_path, exist_ok=True)
        
        # Initialize generator 
        if knn ==0:
            generator = GeneratorResNet(in_channels=nchl_i, out_channels=nchl_o,
                                        n_residual_blocks=opt.residual_blocks,up_factor=opt.up_factor).eval()
        # discriminator = Discriminator(input_shape=(nchl_o, *hr_shape))
        elif knn == 1:
            generator = SRRes(in_channels=nchl_i, out_channels=nchl_o,n_residual_blocks=opt.residual_blocks,
                                        up_factor=opt.up_factor,kernel_size=ker_size,kernel_no=kernel_no).eval()
    
        if key_ep_sort:
            # choose sorted epoc number that gives the smallest rmse
            ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort_rmse.csv' #  rank based on 99 percentile, only saved for last var
            # ofname = "srf_%d_re%d_ep%d_%d_mask" % (opt.up_factor,irep,40,100) + '_epoc_sort'+tstr+'.csv' # rank based on rt_use highest ssh/
            ep_sort = np.loadtxt(opath_st + ofname, dtype=int,delimiter=",")  # load 
            epoc_num = ep_sort.flatten()[0:nepoc*nchl_o] 
            epoc_num = list(set(epoc_num.tolist())) # remove duplicates,order not reseved
        
        txtname = out_path0+"/para_cmp"+'_rp%d_ep%d_%d'%(irep,epoc_num[0],epoc_num[-1])+tstr+'.txt'
        outfile = open(txtname, 'w')
        # outfile.write('# varm_hr_test, varm_lr_test\n')
        # np.savetxt(outfile, np.hstack((varm_hr_test,varm_lr_test)), fmt='%-7.4f,')
        # outfile.write('# irep: {0}\n'.format(irep))
        
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
            
            # hr_scale = transforms.Resize((opt.hr_height, opt.hr_width), Image.Resampling.BICUBIC,antialias=None)
            
            nsta = len(index)
            sr_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res1_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res2_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            hr_res3_sta = np.zeros(shape=(nsta,nchl_o,len(tuser0)))
            
            sr_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            hr_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            dif_varm = np.zeros(shape=(nchl_o,2,len(tuser0)))
            
            for indf in range(0,nfile):

                nc_f = files_hr[indf] 
                indt = indt_hr[indf] # time index in nc_f
                data = nc_normalize_vars([nc_f],var_hr,[indt],varm_hr,
                                                 ll_hr[0],ll_hr[1],kintp[1])  #(H,W,C)
                x = np.transpose(data,(2,0,1)) #(C,H,W)
                hr = torch.from_numpy(x)
                mask = nc_load_vars(nc_f,var_hr[0],[indt_hr[indf]],lats=ll_hr[0],lons=ll_hr[1])[4] #(1,H,W)
                mask = np.squeeze(mask)

                nc_f = files_lr[indf]
                indt = indt_lr[indf] # time index in nc_f
                data = nc_normalize_vars([nc_f],var_lr,[indt],varm_lr,
                                         ll_lr[0],ll_lr[1],kintp[0])  #(H,W,C)
                x = np.transpose(data,(2,0,1)) #(C,H,W)
                lr = torch.from_numpy(x)
                                
                lr = lr.reshape(1,lr.shape[0],lr.shape[1],lr.shape[2]) # 3d to 4d
                hr = hr.reshape(1,hr.shape[0],hr.shape[1],hr.shape[2]) # 3d to 4d
                
                # nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact
                hr_restore1 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bicubic') # default nearest;bicubic; input 4D/5D
                hr_restore2 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='bilinear') # default nearest;
                hr_restore3 = torch.nn.functional.interpolate(lr, scale_factor=opt.up_factor,mode='nearest') # default nearest;
        
                start = time.time()
                sr = generator(lr.float())
                end = time.time()
                elapsed = (end - start)
                print('cost ' + str(elapsed) + 's')
                
                sr_norm0 = var_denormalize(sr.detach().numpy(),varm_hr) # (N,C,H,W), flipud height back
                hr_norm0 = var_denormalize(hr.detach().numpy(),varm_hr)
                hr_restore1_norm0  = var_denormalize(hr_restore1.detach().numpy(),varm_hr)
                hr_restore2_norm0  = var_denormalize(hr_restore2.detach().numpy(),varm_hr)
                hr_restore3_norm0  = var_denormalize(hr_restore3.detach().numpy(),varm_hr)
                
                hr_norm0[:,:,mask] = np.nan
                mse = np.nanmean((sr_norm0 - hr_norm0) ** 2,axis=(0,2,3)) # mean for channel
        
                it = indf
                sr_varm[:,0,it] = sr_norm0.max(axis=(0,2,3)) # max for channel
                sr_varm[:,1,it] = sr_norm0.min(axis=(0,2,3)) # min for channel
                hr_varm[:,0,it] = np.nanmax(hr_norm0,axis=(0,2,3)) # max for channel
                hr_varm[:,1,it] = np.nanmin(hr_norm0,axis=(0,2,3)) # min for channel
                dif = sr_norm0 - hr_norm0
                dif_varm[:,0,it] = np.nanmax(dif,axis=(0,2,3)) # max for channel
                dif_varm[:,1,it] = np.nanmin(dif,axis=(0,2,3)) # min for channel
                
                for ip in range(nsta):
                    sr_sta[ip,:,it]=sr_norm0[:,:,index[ip,0],index[ip,1]]
                    hr_sta[ip,:,it]=hr_norm0[:,:,index[ip,0],index[ip,1]]
                    hr_res1_sta[ip,:,it]=hr_restore1_norm0[:,:,index[ip,0],index[ip,1]]
                    hr_res2_sta[ip,:,it]=hr_restore2_norm0[:,:,index[ip,0],index[ip,1]]
                    hr_res3_sta[ip,:,it]=hr_restore3_norm0[:,:,index[ip,0],index[ip,1]]
                        
            sr_varmt = np.array([sr_varm[:,0,:].max(axis=1), sr_varm[:,1,:].min(axis=1)]).transpose()
            hr_varmt = np.array([hr_varm[:,0,:].max(axis=1), hr_varm[:,1,:].min(axis=1)]).transpose()
            dif_varmt = np.array([dif_varm[:,0,:].max(axis=1), dif_varm[:,1,:].min(axis=1)]).transpose()
            
            outfile.write('# stations sr_varmt, hr_varmt, dif_varmt\n')
            np.savetxt(outfile, np.hstack((sr_varmt,hr_varmt,dif_varmt)), fmt='%-7.4f,')
            
            index_max = np.argmax(hr_sta,axis=2)
            index_min = np.argmin(hr_sta,axis=2)
            
            if keyplot == 1:
                # plot comparison for locations
                axlab = [['Time','ssh (m)'],['Time','u (m/s)'],['Time','v (m/s)'],
                         ['Time','uw (m/s)'],['Time','vw (m/s)'],['Time','swh (m)'],
                         ['Time','pwp (s)'],['Time','swh_ww (m)'],]
                leg = ['hr','sr','bicubic','bilinear','nearest']
                # line_sty=['k.','b','r-','m-','g-','c']
                line_sty=['ko','b','r-','m-','g-','c'] # 'kv',
                for ip in range(nsta):
                    for i in range(len(ivar_hr)):
                        var_sta = hr_sta[ip,i,:]
                        var = sr_sta[ip,i,:]
                        var_res1 = hr_res1_sta[ip,i,:]
                        var_res2 = hr_res2_sta[ip,i,:]
                        var_res3 = hr_res3_sta[ip,i,:]
                        time_lst = [tuser0,tuser0,tuser0,tuser0,tuser0]
                        date_lst = [var_sta,var,var_res1,var_res2,var_res3]
                        ich = ivar_hr[i]
                        # figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoc) +tstr+ sta_user[ip]+'.png'
                        figname = out_path+"/c%d_re%d_ep%d" % (ich,irep,epoch) +tstr+'_ll%4.3f_%4.3f'%(ll_sta[ip,1],ll_sta[ip,0])+'.png'
                        plot_line_list(time_lst,date_lst,tlim,figname,axlab[ich],leg=leg,leg_col=2,line_sty=line_sty)
                    
        outfile.close()
