#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:04:44 2023
compare numerical model output with buoy data

@author: g260218
"""

import os
import numpy as np
# from glob import glob
# from nc_funcs import *
from funs_prepost import nc_load_all,nc_load_cmems,plot_sites_cmp,plot_sites_cmpn,nc_var_normalize_cmems
from funs_sites import read_cmemes_TG,interp_var,check_dry
from datetime import datetime,timedelta

# from matplotlib import pyplot as plt
# from plot_latlon_points import plot_latlon_points


if __name__ == '__main__':
    up_factor = 16
    residual_blocks = 6
    suf = '_res' + str(residual_blocks)
    
    pfd=os.path.dirname(os.path.abspath(__file__)) # directory of this file 
    outdir = 'cmp_TG_figs' #+'SRF_'+str(up_factor)+suf 
    os.makedirs(outdir, exist_ok=True)
    
    tlim = [datetime(2017,10,26),datetime(2017,11,1)]
    tlim = [datetime(2017,1,9),datetime(2017,1,15)]
    # tlim = [datetime(2017,11,1),datetime(2018,2,5)]
    # tlim = [datetime(2017,11,15),datetime(2017,11,21)]
    # tlim = [datetime(2017,12,6),datetime(2017,12,12)]
    # tlim = [datetime(2018,1,2),datetime(2018,1,10)]
    
    # grid files, out2d_interp_001.nc corresponds to 2017.1.2
    nday = (tlim[1] - tlim[0]).days
    iday0 = (tlim[0] - datetime(2017,1,2)).days+1
    iday1 = (tlim[1] - datetime(2017,1,2)).days+1
    dif = int((tlim[1]-tlim[0]).total_seconds()/3600) ## time difference in hours
    tuser0 = [(tlim[0] + timedelta(hours=x)) for x in range(0,dif)]
    tshift = 2 # in hour 2
    tuser = [(tlim[0] + timedelta(hours=x)) for x in range(tshift,dif+tshift)] # time shift for numerical model
    id_test = np.arange(iday0,iday1)
    indir_nm='../../ll_region_lon8_lat53.5/out2d_128'
    files = [indir_nm + '/out2d_interp_' + str(i).zfill(3) + ".nc" for i in id_test]

    indir_cm='../nws7_8.0_8.9_53.5_54.0/' # input path for cmems reanalysis data
        
    # tlim = None
    if tlim is not None:
        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d')
    else:
        tstr = '_tall'
        
    # load interpolated schism grids    
    nc_f = files[0]
    lon = nc_load_all(nc_f)[1]
    lat = nc_load_all(nc_f)[2]
    ssh0 = nc_load_all(nc_f,0)[3]
    mask = nc_load_all(nc_f,0)[10]
    
    # load cmems grids
    varname = ["zos","uo","vo"] # varname from cmems
    subdir = ["ssh","u","v"] # subdir to save each var
    dir_sub = indir_cm + subdir[0]
    lon_cm = nc_load_cmems(dir_sub,0,varname[0])[1]
    lat_cm = nc_load_cmems(dir_sub,0,varname[0])[2]
    mask_cm = nc_load_cmems(dir_sub,0,varname[0],0)[4]
    
    data_dm0 = nc_var_normalize_cmems(indir_cm,indf=301,indt=0,ivar=[3],varmaxmin=None)
    
# only compare schism with cmems, use non-masked point location in cmems 
    for i,j in zip(*np.where(mask_cm)):
        # i,k = 5,7
        ssh_grd= []
        ssh_grd_cm= []
        # use points 
        lon_sta = lon_cm[i]
        lat_sta = lat_cm[j]
        out = check_dry(lon_sta,lat_sta,lon,lat,ssh0) # 0 means in wet area inside
        if out==0: 
            for k in range(0,nday):
                nc_f = files[k]
                ssh = nc_load_all(nc_f)[3] # shape is time, Ny*Nx
                indf = id_test[k] # day index for the selected days, refer to 20170102
                ssh_cm = nc_load_cmems(dir_sub,indf,varname[0])[3] # load all data 24 hours
                for it in range(len(ssh)):
                    temp = interp_var(lon_sta,lat_sta,lon,lat,ssh[it,:,:],method='max')
                    ssh_grd.append(temp)
                    temp = ssh_cm[it,j,i]
                    ssh_grd_cm.append(temp)
    
            figname = outdir+os.sep+'ll%.2f_%.2f'%(lon_sta,lat_sta)+tstr+'_ssh_compare_cm_s%d'%tshift+'.png'
            time_lst = [tuser,tuser0]
            dat_lst = [ssh_grd,ssh_grd_cm]
            leg = ['schism','cmems7']
            plot_sites_cmpn(time_lst,dat_lst,tlim=tlim,figname=figname,axlab=None,leg=leg,
                               leg_col=1, legloc=None,line_sty=None,style='default',capt='')
        
    
    # compare with selected gauging stations
    indir='/work/gg0028/g260218/Data/cmems_obs/NWS_cmems/TG'
    # sta_user = ['BallumTG','HelgolandTG','CuxhavenTG','NorderneyTG','BuesumTG',
    #          'BorkumTG','BremerhavenTG','EiderSPTG','HoernumTG','WilhelmshavenTG',
    #          'EmdenTG','EemshavenTG','HusumTG','HuibertgatTG','AlteWeserTG','WangeroogeTG']
    sta_user = ['CuxhavenTG','BremerhavenTG','AlteWeserTG',] # 'WangeroogeTG'
    ll_shift = [[0,0],[0,0],[0,0],[0,0]] # shift the station to the water region, lon,lat 0.51/128*2.0
    nsta = len(sta_user)
    lat_stas = []
    lon_stas = []
    ts_all = []
    te_all = []
    sids = []
    
    sshlim = [5.2, -4] # upper and lower limit of ssh 
    if sshlim is not None:
        suflim = '_lim'+str(sshlim[0])+'_'+str(sshlim[1])
    else:
        suflim = ''
    cri_QC = 2
    if cri_QC is not None:
        sufqc = '_qc'+str(cri_QC)
    else:
        sufqc = ''
    suf = sufqc+suflim

    ssh_grd = [None] * nsta
    ssh_grd_cm = [None] * nsta 
    
    for i in range(nsta): #
        infile = indir+os.sep+'NO_TS_TG_'+sta_user[i]+'.nc'
        sid,timed,ts,te,lon_sta,lat_sta,ssh_sta,SLEV_QC = read_cmemes_TG(infile,cri_QC,sshlim)
        # it0 = np.where(timed==tlim[0])[0][0] # if timed is array object
        # it1 = np.where(timed==tlim[1])[0][0]
        # it0 = timed.index(tlim[0]) # if timed is list
        # it1 = timed.index(tlim[1])
        ituse = np.where(np.isin(timed,tuser0))[0]  # dt can be 10 min, some t is missing
        tref = datetime(1970, 1, 1)
        times =  [(j - tref).total_seconds() for j in timed]
        tuser0s =[(j - tref).total_seconds() for j in tuser0]
        int_mod = 'linear' #'linear','slinear', 'quadratic' and 'cubic'; nan makes the latter 2 not work
        # valid = np.nonzero(~np.isnan(ssh_sta.squeeze()))[0]
        # valid = np.argwhere(~np.isnan(ssh_sta.squeeze()))
        # f = interpolate.interp1d(np.array(times)[valid].squeeze(),ssh_sta[valid].squeeze(),kind=int_mod) # float, same shape, 
        # ssh_sta_int = f(tuser0s)
        # # figname = outdir+os.sep+sta_user[i]+tstr+'_ssh'+suf+'.png'
        # # plot_sites_var(timed[it0:it1],ssh_TG[it0:it1],tlim=tlim,figname=figname)
        # # figname = outdir+os.sep+sta_user[i]+tstr+'_qc'+suf+'.png'
        # # plot_sites_var(timed,SLEV_QC,tlim=tlim,figname=figname)
        # figname = outdir+os.sep+sta_user[i]+tstr+'_ssh_sta_int_'+int_mod+'.png'
        # plot_sites_cmp(timed,ssh_sta,tuser0,ssh_sta_int,tlim,figname)
        
        # lat_stas.append(lat)
        # lon_stas.append(lon)
        # ts_all.append(ts)
        # te_all.append(te)
        # sids.append(sid)

        ssh_grd[i] = []
        ssh_grd_cm[i] = []
        
        # shift the station to water region
        lon_sta = lon_sta + ll_shift[i][0]
        lat_sta = lat_sta + ll_shift[i][1]
        for j in range(0,nday):
            nc_f = files[j]
            ssh = nc_load_all(nc_f)[3] # shape is time, Ny*Nx
            indf = id_test[j] # day index for the selected days, refer to 20170102
            ssh_cm = nc_load_cmems(dir_sub,indf,varname[0])[3]
            for it in range(len(ssh)):
                temp = interp_var(lon_sta,lat_sta,lon,lat,ssh[it,:],method='max')
                ssh_grd[i].append(temp)
                temp = interp_var(lon_sta,lat_sta,lon_cm,lat_cm,ssh_cm[it,:],method='max')
                ssh_grd_cm[i].append(temp)
        figname = outdir+os.sep+sta_user[i]+tstr+'_ssh_compare'+'.png'
        plot_sites_cmp(timed,ssh_sta,tuser,ssh_grd[i],tlim,figname) 
        
        figname = outdir+os.sep+sta_user[i]+tstr+'_ssh_compare_cm'+'.png'
        time_lst = [timed,tuser,tuser]
        dat_lst = [ssh_sta,ssh_grd[i],ssh_grd_cm[i]]
        leg = ['TG','schism','cmems7']
        plot_sites_cmpn(time_lst,dat_lst,tlim=tlim,figname=figname,axlab=None,leg=leg,
                           leg_col=1, legloc=None,line_sty=None,style='default',capt='')

#     lat_stas=np.array(lat_stas)
#     lon_stas=np.array(lon_stas)
#     ofname=outdir+os.sep+'site_info_user' # 
#     write_sites_info(sids,lat_stas,lon_stas,ts_all,te_all,ofname)
