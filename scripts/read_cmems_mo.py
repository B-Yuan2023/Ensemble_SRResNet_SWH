#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:04:54 2024
read wave buoy data from cmems 
@author: g260218
"""
import os
import numpy as np
from glob import glob
from funs_prepost import nc_load_depth
from funs_sites import (read_cmemes_MO,plot_sites_location,write_sites_info,
                        plt_pcolor_pnt,plot_sites_var) #,interp_var,check_dry
from datetime import datetime,timedelta


if __name__ == '__main__':
    
    # pfd=os.path.dirname(os.path.abspath(__file__)) # directory of this file 
    indir='/work/gg0028/g260218/Data/cmems_wave_BlackSea/obs/history/MO'
    outdir='../'+'obs_figs'
    
    # If it doesn't exist, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.chdir(outdir)
    
    # nc name meaning: 
    #    BS: Black Sea
    #    TS: timeseries/trajactories; PR profiles; WS wave spectra. 
    #    MO: mooring
    pref = 'BS_TS_MO_'
    
    ###################################################################
    # #  for all stations with specifiled string, only need run once
    kall = 1
    if kall: 
        files=list(np.sort(glob(indir+os.sep+pref+'*.nc'))[:]) #
        lats = []
        lons = []
        ts_all = []
        te_all = []
        sids = []
        pcode = []
        for index in range(len(files)): # len(files),1        
            infile=files[index]
            sid,platform_code,timed,ts,te,lon,lat,swh,_,_,_,_ = read_cmemes_MO(infile)
            lats.append(lat)
            lons.append(lon)
            ts_all.append(ts)
            te_all.append(te)
            sids.append(sid)
            pcode.append(platform_code)
    
        lats=np.array(lats)
        lons=np.array(lons)
            
        # figname = outdir+os.sep+'insitu_sites_mo'+'.png'
        # plot_sites_location(lats,lons,figname)
        ofname=outdir+os.sep+'site_info_mo' # 
        write_sites_info(sids,lons,lats,ts_all,te_all,ofname)
        
        # load depth file 
        nc_f = '/work/gg0028/g260218/Data/cmems_wave_BlackSea/blacksea_bathy.nc'
        depth,lon,lat,mask,_= nc_load_depth(nc_f)
        X, Y = np.meshgrid(lon, lat)
        sta_name = pcode
        figname = outdir+os.sep+'insitu_sites_mo_dep'+'.png'
        plt_pcolor_pnt(X,Y,depth, figname,lats,lons,sta_name=sta_name,figsize = [5,4],
                           cmap='bwr',clim=None,unit='Bathymetry (m)',title=None,axoff=0,capt=None)
    
    ###################################################################
    # plot selected stations
    files = []
    # files=list(np.sort(glob(indir+os.sep+pref+'*.nc'))[:]) # all stations

    sta_user = ['SPOT0772', 'SPOT0773', 'SPOT0776', 'WAVEB01', 'WAVEB02', 
                'WAVEB03', 'WAVEB04', 'WAVEB05', 'WAVEB06']
    sta_user = ['WAVEB04', 'WAVEB05', 'WAVEB06']  # within domain 
    # sta_user = ['SPOT0772']

    lats = []
    lons = []
    ts_all = []
    te_all = []
    sids_u = []
    # tlim = [datetime(2017,10,25),datetime(2017,10,30)]
    tlim = None
    if tlim is not None:
        tstr = '_'+tlim[0].strftime('%Y%m%d')+'_'+tlim[1].strftime('%Y%m%d')
    else:
        tstr = '_tall'
    varlim = None # [5.2, -4] upper and lower limit of var 
    if varlim is not None:
        suflim = '_lim'+str(varlim[0])+'_'+str(varlim[1])
    else:
        suflim = ''
    cri_QC = 3 # None or 1-9
    if cri_QC is not None:
        sufqc = '_qc'+str(cri_QC)
    else:
        sufqc = ''
    suf = sufqc+suflim
    for i in range(len(sta_user)): # len(files),1        
#         infile=files[index]  #  use specified files
        infile = indir+os.sep+pref+sta_user[i]+'.nc' # use selected stations
        
        sid,_,timed,ts,te,lon,lat,swh,VHM0_QC,_,_,_ = read_cmemes_MO(infile,cri_QC=cri_QC,varlim=varlim)
        figname = outdir+os.sep+sta_user[i]+tstr+'_swh'+suf+'.png'
        plot_sites_var(timed,swh,tlim=tlim,figname=figname)
        figname = outdir+os.sep+sta_user[i]+tstr+'_qc'+suf+'.png'
        plot_sites_var(timed,VHM0_QC,tlim=tlim,figname=figname)
        
        lats.append(lat)
        lons.append(lon)
        ts_all.append(ts)
        te_all.append(te)
        sids_u.append(sid)

    lats=np.array(lats)
    lons=np.array(lons)
    ofname=outdir+os.sep+'site_info_mo_user' # 
    write_sites_info(sids_u,lons,lats,ts_all,te_all,ofname)
    
#     figname = outdir+os.sep+'insitu_sites_user'+'.png'
#     plot_sites_location(lats,lons,figname)
#     region = [0,52,12, 64] # [lon,lat,lon,lat] lat/lon of bottom left and top right
#     plot_latlon_points(lats,lons,outdir,region)

    

