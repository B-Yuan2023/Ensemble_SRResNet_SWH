#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:46:42 2024
input longitude, latitude and time range for download cmems15 data
@author: g260218
"""

import sys
import os
import datetime as dt
import copernicusmarine as cm

# input in bashfile: python script lon0 lon1 lat0 lat1 t00 t11
lon0=float(sys.argv[1])
lon1=float(sys.argv[2])
lat0=float(sys.argv[3])
lat1=float(sys.argv[4])

# time 
datefmt="%Y-%m-%dT%H:%M:%S"
t00=dt.datetime.strptime(sys.argv[5],datefmt) # "2023-02-01T00:00:00"
t11=dt.datetime.strptime(sys.argv[6],datefmt)

# Black Sea,time resolution 1h, space 0.025 deg, 01/01/1950-31/12/2022
# lon0=27.25
# lon1=42
# lat0=40.5
# lat1=47

# t00=dt.datetime(2023,4,1,0,0,0,0) #start time download 
# t11=dt.datetime(2023,4,2,0,0,0,0) #endtime download time download 

step=dt.timedelta(hours=24)  		 # step witdth e.g. each 24h a new file
endhour=step-dt.timedelta(hours=1)

#folder
outdir='wave_%.1f_%.1f_%.1f_%.1f_h%d'%(lon0,lon1,lat0,lat1,t00.hour)+'/'


varfname = ["wave"]
varname = [["VHM0",]] # "VMDR","VTM02"
subdir = ["swh"]

for j in range(len(varfname)):
    out_path=outdir+subdir[j]+'/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

for i in range((t11-t00).days):
    t0=t00+step*i
    t1=t0+endhour
    print(t0)
    print(t1)
    for j in range(len(varfname)):
        var = varfname[j]
        filename = outdir+subdir[j]+'/'+t0.strftime("%Y%m%d")+'.nc'
        if not os.path.isfile(filename):
            # instantanours elev
            cm.subset(
              dataset_id="cmems_mod_blk_wav_my_2.5km_PT1H-i",
              #dataset_version="202309",
              variables=varname[j],
              minimum_longitude=lon0,
              maximum_longitude=lon1,
              minimum_latitude=lat0,
              maximum_latitude=lat1,
              start_datetime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
              end_datetime=t1.strftime("%Y-%m-%dT%H:%M:%S") ,
              #start_datetime="2023-02-01T00:00:00",
              #end_datetime="2023-02-01T01:00:00",
              force_download=True,
              output_filename = filename,
              output_directory='./'
               #--force-download
            )

            