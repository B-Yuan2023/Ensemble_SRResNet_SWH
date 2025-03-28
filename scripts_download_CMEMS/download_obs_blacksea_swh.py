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

# steps to download insitu data
# https://help.marine.copernicus.eu/en/articles/9133855-how-to-download-insitu-data-using-index-files
# Step 1. Get the list of all index files
# copernicusmarine get -i cmems_obs-ins_blk_phybgcwav_mynrt_na_irr --index-parts -nd
# the -nd option stands for --no-directories and allows to not reproduce the hierarchy of the server.
# Step 2. Filter the chosen index file
# filter files of interest from index_latest.txt
# save the output result in a text file (files_to_download.txt) before downloading:
grep -E "*TG." index_history.txt| cut -d ',' -f 2 | rev | cut -d '/' -f 1,2 | rev > files_to_download.txt
# Step 3. Download data
# cm get --dataset-id cmems_obs-ins_blk_phybgcwav_mynrt_na_irr --dataset-part latest --file-list files_to_download.txt


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


#varfname = ["obs"]
varname = [["VHM0","VMDR","VTM02"]] # "VMDR","VTM02"
subdir = ["obs"]

for j in range(len(subdir)):
    out_path=outdir+subdir[j]+'/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

# for i in range((t11-t00).days):
#     t0=t00+step*i
#     t1=t0+endhour
#     print(t0)
#     print(t1)
# for j in range(len(subdir)):
#     # var = varfname[j]
#     filename = outdir+subdir[j]+'/'+t0.strftime("%Y%m%d")+'.nc'
#     if not os.path.isfile(filename):
#         # instantanours elev
#         cm.subset(
#           dataset_id="cmems_obs-ins_blk_phybgcwav_mynrt_na_irr",
#           #dataset_version="202309",
#           variables=varname[j],
#           minimum_longitude=lon0,
#           maximum_longitude=lon1,
#           minimum_latitude=lat0,
#           maximum_latitude=lat1,
#           start_datetime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
#           end_datetime=t1.strftime("%Y-%m-%dT%H:%M:%S") ,
#           #start_datetime="2023-02-01T00:00:00",
#           #end_datetime="2023-02-01T01:00:00",
#           force_download=True,
#           output_filename = filename,
#           output_directory='./'
#            #--force-download
#         )

            
