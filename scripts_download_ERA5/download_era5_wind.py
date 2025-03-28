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
import cdsapi

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
outdir='wind_%.1f_%.1f_%.1f_%.1f_h%d'%(lon0,lon1,lat0,lat1,t00.hour)+'/'

varname = [['10m_u_component_of_wind', '10m_v_component_of_wind',]]
subdir = ["wind_10m"]

for j in range(len(subdir)):
    out_path=outdir+subdir[j]+'/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

c = cdsapi.Client()

for i in range((t11-t00).days):
    t0=t00+step*i
    t1=t0+endhour
    print(t0)
    print(t1)
    year = t0.strftime("%Y")
    month = t0.strftime("%m")
    day = t0.strftime("%d")
    for j in range(len(varname)):
        filename = outdir+subdir[j]+'/'+t0.strftime("%Y%m%d")+'.nc'
        if not os.path.isfile(filename):
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': varname[j],
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [lat1, lon0, lat0,lon1,],
                },
                filename)

            
