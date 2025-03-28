#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:04:54 2024
read wave buoy data from cmems 
@author: g260218
"""
import os
import numpy as np
# from glob import glob
from funs_prepost import nc_load_depth
from funs_sites import interp_var #,interp_var,check_dry
# from datetime import datetime,timedelta


if __name__ == '__main__':
    
    # pfd=os.path.dirname(os.path.abspath(__file__)) # directory of this file 
    indir='/work/gg0028/g260218/Data/cmems_wave_BlackSea/obs/history/MO'
    outdir='../'+'obs_figs'
    
    # If it doesn't exist, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.chdir(outdir)
    
    sta_name = ['WAVEB01', 'WAVEB02', 'WAVEB03',
                'WAVEB04', 'WAVEB05', 'WAVEB06',
                'SPOT0772','SPOT0773','SPOT0776','Max']
    # sta_name = ['WAVEB04', 'WAVEB05', 'WAVEB06']  # within domain 
    ll_stas = np.array([[27.941500,43.194200],[27.556200,42.511700],[27.927700,42.114400],
                        [28.611600,43.539200],[27.906700,42.696400],[28.343800,43.371700],
                        [27.994700,43.182000],[27.899200,42.957600],[27.633200,42.504400],
                        [33.375,42.8]])
    # [9.969837188720703, 20.1270809173584, 43.50788116455078, 
    # 37.56419372558594, 40.2666130065918, 18.597515106201172, 
    # 19.63839340209961, nan, 20.380855560302734, 2218.39306640625]
    
    # ll_stas = np.array([[27.927700,42.114400],
    #                     [28.611600,43.539200],
    #                     [27.906700,42.696400],
    #                     [33.375,42.8]])
    id_use = [0,1,4,9]
    lat_sta = ll_stas[:,1][id_use]
    lon_sta = ll_stas[:,0][id_use]
    sta_name = ['P' + str(i) for i in range(1,len(lat_sta)+1) ]

    # load depth file 
    nc_f = '/work/gg0028/g260218/Data/cmems_wave_BlackSea/blacksea_bathy.nc'
    depth,lon,lat,mask,_= nc_load_depth(nc_f)
    X, Y = np.meshgrid(lon, lat)
    figname = outdir+os.sep+'study_domain_sites'+'.png'
    # plt_pcolor_pnt(X,Y,depth, figname,lat_sta,lon_sta,sta_name=sta_name,figsize = [5,4],
    #                    cmap='bwr',clim=None,unit='Bathymetry (m)',title=None,axoff=0,capt=None)

    depth_sta = []
    for i in range(len(lon_sta)):
        temp=interp_var(lon_sta[i],lat_sta[i],lon,lat,depth,method='max')
        depth_sta.append(temp)
    ###################################################################
    # 
    from matplotlib import pyplot as plt
    import matplotlib.transforms as mtransforms    
    
    import matplotlib.ticker as ticker
    # Set x/y tick label formatter to show degrees with directional indicators
    def format_lon(value, tick_number):
        direction = 'W' if value < 0 else 'E'
        degrees = int(abs(value))
        minutes = int((abs(value) - degrees) * 60)
        return f'{degrees}°{minutes}\'{direction}' if minutes>0 else f'{degrees}°{direction}'
    
    def format_lat(value, tick_number):
        direction = 'S' if value < 0 else 'N'
        degrees = int(abs(value))
        minutes = int((abs(value) - degrees) * 60)
        return f'{degrees}°{minutes}\'{direction}' if minutes>0 else f'{degrees}°{direction}'
    
    
    # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
    # then adds a percent sign.
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"
        
    figsize = [7.4,5.0]  # # A4 8.3*11.7
    clim=None
    unit='Bathymetry (m)'
    title=None
    axoff=0
    capt=None
    
    # Define a custom colormap with specified color for bathymetry > 0
    from matplotlib.colors import LinearSegmentedColormap
    color_ = [(0, 0, 1), (1, 1, 1)]  # Blue to White
    color_ = [(0.3, 0.3, 1), (0.9, 0.9, 1)]  # Blue to White
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_)
    # cmap= 'bwr' # 'viridis'

    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(1,1,figsize=figsize) # default unit inch 2.54 cm
    size_tick = 12
    size_label = 12
    size_title = 14
    sample = depth
    
    # import matplotlib.colors as colors
    if clim:
        vmin, vmax = clim
        cf = ax.pcolormesh(X, Y, sample, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cf = ax.pcolormesh(X, Y, sample, cmap=cmap) # norm=colors.PowerNorm(gamma=0.5),
    cbar = fig.colorbar(cf, ax=ax,location='top', shrink=0.6)
    
    # add mark for stations if exist
    ll_shift = np.array([[0.06,0.32],[0.35,0.05],[0,0],
                              [0,0],[0.35,0.1],[0,0],
                              [0,0],[0,0],[0,0],
                              [0.35,0.1]])[id_use,:]
    if lat_sta is not None and lon_sta is not None:
        ax.scatter(lon_sta, lat_sta, s=40, marker="^",color='k')
        if sta_name is not None:
            for i in range(len(lon_sta)):
                ax.text(lon_sta[i]+ll_shift[i,0],lat_sta[i]+ll_shift[i,1],sta_name[i],
                        fontsize=size_tick,ha='center', va='top') #add text , transform=ax.transAxes

    cbar.ax.tick_params(labelsize=size_tick)
    if title:
        ax.set_title(title,fontsize=size_title)
    # if unit:
    #     cbar.set_label(unit,fontsize=size_tick-1)
    if not axoff: # keep axes or not 
        # ax.set_xlabel('lon',fontsize=size_label)
        # ax.set_ylabel('lat',fontsize=size_label)
        ax.tick_params(axis="both", labelsize=size_tick) 
    # plt.xticks(fontsize=size_tick)
    # plt.yticks(fontsize=size_tick)
    else:
        ax.axis('off')
    
    txt = 'm'
    loc_txt = [0.78,1.20]
    if txt is not None: 
        plt.text(loc_txt[0],loc_txt[1], txt,fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
    if capt is not None: 
        trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
    plt.tight_layout()
    
    CS=ax.contour(X, Y, depth, levels=[50,1000], colors='gray', linestyles='dashed', linewidths=1)
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=size_tick)

    # add box in the main plot 
    ax_main = ax    
    lon_main = [27.4, 33.4]
    lat_main = [40.6, 46.6]
    color_box = 'gray'
    lw_box = 2
    line_sty = '--'
    ax_main.plot(lon_main, [lat_main[0], lat_main[0]], line_sty, color=color_box, linewidth=lw_box)
    ax_main.plot(lon_main, [lat_main[1], lat_main[1]], line_sty, color=color_box, linewidth=lw_box)
    ax_main.plot([lon_main[0], lon_main[0]], lat_main, line_sty, color=color_box, linewidth=lw_box)
    ax_main.plot([lon_main[1], lon_main[1]], lat_main, line_sty, color=color_box, linewidth=lw_box)
        
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
    ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    
        
    # Plot the inset
    import cartopy
    import cartopy.crs as ccrs
    lon_range = [-14, 60]  # Longitude range (min, max)
    lat_range = [24, 70]  # Latitude range (min, max)
    # ax_inset = fig.add_axes([0.52, 0.18, 0.2, 0.25])  # Main figure as a subdomain of inset
    ax_inset = fig.add_axes([0.70, 0.51, 0.25, 0.25],projection=ccrs.PlateCarree())  # Main figure as a subdomain of inset
    # plot_polygon(coastline_load,linecolor='gray',fillcolor='lightgray',ax=ax_inset)
    ax_inset.coastlines()
    # add country border
    ax_inset.add_feature(cartopy.feature.BORDERS, color="grey", linewidth=0.5) 
    ax_inset.add_feature(cartopy.feature.OCEAN, color="azure") # blue
    # ax_inset.add_feature(cartopy.feature.LAND, color="cornsilk") # yellow
    # ax_inset.set_extent([-20, 60, 20, 70], crs=ccrs.PlateCarree())
    
    ax_inset.set_xlim(lon_range)
    ax_inset.set_ylim(lat_range)
    ax_inset.set_aspect('auto', adjustable='box')
    ax_inset.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
    ax_inset.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    x_ticks = np.linspace(lon_range[0], lon_range[1], 3)[1:3]
    y_ticks = np.linspace(lat_range[0], lat_range[1], 3)[1:3]
    ax_inset.set_xticks(x_ticks) 
    ax_inset.set_yticks(y_ticks) 
    ax_inset.tick_params(axis="both", labelsize=size_label-2)
    # ax_inset.tick_params(colors='white', which='both')  # 'both' refers to minor and major axes
    
    # Add a box indicating the plotted main figure region
    lon_main1 = [lon.min(),lon.max()]
    lat_main1 = [lat.min(),lat.max()]
    color_box = 'red'
    lw_box = 1
    ax_inset.plot(lon_main1, [lat_main1[0], lat_main1[0]], color=color_box, linewidth=lw_box)
    ax_inset.plot(lon_main1, [lat_main1[1], lat_main1[1]], color=color_box, linewidth=lw_box)
    ax_inset.plot([lon_main1[0], lon_main1[0]], lat_main1, color=color_box, linewidth=lw_box)
    ax_inset.plot([lon_main1[1], lon_main1[1]], lat_main1, color=color_box, linewidth=lw_box)
        
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    plt.show()

    
    

