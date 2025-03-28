#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:24:51 2024
compare metrics 
@author: g260218
"""

import os
import numpy as np

import pandas as pd
# import sys
# import importlib
path_par = "../"  # the path of parameter files, also used for output path

# load metrics from ensembes of multiple model with one run
def read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask):
    metrics_md = {'ep':[], 'mae': [], 'rmse': [], 'mae_99': [],'rmse_99': [],
                      'mae_01': [],'rmse_01': [],'mae_m': [],'rmse_m': [],
                      'mae_t': [],'rmse_t': [],} # 'mse': [], 
    irep = 0
    for i,mod in enumerate(mod_name):
        suf = '_res' + str(rs_blocks[i]) + '_v1'+mod
        opath_st = path_par+'stat' + suf +'_mk'+str(kmask)+'_ave/'
        ofname = "srf_%d_re%d_ep%d_%d_mask" % (up_factor[i],irep,epoc_num[0],epoc_num[-1]) + '_test_metrics.npy'
        metrics = np.load(opath_st + ofname,allow_pickle='TRUE').item()
        for key in metrics_md.keys():
            temp = np.concatenate(metrics[key], axis=0) # convert to array, note if nchl>1, 2d array
            metrics_md[key].append(temp)  # list of arrays
    for key in metrics_md.keys():
        metrics_md[key] = np.stack(metrics_md[key], axis=1) # combine lists to array
    return metrics_md


# Function to write dictionary of 2D arrays side by side into a CSV using pandas
def dict_2Darrays_to_csv_pd(data, filename):
    # Create a list to store the DataFrames
    dfs = []
    
    for key, array in data.items():
        # Convert the 2D array to a DataFrame
        df = pd.DataFrame(array)
        # Rename the columns to include the array name
        df.columns = [f'{key}_col{i+1}' for i in range(df.shape[1])]
        # Append the DataFrame to the list
        dfs.append(df)
    
    # Concatenate the DataFrames horizontally (side by side)
    result_df = pd.concat(dfs, axis=1)
    # Write the resulting DataFrame to a CSV file
    result_df.to_csv(filename, index=False)
    

# load metrics from sorted data from direct interpolation
def plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',
             leg_col=2,style='default',ylim=None):
    # data(M,N) array: N bar at M xticks
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 

    x = np.arange(data.shape[0])
    dx = (np.arange(data.shape[1])-data.shape[1]/2.)/(data.shape[1]+2.)
    d = 1./(data.shape[1]+2.)
    
    plt.style.context(style) # 'seaborn-deep', why not working
    plt.style.use(style) # 'seaborn-deep', why not working

    fig, ax=plt.subplots(figsize=figsize,layout="constrained")
    for i in range(data.shape[1]):
        ax.bar(x+dx[i],data[:,i], width=d, label=leg[i], zorder=3)
    ax.set_xticks(np.arange(len(ticklab))) # for missing 1st tick
    ax.set_xticklabels(ticklab,rotation = 30,fontsize=size_label)
    ax.set_xlabel(axlab[0],fontsize=size_label)
    ax.set_ylabel(axlab[1],fontsize=size_label)
    if len(axlab)>=3:
        ax.set_title(axlab[2], fontsize=size_label)

    ax.tick_params(axis="both", labelsize=size_label-1)
    ax.grid(zorder=0)
    if ylim:
        ax.set_ylim(ylim)
    plt.legend(ncol=leg_col,fontsize=size_label-2,borderpad=0.2,handlelength=1.0,
               handleheight=0.3,handletextpad=0.4,labelspacing=0.2,columnspacing=0.5)  # framealpha=1
    if capt is not None: 
        trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        plt.text(0.03, 1.06, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    # plt.show()
    plt.close(fig) 


if __name__ == '__main__':
    
    epoc_num=[40,100]
    epoch0,epoch1 = 100, 100-30
    epoc_num = np.arange(epoch0,epoch1,-1)  # use the last 30 epochs for average
    
    kmask = 1 
    out_path = path_par+'cmp_ens_metrics/'
    os.makedirs(out_path, exist_ok=True)

# =============================================================================
    # # scale factor
    mod_name= ['par55e','par55e_s40','par55e_s80'] 
    up_factor = [20,40,80]
    rs_blocks = [6,6,6]
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    style='default' # '' seaborn-deep
    leg = ['s20','s40','s80']
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.25]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(b)',
             leg_col=3,ylim=ylim)

# =============================================================================
    # # scale factor
    mod_name= ['par534e','par534e_s20','par534e_s40'] 
    up_factor = [10,20,40]
    rs_blocks = [6,6,6]
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    style='default' # '' seaborn-deep
    leg = ['s10','s20','s40']
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_s%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.95]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(b)',
             leg_col=3,ylim=ylim)
    
# =============================================================================
    # input sample length
    id_use = [0,2,3,4,5,6]
    mod_name= ['par55e_dt48','par55e_dt24','par55e_dt12','par55e_dt6',
               'par55e','par55e_dt2','par55e_dt1']
    mod_name = [mod_name[i] for i in id_use]
    up_factor = [20]*len(mod_name)
    rs_blocks = [6]*len(mod_name)
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    style='default' # '' seaborn-deep
    leg = ['dt48','dt24','dt12','dt6','dt3','dt2','dt1']
    leg = [leg[i] for i in id_use]
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.18]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(d)',
             leg_col=3,ylim=ylim)

# =============================================================================
    # input sample length
    id_use = [0,1,2,3,4,5]
    mod_name= ['par534e_dt48','par534e_dt12','par534e_dt6','par534e','par534e_dt2','par534e_dt1'] 
    leg = ['dt48','dt12','dt6','dt3','dt2','dt1']
    mod_name = [mod_name[i] for i in id_use]
    leg = [leg[i] for i in id_use]

    up_factor = [10]*len(mod_name)
    rs_blocks = [6]*len(mod_name)
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    style='default' # '' seaborn-deep
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_dt%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.55]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(d)',
             leg_col=3,ylim=ylim)

# =============================================================================

    # residual block
    mod_name= ['par55e_b2','par55e','par55e_b16']  # ,'par55e_b20','par55e_b24'
    up_factor = [20,20,20,20,20]
    rs_blocks = [2,6,16,20,24]
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_rb%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)
    
    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    leg = ['rb2','rb6','rb16',]
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_rb%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.24] # None
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(a)',ylim=ylim)

# =============================================================================

    # mod_name= ['par534e','par534e_b16','par534e_b20','par534e_b24'] 
    # up_factor = [10,10,10,10]
    # rs_blocks = [6,16,20,24]
    # leg = ['rb6','rb16','rb20','rb24']
    mod_name= ['par534e_b2','par534e','par534e_b16',] 
    up_factor = [10,10,10]
    rs_blocks = [2,6,16]
    leg = ['rb2','rb6','rb16']
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_rb%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)

    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_rb%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = [0,0.24] # None
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(b)',ylim=ylim)

# =============================================================================
# no. of input channels 
    # mod_name= ['par534e','par534e_ct3','par534e_ct6','par534e_ct12','par534e_ct24','par534e_ct3_1',] 
    # up_factor = [10]*len(mod_name)
    # rs_blocks = [6,6,6,6,6,16]
    # # leg = ['0h','3h_dt3','6h_dt3','12h_dt6','24h_dt6']
    # leg = ['0h','3h','6h','12h','24h','3h_1',]
    mod_name= ['par534e','par534e_ct3','par534e_ct6','par534e_ct12','par534e_ct24'] 
    up_factor = [10]*len(mod_name)
    rs_blocks = [6,6,6,6,6]
    # leg = ['0h','3h_dt3','6h_dt3','12h_dt6','24h_dt6']
    leg = ['0h','3h','6h','12h','24h']    
    metrics_ave_md = read_metric_ens_mod(mod_name,up_factor,rs_blocks,epoc_num,kmask)
    ofname = 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_rb%d'%len(mod_name)+'.csv' 
    # Write the dictionary to a CSV file
    dict_2Darrays_to_csv_pd(metrics_ave_md, out_path + ofname)

    ticklab  = ['mae','rmse','mae_m','rmse_m','mae_99','rmse_99','mae_01','rmse_01']
    data =  []
    for key in ticklab:
        data.append(metrics_ave_md[key][19])  # 20th ele, i.e. ensemble of 81-100
    data = np.stack(data,axis=0)
    figname = out_path + 'metrics_ave_mk'+str(kmask)+ mod_name[0]+'_ct%d'%len(mod_name)+'.png'
    axlab = ['','Error in SWH (m)','SR_en']
    figsize = [3.3,2]
    ylim = None # [0,0.18]
    plot_bar(data,figname,figsize,axlab,leg,ticklab,size_label=12,capt='(b)',
             leg_col=3,ylim=ylim)
