#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:15:24 2024

pre-post functions for data

@author: g260218
"""

import numpy as np
from matplotlib import pyplot as plt
import netCDF4 

# make lists for files and the corresponding used time step, with repeated file names
import glob
from datetime import datetime, timedelta # , date
def make_list_file_t(dir_file,tlim,dt,tshift=0,t0_h=0,dt_h=1,ntpf=24):
    nt = int((tlim[1]-tlim[0]).total_seconds()/(dt*3600)) ## total time steps 
    # dir_file: directory of the data files
    # tlim: used time range, e.g., [datetime(2018,1,1,0,0,0),datetime(2022,1,1,0,0,0)]
    # dt: usd time step in hour, should be integer times of original time step in ncfiles
    # t0_h,dt_h = 0,1  # initial time and delta time (time step) in hour in ncfile
    # ntpf = 24  # number of time steps in each ncfile
    t_s = [(tlim[0] + timedelta(hours=x+tshift)) for x in range(0,nt)] # time list with time shift
    files0 = sorted(glob.glob(dir_file + "/*.nc"))  # all files in dir
    files0_ext = [ele for ele in files0 for i in range(ntpf)]  # list of files with repeat base on ntpf
    indt_ = [i for i in range(ntpf)]  # list of time steps in one file 
    indt0_ext = [ele for i in range(len(files0)) for ele in indt_ ]  # list of time steps in all files 
    ymd0 = t_s[0].strftime("%Y%m%d")
    indt0 = int((t_s[0].hour-t0_h)/dt_h)  # index of used inital time in an ncfile
    indf0 =[i for i, item in enumerate(files0) if ymd0 in item][0] # file index of first time step, file name e.g. 20200102.nc
    ind_ext0 = (indf0*ntpf+indt0) + np.arange(0, nt) * (dt/dt_h)  # index of used times in indt0_ext or files0_ext
    ind_ext = [int(i) for i in ind_ext0]  # change array to list 
    files = [files0_ext[i] for i in ind_ext]  # files with repeat, length=no. samples
    indt = [indt0_ext[i] for i in ind_ext]  # selected time indexs in all files, length=no. samples
    return files, indt


# read one var per file per time
def nc_load_vars(nc_f,varname,indt=None,lats=None,lons=None,kintp=0,method='linear'):
    # nc_f: string, nc file name
    # varname: string, variable name in ncfile, 1 var
    # indt: list, time steps in nc_f
    # import glob
    # nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    # print(f'filename:{nc_f}')
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # class Dataset: open ncfile, create ncCDF4 class
    # nc_fid.variables.keys() # 
    ncvar = list(nc_fid.variables)
    lon_var = [i for i in ncvar if 'lon' in i][0] # only work if only one var contain lon
    lat_var = [i for i in ncvar if 'lat' in i][0]
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables[lon_var][:]  # extract/copy the data
    lat = nc_fid.variables[lat_var][:]
    time = nc_fid.variables['time'][:]
    # dt_h = (time[1]-time[0])/3600  #  change time step in second to hour 
    if indt is None:
        indt = np.arange(0,len(time))  # read all times
    # else:
    #     indt = [int(nc_t[i]/dt_h) for i in range(len(nc_t))]  # indt is a list here, gives var[nt,Ny,nx] even nt=1

    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array
    if mask.size == 1: # in case all data are available
        mask = data!=data

    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    return time,lon,lat,data,mask


# normalize the data from ncfile 
def nc_normalize_vars(nc_f,varname,indt,varmaxmin=None,lats=None,lons=None,kintp=0,method='linear'):
    # output: (H,W,C)
    # nc_f: list of nc file name, length = no. of varname
    # varname: list, variable name in ncfile
    # indt: list, time index in nc_f, length = no. of varname
    nvar = len(varname)
    Nx = len(nc_load_vars(nc_f[0],varname[0],indt[0],lats,lons)[1])
    Ny = len(nc_load_vars(nc_f[0],varname[0],indt[0],lats,lons)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        # if len(nc_f) == 1:
        #     nc_fi,indti = nc_f[0],indt[0]
        # else:
        #     nc_fi,indti = nc_f[i],indt[i]

        var = nc_load_vars(nc_f[i],varname[i],[indt[i]],lats,lons,kintp,method)[3] # (NT,H,W) one channel
        
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flip(var,axis=1) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
    return data 


# denormalize 
def var_denormalize(var,varmaxmin):
    # var(N,C,H,W)
    nc = var.shape[1]
    var = np.flip(var,2) # flip the dimenson of height as in nc_normalize_vars, flip makes a copy
    data = np.zeros(shape=var.shape)
    for i in range(nc):
        vmax = varmaxmin[i,0]
        vmin = varmaxmin[i,1]
        data[:,i,:,:] = var[:,i,:,:]*(vmax-vmin) + vmin 
    return data 


# normalize 4d var directly to range (0,1)
def var_normalize(var,varmaxmin=None):
    # var(N,C,H,W), pay attention to the input dimension
    temp = var.copy()  # make a copy to avoid updating var (mutable)
    if len(var.shape)==2:  # (H,W)
        temp = temp.reshape([1,1,var.shape[0],var.shape[1]])
    elif len(var.shape)==3: # (N,H,W)
        temp = temp.reshape([var.shape[0],1,var.shape[1],var.shape[2]])
        
    nc = temp.shape[1]
    temp[np.isnan(temp)] = 0
    data = np.zeros(shape=temp.shape)
    for i in range(nc):
        if varmaxmin is None:
            vmax = var.max()
            vmin = var.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
        data[:,i,:,:] = (temp[:,i,:,:]-vmin)/(vmax-vmin)
    data[data<0] = 0  # make sure data>=0
    return data 


# calculate ssim usign torch for single 2D map, given var(N,C,H,W) in (0,1)
def ssim_tor(var1,var2):
    import torch
    from pytorch_msssim import ssim as ssim_torch

    N,C,H,W = var1.shape
    ssim = np.zeros((N,C))
    
    # if var is array convert to tensor
    if not torch.is_tensor(var1):
        var1 = torch.from_numpy(var1).clone()  # add clone to avoid change items in input array
    if not torch.is_tensor(var2):
        var2 = torch.from_numpy(var2).clone()
    for i in range(N):
        v1 = var1[i,:,:,:].reshape([1,C,H,W]) # 3d to 4d, required by ssim_torch
        v2 = var2[i,:,:,:].reshape([1,C,H,W])
        ssim[i,:] = ssim_torch(v1, v2,data_range=1.0).item()
    if N==1:
        ssim = np.squeeze(ssim,axis=0) # reduce dimension if N=1
    return ssim


# calculate ssim usign skimage for single 2D map, given var(N,C,H,W) in (0,1)
def ssim_skimg(var1,var2):
    from skimage.metrics import structural_similarity as ssim_skimg
    # var1,var2: input 4D array 
    
    N,C,H,W = var1.shape
    ssim = np.zeros((N,C))

    for i in range(N):
        for j in range(C):
            v1 = var1[i,j,:,:]
            v2 = var2[i,j,:,:]
            ssim[i,j] = ssim_skimg(v1, v2,data_range=1.0,
                              use_sample_covariance=False, gaussian_weights=True,
                              ) # win_size=11, sigma=1.5,channel_axis = 2
    return ssim
        

def nc_load_depth(nc_f,lats=None,lons=None,kintp=0,method='linear'):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
    # nc_fid.variables.keys() # list(nc_fid.variables)
    ncvar = list(nc_fid.variables) # 'depth' in schism, 'deptho' in cmems
    dep_var = [i for i in ncvar if 'dep' in i][0] # only work if only one var contain dep
    
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    depth = nc_fid.variables[dep_var][:] # shape is Ny*Nx
    mask = np.ma.getmask(depth)
    FillValue=0.0 # np.nan
    # depth = depth.filled(fill_value=FillValue)
    lon = np.ma.getdata(lon) # data of masked array
    lat = np.ma.getdata(lat) # data of masked array
    temp = np.ma.getdata(depth) # data of masked array
    vmax = temp.max()
    vmin = temp.min()
    data = temp # np.ma.getdata(temp) # strange, why need twice to get data
    data[np.isnan(data)] = FillValue
    if mask.size == 1: # in case all data are available
        mask = data!=data
    
    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    
    depth_nm = (data - vmin)/(vmax-vmin)
    depth_nm = np.flipud(depth_nm) # flipped normalized depth refer to mean sea level
    return depth,lon,lat,mask,depth_nm


# functions for read variables in nc files from schism output
def nc_load_all(nc_f,indt=None):
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
     # and create an instance of the ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))
    ssh = nc_fid.variables['elevation'][indt,:]  # shape is time, Ny*Nx
    uwind = nc_fid.variables['windSpeedX'][indt,:]  # shape is time, Ny*Nx
    vwind = nc_fid.variables['windSpeedY'][indt,:]  # shape is time, Ny*Nx
    swh = nc_fid.variables['sigWaveHeight'][indt,:]  # shape is time, Ny*Nx
    pwp = nc_fid.variables['peakPeriod'][indt,:]  # shape is time, Ny*Nx
    ud = nc_fid.variables['depthAverageVelX'][indt,:]  # shape is time, Ny*Nx
    vd = nc_fid.variables['depthAverageVelY'][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    
    mask = np.ma.getmask(ssh)
    
    FillValue=0.0 # np.nan
    ssh = ssh.filled(fill_value=FillValue)
    uwind = uwind.filled(fill_value=FillValue)
    vwind = vwind.filled(fill_value=FillValue)
    swh = swh.filled(fill_value=FillValue)
    pwp = pwp.filled(fill_value=FillValue)
    ud = ud.filled(fill_value=FillValue)
    vd = vd.filled(fill_value=FillValue)
    
    ssh = np.ma.getdata(ssh) # data of masked array
    uw = np.ma.getdata(uwind) # data of masked array
    vw = np.ma.getdata(vwind) # data of masked array
    swh = np.ma.getdata(swh) # data of masked array
    pwp = np.ma.getdata(pwp) # data of masked array
    ud = np.ma.getdata(ud) # data of masked array
    vd = np.ma.getdata(vd) # data of masked array
    return time,lon,lat,ssh,ud,vd,uw,vw,swh,pwp,mask


# instance or dataset normalization of schims output
# ivar = [3,4,5] # ssh, ud, vd
def nc_var_normalize(nc_f,indt,ivar,varmaxmin=None):
    nvar = len(ivar)
    Nx = len(nc_load_all(nc_f,indt)[1])
    Ny = len(nc_load_all(nc_f,indt)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        var = nc_load_all(nc_f,indt)[ivar[i]]
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flipud(var) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
                
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
        #data = np.array(data).reshape(data.shape[0],data.shape[1],1) # height, width, channel (top to bot)
    # data = np.dstack(data)
    # if nvar==1:
    #     data = np.repeat(data[..., np.newaxis], 3, -1)  # make 1 channel to 3 channels for later interpolation and trained model like vgg19
    return data 

import torch
from scipy.interpolate import griddata, interp2d, RBFInterpolator
# griddata no extrapolation, RBFInterpolator has extrapolation
def interpolate_tensor(tensor, scale_factor=None,x_in=None,y_in=None,x_out=None,y_out=None,
                       kintp=1, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of a tensor using the specified interpolation function.
        using either a scale factor or given xy coordinate
    Parameters:
        tensor (ndarray): Input tensor of shape (N, C, H, W).
        scale_factor (float): Scale factor for the last two dimensions. The new dimensions will be
                              original_dimensions * scale_factor.
        x_in,y_in,x_out,y_out: input and output coordinates of x,y
        kintp (function, optional): Interpolation function to use.
                                    Default is 1 griddata from scipy.interpolate; 2, RBFInterpolator
        method (str, optional): Interpolation method to use .
                                 Default is 'linear'.for 1 ('linear', 'nearest', 'cubic')
                                 for 2 kernel: 'linear', ‘thin_plate_spline’(default), ‘cubic’,...
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        new_tensor: Interpolated tensor of shape (N, C, new_H, new_W).
    """
    N, C, H, W = tensor.shape
    
    if scale_factor:  # use scale factor
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        
        # Create 2D grids for interpolation
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        X, Y = np.meshgrid(x, y)
        
        # Create new 2D grids for interpolated domain
        new_x = np.linspace(0, W - 1, new_W)
        new_y = np.linspace(0, H - 1, new_H)
        new_X, new_Y = np.meshgrid(new_x, new_y)
    else:  # use xy coordinate
        new_H, new_W = len(y_out),len(x_out)
        
        # Create 2D grids for input data
        X, Y = np.meshgrid(x_in, y_in)
    
        # Create new 2D grids for interpolated domain
        new_X, new_Y = np.meshgrid(x_out, y_out)

    new_tensor = np.zeros((N, C, new_H, new_W))
    
    for n in range(N):
        for c in range(C):
            # Flatten the original grid
            points = np.column_stack((X.flatten(), Y.flatten())) # 1D to 2D
            values = tensor[n, c].flatten()
            
            # remove nan points
            # Find the indices of non-NaN values
            ind_nan = torch.nonzero(~torch.isnan(values)).squeeze().tolist()
            values = values[ind_nan]  # filter out nan
            points = points[ind_nan,:]
            
            # Interpolate using the specified interpolation function
            if kintp==1:
                interpolated = griddata(points, values, (new_X, new_Y), method=method, **kwargs)
            elif kintp==2:
                # values[np.isnan(values)] = 0
                new_points = np.stack([new_X.ravel(), new_Y.ravel()], -1)  # shape (N, 2) in 2d
                interpolated = RBFInterpolator(points, values, kernel=method, **kwargs)(new_points).reshape(new_X.shape)
            new_tensor[n, c] = interpolated
    new_tensor = torch.from_numpy(new_tensor).type(torch.float)
    return new_tensor


def interp4d_tensor_nearest_neighbor(tensor4d, scale_factors=None, lr_x=None, lr_y=None, hr_x=None, hr_y=None):
    """
    Perform nearest neighbor interpolation on the last two dimensions of a 4D tensor.

    Args:
        tensor4d (numpy.ndarray): 4D tensor of shape (N, C, height, width) with NaN as masked values.
        scale_factors (tuple): Scale factors (sx, sy) for x and y directions (optional).
        lr_x (numpy.ndarray): 1D array of x-coordinates for the low-resolution grid (optional if scale_factors is provided).
        lr_y (numpy.ndarray): 1D array of y-coordinates for the low-resolution grid (optional if scale_factors is provided).
        hr_x (numpy.ndarray): 1D array of x-coordinates for the high-resolution grid (optional if scale_factors is provided).
        hr_y (numpy.ndarray): 1D array of y-coordinates for the high-resolution grid (optional if scale_factors is provided).

    Returns:
        numpy.ndarray: Interpolated 4D tensor with the same time and channels dimensions, but higher spatial resolution.
    """
    N, C, lr_height, lr_width = tensor4d.shape
    array4d = tensor4d.numpy()

    if scale_factors is not None:
        # Generate LR and HR grid coordinates using scale factors
        # If scale_factors is a single value, use it for both directions
        if isinstance(scale_factors, (int, float)):
            sx = sy = scale_factors
        elif isinstance(scale_factors, (tuple, list)) and len(scale_factors) == 2:
            sx, sy = scale_factors
        else:
            raise ValueError("scale_factors must be a single value or a tuple/list of two values.")
            
        lr_x = np.arange(lr_width)
        lr_y = np.arange(lr_height)
        hr_x = np.linspace(0, lr_width - 1, lr_width * sx)
        hr_y = np.linspace(0, lr_height - 1, lr_height * sy)
    elif lr_x is not None and lr_y is not None and hr_x is not None and hr_y is not None:
        # Validate the input coordinates
        if len(lr_x) != lr_width or len(lr_y) != lr_height:
            raise ValueError("The dimensions of lr_x and lr_y must match the spatial dimensions of the tensor.")
    else:
        raise ValueError("Either scale_factors or all of lr_x, lr_y, hr_x, and hr_y must be provided.")

    # Create coordinate grids
    lr_xx, lr_yy = np.meshgrid(lr_x, lr_y)
    hr_xx, hr_yy = np.meshgrid(hr_x, hr_y)

    # Flatten LR grid coordinates
    lr_coords = np.column_stack([lr_xx.ravel(), lr_yy.ravel()])
    hr_coords = np.column_stack([hr_xx.ravel(), hr_yy.ravel()])

    # Initialize the high-resolution tensor
    hr_height = len(hr_y)
    hr_width = len(hr_x)
    hr_tensor = np.full((N, C, hr_height, hr_width), np.nan)

    # Iterate over time and channel dimensions to interpolate each 2D slice
    for t in range(N):
        for c in range(C):
            lr_data = array4d[t, c]

            # Flatten LR data and apply mask
            lr_data_flat = lr_data.ravel()
            valid_mask = ~np.isnan(lr_data_flat)
            valid_coords = lr_coords[valid_mask]
            valid_data = lr_data_flat[valid_mask]

            # Use broadcasting for nearest-neighbor interpolation
            distances = np.linalg.norm(hr_coords[:, None, :] - valid_coords[None, :, :], axis=2)
            nearest_idx = np.argmin(distances, axis=1)

            # Assign values from the nearest neighbors to HR grid
            hr_data_flat = valid_data[nearest_idx]
            hr_tensor[t, c] = hr_data_flat.reshape(hr_height, hr_width)

    new_tensor = torch.from_numpy(hr_tensor).type(torch.float)
    return new_tensor


def interp4d_tensor_bilinear(tensor4d, scale_factors=None, lr_x=None, lr_y=None,hr_x=None, hr_y=None):
    """
    Perform bilinear interpolation on the last two dimensions of a 4D tensor, handling NaN values.

    Parameters:
    -----------
    tensor4d : numpy.ndarray
        Input tensor of shape (N, C, height, width) with possible NaN values.
    scale_factors : int, float, tuple, or list, optional
        Scaling factor(s) for the x and y dimensions. 
        - A single value for uniform scaling.
        - A tuple/list of two values for independent scaling in x and y directions.
        lr_x (numpy.ndarray): 1D array of x-coordinates for the low-resolution grid (optional if scale_factors is provided).
        lr_y (numpy.ndarray): 1D array of y-coordinates for the low-resolution grid (optional if scale_factors is provided).
        hr_x (numpy.ndarray): 1D array of x-coordinates for the high-resolution grid (optional if scale_factors is provided).
        hr_y (numpy.ndarray): 1D array of y-coordinates for the high-resolution grid (optional if scale_factors is provided).

    Returns:
    --------
    hr_tensor : numpy.ndarray
        High-resolution tensor of shape (N, C, hr_height, hr_width).
    """
    N, C, lr_height, lr_width = tensor4d.shape
    array4d = tensor4d.numpy()

    if scale_factors is not None:
        # Process scale factors
        if isinstance(scale_factors, (int, float)):
            sx = sy = scale_factors
        elif isinstance(scale_factors, (tuple, list)) and len(scale_factors) == 2:
            sx, sy = scale_factors
        else:
            raise ValueError("scale_factors must be a single value or a tuple/list of two values.")

        # Generate LR and HR grid coordinates using scale factors
        lr_x = np.arange(lr_width)
        lr_y = np.arange(lr_height)
        hr_x = np.linspace(0, lr_width - 1, int(lr_width * sx))
        hr_y = np.linspace(0, lr_height - 1, int(lr_height * sy))
    elif lr_x is not None and lr_y is not None and hr_x is not None and hr_y is not None:
        # shift the lr coordinate to 0,lr_width-1 and 0, lr_height-1 for easy weight estimation
        xshift,yshift = lr_x[0],lr_y[0]
        xtrans = (lr_width - 1)/(lr_x[-1]-lr_x[0])
        ytrans = (lr_height - 1)/(lr_y[-1]-lr_y[0])
        lr_x = (lr_x-xshift)*xtrans
        lr_y = (lr_y-yshift)*ytrans
        hr_x = (hr_x-xshift)*xtrans
        hr_y = (hr_y-yshift)*ytrans
    else:
        raise ValueError("Either scale_factors or lr_coords and hr_coords must be provided.")

    # Create 2D coordinate grids for HR and LR
    lr_xx, lr_yy = np.meshgrid(lr_x, lr_y)
    hr_xx, hr_yy = np.meshgrid(hr_x, hr_y)

    # Flatten HR grid coordinates for vectorized computation
    hr_xx_flat = hr_xx.flatten()
    hr_yy_flat = hr_yy.flatten()

    # Find indices for bilinear interpolation
    x0 = np.floor(hr_xx_flat).astype(int)
    x1 = np.clip(x0 + 1, 0, lr_width - 1)
    y0 = np.floor(hr_yy_flat).astype(int)
    y1 = np.clip(y0 + 1, 0, lr_height - 1)

    # Calculate weights for bilinear interpolation
    # there is problem for hr grids outside of lr valid grids. 
    wx1 = hr_xx_flat - x0
    wx0 = 1 - wx1
    wy1 = hr_yy_flat - y0
    wy0 = 1 - wy1

    # Perform bilinear interpolation with NaN handling
    hr_tensor = np.zeros((N, C, hr_y.size, hr_x.size))
    for n in range(N):
        for c in range(C):
            lr_data = array4d[n, c]

            # Gather the values for the 4 surrounding points
            f00 = lr_data[y0, x0]
            f01 = lr_data[y1, x0]
            f10 = lr_data[y0, x1]
            f11 = lr_data[y1, x1]

            # Stack weights and values for NaN handling
            weights = np.stack([wy0 * wx0, wy1 * wx0, wy0 * wx1, wy1 * wx1], axis=1)
            values = np.stack([f00, f01, f10, f11], axis=1)

            # Mask NaN values in values
            valid_mask = ~np.isnan(values)
            valid_weights = weights * valid_mask

            # Normalize weights
            sum_weights = valid_weights.sum(axis=1, keepdims=True)
            sum_weights[sum_weights == 0] = np.nan  # Avoid division by zero
            normalized_weights = valid_weights / sum_weights

            # Compute interpolated values
            interp_values = np.nansum(normalized_weights * values, axis=1)
            hr_tensor[n, c] = interp_values.reshape(hr_y.size, hr_x.size)

    new_tensor = torch.from_numpy(hr_tensor).type(torch.float)

    return new_tensor


# from scipy.interpolate import griddata, interp2d, RBFInterpolator
# griddata: no extropolation; RBFInterpolator: with extropolation; interp2d not suggested
def interpolate_array(array_in,x_in,y_in,x_out,y_out, kintp=1, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of an array using the specified interpolation function.
    
    Parameters:
        array_in (ndarray): Input array of shape (C, H, W).
        x_in,y_in,x_out,y_out: input and output coordinates of x,y
        kintp (function, optional): Interpolation function to use.
              1 Default is griddata from scipy.interpolate, 2 RBFInterpolator
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').for griddata
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated array of shape (C, new_H, new_W).
    """
    C, H, W = array_in.shape
    new_H, new_W = len(y_out),len(x_out)
    array_out = np.zeros((C, new_H, new_W))
    
    # Create 2D grids for input data
    X, Y = np.meshgrid(x_in, y_in)

    # Create new 2D grids for interpolated domain
    new_X, new_Y = np.meshgrid(x_out, y_out)
    
    for c in range(C):
        # Flatten the original grid
        points = np.column_stack((X.flatten(), Y.flatten()))
        values = array_in[c].flatten()
        
        # Interpolate using the specified interpolation function
        if kintp==1:
            array_out[c] = griddata(points, values, (new_X, new_Y), method=method, **kwargs)
        elif kintp==2:
            values[np.isnan(values)] = 0
            new_points = np.stack([new_X.ravel(), new_Y.ravel()], -1)  # shape (N, 2) in 2d
            array_out[c] = RBFInterpolator(points, values, kernel=method, **kwargs)(new_points).reshape(new_X.shape)
    return array_out


def interpolate_array_scale(array_in,scale_factor,kintp=griddata, method='linear', **kwargs):
    """
    Interpolates the last two dimensions of an array using the specified interpolation function.
    
    Parameters:
        array_in (ndarray): Input array of shape (C, H, W).
        scale_factor (float): Scale factor for the last two dimensions. The new dimensions will be
                              original_dimensions * scale_factor.
        kintp (function, optional): Interpolation function to use.
                                                     Default is griddata from scipy.interpolate.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').
                                 Default is 'linear'.
        kwargs: Additional keyword arguments to be passed to the interpolation function.
        
    Returns:
        ndarray: Interpolated array of shape (N, C, new_H, new_W).
    """
    C, H, W = array_in.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    array_out = np.zeros((C, new_H, new_W))
    
    # Create 2D grids for interpolation
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Create new 2D grids for interpolated domain
    new_x = np.linspace(0, W - 1, new_W)
    new_y = np.linspace(0, H - 1, new_H)
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    for c in range(C):
        # Flatten the original grid
        points = np.column_stack((X.flatten(), Y.flatten()))
        values = array_in[c].flatten()
        
        # Interpolate using the specified interpolation function
        interpolated = kintp(points, values, (new_X, new_Y), method=method, **kwargs)
        array_out[c] = interpolated
    return array_out


# functions for read data in nc files from cmems an interpolate when necessary
# ivar = [3,4,5] # ssh, ud, vd
# varname = ["zos","uo","vo"] # varname from cmems
# ymd: e.g.'20170101', year month day string of the file to read
def nc_load_cmems(dir_sub,ymd,varname,indt=None,lats=None,lons=None,kintp=0,method='linear'):
    import glob
    nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    # print(f'filename:{nc_f}')
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # class Dataset: open ncfile, create ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))  # read all times
    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array

    # use user domain when lats,lons are specified
    if lats is not None and lons is not None:
        if kintp==0: # no interpolation, only select user domain, use original coordinate
            # Find indices of x_s and y_s in x and y arrays
            ind_x = np.array([np.argmin(np.abs(lon-lons[i])) for i in range(len(lons))])# np.searchsorted(lon, lons)
            ind_y = np.array([np.argmin(np.abs(lat-lats[i])) for i in range(len(lats))])
            if len(data.shape)==2: # only one time step
                data = data[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
                mask = mask[ind_y[:, np.newaxis],ind_x[np.newaxis, :]]
            else:
                data = data[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
                mask = mask[:, ind_y[:, np.newaxis], ind_x[np.newaxis, :]]
            lon,lat = lon[ind_x],lat[ind_y]
        else:  # interpolated to user domain, use new coordinate 
            data[mask] = np.nan 
            data = interpolate_array(data,lon,lat,lons,lats, kintp=kintp, method=method)
            mask = np.isnan(data)
            data = np.nan_to_num(data,nan=0)
            lon,lat = lons,lats
    return time,lon,lat,data,mask

# functions for read data in nc files from cmems
# ivar = [3,4,5] # ssh, ud, vd
# varname = ["zos","uo","vo"] # varname from cmems
# ymd: e.g.'20170101', year month day string of the file to read
def nc_load_cmems0(dir_sub,ymd,varname,indt=None):
    import glob
    # subdir = ["ssh","u","v"] # subdir to for vars from cmems
    # indf: index of the file to be read
    # varname = ["zos","uo","vo"] # varname from cmems
    # index of the time in a file to be read
    # files = sorted(glob.glob(dir_sub + "/*.nc")) # in this way, each file links to one file from schism
    # nc_f = files[indf]
    # nfiles = len(files)
    # print(f'dir_sub:{dir_sub},indf:{indf},nfile:{nfiles}\n')
    nc_f = sorted(glob.glob(dir_sub + "/*"+ymd+"*.nc"))[0] # use the file contain string ymd
    nc_fid = netCDF4.Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
     # and create an instance of the ncCDF4 class
    # nc_fid.variables.keys() # list(nc_fid.variables)
    # print(nc_fid)
    # Extract data from NetCDF file
    lon = nc_fid.variables['longitude'][:]  # extract/copy the data
    lat = nc_fid.variables['latitude'][:]
    time = nc_fid.variables['time'][:]
    if indt is None:
        indt = np.arange(0,len(time))
    var = nc_fid.variables[varname][indt,:]  # shape is time, Ny*Nx
    nc_fid.close()
    
    mask = np.ma.getmask(var)
    FillValue=0.0 # np.nan
    data = var.filled(fill_value=FillValue)
    data = np.ma.getdata(data) # data of masked array
    return time,lon,lat,data,mask

# normalize the data from cmems
# ivar = [3,4,5] # ssh, ud, vd
def nc_var_normalize_cmems(dir_fl,ymd,ivar,indt,varmaxmin=None,lats=None,lons=None,kintp=0,method='linear'):
    # output: (H,W,C)
    # indt: index of time, should have length of 1 when call this function
    
    varname = ["zos","uo","vo"] # varname from cmems
    subdir = ["ssh","u","v"] # subdir to save each var
    nvar = len(ivar)
    dir_sub = dir_fl + '/'+ subdir[0]
    Nx = len(nc_load_cmems(dir_sub,ymd,varname[0],indt,lats,lons)[1])
    Ny = len(nc_load_cmems(dir_sub,ymd,varname[0],indt,lats,lons)[2])
    data = np.zeros(shape=(Ny,Nx,nvar))
    for i in range(nvar):
        ichl = ivar[i]-3
        dir_sub = dir_fl + '/'+ subdir[ichl]
        var = nc_load_cmems(dir_sub,ymd,varname[ichl],indt,lats,lons,kintp,method)[3] # (NT,H,W) one channel
        
        # data = np.squeeze(data[indt,:,:])  # (Ny,Nx), lat,lon
        temp = np.flipud(var) # original data first row -> lowest latitude
        # convert data to [0,1]
        if varmaxmin is None:
            vmax = temp.max()
            vmin = temp.min()
        else:
            vmax = varmaxmin[i,0]
            vmin = varmaxmin[i,1]
        data[:,:,i] = (temp - vmin)/(vmax-vmin) # convert to [0,1]
    return data 


# find max and min of variable in the files
def find_maxmin_global(files, ivar=[3,3,3]):
    # files = sorted(glob.glob(dirname + "/*.nc"))
    # nfile = len(files)
    # files = files[:int(nfile*rtra)]
    file_varm = [] 
    ind_varm = np.ones((len(ivar),2),dtype= np.int64)
    varmaxmin = np.ones((len(ivar),2))
    varmaxmin[:,0] *= -10e6 # maximum 
    varmaxmin[:,1] *= 10e6 # minimum 
    for i in range(len(ivar)):
        for indf in range(len(files)):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            if varmaxmin[i,0]<var.max():
                varmaxmin[i,0] = var.max()
                ind_varm[i,0] = np.argmax(var)
                file_max = nc_f
            if varmaxmin[i,1]>var.min():
                varmaxmin[i,1] = var.min()
                ind_varm[i,1] = np.argmin(var)
                file_min = nc_f
            # varmaxmin[i,0] = max(varmaxmin[i,0],var.max())
            # varmaxmin[i,1] = min(varmaxmin[i,1],var.min())
        file_varm.append([file_max,file_min])
    return varmaxmin,ind_varm,file_varm


# sorted var in hour
ntpd = 24 # number of time steps in an nc file
def find_max_global(files, ivar=[3]):
    nfile = len(files)
    nvar = len(ivar)
    ind_sort = [[]]*nvar
    var_sort = [[]]*nvar
    # ind_file = [[]]*nvar
    # ind_it = [[]]*nvar
    for i in range(nvar):
        var_comb = []
        # var_file = []
        # var_it = []
        for indf in range(nfile):
            nc_f = files[indf]
            var = nc_load_all(nc_f)[ivar[i]]
            var_max = var.max(axis=(1,2)) # maximum in 2d space, note during ebb sl can be <0
            for indt in range(ntpd):
                var_comb.append(var_max[indt])
                # var_file.append(indf) # the indf th file, not file name index
                # var_it.append(indt)
        ind_sort[i] = sorted(range(len(var_comb)), key=lambda k: var_comb[k], reverse=True)
        var_sort[i] = [var_comb[k] for k in ind_sort[i]]
        # var_sort[i] = sorted(var_comb)
        # ind_file[i] = [var_file[k] for k in ind_sort[i]]
        # ind_it[i] = [var_it[k] for k in ind_sort[i]]
    return var_sort,ind_sort #,ind_file,ind_it


def plt_sub(sample,ncol,figname,ichan=0,clim=[0,1],cmp='bwr',contl=None):  
    # sample: array, normalized sample(nk,1,nx,ny)
    nsub = len(sample)
    columns = ncol
    rows = int(-(-(nsub/columns))//1)
    fig = plt.figure()
    for i in range(0,nsub):
        fig.add_subplot(rows, columns, i+1)        
        plt.imshow(sample[i,ichan,:,:],cmap=cmp) # bwr,coolwarm
        plt.axis('off')
        plt.clim(clim[0],clim[1]) 
        plt.tight_layout()
        #plt.title("First")
        if contl is not None: # add 0 contour
            plt.contour(sample[i,ichan,:,:], levels=contl, colors='black', linewidths=1)
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    
# def plt_contour_list(lat,lon,sample,figname,lev=11,cmap='bwr',clim=None,unit=None,title=None):  # sample is a list with k array[C,nx,ny]
#     nsub = len(sample)
#     fig, axes = plt.subplots(1, nsub, figsize=(5 * nsub, 5))
#     for i in range(0,nsub):
#         ax = axes[i] if nsub > 1 else axes
#         # ax.set_facecolor('xkcd:gray')
#         if clim:
#             vmin, vmax = clim[i]
#             cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#             # cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
#         else:
#             cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
#         cbar = fig.colorbar(cf, ax=ax)
#         ax.set_title(title[i] if title else f'Array {i + 1}')
#         if unit:
#             cbar.set_label(unit[i])
#         ax.set_xlabel('lon',fontsize=16)
#         ax.set_ylabel('lat',fontsize=16)
#         plt.tight_layout()
#     plt.savefig(figname,dpi=100) #,dpi=100    
#     plt.close(fig)
    
    
def plt_contour_list(lat,lon,sample,figname,lev=20,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None):  # sample is a list with k array[C,nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.contourf(lat,lon,sample[i],levels=np.linspace(vmin, vmax, lev),cmap=cmap)
        else:
            cf = ax.contourf(lat,lon,sample[i],levels=lev,cmap=cmap) # bwr,coolwarm
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            plt.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            plt.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)    
    
    
def plt_pcolor_list(lat,lon,sample,figname,subsize = [5,4],cmap='bwr',clim=None,unit=None,
                    title=None,nrow=1,axoff=0,capt=None,txt=None,loc_txt=None,xlim=None,ylim=None):  
    # sample is a list with k array[nx,ny]
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    size_tick = 16
    size_label = 18
    size_title = 18
    axes = axes.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axes[irm_ax[i]])
    for i in range(0,nsub):
        ax = axes[i] if nsub > 1 else axes
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolor(lat, lon, sample[i], cmap=cmap)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=size_tick)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)
        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None: 
            ax.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            ax.text(0.00, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()

    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)


def plt_pcolorbar_list(lat,lon,sample,figname,subsize = [2.5,2],fontsize=12,
                       cmap='bwr',clim=None,kbar=0,unit=None,title=None,nrow=1,
                       axoff=0,capt=None,txt=None,loc_txt=None,xlim=None,ylim=None):  
    # sample is a list with k array[nx,ny], or array [Nt,H,W]
    # kbar: control colorbar appearance
    import matplotlib.transforms as mtransforms    
    nsub = len(sample)
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    # figure layout,layout="constrained"
    fig, axs = plt.subplots(nrow, ncol,layout="constrained", figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    
    # size_tick,size_label,size_title = 16,18,18 # subsize = [5,4]
    size_tick,size_label,size_title = 10,12,12 # subsize = [2.5,2]
    size_tick,size_label,size_title = fontsize-2,fontsize,fontsize

    axs1 = axs.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    cf_a = []
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axs1[irm_ax[i]])
    for i in range(0,nsub):
        ax = axs1[i] if nsub > 1 else axs
        # ax.set_facecolor('xkcd:gray')
        if clim:
            vmin, vmax = clim[i]
            cf = ax.pcolormesh(lat, lon, sample[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolormesh(lat, lon, sample[i], cmap=cmap)
        cf_a.append(cf)
        ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)

        if not axoff: # keep axes or not 
            ax.set_xlabel('lon',fontsize=size_label)
            ax.set_ylabel('lat',fontsize=size_label)
            ax.tick_params(axis="both", labelsize=size_tick-1) 
        # plt.xticks(fontsize=size_tick)
        # plt.yticks(fontsize=size_tick)
        else:
            ax.axis('off')
        if txt is not None:  # adding text, Dont use plt.text that modify subfigs!
            ax.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick,ha='left', va='top', transform=ax.transAxes) #add text
        if capt is not None: 
            # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            # ax.text(0.06, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
            ax.text(0.01, 1.05, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # plt.tight_layout()  # conflict with layout constrained

    # plot colorbar 
    if kbar == 0:  # plot for every subplot
        for i in range(0,nsub):
            cbar = fig.colorbar(cf_a[i], ax=axs1[i])
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==1:  # plot 1 colorbar for each row, on the right, if no constrained layout, subfig size differs
        for i in range(0,nrow):
            cbar = fig.colorbar(cf_a[i*nrow+ncol-1], ax=[axs[i,-1]],location='right') # , shrink=0.6
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==2:  # plot 1 colorbar for each colume, on the bottom 
        for i in range(0,ncol):
            cbar = fig.colorbar(cf_a[i], ax=[axs[-1,i]],location='bottom', shrink=0.6)
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
    elif kbar==3:  # plot 1 colorbar for all rows, on the right, implicit
        cbar = fig.colorbar(cf, ax=axs[:,-1],location='right', shrink=0.6) # , shrink=0.6
    elif kbar==4:  # plot 1 colorbar for all columes, on the bottom 
        cbar = fig.colorbar(cf, ax=axs[-1,:],location='bottom', shrink=0.6)        
    elif kbar==5:  # plot 1 colorbar for all rows, on the right, explicit 
        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.93,
                            wspace=0.02, hspace=0.06)
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with width 0.02 and height 0.8
        cb_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
        cbar = fig.colorbar(cf, cax=cb_ax, shrink=0.6)
    elif kbar==6:  # plot 1 colorbar for all columes, on the bottom, explicit 
        fig.subplots_adjust(bottom=0.1, top=0.8, left=0.1, right=0.9,
                            wspace=0.02, hspace=0.02)
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with width 0.02 and height 0.8
        cb_ax = fig.add_axes([0.1, 0.83, 0.8, 0.02])
        cbar = fig.colorbar(cf, cax=cb_ax)
    elif kbar==7:  # plot 1 colorbar for 1st colume on right, 1 colorbar for bottom row on the bottom 
        # plot colorbar for 1st colume on the right
        for i in range(0,nrow):
            cbar = fig.colorbar(cf_a[i*ncol], ax=[axs[i,0]],location='right', shrink=0.6) # , shrink=0.6
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)
        # plot colorbar for bottom row on the bottom
        for i in range(1,ncol):
            cbar = fig.colorbar(cf_a[i], ax=[axs[-1,i]],location='bottom', shrink=0.6)
            cbar.ax.tick_params(labelsize=size_tick)
            if unit:
                cbar.set_label(unit[i],fontsize=size_tick+1)        
    if kbar in [3,4,5,6]:
        cbar.ax.tick_params(labelsize=size_tick)
        if unit:
            cbar.set_label(unit[i],fontsize=size_tick+1)          
            
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    # plt.savefig(figname,dpi=300) #,dpi=100    
    plt.close(fig)

    
def plotsubs_line_list(time_lst,dat_lst,figname=None,tlim=None,ylim=None,subsize = [2.5,2],
                            fontsize=12,nrow=1,title=None,axlab=None,leg=None,leg_col=1,
                            lloc=9,legloc=None,line_sty=None,line_col=None,marker=None,
                            capt=None,txt=None,loc_txt=None):
    """
    Plot a list of lists/2D arrays using subplots, each subplot corresponds to a sublist/array.
    Each subplot contains multiple lines. A single legend is added on top of the figure.
    
    Parameters:
    time_lst (list of lists)
    dat_lst (list of lists/2D arrays): each sublist/array contains lines to be plotted.
                               each line of sublist/array has the same no. of data as in time_lst[i][j]
    leg (list of str): List of legends for the lines.
                          The length of the list should be equal to the number of lines in each array.
    """
    
    nsub = len(dat_lst)
    nvar = len(dat_lst[0])  # all arrays in the list should have the same length
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    # figure layout,layout="constrained"
    fig, axs = plt.subplots(nrow, ncol, sharex=True,layout="constrained", figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
        
    # Ensure axs is iterable (handles case with a single subplot)
    if nsub == 1:
        axs = [axs]
    
    # A4 8.3*11.7
    # size_tick,size_label,size_title = 16,18,18 # subsize = [5,4]
    # size_tick,size_label,size_title = 10,12,12 # subsize = [2.5,2]
    size_tick,size_label,size_title = fontsize-1,fontsize,fontsize

    axs1 = axs.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axs1[irm_ax[i]])
    
    if line_sty and len(line_sty)>=nvar:
        line_sty = line_sty[:nvar]
    else:
        line_sty = ['-'] *nvar
        
    if line_col and len(line_col)>=nvar:
        line_col = line_col[:nvar]
    else:
        line_col = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if marker and len(marker)>=nvar:
        marker = marker[:nvar]
    else:
        marker = ['None'] *nvar
    
    for i,array in enumerate(dat_lst):
        ax = axs1[i] 
        # ax.set_facecolor('xkcd:gray')
        for j,line in enumerate(array):
            ax.plot(time_lst[i][j],line,linestyle=line_sty[j],color=line_col[j],marker=marker[j])
        
        # ax.set_title(title[i] if title else f'Array {i + 1}',fontsize=size_title)
        if title:
            ax.set_title(title[i],fontsize=size_title)
            
        if txt is not None:  # adding text, Dont use plt.text that modify subfigs!
            ax.text(loc_txt[0],loc_txt[1], txt[i],fontsize=size_tick-1,ha='left', va='top', transform=ax.transAxes) #add text
            
        if capt: 
            # import matplotlib.transforms as mtransforms    
            # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
            # ax.text(0.06, 1.00, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
            ax.text(0.01, 0.99, capt[i],fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    
        ax.tick_params(axis="both", labelsize=size_tick)     
        if tlim is not None:
            ax.set_xlim(tlim)
    #         plt.xlim(tlim)
        if ylim is not None:
            ax.set_ylim(ylim[i])
        if axlab:
            if isinstance(axlab[0], list):
                ax.set_xlabel(axlab[i][0],fontsize=size_label)
                ax.set_ylabel(axlab[i][1],fontsize=size_label)
            else:
                ax.set_xlabel(axlab[0],fontsize=size_label)
                ax.set_ylabel(axlab[1],fontsize=size_label)
    
    fig.autofmt_xdate()  # format xlable

    if leg:
        leg = leg[:nvar]
    else:
        leg = [str(i) for i in range(nvar)]

    # Create a single legend for all subplots
    if legloc: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C; U-upper
        fig.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,bbox_to_anchor=legloc,
                    frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)
    else: 
        fig.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,
                    frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)
        
    # Display the plot
    plt.tight_layout()
    if figname:
        plt.savefig(figname,bbox_inches='tight',dpi=300)
        plt.close(fig)
    plt.show()


def plot_line_list(time_lst,dat_lst,tlim=None,figname='Fig',figsize=None,fontsize=12,
                   axlab=None,leg=None,leg_col=1,lloc=0,legloc=None,
                   line_sty=None,style='default',capt=''):
    import matplotlib.transforms as mtransforms    
    
    # size_tick,size_label = 14,16  # one colume per page
    # size_tick,size_label = 10,12  # two colume per page
    size_tick,size_label = fontsize-2,fontsize  # two colume per page
    # size_title = 18
    
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    ndat = len(time_lst)
    # line_sty=['k','b','r','m','g','c']
    with plt.style.context(style):
        for i in range(ndat): 
            if line_sty is not None and len(line_sty)>=ndat:
                plt.plot(time_lst[i],dat_lst[i],line_sty[i]) # ,mfc='none'
            else:
                plt.plot(time_lst[i],dat_lst[i])
    fig.autofmt_xdate()
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    ax.tick_params(axis="both", labelsize=size_tick)     
    if tlim is not None:
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=size_label)
        plt.ylabel(axlab[1],fontsize=size_label)
    if leg is None:
        leg = [str(i) for i in range(ndat)]
    else:
        leg = leg[:ndat]
    if legloc is None: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C; U-upper
        plt.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,
                   frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)
    else: 
        plt.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,bbox_to_anchor=legloc,
                   frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)    
    plt.tight_layout()
    plt.savefig(figname,bbox_inches='tight',dpi=300)
    plt.close(fig)
    plt.show()
    
    
def plot_errbar_list(xlst,dat_lst,err_lst,tlim=None,figname='Fig',axlab=None,leg=None,
                   leg_col=1, legloc=None,line_sty=None,style='default',capt=''):
    import matplotlib.transforms as mtransforms    
    
    size_tick = 14
    size_label = 16
    # size_title = 18
    fig = plt.figure()
    ndat = len(xlst)
    # line_sty=['k','b','r','m','g','c']
    with plt.style.context(style):
        for i in range(ndat): 
            if line_sty is not None and len(line_sty)>=ndat:
                plt.plot(xlst[i],dat_lst[i],line_sty[i]) # ,mfc='none'
                plt.errorbar(xlst[i],dat_lst[i],err_lst[i], linestyle='None', marker='^', capsize=3)
            else:
                plt.plot(xlst[i],dat_lst[i])
                plt.errorbar(xlst[i],dat_lst[i],err_lst[i], linestyle='None', marker='^', capsize=3)
    fig.autofmt_xdate()
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=size_label,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    ax.tick_params(axis="both", labelsize=size_tick)     
    if tlim is not None:
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=size_label)
        plt.ylabel(axlab[1],fontsize=size_label)
    if leg is None:
        leg = [str(i) for i in range(ndat)]
    else:
        leg = leg[:ndat]
    plt.tight_layout()
    if legloc is None:
        plt.legend(leg,ncol=leg_col,fontsize=size_tick)
    else: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C
        plt.legend(leg,ncol=leg_col,fontsize=size_tick,loc=2,bbox_to_anchor=legloc)    
    plt.savefig(figname,bbox_inches='tight',dpi=300)
    plt.close(fig)
    plt.show()    
    

def plot_sites_cmp(time_TG,ssh_TG,time,ssh,tlim=None,figname=None,axlab=None,leg=None):
    fig = plt.figure()
    plt.plot(time_TG,ssh_TG,'k.')
    plt.plot(time,ssh,'b')
    fig.autofmt_xdate()
    if tlim is not None:
        ax = plt.gca()
        ax.set_xlim(tlim)
#         plt.xlim(tlim)
    if axlab is not None:
        plt.xlabel(axlab[0],fontsize=14)
        plt.ylabel(axlab[1],fontsize=14)
    if leg is None:
        leg = ['ref','mod']
    if figname is None:
        figname = 'Fig'
    plt.legend(leg)     
    plt.savefig(figname,dpi=100)
    plt.close(fig)
    plt.show()        
    
    
def plot_mod_vs_obs(mod,obs,figname,axlab=('Target','Mod',''),leg=None,alpha=0.3,
                    marker='o',figsize=(3,3),fontsize=12,capt=''):
    """
    Plot model data against observation data to visualize bias in the model.
    Parameters:
        mod (list of arrays): Model data to be plotted.
        obs (array-like): Observation data to be plotted.
        label (tuple, optional): Labels for x-axis, y-axis, and title.
    """
    import matplotlib.transforms as mtransforms    

    fig= plt.figure(figsize=figsize)
    # plt.style.use('seaborn-deep')

    if len(marker) < len(mod):
        marker = ['o' for i in range(len(mod))]
    # Plot the scatter plot
    for i in range(len(mod)):
        if leg is not None:
            plt.scatter(obs, mod[i], alpha=alpha, marker=marker[i],label=leg[i]) # marker=marker, color='blue',
        else:
            plt.scatter(obs, mod[i], alpha=alpha, marker=marker[i]) # marker=marker, color='blue',
    if len(mod)>1:
        if leg is None:
            leg = [str(i) for i in range(len(mod))]
        # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C
        hleg = plt.legend(fontsize=fontsize-1,loc=2,frameon=False) # no box
        # hleg.get_frame().set_linewidth(0.0)  # only remove box border
        
    # Set the same limits for x and y axes
    max_val = max(np.nanmax(obs), np.nanmax(np.array(mod)))
    min_val = min(np.nanmin(obs), np.nanmin(np.array(mod)))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)    
    # Set the same ticks for x and y axes
    ticks = plt.xticks()
    plt.xticks(ticks[0],fontsize=fontsize-2)
    plt.yticks(ticks[0],fontsize=fontsize-2)
    
    # Plot the perfect fit line (y = x)
    plt.plot(ticks[0], ticks[0], linestyle='dashed', color='black') 

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)

    plt.grid(True)
    ax = plt.gca()
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    
    
def plotsubs_mod_vs_obs(dat_mod,dat_ref,figname,subsize=(3,3),fontsize=12,nrow=1,
                        axlab=('Ref','Mod',''),leg=None,leg_col=None,lloc=9,legloc=None,
                        alpha=0.3,marker='o',capt='',txt=None,loc_txt=[0,1],lim=None):
    """
    Plot model data against reference data to visualize bias in the model.
    Parameters:
        dat_mod (list of lists/arrays): Model data 
        dat_ref (list of lists/arrays): reference data 
        leg (list of string lists)
        lim (list of lists) dimension (len(dat_mod),2), (min,max) for each subplot
    """
    import matplotlib.transforms as mtransforms    

    nsub = len(dat_mod)
    nvar = len(dat_mod[0])  # all arrays in the list should have the same length
    ncol = int(nsub/nrow+0.5)
    cm = 1/2.54  # centimeters in inches
    # figure layout,layout="constrained", sharex=True, 
    fig, axs = plt.subplots(nrow, ncol, figsize=(subsize[0]*ncol, subsize[1]*nrow)) # default unit inch 2.54 cm
    
    # Ensure axs is iterable (handles case with a single subplot)
    if nsub == 1:
        axs = [axs]
    
    if len(marker) < len(dat_mod[0]):
        marker = ['o' for i in range(len(dat_mod[0]))]
    # A4 8.3*11.7
    size_tick,size_label,size_title = fontsize-1,fontsize,fontsize

    axs1 = axs.flatten()
    irm_ax = np.delete(np.arange(nrow*ncol),np.arange(nsub))
    if irm_ax is not None: # remove empty axis
        for i in range(len(irm_ax)):
            fig.delaxes(axs1[irm_ax[i]])
            
    for i,sublst in enumerate(dat_mod):
        ax = axs1[i] 
        # ax.set_facecolor('xkcd:gray')
        
        for j,data in enumerate(sublst):
            if leg is not None:
                ax.scatter(dat_ref[i][j], data, alpha=alpha, marker=marker[j],label=leg[j]) # marker=marker, color='blue',
            else:
                ax.scatter(dat_ref[i][j], data, alpha=alpha, marker=marker[j]) # marker=marker, color='blue',
            
        # Set the same limits for x and y axes
        if not lim:
            max_val = max(np.nanmax(np.array(dat_ref[i])), np.nanmax(np.array(sublst)))
            min_val = min(np.nanmin(np.array(dat_ref[i])), np.nanmin(np.array(sublst)))
        else:
            min_val = lim[i][0]
            max_val = lim[i][1]
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)    
        # Set the same ticks for x and y axes
        ticks = ax.get_xticks()
        ax.set_xticks(ticks,fontsize=fontsize-2)
        ax.set_yticks(ticks,fontsize=fontsize-2)
        
        # Plot the perfect fit line (y = x)
        ax.plot(ticks, ticks, linestyle='dashed', color='black') 
    
        ax.set_xlabel(axlab[0], fontsize=fontsize)
        if i%ncol==0: # only show ylabel on the left sub-panel
            ax.set_ylabel(axlab[1], fontsize=fontsize)
        ax.set_title(axlab[2], fontsize=fontsize)

        ax.grid(True)
        plt.text(0.01, 0.99, capt[i],fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
        # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
        # ax.text(0.02, 0.98, capt[i],fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption
        
        if txt is not None:
            ax.text(loc_txt[0],loc_txt[1], txt[i],fontsize=fontsize-2,ha='right', va='top', transform=ax.transAxes) #add txt
        
        ax.set_aspect('equal', adjustable='box')
        # plt.tight_layout()
    
    if leg is None:
        leg = [str(i) for i in range(nvar)]
    else:
        leg = leg[:nvar]
    if leg_col is None:
        leg_col = nvar
    # Create a single legend for all subplots
    if legloc is None: # loc: 0best,1Ur,2Ul,3-Ll,4-Lr, 5-R,6-Cl,7-Cr,8-Lc,9Uc,10C; U-upper
        fig.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,
                    frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)
    else: 
        fig.legend(leg,ncol=leg_col,columnspacing=0.8,fontsize=size_tick,loc=lloc,bbox_to_anchor=legloc,
                    frameon=False,handlelength=1, borderpad=0.1, labelspacing=0.1)
    
    # plt.show()
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)
    
    
def plot_distri(data,figname,nbin=10, lim=None, axlab=('Val','P',''),leg=('1', ), 
                   figsize=(3.7, 3.0), fontsize=12,capt='',style='default',xlim=None,ylim=None):
    """
    Compare the distribution of data using histograms.
    Parameters:
        data (list of arrays with same length): data, one array corresponds to 1 histogram.
        nbin (int or sequence, optional): Number of bins or bin edges. Default is 10.
"""
    from matplotlib.ticker import PercentFormatter 
    import matplotlib.transforms as mtransforms    
    # from matplotlib import style 
    fig = plt.figure(figsize=figsize)
    # plt.style.use(style) #'seaborn-deep'
    plt.style.context(style)
    # Calculate the bin edges
    if not lim:
        xmin = min([np.nanmin(np.array(data[i])) for i in range(len(data))])
        xmax = max([np.nanmax(np.array(data[i])) for i in range(len(data))])
    else:
        xmin = lim[0]
        xmax = lim[1]
    hist_range = (xmin, xmax)
    bins = np.linspace(hist_range[0], hist_range[1], nbin+1)

    # Plot histogram for observation data
    # for i in range(length(data)):
    # plt.hist(data[i], bins=bins, color=color[i], alpha=0.5, label=label[i]) # , align='left'

    # to plot the histogam side by side
    weights=[np.ones(len(data[i])) / len(data[i]) for i in range(len(data))]
    plt.hist(data,bins=bins,weights=weights, alpha=0.5, label=leg) # , align='right'

    plt.xlabel(axlab[0], fontsize=fontsize)
    plt.ylabel(axlab[1], fontsize=fontsize)
    plt.title(axlab[2], fontsize=fontsize)
    plt.legend(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
        
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # for array of the same length
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(data[0]))) # for array of the same length

    plt.grid(True)
    ax = plt.gca()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.text(0.01, 0.99, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes) #add fig caption
    # trans = mtransforms.ScaledTranslation(-30/72, 7/72, fig.dpi_scale_trans) # add shift in txt
    # plt.text(0.00, 1.00, capt,fontsize=fontsize,ha='left', va='top', transform=ax.transAxes+ trans) #add fig caption

    plt.tight_layout()
    # plt.show()
    plt.savefig(figname,bbox_inches='tight',dpi=300) #,dpi=100    
    plt.close(fig)