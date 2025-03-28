# SRResNet_ens: introduction

This repository is for downscaling physical fields using 
ensemble super-resolution residual network (SRResNet). 
Here the model is applied to downscale significant wave height (SWH) 
in the Black Sea using low-resolution data from ERA5 and high-resolution 
data from CMEMS. 
Both self-variable downscaling from low-resolutoin SWH and cross-variable 
downscaling from low-resolution wind fields are applied. 

An example has been constructed using the minima dataset. 
Directories *data_ERA5* and *data_CMEMS* contain examples of low and high resoltuion data. 
Data are downloaded using scripts in directories *scripts_download_CMEMS* and 
*scripts_download_ERA5*. 

Batch files are used to run the model on the DKRZ Levante cluster. 
The batch files contain the name of the parameter file (par*.py) to be read for runnning. 
Examples of the parameter files and the corresponding batch files are in *files_par_bash*. 
To run the model, these files should be in the parent directory of the directory 
*scripts*.
You can also modify the scripts to read parameter files from a 
specified directory. 

Inspired by  
[https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan] 
and  
[https://github.com/leftthomas/SRGAN]  

pytorch-msssim is from 
[https://github.com/VainF/pytorch-msssim]  

===========================================================
## How to use 

To run the model without batch file, change 'mod_name' from 'sys.argv[1]' 
to the name of the parameter file (e.g., 'par01') in train.py. 
Next in a linux terminal: python train.py, or run with python IDE like spyder. 

To train the model:   
```	
python train.py, or sbatch run_ml_gpu1_r55e.sh
```
To test the model for each epoch:   
```	
python test_epo.py or sbatch run_test_epo_r55e.sh   
```
To test the ensemble model:   
```	
python test_epo_ave.py or sbatch run_test_epo_ave_r55e.sh   
```
===========================================================
## Mainly used python scripts and steps to produce figures:
  
Main scripts for training and testing:  
1. train.py:   
	
2. test_epo.py:   
	generating hr data from lr testing dataset.  
	save mean, 99th 1st percentiles, and metrics rmse/mae for each epoch.  
	sort rmse and rmse_99 for all repeated runs.   
3. test_epo_ave.py:  
	generating ensemble hr data from lr testing dataset, save every nep_skip epochs.  
	save mean, 99th 1st percentiles, and metrics rmse/mae for each ensemble.  
test_epo_ave_t.py:  
	as test_epo_ave.py, but use testing period defined in par*_tuse*.py (set rtra=0 such that period in tlim is all for testing).  

Scripts needed:  
models.py (SRResNet model)  
datasets.py: class mydataset, loss function.  
funs_prepost.py: functions for pre- and post-processing.  
par*.py: parameter file (located in the parent path of \scripts), including model parameters and information of dataset (file dir, nc variable names etc.).  

===========================================================
## Scripts used for plotting

Comparison of metrics: (plot)  
4. compare_ens_err_gb.py: (plot)  
	compare ensemble SRResNet with original SRResNet global error.  
	need to load saved metrics from steps 2&3.
5. compare_ens_metrics.py: (plot)  
	compare metrics between different experiments (e.g. scale factor) using ensemble model.  

Scripts needed:  
funs_prepost.py: functions for post-processing.  

===========================================================
Comparison of 2D spatial pattern and time series at selected location in user defined period: (plot)  

6. test_epo_ave_tuse.py:  
	testing for the user defined (short) period instead of whole test set.  
	save hr, interpolation results, ensemble sr in the user defined period.  
	stations selected based on plot_study_domain.py (modify select_sta in funs_sites.py for custom use).   
7. compare_2D_pnt.py:  
	compare sr & hr at selected time (2d map plot) and selected stations (line plot).  

cal_metrics_intp.py:  
	get interpolated hr and its 01,99 percentile and time average.  

>[!NOTE]
For comparison with conventional ml models, results from ml models should be saved, 'ml_mod_name' should match with 'mod_name'. Conventional ml models like mulivariate linear regression for spatial wave downscaling see [https://github.com/B-Yuan2023/MLR_SWH]. 

8. compare_ml_2D_pnt.py: (plot)  
	compare MLR with SRResNet, time series of 2d map and points.  
	require saved sr from SRResNet, MLR, saved hr and interpolated hr in the selected period (test_epo_ave_tuse.py).  
9. compare_ml_99per.py: (plot)  
	compare MLR with SRResNet, 99/01 percentile, and mean;  
	require 99/01 percentile of hr and interpolated hr (cal_metrics_intp.py);  
	require saved 99/01 percentile sr from linear regression (MLR/test.py);  
	require saved 99/01 percentile sr from SRResNet (test_epo_ave.py).  
10. compare_ml_dist.py: (plot)  
	compare MLR with SRResNet, distribution of global data and selected stations.  

11. compare_dist_year.py: (plot)  
	compare data distribution of specified periods.   

===========================================================
### Other scripts:  

plot bathymetry and check information on buoy stations
read_cmems_mo.py: 
	read cmems mooring data (coordinate,time range).  
plot_study_domain.py:  
	plot bathymetry from nc, estimate the depth at the mooring stations (coordinates from the read_cmems_mo.py).  

scripts needed: funs_prepost.py, funs_sites.py

=============================

initial check of datasets:  
statistic_hr.py:  
	to obtain the index of hour/batch for the maximum var.  
plot_hist_train_test.py:  
	check data distribution, plot hisogram of the training and testing data.  

=============================

check network structure and referencing time:  
check_nn_structure.py:  
	check the neural network structure and no. of coefficients.  
test_time.py:  
	check the referencing time after training for a selected period.  

=============================

test_epo_user.py:  
	plot 2d map of 01,99,mean for user epoch; plot 2d snapshot of time series;
	save 01,99,mean for user epoch. 

