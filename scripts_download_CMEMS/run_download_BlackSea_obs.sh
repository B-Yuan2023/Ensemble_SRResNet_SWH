#!/bin/bash
#
#SBATCH --job-name=download_cmems        # Specify job name
#SBATCH --partition=compute
#SBATCH --nodes=1           # Specify number of nodes
#SBATCH --ntasks-per-node=4  # Specify number of tasks on each node
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --account=gg0028
#SBATCH --output=out.o%j       # File name for standard output
#SBATCH --error=out.e%j        # File name for standard error output
#
mypy=/home/g/g260218/.conda/envs/cmt_1.0/bin/python

cdir=$(pwd)
cd ${cdir}

lon0=27.25
lon1=42
lat0=40.5
lat1=47
t00="2000-01-01T00:00:00"
t11="2022-01-01T00:00:00"

${mypy} download_obs_blacksea_swh.py ${lon0} ${lon1} ${lat0} ${lat1} ${t00} ${t11}

