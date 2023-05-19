"""
Calculate Hadley circulation strength CESM2-SOM.

Be sure to adjust data_path.
"""

#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import pylab as pl
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy.stats import pearsonr as pears_corr
from scipy import interpolate
from scipy import signal
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti
import geocat.ncomp as geoc
from dask.distributed import Client

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Region_Mean import region_mean


#%% establish connection to the client
c = Client("localhost:8786")


#%% setup case name
try:
    case = sys.argv[2]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% set up paths
data_path = f"/Data/{case}/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + "/mpsi_Files/mpsi_*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-7:-3])
f_sto = int((f_list[-1][-7:-3]))


#%% load the datset
mpsi_nc = Dataset(f_list[0])

# TREFHT data
sat_exp_nc = Dataset(glob.glob(data_path + "TREFHT_*nc")[0])


#%% get the streamfunction values - forced
mpsi_mon = da.ma.masked_array(mpsi_nc.variables["mpsi"], lock=True)
if len(f_list) > 1:
    for i in np.arange(1, len(f_list)):
        dataset = Dataset(f_list[i])
        mpsi_mon = da.concatenate([mpsi_mon, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% aggregate to annual means
mpsi_mon = mpsi_mon.compute()

mpsi = an_mean(mpsi_mon)


#%% store the number of years the Q-flux experiment is run
nyrs = np.shape(mpsi)[0]


#%% get lat and plev
lat =  mpsi_nc.variables["lat"][:]
plev = mpsi_nc.variables["plev"][:] / 100

# extract the streamfunction for the latitudes between 15 and 25 N and S
hadley_n = mpsi[:, :, (lat > 15) & (lat < 25)]
hadley_s = mpsi[:, :,  (lat > -25) & (lat < -15)]

# get the time series via extracting the maximum streamfunction from the Hadley values
had_n_tser = np.max(hadley_n, axis=(1, 2))
had_s_tser = np.min(hadley_s, axis=(1, 2))


#%% store the values
out_name = f"Hadley_Circ_Index_{case}_{f_sta:04}-{f_sto:04}.nc"

os.makedirs(data_path, exist_ok=True)

# generate the file
f = Dataset(data_path + out_name, "w", format="NETCDF4")

f.createDimension("time", len(hadley_n))
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.arange(f_sta, f_sto+1)

# create the wind speed variable
nh_had_nc = f.createVariable("NH_Hadley", "f4", "time")
sh_had_nc = f.createVariable("SH_Hadley", "f4", "time")

print("\nStoring data...\n\n")

nh_had_nc[:] = had_n_tser
sh_had_nc[:] = had_s_tser

# add attributes
f.description = (f"Hadley circulation strength index case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                 "Model set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


