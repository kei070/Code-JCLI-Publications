"""
Calculate global mean quantities for CESM2 runs.

Be sure to adjust data_path and out_path.
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


#%% set the variable name
var_n = "TREFHT"


#%% setup case name
case = "dQ01yr_4xCO2"


#%% set paths
data_path = f"/Data/{case}/CAM_Files/"
out_path = f"/Data/{case}/Glob_Mean/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"/{case}*.cam.h0*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-8:-6])
f_sto = int(f_list[-1][-8:-6])


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% loop over all files and load the ts variable
var = da.ma.masked_array(nc.variables[var_n], lock=True)

lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]

if len(f_list) > 1:
    for i in np.arange(1, len(f_list)):
        dataset = Dataset(f_list[i])
        var = da.concatenate([var, da.ma.masked_array(dataset.variables[var_n], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% calculate the TOM imbalance
var = np.array(var)

dataset.close()

var_g = glob_mean(var, lat, lon)


#%% store the result as a netcdf file
out_name = (f"{var_n}_{case}_{f_sta:04}-{f_sto:04}.nc")

os.makedirs(out_path, exist_ok=True)

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

f.createDimension("time", len(var_g))
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

# create the wind speed variable
var_nc = f.createVariable(var_n, "f4", "time")

print("\nStoring data...\n\n")

var_nc[:] = var_g

# add attributes
f.description = (f"Global mean {var_n} imbalance for case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                 "Model set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()








