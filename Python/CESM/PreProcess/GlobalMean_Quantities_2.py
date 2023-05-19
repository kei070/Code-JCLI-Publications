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
try:
    var_n = sys.argv[1]
except:
    var_n = "sw"
# end try except 


#%% setup case name
case_con = "Proj2_KUE"
try:
    case = sys.argv[2]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% if var_n and vaar_fn are not the same
var_fn = var_n
if (var_n == "sw") | (var_n == "lw"):
    var_fn = "tom"
# end if    
if (var_n == "swC") | (var_n == "lwC"):
    var_fn = "tomC"
# end if    


#%% set paths
data_path = f"/Data/{case}/"
out_path = f"/Data/{case}/Glob_Mean/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"/{var_fn}_{case}*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-12:-8])
f_sto = int((f_list[-1][-7:-3]))


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% loop over all files and load the ts variable
var = da.ma.masked_array(nc.variables[var_n], lock=True)

lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]


#%% calculate the TOM imbalance
var = np.array(var)

var_g = glob_mean(var, lat, lon)


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"{var_n}_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


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








