"""
Concatenate the regridded (via CDO) sea-ice files.
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


#%% case
case = "dQ01yr_4xCO2"


#%% data path
data_path = f"/Data/{case}/CICE_Files_rg/"
out_path = f"/Data/{case}/"


#%% load the file list
f_list = sorted(glob.glob(data_path + case + "*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-10:-6])
f_sto = int(f_list[-1][-10:-6])


#%% get the number of years
n_yrs = int(len(f_list)/12)


#%% load the first file to get lat and lon
nc0 = Dataset(f_list[0])
lat = nc0.variables["lat"][:]
lon = nc0.variables["lon"][:]    


#%% load, concatentate, and store the values
out_name = (f"SeaIce_Fraction_n80_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")
# si_type = f.createVariable("type", "f8", "type")

f.createDimension("time", n_yrs*12)
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

var_nc = f.createVariable("aice", "f4", ("time", "lat", "lon"))

for mon, file in enumerate(f_list):
    nc = Dataset(file)
    var_nc[mon, :, :] = nc.variables["aice"][:]
    
# end for f    

longitude[:] = lon 
latitude[:] = lat

f.description = (f"Monthly mean sea ice fraction for case {case} in CESM2-SOM regridded via CDO to n80 grid." + 
                 "\nModel run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()
