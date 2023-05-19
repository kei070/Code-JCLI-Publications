"""
Calculate SEB.
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

script_direc = "/home/kei070/Documents/Python_Scripts_PhD_Publication1/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Region_Mean import region_mean


#%% establish connection to the client
c = Client("localhost:8786")


#%% setup case name
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% set paths
data_path = f"/Data/{case}/"
out_path = f"/Data/{case}/"


#%% load the necessary files
lhflx_nc = Dataset(glob.glob(data_path + "LHFLX_*.nc")[0])
shflx_nc = Dataset(glob.glob(data_path + "SHFLX_*.nc")[0])
flns_nc = Dataset(glob.glob(data_path + "FLNS_*.nc")[0])
fsns_nc = Dataset(glob.glob(data_path + "FSNS_*.nc")[0])


#%% extract the years from the file names
period = glob.glob(data_path + "LHFLX_*.nc")[0][-12:-3]


#%% get the gird
lat = lhflx_nc.variables["lat"][:]
lon = lhflx_nc.variables["lon"][:]


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"seb_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the result as a netcdf file
out_name = (f"seb_{case}_{period}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("time", lhflx_nc.dimensions["time"].size)
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.arange(lhflx_nc.dimensions["time"].size)

# create the wind speed variable
seb_nc = f.createVariable("seb", "f4", ("time", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

# this should be correct: FLNS appears to be positive up; FSNS positive down
seb_nc[:] = (-lhflx_nc.variables["LHFLX"][:] - shflx_nc.variables["SHFLX"][:] - flns_nc.variables["FLNS"][:] + 
             fsns_nc.variables["FSNS"][:])

# add attributes
f.description = (f"Surface energy balance (SEB) for case {case} in " +
                 "CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

seb_nc.description = "Surface energy balance in Wm^-2 (positive down)"

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()
