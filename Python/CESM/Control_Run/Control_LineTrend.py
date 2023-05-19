"""
Calculate and store the control run linear trends for 2d model variables.

Be sure to set data_path correctly.
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


#%% select a variable to calculate the trends for
var_n = "FSDS"
var_desc = "downwelling solar flux at surface"
units = "W m^-2"


#%% set paths
data_path = ""


#%% set file path and check if that file already exists
f_name = "CAM_2d_control_trends.nc"

file_ext = os.path.isfile(data_path + f_name)

nc_mode = "w"
if file_ext:
    nc_mode = "r+"
# end if


#%% load the variable
var_nc = Dataset(glob.glob(data_path + var_n + "_Proj2_KUE*.nc")[0])

var = var_nc.variables[var_n][:]


#%% load the grid info
lat = var_nc.variables["lat"][:]
lon = var_nc.variables["lon"][:]


#%% extract the dimensions
timd = np.shape(var)[0]
latd = np.shape(var)[-2]
lond = np.shape(var)[-1]


#%% loop over the array and perform the linear regressions
sl, yi = np.zeros((latd, lond)), np.zeros((latd, lond))

for la in np.arange(latd):
    for lo in np.arange(lond):
        sl[la, lo], yi[la, lo] = lr(np.arange(timd), var[:, la, lo])[:2]
    # end for lo
# end for la


#%% test plots
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(15, 6), sharey=True)

p1 = axes[0].imshow(sl, origin="lower")
cb1 = fig.colorbar(p1, ax=axes[0], shrink=0.7)
cb1.set_label("Slopes in unit a$^{-1}$")

axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Slopes")

p2 = axes[1].imshow(yi, origin="lower")
cb2 = fig.colorbar(p2, ax=axes[1], shrink=0.7)
cb2.set_label("Y-intercept in units")

axes[1].set_xlabel("Longitude")
axes[1].set_title("Y-intercepts")

fig.subplots_adjust(wspace=0.1)

pl.show()
pl.close()


#%% either write or update the control trend dataset

# generate/open the file
f = Dataset(data_path + f_name, nc_mode, format="NETCDF4")

# enter grid info as well as a description if necessary
if not file_ext:
    f.createDimension("lon", len(lon))
    f.createDimension("lat", len(lat))
    f.createDimension("2", 2)
    longitude = f.createVariable("lon", "f8", "lon")
    latitude = f.createVariable("lat", "f8", "lat")
    longitude[:] = lon 
    latitude[:] = lat    
    longitude.units = "degrees east"
    latitude.units = "degrees north"
    f.history = "Created " + ti.strftime("%d/%m/%y")
    f.description = ("This file contains the linear trends of variables from the CESM2-SOM case 'Proj2_KUE' which " +
                     "acts as a control case and may be used to calculate the 'delta quantities', i.e., the change " +
                     "induced due to a forcing of the climate system. All variables are 3d arrays and the slopes of " +
                     "the linear trends are accessible via [0, :, :] and the y-intercepts via [1, :, :].")
else:
    hist = f.history
    f.history = hist + " - updated " + ti.strftime("%d/%m/%y")
# end if

nc_var = f.createVariable(var_n, "f8", ("2", "lat", "lon"))

nc_var[0, :, :] = sl
nc_var[1, :, :] = yi

nc_var.description = var_desc
nc_var.units = units

f.close()

    

    