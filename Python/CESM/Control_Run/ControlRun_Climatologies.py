"""
Build climatologies from the control run.

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


#%% set paths
data_path = ""


#%% set the file names
f_tas = glob.glob(data_path + "TREFHT*.nc")[0]
f_tom = glob.glob(data_path + "tom_*.nc")[0]
f_tomc = glob.glob(data_path + "tomC_*.nc")[0]


#%% load the files
nc_tas = Dataset(f_tas)
nc_tom = Dataset(f_tom)
nc_tomc = Dataset(f_tomc)


#%% get the data
tas = nc_tas.variables["TREFHT"][:]
tom = nc_tom.variables["tom"][:]
tomc = nc_tomc.variables["tomC"][:]


#%% calculate the climatology
tas_rs = np.reshape(tas, (int(len(tas)/12), 12))
tom_rs = np.reshape(tom, (int(len(tas)/12), 12))
tomc_rs = np.reshape(tomc, (int(len(tas)/12), 12))


#%% calculate the climatology
tas_clim = np.mean(tas_rs, axis=0)
tom_clim = np.mean(tom_rs, axis=0)
tomc_clim = np.mean(tomc_rs, axis=0)


#%% test plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

ax01 = axes.twinx()

axes.plot(tas_clim, c="red", label="T$_{ref}$")
ax01.plot(tom_clim, c="blue", label="TOM AS")
ax01.plot(tomc_clim, c="blue", linestyle="--", label="TOM CS")

axes.legend(loc="center left")
ax01.legend(loc="center right")

pl.show()
pl.close()


#%% store the data
out_name = ("tas_tom_climatology_Proj2_KUE.nc")

# generate the file
f = Dataset(data_path + out_name, "w", format="NETCDF4")

f.createDimension("time", 12)
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.arange(0, 12)

# create the variable
tas_nc = f.createVariable("tas_clim", "f4", "time")
tom_nc = f.createVariable("tom_clim", "f4", "time")
tomc_nc = f.createVariable("tomC_clim", "f4", "time")

# pass the remaining data into the variables
print("\nStoring data...\n\n")

tas_nc[:] = tas_clim
tom_nc[:] = tom_clim
tomc_nc[:] = tomc_clim

# add attributes
f.description = ("SAT and TOM (all- and clear-sky) control climatology in CESM2 SOM.\nModel run started by " + 
                 "Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


