"""
Calculate AHT.

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
seb_nc = Dataset(glob.glob(data_path + "dseb_Mon_*.nc")[0])
tom_nc = Dataset(glob.glob(data_path + "dtom_Mon_*.nc")[0])


#%% extract the years from the file names
period = glob.glob(data_path + "dseb_*.nc")[0][-12:-3]


#%% get the gird
lat = seb_nc.variables["lat"][:]
lon = seb_nc.variables["lon"][:]


#%% test plot
dseb = np.mean(seb_nc.variables["dseb"][1:, :, :, :], axis=(0, 1))
dtom = np.mean(tom_nc.variables["dtom"][1:, :, :, :], axis=(0, 1))


#%% 
x, y = np.meshgrid(lon, lat)
proj = ccrs.Robinson(central_longitude=0)

fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(16, 6), subplot_kw={'projection':proj})

p1 = axes[0].pcolormesh(x, y, dseb, cmap=cm.RdBu_r, vmin=-50, vmax=50, transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=axes[0], shrink=0.5)
cb1.set_label("dSEB in Wm$^{-2}$K$^{-1}$")

axes[0].set_global()
axes[0].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

axes[0].set_title(f"dSEB CESM-SOM {case}")


p2 = axes[1].pcolormesh(x, y, dtom, cmap=cm.RdBu_r, vmin=-50, vmax=50, transform=ccrs.PlateCarree())

cb2 = fig.colorbar(p2, ax=axes[1], shrink=0.5)
cb2.set_label("dTOM in Wm$^{-2}$K$^{-1}$")

axes[1].set_global()
axes[1].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

axes[1].set_title(f"dTOM CESM-SOM {case}")

pl.show()
pl.close()


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"daht_Mon_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the result as a netcdf file
out_name = (f"daht_Mon_{case}_{period}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("year", tom_nc.dimensions["year"].size)
time_nc = f.createVariable("year", "i4", "year")
time_nc[:] = np.arange(tom_nc.dimensions["year"].size)
f.createDimension("month", tom_nc.dimensions["month"].size)
time_nc = f.createVariable("month", "i4", "month")
time_nc[:] = np.arange(tom_nc.dimensions["month"].size)

# create the wind speed variable
aht_nc = f.createVariable("daht", "f4", ("year", "month", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

aht_nc[:] = seb_nc.variables["dseb"][:] - tom_nc.variables["dtom"][:]

# add attributes
f.description = (f"Atmospheric heat transport (AHT) for case {case} in " +
                 "CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

aht_nc.description = "Surface energy balance in Wm^-2 (positive into atmopsheric column)"

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()
