"""
Calculate surface albedo CESM2-SOM.

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


#%% clear sky
try:
    cs = sys.argv[2]
except:
    cs = ""
# end try except    

cs_desc = "all-sky"
if cs == "C":
    cs_desc = "clear-sky"
# end if cs    


#%% setup case name
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% set paths
data_path = f"/Data/{case}/"
out_path = f"/Data/{case}/"


#%% load the variable
fsnt_fn = glob.glob(data_path + "FSNT_*.nc")[0]
solin_fn = glob.glob(data_path + "SOLIN_*.nc")[0]

fsnt_nc = Dataset(fsnt_fn)
solin_nc = Dataset(solin_fn)

fsnt = fsnt_nc.variables["FSNT"][:]
solin = solin_nc.variables["SOLIN"][:]
fsut = solin - fsnt
pln_alb = fsut / solin

# replace masked surface albedo values with zero
pln_alb[pln_alb.mask] = 0


#%% extract the years from the file names
f_sta = int(fsnt_fn[-12:-8])
f_sto = int(fsnt_fn[-7:-3])


#%% load the grid info
lat = fsnt_nc.variables["lat"][:]
lon = fsnt_nc.variables["lon"][:]


#%% test plot FSNS and FSDS
t_ind = 0

fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(12, 10), sharey=True)

p1 = axes[0, 0].imshow(fsnt[t_ind, :, :], origin="lower", vmin=0, vmax=400)
cb1 = fig.colorbar(p1, ax=axes[0, 0], shrink=0.7)
cb1.set_label("FSNT in Wm$^{-2}$")

axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
axes[0, 0].set_title("Net solar radiation at TOM\n(positive down)")

p2 = axes[0, 1].imshow(solin[t_ind, :, :], origin="lower", vmin=0, vmax=400)
cb2 = fig.colorbar(p2, ax=axes[0, 1], shrink=0.7)
cb2.set_label("SOLIN in Wm$^{-2}$")

axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_title("Insolation at TOM\n(positive down)")

p3 = axes[1, 0].imshow(fsut[t_ind, :, :], origin="lower", vmin=0, vmax=400)
cb3 = fig.colorbar(p3, ax=axes[1, 0], shrink=0.7)
cb3.set_label("FSUT in Wm$^{-2}$")

axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_title("Upwelling solar radiation at TOM\n(positive up)")

p4 = axes[1, 1].imshow(pln_alb[t_ind] * 100, vmin=0, vmax=100, origin="lower")
cb4 = fig.colorbar(p4, ax=axes[1, 1], shrink=0.7)
cb4.set_label("Planetary albedo in %")

axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_title("Planetary albedo")

fig.suptitle("Month " + str(t_ind))
fig.subplots_adjust(wspace=0.1, hspace=0.05)

pl.show()
pl.close()


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"PlanetaryAlbedo_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the result as a netcdf file
out_name = (f"PlanetaryAlbedo_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set up the wind speed createVariable list, Note that it's the wrong way
# round due to the appending and will later be flipped and then converted
# to a tuple
create_var = ["lon", "lat", "time"]


# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("time", np.shape(fsnt)[0])
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

# flip the list and turn it into a tuple
create_var = tuple(create_var[::-1])

# create the wind speed variable
pln_alb_nc = f.createVariable("PlnAlb", "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

pln_alb_nc[:] = pln_alb

# add attributes
f.description = (f"Planetary albedo for case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                 "Model set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()

