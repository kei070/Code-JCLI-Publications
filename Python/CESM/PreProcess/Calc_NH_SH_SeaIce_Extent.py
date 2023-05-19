"""
Calculate NH and SH sea-ice extent.

Be sure to adjust data_path and area_path.
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


#%% set the variable to be extracted
try:
    var_n = sys.argv[2]
except:
    var_n = "aice"
# end try except    


#%% set paths
data_path = f"/Data/{case}/"
area_path = "/POP_Input/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"SeaIce_Fraction_{case}*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-12:-8])
f_sto = int(f_list[-1][-7:-3])


#%% load the areacello variable
f_area = "areacello_Ofx_CESM2-FV2_abrupt-4xCO2_r1i1p1f1_gn.nc"
area_nc = Dataset(area_path + f_area)
area_co = area_nc.variables["areacello"][:] / 1E6  # convert from m^2 to km^2


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% loop over all files and load the ts variable
var = da.ma.masked_array(nc.variables[var_n], lock=True)

lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]


#%% sum up all the categories
# var = var.sum(axis=1).compute()
var = var.compute()


#%% convert the masked values to zero
var[var > 1e25] = 0


#%% multiply areacello field onto the siconc variable
si_area = var * area_co[None, :, :]


#%% calculate the NH and SH sea-ice areas
si_area_n_sum = np.nansum(si_area[:, lat > 0], axis=-1)
si_area_s_sum = np.nansum(si_area[:, lat < 0], axis=-1)


#%% test plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.axhline(y=0, c="gray", linewidth=0.75)
axes.plot(si_area_n_sum / 1e6, c="blue", label="NH")
axes.plot(si_area_s_sum / 1e6, c="red", label="SH")

axes.legend()

axes.set_title("Sea-ice extent CESM2-SOM "+ case)
axes.set_xlabel("Months since forcing")
axes.set_ylabel("Sea-ice extent in 10$^{10}$ km$^2$")

pl.show()
pl.close()


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"SeaIce_Area_NH_SH_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the values in an nc-file
out_name = (f"SeaIce_Area_NH_SH_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(data_path + out_name, "w", format="NETCDF4")

create_var = ["time"]

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", len(si_area_n_sum))
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

# flip the list and turn it into a tuple
create_var = tuple(create_var[::-1])

# create the wind speed variable
si_nh_nc = f.createVariable("si_area_n", "f4", create_var)
si_sh_nc = f.createVariable("si_area_s", "f4", create_var)

print("\nStoring data...\n\n")
si_nh_nc[:] = si_area_n_sum
si_sh_nc[:] = si_area_s_sum

# add attributes
f.description = (f"Monthly mean sea ice area in the northern and southern hemisphere for case {case} in CESM2 SOM." + 
                 "\nModel run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")

si_nh_nc.units = "km^2"
si_sh_nc.units = "km^2"

f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()















