"""
Store sea-ice data.

Be sure to adjust data_path, area_path, and out_path
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
data_path = f"/Data/{case}/CICE_Files/"
area_path = "/POP_Input/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + "*.cice.h*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-10:-6])
f_sto = int(f_list[-1][-10:-6])


#%% load the areacello variable
f_area = "areacello_Ofx_CESM2-FV2_abrupt-4xCO2_r1i1p1f1_gn.nc"
area_nc = Dataset(area_path + f_area)
area_co = area_nc.variables["areacello"][:] / 1E6


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% look into the first file
"""
test = nc.variables[var_n][:]


#%% plot the values
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

p1 = axes.imshow(np.sum(test[0, :, :, :], axis=0), origin="lower")
fig.colorbar(p1)

pl.show()
pl.close()
"""


#%% loop over all files and load the ts variable
var = da.ma.masked_array(nc.variables[var_n], lock=True)

lat = nc.variables["TLAT"][:]
lon = nc.variables["TLON"][:]

if len(f_list) > 1:
    for i in np.arange(1, len(f_list)):
        dataset = Dataset(f_list[i])
        var = da.concatenate([var, da.ma.masked_array(dataset.variables[var_n], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"SeaIce_Fraction_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the data as nc files
out_name = (f"SeaIce_Fraction_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set up the wind speed createVariable list, Note that it's the wrong way
# round due to the appending and will later be flipped and then converted
# to a tuple
# create_var = ["lon", "lat", "type", "time"]
create_var = ["lon", "lat", "time"]


# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", np.shape(lon)[1])
f.createDimension("lat", np.shape(lat)[0])
# f.createDimension("type", 5)

# build the variables
longitude = f.createVariable("lon", "f8", ("lat", "lon"))
latitude = f.createVariable("lat", "f8", ("lat", "lon"))
# si_type = f.createVariable("type", "f8", "type")

f.createDimension("time", np.shape(var)[0])
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

# flip the list and turn it into a tuple
create_var = tuple(create_var[::-1])

# create the wind speed variable
var_nc = f.createVariable(var_n, "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

var_nc[:] = var

# add attributes
f.description = (f"Monthly mean sea ice fraction for case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                 "Model set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()

























