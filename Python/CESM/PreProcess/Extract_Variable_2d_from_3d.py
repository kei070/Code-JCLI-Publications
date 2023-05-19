"""
Extract a 2d field for a give pressure from a 3d variable.

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


#%% set the level to extract in hPa
try:
    exlev = int(sys.argv[3])
except:
    exlev = 700
# end try except


#%% set the variable to be extracted
try:
    var_n = sys.argv[1]
except:
    var_n = "OMEGA"
# end try except


#%% setup case name
try:
    case = sys.argv[2]
except:
    case = "d01yr_4xCO2"
# end try except


#%% units
units = ""
if var_n == "T":
    units = "K"
if var_n == "OMEGA":
    units = "Pa/s"    
# end if    


#%% set paths
data_path = f"/Data/{case}/CAM_Files/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"/{case}*.cam.h0*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-10:-6])
f_sto = int(f_list[-1][-10:-6])


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% get the index of the level closest to 700 hPa (apparently)
i_exlev  = np.argmin(np.abs(nc.variables["lev"][:] - exlev))
p_exlev = nc.variables["lev"][i_exlev]


#%% loop over all files and load the variable
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


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"{var_n}_{exlev}hPa_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the result as a netcdf file
out_name = (f"{var_n}_{exlev}hPa_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set up the wind speed createVariable list, Note that it's the wrong way
# round due to the appending and will later be flipped and then converted
# to a tuple
create_var = ["lon", "lat", "time"]


# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))
f.createDimension("lev", 1)

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")
plev_nc = f.createVariable("lev", "f8", "lev")

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
plev_nc[:] = p_exlev
print("\nStoring data...\n\n")

var_nc[:] = var[:, i_exlev, :, :]

# add the potential temperature if applicable
desc = ""
if var_n == "T":
    var_pot_nc = f.createVariable(var_n + "_pot", "f4", create_var)
    var_pot_nc[:] = var[:, i_exlev, :, :] * (1000 / exlev)**0.286
    desc = " Also contains the potential temperature."
    var_pot_nc.units = units
#  end if    

# add attributes
f.description = (f"{var_n} on the {p_exlev} hPa level for case {case} in CESM2 SOM.{desc}\nModel run started by " + 
                 "Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"
var_nc.units = units

# close the dataset
f.close()


#%% close the opened datasets
dataset.close()
