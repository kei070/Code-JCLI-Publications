"""
Extract a 3d variable to stored one file per year (NOT ANNUAL MEANS, these are monthly means).

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


#%% set the variable to be extracted
try:
    var_n = sys.argv[1]
except:
    var_n = "V"
# end try except    


#%% setup case name
try:
    case = sys.argv[2]
except:
    case = "dQ01yr_4xCO2"
# end try except 


#%% set paths
data_path = f"/Data/{case}/CAM_Files/"
out_path = f"/Data/{case}/Files_3d/{var_n}_Files/"
os.makedirs(out_path, exist_ok=True)


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"/{case}*.cam.h0*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-10:-6])
f_sto = int(f_list[-1][-10:-6])


#%% get the number of years
n_yrs = int(len(f_list)/12)


#%% load one file to lat and lon
lat = Dataset(f_list[0]).variables["lat"][:]
lon = Dataset(f_list[0]).variables["lon"][:]
lev = Dataset(f_list[0]).variables["lev"][:]


#%% loop over all files and load the variable
for count, yr in enumerate(np.arange(0, n_yrs*12, 12)):
    
    # set up the year for the file name
    yr_fn = f_sta + count
    
    nc = Dataset(f_list[yr])
    var = da.ma.masked_array(nc.variables[var_n], lock=True)

    if len(f_list) > 1:
        for i in np.arange(yr+1, yr+12):
            dataset = Dataset(f_list[i])
            var = da.concatenate([var, da.ma.masked_array(dataset.variables[var_n], lock=True)])
            # dataset.close()
        # end for i
    # end if
    
    # store the result as a netcdf file
    out_name = (f"{var_n}_{case}_{yr_fn:04}.nc")
    
    # generate the file
    f = Dataset(out_path + out_name, "w", format="NETCDF4")
    
    # set up the wind speed createVariable list, Note that it's the wrong way
    # round due to the appending and will later be flipped and then converted
    # to a tuple
    create_var = ["lon", "lat", "lev", "time"]
    
    
    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("lon", len(lon))
    f.createDimension("lat", len(lat))
    f.createDimension("lev", len(lev))
    
    # build the variables
    longitude = f.createVariable("lon", "f8", "lon")
    latitude = f.createVariable("lat", "f8", "lat")
    level = f.createVariable("lev", "f8", "lev")
    
    f.createDimension("time", np.shape(var)[0])
    time_nc = f.createVariable("time", "i4", "time")
    time_nc[:] = np.arange(12)
    
    # flip the list and turn it into a tuple
    create_var = tuple(create_var[::-1])
    
    # create the wind speed variable
    var_nc = f.createVariable(var_n, "f4", create_var)
    
    # pass the remaining data into the variables
    longitude[:] = lon 
    latitude[:] = lat
    level[:] = lev
    print("\nStoring data...\n\n")
    
    var_nc[:] = var
    
    # add attributes
    f.description = (f"{var_n} annual file for case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                     "Model set up by Rune Grand Graversen.")
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    longitude.units = "degrees east"
    latitude.units = "degrees north"
    level.units = "hPa"
    
    # close the dataset
    f.close()
# end for yr    

