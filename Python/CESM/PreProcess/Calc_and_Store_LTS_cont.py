"""
Calculate LTS. --> Intended only for control runs.

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
    case = "Proj2_KUE"
# end try except


#%% set paths
data_path = f"/Data/{case}/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_sat = sorted(glob.glob(data_path + f"/TREFHT_{case}_*.nc"), key=str.casefold)
f_ps = sorted(glob.glob(data_path + f"/PS_{case}_*.nc"), key=str.casefold)
f_t700 = sorted(glob.glob(data_path + f"/T_700hPa_{case}_*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_sat[0][-12:-8])
f_sto = int((f_sat[-1][-7:-3]))


#%% load one file to lat and lon
nc_sat = Dataset(f_sat[0])
nc_ps = Dataset(f_ps[0])
nc_t700 = Dataset(f_t700[0])


#%% load the values
sat = an_mean(da.ma.masked_array(nc_sat.variables["TREFHT"], lock=True).compute())
ps = an_mean(da.ma.masked_array(nc_ps.variables["PS"], lock=True).compute())
t700 = an_mean(da.ma.masked_array(nc_t700.variables["T"], lock=True).compute())

lat = nc_sat.variables["lat"][:]
lon = nc_sat.variables["lon"][:]
lev = nc_t700.variables["lev"][:][0]


#%% calculate the LTS
lts = np.zeros(np.shape(sat))
for yr in np.arange(np.shape(sat)[0]):
    lts[yr, :, :] = (t700[yr, :, :] * (1000 / lev)**0.286 - sat[yr, :, :] * (100000 / ps[yr, :, :])**0.286)
# end for yr    


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"LTS_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the result as a netcdf file
out_name = (f"LTS_{case}_{f_sta:04}-{f_sto:04}.nc")

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

f.createDimension("time", len(np.arange(f_sta, f_sto+1)))
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.arange(f_sta, f_sto+1)

# flip the list and turn it into a tuple
create_var = tuple(create_var[::-1])

# create the wind speed variable
lts_nc = f.createVariable("lts", "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

lts_nc[:] = lts

# add attributes
f.description = (f"LTS for case {case} in CESM2 SOM.\nModel run started by Kai-Uwe Eiselt.\n" + 
                 "Model set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()


