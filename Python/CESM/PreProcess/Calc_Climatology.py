"""
Calculate climatology of a given variable.

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
import matplotlib.colors as colors
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
import dask.array as da
import time as ti
import timeit
import Ngl as ngl
import geocat.ncomp as geoc
import smtplib, ssl
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)
from Functions.Func_RunMean import run_mean
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_calcsatspechum import calcsatspechum, da_calcsatspechum


#%% establish connection to the client
c = Client("localhost:8786")


#%% set the variable of which the climatology is to be calculated
try:
    var_n = sys.argv[1]
except:
    var_n = "T"
# end try except  


#%% setup case name
try:
    case = sys.argv[1]
except:
    case = "Proj2_KUE"
# end try except


#%% set paths
data_path = f"/Data/{case}/CAM_Files/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_list = sorted(glob.glob(data_path + f"/{case}*.cam.h0*.nc"), key=str.casefold)[50*12:]


#%% extract the years from the file names
f_sta = int(f_list[0][-10:-6])
f_sto = int(f_list[-1][-10:-6])


#%% load one file to lat and lon
nc = Dataset(f_list[0])


#%% loop over all files and load the variable
var = da.ma.masked_array(nc.variables[var_n], lock=True)

lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]

if len(da.shape(var)) > 3:
    lev = nc.variables["lev"][:]
# end if    

if len(f_list) > 1:
    for i in np.arange(1, len(f_list)):
        dataset = Dataset(f_list[i])
        var = da.concatenate([var, da.ma.masked_array(dataset.variables[var_n], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% reshape the array
nyr = int(da.shape(var)[0] / 12)

if len(da.shape(var)) > 3:
    var_rs = da.reshape(var, (nyr, 12, len(lev), len(lat), len(lon)))
else:
    var_rs = da.reshape(var, (nyr, 12, len(lat), len(lon)))
# end if else    

clim = da.mean(var_rs, axis=0)


#%% store the result as a netcdf file
out_name = (f"{var_n}_Climatology_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

dims = ("month", "lat", "lon")
if len(da.shape(var)) > 3:  # if the variable also has a vertical dimension
    f.createDimension("lev", len(lev))
    level = f.createVariable("lev", "f8", "lev")
    level[:] = lev
    level.units = "hPa"
    dims = ("month", "lev", "lat", "lon")
# end if    

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("month", 12)
time_nc = f.createVariable("month", "i4", "month")
time_nc[:] = np.arange(12)

# create the wind speed variable
var_nc = f.createVariable(var_n, "f4", dims)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

var_nc[:] = clim

# add attributes
f.description = (f"{var_n} climatology for case {case} in CESM2 SOM calculated over the years given in the file name." +
                 "\nModel run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()
