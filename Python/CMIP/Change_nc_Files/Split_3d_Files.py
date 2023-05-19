"""
Split up a given 3D (+time) netCDF file into multiple shorter files.

So far only implemented for CMIP6 (not CMIP5)!

Be sure to change data_path and out_path.
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
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti
import timeit
import Ngl as ngl
import geocat.ncomp as geoc


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


#%% Choose the length of the data set. If tim_set=0 the actual length of the dataset will be used.
len_tim = 250  # in years

# set the number of years per output file
n_yr = 50

# convert to length in months
len_tim = int(12 * len_tim)


#%% set the required parameters
try:
    exp = int(sys.argv[1])
except:
    exp = 0
try:
    var_name = sys.argv[2]
except:
    var_name = "cl"
try:    
    ensemble = sys.argv[3]
except:    
    ensemble = "r1i1p1f2"
try:
    model = sys.argv[4]
except:    
    model = "UKESM1-0-LL" 
try:
    start_yr = int(sys.argv[5])    
except:
    start_yr = 2100
try:
    stop_yr = int(sys.argv[6])    
except:
    stop_yr = 2149
# end try except  

exp_name = ["piControl", "abrupt-4xCO2", "abrupt-2xCO2", "1pctCO2", "ssp585"]

print(model)
print("\n" + var_name + "\n")
print(exp_name[exp] + "\n")


#%% handle the grid 
gr_lab = "gn"

if model in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", "EC-Earth3-AerChem", 
             "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
    gr_lab = "gr"
elif model in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5", "KIOST-ESM"]:
    gr_lab = "gr1"
# end if


#%% handle the ensemble
ensemble_d = ""
if ensemble != "r1i1p1f1":
    ensemble_d = "_" + ensemble
# end if


#%% set paths and names

# data path
data_path = model + "/" + exp_name[exp] + "_" + var_name + "_Files" + ensemble_d + "/"

# output name
out_path = model + "/" + exp_name[exp] + "_" + var_name + "_Files" + ensemble_d + "/"

# file name
f_name = (var_name + "_Amon_" + model + "_" + exp_name[exp] + "_" + ensemble + "_" + gr_lab + "_" + f'{start_yr:04}' + 
          "01-" + f'{stop_yr:04}' + "12.nc")


#%% load the nc file
nc = Dataset(data_path + f_name)


#%% load the necessary grid information
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]
plev = nc.variables["plev"][:]
tim = nc.variables["time"][:]

if len_tim == 0:
    len_tim = len(tim)
# end if    


#%% load part of the file and store it as a stand-alone nc file
yrs = np.arange(start_yr, start_yr+int(len(tim)/12), n_yr)

yr = 0
for i in np.arange(0, len_tim, int(n_yr*12)):
    
    # set up the period string for the file name
    period = f'{yrs[yr]:04}' + "01-" + f'{yrs[yr]+n_yr-1:04}' + "12"
    
    # set up an output name
    out_name = f_name[:-16] + period + ".nc"
    
    # get the variable to be stored
    var = nc.variables[var_name][i:int(i+n_yr*12), :, :, :]
    
    # generate the file
    f = Dataset(out_path + out_name, "w", format="NETCDF4")
    
    # set the branch_time attribute
    f.setncattr("branch_time_in_child", nc.branch_time_in_child)
    f.setncattr("branch_time_in_parent", nc.branch_time_in_parent)
    f.setncattr("parent_experiment_id", nc.parent_experiment_id)
    f.setncattr("parent_time_units", nc.parent_time_units)
    f.setncattr("variant_label", nc.variant_label)
    f.setncattr("further_info_url", nc.further_info_url)
    
    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("lon", len(lon))
    f.createDimension("lat", len(lat))
    f.createDimension("plev", len(plev))
    f.createDimension("time", len(tim[i:int(i+n_yr*12)]))
    
    # build the variables
    longitude = f.createVariable("lon", "f8", "lon")
    latitude = f.createVariable("lat", "f8", "lat")
    plev_nc = f.createVariable("plev", "f8", "plev")
    time_nc = f.createVariable("time", "f8", "time")
    
    
    # create the wind speed variable
    var_nc = f.createVariable(var_name, "f4", ("time", "plev", "lat", "lon"))
    
    # pass the remaining data into the variables
    longitude[:] = lon 
    latitude[:] = lat
    plev_nc[:] = plev
    time_nc[:] = tim[i:int(i+n_yr*12)]
    var_nc[:] = var
    
    # add attributes
    f.description = (var_name + " file with years " + str(yrs[yr]) + " to " + str(yrs[yr]+n_yr-1) + " of the file " + 
                     f_name + "\n.\n.\n.")
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    longitude.units = nc.variables["lat"].units
    latitude.units = nc.variables["lon"].units
    plev_nc.units = nc.variables["plev"].units
    time_nc.units = nc.variables["time"].units
    
    # close the dataset
    f.close()
    
    yr += 1

    print_one_line(str(i) + "/" + str(len_tim) + "   ")
    
# end for i