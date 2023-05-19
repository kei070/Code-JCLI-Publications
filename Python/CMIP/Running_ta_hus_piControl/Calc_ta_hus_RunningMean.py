"""
Generate piControl 21-year running mean values of the 4d-variables ta and hus (should also work for other 4d variables).

Be sure to change data_path and out_path.
"""

#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import numexpr as ne
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
import smtplib, ssl
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Print_In_One_Line import print_one_line
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% establish connection to the client
c = Client("localhost:8786")


#%% variable, either "ta", or "hus"
try:
    var = sys.argv[4]
except:
    var = "ta"
# end try except

# did the file have to be regridded (i.e., vertically interpolated) first?
rg = ""  # "_rg"

if var == "ta":
    var_unit = "K"
elif var == "hus":
    var_unit = "kg/kg"
elif var == "cl":
    var_unit = "1"    
elif var == "wap":
    var_unit = "Pa s-1"    
# end if elif


#%% set the runnning mean
run = 21


#%% length of the experiment in years
n_yr = 150


#%% set the experiment - either "4x" or "2x"
try:
    exp = sys.argv[3]
except:
    exp = "4x"
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 5
# end try except

if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    a4x = "abrupt" + exp + "CO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    a4x = "abrupt-" + exp + "CO2"
# end if elif


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 12
# end try except

print(nl.models_pl[mod])

# set the branch year for CESM2 to 100
# nl.b_times["CAMS-CSM1-0"] = 25
nl.b_times["CESM2"] = 100


#%% set the ensemble
try:    
    ensemble = sys.argv[5]
except:    
    ensemble = "r1i1p1"
# end try ensemble    


#%% handle the ensemble
ensemble_d = ""

if (ensemble != "r1i1p1f1") & (ensemble != "r1i1p1"):
    ensemble_d = "_" + ensemble
# end if


#%% select the forced experiment file name
#if exp == "4x":
#    f_n = nl.a4x_f[mod]
#elif exp == "2x":
#    f_n = nl.a2x_f[mod]
# end if elif


#%% get the branching index
exp_sta = int(nl.b_times[nl.models[mod]] * 12)
if nl.b_times[nl.models[mod]] < 10:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)    
    b_end_ind = int(n_yr * 12) + b_sta_ind
    print("\npiControl does not start at least 10 year before forcing experiment. " + 
          "Taking start index from namelist: " + str(b_sta_ind) + ".\n")
else:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)
    b_sta_ind = int(b_sta_ind - ((run - 1) / 2) * 12)
    print("\npiControl does start at least 10 year before forcing experiment. " + 
          "Adjusting start index from namelist: " + str(b_sta_ind) + ".\n")
# end if else


#%% set paths

# model data path
# --> data path must contain the folders
# piControl_ta_Files and piControal_hus_Files
data_path = ""

# output path
out_path = "/" + var + "_piControl_" + str(run) + "YrRunMean_Files" + ensemble_d + "/"


#%% get the lists of hus and ta files
f_list = sorted(glob.glob(data_path + "piControl_" + var + "_Files" + rg + ensemble_d + "/*.nc"), key=str.casefold)


#%% load first file

# load in the respective first hus file
nc = Dataset(f_list[0])


#%% grid variables

# get the levels
levs = nc.variables["plev"][:]

# get lat, lon, and levels for the model data
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]


#%% load all files and concatenate them
if len(f_list) > 1:
    nc_l = []
    for i in np.arange(0, len(f_list)):
        nc_l.append(da.ma.masked_array(Dataset(f_list[i]).variables[var], lock=True))
    # end for i
    
    # concatenate the loaded files
    vals_da = da.concatenate(nc_l, axis=0)
else:
    print("\nOnly one " + var + " file...\n")
    vals_da = da.ma.masked_array(Dataset(f_list[0]).variables[var], lock=True)
# end if else


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
if ((da.shape(vals_da)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print("\npiControl run after branch year is at least 10 years longer than abrupt4xCO2. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print("\npiControl run after branch year is less than 10 years longer than abrupt4xCO2. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")
# end if

vals_da = vals_da[b_sta_ind:b_end_ind, :, :]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    f_add = sorted(glob.glob(data_path + "piControl_" + var + "_Files" + rg + ensemble_d + "/AddFiles/*AddBe.nc"), 
                   key=str.casefold)
    vals_da = da.concatenate([da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True), vals_da], axis=0)
    print("\nAdded AddBe file. New piControl shape:" + str(da.shape(vals_da)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_add = sorted(glob.glob(data_path + "piControl_" + var + "_Files" + rg + ensemble_d + "/AddFiles/*AddAf.nc"), 
                   key=str.casefold)
    vals_da = da.concatenate([vals_da, da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True)], axis=0)
    print("\nAdded AddAf file. New piControl shape:" + str(da.shape(vals_da)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except


#%% rechunk if necessary (e.g. for models like HadGEM3-GC31-MM)
if len(lat) * len(lon) > 130000:
    print("\nRechunking to (204, 1, " + str(len(lat)) + ", " + str(len(lon)) + ")")
    vals_da = da.rechunk(vals_da, chunks=(204, 1, len(lat), len(lon)))
# end if


#%% load the
start = timeit.default_timer()
ti_old = copy.deepcopy(start)
for lev in np.arange(len(levs)):

    #% open a netCDF file
    
    print("\nOpening the nc file...\n")
    f = Dataset(out_path + var + "_Amon_" + nl.models_n[mod] + "_piControl_plev" + f"{lev:02}" + "_0001-015012.nc", 
                "w", format="NETCDF4")
    
    # add the ensemble as an attribute
    f.setncattr("variant_label", ensemble)
    
    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("mon", 12)
    f.createDimension("year", n_yr)
    f.createDimension("lat", len(lat))
    f.createDimension("lon", len(lon))
    f.createDimension("plev", 1)
    
    # create the variables
    lat_nc = f.createVariable("lat", "f8", "lat")
    lon_nc = f.createVariable("lon", "f8", "lon")
    lev_nc = f.createVariable("plev", "f8", "plev")
    var_nc = f.createVariable(var, "f4", ("mon", "year", "lat", "lon"))    
    
    print("\nComputing level " + str(lev) + "...")
    vals = np.zeros((2040, len(lat), len(lon)))
    for i in np.arange(0, 2040, 204):
        vals[i:i+204, :, :] = np.array(np.ma.filled(vals_da[i:i+204, lev, :, :].compute(), np.nan))
        print_one_line(str(i+204) + "/2040  ")
    # end for i        
    vals[vals == -1.00000006e+27] = np.nan
    
    """ the following should not be necessary because of the np.ma.filled() function
    vals[vals == -1.00000006e+27] = np.nan
    vals[vals == 1e+20] = np.nan
    """
    
    # calculate the running means of the piControl    
    vals_run = np.zeros((12, n_yr, len(lat), len(lon)))
    print("\nCalculating running means...\n")
    for mon in np.arange(12):
        print(mon)
        vals_run[mon, :, :, :] = run_mean_da(vals[mon::12, :, :], running=run)
    # end for mon
    
    desc_add = ""
    if add_yrs:
        desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                    "mean values completely corresponding to the forcing experiment which is why the were\n" + 
                    "extrap0lated linearly from the adjacent 21 years of simulation.")
    # end if
    
    # fill in the values to the nc variables
    lat_nc[:] = lat
    lon_nc[:] = lon
    lev_nc[:] = levs[lev]
    var_nc[:] = vals_run
    
    # set the variable units
    var_nc.units = var_unit
    lat_nc.units = "degrees_north"
    lon_nc.units = "degrees_east"
    lev_nc.units = "Pa"
    
    # enter variable descriptions
    var_nc.description = var + " " + str(run) + "-year running mean"
    
    # enter a file description
    f.description = (str(run) + "-year running mean of " + var + " of the piControl run of the " + nl.models[mod] + 
                     " model. For memory reasons this file only contains one level of the data (level " + 
                     str(levs[lev]/100) + " hPa)." + desc_add)
    
    # enter the file creation date
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    # close the dataset
    f.close()
    
    ti_temp = timeit.default_timer()
    del_t = ti_temp - start
    frac = (lev+1) / len(levs)
    remaining = np.round((del_t / frac - del_t) / 60, decimals=1)
    print_one_line(str(np.round((lev+1) / len(levs) * 100, decimals=1)) + " %  " + str(remaining) + " min remain\n\n")  
        
    # set ti_old to the value of ti_temp
    ti_old = copy.deepcopy(ti_temp)
    
    # raise Exception("")
# end for lev
print("\nTotal time: " + str((timeit.default_timer() - start)/60) + " minutes")
