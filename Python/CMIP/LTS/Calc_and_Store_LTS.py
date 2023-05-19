"""
Calculate Lower Tropospheric Stability (LTS).

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
import numexpr as ne
import matplotlib.colors as colors
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti
import timeit
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_AnMean import an_mean
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Regrid_Data import remap as remap2
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_calcsatspechum import calcsatspechum, da_calcsatspechum
from Functions.Func_Set_ColorLevels import set_levels, set_levs_norm_colmap

from Extra_Functions.lcl import lcl as lcl_func


#%% experiment
# exp = "abrupt-4xCO2"
exp = "piControl"


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 24
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 5
# end try except


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    a4x = "abrupt4xCO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    a4x = "abrupt-4xCO2"
# end if elif


#%% get the ensemble
import Namelists.CMIP_Dictionaries as cmip_d

if (exp == "abrupt4xCO2") | (exp == "abrupt-4xCO2"):
    ens_n = cmip_d.mod_ens_a4x[cmip[-1]][nl.models[mod]]
    ens_d = "_" + ens_n
elif exp == "piControl":
    ens_n = cmip_d.mod_ens_pic[cmip[-1]][nl.models[mod]]
    ens_d = "_" + ens_n
# end if elif

if (ens_d == "_r1i1p1f1") | (ens_d == "_r1i1p1"):
    ens_d = ""
# end if


#%% number of years and runnig mean years
n_yr = 150
run = 21


#%% tas variable
t_var = "tas"
thr_str = 20
lts_var = "lts"  # because of the error I made originally naming LTS EIS


#%% set paths

# model data path
# --> must contain tas, ps, and ta
data_path = ""

# output path
out_path = ""


#%% load ta and hus file list
ta_f_list = sorted(glob.glob(data_path + exp + "_ta_Files" + ens_d + "/*.nc"), key=str.casefold)

# load in the respective first ta and hus file
ta_nc = Dataset(ta_f_list[0])


#%% load the nc files
tas_nc = Dataset(glob.glob(data_path + f"tas_Amon*{exp}*.nc")[0])
ps_nc = Dataset(glob.glob(data_path + f"ps_Amon*{exp}*.nc")[0])


#%%# get lat, lon, and levels for the model data
lat2d = tas_nc.variables["lat"][:]
lon2d = tas_nc.variables["lon"][:]
lat = ta_nc.variables["lat"][:]
lon = ta_nc.variables["lon"][:]

# get the levels
levs = ta_nc.variables["plev"][:] / 100  # convert to hPa


#%% find the vertical index of 700 and 850 hPa
i700 = np.argmin(np.abs(levs - 700))


#%% if the experiment is piControl adjust the number of years
if exp == "piControl":
    n_yr = int(tas_nc.dimensions["time"].size / 12)
# end if    
# n_yr = 300


#%% load files
tas = tas_nc.variables["tas"][:(n_yr*12), :, :]
ps = ps_nc.variables["ps"][:(n_yr*12), :, :]
ta700 = da.ma.masked_array(ta_nc.variables["ta"][:, i700, :, :], lock=True)


#%% if the lats of 3d and 2d differ from one another interpolate the 2d variables to the 3d grid

if len(lat) != len(lat2d):
    ps_re = np.zeros((12*n_yr, len(lat), len(lon)))
    tas_re = np.zeros((12*n_yr, len(lat), len(lon)))
    # huss_re = np.zeros((12*n_yr, len(lat), len(lon)))
    
    lat_o = lat2d + 90
    lat_t = lat + 90
    
    print("\n\nRegridding surfaces variables because of different latitudes from atmospheric variables...\n\n")
    
    for mon in np.arange(12*n_yr):
        ps_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, ps[mon, :, :], verbose=True)
        tas_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, tas[mon, :, :], verbose=True)
        # huss_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, huss[mon, :, :], verbose=True)
    # end for mon
    
    # replace the variabels with the regridded ones
    ps = ps_re
    tas = tas_re
    # huss = huss_re
    
# end if


#%%  loop over the files and concatenate the values via dask
print("\n\nLoading the ta data...\n\n")
if (len(ta_f_list) > 1):
    for i in np.arange(1, len(ta_f_list)):
        print_one_line(f"{i+1}/{len(ta_f_list)}  ")
        ta700 = da.concatenate([ta700, da.ma.masked_array(Dataset(ta_f_list[i]).variables["ta"][:, i700, :, :], 
                                                          lock=True)])
        # hus700 = da.concatenate([hus700, da.ma.masked_array(Dataset(hus_f_list[i]).variables["hus"][:, i700, :, :], 
        #                                                     lock=True)])
    # end for i
else:
    print("\nOnly one ta and q a4x file...\n")
# end if else

print("level " + str(Dataset(ta_f_list[0]).variables["plev"][:][i700]/100) + " hPa")

# reduce the arrays to the 150 years of the experiment and the corresponding control run
ta700 = ta700[:(n_yr*12), :, :]
# hus700 = hus700[:(n_yr*12), :, :]


#%% calculate the LTS
lts = np.zeros((n_yr*12, len(lat), len(lon)))

lts = np.zeros((n_yr*12, len(lat), len(lon)))
for yr, i in enumerate(np.arange(0, 12*n_yr, 12)):
    print_one_line(str(yr+1) + "/" + str(n_yr) + "  ")

    lts[i:i+12, :, :] = (ta700[i:i+12, :, :] * (1000 / 700)**0.286 - tas[i:i+12, :, :] * 
                         (100000 / ps[i:i+12, :, :])**0.286)
# end for yr, i    


#%% store the regression results as netCDF files
print("\nGenerating nc file...\n")

out_name = (f"{exp}/LTS_{nl.models_n[mod]}_{exp}_{ens_n}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
if cmip == "CMIP6":
    f.setncattr("variant_label", ens_n)
# end if        

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))
f.createDimension("time", n_yr*12)

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

# create the wind speed variable
lts_nc = f.createVariable("lts", "f4", ("time", "lat", "lon"))
ta700_nc = f.createVariable("ta700", "f4", ("time", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat

lts_nc[:] = lts
ta700_nc[:] = ta700

add_desc = ""

lts_nc.description = "LTS in K"
ta700_nc.description = "Ta at 700 hPa in K"

# add attributes
f.description = ("LTS and Ta at 700 hPa; see Ceppi and Gregory (2017) in PNAS.  Model: " + nl.models_pl[mod])
f.history = "Created " + ti.strftime("%d/%m/%y")

lts_nc.units = "K"
ta700_nc.units = "K"

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()



