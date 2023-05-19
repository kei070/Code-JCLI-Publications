"""
Calculate delta quantities for a given variable based on a linear regression in the control run.

Be sure to adjust paths in code block "set paths".
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


#%% set the variable name
try:
    var_n = sys.argv[1]
except:
    var_n = "seb"
# end try except 


#%% if the variable is not in the file name
var_fn = var_n
if (var_n == "sw") | (var_n == "lw"):
    var_fn = "tom"
# end if   
if (var_n == "swC") | (var_n == "lwC"):
    var_fn = "tomC"
# end if


#%% add the pressure level if applicable
try:
    plev = sys.argv[2]
except:    
    plev = ""  # "_700hPa"
# end if    
var_fn = var_fn + plev


#%% adjust the output name
var_on = var_n + plev 


#%% setup case name
try:
    case_con = sys.argv[4]
except:    
    case_con = "Proj2_KUE"      # CESM2.1
# end try except

# select the timespan to be taken from the control run
try:
    sta_yr = int(sys.argv[5])
    end_yr = int(sys.argv[6])
except:
    sta_yr = 51
    end_yr = 80
# end try except   

try:
    case = sys.argv[3]
except:  
    case = "dQ01yr_4xCO2"
# end try except


#%% set paths
data_path = f"/Data/{case}/"
data_con_path = f"/Data/{case_con}/"
out_path = f"/Data/{case}/"


#%% load the data
exp_fn = glob.glob(data_path + f"{var_fn}_{case}*.nc")[0]
var_nc = Dataset(exp_fn)

con_fn = glob.glob(data_con_path + f"{var_fn}_{case_con}*.nc")[0]
var_con_nc = Dataset(con_fn)


#%% get lat and lon
lat = var_nc.variables["lat"][:]
lon = var_nc.variables["lon"][:]


#%% extract the years from the file names
con_sta = int(con_fn[-12:-8])
con_sto = int(con_fn[-7:-3])
exp_sta = int(exp_fn[-12:-8])
exp_sto = int(exp_fn[-7:-3])


#%% get the indices of the start and end year in the control run
sta_i = sta_yr - con_sta
end_i = sta_i + (end_yr - sta_yr) + 1


#%% load data and directly calculate the annual mean
var = var_nc.variables[var_n][:]
var_con = var_con_nc.variables[var_n][sta_i*12:end_i*12]


#%% reshape the data to (year, month, lat, lon)
nmon = np.shape(var)[0]
nyr = int(nmon / 12)

nmon_con = np.shape(var_con)[0]
nyr_con = int(nmon_con / 12)

var_rs = np.reshape(var, (nyr, 12, len(lat), len(lon)))
var_con_rs = np.reshape(var_con, (nyr_con, 12, len(lat), len(lon)))

# calculate the control run climatology
con_clim = np.mean(var_con_rs, axis=0)


#%% generate the control run reference data which will be subtracted from the experiment data
sl = np.zeros((12, len(lat), len(lon)))
yi = np.zeros((12, len(lat), len(lon)))

for m in np.arange(12):
    for i in np.arange(len(lat)):
        for j in np.arange(len(lon)):
            sl[m, i, j], yi[m, i, j] = lr(np.arange(sta_yr, end_yr+1), var_con_rs[:, m, i, j])[:2]
        # end for i
    # end for j
# end for m


#%% test plot
# pl.imshow((yi + np.arange(exp_sta, exp_sto+1)[:, None, None] * sl)[10, :, :], origin="lower")
# pl.colorbar()


#%% calculate the delta quantity
dvar = np.zeros(np.shape(var_rs))

# calculate it with respect to the control trend
for m in np.arange(12):
    dvar[:, m, :, :] = var_rs[:, m, :, :] - (yi[m, :, :] + np.arange(exp_sta, exp_sto+1)[:, None, None] * sl[m, :, :])
# end for m

# calculate it with respect to the control climatology
dvar_clim = var_rs - con_clim[None, :, :, :]


#%% test plot the global mean
"""
gl_mean = glob_mean(dvar, lat, lon)

fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(14, 6))
axes[0].plot(gl_mean[:, 0], label="Jan")
axes[0].plot(gl_mean[:, 5], label="Jun")
axes[0].plot(gl_mean[:, 8], label="Sep")
axes[0].legend()

axes[1].plot(gl_mean[0, :], label="Year 0")
axes[1].plot(gl_mean[5, :], label="Year 5")
axes[1].plot(gl_mean[10, :], label="Year 10")
# axes[1].plot(gl_mean[50, :], label="Year 50")
axes[1].legend()

pl.show()
pl.close()
"""


#%% if applicable calculate the potential temperature change
"""
if var_n == "T":
    
    var_pot = an_mean(var_nc.variables[var_n + "_pot"][:])
    var_pot_con = an_mean(var_con_nc.variables[var_n + "_pot"][:])
    
    sl_pot = np.zeros((len(lat), len(lon)))
    yi_pot = np.zeros((len(lat), len(lon)))
    for i in np.arange(len(lat)):
        for j in np.arange(len(lon)):
            sl_pot[i, j], yi_pot[i, j] = lr(np.arange(con_sta, con_sto+1), var_pot_con[:, i, j])[:2]
        # end for i
    # end for j

    # calculate the delta quantity
    dvar_pot = var_pot - (yi_pot + np.arange(exp_sta, exp_sto+1)[:, None, None] * sl_pot)
    
# end if    
"""

#%% remove any exisiting file
pre_path = data_path
pre_name = (f"d{var_on}_Mon_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else


#%% compare the two differently calculated changes
yr = 1
mon = 6

fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(14, 7))

p1 = axes[0].pcolormesh(dvar[yr, mon, :, :])
fig.colorbar(p1, ax=axes[0])
axes[0].set_title(f"d{var_n} with respect to control trend")

p2 = axes[1].pcolormesh(dvar_clim[yr, mon, :, :])
fig.colorbar(p2, ax=axes[1])
axes[1].set_title(f"d{var_n} with respect to control climatology")

pl.show()
pl.close()


#%% store the delta file in netcdf file
out_name = (f"d{var_on}_Mon_{case}_{exp_sta:04}-{exp_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("year", nyr)
year_nc = f.createVariable("year", "i4", "year")
year_nc[:] = np.arange(exp_sta, exp_sto+1)

f.createDimension("month", 12)
month_nc = f.createVariable("month", "i4", "month")
month_nc[:] = np.arange(12)

# create the variable
dvar_nc = f.createVariable(f"d{var_n}", "f4", ("year", "month", "lat", "lon"))
dvar_clim_nc = f.createVariable(f"d{var_n}_clim", "f4", ("year", "month", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

dvar_nc[:] = dvar
dvar_clim_nc[:] = dvar_clim

desc = ""
"""
# if applicable add the potential temperature change
if var_n == "T":
    # create the wind speed variable
    dvar_pot_nc = f.createVariable(f"d{var_n}_pot", "f4", create_var)
    dvar_pot_nc[:] = dvar_pot
    desc = " The file also contains the potential temperature."    
    dvar_pot_nc.units = "K"
# end if    
"""

dvar_nc.description = var_on + " difference to control run calculated via control-run trend"
dvar_clim_nc.description = var_on + " difference to control run calculated via control-run climatology"

# add attributes
f.description = (f"{var_on} difference to control run (control case: {case_con}) for case {case} in CESM2 SOM.{desc}\n" + 
                 f"Years {sta_yr} to {end_yr} of the control run were used to generate a linear regression which\n" +
                 "is subtracted from the experiment run to obtain the change over time of the quantity.\n" +
                 "The data correspond to monthly means.\n" +
                 "Model run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()





