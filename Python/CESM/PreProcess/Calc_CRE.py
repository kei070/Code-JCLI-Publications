"""
Calculate CRE.

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
    case = "dQ01yr_4xCO2"
# end try except
    

#%% set paths
data_path = f"Data/{case}/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_lw = sorted(glob.glob(data_path + f"/FLNT_{case}_*.nc"), key=str.casefold)
f_lwc = sorted(glob.glob(data_path + f"/FLNTC_{case}_*.nc"), key=str.casefold)
f_sw = sorted(glob.glob(data_path + f"/FSNT_{case}_*.nc"), key=str.casefold)
f_swc = sorted(glob.glob(data_path + f"/FSNTC_{case}_*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_lw[0][-12:-8])
f_sto = int((f_lw[-1][-7:-3]))


#%% number of years
years = np.arange(f_sta, f_sto + 1)
n_yr = len(years)


#%% load the files
nc_lw = Dataset(f_lw[0])
nc_lwc = Dataset(f_lwc[0])
nc_sw = Dataset(f_sw[0])
nc_swc = Dataset(f_swc[0])


#%% load the values
lw = nc_lw.variables["FLNT"][:]
lwc = nc_lwc.variables["FLNTC"][:]
sw = nc_sw.variables["FSNT"][:]
swc = nc_swc.variables["FSNTC"][:]

lat = nc_lw.variables["lat"][:]
lon = nc_lw.variables["lon"][:]


#%% calculate the dCRE
lw_cre = lw - lwc
sw_cre = sw - swc
cre = -lw + sw


#%% test plots
lw_an = glob_mean(an_mean(lw), lat, lon)
sw_an = glob_mean(an_mean(sw), lat, lon)
lwc_an = glob_mean(an_mean(lwc), lat, lon)
swc_an = glob_mean(an_mean(swc), lat, lon)

lw_cre_an = glob_mean(an_mean(lw_cre), lat, lon)
sw_cre_an = glob_mean(an_mean(sw_cre), lat, lon)


#%%
fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

# axes.plot(lw_an, label="LW", c="blue")
# axes.plot(lwc_an, label="LWC", c="green")
# axes.plot(sw_an, label="SW", c="red")
# axes.plot(swc_an, label="SWC", c="orange")

# axes.plot(sw_an - swc_an, label="SW - SWC (positive-down)", c="red")
# axes.plot(lw_an - lwc_an, label="LW - LWC (positive-down)", c="red")
axes.plot(-lw_an+sw_an, label="TOM")
axes.plot(-lwc_an+swc_an, label="TOMC")
axes.plot(-lw_cre_an+sw_cre_an, label="CRE")

# axes.plot(sw_an-lw_an, label="SW-LW", c="blue")
# axes.plot(swc_an-lwc_an, label="SWC-LWC", c="red")

axes.legend()

pl.show()
pl.close()


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"CRE_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% stor the CRE
out_name = (f"CRE_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("time", np.shape(cre)[0])
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = np.repeat(np.arange(f_sta, f_sto+1), 12)

# create the variable
cre_nc = f.createVariable("CRE", "f4", ("time", "lat", "lon"))
lw_cre_nc = f.createVariable("LW_CRE", "f4", ("time", "lat", "lon"))
sw_cre_nc = f.createVariable("SW_CRE", "f4", ("time", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

lw_cre_nc[:] = lw_cre
sw_cre_nc[:] = sw_cre
cre_nc[:] = cre

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

cre_nc.description = "cloud radiatice effect: CRE = -LW_CRE + SW_CRE; positive down"
lw_cre_nc.description = "long-wave CRE; positive up"
sw_cre_nc.description = "short-wave CRE; positive down"

# add attributes
f.description = (f"Cloud radiative effect (CRE) for case {case} simulated with CESM2-SOM.\n" +
                 "The data correspond to monthly means.\n" +
                 "Model run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()