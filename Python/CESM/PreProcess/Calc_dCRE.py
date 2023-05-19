"""
Calculate CESM2-SOM dCRE.

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


#%% setup case name
case_con = "Proj2_KUE"
case_com = "Proj2_KUE_4xCO2"
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except
    

#%% set paths
data_con_path = f"/Data/{case_con}/"
data_path = f"/Data/{case}/"
out_path = f"/Data/{case}/"


#%% list and sort the files
f_cre = sorted(glob.glob(data_path + f"/CRE_{case}_*.nc"), key=str.casefold)
f_cre_con = sorted(glob.glob(data_con_path + f"/CRE_{case_con}_*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_cre[0][-12:-8])
f_sto = int((f_cre[-1][-7:-3]))


#%% number of years
years = np.arange(f_sta, f_sto + 1)
n_yr = len(years)


#%% load one file to lat and lon
nc_cre = Dataset(f_cre[0])
nc_cre_con = Dataset(f_cre_con[0])


#%% load the values
cre_lw = nc_cre.variables["LW_CRE"][:]
cre_sw = nc_cre.variables["SW_CRE"][:]

cre_lw_con = nc_cre_con.variables["LW_CRE"][:]
cre_sw_con = nc_cre_con.variables["SW_CRE"][:]

lat = nc_cre.variables["lat"][:]
lon = nc_cre.variables["lon"][:]


#%% calcultate control climatologies
n_yr_con = int(np.shape(cre_lw_con)[0]/12)
cre_lw_con_rs = np.reshape(cre_lw_con, (n_yr_con, 12, len(lat), len(lon)))
cre_sw_con_rs = np.reshape(cre_sw_con, (n_yr_con, 12, len(lat), len(lon)))

cre_lw_con_clim = np.mean(cre_lw_con_rs, axis=0)
cre_sw_con_clim = np.mean(cre_sw_con_rs, axis=0)


#%% calculate the dCREs
dcre_lw = np.reshape(cre_lw, (n_yr, 12, len(lat), len(lon))) - cre_lw_con_clim[None, :, :, :]
dcre_sw = np.reshape(cre_sw, (n_yr, 12, len(lat), len(lon))) - cre_sw_con_clim[None, :, :, :]


#%% remove any exisiting file
pre_path = data_path
pre_name = (f"dCRE_Mon_{case}_*.nc")

try:
    pre_fn = glob.glob(pre_path + pre_name)[0]
    os.remove(pre_fn)
    print("Existing file removed...")
except:
    print("No file to remove...")
# end if else  


#%% store the delta file in netcdf file
out_name = (f"dCRE_Mon_{case}_{f_sta:04}-{f_sto:04}.nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("year", n_yr)
year_nc = f.createVariable("year", "i4", "year")
year_nc[:] = years

f.createDimension("month", 12)
month_nc = f.createVariable("month", "i4", "month")
month_nc[:] = np.arange(12)

# create the variable
dcre_lw_nc = f.createVariable("dcre_lw", "f4", ("year", "month", "lat", "lon"))
dcre_sw_nc = f.createVariable("dcre_sw", "f4", ("year", "month", "lat", "lon"))

cre_lw_clim_nc = f.createVariable("cre_lw_clim", "f4", ("month", "lat", "lon"))
cre_sw_clim_nc = f.createVariable("cre_sw_clim", "f4", ("month", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
print("\nStoring data...\n\n")

dcre_lw_nc[:] = dcre_lw
dcre_sw_nc[:] = dcre_sw

cre_lw_clim_nc[:] = cre_lw_con_clim
cre_sw_clim_nc[:] = cre_sw_con_clim

dcre_lw_nc.description = ("long-wave CRE difference to control run calculated with respect to control-run climatology " +
                          "(positive up)")
dcre_sw_nc.description = ("short-wave CRE difference to control run calculated with respect to control-run climatology" +
                          "(positive down)")

cre_lw_clim_nc.description = "long-wave CRE control-run climatology"
cre_sw_clim_nc.description = "short-wave CRE control-run climatology"

# add attributes
f.description = (f"CRE difference to control run (control case: {case_con}) for case {case} in CESM2 SOM.\n" + 
                 "28 years of the control run were used to generate a climatology which\n" +
                 "is subtracted from the experiment run to obtain the change over time of the quantity.\n" +
                 "Also contains said climatology.\n" +
                 "The data correspond to monthly means.\n" +
                 "Model run started by Kai-Uwe Eiselt.\nModel set up by Rune Grand Graversen.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()