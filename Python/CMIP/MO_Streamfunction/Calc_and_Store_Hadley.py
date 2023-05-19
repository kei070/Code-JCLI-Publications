"""
Calculate, plot, and store the Hadley cell from mpsi.

Be sure to set data_dir and possibly out_path.
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
import progressbar as pg
import dask.array as da
import time as ti
import xarray as xr

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)
from Functions.Func_Extract_Geographic_Region import extract_region
from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set CMIP
cmip_v = 6


#%% set model number
mod = 48


#%% set the variable name
var = "mpsi"


#%% load the namelist and the model name
import Namelists.CMIP_Dictionaries as cmip_dic
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    a4x = "abrupt4xCO2"
    cmip = "CMIP5"
elif cmip_v == 6:
    import Namelists.Namelist_CMIP6 as nl
    a4x = "abrupt-4xCO2"
    cmip = "CMIP6"
# end if elif

mod_d = nl.models[mod]
mod_n = nl.models_n[mod]
mod_pl = nl.models_pl[mod]


#%% set paths
data_dir = ""
out_path = data_dir + f"/{cmip}/Data/{mod_d}/"


#%% get the branch year
b_yr = nl.b_times[mod_d]


#%% load data
mpsi_nc = Dataset(glob.glob(data_dir + f"{cmip}/Data/{nl.models[mod]}/{a4x}_mpsi_Files*/*.nc")[0])
mpsi_pi_nc = Dataset(glob.glob(data_dir + f"{cmip}/Data/{nl.models[mod]}/piControl_mpsi_Files*/*.nc")[0])


# get the streamfunction values
mpsi_mon = np.array(xr.open_mfdataset(glob.glob(data_dir + 
                                                f"{cmip}/Data/{nl.models[mod]}/{a4x}_mpsi_Files*/*.nc")).mpsi)
# mpsi_mon = mpsi_mon[:, :, :]

mpsi_pi_mon = np.array(xr.open_mfdataset(glob.glob(data_dir + f"/{cmip}/Data/{nl.models[mod]}/" + 
                                                   "piControl_mpsi_Files*/*.nc")).mpsi)


mpsi_21pi_mon = np.array(xr.open_mfdataset(glob.glob(data_dir + f"/{cmip}/Data/{nl.models[mod]}/" + 
                                                     "mpsi_piControl_21YrRunMean_Files*/*.nc")).mpsi)

# aggregate to annual means
mpsi = an_mean(mpsi_mon)
mpsi_21pi = np.mean(mpsi_21pi_mon, axis=0)
mpsi_pi = an_mean(mpsi_pi_mon)

lat =  mpsi_nc.variables["lat"][:]

# extract the streamfunction for the latitudes between 15 and 25 N and S
hadley_n = mpsi[:, :, (lat > 15) & (lat < 25)]
hadley_s = mpsi[:, :,  (lat > -25) & (lat < -15)]
hadley_21pi_n = mpsi_21pi[:, :, (lat > 15) & (lat < 25)]
hadley_21pi_s = mpsi_21pi[:, :,  (lat > -25) & (lat < -15)]
hadley_pi_n = mpsi_pi[:, :, (lat > 15) & (lat < 25)]
hadley_pi_s = mpsi_pi[:, :,  (lat > -25) & (lat < -15)]

# get the time series via extracting the maximum streamfunction from the Hadley values
had_n_tser = np.max(hadley_n, axis=(1, 2))
had_s_tser = np.min(hadley_s, axis=(1, 2))
had_21pi_n_tser = np.max(hadley_21pi_n, axis=(1, 2))
had_21pi_s_tser = np.min(hadley_21pi_s, axis=(1, 2))
had_pi_n_tser = np.max(hadley_pi_n, axis=(1, 2))
had_pi_s_tser = np.min(hadley_pi_s, axis=(1, 2))


#%% test plots of the Hadley circulation metric
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=True)

axes[0].plot(np.arange(b_yr, b_yr+len(had_n_tser)), had_n_tser / 1E10, c="red", label="4xCO$_2$", linewidth=0.75)
axes[0].plot(had_pi_n_tser / 1E10, c="grey", label="piC", linewidth=0.5)
axes[0].plot(np.arange(b_yr, b_yr+150), had_21pi_n_tser / 1E10, c="black", label="21-yr piC")
axes[0].set_title("Northern hemispheric Hadley")

axes[0].set_xlabel("Years")
axes[0].set_ylabel("Hadley circulation strength in 10$^{10}$ kg s$^{-1}$")

axes[1].plot(np.arange(b_yr, b_yr+len(had_n_tser)), -had_s_tser / 1E10, c="red", label="4xCO$_2$", linewidth=0.75)
axes[1].plot(-had_pi_s_tser / 1E10, c="grey", label="piC", linewidth=0.5)
axes[1].plot(np.arange(b_yr, b_yr+150), -had_21pi_s_tser / 1E10, c="black", label="21-yr piC")
axes[1].set_title("Southern hemispheric Hadley (negative)")

axes[1].legend()

axes[1].set_xlabel("Years")
# axes[1].set_ylabel("Hadley circulation strength in kg s$^{-1}$")

fig.suptitle(mod_pl)

fig.subplots_adjust(wspace=0.05)

pl.show()
pl.close()


#%% store the annual values in a netcdf file
print("\nGenerating nc file...\n")
out_name = (f"Hadley_Strength_{mod_d}_piControl_{a4x}.nc")

f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
f.setncattr("branch_time", b_yr)
f.setncattr("variant_label_pic", cmip_dic.mod_ens_pic[cmip[-1]][mod_d])
f.setncattr("variant_label_a4x", cmip_dic.mod_ens_a4x[cmip[-1]][mod_d])

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("years_a4x", len(had_n_tser))
f.createDimension("years_21yr_piC", len(had_21pi_n_tser))
f.createDimension("years_piC", len(had_pi_n_tser))

# create the variables
hadn_a4x_nc = f.createVariable("Had_N_a4x", "f4", "years_a4x")
hadn_21yrpic_nc = f.createVariable("Had_N_21yr_pic", "f4", "years_21yr_piC")
hadn_pic_nc = f.createVariable("Had_N_pic", "f4", "years_piC")
hads_a4x_nc = f.createVariable("Had_S_a4x", "f4", "years_a4x")
hads_21yrpic_nc = f.createVariable("Had_S_21yr_pic", "f4", "years_21yr_piC")
hads_pic_nc = f.createVariable("Had_S_pic", "f4", "years_piC")

# pass the data into the variables
hadn_a4x_nc[:] = had_n_tser
hadn_21yrpic_nc[:] = had_21pi_n_tser
hadn_pic_nc[:] = had_pi_n_tser
hads_a4x_nc[:] = had_s_tser
hads_21yrpic_nc[:] = had_21pi_s_tser
hads_pic_nc[:] = had_pi_s_tser

# variable units
hadn_a4x_nc.units = "kg / s"
hadn_21yrpic_nc.units = "kg / s"
hadn_pic_nc.units = "kg / s"
hads_a4x_nc.units = "kg / s"
hads_21yrpic_nc.units = "kg / s"
hads_pic_nc.units = "kg / s"

# variable description
hadn_a4x_nc.description = (f"annual mean {a4x} NH Hadley circulation strength as calculated according to" + 
                           "Feldl and Bordoni (2016)")
hadn_21yrpic_nc.description = ("annual 21-year running mean piControl NH Hadley circulation strength as calculated " + 
                               "according to Feldl and Bordoni (2016)")
hadn_pic_nc.description = ("annual mean piControl NH Hadley circulation strength as calculated according to" + 
                           "Feldl and Bordoni (2016)")

# add some attributes
f.description = ("This file contains the annual mean Hadley circulation strength for the piControl (both annual and " +
                 f"21-year running mean) and {a4x} experiments for {mod_pl} in both NH and SH.\n" +
                 "The calculation of the Hadley circulation strength follows Feldl, N. and S. Bordoni (2016) in " +
                 "Journal of Climate 29, pp. 613-622. doi:10.1175/JCLI-D-150424.1")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()