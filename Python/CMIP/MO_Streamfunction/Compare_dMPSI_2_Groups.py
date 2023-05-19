"""
Compare the mpsi (regridded) between two model groups.

Generates Fig. S19 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir5, data_dir6, and pl_path as well as possibly mpsi_path.
"""

#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import xarray as xr
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
from scipy.interpolate import interp2d as i2d
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Region_Mean import region_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]  # AMOC models
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]               # AMOC models   

models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "ACCESS-ESM1-5", "NorESM1_M", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models = {"G1":models1, "G2":models2}


#%% radiative kernels
kl = "Sh08"


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = "/MultModPlots/G1_G2_MPSI/"

os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% kernel dictionaries
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% load the CanESM5 grid as target for regridding
can_nc = Dataset(glob.glob(data_dir6 + "/CMIP6/Data/CanESM5/ts_*4xCO2*.nc")[0])
tlat = can_nc.variables["lat"][:]
tlon = can_nc.variables["lon"][:]
pllat = can_nc.variables["lat"][:]
pllon = can_nc.variables["lon"][:]


#%% loop over group one
mpsi_dic = {}
mpsi_pi_dic = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    mpsi_l = []
    mpsi_pi_l = []
    
    for i, mod_d in enumerate(group):
        
        print(f"\n{g_n} {mod_d}\n")
        
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            mod_p = nl5.models[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_p = nl6.models[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
        # end if
        
        # data path and name
        mpsi_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        
        mpsi_nc = Dataset(glob.glob(mpsi_path + f"/{a4x}_mpsi_Files*/*.nc")[0])
        mpsi_pi_nc = Dataset(glob.glob(mpsi_path + f"/piControl_mpsi_Files*/*.nc")[0])
        
        # load the files
        mpsi_an = an_mean(np.array(xr.open_mfdataset(sorted(glob.glob(mpsi_path + f"/{a4x}_mpsi_Files*/*.nc"), 
                                                            key=str.casefold)).mpsi))[:150, :, :]
        mpsi_pi_an = an_mean(np.array(xr.open_mfdataset(sorted(glob.glob(mpsi_path + f"/piControl_mpsi_Files*/*.nc"), 
                                                               key=str.casefold)).mpsi))[b_time:(b_time+150), :, :]
        
        # get lat and lon
        lat = mpsi_nc.variables["lat"][:]
        plev = mpsi_nc.variables["plev"][:]
        
        time_coord = np.arange(150)
        
        mpsi_an_i = np.zeros((150, len(plev), len(tlat)))
        mpsi_pi_an_i = np.zeros((150, len(plev), len(tlat)))
        
        for lev in np.arange(len(plev)):
            f = i2d(lat, time_coord, mpsi_an[:, lev, :])
            mpsi_an_i[:, lev, :] = f(tlat, time_coord)
            f = i2d(lat, time_coord, mpsi_pi_an[:, lev, :])
            mpsi_pi_an_i[:, lev, :] = f(tlat, time_coord)
        # end for lev            
        
        mpsi_l.append(mpsi_an_i)
        mpsi_pi_l.append(mpsi_pi_an_i)
        
    # end for i, mod_d
    
    mpsi_dic[g_n] = np.array(mpsi_l)
    mpsi_pi_dic[g_n] = np.array(mpsi_pi_l)
    
# end for g_n, group


#%% group averages
g1_mpsi = np.mean(mpsi_dic["G1"], axis=0)
g2_mpsi = np.mean(mpsi_dic["G2"], axis=0)
g1_mpsi_pi = np.mean(mpsi_pi_dic["G1"], axis=0)
g2_mpsi_pi = np.mean(mpsi_pi_dic["G2"], axis=0)


#%% average over the control run
g1_mpsi_pi = np.mean(g1_mpsi_pi, axis=0)
g2_mpsi_pi = np.mean(g2_mpsi_pi, axis=0)


#%% set the start and end year
stayr = 17
endyr = 22

# stayr = 34
# endyr = 45

#% average over the given years
g1_mpsi_l50 = np.mean(g1_mpsi[stayr:endyr], axis=0)
g2_mpsi_l50 = np.mean(g2_mpsi[stayr:endyr], axis=0)


#% cacluclate the delta
g1_dmpsi_l50 = g1_mpsi_l50 - g1_mpsi_pi
g2_dmpsi_l50 = g2_mpsi_l50 - g2_mpsi_pi


#% set the converted latitude
clat = np.sin(tlat / 180 * np.pi)
xticks = np.array([-75, -50, -30, -15, 0, 15, 30, 50, 75])


#% plot
levels = np.arange(-2.0, 2.1, 0.1)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
ticks = np.arange(-2, 2.1, 0.5)
dlevels = np.arange(-2, 2.1, 0.1)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
dticks = np.arange(-2, 2.1, 0.5)
x, y = np.meshgrid(clat, plev/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax3 = pl.subplot(gs[0, 14])
ax4 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_dmpsi_l50 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_dmpsi_l50 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

cb2 = fig.colorbar(p2, cax=ax3, ticks=ticks)
cb2.set_label("Change of MPSI in 10$^{10}$ kg s$^{-1}$")


p4 = ax4.contourf(x, y, (g2_mpsi_l50 - g1_mpsi_l50) / 1e10, cmap=cm.RdBu_r, extend="both", levels=dlevels)
cb4 = fig.colorbar(p4, cax=ax5, ticks=dticks)
cb4.set_label("Difference in 10$^{10}$ kg s$^{-1}$")

ax4.axvline(x=0, c="gray", linewidth=0.5)

ax4.set_xticks(np.sin(xticks / 180 * np.pi))
ax4.set_xticklabels(xticks)

ax4.set_title("G2 minus G1")
ax4.set_xlabel("Latitude")
ax4.set_yticklabels([])
ax4.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 50, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 50, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax4.text(np.sin(-55/180*np.pi), 50, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"Change of meridional overturning streamfunction G1 and G2 averaged over years {stayr+1}$-${endyr}")

pl.savefig(pl_path + f"/PDF/dMPSI_G1_G2_yr{stayr+1}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/dMPSI_G1_G2_yr{stayr+1}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
