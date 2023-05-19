"""
Compare vertical cross section of humidity change.

Generates Figs. S20 and S21 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir5 and data_dir6 as well as pl_path. Potentially change the paths in the for-loop.
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
from scipy.interpolate import interp1d as i1d
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
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "BCC-CSM2-MR", 
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

pl_path = ""

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
cmip_p = np.array([100000.,  92500.,  85000.,  70000.,  60000.,  50000.,
                    40000.,  30000.,  25000.,  20000.,  15000.,  10000.,
                     7000.,   5000.,   3000.,   2000.,   1000.,    500.,
                      100.])[::-1]


#%% loop over group one

# set the start and end year
stayr = 35
endyr = 45

stayr = 130
endyr = 150

dq_dic = {}
dt_dic = {}
dsat_dic = {}
dsat_zm_dic = {}
# dt_dic2 = {}
for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    dq_l = []
    dt_l = []
    dsat_l = []
    dsat_zm_l = []
    # dt_l2 = []
    
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
        data_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        
        # list files
        f_list = sorted(glob.glob(data_path + "/dQ_and_dT_Files*/*.nc"), key=str.casefold)[stayr:endyr]
        
        dsat_nc = Dataset(glob.glob(data_path + "dtas_Amon_*.nc")[0])
        lat2d = dsat_nc.variables["lat"][:]
        lon2d = dsat_nc.variables["lon"][:]
        dsat = glob_mean(np.mean(dsat_nc.variables["tas_ch"][:, stayr:endyr, :, :], axis=(0, 1)), lat2d, lon2d)
        dsat_zm = np.mean(dsat_nc.variables["tas_ch_rg"][:, stayr:endyr, :, :], axis=(0, 1, -1))
        
        dq_dt_nc = Dataset(f_list[0])
        dq = dq_dt_nc.variables["dQ"][:]
        dt = dq_dt_nc.variables["dT"][:]
        
        if len(f_list) > 1:
            for i in np.arange(1, len(f_list)):
                dq = da.concatenate([dq, da.ma.masked_array(Dataset(f_list[i]).variables["dQ"], lock=True)])
                dt = da.concatenate([dt, da.ma.masked_array(Dataset(f_list[i]).variables["dT"], lock=True)])
            # end for i
        else:
            print("\nOnly one dQ_and_dT file...\n")
        # end if
        
        # calculate zonal and annual mean
        dq = np.nanmean(dq.compute(), axis=(0, -1))
        dt = np.nanmean(dt.compute(), axis=(0, -1))
                
        # get lat and lon
        lat = dq_dt_nc.variables["lat"][:]
        plev = dq_dt_nc.variables["plev"][:]
        
        # time_coord = np.arange(endyr-stayr)

        # loop over all latitudes to extrapolate (if necessary) values to the surface
        if np.any(np.isnan(dq)):
            for la in np.arange(len(lat)):
                if np.any(np.isnan(dq[:, la])):
                    f_vert = i1d(x=plev[~np.isnan(dq[:, la])], y=dq[~np.isnan(dq[:, la]), la], fill_value="extrapolate")
                    dq[:, la] = f_vert(np.array(plev))
                    f_vert = i1d(x=plev[~np.isnan(dt[:, la])], y=dt[~np.isnan(dt[:, la]), la], fill_value="extrapolate")
                    dt[:, la] = f_vert(np.array(plev))
                # end if
            # end for la
        # end if            
        
        dq_i = np.zeros((len(plev), len(tlat)))
        f = i2d(lat, plev, dq)
        dq_i = f(tlat, cmip_p)

        dt_i = np.zeros((len(plev), len(tlat)))
        f = i2d(lat, plev, dt)
        dt_i = f(tlat, cmip_p)
        
        """
        for lev in np.arange(len(plev)):
            f = i2d(lat, time_coord, dq_an[:, lev, :])
            dq_an_i[:, lev, :] = f(tlat, time_coord)
        # end for lev
        
        dq_an_i2 = np.zeros((endyr-stayr, len(cmip_p), len(tlat)))
        for la in np.arange(len(tlat)):
            f = i2d(plev, time_coord, dq_an_i[:, :, la])
            dq_an_i2[:, :, la] = f(cmip_p, time_coord)
        # end for lev      
        """
        dq_l.append(dq_i)
        dt_l.append(dt_i)
        dsat_l.append(dsat)
        dsat_zm_l.append(dsat_zm)
        # dt_l2.append(dt)
        
    # end for i, mod_d
    
    dq_dic[g_n] = np.array(dq_l)
    dt_dic[g_n] = np.array(dt_l)
    dsat_dic[g_n] = np.array(dsat_l)
    dsat_zm_dic[g_n] = np.array(dsat_zm_l)
    # dt_dic2[g_n] = dt_l2
    
# end for g_n, group


#%% test
g1_hus = np.mean(dq_dic["G1"], axis=0) * 1000
g2_hus = np.mean(dq_dic["G2"], axis=0) * 1000
g1_ta = np.mean(dt_dic["G1"], axis=0)
g2_ta = np.mean(dt_dic["G2"], axis=0)
g1_sat = np.mean(dsat_dic["G1"], axis=0)
g2_sat = np.mean(dsat_dic["G2"], axis=0)
g1_sat_zm = np.mean(dsat_zm_dic["G1"], axis=0)
g2_sat_zm = np.mean(dsat_zm_dic["G2"], axis=0)

# calculate the normalised temperature
g1_ta_norm = g1_ta / g1_sat
g2_ta_norm = g2_ta / g2_sat
g1_hus_norm = g1_hus / g1_sat
g2_hus_norm = g2_hus / g2_sat

# calculate the temperature departure
g1_ta_dep = g1_ta - g1_sat_zm[None, :]
g2_ta_dep = g2_ta - g2_sat_zm[None, :]


#%% set the converted latitude
clat = np.sin(tlat / 180 * np.pi)
xticks = np.array([-75, -50, -30, -15, 0, 15, 30, 50, 75])


#%% plot the year 100-150 averages - specific humidity
pl_title = "Specific humidity"
col_map = cm.RdBu_r
levels = np.arange(-5, 5.1, 0.5)
ticks = np.arange(-5, 5.1, 1)
dlevels = np.arange(-2, 2.1, 0.2)
dticks = np.arange(-2, 2.1, 0.5)
    
x, y = np.meshgrid(clat, cmip_p/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_hus, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_hus, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Specific humidity in g kg$^{-1}$")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y,  g2_hus - g1_hus, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in g kg$^{-1}$")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("G2 minus G1")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 50, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 50, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax3.text(np.sin(-55/180*np.pi), 50, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(pl_title + f" anomaly averaged over years {stayr}$-${endyr} G1 and G2")

pl.savefig(pl_path + f"/PDF/SpecHum_G1_G2_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/SpecHum_G1_G2_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()

raise Exception
#%% plot the year 100-150 averages - temperature
pl_title = "Temperature"
col_map = cm.RdBu_r
levels = np.arange(-15, 15.1, 1)
ticks = np.arange(-15, 15.1, 3)
dlevels = np.arange(-5, 5.1, 0.5)
dticks = np.arange(-5, 5.1, 1)
    
x, y = np.meshgrid(clat, cmip_p/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_ta, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_ta, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Temperature in K")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y,  g2_ta - g1_ta, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in K")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("G2 minus G1")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

fig.suptitle(pl_title + f" anomaly averaged over years {stayr}$-${endyr} G1 and G2")

pl.savefig(pl_path + f"/PDF/Ta_G1_G2_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Ta_G1_G2_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 100-150 averages - normalised temperature
pl_title = "Normalised humidity"
col_map = cm.Blues
col_map2 = cm.RdBu
levels = np.arange(0, 1, 0.05)
ticks = np.arange(0, 1, 0.1)
dlevels = np.arange(-0.05, 0.051, 0.001)
dticks = np.arange(-0.05, 0.051, 0.01)
    
x, y = np.meshgrid(clat, cmip_p/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_hus_norm, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_hus_norm, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Normalised humidity in g kg$^{-1}$ K$^{-1}$")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y,  g2_hus_norm - g1_hus_norm, cmap=col_map2, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in g kg$^{-1}$ K$^{-1}$")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("G2 minus G1")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

fig.suptitle(pl_title + f" anomaly averaged over years {stayr}$-${endyr} G1 and G2")

pl.savefig(pl_path + f"/PDF/HusNorm_G1_G2_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/HusNorm_G1_G2_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 100-150 averages - normalised temperature
pl_title = "Normalised temperature"
col_map = cm.RdBu_r
levels = np.arange(-2, 2.1, 0.2)
ticks = np.arange(-2, 2.1, 0.5)
dlevels = np.arange(-0.5, 0.51, 0.05)
dticks = np.arange(-0.5, 0.51, 0.1)
    
x, y = np.meshgrid(clat, cmip_p/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_ta_norm, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_ta_norm, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Normalised temperature in KK$^{-1}$")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y,  g2_ta_norm - g1_ta_norm, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in KK$^{-1}$")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("G2 minus G1")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

fig.suptitle(pl_title + f" anomaly averaged over years {stayr}$-${endyr} G1 and G2")

pl.savefig(pl_path + f"/PDF/TaNorm_G1_G2_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/TaNorm_G1_G2_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 100-150 averages - normaised temperature
pl_title = "Temperature anomaly departure"
col_map = cm.RdBu_r
levels = np.arange(-10, 10.1, 0.5)
ticks = np.arange(-10, 10.1, 2)
dlevels = np.arange(-5, 5.1, 0.5)
dticks = np.arange(-5, 5.1, 1)
    
x, y = np.meshgrid(clat, cmip_p/100)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, g1_ta_dep, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("G1")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, g2_ta_dep, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Temperature departure in K")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("G2")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y,  g2_ta_dep - g1_ta_dep, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in K")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("G2 minus G1")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

fig.suptitle(pl_title + f" averaged over years {stayr}$-${endyr} G1 and G2")

pl.savefig(pl_path + f"/PDF/TaDep_G1_G2_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/TaDep_G1_G2_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the zonal mean dSAT normalised by global mean
clat = np.sin(tlat / 180 * np.pi)
xticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])

fig = pl.figure(figsize=(6, 4))
gs = gridspec.GridSpec(nrows=1, ncols=1) 
ax1 = pl.subplot(gs[:, :])

ax1.plot(clat, g1_sat_zm / g1_sat, c="blue", label="G1")
ax1.plot(clat, g2_sat_zm / g2_sat, c="red", label="G2")
ax1.axvline(x=0, c="black", linewidth=0.5)
ax1.legend()

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_xlabel("Latitude")
ax1.set_ylabel("Normalised $\Delta$SAT in KK$^{-1}$")
ax1.set_title(f"Normalised zonal mean $\Delta$SAT (years {stayr}$-${endyr})")

pl.show()
pl.close()


#%% plot
"""
yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
clat = np.sin(tlat / 180 * np.pi)
levels = np.linspace(-5, 5, 25)
ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
lev_pl = cmip_p / 100

x, y = np.meshgrid(clat, lev_pl)

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(16, 6), sharey=True)

p1 = axes[0].contourf(x, y, g1_test, cmap=cm.RdBu, levels=levels, extend="both")
# axes[0].invert_yaxis()
cb1 = fig.colorbar(p1, ax=axes[0], ticks=ticks)
cb1.set_label("Spec humidity in g/kg")

axes[0].set_title("G1")
axes[0].set_xlabel("Latitude")
axes[0].set_ylabel("Pressure in hPa")
axes[0].set_xticks(np.sin(yticks / 180 * np.pi))
axes[0].set_xticklabels(yticks)


p2 = axes[1].contourf(x, y, g2_test, cmap=cm.RdBu, levels=levels, extend="both")
axes[1].invert_yaxis()
cb2 = fig.colorbar(p2, ax=axes[1], ticks=ticks)
cb2.set_label("Spec humidity in g/kg")

axes[1].set_title("G2")
axes[1].set_xlabel("Latitude")
axes[1].set_xticks(np.sin(yticks / 180 * np.pi))
axes[1].set_xticklabels(yticks)

fig.subplots_adjust(wspace=0.05)
fig.suptitle("Profiles of specific humidity change abrupt4xCO2\n")

pl.show()
pl.close()
"""



