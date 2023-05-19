"""
Calculate annual mean zonal mean vertical cross section for a given variable Ta.

Generates Fig. S15 and S18 in Eiselt and Graversen (2023), JCLI.

Be sure to set the paths in code block "set paths" correctly.
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


#%% variable
var = "T"


#%% set up colorbars
cmp_ch = cm.RdBu_r
cmp_abs = cm.Reds


#%% setup case name
case_con = "Proj2_KUE"
case_com = "Proj2_KUE_4xCO2"
try:
    case = sys.argv[2]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% set up some dictionaries with title etc.
cb_labs = {"T":"T in K", "V":"Northward wind in ms$^{-1}$", "Q":"Q in kg/kg", "RELHUM":"Relative humidity in %", 
           "VT":"Meridional heat transport", "OMEGA":"Vertical wind speed", "OMEGAT":"Vertical heat transport",
           "QRL":"Longwave heating rate"}

pl_titles = {"T":"Vertical temperature profile ", "V":"Vertical v-wind profile ", "Q":"Vertical Q profile ", 
             "RELHUM":"Vertical Q$_{rel}$ profile ", "VT":"Vertical VT profile ", "OMEGA":"Vertical omgea profile ", 
             "OMEGAT":"Vertical OMEGAT profile ", "QRL":"Longwave heating rate profile "}

levels_chs = {"T":np.array([-15, -12.5, -10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 12.5, 15]) / 1.5, 
              "V":np.array([-15, -12.5, -10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 12.5, 15]), 
              "Q":np.array([-15, -12.5, -10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 12.5, 15]) / 1000, 
             "RELHUM":np.array([-35, -25, -15, -5, -1, 1, 5, 15, 25, 35]), 
             "VT":np.array([-1000, -800, -600, -400, -200, -100, 100, 200, 400, 600, 800, 1000]), 
             "OMEGA":np.array([-0.02, -0.0175, -0.015, -0.0125, -0.01, -0.0075, -0.005, -0.0025, 0.0025, 0.005, 0.0075, 
                               0.01, 0.0125, 0.015, 0.0175, 0.02]), 
             "OMEGAT":np.array([-10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10]),
             "QRL":np.array([-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 
                             2]) / 1E5}

levels_abss = {"T":np.array([200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]), 
               "V":np.array([-15, -12.5, -10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 12.5, 15]), 
               "Q":np.array([-15, -12.5, -10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 12.5, 15]) / 1000, 
               "RELHUM":np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 
               "VT":np.array([-1000, -800, -600, -400, -200, -100, 100, 200, 400, 600, 800, 1000]), 
               "OMEGA":np.array([-0.02, -0.0175, -0.015, -0.0125, -0.01, -0.0075, -0.005, -0.0025, 0.0025, 0.005, 0.0075, 
                                 0.01, 0.0125, 0.015, 0.0175, 0.02]), 
               "OMEGAT":np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
               "QRL":np.array([-7, -6, -5, -4, -3, -2, -1, 0]) / 1E5}

cb_lab = cb_labs[var]

pl_title = pl_titles[var]

levels_ch = levels_chs[var]

levels_abs = levels_abss[var]


#%% set paths
data_con_path = f"/Data/{case_con}/"
data_com_path = f"/Data/{case_com}/"
data_path = f"/Data/{case}/"

pl_path = ""


#%% load TREFHT
dsat_com_nc = Dataset(glob.glob(data_com_path + "/dTREFHT_Mon_*nc")[0])
dsat_exp_nc = Dataset(glob.glob(data_path + "/dTREFHT_Mon_*nc")[0])

lat = dsat_com_nc.variables["lat"][:]
lon = dsat_com_nc.variables["lon"][:]

dsat_com = np.mean(dsat_com_nc.variables["dTREFHT"][:40, :, :, :], axis=1)
dsat_exp = np.mean(dsat_exp_nc.variables["dTREFHT"][:40, :, :, :], axis=1)

dsat_com_zm = np.mean(dsat_com, axis=-1)
dsat_exp_zm = np.mean(dsat_exp, axis=-1)

dsat_com_nt = region_mean(dsat_com, [0], [360], [0], [30], lat, lon)
dsat_com_st = region_mean(dsat_com, [0], [360], [-30], [0], lat, lon)
dsat_exp_nt = region_mean(dsat_exp, [0], [360], [0], [30], lat, lon)
dsat_exp_st = region_mean(dsat_exp, [0], [360], [-30], [0], lat, lon)


#%% list and sort the files
f_con = sorted(glob.glob(data_con_path + f"Files_3d/{var}_Files/{var}_{case_con}_*.nc"), key=str.casefold)
f_com = sorted(glob.glob(data_com_path + f"Files_3d/{var}_Files/{var}_{case_com}_*.nc"), key=str.casefold)
f_exp = sorted(glob.glob(data_path + f"Files_3d/{var}_Files/{var}_{case}_*.nc"), key=str.casefold)


#%% load nc files
nc_con = Dataset(f_con[0])
nc_com = Dataset(f_com[0])
nc_exp = Dataset(f_exp[0])


#%% load the grid
plev = nc_con.variables["lev"][:]
lat = nc_con.variables["lat"][:]
lon = nc_con.variables["lon"][:]


#%% calculate the global mean dTREFHT
dsat_com_gm = glob_mean(dsat_com_zm, lat, from_zon=True)
dsat_exp_gm = glob_mean(dsat_com_zm, lat, from_zon=True)


#%% load the data
con = da.ma.masked_array(nc_con.variables[var], lock=True)
com = da.ma.masked_array(nc_com.variables[var], lock=True)
exp = da.ma.masked_array(nc_exp.variables[var], lock=True)


#%% load the rest of the data

# load the forced run data
if len(f_exp) > 1:
    for i in np.arange(1, 40): # len(f_exp)):
        dataset_com = Dataset(f_com[i])
        dataset_exp = Dataset(f_exp[i])
        com = da.concatenate([com, da.ma.masked_array(dataset_com.variables[var], lock=True)])
        exp = da.concatenate([exp, da.ma.masked_array(dataset_exp.variables[var], lock=True)])
        # dataset.close()
    # end for i
# end if

# load the control run data separately
if len(f_con) > 1:
    for i in np.arange(1, len(f_con)):
        dataset = Dataset(f_con[i])
        con = da.concatenate([con, da.ma.masked_array(dataset.variables[var], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% calculate the monthly zonal mean slices
con_zm = np.mean(con.compute(), axis=-1)
com_zm = np.mean(com.compute(), axis=-1)
exp_zm = np.mean(exp.compute(), axis=-1)


#%% region mean
con_nt = np.mean(region_mean(con.compute(), [0], [360], [0], [30], lat, lon), axis=0)
com_nt = an_mean(region_mean(com.compute(), [0], [360], [0], [30], lat, lon))
exp_nt = an_mean(region_mean(exp.compute(), [0], [360], [0], [30], lat, lon))
con_st = np.mean(region_mean(con.compute(), [0], [360], [-30], [0], lat, lon), axis=0)
com_st = an_mean(region_mean(com.compute(), [0], [360], [-30], [0], lat, lon))
exp_st = an_mean(region_mean(exp.compute(), [0], [360], [-30], [0], lat, lon))


#%% calculate the anomalies
dcom_nt = com_nt - con_nt[None, :]
dcom_st = com_st - con_st[None, :]
dexp_nt = exp_nt - con_nt[None, :]
dexp_st = exp_st - con_st[None, :]


#%% calculate normalised regional means
dcom_nt_norm = dcom_nt / dsat_com_nt[:, None]
dcom_st_norm = dcom_st / dsat_com_st[:, None]
dexp_nt_norm = dexp_nt / dsat_exp_nt[:, None]
dexp_st_norm = dexp_st / dsat_exp_st[:, None]

ddcom_nt = dcom_nt - dsat_com_nt[:, None]
ddcom_st = dcom_st - dsat_com_st[:, None]
ddexp_nt = dexp_nt - dsat_exp_nt[:, None]
ddexp_st = dexp_st - dsat_exp_st[:, None]


#%% calculate the annual zonal mean slices
con_an_zm = an_mean(con_zm)
com_an_zm = an_mean(com_zm)
exp_an_zm = an_mean(exp_zm)


#%% calculate the linear trends for all control run cells
sl, yi = np.zeros((len(plev), len(lat))), np.zeros((len(plev), len(lat)))

ntim = np.arange(np.shape(con_zm)[0])

for le in np.arange(len(plev)):
    for la in np.arange(len(lat)):
        sl[le, la], yi[le, la] = lr(ntim, con_zm[:, le, la])[:2]
    # end for la
# end for le


#%%  calculate the control run trend data
ntim_exp = np.arange(np.shape(exp_zm)[0])  # number of years of the experiment available

con_trend = np.zeros(np.shape(exp_zm))
con_trend[:] = sl[None, :, :] * ntim_exp[:, None, None] + yi[None, :, :]


#%% calculate the delta quantities
dcom_zm = com_zm - con_trend
dexp_zm = exp_zm - con_trend


#%% calculate the annual mean delta quantities
dcom_an_zm = an_mean(dcom_zm)
dexp_an_zm = an_mean(dexp_zm)
nyrs_exp = np.arange(len(dcom_an_zm))


#%% calculate the normalised temperature
dcom_zm_norm = dcom_an_zm / dsat_com_gm[:len(f_exp), None, None]
dexp_zm_norm = dexp_an_zm / dsat_exp_gm[:len(f_exp), None, None]

ddcom_zm = dcom_an_zm - dsat_com_zm[:len(f_exp), None, :]
ddexp_zm = dexp_an_zm - dsat_exp_zm[:len(f_exp), None, :]


#%% set start and end year
stayr = 20
endyr = 40


#%% average over years 20-40
exp_20_40_zm = np.mean(exp_an_zm[stayr:endyr, :, :], axis=0)
com_20_40_zm = np.mean(com_an_zm[stayr:endyr, :, :], axis=0)
dexp_20_40_zm = np.mean(dexp_an_zm[stayr:endyr, :, :], axis=0)
dcom_20_40_zm = np.mean(dcom_an_zm[stayr:endyr, :, :], axis=0)
dexp_20_40_zm_norm = np.mean(dexp_zm_norm[stayr:endyr, :, :], axis=0)
dcom_20_40_zm_norm = np.mean(dcom_zm_norm[stayr:endyr, :, :], axis=0)

ddexp_20_40_zm = np.mean(ddexp_zm[stayr:endyr, :, :], axis=0)
ddcom_20_40_zm = np.mean(ddcom_zm[stayr:endyr, :, :], axis=0)


dcom_nt_20_40_norm = np.mean(dcom_nt_norm[stayr:endyr, :], axis=0)
dcom_st_20_40_norm = np.mean(dcom_st_norm[stayr:endyr, :], axis=0)
dexp_nt_20_40_norm = np.mean(dexp_nt_norm[stayr:endyr, :], axis=0)
dexp_st_20_40_norm = np.mean(dexp_st_norm[stayr:endyr, :], axis=0)

dcom_nt_20_40 = np.mean(dcom_nt[stayr:endyr, :], axis=0)
dcom_st_20_40 = np.mean(dcom_st[stayr:endyr, :], axis=0)
dexp_nt_20_40 = np.mean(dexp_nt[stayr:endyr, :], axis=0)
dexp_st_20_40 = np.mean(dexp_st[stayr:endyr, :], axis=0)

ddcom_nt_20_40 = np.mean(ddcom_nt[stayr:endyr, :], axis=0)
ddcom_st_20_40 = np.mean(ddcom_st[stayr:endyr, :], axis=0)
ddexp_nt_20_40 = np.mean(ddexp_nt[stayr:endyr, :], axis=0)
ddexp_st_20_40 = np.mean(ddexp_st[stayr:endyr, :], axis=0)


#%% set the converted latitude
clat = np.sin(lat / 180 * np.pi)
xticks = np.array([-75, -50, -30, -15, 0, 15, 30, 50, 75])


#%% plot the year 20-40 averages
if var == "Q":
    pl_title = "Specific humidity"
    col_map = cm.RdBu
    levels = np.arange(-15, 16, 1) / 1000
    ticks = np.arange(-15, 16, 5) / 1000
elif var == "T":
    pl_title = "Temperature"
    col_map = cm.RdBu_r
    levels = np.arange(-1.5, 1.51, 0.1)
    ticks = np.arange(-1.5, 1.51, 0.5)
    dlevels = np.arange(-0.5, 0.51, 0.05)
    dticks = np.arange(-0.5, 0.51, 0.1)
# end if elif
    
x, y = np.meshgrid(clat, plev)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, dcom_20_40_zm_norm, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("no-dQ")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, dexp_20_40_zm_norm, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Normalised temperature in K K$^{-1}$")

ax2.axvline(x=0, c="gray", linewidth=0.5)
ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("dQ")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y, dexp_20_40_zm_norm - dcom_20_40_zm_norm, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in K K$^{-1}$")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("dQ minus no-dQ")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 50, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 50, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax3.text(np.sin(-55/180*np.pi), 50, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle("Normalised " + pl_title + f" anomaly averaged over years {stayr}$-${endyr} CESM2-SOM")

#pl.savefig(pl_path + "SpecHum_dQ_nodQ_yr20to40_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"{var}_VertCrossSec_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the deperature from the dSAT
pl_title = "Temperature"
col_map = cm.RdBu_r
levels = np.arange(-10, 10.1, 0.5)
ticks = np.arange(-10, 10.1, 2)
dlevels = np.arange(-4, 4.1, 0.2)
dticks = np.arange(-4, 4.1, 1)

x, y = np.meshgrid(clat, plev)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax4 = pl.subplot(gs[0, 14])
ax3 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, ddcom_20_40_zm, cmap=col_map, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("no-dQ")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, ddexp_20_40_zm, cmap=col_map, extend="both", levels=levels)

cb2 = fig.colorbar(p2, cax=ax4, ticks=ticks)
cb2.set_label("Normalised temperature in K K$^{-1}$")

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("dQ")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

p3 = ax3.contourf(x, y, ddexp_20_40_zm - ddcom_20_40_zm, cmap=col_map, extend="both", levels=dlevels)
cb3 = fig.colorbar(p3, cax=ax5, ticks=dticks)
cb3.set_label("Difference in K K$^{-1}$")

ax3.axvline(x=0, c="gray", linewidth=0.5)

ax3.set_xticks(np.sin(xticks / 180 * np.pi))
ax3.set_xticklabels(xticks)

ax3.set_title("dQ minus no-dQ")
ax3.set_xlabel("Latitude")
ax3.set_yticklabels([])
ax3.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 50, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 50, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax3.text(np.sin(-55/180*np.pi), 50, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(pl_title + f" anomaly departure averaged over years {stayr}$-${endyr} CESM2-SOM")

#pl.savefig(pl_path + "SpecHum_dQ_nodQ_yr20to40_Avg.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"{var}_Departure_VertCrossSec_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the lapse rate
yr = 0
"""
for yr in np.arange(40):
    fig = pl.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig, wspace=0.1)
    
    ax1 = pl.subplot(gs[0, 0])
    ax2 = pl.subplot(gs[0, 1])
    ax3 = pl.subplot(gs[0, 2])
    
    ax1.plot(dcom_nt_norm[yr, :], plev, c="blue", label="no-dQ")
    ax1.plot(dexp_nt_norm[yr, :], plev, c="red", label="dQ")
    ax2.plot(dcom_st_norm[yr, :], plev, c="blue")
    ax2.plot(dexp_st_norm[yr, :], plev, c="red")
    ax3.plot(dexp_nt_norm[yr, :] - dcom_nt_norm[yr, :], plev, c="black", label="NT")
    ax3.plot(dexp_st_norm[yr, :] - dcom_st_norm[yr, :], plev, c="gray", label="ST")
    ax1.axvline(x=1, c="black", linewidth=0.5)
    ax2.axvline(x=1, c="black", linewidth=0.5)
    ax3.axvline(x=0, c="black", linewidth=0.5)
    
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    
    ax1.legend(loc="lower right")
    ax3.legend(loc="lower left")
    
    ax1.set_xlim((0, 2.5))
    ax2.set_xlim((0, 2.5))
    ax3.set_xlim((-0.5, 0.5))
    
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    
    ax1.set_title("Northern Tropics")
    ax2.set_title("Southern Tropics")
    ax3.set_title("dQ minus no-dQ")
    
    pl.show()
    pl.close()
    
# end for yr    
"""    
    
#%% plot the year 20-40 mean
fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig, wspace=0.1, top=0.85)

ax1 = pl.subplot(gs[0, 0])
ax2 = pl.subplot(gs[0, 1])
ax3 = pl.subplot(gs[0, 2])

ax1.plot(dcom_nt_20_40_norm, plev, c="blue", label="no-dQ")
ax1.plot(dexp_nt_20_40_norm, plev, c="red", label="dQ")
ax2.plot(dcom_st_20_40_norm, plev, c="blue")
ax2.plot(dexp_st_20_40_norm, plev, c="red")
ax3.plot(dexp_nt_20_40_norm - dcom_nt_20_40_norm, plev, c="black", label="NT")
ax3.plot(dexp_st_20_40_norm - dcom_st_20_40_norm, plev, c="gray", label="ST")
ax1.axvline(x=1, c="black", linewidth=0.5)
ax2.axvline(x=1, c="black", linewidth=0.5)
ax3.axvline(x=0, c="black", linewidth=0.5)

ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()

ax1.axhline(y=1000, c="black", linewidth=0.5)
ax2.axhline(y=1000, c="black", linewidth=0.5)
ax3.axhline(y=1000, c="black", linewidth=0.5)

ax1.legend(loc="lower right")
ax3.legend(loc="lower left")

ax1.set_xlim((0, 2.1))
ax2.set_xlim((0, 2.1))
ax3.set_xlim((-0.5, 0.5))

ax2.set_yticklabels([])
ax3.set_yticklabels([])

ax1.set_xlabel("Normalised temperature in K K$^{-1}$")
ax2.set_xlabel("Normalised temperature in K K$^{-1}$")
ax3.set_xlabel("Normalised temperature in K K$^{-1}$")
ax1.set_ylabel("Pressure in hPa")

ax1.set_title("Northern Tropics")
ax2.set_title("Southern Tropics")
ax3.set_title("dQ minus no-dQ")

ax1.text(1.95, 0, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(1.95, 0, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax3.text(0.43, 0, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"Tropical normalised temperature anomaly CESM2-SOM years {stayr}$-${endyr}")

# pl.savefig(pl_path + f"{var}_NT_ST_NormLapseRate_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 20-40 mean
fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig, wspace=0.1, top=0.85)

ax1 = pl.subplot(gs[0, 0])
ax2 = pl.subplot(gs[0, 1])
ax3 = pl.subplot(gs[0, 2])

ax1.plot(ddcom_nt_20_40, plev, c="blue", label="no-dQ")
ax1.plot(ddexp_nt_20_40, plev, c="red", label="dQ")
ax2.plot(ddcom_st_20_40, plev, c="blue")
ax2.plot(ddexp_st_20_40, plev, c="red")
ax3.plot(ddexp_nt_20_40 - ddcom_nt_20_40, plev, c="black", label="R<30N")
ax3.plot(ddexp_st_20_40 - ddcom_st_20_40, plev, c="gray", label="R<30S")
ax1.axvline(x=0, c="black", linewidth=0.5)
ax2.axvline(x=0, c="black", linewidth=0.5)
ax3.axvline(x=0, c="black", linewidth=0.5)

ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()

ax1.axhline(y=1000, c="black", linewidth=0.5)
ax2.axhline(y=1000, c="black", linewidth=0.5)
ax3.axhline(y=1000, c="black", linewidth=0.5)

ax1.legend(loc="lower right")
ax3.legend(loc="center right")

ax1.set_xlim((-0.5, 9))
ax2.set_xlim((-0.5, 9))
# ax3.set_xlim((-0.5, 0.5))

ax2.set_yticklabels([])
ax3.set_yticklabels([])

ax1.set_xlabel("Departure in K")
ax2.set_xlabel("Departure in K")
ax3.set_xlabel("Difference in K")
ax1.set_ylabel("Pressure in hPa")

ax1.set_title("R<30N")
ax2.set_title("R<30S")
ax3.set_title("dQ minus no-dQ")

ax1.text(0.5, 0, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(0.5, 0, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax3.text(-1.2, 0, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"Tropical temperature anomaly departure CESM2-SOM years {stayr}$-${endyr}")

# pl.savefig(pl_path + f"{var}_NT_ST_DepartureFromdSAT_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 20-40 mean
fig = pl.figure(figsize=(8, 6))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, wspace=0.1, hspace=0.25, top=0.9)

ax1 = pl.subplot(gs[0, 0])
ax2 = pl.subplot(gs[0, 1])
ax3 = pl.subplot(gs[1, 0])
ax4 = pl.subplot(gs[1, 1])

ax1.plot(dcom_nt_20_40, plev, c="blue", label="no-dQ")
ax1.plot(dexp_nt_20_40, plev, c="red", label="dQ")
ax2.plot(dcom_st_20_40, plev, c="blue")
ax2.plot(dexp_st_20_40, plev, c="red")
ax3.plot(dexp_nt_20_40 - dcom_nt_20_40, plev, c="black", label="NT")
ax3.plot(dexp_st_20_40 - dcom_st_20_40, plev, c="gray", label="ST")
ax3.axvline(x=0, c="black", linewidth=0.5)
ax4.plot(dexp_nt_20_40 - dexp_st_20_40, plev, c="red", label="dQ")
ax4.plot(dcom_nt_20_40 - dcom_st_20_40, plev, c="blue", label="no-dQ")
ax4.axvline(x=0, c="black", linewidth=0.5)

ax1.axhline(y=1000, c="black", linewidth=0.5)
ax2.axhline(y=1000, c="black", linewidth=0.5)
ax3.axhline(y=1000, c="black", linewidth=0.5)
ax4.axhline(y=1000, c="black", linewidth=0.5)

ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()

ax1.legend(loc="lower right")
ax3.legend(loc="lower right")

ax1.set_xlim((5, 18))
ax2.set_xlim((5, 18))
# ax3.set_xlim((-0.5, 0.5))

ax2.set_yticklabels([])
ax4.set_yticklabels([])

# ax1.set_xlabel("Normalised temperature in K K$^{-1}$")
# ax2.set_xlabel("Normalised temperature in K K$^{-1}$")
ax3.set_xlabel("Temperature in K")
ax4.set_xlabel("Temperature in K")
ax1.set_ylabel("Pressure in hPa")
ax3.set_ylabel("Pressure in hPa")

ax1.set_title("Northern Tropics")
ax2.set_title("Southern Tropics")
ax3.set_title("dQ minus no-dQ")
ax4.set_title("NT minus ST")

fig.suptitle(f"Tropical temperature anomaly CESM2-SOM year {stayr}$-${endyr}")

# pl.savefig(pl_path + f"{var}_NT_ST_LapseRate_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
    

#%% plot NT and ST TREFHT
fig = pl.figure(figsize=(8, 3))
gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, wspace=0.1, top=0.8)

ax1 = pl.subplot(gs[0, 0])
ax2 = pl.subplot(gs[0, 1])

ax1.plot(dsat_com_nt, c="blue", label="no-dQ")
ax1.plot(dsat_exp_nt, c="red", label="dQ")
ax1.plot(dsat_com_nt - dsat_exp_nt, c="black", label="no-dQ minus dQ")
ax1.axhline(y=0, c="black", linewidth=0.5)


ax2.plot(dsat_com_st, c="blue", label="no-dQ")
ax2.plot(dsat_exp_st, c="red", label="dQ")
ax2.plot(dsat_com_st - dsat_exp_st, c="black", label="no-dQ minus dQ")
ax2.axhline(y=0, c="black", linewidth=0.5)
ax2.legend()    

ax1.set_title("Northern Tropics")
ax2.set_title("Southern Tropics")

ax1.set_ylim((-1, 11))
ax2.set_ylim((-1, 11))

ax2.set_yticklabels([])

ax1.set_xlabel("Years of simulation")
ax2.set_xlabel("Years of simulation")

ax1.set_ylabel("$\Delta$SAT in K")

fig.suptitle(f"Temperature at reference height anomaly CESM2-SOM years {stayr}$-${endyr}")

# pl.savefig(pl_path + f"dSAT_NT_ST_dQ_nodQ_yr{stayr}to{endyr}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
        
    
