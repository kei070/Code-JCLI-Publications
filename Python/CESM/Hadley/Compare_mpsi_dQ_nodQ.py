"""
Compare mpsi for different cases.

Generates Fig. 11 and S11 in Eiselt and Graversen (2023), JCLI.

Be sure to set the paths in code block "set up paths" correctly.
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
import xarray as xr
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
case1 = "dQ01yr_4xCO2"
case2 = "Proj2_KUE_4xCO2"
case3 = "Proj2_KUE"


#%% set up paths
data_path1 = f"/Data/{case1}/mpsi_Files/"
data_path2 = f"/Data/{case2}/mpsi_Files/"
data_path3 = f"/Data/{case3}/mpsi_Files/"

pl_path = ""

os.makedirs(pl_path, exist_ok=True)


#%% list and sort the files
f_list1 = sorted(glob.glob(data_path1 + "/mpsi_*.nc"), key=str.casefold)
f_list2 = sorted(glob.glob(data_path2 + "/mpsi_*.nc"), key=str.casefold)
f_list3 = sorted(glob.glob(data_path3 + "/mpsi_*.nc"), key=str.casefold)


#%% load the datset
mpsi_nc1 = Dataset(f_list1[0])
mpsi_nc2 = Dataset(f_list2[0])
mpsi_nc3 = Dataset(f_list3[0])


#%% get the streamfunction values
mpsi_mon1 = da.ma.masked_array(mpsi_nc1.variables["mpsi"], lock=True)
if len(f_list1) > 1:
    for i in np.arange(1, len(f_list1)):
        dataset = Dataset(f_list1[i])
        mpsi_mon1 = da.concatenate([mpsi_mon1, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if

mpsi_mon2 = da.ma.masked_array(mpsi_nc2.variables["mpsi"], lock=True)
if len(f_list2) > 1:
    for i in np.arange(1, len(f_list2)):
        dataset = Dataset(f_list2[i])
        mpsi_mon2 = da.concatenate([mpsi_mon2, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if

mpsi_mon3 = da.ma.masked_array(mpsi_nc3.variables["mpsi"], lock=True)
if len(f_list3) > 1:
    for i in np.arange(1, len(f_list3)):
        dataset = Dataset(f_list3[i])
        mpsi_mon3 = da.concatenate([mpsi_mon3, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% aggregate to annual means
mpsi_mon1 = mpsi_mon1.compute()
mpsi1 = an_mean(mpsi_mon1)
mpsi_mon2 = mpsi_mon2.compute()
mpsi2 = an_mean(mpsi_mon2)
mpsi_mon3 = mpsi_mon3.compute()
mpsi3 = an_mean(mpsi_mon3)


#%% set start and end year
stayr = 20
endyr = 40


#%% average the mpsi for years 20-40
mpsi1_20_40 = np.mean(mpsi1[stayr:endyr, :, :], axis=0)
mpsi2_20_40 = np.mean(mpsi2[stayr:endyr, :, :], axis=0)

mpsi3_mean = np.mean(mpsi3[:, :, :], axis=0)

dmpsi1_20_40 = mpsi1_20_40 - mpsi3_mean
dmpsi2_20_40 = mpsi2_20_40 - mpsi3_mean


#%% get lat and plev
lat =  mpsi_nc1.variables["lat"][:]
plev = mpsi_nc1.variables["plev"][:] / 100


#%% months list
months_l = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


#%% set the converted latitude
clat = np.sin(lat / 180 * np.pi)
xticks = np.array([-75, -50, -30, -15, 0, 15, 30, 50, 75])


#%% test plot - monthly
"""
for nti in np.arange(np.shape(mpsi_mon)[0]):
    
    month_l_i = nti % 12
    
    levels =[-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
    x, y = np.meshgrid(lat, plev)
    
    fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))
    
    p1 = axes.contourf(x, y, mpsi_mon[nti, :, :] / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)
    cb1 = fig.colorbar(p1, ax=axes)
    cb1.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")
    
    axes.axvline(x=0, c="gray", linewidth=0.5)
    
    axes.set_title(f"MPSI CESM2-SOM {case} month {nti} ({months_l[month_l_i]})")
    axes.set_xlabel("Latitude")
    axes.set_ylabel("Pressure in hPa")
    
    axes.invert_yaxis()
    
    pl.show()
    pl.close()
# end for nti   
"""    
    
#%% test plot - annual

for nti in np.arange(np.shape(mpsi1)[0]):
# for nti in [0]:    
    levels = np.arange(-7.5, 7.6, 0.5)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
    x, y = np.meshgrid(lat, plev)
    
    fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3), sharey=True)
    
    p1 = axes[0].contourf(x, y, mpsi1[nti, :, :] / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)
    cb1 = fig.colorbar(p1, ax=axes[0])
    cb1.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")
    
    axes[0].axvline(x=0, c="gray", linewidth=0.5)
    
    axes[0].set_title(f"MPSI CESM2-SOM {case1} year {nti}")
    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Pressure in hPa")

    p2 = axes[1].contourf(x, y, mpsi2[nti, :, :] / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)
    cb2 = fig.colorbar(p2, ax=axes[1])
    cb2.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")
    
    axes[1].axvline(x=0, c="gray", linewidth=0.5)
    axes[1].set_title(f"MPSI CESM2-SOM {case2} year {nti}")
    axes[1].set_xlabel("Latitude")
    axes[1].invert_yaxis()
    
    pl.show()
    pl.close()
# end for yr    

    
#%% difference plot    
# for nti in np.arange(np.shape(mpsi1)[0]):
for nti in [0]:    
    levels = np.arange(-7.5, 7.6, 0.5)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
    x, y = np.meshgrid(lat, plev)
    
    fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))
    
    p1 = axes.contourf(x, y, (mpsi1[nti, :, :] - mpsi2[nti, :, :]) / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)
    # p1 = axes.pcolormesh(x, y, mpsi[nti, :, :] / 1e10, cmap=cm.RdBu_r, vmin=-7.5, vmax=7.5)
    cb1 = fig.colorbar(p1, ax=axes)
    cb1.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")
    
    axes.axvline(x=0, c="gray", linewidth=0.5)
    
    axes.set_title(f"MPSI CESM2-SOM dQ minus no-dQ year {nti}")
    axes.set_xlabel("Latitude")
    axes.set_ylabel("Pressure in hPa")
    
    axes.invert_yaxis()
    
    pl.show()
    pl.close()
# end for yr 


#%% plot the year 20-40 averages
levels = np.arange(-10, 11, 0.5)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
ticks = np.arange(-10, 11, 2)
x, y = np.meshgrid(clat, plev)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.5, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax3 = pl.subplot(gs[0, 14])
ax4 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, mpsi2_20_40 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("no-dQ")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, mpsi1_20_40 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("dQ")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

cb2 = fig.colorbar(p2, cax=ax3, ticks=ticks)
cb2.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")


p4 = ax4.contourf(x, y, (mpsi1_20_40 - mpsi2_20_40) / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)
cb4 = fig.colorbar(p4, cax=ax5, ticks=ticks)
cb4.set_label("Difference in 10$^{10}$ kg s$^{-1}$")

ax4.axvline(x=0, c="gray", linewidth=0.5)

ax4.set_xticks(np.sin(xticks / 180 * np.pi))
ax4.set_xticklabels(xticks)

ax4.set_title("dQ minus no-dQ")
ax4.set_xlabel("Latitude")
ax4.set_yticklabels([])
ax4.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 65, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 65, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax4.text(np.sin(-55/180*np.pi), 65, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"Meridional overturning streamfunction CESM2-SOM years {stayr}$-${endyr}")

pl.savefig(pl_path + f"MPSI_dQ_nodQ_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"MPSI_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the year 20-40 averages minus control
levels = np.arange(-5.0, 5.1, 0.25)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
ticks = np.arange(-5, 5.1, 1)
dlevels = np.arange(-10, 11, 0.5)  # [-7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5]
dticks = np.arange(-10, 11, 2)
x, y = np.meshgrid(clat, plev)

# fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(12, 3), sharey=True)

fig = pl.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=25, wspace=0.55, top=0.85) 

ax1 = pl.subplot(gs[0, :7])
ax2 = pl.subplot(gs[0, 7:14])
ax3 = pl.subplot(gs[0, 14])
ax4 = pl.subplot(gs[0, 17:24])
ax5 = pl.subplot(gs[0, 24])

p1 = ax1.contourf(x, y, dmpsi2_20_40 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax1.axvline(x=0, c="gray", linewidth=0.5)

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

ax1.set_title("no-dQ")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Pressure in hPa")
ax1.invert_yaxis()

p2 = ax2.contourf(x, y, dmpsi1_20_40 / 1e10, cmap=cm.RdBu_r, extend="both", levels=levels)

ax2.axvline(x=0, c="gray", linewidth=0.5)

ax2.set_xticks(np.sin(xticks / 180 * np.pi))
ax2.set_xticklabels(xticks)

ax2.set_title("dQ")
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_yticklabels([])

cb2 = fig.colorbar(p2, cax=ax3, ticks=ticks)
cb2.set_label("Change of MPSI in 10$^{10}$ kg s$^{-1}$")


p4 = ax4.contourf(x, y, (dmpsi1_20_40 - dmpsi2_20_40) / 1e10, cmap=cm.RdBu_r, extend="both", levels=dlevels)
cb4 = fig.colorbar(p4, cax=ax5, ticks=dticks)
cb4.set_label("Difference in 10$^{10}$ kg s$^{-1}$")

ax4.axvline(x=0, c="gray", linewidth=0.5)

ax4.set_xticks(np.sin(xticks / 180 * np.pi))
ax4.set_xticklabels(xticks)

ax4.set_title("dQ minus no-dQ")
ax4.set_xlabel("Latitude")
ax4.set_yticklabels([])
ax4.invert_yaxis()

ax1.text(np.sin(-55/180*np.pi), 65, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax2.text(np.sin(-55/180*np.pi), 65, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax4.text(np.sin(-55/180*np.pi), 65, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"Change of meridional overturning streamfunction CESM2-SOM years {stayr}$-${endyr}")

pl.savefig(pl_path + f"dMPSI_dQ_nodQ_yr{stayr}to{endyr}_Avg.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"dMPSI_dQ_nodQ_yr{stayr}to{endyr}_Avg.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
    
