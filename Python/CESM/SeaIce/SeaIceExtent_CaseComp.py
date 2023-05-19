"""
Compare sea-ice extent (spatial data; maps) for two difference CESM2-SOM cases. 

Be sure to adjust the paths in code block "set paths".
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


#%% case
case1 = "Proj2_KUE_4xCO2"
case2 = "dQ01yr_4xCO2"


#%% set the number of years
nyrs = 25


#%% set paths
data_path1 = f"/Data/{case1}/"
data_path2 = f"/Data/{case2}/"

pl_path = (f"/Plots/Case_Comparison/{case2}_{case1}/SeaIce/")
# os.makedirs(pl_path, exist_ok=True)


#%% load a file
nc1 = Dataset(sorted(glob.glob(data_path1 + "SeaIce_Fraction_n80_" + case1 + "*.nc"), key=str.casefold)[0])
nc2 = Dataset(sorted(glob.glob(data_path2 + "SeaIce_Fraction_n80_" + case2 + "*.nc"), key=str.casefold)[0])


#%% load values
aice1 = an_mean(nc1.variables["aice"][:nyrs*12, :, :])
aice2 = an_mean(nc2.variables["aice"][:nyrs*12, :, :])

lat = nc1.variables["lat"][:]
lon = nc1.variables["lon"][:]


#%% where sea-ice concentration is zero set cell to NaN
aice1[aice1 == 0] = np.nan
aice2[aice2 == 0] = np.nan


#%% calculate the sea-ice extent
si_ext1 = np.zeros(np.shape(aice1))
si_ext2 = np.zeros(np.shape(aice2))

si_ext1[aice1 >= 0.15] = 1
si_ext2[aice2 >= 0.15] = 1
si_ext1[si_ext1 == 0] = np.nan
si_ext2[si_ext2 == 0] = np.nan


#%% plot the sea-ice concentration
"""
yr = 0
x, y = np.meshgrid(lon, lat)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=90)
    
for yr in np.arange(15):
    
    
    fig = pl.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=25, figure=fig)
    ax0 = pl.subplot(gs[0, :12], projection=proj)
    ax1 = pl.subplot(gs[0, 12:24], projection=proj)
    ax2 = pl.subplot(gs[0, 24])
    
    p0 = ax0.pcolormesh(x, y, aice1[yr, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax0.stock_img()
    ax0.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax0.set_title(case1)
    
    p1 = ax1.pcolormesh(x, y, aice2[yr, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax1.stock_img()
    ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax1.set_title(case2)
    
    cb0 = fig.colorbar(p0, cax=ax2)
    cb0.set_label("Sea-ice concentration")
    
    fig.suptitle(f"Year {yr}")
    
    pl.show()
    pl.close()
# for yr    
"""

#%% plot the sea-ice extent
"""
yr = 0
x, y = np.meshgrid(lon, lat)
# proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
proj = ccrs.Mollweide(central_longitude=0)

for yr in np.arange(nyrs):
    
    
    fig = pl.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=24, figure=fig)
    ax0 = pl.subplot(gs[0, :12], projection=proj)
    ax1 = pl.subplot(gs[0, 12:24], projection=proj)
    # ax2 = pl.subplot(gs[0, 24])
    
    p0 = ax0.pcolormesh(x, y, si_ext1[yr, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax0.stock_img()
    ax0.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax0.set_title(case1)
    
    p1 = ax1.pcolormesh(x, y, si_ext2[yr, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax1.stock_img()
    ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax1.set_title(case2)
    
    fig.suptitle(f"Year {yr}")
    
    pl.show()
    pl.close()
# for yr
"""

#%% plot the sea-ice extent
x, y = np.meshgrid(lon, lat)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)

# fig, axes = pl.subplots(ncols=3, nrows=2, figsize=(13, 7), subplot_kw={'projection':proj})

fig = pl.figure(figsize=(14, 7))
gs = gridspec.GridSpec(nrows=2, ncols=4, wspace=0.05, hspace=0.3) 

# upper panel -----------------------------------------------------------------------------------------------------------    
ax00 = pl.subplot(gs[0, 0], projection=proj)
ax01 = pl.subplot(gs[0, 1], projection=proj)
ax02 = pl.subplot(gs[0, 2], projection=proj)
ax03 = pl.subplot(gs[0, 3], projection=proj)
ax10 = pl.subplot(gs[1, 0], projection=proj)
ax11 = pl.subplot(gs[1, 1], projection=proj)
ax12 = pl.subplot(gs[1, 2], projection=proj)
ax13 = pl.subplot(gs[1, 3], projection=proj)

ax00.pcolormesh(x, y, si_ext2[2, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax00.stock_img()
ax00.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax00.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax00.set_title("no-dQ $-$ year 3")

ax01.pcolormesh(x, y, si_ext2[5, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax01.stock_img()
ax01.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax01.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax01.set_title("no-dQ $-$ year 6")

ax02.pcolormesh(x, y, si_ext2[8, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax02.stock_img()
ax02.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax02.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax02.set_title("no-dQ $-$ year 9")

ax03.pcolormesh(x, y, si_ext2[11, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax03.stock_img()
ax03.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax03.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax03.set_title("no-dQ $-$ year 12")

ax10.pcolormesh(x, y, si_ext1[2, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax10.stock_img()
ax10.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax10.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax10.set_title("dQ $-$ year 3")

ax11.pcolormesh(x, y, si_ext1[5, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax11.stock_img()
ax11.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax11.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax11.set_title("dQ $-$ year 6")

ax12.pcolormesh(x, y, si_ext1[8, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax12.stock_img()
ax12.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax12.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax12.set_title("dQ $-$ year 9")

ax13.pcolormesh(x, y, si_ext1[11, :, :], cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax13.stock_img()
ax13.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax13.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax13.set_title("dQ $-$ year 12")

pl.savefig(pl_path + "SeaIce_Extent_MultTimeSlice_CESM2-SOM_CaseComp.pdf", bbox_inches="tight", dpi=25)

pl.show()
pl.close()
