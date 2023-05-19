"""
Plot function for comparing APRP indices.
"""

#%% imports
import os
import sys
import copy
import time
import numpy as np
import pylab as pl
import matplotlib.colors as colors
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
import progressbar as pg
import dask.array as da
import time as ti

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Set_ColorLevels import set_levs_norm_colmap


def pl_maps(a1, a2, lat, lon, title1, title2, cb_lab1, cb_lab2=None, l_num=14, level_a=1, factor=1):

    # if there is no second colorbar label given set it to the value of cb_lab1
    if cb_lab2 is None:
        cb_lab2 = cb_lab1
    # end if
    
    # handle colormap labels
    if level_a == 1:  # use colormap levels of array 1
        levels, norm, cmp = set_levs_norm_colmap(a1, l_num, factor=factor)
        levels2, norm2, cmp2 = levels, norm, cmp
    elif level_a == 2:  # use colormap levels of array 2
        levels, norm, cmp = set_levs_norm_colmap(a2, l_num, factor=factor)
        levels2, norm2, cmp2 = levels, norm, cmp
    elif level_a == 3:  # use different colormap levels of array 1 and 2
        levels, norm, cmp = set_levs_norm_colmap(a1, l_num, factor=factor)
        levels2, norm2, cmp2 = set_levs_norm_colmap(a2, l_num, factor=factor)
    # end if elif
    
    x, y = np.meshgrid(lon, lat)
    proj = ccrs.PlateCarree(central_longitude=0)
    
    # surface albedo kernel
    fig, (ax1, ax2) = pl.subplots(ncols=1, nrows=2, figsize=(18, 14), subplot_kw=dict(projection=proj))
    
    # colorbar padding
    padding = 0.05
    
    p1 = ax1.contourf(x, y, a1, transform=ccrs.PlateCarree(), cmap=cmp, norm=norm, levels=levels, 
                      extend="both")
    ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    cb1 = pl.colorbar(p1, ax=ax1, ticks=levels, pad=padding)
    cb1.set_label(cb_lab1)
    ax1.set_title(title1)
    
    p2 = ax2.contourf(x, y, a2, transform=ccrs.PlateCarree(), cmap=cmp2, norm=norm2, levels=levels2,
                      extend="both")
    ax2.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    cb2 = pl.colorbar(p2, ax=ax2, ticks=levels2, pad=padding)
    cb2.set_label(cb_lab2)
    ax2.set_title(title2)
    
    pl.show()
    pl.close()

# end function pl_maps
