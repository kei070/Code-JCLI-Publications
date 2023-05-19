"""
Sea-ice extent in abrupt-4xCO2 in two model groups.

Generates Fig. S7 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir5 and data_dir6 as well as pl_path.
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
from scipy import interpolate
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)
from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set sea-ice extent threshold
thr = 15


#%% set model groups
models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]
mod_dic = {"G1":models1, "G2":models2}


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""
data_dir6 = ""

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""
# os.makedirs(pl_path + "/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% loop over group one
dtas_dic = {}
sic_e_dic = {}
sic_m_dic = {}
sic_m2_dic = {}
sic_l_dic = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    # dtas = np.zeros((150, len(group)))
    sic_e = []
    sic_m = []
    sic_m2 = []
    sic_l = []
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
            var = "sic"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
            var = "siconc"
        # end if
        
        # data path and name
        data_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        # f_name_dtas = "dtas_Amon_*.nc"
        f_name_sic = f"{var}_*_{a4x}_*_n80_*.nc"
        # f_name_damoc = f"damoc_and_piControl_21YrRun_{mod_d}.nc"
        
        # load the amoc file
        # dtas_nc = Dataset(glob.glob(data_path + f_name_dtas)[0])
        sic_nc = Dataset(glob.glob(data_path + f_name_sic)[0])
        
        # lat = sic_nc.variables["lat"][:]
        # lon = sic_nc.variables["lon"][:]
        
        # load the data    
        # dtas[:, i] = an_mean(glob_mean(dtas_nc.variables["tas_ch"][:150*12, :, :], lat, lon))
        sic_e.append(sic_nc.variables[var][:5*12, :, :])
        sic_m.append(sic_nc.variables[var][18*12:22*12, :, :])
        sic_m2.append(sic_nc.variables[var][45*12:55*12, :, :])
        sic_l.append(sic_nc.variables[var][-10*12:, :, :])
        # damoc = damoc_nc.variables["damoc"][:]
        
    # end for i, mod_d
    
    # dtas_dic[g_n] = dtas
    sic_e_dic[g_n] = np.stack(sic_e, axis=0)
    sic_m_dic[g_n] = np.stack(sic_m, axis=0)
    sic_m2_dic[g_n] = np.stack(sic_m2, axis=0)
    sic_l_dic[g_n] = np.stack(sic_l, axis=0)

# end for g_n, group


#%% load lats and lons
lat = sic_nc.variables["lat"][:]
lon = sic_nc.variables["lon"][:]


#%% where there are large (masked) values insert nan
sic_e_dic["G1"][sic_e_dic["G1"] > 100] = np.nan
sic_e_dic["G2"][sic_e_dic["G2"] > 100] = np.nan
sic_m_dic["G1"][sic_m_dic["G1"] > 100] = np.nan
sic_m_dic["G2"][sic_m_dic["G2"] > 100] = np.nan
sic_m2_dic["G1"][sic_m2_dic["G1"] > 100] = np.nan
sic_m2_dic["G2"][sic_m2_dic["G2"] > 100] = np.nan
sic_l_dic["G1"][sic_l_dic["G1"] > 100] = np.nan
sic_l_dic["G2"][sic_l_dic["G2"] > 100] = np.nan


#%% calculate the group means
# dtas_grm_dic = {"G1":np.mean(dtas_dic["G1"], axis=-1), "G2":np.mean(dtas_dic["G2"], axis=-1)}
sic_e_g1 = np.mean(sic_e_dic["G1"], axis=0)
sic_e_g2 = np.mean(sic_e_dic["G2"], axis=0)
sic_m_g1 = np.mean(sic_m_dic["G1"], axis=0)
sic_m_g2 = np.mean(sic_m_dic["G2"], axis=0)
sic_m2_g1 = np.mean(sic_m2_dic["G1"], axis=0)
sic_m2_g2 = np.mean(sic_m2_dic["G2"], axis=0)
sic_l_g1 = np.mean(sic_l_dic["G1"], axis=0)
sic_l_g2 = np.mean(sic_l_dic["G2"], axis=0)


#%% calculate the piC mean
sic_e_g1_pm = np.mean(sic_e_g1, axis=0)
sic_e_g2_pm = np.mean(sic_e_g2, axis=0)
sic_m_g1_pm = np.mean(sic_m_g1, axis=0)
sic_m_g2_pm = np.mean(sic_m_g2, axis=0)
sic_m2_g1_pm = np.mean(sic_m2_g1, axis=0)
sic_m2_g2_pm = np.mean(sic_m2_g2, axis=0)
sic_l_g1_pm = np.mean(sic_l_g1, axis=0)
sic_l_g2_pm = np.mean(sic_l_g2, axis=0)


#%% calculate the sea-ice extent (>15% siconc)
si_ext_e_g1_pm = np.zeros(np.shape(sic_e_g1_pm))
si_ext_e_g2_pm = np.zeros(np.shape(sic_e_g2_pm))
si_ext_m_g1_pm = np.zeros(np.shape(sic_m_g1_pm))
si_ext_m_g2_pm = np.zeros(np.shape(sic_m_g2_pm))
si_ext_m2_g1_pm = np.zeros(np.shape(sic_m2_g1_pm))
si_ext_m2_g2_pm = np.zeros(np.shape(sic_m2_g2_pm))
si_ext_l_g1_pm = np.zeros(np.shape(sic_l_g1_pm))
si_ext_l_g2_pm = np.zeros(np.shape(sic_l_g2_pm))

si_ext_e_g1_pm[sic_e_g1_pm > thr] = 1
si_ext_e_g2_pm[sic_e_g2_pm > thr] = 1
si_ext_m_g1_pm[sic_m_g1_pm > thr] = 1
si_ext_m_g2_pm[sic_m_g2_pm > thr] = 1
si_ext_m2_g1_pm[sic_m2_g1_pm > thr] = 1
si_ext_m2_g2_pm[sic_m2_g2_pm > thr] = 1
si_ext_l_g1_pm[sic_l_g1_pm > thr] = 1
si_ext_l_g2_pm[sic_l_g2_pm > thr] = 1


#%% caluclate the difference between G1 and G2 sea-ice extent (!)
si_ext_e_d_pm = si_ext_e_g2_pm - si_ext_e_g1_pm
si_ext_m_d_pm = si_ext_m_g2_pm - si_ext_m_g1_pm
si_ext_m2_d_pm = si_ext_m2_g2_pm - si_ext_m2_g1_pm
si_ext_l_d_pm = si_ext_l_g2_pm - si_ext_l_g1_pm

si_ext_e_d_pm[si_ext_e_d_pm == 0] = np.nan
si_ext_m_d_pm[si_ext_m_d_pm == 0] = np.nan
si_ext_m2_d_pm[si_ext_m2_d_pm == 0] = np.nan
si_ext_l_d_pm[si_ext_l_d_pm == 0] = np.nan


#%% set the cells without sea ice to NaN that they are transparent in the plot
si_ext_e_g1_pm[si_ext_e_g1_pm != 1] = np.nan
si_ext_e_g2_pm[si_ext_e_g2_pm != 1] = np.nan
si_ext_m_g1_pm[si_ext_m_g1_pm != 1] = np.nan
si_ext_m_g2_pm[si_ext_m_g2_pm != 1] = np.nan
si_ext_m2_g1_pm[si_ext_m2_g1_pm != 1] = np.nan
si_ext_m2_g2_pm[si_ext_m2_g2_pm != 1] = np.nan
si_ext_l_g1_pm[si_ext_l_g1_pm != 1] = np.nan
si_ext_l_g2_pm[si_ext_l_g2_pm != 1] = np.nan


#%% generate the sea-ice extent for all 150 years
si_ext_e_g1 = np.zeros(np.shape(sic_e_g1))
si_ext_e_g2 = np.zeros(np.shape(sic_e_g2))
si_ext_m_g1 = np.zeros(np.shape(sic_m_g1))
si_ext_m_g2 = np.zeros(np.shape(sic_m_g2))
si_ext_m2_g1 = np.zeros(np.shape(sic_m2_g1))
si_ext_m2_g2 = np.zeros(np.shape(sic_m2_g2))
si_ext_l_g1 = np.zeros(np.shape(sic_l_g1))
si_ext_l_g2 = np.zeros(np.shape(sic_l_g2))

si_ext_e_g1[sic_e_g1 > thr] = 1
si_ext_e_g2[sic_e_g2 > thr] = 1
si_ext_m_g1[sic_m_g1 > thr] = 1
si_ext_m_g2[sic_m_g2 > thr] = 1
si_ext_m2_g1[sic_m2_g1 > thr] = 1
si_ext_m2_g2[sic_m2_g2 > thr] = 1
si_ext_l_g1[sic_l_g1 > thr] = 1
si_ext_l_g2[sic_l_g2 > thr] = 1

si_ext_e_g1[si_ext_e_g1 != 1] = np.nan
si_ext_e_g2[si_ext_e_g2 != 1] = np.nan
si_ext_m_g1[si_ext_m_g1 != 1] = np.nan
si_ext_m_g2[si_ext_m_g2 != 1] = np.nan
si_ext_m2_g1[si_ext_m2_g1 != 1] = np.nan
si_ext_m2_g2[si_ext_m2_g2 != 1] = np.nan
si_ext_l_g1[si_ext_l_g1 != 1] = np.nan
si_ext_l_g2[si_ext_l_g2 != 1] = np.nan


#%% where sic == 0 set to nan
sic_e_g1[sic_e_g1 == 0] = np.nan
sic_e_g2[sic_e_g2 == 0] = np.nan
sic_m_g1[sic_m_g1 == 0] = np.nan
sic_m_g2[sic_m_g2 == 0] = np.nan
sic_m2_g1[sic_m2_g1 == 0] = np.nan
sic_m2_g2[sic_m2_g2 == 0] = np.nan
sic_l_g1[sic_l_g1 == 0] = np.nan
sic_l_g2[sic_l_g2 == 0] = np.nan

sic_e_g1_pm[sic_e_g1_pm == 0] = np.nan
sic_e_g2_pm[sic_e_g2_pm == 0] = np.nan
sic_m_g1_pm[sic_m_g1_pm == 0] = np.nan
sic_m_g2_pm[sic_m_g2_pm == 0] = np.nan
sic_m2_g1_pm[sic_m2_g1_pm == 0] = np.nan
sic_m2_g2_pm[sic_m2_g2_pm == 0] = np.nan
sic_l_g1_pm[sic_l_g1_pm == 0] = np.nan
sic_l_g2_pm[sic_l_g2_pm == 0] = np.nan


#%% calculate the individual member piControl mean (from monthly data over the first 10 years)
sic_e_means = {"G1":np.mean(sic_e_dic["G1"], axis=1), "G2":np.mean(sic_e_dic["G2"], axis=1)}
sic_e_means["G1"][sic_e_means["G1"] == 0] = np.nan
sic_e_means["G2"][sic_e_means["G2"] == 0] = np.nan
sic_m_means = {"G1":np.mean(sic_m_dic["G1"], axis=1), "G2":np.mean(sic_m_dic["G2"], axis=1)}
sic_m_means["G1"][sic_m_means["G1"] == 0] = np.nan
sic_m_means["G2"][sic_m_means["G2"] == 0] = np.nan
sic_m2_means = {"G1":np.mean(sic_m2_dic["G1"], axis=1), "G2":np.mean(sic_m2_dic["G2"], axis=1)}
sic_m2_means["G1"][sic_m2_means["G1"] == 0] = np.nan
sic_m2_means["G2"][sic_m2_means["G2"] == 0] = np.nan
sic_l_means = {"G1":np.mean(sic_l_dic["G1"], axis=1), "G2":np.mean(sic_l_dic["G2"], axis=1)}
sic_l_means["G1"][sic_l_means["G1"] == 0] = np.nan
sic_l_means["G2"][sic_l_means["G2"] == 0] = np.nan

si_ext_e_means = {"G1":np.zeros(np.shape(sic_e_means["G1"])), "G2":np.zeros(np.shape(sic_e_means["G2"]))}
si_ext_e_means["G1"][sic_e_means["G1"] > 15] = 1
si_ext_e_means["G2"][sic_e_means["G2"] > 15] = 1
si_ext_e_means["G1"][si_ext_e_means["G1"] == 0] = np.nan
si_ext_e_means["G2"][si_ext_e_means["G2"] == 0] = np.nan
si_ext_m_means = {"G1":np.zeros(np.shape(sic_m_means["G1"])), "G2":np.zeros(np.shape(sic_m_means["G2"]))}
si_ext_m_means["G1"][sic_m_means["G1"] > 15] = 1
si_ext_m_means["G2"][sic_m_means["G2"] > 15] = 1
si_ext_m_means["G1"][si_ext_m_means["G1"] == 0] = np.nan
si_ext_m_means["G2"][si_ext_m_means["G2"] == 0] = np.nan
si_ext_m2_means = {"G1":np.zeros(np.shape(sic_m2_means["G1"])), "G2":np.zeros(np.shape(sic_m2_means["G2"]))}
si_ext_m2_means["G1"][sic_m2_means["G1"] > 15] = 1
si_ext_m2_means["G2"][sic_m2_means["G2"] > 15] = 1
si_ext_m2_means["G1"][si_ext_m2_means["G1"] == 0] = np.nan
si_ext_m2_means["G2"][si_ext_m2_means["G2"] == 0] = np.nan
si_ext_l_means = {"G1":np.zeros(np.shape(sic_l_means["G1"])), "G2":np.zeros(np.shape(sic_l_means["G2"]))}
si_ext_l_means["G1"][sic_l_means["G1"] > 15] = 1
si_ext_l_means["G2"][sic_l_means["G2"] > 15] = 1
si_ext_l_means["G1"][si_ext_l_means["G1"] == 0] = np.nan
si_ext_l_means["G2"][si_ext_l_means["G2"] == 0] = np.nan


#%% cyclic point
si_ext_e_g1_pm_cy, lon_cy = cu.add_cyclic_point(si_ext_e_g1_pm, lon)
si_ext_e_g2_pm_cy = cu.add_cyclic_point(si_ext_e_g2_pm)
si_ext_m_g1_pm_cy, lon_cy = cu.add_cyclic_point(si_ext_m_g1_pm, lon)
si_ext_m_g2_pm_cy = cu.add_cyclic_point(si_ext_m_g2_pm)
si_ext_m2_g1_pm_cy, lon_cy = cu.add_cyclic_point(si_ext_m2_g1_pm, lon)
si_ext_m2_g2_pm_cy = cu.add_cyclic_point(si_ext_m2_g2_pm)
si_ext_l_g1_pm_cy, lon_cy = cu.add_cyclic_point(si_ext_l_g1_pm, lon)
si_ext_l_g2_pm_cy = cu.add_cyclic_point(si_ext_l_g2_pm)


#%% plot the siconc
x, y = np.meshgrid(lon, lat)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)

# fig, axes = pl.subplots(ncols=3, nrows=2, figsize=(13, 7), subplot_kw={'projection':proj})

fig = pl.figure(figsize=(14, 7))
gs = gridspec.GridSpec(nrows=2, ncols=65, wspace=0.05, hspace=0.3) 

# upper panel -----------------------------------------------------------------------------------------------------------    
ax00 = pl.subplot(gs[0, 0:16], projection=proj)
ax01 = pl.subplot(gs[0, 16:32], projection=proj)
ax02 = pl.subplot(gs[0, 32:48], projection=proj)
ax03 = pl.subplot(gs[0, 48:64], projection=proj)
ax04 = pl.subplot(gs[0, 64])
ax10 = pl.subplot(gs[1, 0:16], projection=proj)
ax11 = pl.subplot(gs[1, 16:32], projection=proj)
ax12 = pl.subplot(gs[1, 32:48], projection=proj)
ax13 = pl.subplot(gs[1, 48:64], projection=proj)
ax14 = pl.subplot(gs[1, 64])

p1 = ax00.pcolormesh(x, y, sic_e_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
cb1 = fig.colorbar(p1, cax=ax04)
cb1.set_label("SICONC in %")
ax00.stock_img()
ax00.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax00.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax00.set_title("G1 $-$ years 1-5")

ax01.pcolormesh(x, y, sic_m_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax01.stock_img()
ax01.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax01.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax01.set_title("G1 $-$ years 18-22")

ax02.pcolormesh(x, y, sic_m2_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax02.stock_img()
ax02.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax02.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax02.set_title("G1 $-$ years 45-55")

ax03.pcolormesh(x, y, sic_l_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax03.stock_img()
ax03.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax03.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax03.set_title("G1 $-$ years 141-150")

p2 = ax10.pcolormesh(x, y, sic_e_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
cb2 = fig.colorbar(p2, cax=ax14)
cb2.set_label("SICONC in %")
ax10.stock_img()
ax10.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax10.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax10.set_title("G2 $-$ years 1-5")

ax11.pcolormesh(x, y, sic_m_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax11.stock_img()
ax11.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax11.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax11.set_title("G2 $-$ years 18-22")

ax12.pcolormesh(x, y, sic_m2_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax12.stock_img()
ax12.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax12.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax12.set_title("G2 $-$ years 45-55")

ax13.pcolormesh(x, y, sic_l_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=100, transform=ccrs.PlateCarree())
ax13.stock_img()
ax13.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax13.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax13.set_title("G2 $-$ years 141-150")

pl.show()
pl.close()


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

ax00.pcolormesh(x, y, si_ext_e_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax00.stock_img()
ax00.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax00.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax00.set_title("G1 $-$ years 1-5")

ax01.pcolormesh(x, y, si_ext_m_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax01.stock_img()
ax01.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax01.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax01.set_title("G1 $-$ years 18-22")

ax02.pcolormesh(x, y, si_ext_m2_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax02.stock_img()
ax02.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax02.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax02.set_title("G1 $-$ years 45-55")

ax03.pcolormesh(x, y, si_ext_l_g1_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax03.stock_img()
ax03.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax03.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax03.set_title("G1 $-$ years 141-150")

ax10.pcolormesh(x, y, si_ext_e_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax10.stock_img()
ax10.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax10.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax10.set_title("G2 $-$ years 1-5")

ax11.pcolormesh(x, y, si_ext_m_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax11.stock_img()
ax11.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax11.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax11.set_title("G2 $-$ years 18-22")

ax12.pcolormesh(x, y, si_ext_m2_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax12.stock_img()
ax12.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax12.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax12.set_title("G2 $-$ years 45-55")

ax13.pcolormesh(x, y, si_ext_l_g2_pm, cmap=cm.Blues_r, vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax13.stock_img()
ax13.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax13.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax13.set_title("G2 $-$ years 141-150")

pl.savefig(pl_path + "/PDF/Antarctic_SeaIce_Extent_MultTimeSlice_G1_G2.pdf", bbox_inches="tight", dpi=25)

pl.show()
pl.close()


#%% plot the sea-ice extent
x, y = np.meshgrid(lon, lat)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)

# fig, axes = pl.subplots(ncols=3, nrows=2, figsize=(13, 7), subplot_kw={'projection':proj})

fig = pl.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.05, hspace=0.3) 

# upper panel -----------------------------------------------------------------------------------------------------------    
ax00 = pl.subplot(gs[0, 0], projection=proj)
ax01 = pl.subplot(gs[0, 1], projection=proj)
ax10 = pl.subplot(gs[1, 0], projection=proj)
ax11 = pl.subplot(gs[1, 1], projection=proj)

ax00.pcolormesh(x, y, si_ext_e_d_pm, cmap=cm.RdBu_r, vmin=-2, vmax=2, transform=ccrs.PlateCarree())
ax00.stock_img()
ax00.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax00.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax00.set_title("G2$-$G1 years 1-5")

ax01.pcolormesh(x, y, si_ext_m_d_pm, cmap=cm.RdBu_r, vmin=-2, vmax=2, transform=ccrs.PlateCarree())
ax01.stock_img()
ax01.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax01.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax01.set_title("G2$-$G1 years 18-22")

ax10.pcolormesh(x, y, si_ext_m2_d_pm, cmap=cm.RdBu_r, vmin=-2, vmax=2, transform=ccrs.PlateCarree())
ax10.stock_img()
ax10.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax10.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax10.set_title("G2$-$G1 years 45-55")

ax11.pcolormesh(x, y, si_ext_l_d_pm, cmap=cm.RdBu_r, vmin=-2, vmax=2, transform=ccrs.PlateCarree())
ax11.stock_img()
ax11.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax11.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax11.set_title("G2$-$G1 years 141-150")

pl.show()
pl.close()
