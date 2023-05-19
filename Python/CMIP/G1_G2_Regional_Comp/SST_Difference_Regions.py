"""
Compare the difference between two regions given by the user for two model groups given by the user for the abrupt4xCO2
and piControl experiments.

The purpose is to compare the Arctic--Tropics temperature difference in G1 and G2 (models with AMOC data!) so this the
standard set up.

Be sure to adjust data_dir5 and data_dir6 and pl_path.
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


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Region_Mean import region_mean
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


#%% set model groups
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]
# models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
# models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "BCC-CSM2-MR", 
#            "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]


#%% set up the regions

# "Atlantic corridor" Arctic
x11, x21 = [285], [25]
y11, y21 = [50], [80]

# Bellomo et al. (2021)
x11, x21 = [280], [10]
y11, y21 = [50], [70]

# MLNA
# x11, x21 = [300], [350]
# y11, y21 = [40], [60]

# "Atlantic corridor" Tropics
x12, x22 = [285], [25]
y12, y22 = [-15], [15]

# Southern Ocean
x13, x23 = [0], [360]
y13, y23 = [-55], [-35]


#%% choose the variable (tas or ts)
var = "ts"
var_s = "Ts"


#%% choose the threshold year; if this is set to 0, the "year of maximum feedback change" will be taken from the below
#   dictionary
thr_yr = 20

thr_min = 15
thr_max = 75


#%% load the namelist
import Namelists.Namelist_CMIP5 as nl5
a4x5 = "abrupt4xCO2"

import Namelists.Namelist_CMIP6 as nl6
a4x6 = "abrupt-4xCO2"


#%% add a a plot name suffix for excluded models

all_mod = np.array(['ACCESS1.0', 'ACCESS1.3', 'BCC-CSM1.1', 'BCC-CSM1.1(m)', 'BNU-ESM', 'CanESM2', "CCSM4", 'CNRM-CM5', 
                    "FGOALS-s2", 'GFDL-CM3', "GFDL-ESM2G", 'GFDL-ESM2M', 'GISS-E2-H', 'GISS-E2-R', 'HadGEM2-ES', 
                    "INMCM4", 'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', "MIROC-ESM", 'MPI-ESM-LR', 'MPI-ESM-MR', 
                    'MPI-ESM-P', 'MRI-CGCM3', 'NorESM1-M',
                    'ACCESS-CM2', 'ACCESS-ESM1.5', 'AWI-CM-1.1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1.0',
                    'CanESM5', "CAS-ESM2.0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", 
                    'CMCC-CM2-SR5', "CMCC-ESM2", 'CNRM-CM6.1', "CNRM-CM6.1-HR", 'CNRM-ESM2.1', 'E3SM-1.0', 'EC-Earth3', 
                    "EC-Earth3-AerChem", 'EC-Earth3-Veg', 'FGOALS-f3-L', 'FGOALS-g3', "FIO-ESM2.0", 'GFDL-CM4', 
                    'GFDL-ESM4', 'GISS-E2.1-G', 'GISS-E2.1-H', 'GISS-E2.2-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 
                    "IITM-ESM",  'INM-CM4.8', 'INM-CM5', 'IPSL-CM6A-LR', "KACE-1.0-G", "KIOST-ESM", 'MIROC-ES2L', 
                    'MIROC6', 'MPI-ESM1.2-HAM', 'MPI-ESM1.2-HR', 'MPI-ESM1.2-LR', 'MRI-ESM2.0', "NESM3", 'NorCPM1', 
                    "NorESM2-LM", "NorESM2-MM", 'SAM0-UNICON', 'TaiESM1', 'UKESM1.0-LL'])


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
mods_d = dict()

pl_add = ""
tl_add = ""

ar_ts_a4x_d = dict()
ar_ts_pic_d = dict()
tr_ts_a4x_d = dict()
tr_ts_pic_d = dict()
so_ts_a4x_d = dict()
so_ts_pic_d = dict()
gl_ts_a4x_d = dict()
gl_ts_pic_d = dict()
ts_pic_d = dict()

for g12, group in zip(["G1", "G2"], [models1, models2]):
    
    mods = []
    mod_count = 0

    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    ar_ts_a4x_l = []
    ar_ts_pic_l = []
    tr_ts_a4x_l = []
    tr_ts_pic_l = []
    so_ts_a4x_l = []
    so_ts_pic_l = []
    gl_ts_a4x_l = []
    gl_ts_pic_l = []
    ts_pic_l = []
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            mod_dir = nl5.models[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_dir = nl6.models[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
        # end if        
    
        # load nc files
        ts_a4x_nc = Dataset(glob.glob(data_dir + f"{mod_d}/ts_*4xCO2*.nc")[0])
        ts_pic_nc = Dataset(glob.glob(data_dir + f"{mod_d}/ts_*piControl*.nc")[0])
        
        #% get lat and lon
        lat = ts_a4x_nc.variables["lat"][:]
        lon = ts_a4x_nc.variables["lon"][:]
        
        # set start and end index for the piControl experiment depending on the branch year
        bst = b_time * 12
        ben = bst + 150*12
        
        # load the values
        ar_ts_a4x_l.append(region_mean(an_mean(ts_a4x_nc.variables["ts"][:150*12, :, :]), x11, x21, y11, y21, lat, lon))
        ar_ts_pic_l.append(region_mean(an_mean(ts_pic_nc.variables["ts"][bst:ben, :, :]), x11, x21, y11, y21, lat, lon))
        tr_ts_a4x_l.append(region_mean(an_mean(ts_a4x_nc.variables["ts"][:150*12, :, :]), x12, x22, y12, y22, lat, lon))
        tr_ts_pic_l.append(region_mean(an_mean(ts_pic_nc.variables["ts"][bst:ben, :, :]), x12, x22, y12, y22, lat, lon))
        so_ts_a4x_l.append(region_mean(an_mean(ts_a4x_nc.variables["ts"][:150*12, :, :]), x13, x23, y13, y23, lat, lon))
        so_ts_pic_l.append(region_mean(an_mean(ts_pic_nc.variables["ts"][bst:ben, :, :]), x13, x23, y13, y23, lat, lon))
        gl_ts_a4x_l.append(glob_mean(an_mean(ts_a4x_nc.variables["ts"][:150*12, :, :]), lat, lon))
        gl_ts_pic_l.append(glob_mean(an_mean(ts_pic_nc.variables["ts"][bst:ben, :, :]), lat, lon))
        
        # ts_pic_l.append(np.mean(ts_pic_nc.variables["ts"][bst:ben, :, :], axis=0)) 
        
        print(g12 + " " + str(i))
        
        mods.append(mod_pl)
        
        mod_count += 1
    # end for i, mod_d
    
    print("\n\n")

    #% convert the lists into numpy arrays
    ar_ts_a4x_a = np.array(ar_ts_a4x_l)
    ar_ts_pic_a = np.array(ar_ts_pic_l)
    tr_ts_a4x_a = np.array(tr_ts_a4x_l)
    tr_ts_pic_a = np.array(tr_ts_pic_l)
    so_ts_a4x_a = np.array(so_ts_a4x_l)
    so_ts_pic_a = np.array(so_ts_pic_l)
    gl_ts_a4x_a = np.array(gl_ts_a4x_l)
    gl_ts_pic_a = np.array(gl_ts_pic_l)

    print(f"{g12} shape: {np.shape(ar_ts_a4x_a)}")

    # store everything in dictionaries
    ar_ts_a4x_d[g12] = ar_ts_a4x_a
    ar_ts_pic_d[g12] = ar_ts_pic_a
    tr_ts_a4x_d[g12] = tr_ts_a4x_a
    tr_ts_pic_d[g12] = tr_ts_pic_a
    so_ts_a4x_d[g12] = so_ts_a4x_a
    so_ts_pic_d[g12] = so_ts_pic_a
    gl_ts_a4x_d[g12] = gl_ts_a4x_a
    gl_ts_pic_d[g12] = gl_ts_pic_a
    # ts_pic_d[g12] = ts_pic_l
    
    mods_d[g12] = np.array(mods)
    
# end for g12


#%% calculate the group means and standard deviations
g1_ts_ar_a4x_m = np.mean(ar_ts_a4x_d["G1"], axis=0)
g1_ts_ar_a4x_s = np.std(ar_ts_a4x_d["G1"], axis=0)
g1_ts_ar_pic_m = np.mean(ar_ts_pic_d["G1"], axis=0)
g1_ts_ar_pic_s = np.std(ar_ts_pic_d["G1"], axis=0)
g2_ts_ar_a4x_m = np.mean(ar_ts_a4x_d["G2"], axis=0)
g2_ts_ar_a4x_s = np.std(ar_ts_a4x_d["G2"], axis=0)
g2_ts_ar_pic_m = np.mean(ar_ts_pic_d["G2"], axis=0)
g2_ts_ar_pic_s = np.std(ar_ts_pic_d["G2"], axis=0)

g1_ts_tr_a4x_m = np.mean(tr_ts_a4x_d["G1"], axis=0)
g1_ts_tr_a4x_s = np.std(tr_ts_a4x_d["G1"], axis=0)
g1_ts_tr_pic_m = np.mean(tr_ts_pic_d["G1"], axis=0)
g1_ts_tr_pic_s = np.std(tr_ts_pic_d["G1"], axis=0)
g2_ts_tr_a4x_m = np.mean(tr_ts_a4x_d["G2"], axis=0)
g2_ts_tr_a4x_s = np.std(tr_ts_a4x_d["G2"], axis=0)
g2_ts_tr_pic_m = np.mean(tr_ts_pic_d["G2"], axis=0)
g2_ts_tr_pic_s = np.std(tr_ts_pic_d["G2"], axis=0)

g1_ts_so_a4x_m = np.mean(so_ts_a4x_d["G1"], axis=0)
g1_ts_so_a4x_s = np.std(so_ts_a4x_d["G1"], axis=0)
g1_ts_so_pic_m = np.mean(so_ts_pic_d["G1"], axis=0)
g1_ts_so_pic_s = np.std(so_ts_pic_d["G1"], axis=0)
g2_ts_so_a4x_m = np.mean(so_ts_a4x_d["G2"], axis=0)
g2_ts_so_a4x_s = np.std(so_ts_a4x_d["G2"], axis=0)
g2_ts_so_pic_m = np.mean(so_ts_pic_d["G2"], axis=0)
g2_ts_so_pic_s = np.std(so_ts_pic_d["G2"], axis=0)

g1_ts_gl_a4x_m = np.mean(gl_ts_a4x_d["G1"], axis=0)
g1_ts_gl_a4x_s = np.std(gl_ts_a4x_d["G1"], axis=0)
g1_ts_gl_pic_m = np.mean(gl_ts_pic_d["G1"], axis=0)
g1_ts_gl_pic_s = np.std(gl_ts_pic_d["G1"], axis=0)
g2_ts_gl_a4x_m = np.mean(gl_ts_a4x_d["G2"], axis=0)
g2_ts_gl_a4x_s = np.std(gl_ts_a4x_d["G2"], axis=0)
g2_ts_gl_pic_m = np.mean(gl_ts_pic_d["G2"], axis=0)
g2_ts_gl_pic_s = np.std(gl_ts_pic_d["G2"], axis=0)


#%% calculate the temperature difference between Tropics and Arctic
g1_dts_a4x = g1_ts_ar_a4x_m - g1_ts_tr_a4x_m
g1_dts_pic = g1_ts_ar_pic_m - g1_ts_tr_pic_m

g2_dts_a4x = g2_ts_ar_a4x_m - g2_ts_tr_a4x_m
g2_dts_pic = g2_ts_ar_pic_m - g2_ts_tr_pic_m


#%% generate some test plots
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].plot(g1_ts_ar_a4x_m, c="blue", label="G1 4xCO$_2$")
axes[0].plot(g2_ts_ar_a4x_m, c="red", label="G2 4xCO$_2$")
axes[0].plot(g1_ts_ar_pic_m, c="blue", label="G1 piControl", linestyle="--")
axes[0].plot(g2_ts_ar_pic_m, c="red", label="G2 piControl", linestyle="--")

axes[0].legend()
axes[0].set_title("Arctic")

axes[1].plot(g1_ts_tr_a4x_m, c="blue", label="G1 Tropics")
axes[1].plot(g2_ts_tr_a4x_m, c="red", label="G2 Tropics")
axes[1].plot(g1_ts_tr_pic_m, c="blue", label="G1 Tropics", linestyle="--")
axes[1].plot(g2_ts_tr_pic_m, c="red", label="G2 Tropics", linestyle="--")

axes[1].legend()
axes[1].set_title("Tropics")

pl.show()
pl.close()


#%% plot the temperature difference between Arctic and Tropics in piControl and abrupt4xCO2
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(g1_dts_a4x, c="blue", label="G1 a4xCO$_2$")
axes.plot(g2_dts_a4x, c="red", label="G2 a4xCO$_2$")
axes.plot(g1_dts_pic, c="blue", linestyle="--", label="G1 piC")
axes.plot(g2_dts_pic, c="red", linestyle="--", label="G2 piC")

axes.legend()

axes.set_title("Arctic $-$ Tropics SST difference")

pl.show()
pl.close()


#%% print the values for Arctic -- Tropics temperature difference for the piControl mean and for a 5-year mean centered
#   around year 20 in the abrupt4xCO2 run
g1_dts_pic_mean = np.mean(g1_dts_pic)
g2_dts_pic_mean = np.mean(g2_dts_pic)

g1_dts_a4x_y20m = np.mean(g1_dts_a4x[12:17])
g2_dts_a4x_y20m = np.mean(g2_dts_a4x[12:17])

# g1_dts_a4x_y20m = np.mean(g1_dts_a4x[12:17])
# g2_dts_a4x_y20m = np.mean(g2_dts_a4x[12:17])

g1_ddts = g1_dts_a4x_y20m - g1_dts_pic_mean
g2_ddts = g2_dts_a4x_y20m - g2_dts_pic_mean


# print the values
print(f"G1 piC mean: {np.round(g1_dts_pic_mean, 2)} K")
print(f"G2 piC mean: {np.round(g2_dts_pic_mean, 2)} K")

# NA = North Atlantic, TA = Tropical Atlantic
print(f"G1 a4x year-15 mean NA-TA SST: {np.round(g1_dts_a4x_y20m, 2)} K")
print(f"G2 a4x year-15 mean NA-TA SST: {np.round(g2_dts_a4x_y20m, 2)} K")

print(f"G1 a4x minus piC: {np.round(g1_ddts, 2)}")
print(f"G2 a4x minus piC: {np.round(g2_ddts, 2)}")


#%% look into the Southern Ocean
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(g1_ts_so_a4x_m, c="blue", label="G1 a4xCO$_2$")
axes.fill_between(np.arange(150), g1_ts_so_a4x_m - g1_ts_so_a4x_s, g1_ts_so_a4x_m + g1_ts_so_a4x_s, facecolor="blue", 
                  alpha=0.15)
axes.plot(g2_ts_so_a4x_m, c="red", label="G2 a4xCO$_2$")
axes.fill_between(np.arange(150), g2_ts_so_a4x_m - g2_ts_so_a4x_s, g2_ts_so_a4x_m + g2_ts_so_a4x_s, facecolor="red", 
                  alpha=0.15)
axes.plot(g1_ts_so_pic_m, c="blue", linestyle="--", label="G1 piC")
axes.fill_between(np.arange(150), g1_ts_so_pic_m - g1_ts_so_pic_s, g1_ts_so_pic_m + g1_ts_so_pic_s, facecolor="blue", 
                  alpha=0.15)
axes.plot(g2_ts_so_pic_m, c="red", linestyle="--", label="G2 piC")
axes.fill_between(np.arange(150), g2_ts_so_pic_m - g2_ts_so_pic_s, g2_ts_so_pic_m + g2_ts_so_pic_s, facecolor="red", 
                  alpha=0.15)

axes.legend()

axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("SST in K")
axes.set_title(f"Southern Ocean SST ({abs(y23[0])}-{abs(y13[0])}$\degree$S)")

pl.show()
pl.close()


#%% look into the global mean
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(g1_ts_gl_a4x_m, c="blue", label="G1 a4xCO$_2$")
axes.fill_between(np.arange(150), g1_ts_gl_a4x_m - g1_ts_gl_a4x_s, g1_ts_gl_a4x_m + g1_ts_gl_a4x_s, facecolor="blue", 
                  alpha=0.15)
axes.plot(g2_ts_gl_a4x_m, c="red", label="G2 a4xCO$_2$")
axes.fill_between(np.arange(150), g2_ts_gl_a4x_m - g2_ts_gl_a4x_s, g2_ts_gl_a4x_m + g2_ts_gl_a4x_s, facecolor="red", 
                  alpha=0.15)
axes.plot(g1_ts_gl_pic_m, c="blue", linestyle="--", label="G1 piC")
axes.fill_between(np.arange(150), g1_ts_gl_pic_m - g1_ts_gl_pic_s, g1_ts_gl_pic_m + g1_ts_gl_pic_s, facecolor="blue", 
                  alpha=0.15)
axes.plot(g2_ts_gl_pic_m, c="red", linestyle="--", label="G2 piC")
axes.fill_between(np.arange(150), g2_ts_gl_pic_m - g2_ts_gl_pic_s, g2_ts_gl_pic_m + g2_ts_gl_pic_s, facecolor="red", 
                  alpha=0.15)

axes.legend()

axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("Ts in K")
axes.set_title("Global mean Ts")

pl.show()
pl.close()


#%% plot the group difference
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(g1_ts_gl_a4x_m - g2_ts_gl_a4x_m, c="red", label="a4xCO$_2$")
axes.plot(g1_ts_gl_pic_m - g2_ts_gl_pic_m, c="blue", label="piControl")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.legend()

axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("Ts in K")
axes.set_title("Global mean Ts $-$ G1 minus G2")

pl.show()
pl.close()


