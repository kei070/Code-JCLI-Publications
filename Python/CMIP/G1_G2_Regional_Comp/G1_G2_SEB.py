"""
Compare regional SEB. 

Generates Figs. 2 and S2 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir5 and data_dir6 as well as pl_path. Potentially change the paths in the for-loop.
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
from scipy.stats import ttest_ind
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
from Functions.Func_Climatology import climatology
from Functions.Func_SeasonalMean import sea_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Regrid_Data import remap as remap2
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups
models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]  # AMOC models
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]               # AMOC models   

# models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
# models2 = ["EC-Earth3", "FGOALS-g3", "INM-CM4-8", "ACCESS-ESM1-5"]

models = {"G1":models1, "G2":models2}
g_cols = {"G1":"blue", "G2":"red"}


#%% how many years
n_months = 60


#%% set the region
### --> Subpolar North Atlantic (SPNA) in Bellomo et al. (2021):  50-70N, 80W-10E

# South Atlantic
# x1, x2 = [305], [30]
# y1, y2 = [-60], [-40]

# South American Coast
# x1, x2 = [300], [320]
# y1, y2 = [-45], [-40]

x1, x2 = [300], [350]
y1, y2 = [40], [60]
# y1, y2 = [50], [80]

# x1, x2 = [280], [10]
# y1, y2 = [50], [70]

# x1, x2 = [300], [360]
# y1, y2 = [-15], [15]

# x1 = [145]
# x2 = [160]
# y1 = [35]
# y2 = [45]

# x1 = [178]
# x2 = [182]
# y1 = [42]
# y2 = [47]

# Bellomo et al. (2021)
# x1 = [280]
# x2 = [10]
# y1 = [50]
# y2 = [70]

# set the region name
reg_name = "2"
reg_name_title = {"1":"Barents-Norwegian-Greenland Sea", "2":"mid-latitude North Atlantic", 
                  "3":"Western Pacific", "4":"Central Mid-Latitude North Pacific", "5":"Eastern Japanese Coast",
                  "6":"Tropical Atlantic", "7":"Sub-Polar North Atlantic"}
reg_name_fn = {"1":"BNG_Sea", "2":"MLNA", "3":"NWP", "4":"CMLP", "5":"EJC", "6":"TrAt", "7":"SPNA"}


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
data_dir5 = "/CMIP5/"  # SETS BASIC DATA DIRECTORY
data_dir6 = "/CMIP6/"  # SETS BASIC DATA DIRECTORY

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

seb_pic_d = dict()
seb_a4x_d = dict()
dseb_d = dict()
seb_pic_run_d = dict()

for g12, group in zip(["G1", "G2"], [models1, models2]):
    
    mods = []
    mod_count = 0

    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    seb_pic_l = []
    seb_a4x_l = []
    dseb_l = []
    seb_pic_run_l = []

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
            si_varn = "sic"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_dir = nl6.models[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
            si_varn = "siconc"
        # end if        
        
        # set start and end index for the piControl experiment depending on the branch year
        bst = b_time * 12
        ben = bst + 150*12
        
        # load nc files
        seb_pic_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/seb_*piControl*.nc")[0])
        seb_a4x_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/seb_*{a4x}*.nc")[0])
        dseb_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/dseb_Amon_*.nc")[0])
        sftlf_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/sftlf_*.nc")[0])
       
        #% get lat and lon
        lat = seb_pic_nc.variables["lat"][:]
        lon = seb_pic_nc.variables["lon"][:]
        olat = lat + 90
        olon = lon + 0
        
        # extract region
        sftlf = sftlf_nc.variables["sftlf"][:]
        seb_temp = an_mean(seb_pic_nc.variables["seb"][bst:ben, :, :])
        sftlf = np.broadcast_to(sftlf, shape=np.shape(seb_temp))
        seb_temp = np.ma.masked_where(sftlf, seb_temp)
        seb_pic = region_mean(seb_temp, x1, x2, y1, y2, lat, lon, test_plt=True, plot_title="piC SEB")
        
        seb_temp = an_mean(seb_a4x_nc.variables["seb"][:150*12, :, :])
        seb_temp = np.ma.masked_where(sftlf, seb_temp)
        seb_a4x = region_mean(seb_temp, x1, x2, y1, y2, lat, lon, test_plt=True, plot_title="a4x SEB")        
        
        seb_temp = np.mean(dseb_nc.variables["dseb"][:], axis=0)
        seb_temp = np.ma.masked_where(sftlf, seb_temp)
        dseb = region_mean(seb_temp, x1, x2, y1, y2, lat, lon)
        
        seb_temp = np.mean(dseb_nc.variables["seb_pi_run"][:], axis=0)
        seb_temp = np.ma.masked_where(sftlf, seb_temp)
        seb_pic_run = region_mean(seb_temp, x1, x2, y1, y2, lat, lon)        
                
        seb_pic_l.append(seb_pic)
        seb_a4x_l.append(seb_a4x)
        dseb_l.append(dseb)
        seb_pic_run_l.append(seb_pic_run)
        
        print(g12 + " " + str(i) + ": " + mod_pl)
        
        mods.append(mod_pl)
        
        mod_count += 1
    # end for i, mod_d
    
    print("\n")
    
    # print(f"{g12} shape: {np.shape(cre_lw_pic)}")

    print("\n\n")
    # store everything in dictionaries
    seb_pic_d[g12] = np.array(seb_pic_l)
    seb_a4x_d[g12] = np.array(seb_a4x_l)
    dseb_d[g12] = np.array(dseb_l)
    seb_pic_run_d[g12] = np.array(seb_pic_run_l)
    
    mods_d[g12] = np.array(mods)
    
# end for g12


#%% calculate the group averages

# annual means
seb_pic_g1 = np.mean(seb_pic_d["G1"], axis=0)
seb_pic_g2 = np.mean(seb_pic_d["G2"], axis=0)
seb_a4x_g1 = np.mean(seb_a4x_d["G1"], axis=0)
seb_a4x_g2 = np.mean(seb_a4x_d["G2"], axis=0)
dseb_g1 = np.mean(dseb_d["G1"], axis=0)
dseb_g2 = np.mean(dseb_d["G2"], axis=0)
seb_pic_run_g1 = np.mean(seb_pic_run_d["G1"], axis=0)
seb_pic_run_g2 = np.mean(seb_pic_run_d["G2"], axis=0)

seb_pic_g1_std = np.std(seb_pic_d["G1"], axis=0)
seb_pic_g2_std = np.std(seb_pic_d["G2"], axis=0)
seb_a4x_g1_std = np.std(seb_a4x_d["G1"], axis=0)
seb_a4x_g2_std = np.std(seb_a4x_d["G2"], axis=0)
dseb_g1_std = np.std(dseb_d["G1"], axis=0)
dseb_g2_std = np.std(dseb_d["G2"], axis=0)

seb_pic_ttest = ttest_ind(seb_pic_d["G1"], seb_pic_d["G2"], axis=0)
seb_a4x_ttest = ttest_ind(seb_a4x_d["G1"], seb_a4x_d["G2"], axis=0)
dseb_ttest = ttest_ind(dseb_d["G1"], dseb_d["G2"], axis=0)


#%% plot the time series - version 1 (3 panels)
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(21.3, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(seb_pic_g1, c="blue")
axes[0].plot(seb_pic_g2, c="red")
axes[0].plot(seb_pic_run_g1, c="blue", linestyle="--")
axes[0].plot(seb_pic_run_g2, c="red", linestyle="--")

axes[0].fill_between(np.arange(len(seb_pic_g1)), seb_pic_g1 - seb_pic_g1_std, seb_pic_g1 + seb_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(len(seb_pic_g2)), seb_pic_g2 - seb_pic_g2_std, seb_pic_g2 + seb_pic_g2_std, 
                     facecolor="red", alpha=0.25)
axes[0].axhline(y=0.05, c="grey", linewidth=0.5)

ax01.plot(seb_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0].set_xlabel("Year of simulation", fontsize=14)
axes[0].set_ylabel("SEB in Wm$^{-2}$", fontsize=14)

axes[0].set_title("piControl", fontsize=16)


axes[1].plot(seb_a4x_g1, c="blue", label="G1")
axes[1].plot(seb_a4x_g2, c="red", label="G2")

axes[1].axhline(y=0.05, c="grey", linewidth=0.5)

axes[1].fill_between(np.arange(len(seb_a4x_g1)), seb_a4x_g1 - seb_a4x_g1_std, seb_a4x_g1 + seb_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(len(seb_a4x_g2)), seb_a4x_g2 - seb_a4x_g2_std, seb_a4x_g2 + seb_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

ax11.plot(seb_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1].set_xlabel("Year of simulation", fontsize=14)
axes[1].set_ylabel("SEB in Wm$^{-2}$", fontsize=14)

axes[1].set_title("abrupt4xCO2", fontsize=16)


axes[2].plot(dseb_g1, c="blue", label="G1")
axes[2].plot(dseb_g2, c="red", label="G2")
axes[2].axhline(y=0.05, c="grey", linewidth=0.5)
axes[2].legend(loc="lower center", fontsize=14)

axes[2].fill_between(np.arange(len(dseb_g1)), dseb_g1 - dseb_g1_std, dseb_g1 + dseb_g1_std, facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(len(dseb_g2)), dseb_g2 - dseb_g2_std, dseb_g2 + dseb_g2_std, facecolor="red", alpha=0.25)

# ax21.plot(dseb_ttest.pvalue, linewidth=0.5, c="black")
# ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))
ax21.set_yticks([])
ax21.set_yticklabels(labels=[])

axes[2].set_xlabel("Year of simulation", fontsize=14)
axes[2].set_ylabel("SEB in Wm$^{-2}$", fontsize=14)

axes[2].set_title("Change", fontsize=16)

axes[0].tick_params(axis='x', labelsize=13)
axes[0].tick_params(axis='y', labelsize=13)
axes[1].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)
axes[2].tick_params(axis='x', labelsize=13)
axes[2].tick_params(axis='y', labelsize=13)
ax01.tick_params(axis='x', labelsize=13)
ax01.tick_params(axis='y', labelsize=13)
ax11.tick_params(axis='x', labelsize=13)
ax11.tick_params(axis='y', labelsize=13)

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)

ax01.text(5, 0.68, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.68, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax21.text(5, 0.68, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"SEB {reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)", fontsize=17)
fig.subplots_adjust(wspace=0.35, top=0.85)

# pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_AMOC.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_AMOC.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_seb_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_seb_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - version 2 (4 panels)
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(13, 9))

ax01 = axes[0, 0].twinx()
ax11 = axes[0, 1].twinx()
ax21 = axes[1, 0].twinx()
ax31 = axes[1, 1].twinx()

axes[0, 0].plot(seb_pic_g1, c="blue")
axes[0, 0].plot(seb_pic_g2, c="red")
axes[0, 0].plot(seb_pic_run_g1, c="blue", linewidth=1.5, linestyle="--", label="G1 21-year running mean")
axes[0, 0].plot(seb_pic_run_g2, c="red", linewidth=1.5, linestyle="--", label="G2 21-year running mean")

axes[0, 0].legend(loc="upper center", fontsize=12)

axes[0, 0].fill_between(np.arange(len(seb_pic_g1)), seb_pic_g1 - seb_pic_g1_std, seb_pic_g1 + seb_pic_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 0].fill_between(np.arange(len(seb_pic_g2)), seb_pic_g2 - seb_pic_g2_std, seb_pic_g2 + seb_pic_g2_std, 
                        facecolor="red", alpha=0.25)
axes[0, 0].axhline(y=0, c="black", linewidth=0.5)

ax01.plot(seb_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0, 0].set_ylabel("SEB in Wm$^{-2}$", fontsize=14)

axes[0, 0].set_title("piControl", fontsize=16)


axes[0, 1].plot(seb_a4x_g1, c="blue", label="G1")
axes[0, 1].plot(seb_a4x_g2, c="red", label="G2")

axes[0, 1].axhline(y=0, c="black", linewidth=0.5)

axes[0, 1].fill_between(np.arange(len(seb_a4x_g1)), seb_a4x_g1 - seb_a4x_g1_std, seb_a4x_g1 + seb_a4x_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 1].fill_between(np.arange(len(seb_a4x_g2)), seb_a4x_g2 - seb_a4x_g2_std, seb_a4x_g2 + seb_a4x_g2_std, 
                        facecolor="red", alpha=0.25)

ax11.plot(seb_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[0, 1].set_ylabel("SEB in Wm$^{-2}$", fontsize=14)

axes[0, 1].set_title("abrupt4xCO2", fontsize=16)


axes[1, 0].plot(dseb_g1, c="blue", label="G1")
axes[1, 0].plot(dseb_g2, c="red", label="G2")
axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 0].legend(loc="lower center", fontsize=14)

axes[1, 0].fill_between(np.arange(len(dseb_g1)), dseb_g1 - dseb_g1_std, dseb_g1 + dseb_g1_std, facecolor="blue", 
                        alpha=0.25)
axes[1, 0].fill_between(np.arange(len(dseb_g2)), dseb_g2 - dseb_g2_std, dseb_g2 + dseb_g2_std, facecolor="red", 
                        alpha=0.25)
axes[1, 0].set_ylim((0, 40))

ax21.plot(dseb_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))
# ax21.set_yticks([])
# ax21.set_yticklabels(labels=[])

# ax31.plot(dseb_ttest.pvalue, linewidth=0.5, c="black", linestyle="--")
# ax31.axhline(y=0.05, c="grey", linewidth=0.5)
ax31.set_ylim((-0.01, 0.7))
ax31.set_yticks([])
ax31.set_yticklabels(labels=[])

axes[1, 0].set_xlabel("Year of simulation", fontsize=14)
axes[1, 0].set_ylabel("SEB change in Wm$^{-2}$", fontsize=14)
axes[1, 0].set_title("Change", fontsize=16)

axes[1, 1].plot(dseb_g2 - dseb_g1, c="black", label="change")
axes[1, 1].plot(seb_a4x_g2 - seb_a4x_g1, c="black", linestyle="--", label="abrupt4xCO2")
axes[1, 1].plot(seb_pic_g2 - seb_pic_g1, c="black", linestyle=":", label="piControl")
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].legend(loc="center", fontsize=14)

axes[1, 1].set_xlabel("Year of simulation", fontsize=14)
axes[1, 1].set_ylabel("SEB (change) difference in Wm$^{-2}$", fontsize=14)
axes[1, 1].set_title("Group difference (G2 $-$ G1)", fontsize=16)

axes[0, 0].tick_params(axis='x', labelsize=13)
axes[0, 0].tick_params(axis='y', labelsize=13)
axes[0, 1].tick_params(axis='x', labelsize=13)
axes[0, 1].tick_params(axis='y', labelsize=13)
axes[1, 0].tick_params(axis='x', labelsize=13)
axes[1, 0].tick_params(axis='y', labelsize=13)
axes[1, 1].tick_params(axis='x', labelsize=13)
axes[1, 1].tick_params(axis='y', labelsize=13)
ax01.tick_params(axis='x', labelsize=13)
ax01.tick_params(axis='y', labelsize=13)
ax11.tick_params(axis='x', labelsize=13)
ax11.tick_params(axis='y', labelsize=13)
ax21.tick_params(axis='x', labelsize=13)
ax21.tick_params(axis='y', labelsize=13)
ax31.tick_params(axis='x', labelsize=13)
ax31.tick_params(axis='y', labelsize=13)

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)
ax21.set_ylabel("$p$ value", fontsize=14)

ax01.text(5, 0.68, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.68, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax21.text(5, 0.68, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax31.text(5, 0.68, "(d)", fontsize=14, horizontalalignment="center", verticalalignment="center")

# fig.suptitle(f"SEB G1$-$G2 comparison {reg_name_title[reg_name]}\n" + 
#             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)", fontsize=17)
fig.suptitle(f"SEB G1$-$G2 comparison {reg_name_title[reg_name]}\n" + 
             f"({y1[0]}$-${y2[0]}$\degree$N, 10$-$60$\degree$W)", fontsize=17)
fig.subplots_adjust(wspace=0.4, top=0.9)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_AMOC_V2.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_seb_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_AMOC_V2.png", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_V2.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_seb_G1_G2_Comp_V2.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()