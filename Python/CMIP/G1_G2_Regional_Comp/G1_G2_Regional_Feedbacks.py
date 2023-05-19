"""
Compare regional feedbacks.

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
# models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
# models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]
models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "BCC-CSM2-MR", "EC-Earth3", "FGOALS-g3", 
           "INM-CM4-8"]

models = {"G1":models1, "G2":models2}
g_cols = {"G1":"blue", "G2":"red"}


#%% set the region
### --> Subpolar North Atlantic (SPNA) in Bellomo et al. (2021):  50-70N, 80W-10E

# 2: mid-latitude North Atlantic
x1 = [300]
x2 = [350]
y1 = [40]
y2 = [60]

x1 = [145]
x2 = [160]
y1 = [35]
y2 = [45]

x1 = [178]
x2 = [182]
y1 = [42]
y2 = [47]

# 6: excluding northern extra-tropics
x1, x2 = [0], [360]
y1, y2 = [-90], [30]

# 7: northern extra-tropics
# x1, x2 = [0], [360]
# y1, y2 = [30], [90]

# Bellomo et al. (2021)
# x1 = [280]
# x2 = [10]
# y1 = [50]
# y2 = [70]

# set the region name
reg_name = "6"
reg_name_title = {"1":"Barents-Norwegian-Greenland Sea", "2":"Mid-latitude North Atlantic", 
                  "3":"Western Pacific", "4":"Central Mid-Latitude North Pacific", "5":"Eastern Japanese Coast",
                  "6":"Excl. Northern Extra-tropics", "7":"Northern Extra-tropics"}
reg_name_fn = {"1":"BNG_Sea", "2":"MLNA", "3":"NWP", "4":"CMLP", "5":"EJC", "6":"ExNExTr", "7":"NExTr"}


#%% set up some initial dictionaries and lists --> NEEDED FOR CSLT !!!

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% choose the threshold year; if this is set to 0, the "year of maximum feedback change" will be taken from the below
#   dictionary
thr_yr = 20

thr_min = 15
thr_max = 75


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

a4x5 = "abrupt4xCO2"
a4x6 = "abrupt-4xCO2"

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = "/MultModPlots/G1_G2_Regional_Comparison_piC_a4x/"

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
mods_d = dict()

pl_add = ""
tl_add = ""

lr_fb_e_d = dict()
lr_fb_l_d = dict()
lr_fb_d_d = dict()
c_fb_e_d = dict()
c_fb_l_d = dict()
c_fb_d_d = dict()
c_lw_fb_e_d = dict()
c_lw_fb_l_d = dict()
c_lw_fb_d_d = dict()
c_sw_fb_e_d = dict()
c_sw_fb_l_d = dict()
c_sw_fb_d_d = dict()

for g12, group in zip(["G1", "G2"], [models1, models2]):
    
    mods = []
    mod_count = 0

    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    lr_fb_e_l = []
    lr_fb_l_l = []
    lr_fb_d_l = []
    c_fb_e_l = []
    c_fb_l_l = []
    c_fb_d_l = []
    c_lw_fb_e_l = []
    c_lw_fb_l_l = []
    c_lw_fb_d_l = []
    c_sw_fb_e_l = []
    c_sw_fb_l_l = []
    c_sw_fb_d_l = []
    
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
        fb_nc = Dataset(glob.glob(data_dir + f"Outputs/Feedbacks_Local/Kernel/{a4x}/{k_p[kl]}/tas_Based/*{mod_d}.nc")[0])
        
        #% get lat and lon
        lat = fb_nc.variables["lat"][:]
        lon = fb_nc.variables["lon"][:]
        olat = lat + 90
        olon = lon + 0
        
        # extract region
        lr_fb_e = region_mean(fb_nc.variables["LR_fb_e"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                              plot_title="LR")
        lr_fb_l = region_mean(fb_nc.variables["LR_fb_l"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                              plot_title="LR")
        lr_fb_d = region_mean(fb_nc.variables["LR_dfb"][:, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                              plot_title="LR")
        c_fb_e = region_mean(fb_nc.variables["C_fb_e"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                             plot_title="C")
        c_fb_l = region_mean(fb_nc.variables["C_fb_l"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                             plot_title="C")
        c_fb_d = region_mean(fb_nc.variables["C_dfb"][:, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                             plot_title="C")
        clw_fb_e = region_mean(fb_nc.variables["C_lw_fb_e"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        clw_fb_l = region_mean(fb_nc.variables["C_lw_fb_l"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        clw_fb_d = region_mean(fb_nc.variables["C_lw_dfb"][:, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        csw_fb_e = region_mean(fb_nc.variables["C_sw_fb_e"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        csw_fb_l = region_mean(fb_nc.variables["C_sw_fb_l"][0, :, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        csw_fb_d = region_mean(fb_nc.variables["C_sw_dfb"][:, :], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="C")
        
        lr_fb_e_l.append(lr_fb_e)
        lr_fb_l_l.append(lr_fb_l)
        lr_fb_d_l.append(lr_fb_d)
        c_fb_e_l.append(c_fb_e)
        c_fb_l_l.append(c_fb_l)
        c_fb_d_l.append(c_fb_d)
        c_lw_fb_e_l.append(clw_fb_e)
        c_lw_fb_l_l.append(clw_fb_l)
        c_lw_fb_d_l.append(clw_fb_d)
        c_sw_fb_e_l.append(csw_fb_e)
        c_sw_fb_l_l.append(csw_fb_l)
        c_sw_fb_d_l.append(csw_fb_d)
        
        print(g12 + " " + str(i) + ": " + mod_pl)
        
        mods.append(mod_pl)
        
        mod_count += 1
    # end for i, mod_d
    
    print("\n")
    
    # print(f"{g12} shape: {np.shape(cre_lw_pic)}")

    print("\n\n")
    # store everything in dictionaries
    lr_fb_e_d[g12] = np.array(lr_fb_e_l)
    lr_fb_l_d[g12] = np.array(lr_fb_l_l)
    lr_fb_d_d[g12] = np.array(lr_fb_d_l)
    c_fb_e_d[g12] = np.array(c_fb_e_l)
    c_fb_l_d[g12] = np.array(c_fb_l_l)
    c_fb_d_d[g12] = np.array(c_fb_d_l)
    c_lw_fb_e_d[g12] = np.array(c_lw_fb_e_l)
    c_lw_fb_l_d[g12] = np.array(c_lw_fb_l_l)
    c_lw_fb_d_d[g12] = np.array(c_lw_fb_d_l)
    c_sw_fb_e_d[g12] = np.array(c_sw_fb_e_l)
    c_sw_fb_l_d[g12] = np.array(c_sw_fb_l_l)
    c_sw_fb_d_d[g12] = np.array(c_sw_fb_d_l)
    
    mods_d[g12] = np.array(mods)
    
# end for g12


#%% t-test
lr_e_ttest = ttest_ind(lr_fb_e_d["G1"], lr_fb_e_d["G2"], axis=0)
lr_l_ttest = ttest_ind(lr_fb_l_d["G1"], lr_fb_l_d["G2"], axis=0)
lr_d_ttest = ttest_ind(lr_fb_d_d["G1"], lr_fb_d_d["G2"], axis=0)
c_e_ttest = ttest_ind(c_fb_e_d["G1"], c_fb_e_d["G2"], axis=0)
c_l_ttest = ttest_ind(c_fb_l_d["G1"], c_fb_l_d["G2"], axis=0)
c_d_ttest = ttest_ind(c_fb_d_d["G1"], c_fb_d_d["G2"], axis=0)
c_lw_e_ttest = ttest_ind(c_lw_fb_e_d["G1"], c_lw_fb_e_d["G2"], axis=0)
c_lw_l_ttest = ttest_ind(c_lw_fb_l_d["G1"], c_lw_fb_l_d["G2"], axis=0)
c_lw_d_ttest = ttest_ind(c_lw_fb_d_d["G1"], c_lw_fb_d_d["G2"], axis=0)
c_sw_e_ttest = ttest_ind(c_sw_fb_e_d["G1"], c_sw_fb_e_d["G2"], axis=0)
c_sw_l_ttest = ttest_ind(c_sw_fb_l_d["G1"], c_sw_fb_l_d["G2"], axis=0)
c_sw_d_ttest = ttest_ind(c_sw_fb_d_d["G1"], c_sw_fb_d_d["G2"], axis=0)


#%% print values
print(f"G1 LR early: {r2(np.mean(lr_fb_e_d['G1']))} +/- {r2(np.std(lr_fb_e_d['G1']))}")
print(f"G1 LR late: {r2(np.mean(lr_fb_l_d['G1']))} +/- {r2(np.std(lr_fb_l_d['G1']))}")
print(f"G1 LR change: {r2(np.mean(lr_fb_d_d['G1']))} +/- {r2(np.std(lr_fb_d_d['G1']))}")
print(f"G1 C early: {r2(np.mean(c_fb_e_d['G1']))} +/- {r2(np.std(c_fb_e_d['G1']))}")
print(f"G1 C late: {r2(np.mean(c_fb_l_d['G1']))} +/- {r2(np.std(c_fb_l_d['G1']))}")
print(f"G1 C change: {r2(np.mean(c_fb_d_d['G1']))} +/- {r2(np.std(c_fb_d_d['G1']))}\n")

print(f"G2 LR early: {r2(np.mean(lr_fb_e_d['G2']))} +/- {r2(np.std(lr_fb_e_d['G2']))}")
print(f"G2 LR late: {r2(np.mean(lr_fb_l_d['G2']))} +/- {r2(np.std(lr_fb_l_d['G2']))}")
print(f"G2 LR change: {r2(np.mean(lr_fb_d_d['G2']))} +/- {r2(np.std(lr_fb_d_d['G2']))}")
print(f"G2 C early: {r2(np.mean(c_fb_e_d['G2']))} +/- {r2(np.std(c_fb_e_d['G2']))}")
print(f"G2 C late: {r2(np.mean(c_fb_l_d['G2']))} +/- {r2(np.std(c_fb_l_d['G2']))}")
print(f"G2 C change: {r2(np.mean(c_fb_d_d['G2']))} +/- {r2(np.std(c_fb_d_d['G2']))}")


#%% plot group means
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 6), sharey=True)

ax00 = axes[0].twinx()
ax01 = axes[1].twinx()

axes[0].errorbar(0, np.mean(lr_fb_e_d["G1"]), yerr=np.std(lr_fb_e_d["G1"]), c="blue", label="G1", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(0, np.mean(lr_fb_e_d["G2"]), yerr=np.std(lr_fb_e_d["G2"]), c="red", label="G2", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(1, np.mean(lr_fb_l_d["G1"]), yerr=np.std(lr_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(1, np.mean(lr_fb_l_d["G2"]), yerr=np.std(lr_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(2, np.mean(lr_fb_d_d["G1"]), yerr=np.std(lr_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(2, np.mean(lr_fb_d_d["G2"]), yerr=np.std(lr_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)

ax00.scatter(0, lr_e_ttest.pvalue, c="gray")
ax00.scatter(1, lr_l_ttest.pvalue, c="gray")
ax00.scatter(2, lr_d_ttest.pvalue, c="gray")
ax00.set_ylim((0, 0.1))
ax00.set_yticks([0, 0.025, 0.05, 0.075, 0.1])

axes[0].set_ylim(-3.5, 3.5)

axes[0].axhline(y=0, c="gray", linewidth=1.5, zorder=102)

axes[0].legend(loc="upper left")

axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(["early", "late", "change"])

axes[0].set_ylabel("Feedback in Wm$^{-1}$K$^{-2}$")
axes[0].set_title("Lapse-rate feedback")


axes[1].errorbar(0, np.mean(c_fb_e_d["G1"]), yerr=np.std(lr_fb_e_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(0, np.mean(c_fb_e_d["G2"]), yerr=np.std(lr_fb_e_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(1, np.mean(c_fb_l_d["G1"]), yerr=np.std(lr_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(1, np.mean(c_fb_l_d["G2"]), yerr=np.std(lr_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(2, np.mean(c_fb_d_d["G1"]), yerr=np.std(lr_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(2, np.mean(c_fb_d_d["G2"]), yerr=np.std(lr_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)

ax01.scatter(0, c_e_ttest.pvalue, c="gray")
ax01.scatter(1, c_l_ttest.pvalue, c="gray")
ax01.scatter(2, c_d_ttest.pvalue, c="gray")
ax01.set_ylim((0, 0.1))
ax01.set_yticks([0, 0.025, 0.05, 0.075, 0.1])
# ax01.set_xticklabels(["early", "late", "change"])

axes[1].axhline(y=0, c="gray", linewidth=1.5, zorder=102)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["early", "late", "change"])
axes[1].set_title("Cloud feedback")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)")
fig.subplots_adjust(top=0.89, wspace=0.3)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_LR_C_Fb_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_LR_C_Fb_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot group means
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 6), sharey=True)

ax00 = axes[0].twinx()
ax01 = axes[1].twinx()

axes[0].errorbar(0, np.mean(c_lw_fb_e_d["G1"]), yerr=np.std(c_lw_fb_e_d["G1"]), c="blue", label="G1", fmt="o", 
                 zorder=101, linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(0, np.mean(c_lw_fb_e_d["G2"]), yerr=np.std(c_lw_fb_e_d["G2"]), c="red", label="G2", fmt="o", 
                 zorder=101, linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(1, np.mean(c_lw_fb_l_d["G1"]), yerr=np.std(c_lw_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(1, np.mean(c_lw_fb_l_d["G2"]), yerr=np.std(c_lw_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(2, np.mean(c_lw_fb_d_d["G1"]), yerr=np.std(c_lw_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[0].errorbar(2, np.mean(c_lw_fb_d_d["G2"]), yerr=np.std(c_lw_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)

ax00.scatter(0, c_lw_e_ttest.pvalue, c="gray")
ax00.scatter(1, c_lw_l_ttest.pvalue, c="gray")
ax00.scatter(2, c_lw_d_ttest.pvalue, c="gray")
ax00.set_ylim((0, 0.1))
ax00.set_yticks([0, 0.025, 0.05, 0.075, 0.1])

axes[0].set_ylim(-3.5, 3.5)

axes[0].axhline(y=0, c="gray", linewidth=1.5, zorder=102)

axes[0].legend(loc="upper left")

axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(["early", "late", "change"])

axes[0].set_ylabel("Feedback in Wm$^{-1}$K$^{-2}$")
axes[0].set_title("Long-wave cloud feedback")


axes[1].errorbar(0, np.mean(c_sw_fb_e_d["G1"]), yerr=np.std(c_sw_fb_e_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(0, np.mean(c_sw_fb_e_d["G2"]), yerr=np.std(c_sw_fb_e_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(1, np.mean(c_sw_fb_l_d["G1"]), yerr=np.std(c_sw_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(1, np.mean(c_sw_fb_l_d["G2"]), yerr=np.std(c_sw_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(2, np.mean(c_sw_fb_d_d["G1"]), yerr=np.std(c_sw_fb_l_d["G1"]), c="blue", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)
axes[1].errorbar(2, np.mean(c_sw_fb_d_d["G2"]), yerr=np.std(c_sw_fb_l_d["G2"]), c="red", fmt="o", zorder=101,
                linewidth=1, capsize=3.5, markersize=8)

ax01.scatter(0, c_sw_e_ttest.pvalue, c="gray")
ax01.scatter(1, c_sw_l_ttest.pvalue, c="gray")
ax01.scatter(2, c_sw_d_ttest.pvalue, c="gray")
ax01.set_ylim((0, 0.1))
ax01.set_yticks([0, 0.025, 0.05, 0.075, 0.1])
# ax01.set_xticklabels(["early", "late", "change"])

axes[1].axhline(y=0, c="gray", linewidth=1.5, zorder=102)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["early", "late", "change"])
axes[1].set_title("Short-wave cloud feedback")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)")
fig.subplots_adjust(top=0.89, wspace=0.3)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_LW_SW_C_Fb_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_LW_SW_C_Fb_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()