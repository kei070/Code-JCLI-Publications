"""
Compare the differences of multiple "sub-parts" of the Northern Hemisphere between G1 and G2 for kernel-derived radiative
fluxes and feedbacks.

Generates Fig. 4, S4, S5, S9, and S10 in Eiselt and Graversen (2023), JCLI.

Be sure to set data_dir5 and data_dir6 correctly and change pl_path as well as set data_path (further below).
Also, make sure the data set with the target coordinates for the regridding exists.
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
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
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


#%% set the region
### --> Subpolar North Atlantic (SPNA) in Bellomo et al. (2021):  50-70N, 80W-10E

# set the region name
reg_name = "2"

reg_name_title = {"1":"Mid-latitude North Atlantic", "2":"Arctic", "3":"Northern Polar", "4":"Northern Extratropics", 
                  "5":"Northern Hemisphere"}
reg_name_fn = {"1":"MLNA", "2":"Arctic", "3":"NHP", "4":"NHExT", "5":"NH"}


# region coordinates
x1, x2 = [300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [350, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360]
y1, y2 = [40, 75, 60, 30, 0, 0, -90, -30, -90, -90, -90], [60, 90, 90, 90, 30, 90, 0, 0, -30, -60, -75]

# x1, x2 = [0, 0, 0, 0, 0], [360, 360, 360, 360, 360]
# y1, y2 = [-30, -90, -90, -90, -90], [0, 0, -30, -60, -75]


#%% set model groups
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]  # AMOC models
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]               # AMOC models   

models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "ACCESS-ESM1-5", "NorESM1_M", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models = {"G1":models1, "G2":models2}


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = "/"  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""

# os.makedirs(pl_path + "/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% kernel dictionaries
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% load the CanESM5 grid as target for regridding
can_nc = Dataset(glob.glob(data_dir6 + "/CMIP6/Data/CanESM5/ts_*4xCO2*.nc")[0])
tlat = can_nc.variables["lat"][:] + 90 
tlon = can_nc.variables["lon"][:]
pllat = can_nc.variables["lat"][:]
pllon = can_nc.variables["lon"][:]


#%% loop over group one
dtas_d = dict()
dtas_gl_d = dict()


for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    dtas_rg_l = []
    dtas_gl_l = []
    
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
        
        # load the files
        dtas_nc = Dataset(glob.glob(data_dir + f"/CMIP{cmip}/Data/{mod_d}/dtas_*{a4x}*.nc")[0])
        
        # load the ts grid
        lat = dtas_nc.variables["lat"][:]
        lon = dtas_nc.variables["lon"][:]
        
        dtas_temp = np.mean(dtas_nc.variables["tas_ch"][:], axis=0)
        
        dtas_rg_l.append(region_mean(dtas_temp, x1, x2, y1, y2, lat, lon, test_plt=False, plot_title="fl", 
                                     multi_reg_mean=False)) 
        
        dtas_gl = glob_mean(dtas_temp, lat, lon)

        dtas_gl_l.append(dtas_gl)
                
    # end for i, mod_d
    
    dtas_d[g_n] = np.array(dtas_rg_l)
    dtas_gl_d[g_n] = np.array(dtas_gl_l)

    
# end for g_n, group
"""
pl.plot(np.mean(dtas_d["G2"][:, 0, :], axis=0) - np.mean(dtas_d["G1"][:, 0, :], axis=0), c="violet")
pl.plot(np.mean(dtas_d["G2"][:, 1, :], axis=0) - np.mean(dtas_d["G1"][:, 1, :], axis=0), c="gray")
pl.plot(np.mean(dtas_d["G2"][:, 2, :], axis=0) - np.mean(dtas_d["G1"][:, 2, :], axis=0), c="red")
pl.plot(np.mean(dtas_d["G2"][:, 3, :], axis=0) - np.mean(dtas_d["G1"][:, 3, :], axis=0), c="orange")
pl.plot(np.mean(dtas_d["G2"][:, 4, :], axis=0) - np.mean(dtas_d["G1"][:, 4, :], axis=0), c="blue")
raise Exception
"""


#%% calculate the group mean

# MLNA
g1_dtas_rg = np.mean(dtas_d["G1"], axis=0)
g2_dtas_rg = np.mean(dtas_d["G2"], axis=0)

g1_dtas_rg_std = np.std(dtas_d["G1"], axis=0)
g2_dtas_rg_std = np.std(dtas_d["G2"], axis=0)

dg12_dtas_rg = g2_dtas_rg - g1_dtas_rg

dtas_rg_ttest = []
for i in np.arange(len(x1)):
    dtas_rg_ttest.append(ttest_ind(dtas_d["G1"][:, i, :], dtas_d["G2"][:, i, :], axis=0))
# end for i

# global
g1_dtas_gl = np.mean(dtas_gl_d["G1"], axis=0)
g2_dtas_gl = np.mean(dtas_gl_d["G2"], axis=0)

g1_dtas_gl_std = np.std(dtas_gl_d["G1"], axis=0)
g2_dtas_gl_std = np.std(dtas_gl_d["G2"], axis=0)

dg12_dtas_gl = g2_dtas_gl - g1_dtas_gl

dtas_gl_ttest = ttest_ind(dtas_gl_d["G1"], dtas_gl_d["G2"], axis=0)


#%% LOAD CESM2-SOM data: setup case name --------------------------------------------------------------------------------
case_4x51 = "Proj2_KUE_4xCO2"
case_4x41 = "Yr41_4xCO2"
case_4x61 = "Proj2_KUE_4xCO2_61"

case_dq4x51 = "dQ01yr_4xCO2"
case_dq4x41 = "Y41_dQ01"
case_dq4x61 = "Y61_dQ01"

# case_dq4x51 = "dQ01_r30_4xCO2"
# case_dq4x41 = "dQ01_r30_4xCO2_41"
# case_dq4x61 = "dQ01_r30_4xCO2_61"


#%% choose the variable (tas or ts)
t_var = "dTREFHT"
var_s = "TS"


#%% set the region

# set the region name
reg_name = "3"

reg_name_title = {"1":"Mid-latitude North Atlantic", 
                  "2":"Arctic", 
                  "3":"Northern Polar", 
                  "4":"Northern Extratropics", 
                  "5":"Northern Tropics", 
                  "6":"Northern Hemisphere", 
                  "7":"Southern Hemisphere", 
                  "8":"Southern Tropics", 
                  "9":"Southern Extratropics", 
                  "10":"Southern Polar",
                  "11":"Antarctica",
                  "12":"Global"}
reg_name_fn = {"1":"MLNA", 
               "2":"Arctic", 
               "3":"NP", 
               "4":"NExT",
               "5":"NT",
               "6":"NH", 
               "7":"SH", 
               "8":"ST", 
               "9":"SExT", 
               "10":"SP",
               "11":"Antarctica", 
               "12":"Global"}

region_n = reg_name_fn[reg_name]


#%% set paths
data_path = ""


#%% load the kernel-derived radiative fluxes
dtas_4x51_nc = Dataset(glob.glob(data_path + f"{case_4x51}/{t_var}_{case_4x51}_*.nc")[0])
dtas_4x41_nc = Dataset(glob.glob(data_path + f"{case_4x41}/{t_var}_{case_4x41}_*.nc")[0])
dtas_4x61_nc = Dataset(glob.glob(data_path + f"{case_4x61}/{t_var}_{case_4x61}_*.nc")[0])

dtas_dq4x51_nc = Dataset(glob.glob(data_path + f"{case_dq4x51}/{t_var}_{case_dq4x51}_*.nc")[0])
dtas_dq4x41_nc = Dataset(glob.glob(data_path + f"{case_dq4x41}/{t_var}_{case_dq4x41}_*.nc")[0])
dtas_dq4x61_nc = Dataset(glob.glob(data_path + f"{case_dq4x61}/{t_var}_{case_dq4x61}_*.nc")[0])


#%% load lat and lon
lat = dtas_4x51_nc.variables["lat"][:]
lon = dtas_4x51_nc.variables["lon"][:]


#%% calculate global mean surface temperature    
nyrs = 40
dtas_4x51 = glob_mean(dtas_4x51_nc.variables[t_var][:nyrs, :, :], lat, lon)
dtas_4x41 = glob_mean(dtas_4x41_nc.variables[t_var][:nyrs, :, :], lat, lon)
dtas_4x61 = glob_mean(dtas_4x61_nc.variables[t_var][:nyrs, :, :], lat, lon)

dtas_dq4x51 = glob_mean(dtas_dq4x51_nc.variables[t_var][:nyrs, :, :], lat, lon)
dtas_dq4x41 = glob_mean(dtas_dq4x41_nc.variables[t_var][:nyrs, :, :], lat, lon)
dtas_dq4x61 = glob_mean(dtas_dq4x61_nc.variables[t_var][:nyrs, :, :], lat, lon)


#%% loop over the 5 regions
dtas_dic = {}
avdtas_dic = {}
dtas_dq_dic = {}
avdtas_dq_dic = {}

for reg_name in reg_name_title.keys():
    # MLNA
    if reg_name == "1":
        x1, x2 = [300], [350]
        y1, y2 = [40], [60]
    
    # Arctic
    elif reg_name == "2":
        x1, x2 = [0], [360]
        y1, y2 = [75], [90]
    
    # North Hemisphere Polar
    elif reg_name == "3":
        x1, x2 = [0], [360]
        y1, y2 = [60], [90]
    
    # North Hemisphere Extratropics
    elif reg_name == "4":
        x1, x2 = [0], [360]
        y1, y2 = [30], [90]
    
    # Northern Tropics
    elif reg_name == "5":
        x1, x2 = [0], [360]
        y1, y2 = [0], [30]
        
    # North Hemisphere
    elif reg_name == "6":
        x1, x2 = [0], [360]
        y1, y2 = [0], [90]    
    
    # South Hemisphere
    elif reg_name == "7":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [0]
    
    # Southern Tropics
    elif reg_name == "8":
        x1, x2 = [0], [360]
        y1, y2 = [-30], [0]
    
    # Southern Extratropocs
    elif reg_name == "9":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-30]
    
    # Southern Polar
    elif reg_name == "10":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-60]        
        
    # Antarctica    
    elif reg_name == "11":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-75]            
    
    # Global
    elif reg_name == "12":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [90]            
    # end if elif
    
    dtas_dic[reg_name_fn[reg_name]] = np.zeros((3, nyrs))
    dtas_dq_dic[reg_name_fn[reg_name]] = np.zeros((3, nyrs))
    
    dtas_dic[reg_name_fn[reg_name]][0, :] = region_mean(dtas_4x51_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, lat, 
                                                        lon)
    dtas_dic[reg_name_fn[reg_name]][1, :] = region_mean(dtas_4x41_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, lat, 
                                                        lon)
    dtas_dic[reg_name_fn[reg_name]][2, :] = region_mean(dtas_4x61_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, lat, 
                                                        lon)
    avdtas_dic[reg_name_fn[reg_name]] = np.mean(dtas_dic[reg_name_fn[reg_name]], axis=0)
    
    dtas_dq_dic[reg_name_fn[reg_name]][0, :] = region_mean(dtas_dq4x51_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, 
                                                           lat, lon)
    dtas_dq_dic[reg_name_fn[reg_name]][1, :] = region_mean(dtas_dq4x41_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, 
                                                           lat, lon)
    dtas_dq_dic[reg_name_fn[reg_name]][2, :] = region_mean(dtas_dq4x61_nc.variables[t_var][:nyrs, :, :], x1, x2, y1, y2, 
                                                           lat, lon)
    avdtas_dq_dic[reg_name_fn[reg_name]] = np.mean(dtas_dq_dic[reg_name_fn[reg_name]], axis=0)

# end for reg_name


#%% calculate the p-values for the CESM2-SOM experiments (only possible if multiple ensembles exist)
dtas_dq_ttest = dict()

for name in dtas_dq_dic.keys():
    dtas_dq_ttest[name] = ttest_ind(dtas_dq_dic[name], dtas_dic[name], axis=0)
# end for name    


#%% plot the inter-group differences in radiative flux in one panel
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=False, sharex=False)

axes[0, 0].plot(dg12_dtas_rg[1, :], c="blue", label="R>75N")
axes[0, 0].plot(dg12_dtas_rg[2, :], c="orange", label="R>60N")
axes[0, 0].plot(dg12_dtas_rg[3, :], c="red", label="R>30N")
axes[0, 0].plot(dg12_dtas_rg[4, :], c="violet", label="R<30N")
axes[0, 0].plot(dg12_dtas_rg[5, :], c="gray", label="R>0N")

axes[0, 1].plot(dg12_dtas_rg[10, :], c="blue", linestyle="-", label="R>75S")
axes[0, 1].plot(dg12_dtas_rg[9, :], c="orange", linestyle="-", label="R>60S")
axes[0, 1].plot(dg12_dtas_rg[8, :], c="red", linestyle="-", label="R>30S")
axes[0, 1].plot(dg12_dtas_rg[7, :], c="violet", linestyle="-", label="R<30S")
axes[0, 1].plot(dg12_dtas_rg[6, :], c="gray", linestyle="-", label="R>0S")

axes[0, 0].plot(dg12_dtas_gl, c="black", linestyle="-", label="global")
axes[0, 1].plot(dg12_dtas_gl, c="black", linestyle="-", label="global")

axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)
axes[0, 0].legend(ncol=3, loc="upper right")
axes[0, 1].legend(ncol=3, loc="lower right")

# axes[0, 0].set_xlabel("Year of simulation")
# axes[0, 1].set_xlabel("Year of simulation")
axes[0, 0].set_ylabel("$\Delta$SAT in K")

axes[0, 0].set_title("Northern Hemisphere $-$ G2 minus G1")
axes[0, 1].set_title("Southern Hemisphere $-$ G2 minus G1")

axes[0, 0].text(0, -4.75, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(0, -4.75, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")


axes[1, 0].plot(avdtas_dq_dic["Arctic"] - avdtas_dic["Arctic"], c="blue", label="Arctic")
axes[1, 0].plot(avdtas_dq_dic["NP"] - avdtas_dic["NP"], c="orange", label="NP")
axes[1, 0].plot(avdtas_dq_dic["NExT"] - avdtas_dic["NExT"], c="red", label="NExT")
axes[1, 0].plot(avdtas_dq_dic["NT"] - avdtas_dic["NT"], c="violet", label="NT")
axes[1, 0].plot(avdtas_dq_dic["NH"] - avdtas_dic["NH"], c="gray", label="NH")

axes[1, 1].plot(avdtas_dq_dic["Antarctica"] - avdtas_dic["Antarctica"], c="blue", label="Antarctica")
axes[1, 1].plot(avdtas_dq_dic["SP"] - avdtas_dic["SP"], c="orange", label="SP")
axes[1, 1].plot(avdtas_dq_dic["SExT"] - avdtas_dic["SExT"], c="red", label="SExT")
axes[1, 1].plot(avdtas_dq_dic["ST"] - avdtas_dic["ST"], c="violet", label="ST")
axes[1, 1].plot(avdtas_dq_dic["SH"] - avdtas_dic["SH"], c="gray", label="SH")

axes[1, 0].plot(avdtas_dq_dic["Global"] - avdtas_dic["Global"], c="black", linestyle="-", label="global")
axes[1, 1].plot(avdtas_dq_dic["Global"] - avdtas_dic["Global"], c="black", linestyle="-", label="global")

axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)

axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")
axes[1, 0].set_ylabel("$\Delta$SAT in K")

axes[1, 0].set_title("Northern Hemisphere $-$ dQ minus no-dQ")
axes[1, 1].set_title("Southern Hemisphere $-$ dQ minus no-dQ")

axes[1, 0].text(0, -8.8, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(0, -8.8, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")


axes[0, 1].set_yticklabels([])
axes[1, 1].set_yticklabels([])
axes[0, 0].set_ylim(-5.2, 0.3)
axes[0, 1].set_ylim(-5.2, 0.3)
# axes[1, 0].set_ylim(-5.2, 0.3)
# axes[1, 1].set_ylim(-5.2, 0.3)
axes[1, 0].set_ylim(-9.5, 1)
axes[1, 1].set_ylim(-9.5, 1)

# axes[1, 0].set_xlim((-7.5, 157.5))

# fig.suptitle("G2$-$G1 difference in SAT anomaly")
fig.subplots_adjust(wspace=0.05, hspace=0.3)

pl.savefig(pl_path + "/PDF/MultiRegion_dTas_G1_G2_CESM2_SOM_Diff.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/MultiRegion_dTas_G1_G2_CESM2_SOM_Diff.png", bbox_inches="tight", dpi=250)
pl.show()
pl.close()


#%% plot the p-values of the inter-group differences in radiative flux in one panel
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=True, sharex=False)

axes[0, 0].plot(dtas_rg_ttest[1].pvalue, c="blue", label="R>75N")
axes[0, 0].plot(dtas_rg_ttest[2].pvalue, c="orange", label="R>60N")
axes[0, 0].plot(dtas_rg_ttest[3].pvalue, c="red", label="R>30N")
axes[0, 0].plot(dtas_rg_ttest[4].pvalue, c="violet", label="R<30N")
axes[0, 0].plot(dtas_rg_ttest[5].pvalue, c="gray", label="R>0N")

axes[0, 1].plot(dtas_rg_ttest[10].pvalue, c="blue", linestyle="-", label="R>75S")
axes[0, 1].plot(dtas_rg_ttest[9].pvalue, c="orange", linestyle="-", label="R>60S")
axes[0, 1].plot(dtas_rg_ttest[8].pvalue, c="red", linestyle="-", label="R>30S")
axes[0, 1].plot(dtas_rg_ttest[7].pvalue, c="violet", linestyle="-", label="R<30S")
axes[0, 1].plot(dtas_rg_ttest[6].pvalue, c="gray", linestyle="-", label="R>0S")

axes[0, 0].plot(dtas_gl_ttest.pvalue, c="black", linestyle="-", label="global")
axes[0, 1].plot(dtas_gl_ttest.pvalue, c="black", linestyle="-", label="global")

axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)

axes[0, 0].axhline(y=0.05, c="red", linewidth=1, linestyle="--")
axes[0, 1].axhline(y=0.05, c="red", linewidth=1, linestyle="--")

axes[0, 0].legend(ncol=2, loc="upper center")
axes[0, 1].legend(ncol=2, loc="upper center")

# axes[0, 0].set_xlabel("Year of simulation")
# axes[0, 1].set_xlabel("Year of simulation")
axes[0, 0].set_ylabel("$p$ value")

axes[0, 0].set_title("Northern Hemisphere $-$ G2 minus G1")
axes[0, 1].set_title("Southern Hemisphere $-$ G2 minus G1")

# axes[0, 1].set_yticklabels([])
# axes[1, 1].set_yticklabels([])
# axes[0, 0].set_ylim((-0.05, 0.5))
axes[0, 1].set_ylim((-0.05, 0.5))

axes[0, 0].text(140, 0.47, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(140, 0.47, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")

# lower panels
axes[1, 0].plot(dtas_dq_ttest["Arctic"].pvalue, c="blue", label="R>75N")
axes[1, 0].plot(dtas_dq_ttest["NP"].pvalue, c="orange", label="R>60N")
axes[1, 0].plot(dtas_dq_ttest["NExT"].pvalue, c="red", label="R>30N")
axes[1, 0].plot(dtas_dq_ttest["NT"].pvalue, c="violet", label="R<30N")
axes[1, 0].plot(dtas_dq_ttest["NH"].pvalue, c="gray", label="R>0N")
axes[1, 0].plot(dtas_dq_ttest["Global"].pvalue, c="black", label="global")

axes[1, 1].plot(dtas_dq_ttest["Antarctica"].pvalue, c="blue", label="R>75S")
axes[1, 1].plot(dtas_dq_ttest["SP"].pvalue, c="orange", label="R>60S")
axes[1, 1].plot(dtas_dq_ttest["SExT"].pvalue, c="red", label="R>30S")
axes[1, 1].plot(dtas_dq_ttest["ST"].pvalue, c="violet", label="R<30S")
axes[1, 1].plot(dtas_dq_ttest["SH"].pvalue, c="gray", label="R>0S")
axes[1, 1].plot(dtas_dq_ttest["Global"].pvalue, c="black", label="global")

axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)

axes[1, 0].axhline(y=0.05, c="red", linewidth=1, linestyle="--")
axes[1, 1].axhline(y=0.05, c="red", linewidth=1, linestyle="--")

# axes[1, 0].legend(ncol=2, loc="upper center")

axes[1, 0].set_ylabel("$p$ value")
axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")

axes[1, 0].set_title("Northern Hemisphere $-$ dQ minus no-dQ")
axes[1, 1].set_title("Southern Hemisphere $-$ dQ minus no-dQ")

axes[1, 0].text(36.5, 0.47, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(36.5, 0.47, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.05, hspace=0.3)

pl.savefig(pl_path + "/PDF/MultiRegion_dTas_G1_G2_CESM2_SOM_Diff_p_values.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/MultiRegion_dTas_G1_G2_CESM2_SOM_Diff_p_values.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% only the CESM2 case comparison

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3.25), sharey=True, sharex=False)

axes[0].plot(avdtas_dq_dic["Arctic"] - avdtas_dic["Arctic"], c="blue", label="R>75N")
axes[0].plot(avdtas_dq_dic["NP"] - avdtas_dic["NP"], c="orange", label="R>60N")
axes[0].plot(avdtas_dq_dic["NExT"] - avdtas_dic["NExT"], c="red", label="R>30N")
axes[0].plot(avdtas_dq_dic["NT"] - avdtas_dic["NT"], c="violet", label="R<30N")
axes[0].plot(avdtas_dq_dic["NH"] - avdtas_dic["NH"], c="gray", label="R>0N")

axes[1].plot(avdtas_dq_dic["Antarctica"] - avdtas_dic["Antarctica"], c="blue", label="R>75S")
axes[1].plot(avdtas_dq_dic["SP"] - avdtas_dic["SP"], c="orange", label="R>60S")
axes[1].plot(avdtas_dq_dic["SExT"] - avdtas_dic["SExT"], c="red", label="R>30S")
axes[1].plot(avdtas_dq_dic["ST"] - avdtas_dic["ST"], c="violet", label="R<30S")
axes[1].plot(avdtas_dq_dic["SH"] - avdtas_dic["SH"], c="gray", label="R>0S")

axes[0].plot(avdtas_dq_dic["Global"] - avdtas_dic["Global"], c="black", linestyle="-", label="global")
axes[1].plot(avdtas_dq_dic["Global"] - avdtas_dic["Global"], c="black", linestyle="-", label="global")

axes[0].legend(ncol=1, loc="lower right")
axes[1].legend(ncol=2, loc="lower right")

axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[1].axhline(y=0, c="black", linewidth=0.5)

axes[0].set_xlabel("Year of simulation")
axes[1].set_xlabel("Year of simulation")
axes[0].set_ylabel("$\Delta$SAT in K")

axes[0].set_title("Northern Hemisphere")
axes[1].set_title("Southern Hemisphere")

# axes[0].text(36.5, 0.47, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
# axes[1].text(36.5, 0.47, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")
fig.suptitle("dQ $-$ no-dQ difference in surface-air temperature")

fig.subplots_adjust(wspace=0.05, hspace=0.3, top=0.83)

pl.savefig(pl_path + "/PDF/MultiRegion_dTas_CESM2_SOM_Diff.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/MultiRegion_dTas_CESM2_SOM_Diff.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
