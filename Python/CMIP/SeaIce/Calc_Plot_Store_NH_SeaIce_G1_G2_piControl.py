"""
Calculate, plot (several comparisons), and store the piControl sea-ice extent/area for model groups G1 and G2 (for 
details see our paper). Generates a plot looking similar to the panels of Fig. 6 in Eiselt and Graversen (2022), JCLI, 
but just for the piControl case.

NOTE: GISS-E2-H seems to have unreasonable values so for now exclude it! Because the areacello grid is unreasonable.

Be sure to set data_dir5 and data_dir6 as well as possibly change data_path.
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


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Extract_Geographic_Region import extract_region
from Functions.Func_Region_Mean import region_mean
from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_SeasonalMean import sea_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups
models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", 
           "INM-CM4-8"]


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models


#%% loop over group one
si_ar_nh_gr = {}
si_ex_nh_gr = {}
si_ar_sh_gr = {}
si_ex_sh_gr = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    si_ar_nh = np.zeros((150, len(group)))
    si_ex_nh = np.zeros((150, len(group)))
    si_ar_sh = np.zeros((150, len(group)))
    si_ex_sh = np.zeros((150, len(group)))
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
        # end if
        
        # data path and name
        data_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        
        # load the amoc file
        sia_nc = Dataset(glob.glob(data_path + "/si_ext_area_*piControl*.nc")[0])
        sie_nc = Dataset(glob.glob(data_path + "/si_extent_*piControl*.nc")[0])
        
        
        # load the data    
        si_ar_nh[:, i] = an_mean(sia_nc.variables["si_area_n"][:150*12])
        si_ar_sh[:, i] = an_mean(sia_nc.variables["si_area_s"][:150*12])
        si_ex_nh[:, i] = an_mean(sie_nc.variables["si_ext_n"][:150*12])
        si_ex_sh[:, i] = an_mean(sie_nc.variables["si_ext_s"][:150*12])
        
    # end for i, mod_d
    
    si_ar_nh_gr[g_n] = si_ar_nh / 1E12
    si_ar_sh_gr[g_n] = si_ar_sh / 1E12
    si_ex_nh_gr[g_n] = si_ex_nh / 1E12
    si_ex_sh_gr[g_n] = si_ex_sh / 1E12
    
# end for g_n, group


#%% calculate the means over G1 and over G2
si_ar_nh_ga = {}
si_ar_sh_ga = {}
si_ex_nh_ga = {}
si_ex_sh_ga = {}

for gr in ["G1", "G2"]:
    si_ar_nh_ga[gr] = {}
    si_ar_nh_ga[gr]["mean"] = np.mean(si_ar_nh_gr[gr], axis=1)
    si_ar_nh_ga[gr]["std"] = np.std(si_ar_nh_gr[gr], axis=1)

    si_ar_sh_ga[gr] = {}
    si_ar_sh_ga[gr]["mean"] = np.mean(si_ar_sh_gr[gr], axis=1)
    si_ar_sh_ga[gr]["std"] = np.std(si_ar_sh_gr[gr], axis=1)
    
    si_ex_nh_ga[gr] = {}
    si_ex_nh_ga[gr]["mean"] = np.mean(si_ex_nh_gr[gr], axis=1)
    si_ex_nh_ga[gr]["std"] = np.std(si_ex_nh_gr[gr], axis=1)

    si_ex_sh_ga[gr] = {}
    si_ex_sh_ga[gr]["mean"] = np.mean(si_ex_sh_gr[gr], axis=1)
    si_ex_sh_ga[gr]["std"] = np.std(si_ex_sh_gr[gr], axis=1)
    
# end for gr


#%% try the t-test to compare the means
n_ttest_ar = ttest_ind(si_ar_nh_gr["G1"], si_ar_nh_gr["G2"], axis=-1)
s_ttest_ar = ttest_ind(si_ar_nh_gr["G1"], si_ar_sh_gr["G2"], axis=-1)
n_ttest_ex = ttest_ind(si_ex_nh_gr["G1"], si_ex_nh_gr["G2"], axis=-1)
s_ttest_ex = ttest_ind(si_ex_nh_gr["G1"], si_ex_sh_gr["G2"], axis=-1)


#%% plot the time series - NH AREA
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(si_ar_nh_ga["G1"]["mean"], c="blue", label="G1")
axes.fill_between(np.arange(len(si_ar_nh_ga["G1"]["mean"])), si_ar_nh_ga["G1"]["mean"] - si_ar_nh_ga["G1"]["std"], 
                  si_ar_nh_ga["G1"]["mean"] + si_ar_nh_ga["G1"]["std"], facecolor="blue", alpha=0.25)
axes.plot(si_ar_nh_ga["G2"]["mean"], c="red", label="G2")
axes.fill_between(np.arange(len(si_ar_nh_ga["G2"]["mean"])), si_ar_nh_ga["G2"]["mean"] - si_ar_nh_ga["G2"]["std"], 
                  si_ar_nh_ga["G2"]["mean"] + si_ar_nh_ga["G2"]["std"], facecolor="red", alpha=0.25)
axr.plot(n_ttest_ar.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title("Northern hemispheric sea ice area\nfor G1 and G2 piControl", fontsize=fsz+1)
axes.set_xlabel("Years", fontsize=fsz)
axes.set_ylabel("Sea ice area 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

pl.show()
pl.close()


#%% plot the time series - NH EXTENT
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(si_ex_nh_ga["G1"]["mean"], c="blue", label="G1")
axes.fill_between(np.arange(len(si_ex_nh_ga["G1"]["mean"])), si_ex_nh_ga["G1"]["mean"] - si_ex_nh_ga["G1"]["std"], 
                  si_ex_nh_ga["G1"]["mean"] + si_ex_nh_ga["G1"]["std"], facecolor="blue", alpha=0.25)
axes.plot(si_ex_nh_ga["G2"]["mean"], c="red", label="G2")
axes.fill_between(np.arange(len(si_ex_nh_ga["G2"]["mean"])), si_ex_nh_ga["G2"]["mean"] - si_ex_nh_ga["G2"]["std"], 
                  si_ex_nh_ga["G2"]["mean"] + si_ex_nh_ga["G2"]["std"], facecolor="red", alpha=0.25)
axr.plot(n_ttest_ex.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title("Northern hemispheric sea ice extent\nfor G1 and G2 piControl", fontsize=fsz+1)
axes.set_xlabel("Years", fontsize=fsz)
axes.set_ylabel("Sea ice extent 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

pl.show()
pl.close()


#%% plot the time series - SH AREA
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(si_ar_sh_ga["G1"]["mean"], c="blue", label="G1")
axes.fill_between(np.arange(len(si_ar_sh_ga["G1"]["mean"])), si_ar_sh_ga["G1"]["mean"] - si_ar_sh_ga["G1"]["std"], 
                  si_ar_sh_ga["G1"]["mean"] + si_ar_sh_ga["G1"]["std"], facecolor="blue", alpha=0.25)
axes.plot(si_ar_sh_ga["G2"]["mean"], c="red", label="G2")
axes.fill_between(np.arange(len(si_ar_sh_ga["G2"]["mean"])), si_ar_sh_ga["G2"]["mean"] - si_ar_sh_ga["G2"]["std"], 
                  si_ar_sh_ga["G2"]["mean"] + si_ar_sh_ga["G2"]["std"], facecolor="red", alpha=0.25)
axr.plot(s_ttest_ar.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title("Southern hemispheric sea ice area\nfor G1 and G2 piControl", fontsize=fsz+1)
axes.set_xlabel("Years", fontsize=fsz)
axes.set_ylabel("Sea ice area 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

pl.show()
pl.close()


#%% plot the time series - SH EXTENT
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(si_ex_sh_ga["G1"]["mean"], c="blue", label="G1")
axes.fill_between(np.arange(len(si_ex_sh_ga["G1"]["mean"])), si_ex_sh_ga["G1"]["mean"] - si_ex_sh_ga["G1"]["std"], 
                  si_ex_sh_ga["G1"]["mean"] + si_ex_sh_ga["G1"]["std"], facecolor="blue", alpha=0.25)
axes.plot(si_ex_sh_ga["G2"]["mean"], c="red", label="G2")
axes.fill_between(np.arange(len(si_ex_sh_ga["G2"]["mean"])), si_ex_sh_ga["G2"]["mean"] - si_ex_sh_ga["G2"]["std"], 
                  si_ex_sh_ga["G2"]["mean"] + si_ex_sh_ga["G2"]["std"], facecolor="red", alpha=0.25)
axr.plot(s_ttest_ex.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title("Southern hemispheric sea ice extent\nfor G1 and G2 piControl", fontsize=fsz+1)
axes.set_xlabel("Years", fontsize=fsz)
axes.set_ylabel("Sea ice extent 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

pl.show()
pl.close()


#%% store these data as netcdf file
f = Dataset(out_path + f"G1_G2_NH_SeaIce{cslt_fadd}_piC.nc", "w", format="NETCDF4")

# create the dimensions
f.createDimension("years", n_yrs)
f.createDimension("mods_G1", len(sian_an_d["G1"]))
f.createDimension("mods_G2", len(sian_an_d["G2"]))

# create the variables
mods_g1_nc = f.createVariable("mods_G1", "S25", "mods_G1")
mods_g2_nc = f.createVariable("mods_G2", "S25", "mods_G2")
sia_g1_nc = f.createVariable("sia_G1", "f4", ("mods_G1", "years"))
sia_g2_nc = f.createVariable("sia_G2", "f4", ("mods_G2", "years"))

p_g1_g2_nc = f.createVariable("p_G1_G2", "f4", "years")

# pass the data into the variables
mods_g1_nc[:] = np.array(mods_d["G1"])
mods_g2_nc[:] = np.array(mods_d["G2"])

sia_g1_nc[:] = sian_an_d["G1"]
sia_g2_nc[:] = sian_an_d["G2"]

p_g1_g2_nc[:] = ttest.pvalue

# set units of the variables
sia_g1_nc.units = "m^2"

# descriptions of the variables
mods_g1_nc.description = "names of the model in G1 (weak lapse rate feedback change)"
mods_g2_nc.description = "names of the model in G2 (weak lapse rate feedback change)"

sia_g1_nc.description = "northern hemispheric mean sea ice area G1 piControl"
sia_g2_nc.description = "northern hemispheric mean sea ice area G2 piControl"

p_g1_g2_nc.description = "p-value for group mean difference Wald t-test between G1 and G2"

# date of file creation
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the file
f.close()



