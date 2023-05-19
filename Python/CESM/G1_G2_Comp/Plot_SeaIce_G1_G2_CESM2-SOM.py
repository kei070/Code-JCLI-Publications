"""
Generates Fig. 5 in Eiselt and Graversen (2023), JCLI.

Be sure to set data_dir5 and data_dir6 correctly as well as data_path and pl_path.
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
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "FGOALS-g3",
           "EC-Earth3", "INM-CM4-8"]
mod_dic = {"G1":models1, "G2":models2}


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""
data_dir6 = ""

cmip5_list = nl5.models
cmip6_list = nl6.models


#%% loop over the groups and collect their data to compare later
n_yrs = 150

# set up dictionaries to collect the data
sien_an_d = dict()
sian_an_d = dict()
sias_an_d = dict()
sian_mo_d = dict()
sias_mo_d = dict()

mods_d = dict()

pl_add = ""
tl_add = ""

for g12, group in zip(["G1", "G2"], [models1, models2]):
    
    # set up a counter for the models
    mod_count = 0
    
    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    sien_an_l = []
    sian_an_l = []
    sias_an_l = []
    sian_mo_l = []
    sias_mo_l = []
    
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

        si_nc = Dataset(glob.glob(data_dir + f"/CMIP{cmip}/Data/{mod_d}/si_ext_area_*{a4x}*.nc")[0])
              
        # load the values
        sian_an_l.append(an_mean(si_nc.variables["si_area_n"][:])[:n_yrs])
        sias_an_l.append(an_mean(si_nc.variables["si_area_s"][:])[:n_yrs])
        sian_mo_l.append(si_nc.variables["si_area_n"][:n_yrs*12])
        sias_mo_l.append(si_nc.variables["si_area_s"][:n_yrs*12])
        
    # end for mod
    
    # sien_an_d[g12] = np.array(sien_an_l)
    sian_an_d[g12] = np.array(sian_an_l)
    sias_an_d[g12] = np.array(sias_an_l)
    sian_mo_d[g12] = np.array(sian_mo_l)
    sias_mo_d[g12] = np.array(sias_mo_l)
    
# end for g12


#%% calculate the means over the model groups
avg = "mean"
quant = 0.75

if avg == "mean":
    g1_avg = np.mean(sian_an_d["G1"] / 1E12, axis=0)
    g1_lo = np.std(sian_an_d["G1"] / 1E12, axis=0)
    g1_hi = np.std(sian_an_d["G1"] / 1E12, axis=0)
    g2_avg = np.mean(sian_an_d["G2"] / 1E12, axis=0)
    g2_lo = np.std(sian_an_d["G2"] / 1E12, axis=0)
    g2_hi = np.std(sian_an_d["G2"] / 1E12, axis=0)
    
    g1s_avg = np.mean(sias_an_d["G1"] / 1E12, axis=0)
    g1s_lo = np.std(sias_an_d["G1"] / 1E12, axis=0)
    g1s_hi = np.std(sias_an_d["G1"] / 1E12, axis=0)
    g2s_avg = np.mean(sias_an_d["G2"] / 1E12, axis=0)
    g2s_lo = np.std(sias_an_d["G2"] / 1E12, axis=0)
    g2s_hi = np.std(sias_an_d["G2"] / 1E12, axis=0)    
    
    g1_avg_mo = np.mean(sian_mo_d["G1"] / 1E12, axis=0)
    g1_lo_mo = np.std(sian_mo_d["G1"] / 1E12, axis=0)
    g1_hi_mo = np.std(sian_mo_d["G1"] / 1E12, axis=0)
    g2_avg_mo = np.mean(sian_mo_d["G2"] / 1E12, axis=0)
    g2_lo_mo = np.std(sian_mo_d["G2"] / 1E12, axis=0)
    g2_hi_mo = np.std(sian_mo_d["G2"] / 1E12, axis=0)
    
    g1s_avg_mo = np.mean(sias_mo_d["G1"] / 1E12, axis=0)
    g1s_lo_mo = np.std(sias_mo_d["G1"] / 1E12, axis=0)
    g1s_hi_mo = np.std(sias_mo_d["G1"] / 1E12, axis=0)
    g2s_avg_mo = np.mean(sias_mo_d["G2"] / 1E12, axis=0)
    g2s_lo_mo = np.std(sias_mo_d["G2"] / 1E12, axis=0)
    g2s_hi_mo = np.std(sias_mo_d["G2"] / 1E12, axis=0)        
elif avg == "median":
    g1_avg = np.median(sian_an_d["G1"] / 1E12, axis=0)
    g1_lo = np.median(sian_an_d["G1"] / 1E12, axis=0) - np.quantile(sian_an_d["G1"] / 1E12, 1-quant, axis=0)
    g1_hi = np.quantile(sian_an_d["G1"] / 1E12, quant, axis=0) - np.median(sian_an_d["G1"] / 1E12, axis=0)
    g2_avg = np.median(sian_an_d["G2"] / 1E12, axis=0)
    g2_lo = np.median(sian_an_d["G2"] / 1E12, axis=0) - np.quantile(sian_an_d["G2"] / 1E12, 1-quant, axis=0)
    g2_hi = np.quantile(sian_an_d["G2"] / 1E12, quant, axis=0) - np.median(sian_an_d["G2"] / 1E12, axis=0)
# end if elif  


#%% calculate the p value of the differences of the group means
g1g2_nh_ttest = ttest_ind(sian_an_d["G1"], sian_an_d["G2"], axis=0)
g1g2_sh_ttest = ttest_ind(sias_an_d["G1"], sias_an_d["G2"], axis=0)


#%% calculate seasonal means for the G1 and G2 means
g1_avg_se = sea_mean(g1_avg_mo)
g2_avg_se = sea_mean(g2_avg_mo)
g1s_avg_se = sea_mean(g1s_avg_mo)
g2s_avg_se = sea_mean(g2s_avg_mo)


#%% LOAD CESM2-SOM data: setup case name
case_4x51 = "Proj2_KUE_4xCO2" #  "dQ05yr_4xCO2"
case_4x41 = "Yr41_4xCO2"
case_4x61 = "Proj2_KUE_4xCO2_61"

case_dq4x51 = "dQ01yr_4xCO2"
case_dq4x41 = "Y41_dQ01"
case_dq4x61 = "Y61_dQ01"

# case_dq4x51 = "dQ01_r30_4xCO2"
# case_dq4x41 = "dQ01_r30_4xCO2_41"
# case_dq4x61 = "dQ01_r30_4xCO2_61"


#%% set paths
data_path = ""
pl_path = ""

os.makedirs(pl_path + "PDF/", exist_ok=True)
os.makedirs(pl_path + "PNG/", exist_ok=True)


#%% load nc files
nc_4x51 = Dataset(glob.glob(data_path + case_4x51 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_4x41 = Dataset(glob.glob(data_path + case_4x41 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_4x61 = Dataset(glob.glob(data_path + case_4x61 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x51 = Dataset(glob.glob(data_path + case_dq4x51 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x41 = Dataset(glob.glob(data_path + case_dq4x41 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x61 = Dataset(glob.glob(data_path + case_dq4x61 + "/SeaIce_Area_NH_SH_*.nc")[0])


#%% load the data
nh_4x51 = da.ma.masked_array(nc_4x51.variables["si_area_n"], lock=True).compute()
nh_4x41 = da.ma.masked_array(nc_4x41.variables["si_area_n"], lock=True).compute()
nh_4x61 = da.ma.masked_array(nc_4x61.variables["si_area_n"], lock=True).compute()

nh_dq4x51 = da.ma.masked_array(nc_dq4x51.variables["si_area_n"], lock=True).compute()
nh_dq4x41 = da.ma.masked_array(nc_dq4x41.variables["si_area_n"], lock=True).compute()
nh_dq4x61 = da.ma.masked_array(nc_dq4x61.variables["si_area_n"], lock=True).compute()

sh_4x51 = da.ma.masked_array(nc_4x51.variables["si_area_s"], lock=True).compute()
sh_4x41 = da.ma.masked_array(nc_4x41.variables["si_area_s"], lock=True).compute()
sh_4x61 = da.ma.masked_array(nc_4x61.variables["si_area_s"], lock=True).compute()

sh_dq4x51 = da.ma.masked_array(nc_dq4x51.variables["si_area_s"], lock=True).compute()
sh_dq4x41 = da.ma.masked_array(nc_dq4x41.variables["si_area_s"], lock=True).compute()
sh_dq4x61 = da.ma.masked_array(nc_dq4x61.variables["si_area_s"], lock=True).compute()


#%% calculate annual means
nh_4x51_an = an_mean(nh_4x51)
nh_4x41_an = an_mean(nh_4x41)
nh_4x61_an = an_mean(nh_4x61)

nh_dq4x51_an = an_mean(nh_dq4x51)
nh_dq4x41_an = an_mean(nh_dq4x41)
nh_dq4x61_an = an_mean(nh_dq4x61)

sh_4x51_an = an_mean(sh_4x51)
sh_4x41_an = an_mean(sh_4x41)
sh_4x61_an = an_mean(sh_4x61)

sh_dq4x51_an = an_mean(sh_dq4x51)
sh_dq4x41_an = an_mean(sh_dq4x41)
sh_dq4x61_an = an_mean(sh_dq4x61)


#%% stack the ensembles
tlen = 40
nh_4x_ens = np.stack([nh_4x51_an[:tlen], nh_4x41_an[:tlen], nh_4x61_an[:tlen]], axis=0)
nh_dq4x_ens = np.stack([nh_dq4x51_an[:tlen], nh_dq4x41_an[:tlen], nh_dq4x61_an[:tlen]], axis=0)
sh_4x_ens = np.stack([sh_4x51_an[:tlen], sh_4x41_an[:tlen], sh_4x61_an[:tlen]], axis=0)
sh_dq4x_ens = np.stack([sh_dq4x51_an[:tlen], sh_dq4x41_an[:tlen], sh_dq4x61_an[:tlen]], axis=0)


#%% calculate ensemble mean
nh_4x_an = np.mean(nh_4x_ens, axis=0)
nh_dq4x_an = np.mean(nh_dq4x_ens, axis=0)
nh_4x_an_std = np.std(nh_4x_ens, axis=0)
nh_dq4x_an_std = np.std(nh_dq4x_ens, axis=0)

sh_4x_an = np.mean(sh_4x_ens, axis=0)
sh_dq4x_an = np.mean(sh_dq4x_ens, axis=0)
sh_4x_an_std = np.std(sh_4x_ens, axis=0)
sh_dq4x_an_std = np.std(sh_dq4x_ens, axis=0)


#%% calculate the p value of the differences of the ensemble means
dq_nh_ttest = ttest_ind(nh_4x_ens, nh_dq4x_ens, axis=0)
dq_sh_ttest = ttest_ind(sh_4x_ens, sh_dq4x_ens, axis=0)


#%% compare seasonal means NH
"""
fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=True, sharey=True)

axes[0, 0].plot(g1_avg_se["DJF"], c="blue", label="G1")
axes[0, 0].plot(g2_avg_se["DJF"], c="red", label="G2")
axes[0, 0].plot(g1_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[0, 0].plot(g2_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[0, 0].axhline(y=0, c="gray", linewidth=0.5)
axes[0, 0].legend()
axes[0, 0].set_title("DJF")
axes[0, 0].set_ylabel("Sea-ice area in 10$^6$ km$^2$")

axes[0, 1].plot(g1_avg_se["MAM"], c="blue", label="G1")
axes[0, 1].plot(g2_avg_se["MAM"], c="red", label="G2")
axes[0, 1].plot(g1_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[0, 1].plot(g2_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[0, 1].axhline(y=0, c="gray", linewidth=0.5)
axes[0, 1].set_title("MAM")

axes[1, 0].plot(g1_avg_se["JJA"], c="blue", label="G1")
axes[1, 0].plot(g2_avg_se["JJA"], c="red", label="G2")
axes[1, 0].plot(g1_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[1, 0].plot(g2_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[1, 0].axhline(y=0, c="gray", linewidth=0.5)
axes[1, 0].set_title("JJA")
axes[1, 0].set_xlabel("Years since 4xCO$_2$")
axes[1, 0].set_ylabel("Sea-ice area in 10$^6$ km$^2$")

axes[1, 1].plot(g1_avg_se["SON"], c="blue", label="G1")
axes[1, 1].plot(g2_avg_se["SON"], c="red", label="G2")
axes[1, 1].plot(g1_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[1, 1].plot(g2_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[1, 1].axhline(y=0, c="gray", linewidth=0.5)
axes[1, 1].set_title("SON")
axes[1, 1].set_xlabel("Years since 4xCO$_2$")

fig.suptitle("G1$-$G2 NH sea-ice area", fontsize=15)
fig.subplots_adjust(wspace=0.1, top=0.925)

pl.show()
pl.close()


#%% compare seasonal means SH
fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=True, sharey=True)

axes[0, 0].plot(g1s_avg_se["DJF"], c="blue", label="G1")
axes[0, 0].plot(g2s_avg_se["DJF"], c="red", label="G2")
axes[0, 0].plot(g1s_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[0, 0].plot(g2s_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[0, 0].axhline(y=0, c="gray", linewidth=0.5)
axes[0, 0].legend()
axes[0, 0].set_title("DJF")
axes[0, 0].set_ylabel("Sea-ice area in 10$^6$ km$^2$")

axes[0, 1].plot(g1s_avg_se["MAM"], c="blue", label="G1")
axes[0, 1].plot(g2s_avg_se["MAM"], c="red", label="G2")
axes[0, 1].plot(g1s_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[0, 1].plot(g2s_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[0, 1].axhline(y=0, c="gray", linewidth=0.5)
axes[0, 1].set_title("MAM")

axes[1, 0].plot(g1s_avg_se["JJA"], c="blue", label="G1")
axes[1, 0].plot(g2s_avg_se["JJA"], c="red", label="G2")
axes[1, 0].plot(g1s_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[1, 0].plot(g2s_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[1, 0].axhline(y=0, c="gray", linewidth=0.5)
axes[1, 0].set_title("JJA")
axes[1, 0].set_xlabel("Years since 4xCO$_2$")
axes[1, 0].set_ylabel("Sea-ice area in 10$^6$ km$^2$")

axes[1, 1].plot(g1s_avg_se["SON"], c="blue", label="G1")
axes[1, 1].plot(g2s_avg_se["SON"], c="red", label="G2")
axes[1, 1].plot(g1s_avg, c="blue", linestyle="--", linewidth=0.65, label="G1 annual")
axes[1, 1].plot(g2s_avg, c="red", linestyle="--", linewidth=0.65, label="G2 annual")
axes[1, 1].axhline(y=0, c="gray", linewidth=0.5)
axes[1, 1].set_title("SON")
axes[1, 1].set_xlabel("Years since 4xCO$_2$")

fig.suptitle("G1$-$G2 SH sea-ice area", fontsize=15)
fig.subplots_adjust(wspace=0.1, top=0.925)

pl.show()
pl.close()


#%% try the t-test to compare the means
ttest = ttest_ind(sian_an_d["G1"], sian_an_d["G2"], axis=0)
ttests = ttest_ind(sias_an_d["G1"], sias_an_d["G2"], axis=0)


#%% plot the time series - NH
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(g1_avg, c="blue", label="G1")
# for i in np.arange(np.shape(sian_an_d["G1"])[0]):
#     axes.plot(sian_an_d["G1"][i, :] / 1E12, c="blue", linewidth=0.25)
# end for i
axes.fill_between(np.arange(len(g1_avg)), g1_avg - g1_lo, g1_avg + g1_hi, facecolor="blue", alpha=0.25)
axes.plot(g2_avg, c="red", label="G2")
axes.axhline(y=0, c="black", linewidth=0.5)
axes.set_ylim((-0.5, 13))
# for i in np.arange(np.shape(sian_an_d["G2"])[0]):
#     axes.plot(sian_an_d["G2"][i, :] / 1E12, c="red", linewidth=0.25)
# end for i
axes.fill_between(np.arange(len(g2_avg)), g2_avg - g2_lo, g2_avg + g2_hi, facecolor="red", alpha=0.25)

axr.plot(ttest.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title(f"Northern hemispheric sea ice area\nfor G1 and G2{tl_add}", fontsize=fsz+1)
axes.set_xlabel("Years since branching of abrupt-4xCO2", fontsize=fsz)
axes.set_ylabel("Sea ice area 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

# pl.savefig(pl_path + f"/PNG/SIA_NH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/SIA_NH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - SH
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(g1s_avg, c="blue", label="G1")
# for i in np.arange(np.shape(sian_an_d["G1"])[0]):
#     axes.plot(sian_an_d["G1"][i, :] / 1E12, c="blue", linewidth=0.25)
# end for i
axes.fill_between(np.arange(len(g1s_avg)), g1s_avg - g1s_lo, g1s_avg + g1s_hi, facecolor="blue", alpha=0.25)
axes.plot(g2s_avg, c="red", label="G2")
axes.axhline(y=0, c="black", linewidth=0.5)
axes.set_ylim((-0.5, 13))

# for i in np.arange(np.shape(sian_an_d["G2"])[0]):
#     axes.plot(sian_an_d["G2"][i, :] / 1E12, c="red", linewidth=0.25)
# end for i
axes.fill_between(np.arange(len(g2s_avg)), g2s_avg - g2s_lo, g2s_avg + g2s_hi, facecolor="red", alpha=0.25)

axr.plot(ttests.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=0.5, c="gray")
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc="upper center", ncol=3)
axes.set_title(f"Southern hemispheric sea ice area\nfor G1 and G2{tl_add}", fontsize=fsz+1)
axes.set_xlabel("Years since branching of abrupt-4xCO2", fontsize=fsz)
axes.set_ylabel("Sea ice area 10$^6$ km$^2$", fontsize=fsz)
axes.tick_params(labelsize=fsz)
axr.tick_params(labelsize=fsz)

# pl.savefig(pl_path + f"/PNG/SIA_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/SIA_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
"""

#%% NH and SH combined
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=True, sharex=False)

axes[0, 0].plot(g1_avg, c="blue", label="G1")
axes[0, 0].plot(g2_avg, c="red", label="G2")
axes[0, 0].fill_between(np.arange(len(g1_avg)), g1_avg - g1_lo, g1_avg + g1_hi, facecolor="blue", alpha=0.15)
axes[0, 0].fill_between(np.arange(len(g2_avg)), g2_avg - g2_lo, g2_avg + g2_hi, facecolor="red", alpha=0.15)
axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 0].legend()

axes[0, 1].plot(g1s_avg, c="blue", label="G1")
axes[0, 1].plot(g2s_avg, c="red", label="G2")
axes[0, 1].fill_between(np.arange(len(g1s_avg)), g1s_avg - g1s_lo, g1s_avg + g1s_hi, facecolor="blue", alpha=0.15)
axes[0, 1].fill_between(np.arange(len(g2s_avg)), g2s_avg - g2s_lo, g2s_avg + g2s_hi, facecolor="red", alpha=0.15)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].legend()

# axes[0, 0].set_xlabel("Year of simulation")
# axes[0, 1].set_xlabel("Year of simulation")
axes[0, 0].set_ylabel("Sea-ice area 10$^6$ km$^2$")

axes[0, 0].set_title("Northern Hemisphere $-$ G1 and G2")
axes[0, 1].set_title("Southern Hemisphere $-$ G1 and G2")

axes[1, 0].plot(nh_4x_an[:tlen] / 1e6, c="blue", label="no-dQ")
axes[1, 0].plot(nh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1, 0].fill_between(np.arange(tlen), (nh_4x_an[:tlen] + nh_4x_an_std[:tlen]) / 1e6, 
                        (nh_4x_an[:tlen] - nh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1, 0].fill_between(np.arange(tlen), (nh_dq4x_an[:tlen] + nh_dq4x_an_std[:tlen]) / 1e6, 
                        (nh_dq4x_an[:tlen] - nh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 0].legend()

axes[1, 1].plot(sh_4x_an[:tlen] / 1e6, c="blue", label="no-dQ")
axes[1, 1].plot(sh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1, 1].fill_between(np.arange(tlen), (sh_4x_an[:tlen] + sh_4x_an_std[:tlen]) / 1e6, 
                        (sh_4x_an[:tlen] - sh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1, 1].fill_between(np.arange(tlen), (sh_dq4x_an[:tlen] + sh_dq4x_an_std[:tlen]) / 1e6, 
                        (sh_dq4x_an[:tlen] - sh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].legend()

axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")
axes[1, 0].set_ylabel("Sea-ice area 10$^6$ km$^2$")
axes[1, 0].set_title("Northern Hemisphere $-$ CESM2-SOM")
axes[1, 1].set_title("Southern Hemisphere $-$ CESM2-SOM")


# fig.suptitle("G1 and G2 sea-ice area")
fig.subplots_adjust(wspace=0.05, hspace=0.3)

axes[1, 0].set_xlim((-2.5, 40.5))
axes[1, 1].set_xlim((-2.5, 40.5))

axes[0, 0].text(0, 1, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(0, 1, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 0].text(0, 1, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(0, 1, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

pl.savefig(pl_path + f"/PNG/SIA_NH_SH_AnMean_G1_G2_and_CESM2_SOM_{case_dq4x51}_{avg}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/SIA_NH_SH_AnMean_G1_G2_and_CESM2_SOM_{case_dq4x51}_{avg}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% NH and SH combined WITH p value
p_lo_lim = -0.02

fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=False, sharex=False)

ax00 = axes[0, 0].twinx()
ax01 = axes[0, 1].twinx()
ax10 = axes[1, 0].twinx()
ax11 = axes[1, 1].twinx()

axes[0, 0].plot(g1_avg, c="blue", label="G1")
axes[0, 0].plot(g2_avg, c="red", label="G2")
axes[0, 0].fill_between(np.arange(len(g1_avg)), g1_avg - g1_lo, g1_avg + g1_hi, facecolor="blue", alpha=0.15)
axes[0, 0].fill_between(np.arange(len(g2_avg)), g2_avg - g2_lo, g2_avg + g2_hi, facecolor="red", alpha=0.15)
axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 0].legend()

axes[0, 0].set_ylabel("Sea-ice area 10$^6$ km$^2$")

ax00.plot(g1g2_nh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax00.set_ylim((p_lo_lim, 0.5))
ax00.axhline(y=0.05, c="gray", linewidth=0.5)

axes[0, 1].plot(g1s_avg, c="blue", label="G1")
axes[0, 1].plot(g2s_avg, c="red", label="G2")
axes[0, 1].fill_between(np.arange(len(g1s_avg)), g1s_avg - g1s_lo, g1s_avg + g1s_hi, facecolor="blue", alpha=0.15)
axes[0, 1].fill_between(np.arange(len(g2s_avg)), g2s_avg - g2s_lo, g2s_avg + g2s_hi, facecolor="red", alpha=0.15)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].legend()

ax01.plot(g1g2_sh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax01.set_ylim((p_lo_lim, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)

# axes[0, 0].set_xlabel("Year of simulation")
# axes[0, 1].set_xlabel("Year of simulation")
axes[0, 1].set_ylabel("Sea-ice area 10$^6$ km$^2$")

axes[0, 0].set_title("Northern Hemisphere $-$ G1 and G2")
axes[0, 1].set_title("Southern Hemisphere $-$ G1 and G2")

axes[1, 0].plot(nh_4x_an[:tlen] / 1e6, c="blue", label="no-dQ")
axes[1, 0].plot(nh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1, 0].fill_between(np.arange(tlen), (nh_4x_an[:tlen] + nh_4x_an_std[:tlen]) / 1e6, 
                        (nh_4x_an[:tlen] - nh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1, 0].fill_between(np.arange(tlen), (nh_dq4x_an[:tlen] + nh_dq4x_an_std[:tlen]) / 1e6, 
                        (nh_dq4x_an[:tlen] - nh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 0].legend()
axes[1, 0].set_ylabel("Sea-ice area 10$^6$ km$^2$")

ax10.plot(dq_nh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax10.set_ylim((p_lo_lim, 0.5))
ax10.axhline(y=0.05, c="gray", linewidth=0.5)

axes[1, 1].plot(sh_4x_an[:tlen] / 1e6, c="blue", label="no-dQ")
axes[1, 1].plot(sh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1, 1].fill_between(np.arange(tlen), (sh_4x_an[:tlen] + sh_4x_an_std[:tlen]) / 1e6, 
                        (sh_4x_an[:tlen] - sh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1, 1].fill_between(np.arange(tlen), (sh_dq4x_an[:tlen] + sh_dq4x_an_std[:tlen]) / 1e6, 
                        (sh_dq4x_an[:tlen] - sh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].legend()

axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")
axes[1, 0].set_ylabel("Sea-ice area 10$^6$ km$^2$")
axes[1, 1].set_ylabel("Sea-ice area 10$^6$ km$^2$")
axes[1, 0].set_title("Northern Hemisphere $-$ CESM2-SOM")
axes[1, 1].set_title("Southern Hemisphere $-$ CESM2-SOM")

ax11.plot(dq_sh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax11.set_ylim((p_lo_lim, 0.5))
ax11.axhline(y=0.05, c="gray", linewidth=0.5)

axes[1, 0].set_xlim((-2.5, 40.5))
axes[1, 1].set_xlim((-2.5, 40.5))

axes[0, 0].set_ylim((-0.5, 13))
axes[0, 1].set_ylim((-0.5, 13))
axes[1, 0].set_ylim((-0.5, 13))
axes[1, 1].set_ylim((-0.5, 13))

ax00.set_ylabel("$p$ value")
ax01.set_ylabel("$p$ value")
ax10.set_ylabel("$p$ value")
ax11.set_ylabel("$p$ value")

axes[0, 0].text(0, 2, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(0, 2, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 0].text(0, 2, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(0, 2, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

# fig.suptitle("G1 and G2 sea-ice area")
fig.subplots_adjust(wspace=0.33, hspace=0.3)

pl.savefig(pl_path + f"/PNG/SIA_NH_SH_AnMean_G1_G2_and_CESM2_SOM_{case_dq4x51}_{avg}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/SIA_NH_SH_AnMean_G1_G2_and_CESM2_SOM_{case_dq4x51}_{avg}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()