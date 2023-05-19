"""
Same as Fig. 9 in Eiselt and Graversen (2022), just for arbitrary model groups.

Comparing Gregory plots of the respective means over two arbitrarily chosen model groups.

Be sure to change data_dir5, data_dir6, and pl_path. Also check data_path.
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
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups

# G1 and G2 models with AMOC data
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models1 = ["CanESM5", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["ACCESS-ESM1-5", "BCC-CSM2-MR", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models1 = ['GFDL_CM3', 'ACCESS-CM2', 'MIROC6']  # strong delta AABW
models2 = ['GFDL_ESM2M', 'MRI_CGCM3', 'CNRM-CM6-1', 'E3SM-1-0', 'FGOALS-g3', 'INM-CM4-8', 'INM-CM5']  # weak delta AABW

models1 = ['GFDL_CM3', 'GISS_E2_R', 'ACCESS-CM2', 'FGOALS-g3', 'MIROC6']  # strong piC AABW
models2 = ['GFDL_ESM2M', 'MRI_CGCM3', 'CNRM-CM6-1', 'E3SM-1-0', 'EC-Earth3', 'EC-Earth3-Veg', 'INM-CM4-8', 'INM-CM5', 
           'NorESM2-MM']  # weak piC AABW

models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "ACCESS-ESM1-5", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]


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


#%% loop over group one
dtas_dic = {}
dtoa_dic = {}
dtoacs_dic = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    dtas = np.zeros((150, len(group)))
    dtoa = np.zeros((150, len(group)))
    dtoacs = np.zeros((150, len(group)))
    
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
        f_name_dtas = "dtas_Amon_*.nc"
        f_name_dtoa = "dtoa_as_cs_Amon_*.nc"
        # f_name_damoc = f"damoc_and_piControl_21YrRun_{mod_d}.nc"
        
        # load the tas and TOA files
        dtas_nc = Dataset(glob.glob(data_path + f_name_dtas)[0])
        dtoa_nc = Dataset(glob.glob(data_path + f_name_dtoa)[0])
        
        lat = dtas_nc.variables["lat"][:]
        lon = dtas_nc.variables["lon"][:]
        
        # load the data    
        dtas[:, i] = an_mean(glob_mean(dtas_nc.variables["tas_ch"][:150*12, :, :], lat, lon))
        dtoa[:, i] = an_mean(glob_mean(dtoa_nc.variables["dtoa_as"][:150*12, :, :], lat, lon))
        dtoacs[:, i] = an_mean(glob_mean(dtoa_nc.variables["dtoa_cs"][:150*12, :, :], lat, lon))
        # damoc = damoc_nc.variables["damoc"][:]
        
    # end for i, mod_d
    
    dtas_dic[g_n] = dtas
    dtoa_dic[g_n] = dtoa
    dtoacs_dic[g_n] = dtoacs

# end for g_n, group


#%% calculate the group means
dtas_g1 = np.mean(dtas_dic["G1"], axis=-1)
dtas_g2 = np.mean(dtas_dic["G2"], axis=-1)
dtoa_g1 = np.mean(dtoa_dic["G1"], axis=-1)
dtoa_g2 = np.mean(dtoa_dic["G2"], axis=-1)
dtoacs_g1 = np.mean(dtoacs_dic["G1"], axis=-1)
dtoacs_g2 = np.mean(dtoacs_dic["G2"], axis=-1)

dtas_g1_std = np.std(dtas_dic["G1"], axis=-1)
dtas_g2_std = np.std(dtas_dic["G2"], axis=-1)
dtoa_g1_std = np.std(dtoa_dic["G1"], axis=-1)
dtoa_g2_std = np.std(dtoa_dic["G2"], axis=-1)
dtoacs_g1_std = np.std(dtoacs_dic["G1"], axis=-1)
dtoacs_g2_std = np.std(dtoacs_dic["G2"], axis=-1)

dcre_g1 = dtoa_g1 - dtoacs_g1
dcre_g2 = dtoa_g2 - dtoacs_g2


#%% Gregory regressions
elp = 20

lr_e_g1 = lr(dtas_g1[:elp], dtoa_g1[:elp])
lr_e_g2 = lr(dtas_g2[:elp], dtoa_g2[:elp])
sle_g1, yie_g1, re_g1, pe_g1, ere_g1 = lr_e_g1
sll_g1, yil_g1, rl_g1, pl_g1, erl_g1 = lr(dtas_g1[elp:], dtoa_g1[elp:])
sle_g2, yie_g2, re_g2, pe_g2, ere_g2 = lr_e_g2
sll_g2, yil_g2, rl_g2, pl_g2, erl_g2 = lr(dtas_g2[elp:], dtoa_g2[elp:])


#%% calculate the estimated forcing and estimate from the first two years
lr_2_g1 = lr(dtas_g1[:2], dtoa_g1[:2])
lr_2_g2 = lr(dtas_g2[:2], dtoa_g2[:2])
sl2_g1, yi2_g1, r2_g1, p2_g1, er2_g1 = lr_2_g1
sl2_g2, yi2_g2, r2_g2, p2_g2, er2_g2 = lr_2_g2

#%% calculate running mean dTas
dtas_g1_run = run_mean(dtas_g1, running=50)
dtas_g2_run = run_mean(dtas_g2, running=50)


#%% running feedbacks
nyr = 150
runyr = 50
xtick = 10
run_fb = {}
for g12 in ["G1", "G2"]:
    
    run_fb_l = []
    
    for mem in np.arange(np.shape(dtas_dic[g12])[1]):
        
        run_fb_ll = []
        
        for yr in np.arange(nyr-runyr):
            run_fb_ll.append(lr(dtas_dic[g12][yr:yr+runyr, mem], dtoa_dic[g12][yr:yr+runyr, mem])[0])
        # end for yr
        
        run_fb_l.append(np.array(run_fb_ll))
               
    # end for mem
    
    run_fb[g12] = np.array(run_fb_l)
    
# end for g12

run_fb_g1 = np.mean(run_fb["G1"], axis=0)
run_fb_g2 = np.mean(run_fb["G2"], axis=0)

run_fb_g1_std = np.std(run_fb["G1"], axis=0)
run_fb_g2_std = np.std(run_fb["G2"], axis=0)

# running feedback plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.plot(run_fb_g1, label="G1", c="blue")
axes.fill_between(np.arange(len(run_fb_g1)), run_fb_g1 - run_fb_g1_std, run_fb_g1 + run_fb_g1_std, facecolor="blue", 
                  alpha=0.15)

axes.plot(run_fb_g2, label="G2", c="red")
axes.fill_between(np.arange(len(run_fb_g2)), run_fb_g2 - run_fb_g2_std, run_fb_g2 + run_fb_g2_std, facecolor="red", 
                  alpha=0.15)

axes.set_xticks(np.arange(0, nyr-runyr, xtick))
axes.set_xticklabels(np.arange(int(runyr/2), int(nyr-runyr/2), xtick))

axes.set_xlabel("Center year of feedback regression")
axes.set_ylabel("Feedback in Wm$^{-2}$K$^{-1}$")
axes.set_title(f"G1 and G2 {runyr}-year running feedback")

pl.savefig(pl_path + f"/PDF/Running_{runyr}Yr_Feedback_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Running_{runyr}Yr_Feedback_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% running feedback plot in "tempterature space"
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.plot(dtas_g1_run[1:], run_fb_g1, label="G1", c="blue")
# axes.fill_between(np.arange(len(run_fb_g1)), run_fb_g1 - run_fb_g1_std, run_fb_g1 + run_fb_g1_std, facecolor="blue", 
#                   alpha=0.15)

axes.plot(dtas_g2_run[1:], run_fb_g2, label="G2", c="red")
# axes.fill_between(np.arange(len(run_fb_g2)), run_fb_g2 - run_fb_g2_std, run_fb_g2 + run_fb_g2_std, facecolor="red", 
#                   alpha=0.15)

# axes.set_xticks(np.arange(0, nyr-runyr, xtick))
# axes.set_xticklabels(np.arange(int(runyr/2), int(nyr-runyr/2), xtick))

axes.set_xlabel("$\Delta$SAT in K at center year of feedback regression")
axes.set_ylabel("Feedback in Wm$^{-2}$K$^{-1}$")
axes.set_title(f"G1 and G2 {runyr}-year running feedback")

# pl.savefig(pl_path + f"/PDF/Running_{runyr}Yr_Feedback_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/Running_{runyr}Yr_Feedback_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% running feedbacks - method 2
nyr = 150
sta_yr = 5
xtick = 20
run_fb = {}
for g12 in ["G1", "G2"]:
    
    run_fb_l = []
    
    for mem in np.arange(np.shape(dtas_dic[g12])[1]):
        
        run_fb_ll = []
        
        for yr in np.arange(sta_yr, nyr):
            run_fb_ll.append(lr(dtas_dic[g12][:yr, mem], dtoa_dic[g12][:yr, mem])[0])
        # end for yr
        
        run_fb_l.append(np.array(run_fb_ll))
               
    # end for mem
    
    run_fb[g12] = np.array(run_fb_l)
    
# end for g12

run_fb_g1 = np.mean(run_fb["G1"], axis=0)
run_fb_g2 = np.mean(run_fb["G2"], axis=0)

run_fb_g1_std = np.std(run_fb["G1"], axis=0)
run_fb_g2_std = np.std(run_fb["G2"], axis=0)

# running feedback plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.plot(run_fb_g1, label="G1", c="blue")
axes.fill_between(np.arange(len(run_fb_g1)), run_fb_g1 - run_fb_g1_std, run_fb_g1 + run_fb_g1_std, facecolor="blue", 
                  alpha=0.15)

axes.plot(run_fb_g2, label="G2", c="red")
axes.fill_between(np.arange(len(run_fb_g2)), run_fb_g2 - run_fb_g2_std, run_fb_g2 + run_fb_g2_std, facecolor="red", 
                  alpha=0.15)

axes.legend(loc="lower right")

axes.set_xticks(np.arange(0, nyr-sta_yr+1, xtick))
axes.set_xticklabels(np.arange(sta_yr, nyr+1, xtick))

axes.set_xlabel("End year of feedback regression")
axes.set_ylabel("Feedback in Wm$^{-2}$K$^{-1}$")
axes.set_title("G1 and G2 running feedback")

# pl.savefig(pl_path + f"/PDF/Running_StaYr{sta_yr}_Feedback_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/Running_StaYr{sta_yr}_Feedback_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% Gregory plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.scatter(dtas_g1[:elp], dtoa_g1[:elp], label=f"G1 early {np.round(sle_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="blue", edgecolor="blue")
axes.scatter(dtas_g1[elp:], dtoa_g1[elp:], label=f"G1 late {np.round(sll_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="white",  edgecolor="blue")
axes.scatter(dtas_g2[:elp], dtoa_g2[:elp], label=f"G2 early {np.round(sle_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="red",  edgecolor="red")
axes.scatter(dtas_g2[elp:], dtoa_g2[elp:], label=f"G2 late {np.round(sll_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="white",  edgecolor="red")

axes.plot(dtas_g1[:elp], dtas_g1[:elp] * sle_g1 + yie_g1, c="black", linewidth=2)
axes.plot(dtas_g1[elp:], dtas_g1[elp:] * sll_g1 + yil_g1, c="black", linewidth=2)
axes.plot(dtas_g2[:elp], dtas_g2[:elp] * sle_g2 + yie_g2, c="gray", linewidth=2)
axes.plot(dtas_g2[elp:], dtas_g2[elp:] * sll_g2 + yil_g2, c="gray", linewidth=2)

axes.legend(loc="lower left", fontsize=9)

axes.axvline(x=0, c="gray", linewidth=0.5)
axes.axhline(y=0, c="gray", linewidth=0.5)

# axes.set_title("Gregory plot for G1 and G2 models with AMOC data")
axes.set_title("Gregory plot for G1 and G2")
axes.set_xlabel("Global mean surface temperature in K")
axes.set_ylabel("Global mean TOA imbalance in Wm$^{{-2}}$")

# pl.savefig(pl_path + "/PDF/Gregory_G1_G2_models_with_AMOC_data.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Gregory_G1_G2_models_with_AMOC_data.png", bbox_inches="tight", dpi=250)

# pl.savefig(pl_path + "/PDF/Gregory_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Gregory_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% Gregory plot CRE
sle_g1, yie_g1, re_g1, pe_g1, ere_g1 = lr(dtas_g1[:elp], dcre_g1[:elp])
sll_g1, yil_g1, rl_g1, pl_g1, erl_g1 = lr(dtas_g1[elp:], dcre_g1[elp:])
sle_g2, yie_g2, re_g2, pe_g2, ere_g2 = lr(dtas_g2[:elp], dcre_g2[:elp])
sll_g2, yil_g2, rl_g2, pl_g2, erl_g2 = lr(dtas_g2[elp:], dcre_g2[elp:])

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.scatter(dtas_g1[:elp], dcre_g1[:elp], label=f"G1 early {np.round(sle_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="blue", edgecolor="blue")
axes.scatter(dtas_g1[elp:], dcre_g1[elp:], label=f"G1 late {np.round(sll_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="white",  edgecolor="blue")
axes.scatter(dtas_g2[:elp], dcre_g2[:elp], label=f"G2 early {np.round(sle_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="red",  edgecolor="red")
axes.scatter(dtas_g2[elp:], dcre_g2[elp:], label=f"G2 late {np.round(sll_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$", marker="o", 
             c="white",  edgecolor="red")

axes.plot(dtas_g1[:elp], dtas_g1[:elp] * sle_g1 + yie_g1, c="black", linewidth=2)
axes.plot(dtas_g1[elp:], dtas_g1[elp:] * sll_g1 + yil_g1, c="black", linewidth=2)
axes.plot(dtas_g2[:elp], dtas_g2[:elp] * sle_g2 + yie_g2, c="gray", linewidth=2)
axes.plot(dtas_g2[elp:], dtas_g2[elp:] * sll_g2 + yil_g2, c="gray", linewidth=2)

axes.legend(loc="upper left", fontsize=9)

axes.axvline(x=0, c="gray", linewidth=0.5)
axes.axhline(y=0, c="gray", linewidth=0.5)

# axes.set_title("Gregory plot for G1 and G2 models with AMOC data")
axes.set_title("Gregory plot for G1 and G2")
axes.set_xlabel("Global mean surface temperature in K")
axes.set_ylabel("Global mean TOA CRE in Wm$^{{-2}}$")

# pl.savefig(pl_path + "/PDF/Gregory_G1_G2_models_with_AMOC_data.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Gregory_G1_G2_models_with_AMOC_data.png", bbox_inches="tight", dpi=250)

# pl.savefig(pl_path + "/PDF/Gregory_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Gregory_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% dTas and dTOA plot
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

ax00 = axes.twinx()

axes.plot(dtoa_g1, label="G1 $\Delta$TOA", c="blue")
axes.plot(dtoa_g2, label="G2 $\Delta$TOA", c="red")
ax00.plot(dtas_g1, label="G1 $\Delta$SAT", c="blue", linestyle="--")
ax00.plot(dtas_g2, label="G2 $\Delta$SAT", c="red", linestyle="--")

axes.fill_between(np.arange(len(dtas_g1)), dtas_g1 - dtas_g1_std, dtas_g1 + dtas_g1_std, facecolor="blue", alpha=0.25)
axes.fill_between(np.arange(len(dtas_g2)), dtas_g2 - dtas_g2_std, dtas_g2 + dtas_g2_std, facecolor="red", alpha=0.25)

ax00.fill_between(np.arange(len(dtoa_g1)), dtoa_g1 - dtoa_g1_std, dtoa_g1 + dtoa_g1_std, facecolor="blue", alpha=0.25)
ax00.fill_between(np.arange(len(dtoa_g2)), dtoa_g2 - dtoa_g2_std, dtoa_g2 + dtoa_g2_std, facecolor="red", alpha=0.25)

axes.legend(loc=(0.05, 0.01), fontsize=9)
ax00.legend(loc=(0.05, 0.84), fontsize=9)

axes.axvline(x=0, c="gray", linewidth=0.5)
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.set_title("$\Delta$Tas and $\Delta$TOA imbalance for G1 and G2")

axes.set_xlabel("Years of simulation")
axes.set_ylabel("Global mean TOA imbalance in Wm$^{{-2}}$")
ax00.set_ylabel("Global mean surface temperature in K")

axes.set_ylim((-0.25, 8.5))
ax00.set_ylim((-0.25, 8.5))

# pl.savefig(pl_path + "/PDF/dTas_dTOA_G1_G2_models.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/dTas_dTOA_G1_G2_models.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the forcing estimate
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].scatter(0, yie_g1, marker="o", c="blue")
axes[0].scatter(1, yie_g2, marker="o", c="red")
axes[0].scatter(0, yi2_g1, marker="x", c="blue")
axes[0].scatter(1, yi2_g2, marker="x", c="red")
axes[0].set_xlim((-0.5, 1.5))
axes[0].set_ylabel("Forcing estimate in Wm$^{-2}$")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["G1", "G2"])
axes[0].set_title("Forcing estimate")

axes[1].scatter(0, yie_g2 - yie_g1, marker="o", c="black")
axes[1].scatter(0, yi2_g2 - yi2_g1, marker="x", c="black")
axes[1].set_xlim((-0.5, 1.5))
ax2 = axes[1].twinx()
ax2.scatter(1, -(yie_g2 - yie_g1) / yie_g1 * 100, marker="o", c="blue", label="with respect to G1 (20yr)")
ax2.scatter(1, -(yie_g2 - yie_g1) / yie_g2 * 100, marker="o", c="red", label="with respect to G2 (20yr)")
ax2.scatter(1, -(yi2_g2 - yi2_g1) / yi2_g1 * 100, marker="x", c="blue", label="with respect to G1 (2yr)")
ax2.scatter(1, -(yi2_g2 - yi2_g1) / yi2_g2 * 100, marker="x", c="red", label="with respect to G2 (2yr)")
ax2.legend(loc="center right")
ax2.set_ylabel("Forcing estimate relative difference in %")
axes[1].set_ylabel("Forcing estimate difference in Wm$^{-2}$")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["absolute", "relative"])
axes[1].set_title("Forcing estimate dfference (G2 minus G1)")

pl.show()
pl.close()





