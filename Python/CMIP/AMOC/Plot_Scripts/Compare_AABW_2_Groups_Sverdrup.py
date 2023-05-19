"""
Compare AABW and dAABW for two arbitrarily chosen model groups.

Generates Fig. S22 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir and possibly data_path.
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
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["GFDL_CM3", "NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models = {"G1":models1, "G2":models2}


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = "/MultModPlots/AABW/"


#%% loop over group one
h2o_rho = 1035

amoc_a4x_dic = {}
amoc_pic_dic = {}
amoc_pic_run_dic = {}
damoc_dic = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    amoc_a4x = np.zeros((150, len(group)))
    amoc_pic = np.zeros((150, len(group)))
    amoc_pic_run = np.zeros((150, len(group)))
    damoc = np.zeros((150, len(group)))
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
            ens = cmip_dic.mod_ens_a4x[cmip][mod_d]
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
            ens = cmip_dic.mod_ens_a4x[cmip][mod_d]
        # end if
        
        # data path and name
        data_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        f_name_amoc = f"aabw_{mod_d}_piControl_{a4x}_{ens}.nc"
        f_name_damoc = f"daabw_and_piControl_21YrRun_{mod_d}.nc"
        
        # load the amoc file
        amoc_nc = Dataset(data_path + f_name_amoc)
        damoc_nc = Dataset(data_path + f_name_damoc)
        
        # load the data    
        amoc_a4x[:, i] = amoc_nc.variables["aabw_a4x_67S"][:150] / h2o_rho /  1e6
        amoc_pic[:, i] = amoc_nc.variables["aabw_pic_67S"][b_time:b_time+150] / h2o_rho /  1e6
        amoc_pic_run[:, i] = damoc_nc.variables["aabw_pic_run_67S"][:] / h2o_rho /  1e6
        damoc[:, i] = damoc_nc.variables["daabw_67S"][:] / h2o_rho /  1e6
        
    # end for i, mod_d
    
    amoc_a4x_dic[g_n] = amoc_a4x
    amoc_pic_dic[g_n] = amoc_pic
    amoc_pic_run_dic[g_n] = amoc_pic_run
    damoc_dic[g_n] = damoc

# end for g_n, group


#%% calculate standard dev
avg_a4x_g1 = np.mean(amoc_a4x_dic["G1"], axis=-1)
avg_a4x_g2 = np.mean(amoc_a4x_dic["G2"], axis=-1)
avg_pic_g1 = np.mean(amoc_pic_dic["G1"], axis=-1)
avg_pic_g2 = np.mean(amoc_pic_dic["G2"], axis=-1)
avg_pic_run_g1 = np.mean(amoc_pic_run_dic["G1"], axis=-1)
avg_pic_run_g2 = np.mean(amoc_pic_run_dic["G2"], axis=-1)
avg_d_g1 = np.mean(damoc_dic["G1"], axis=-1)
avg_d_g2 = np.mean(damoc_dic["G2"], axis=-1)

std_a4x_g1 = np.std(amoc_a4x_dic["G1"], axis=-1)
std_a4x_g2 = np.std(amoc_a4x_dic["G2"], axis=-1)
std_pic_g1 = np.std(amoc_pic_dic["G1"], axis=-1)
std_pic_g2 = np.std(amoc_pic_dic["G2"], axis=-1)
std_pic_run_g1 = np.std(amoc_pic_run_dic["G1"], axis=-1)
std_pic_run_g2 = np.std(amoc_pic_run_dic["G2"], axis=-1)
std_d_g1 = np.std(damoc_dic["G1"], axis=-1)
std_d_g2 = np.std(damoc_dic["G2"], axis=-1)


#%% try the t-test to compare the means
ttest_a4x = ttest_ind(amoc_a4x_dic["G1"], amoc_a4x_dic["G2"], axis=-1)
ttest_pic = ttest_ind(amoc_pic_dic["G1"], amoc_pic_dic["G2"], axis=-1)
ttest_pic_run = ttest_ind(amoc_pic_run_dic["G1"], amoc_pic_run_dic["G2"], axis=-1)
ttest_d = ttest_ind(damoc_dic["G1"], damoc_dic["G2"], axis=-1)


#%% linear regression
elp = 15
sle_g1, yie_g1 = lr(np.arange(elp), avg_a4x_g1[:elp])[:2]
sll_g1, yil_g1 = lr(np.arange(elp, len(avg_a4x_g1)), avg_a4x_g1[elp:])[:2]

sle_g2, yie_g2 = lr(np.arange(elp), avg_a4x_g2[:elp])[:2]
sll_g2, yil_g2 = lr(np.arange(elp, len(avg_a4x_g2)), avg_a4x_g2[elp:])[:2]

sl50_g1, yi50_g1 = lr(np.arange(50), avg_a4x_g1[:50])[:2]
sl50_g1, yi50_g1 = lr(np.arange(50, len(avg_a4x_g1)), avg_a4x_g1[50:])[:2]
sl50_g2, yi50_g2 = lr(np.arange(50), avg_a4x_g2[:50])[:2]
sl50_g2, yi50_g2 = lr(np.arange(50, len(avg_a4x_g2)), avg_a4x_g2[50:])[:2]


#%% loop over the models and print the piControl mean
ep = 15
lp = 50

print(f"G1 mean: {np.mean(amoc_pic_dic['G1'])} Sv  G2 mean: {np.mean(amoc_pic_dic['G2'])} Sv\n\n")

print(f"Year 50 AMOC change: G1 mean: {np.mean(damoc_dic['G1'][48:52, :])} Sv  " + 
      f"G2 mean: {np.mean(damoc_dic['G2'][48:52, :])} Sv\n\n")

print(f"Difference: {np.mean(amoc_pic_dic['G2']) - np.mean(amoc_pic_dic['G1'])} Sv")

sle1, yie1, rve1, pve1, erre1 = lr(np.arange(ep), avg_a4x_g1[:ep] * 10) 
sll1, yil1, rvl1, pvl1, errl1 = lr(np.arange(150-lp), avg_a4x_g1[lp:] * 10) 
sle2, yie2, rve2, pve2, erre2 = lr(np.arange(ep), avg_a4x_g2[:ep] * 10) 
sll2, yil2, rvl2, pvl2, errl2 = lr(np.arange(150-lp), avg_a4x_g2[lp:] * 10) 
print(f"G1 early {sle1 * 10} +/- {erre1 * 10} Sv/dec")
print(f"G1 late {sll1 * 10} +/- {errl1 * 10} Sv/dec\n")
print(f"G2 early {sle2 * 10} +/- {erre2 * 10} Sv/dec")
print(f"G2 late {sll2 * 10} +/- {errl2 * 10} Sv/dec\n\n")

sle, sll = {}, {}
for g12 in ["G1", "G2"]:
    
    sle[g12] = np.zeros(len(models[g12]))
    sll[g12] = np.zeros(len(models[g12]))
    
    for mem in np.arange(len(models[g12])):
        
        print(f"{g12} {models[g12][mem]} {np.mean(amoc_pic_dic[g12][:, mem])} Sv")
        
        sle[g12][mem], yie, rve, pve, erre = lr(np.arange(ep), amoc_a4x_dic[g12][:ep, mem] / 1e9) 
        sll[g12][mem], yil, rvl, pvl, errl = lr(np.arange(150-lp), amoc_a4x_dic[g12][lp:, mem] / 1e9) 
        
        print(f"{g12} {models[g12][mem]} early {sle[g12][mem] * 10} +/- {erre * 10} Sv/dec")
        print(f"{g12} {models[g12][mem]} late {sll[g12][mem] * 10} +/- {errl * 10} Sv/dec\n")
    # end for mem

# end for g12   
        

#%% plot the group means
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(13, 9))

ax01 = axes[0, 0].twinx()
ax11 = axes[0, 1].twinx()
ax21 = axes[1, 0].twinx()
ax31 = axes[1, 1].twinx()

axes[0, 0].plot(avg_pic_g1, c="blue", linestyle="-")
axes[0, 0].plot(avg_pic_g2, c="red", linestyle="-")
axes[0, 0].plot(avg_pic_run_g1, c="blue", linestyle="--", linewidth=1.5, label="G1 21-year running mean")
axes[0, 0].plot(avg_pic_run_g2, c="red", linestyle="--", linewidth=1.5, label="G2 21-year running mean")
p01, = axes[0, 1].plot(avg_a4x_g1, c="blue", linestyle="-", label="G1")
p02, = axes[0, 1].plot(avg_a4x_g2, c="red", linestyle="-", label="G2")
axes[1, 0].plot(avg_d_g1, c="blue", linestyle="-", label="G1")
axes[1, 0].plot(avg_d_g2, c="red", linestyle="-", label="G2")
axes[1, 0].axhline(y=0, c="black", linewidth=0.5)

axes[1, 1].plot(avg_d_g2 - avg_d_g1, c="black", linestyle="-", label="change")
axes[1, 1].plot(avg_a4x_g2 - avg_a4x_g1, c="black", linestyle="--", label="abrupt4xCO2")
axes[1, 1].plot(avg_pic_g2 - avg_pic_g1, c="black", linestyle=":", label="piControl")
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)

# axes[2].plot(avg_a4x_g1 - np.mean(avg_pic_g1), c="blue", linestyle="--")
# axes[2].plot(avg_a4x_g2 - np.mean(avg_pic_g2), c="red", linestyle="--")

p03, = axes[0, 1].plot(np.arange(elp), np.arange(elp) * sle_g1 + yie_g1, c="cyan", 
                       label=f"sl={np.round(sle_g1*10, 2)} Sv dec$^{{-1}}$", linewidth=3)
p04, = axes[0, 1].plot(np.arange(elp), np.arange(elp) * sle_g2 + yie_g2, c="black",
                       label=f"sl={np.round(sle_g2*10, 2)} Sv dec$^{{-1}}$", linewidth=3)
# axes[0].plot(np.arange(elp, len(avg_a4x_g1)), np.arange(elp, len(avg_a4x_g1)) * sll_g1 + yil_g1, linestyle="--", 
#              c="black", label=f"sl={np.round(sll_g1*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
# axes[0].plot(np.arange(elp, len(avg_a4x_g2)), np.arange(elp, len(avg_a4x_g2)) * sll_g2 + yil_g2, linestyle="--", 
#              c="gray", label=f"sl={np.round(sll_g2*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")

p05, = axes[0, 1].plot(np.arange(50, len(avg_a4x_g1)), np.arange(50, len(avg_a4x_g1)) * sl50_g1 + yi50_g1, 
                       linestyle="--", c="cyan", linewidth=3, 
                       label=f"sl={np.round(sl50_g1*10, 2)} Sv dec$^{{-1}}$")
p06, = axes[0, 1].plot(np.arange(50, len(avg_a4x_g2)), np.arange(50, len(avg_a4x_g2)) * sl50_g2 + yi50_g2, 
                       linestyle="--", c="black", linewidth=3, 
                       label=f"sl={np.round(sl50_g2*10, 2)} Sv dec$^{{-1}}$")

# legends
axes[0, 0].legend(fontsize=13)
l2 = axes[0, 1].legend(handles=[p03, p04, p05, p06], fontsize=13)
axes[1, 0].legend(loc="upper right", fontsize=13)
axes[1, 1].legend(loc="center", fontsize=13)

axes[0, 0].set_ylim((-5, 20))
axes[0, 1].set_ylim((-5, 20))

ax01.plot(ttest_pic.pvalue, c="black", linewidth=0.5, linestyle=":")
ax01.set_ylim((-0.01, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)

ax11.plot(ttest_a4x.pvalue, c="black", linewidth=0.5, linestyle=":")
ax11.set_ylim((-0.01, 0.5))
ax11.axhline(y=0.05, c="gray", linewidth=0.5)

# ax21.plot(ttest_d.pvalue, c="black", linewidth=0.5, linestyle=":")
ax21.set_ylim((-0.01, 0.5))
ax21.set_yticks([])
ax21.set_yticklabels(labels=[])
# ax21.axhline(y=0.05, c="gray", linewidth=0.5)

ax31.set_ylim((-0.01, 0.5))
ax31.set_yticks([])
ax31.set_yticklabels(labels=[])

axes[0, 0].fill_between(np.arange(150), avg_pic_g1 - std_pic_g1, avg_pic_g1 + std_pic_g1, facecolor="blue", alpha=0.25)
axes[0, 0].fill_between(np.arange(150), avg_pic_g2 - std_pic_g2, avg_pic_g2 + std_pic_g2, facecolor="red", alpha=0.25)
axes[0, 1].fill_between(np.arange(150), avg_a4x_g1 - std_a4x_g1, avg_a4x_g1 + std_a4x_g1, facecolor="blue", alpha=0.25)
axes[0, 1].fill_between(np.arange(150), avg_a4x_g2 - std_a4x_g2, avg_a4x_g2 + std_a4x_g2, facecolor="red", alpha=0.25)
axes[1, 0].fill_between(np.arange(150), avg_d_g1 - std_d_g1, avg_d_g1 + std_d_g1, facecolor="blue", alpha=0.25)
axes[1, 0].fill_between(np.arange(150), avg_d_g2 - std_d_g2, avg_d_g2 + std_d_g2, facecolor="red", alpha=0.25)

# axes[1].legend(loc="lower right")

fig.suptitle("AABW G1$-$G2 comparison", fontsize=17)
axes[0, 0].set_title("piControl", fontsize=16)
axes[0, 1].set_title("abrupt4xCO2", fontsize=16)
axes[1, 0].set_title("Change", fontsize=16)
axes[1, 1].set_title("Group difference (G2 $-$ G1)", fontsize=16)
# axes[0, 0].set_xlabel("Years since 4xCO$_2$", fontsize=14)
# axes[0, 1].set_xlabel("Years since 4xCO$_2$", fontsize=14)
axes[1, 0].set_xlabel("Years since 4xCO$_2$", fontsize=14)
axes[1, 1].set_xlabel("Years since 4xCO$_2$", fontsize=14)
axes[0, 0].set_ylabel("AABW in Sv", fontsize=14)
axes[0, 1].set_ylabel("AABW in Sv", fontsize=14)
axes[1, 0].set_ylabel("AABW change in Sv", fontsize=14)
axes[1, 1].set_ylabel("AABW (change) difference in Sv", fontsize=14)

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

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)

ax01.text(5, 0.48, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.48, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax21.text(5, 0.48, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax31.text(5, 0.48, "(d)", fontsize=14, horizontalalignment="center", verticalalignment="center")

pl.subplots_adjust(wspace=0.4, top=0.92)

pl.savefig(pl_path + "AABW_G1_G2_Comparison_V2.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
        