"""
Compare AMOC and dAMOC for two arbitrarily chosen model groups.

Be sure to set data_dir5 and data_dir6 correctly and possibly change data_path.
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
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["GFDL_CM3", "NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""


#%% loop over group one
amoc_a4x_dic = {}
amoc_pic_dic = {}
damoc_dic = {}
si_a4x_dic = {}
si_pic_dic = {}
had_a4x_dic = {}
had_pic_dic = {}

for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    amoc_a4x = np.zeros((150, len(group)))
    amoc_pic = np.zeros((150, len(group)))
    damoc = np.zeros((150, len(group)))
    si_a4x = np.zeros((150, len(group)))
    si_pic = np.zeros((150, len(group)))
    had_a4x = np.zeros((150, len(group)))
    had_pic = np.zeros((150, len(group)))
    
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
        f_name_amoc = f"amoc_{mod_d}_piControl_{a4x}.nc"
        f_name_damoc = f"damoc_and_piControl_21YrRun_{mod_d}.nc"
        f_name_si_a4x = f"si_ext_area_{mod_d}_{a4x}_*.nc"
        f_name_si_pic = f"si_ext_area_{mod_d}_piControl_*.nc"
        # f_name_dsi = f"dsiextentn_SImon_{mod_d}_{a4x}_piC21Run_*.nc"
        f_name_hadley = f"Hadley_Strength_{mod_d}_piControl_{a4x}.nc"
        
        # load the nc files
        amoc_nc = Dataset(data_path + f_name_amoc)
        damoc_nc = Dataset(data_path + f_name_damoc)
        si_a4x_nc = Dataset(glob.glob(data_path + f_name_si_a4x)[0])
        si_pic_nc = Dataset(glob.glob(data_path + f_name_si_pic)[0])
        hadley_nc = Dataset(glob.glob(data_path + f_name_hadley)[0])
        
        # load the data    
        amoc_a4x[:, i] = amoc_nc.variables["amoc_a4x"][:150]
        amoc_pic[:, i] = amoc_nc.variables["amoc_pic"][b_time:b_time+150]
        damoc[:, i] = damoc_nc.variables["damoc"][:]
        si_a4x[:, i] = an_mean(si_a4x_nc.variables["si_area_n"][:150*12])
        si_pic[:, i] = an_mean(si_pic_nc.variables["si_area_n"][b_time*12:(b_time+150)*12])
        had_a4x[:, i] = hadley_nc.variables["Had_N_a4x"][:150]
        had_pic[:, i] = hadley_nc.variables["Had_N_21yr_pic"][:150]
    # end for i, mod_d
    
    amoc_a4x_dic[g_n] = amoc_a4x
    amoc_pic_dic[g_n] = amoc_pic
    damoc_dic[g_n] = damoc
    si_a4x_dic[g_n] = si_a4x
    si_pic_dic[g_n] = si_pic
    had_a4x_dic[g_n] = had_a4x
    had_pic_dic[g_n] = had_pic

# end for g_n, group


#%% calculate group mean
avg_a4x_g1 = np.mean(amoc_a4x_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
avg_a4x_g2 = np.mean(amoc_a4x_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s
avg_pic_g1 = np.mean(amoc_pic_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
avg_pic_g2 = np.mean(amoc_pic_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s
avg_d_g1 = np.mean(damoc_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
avg_d_g2 = np.mean(damoc_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s

si_avg_a4x_g1 = np.mean(si_a4x_dic["G1"] / 1e12, axis=-1)  # 10^6 km^2
si_avg_a4x_g2 = np.mean(si_a4x_dic["G2"] / 1e12, axis=-1)  # 10^6 km^2
si_avg_pic_g1 = np.mean(si_pic_dic["G1"] / 1e12, axis=-1)  # 10^6 km^2
si_avg_pic_g2 = np.mean(si_pic_dic["G2"] / 1e12, axis=-1)  # 10^6 km^2

had_avg_a4x_g1 = np.mean(had_a4x_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
had_avg_a4x_g2 = np.mean(had_a4x_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s
had_avg_pic_g1 = np.mean(had_pic_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
had_avg_pic_g2 = np.mean(had_pic_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s


#%% calculate group standard dev
std_a4x_g1 = np.std(amoc_a4x_dic["G1"] / 1e10, axis=-1)
std_a4x_g2 = np.std(amoc_a4x_dic["G2"] / 1e10, axis=-1)
std_pic_g1 = np.std(amoc_pic_dic["G1"] / 1e10, axis=-1)
std_pic_g2 = np.std(amoc_pic_dic["G2"] / 1e10, axis=-1)
std_d_g1 = np.std(damoc_dic["G1"] / 1e10, axis=-1)
std_d_g2 = np.std(damoc_dic["G2"] / 1e10, axis=-1)

si_std_a4x_g1 = np.std(si_a4x_dic["G1"] / 1e12, axis=-1)  # 10^6 km^2
si_std_a4x_g2 = np.std(si_a4x_dic["G2"] / 1e12, axis=-1)  # 10^6 km^2
si_std_pic_g1 = np.std(si_pic_dic["G1"] / 1e12, axis=-1)  # 10^6 km^2
si_std_pic_g2 = np.std(si_pic_dic["G2"] / 1e12, axis=-1)  # 10^6 km^2

had_std_a4x_g1 = np.std(had_a4x_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
had_std_a4x_g2 = np.std(had_a4x_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s
had_std_pic_g1 = np.std(had_pic_dic["G1"] / 1e10, axis=-1)  # 10^10 kg/s
had_std_pic_g2 = np.std(had_pic_dic["G2"] / 1e10, axis=-1)  # 10^10 kg/s


#%% try the t-test to compare the means
ttest_a4x = ttest_ind(amoc_a4x_dic["G1"], amoc_a4x_dic["G2"], axis=-1)
ttest_pic = ttest_ind(amoc_pic_dic["G1"], amoc_pic_dic["G2"], axis=-1)
ttest_d = ttest_ind(damoc_dic["G1"], damoc_dic["G2"], axis=-1)

si_ttest_a4x = ttest_ind(si_a4x_dic["G1"], si_a4x_dic["G2"], axis=-1)
had_ttest_a4x = ttest_ind(had_a4x_dic["G1"], had_a4x_dic["G2"], axis=-1)


#%% linear regression
elp = 20
sle_g1, yie_g1 = lr(np.arange(elp), avg_a4x_g1[:elp])[:2]
sll_g1, yil_g1 = lr(np.arange(elp, len(avg_a4x_g1)), avg_a4x_g1[elp:])[:2]

sle_g2, yie_g2 = lr(np.arange(elp), avg_a4x_g2[:elp])[:2]
sll_g2, yil_g2 = lr(np.arange(elp, len(avg_a4x_g2)), avg_a4x_g2[elp:])[:2]

sl50_g1, yi50_g1 = lr(np.arange(50), avg_a4x_g1[:50])[:2]
sl50_g1, yi50_g1 = lr(np.arange(50, len(avg_a4x_g1)), avg_a4x_g1[50:])[:2]
sl50_g2, yi50_g2 = lr(np.arange(50), avg_a4x_g2[:50])[:2]
sl50_g2, yi50_g2 = lr(np.arange(50, len(avg_a4x_g2)), avg_a4x_g2[50:])[:2]


#%% plot the group means
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(20, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
# ax21 = axes[2].twinx()

axes[0].plot(avg_a4x_g1, c="blue", linestyle="-", label="G1")
axes[0].plot(avg_a4x_g2, c="red", linestyle="-", label="G2")
axes[1].plot(avg_pic_g1, c="blue", linestyle="-")
axes[1].plot(avg_pic_g2, c="red", linestyle="-")
axes[2].plot(avg_d_g1, c="blue", linestyle="-")
axes[2].plot(avg_d_g2, c="red", linestyle="-")

axes[0].plot(np.arange(elp), np.arange(elp) * sle_g1 + yie_g1, c="black", 
             label=f"sl={np.round(sle_g1*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
axes[0].plot(np.arange(elp), np.arange(elp) * sle_g2 + yie_g2, c="gray",
             label=f"sl={np.round(sle_g2*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
# axes[0].plot(np.arange(elp, len(avg_a4x_g1)), np.arange(elp, len(avg_a4x_g1)) * sll_g1 + yil_g1, linestyle="--", 
#              c="black", label=f"sl={np.round(sll_g1*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
# axes[0].plot(np.arange(elp, len(avg_a4x_g2)), np.arange(elp, len(avg_a4x_g2)) * sll_g2 + yil_g2, linestyle="--", 
#              c="gray", label=f"sl={np.round(sll_g2*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")

axes[0].plot(np.arange(50, len(avg_a4x_g1)), np.arange(50, len(avg_a4x_g1)) * sl50_g1 + yi50_g1, linestyle=":", 
             c="black", label=f"sl={np.round(sl50_g1*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
axes[0].plot(np.arange(50, len(avg_a4x_g2)), np.arange(50, len(avg_a4x_g2)) * sl50_g2 + yi50_g2, linestyle=":", 
             c="gray", label=f"sl={np.round(sl50_g2*10, 2)} 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")


axes[0].set_ylim((0, 3))
axes[1].set_ylim((0, 3))

ax01.plot(ttest_a4x.pvalue, c="black", linewidth=0.5)
ax01.set_ylim((-0.01, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)

ax11.plot(ttest_pic.pvalue, c="black", linewidth=0.5)
ax11.set_ylim((-0.01, 0.5))
ax11.axhline(y=0.05, c="gray", linewidth=0.5)

axes[2].axhline(y=0, c="gray", linewidth=0.5)
# ax21.plot(ttest_d.pvalue, c="black", linewidth=0.5)
# ax21.set_ylim((-0.01, 0.5))
# ax21.axhline(y=0.05, c="gray", linewidth=0.5)

axes[0].fill_between(np.arange(150), avg_a4x_g1 - std_a4x_g1, avg_a4x_g1 + std_a4x_g1, facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(150), avg_a4x_g2 - std_a4x_g2, avg_a4x_g2 + std_a4x_g2, facecolor="red", alpha=0.25)
axes[1].fill_between(np.arange(150), avg_pic_g1 - std_pic_g1, avg_pic_g1 + std_pic_g1, facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(150), avg_pic_g2 - std_pic_g2, avg_pic_g2 + std_pic_g2, facecolor="red", alpha=0.25)
axes[2].fill_between(np.arange(150), avg_d_g1 - std_d_g1, avg_d_g1 + std_d_g1, facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(150), avg_d_g2 - std_d_g2, avg_d_g2 + std_d_g2, facecolor="red", alpha=0.25)

axes[0].legend()
# axes[1].legend(loc="lower right")

fig.suptitle("AMOC G1$-$G2 comparison", fontsize=15)
axes[0].set_title("abrupt4xCO2")
axes[1].set_title("piControl")
axes[2].set_title("abrupt4xCO2 minus piControl (21-year running)")
axes[0].set_xlabel("Years since 4xCO$_2$")
axes[1].set_xlabel("Years since 4xCO$_2$")
axes[2].set_xlabel("Years since 4xCO$_2$")
axes[0].set_ylabel("AMOC in 10$^{10}$ kg s$^{-1}$")
axes[2].set_ylabel("AMOC change in 10$^{10}$ kg s$^{-1}$")

pl.subplots_adjust(wspace=0.3, top=0.875)

pl.savefig(pl_path + "AMOC_G1_G2_Comparison.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the three quantities
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(20, 6))

ax01 = axes[0].twinx()
# ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(avg_a4x_g1, c="blue", linestyle="-", label="G1")
axes[0].plot(avg_a4x_g2, c="red", linestyle="-", label="G2")
axes[1].plot(had_avg_a4x_g1, c="blue", linestyle="-")
axes[1].plot(had_avg_a4x_g2, c="red", linestyle="-")
axes[2].plot(si_avg_a4x_g1, c="blue", linestyle="-")
axes[2].plot(si_avg_a4x_g2, c="red", linestyle="-")

axes[0].fill_between(np.arange(150), avg_a4x_g1 - std_a4x_g1, avg_a4x_g1 + std_a4x_g1, facecolor="blue", alpha=0.15)
axes[0].fill_between(np.arange(150), avg_a4x_g2 - std_a4x_g2, avg_a4x_g2 + std_a4x_g2, facecolor="red", alpha=0.15)
axes[1].fill_between(np.arange(150), had_avg_a4x_g1 - had_std_a4x_g1, had_avg_a4x_g1 + had_std_a4x_g1, facecolor="blue", 
                     alpha=0.15)
axes[1].fill_between(np.arange(150), had_avg_a4x_g2 - had_std_a4x_g2, had_avg_a4x_g2 + had_std_a4x_g2, facecolor="red", 
                     alpha=0.15)
axes[2].fill_between(np.arange(150), si_avg_a4x_g1 - si_std_a4x_g1, si_avg_a4x_g1 + si_std_a4x_g1, facecolor="blue", 
                     alpha=0.15)
axes[2].fill_between(np.arange(150), si_avg_a4x_g2 - si_std_a4x_g2, si_avg_a4x_g2 + si_std_a4x_g2, facecolor="red", 
                     alpha=0.15)

axes[0].legend()

ax01.plot(ttest_a4x.pvalue, c="black", linewidth=0.5)
ax01.set_ylim((-0.01, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)

# ax11.plot(had_ttest_a4x.pvalue, c="black", linewidth=0.5)
# ax11.set_ylim((-0.01, 1))
# ax11.axhline(y=0.05, c="gray", linewidth=0.5)

ax21.plot(si_ttest_a4x.pvalue, c="black", linewidth=0.5)
ax21.set_ylim((-0.01, 0.5))
ax21.axhline(y=0.05, c="gray", linewidth=0.5)

fig.suptitle("G1$-$G2 comparison", fontsize=15)

axes[0].set_title("AMOC index")
axes[1].set_title("NH Hadley cell strength")
axes[2].set_title("NH sea-ice area")
axes[0].set_xlabel("Years since 4xCO$_2$")
axes[1].set_xlabel("Years since 4xCO$_2$")
axes[2].set_xlabel("Years since 4xCO$_2$")
axes[0].set_ylabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[1].set_ylabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[2].set_ylabel("Sea-ice area in 10$^{6}$ km")

pl.show()
pl.close()


#%% regression of NH Hadley on AMOC
elp1 = 20
elp2 = 20

sle1, yie1, r_e1, p_e1 = lr(avg_a4x_g1[:elp1], had_avg_a4x_g1[:elp1])[:4]
sll1, yil1, r_l1, p_l1 = lr(avg_a4x_g1[elp1:], had_avg_a4x_g1[elp1:])[:4]
sle2, yie2, r_e2, p_e2 = lr(avg_a4x_g2[:elp2], had_avg_a4x_g2[:elp2])[:4]
sll2, yil2, r_l2, p_l2 = lr(avg_a4x_g2[elp2:], had_avg_a4x_g2[elp2:])[:4]

ssle1, syie1, sr_e1, sp_e1 = lr(avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1])[:4]
ssll1, syil1, sr_l1, sp_l1 = lr(avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:])[:4]
ssle2, syie2, sr_e2, sp_e2 = lr(avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2])[:4]
ssll2, syil2, sr_l2, sp_l2 = lr(avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:])[:4]

hssle1, hsyie1, hsr_e1, hsp_e1 = lr(had_avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1])[:4]
hssll1, hsyil1, hsr_l1, hsp_l1 = lr(had_avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:])[:4]
hssle2, hsyie2, hsr_e2, hsp_e2 = lr(had_avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2])[:4]
hssll2, hsyil2, hsr_l2, hsp_l2 = lr(had_avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:])[:4]


#%% scatter plots of AMOC vs. NH Hadley and AMOC vs. NH sea-ice area as well as NH Hadley vs. NH sea-ice area
fig, axes = pl.subplots(nrows=1, ncols=3, figsize=(20, 6))

p1 = axes[0].scatter(avg_a4x_g1[:elp1], had_avg_a4x_g1[:elp1], c="blue", marker="o", label="G1, early")
p2 = axes[0].scatter(avg_a4x_g1[elp1:], had_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o", label="G1, late")
p3 = axes[0].scatter(avg_a4x_g2[:elp2], had_avg_a4x_g2[:elp2], c="red", marker="o", label="G2, early")
p4 = axes[0].scatter(avg_a4x_g2[elp2:], had_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o", label="G2, late")

p5, = axes[0].plot(avg_a4x_g1[:elp1], avg_a4x_g1[:elp1] * sle1 + yie1, c="black", linestyle="--", 
                   label=f"sl={np.round(sle1, 2)}  p={np.round(p_e1, 2)}  R={np.round(r_e1, 2)}")
p6, = axes[0].plot(avg_a4x_g2[:elp2], avg_a4x_g2[:elp2] * sle2 + yie2, c="gray", linestyle="--",
                   label=f"sl={np.round(sle2, 2)}  p={np.round(p_e2, 2)}  R={np.round(r_e2, 2)}")
p7, = axes[0].plot(avg_a4x_g1[elp1:], avg_a4x_g1[elp1:] * sll1 + yil1, c="black",
                   label=f"sl={np.round(sll1, 2)}  p={np.round(p_l1, 2)}  R={np.round(r_l1, 2)}")
p8, = axes[0].plot(avg_a4x_g2[elp2:], avg_a4x_g2[elp2:] * sll2 + yil2, c="gray",
                   label=f"sl={np.round(sll2, 2)}  p={np.round(p_l2, 2)}  R={np.round(r_l2, 2)}")

l1 = axes[0].legend(handles=[p1, p2, p3, p4], loc="lower right")
axes[0].legend(handles=[p5, p6, p7, p8], loc="upper left")
axes[0].add_artist(l1)

axes[1].scatter(avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1], c="blue", marker="o",)
axes[1].scatter(avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o")
axes[1].scatter(avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2], c="red", marker="o")
axes[1].scatter(avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o")

axes[1].plot(avg_a4x_g1[:elp1], avg_a4x_g1[:elp1] * ssle1 + syie1, c="black", linestyle="--", 
             label=f"sl={np.round(ssle1, 2)}  p={np.round(sp_e1, 2)}  R={np.round(sr_e1, 2)}")
axes[1].plot(avg_a4x_g2[:elp2], avg_a4x_g2[:elp2] * ssle2 + syie2, c="gray", linestyle="--",
             label=f"sl={np.round(ssle2, 2)}  p={np.round(sp_e2, 2)}  R={np.round(sr_e2, 2)}")
axes[1].plot(avg_a4x_g1[elp1:], avg_a4x_g1[elp1:] * ssll1 + syil1, c="black",
             label=f"sl={np.round(ssll1, 2)}  p={np.round(sp_l1, 2)}  R={np.round(sr_l1, 2)}")
axes[1].plot(avg_a4x_g2[elp2:], avg_a4x_g2[elp2:] * ssll2 + syil2, c="gray",
             label=f"sl={np.round(ssll2, 2)}  p={np.round(sp_l2, 2)}  R={np.round(sr_l2, 2)}")

axes[1].legend()

axes[2].scatter(had_avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1], c="blue", marker="o",)
axes[2].scatter(had_avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o")
axes[2].scatter(had_avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2], c="red", marker="o")
axes[2].scatter(had_avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o")

axes[2].plot(had_avg_a4x_g1[:elp1], had_avg_a4x_g1[:elp1] * hssle1 + hsyie1, c="black", linestyle="--", 
             label=f"sl={np.round(hssle1, 2)}  p={np.round(hsp_e1, 2)}  R={np.round(hsr_e1, 2)}")
axes[2].plot(had_avg_a4x_g2[:elp2], had_avg_a4x_g2[:elp2] * hssle2 + hsyie2, c="gray", linestyle="--",
             label=f"sl={np.round(hssle2, 2)}  p={np.round(hsp_e2, 2)}  R={np.round(hsr_e2, 2)}")
axes[2].plot(had_avg_a4x_g1[elp1:], had_avg_a4x_g1[elp1:] * hssll1 + hsyil1, c="black",
             label=f"sl={np.round(hssll1, 2)}  p={np.round(hsp_l1, 2)}  R={np.round(hsr_l1, 2)}")
axes[2].plot(had_avg_a4x_g2[elp2:], had_avg_a4x_g2[elp2:] * hssll2 + hsyil2, c="gray",
             label=f"sl={np.round(hssll2, 2)}  p={np.round(hsp_l2, 2)}  R={np.round(hsr_l2, 2)}")

axes[2].legend()

axes[0].set_title("AMOC v. NH Hadley")
axes[1].set_title("AMOC v. NH sea-ice area")
axes[2].set_title("NH Hadley v. NH sea-ice area")

axes[0].set_xlabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[0].set_ylabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")

axes[1].set_xlabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[1].set_ylabel("Sea-ice area in 10$^{6}$ km")

axes[2].set_xlabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[2].set_ylabel("Sea-ice area in 10$^{6}$ km")

pl.show()
pl.close()


#%% plot the two previous three-panel plots as one six-panel plot
fig, axes = pl.subplots(ncols=3, nrows=2, figsize=(20, 12))

# fig.suptitle("G1$-$G2 comparison", fontsize=15)

# first row --------------------------------------------------------
ax01 = axes[0, 0].twinx()
ax11 = axes[0, 1].twinx()
# ax21 = axes[0, 2].twinx()

axes[0, 0].plot(avg_a4x_g1, c="blue", linestyle="-", label="G1")
axes[0, 0].plot(avg_a4x_g2, c="red", linestyle="-", label="G2")
axes[0, 1].plot(si_avg_a4x_g1, c="blue", linestyle="-")
axes[0, 1].plot(si_avg_a4x_g2, c="red", linestyle="-")
axes[0, 2].plot(had_avg_a4x_g1, c="blue", linestyle="-")
axes[0, 2].plot(had_avg_a4x_g2, c="red", linestyle="-")

axes[0, 0].fill_between(np.arange(150), avg_a4x_g1 - std_a4x_g1, avg_a4x_g1 + std_a4x_g1, facecolor="blue", alpha=0.15)
axes[0, 0].fill_between(np.arange(150), avg_a4x_g2 - std_a4x_g2, avg_a4x_g2 + std_a4x_g2, facecolor="red", alpha=0.15)
axes[0, 1].fill_between(np.arange(150), si_avg_a4x_g1 - si_std_a4x_g1, si_avg_a4x_g1 + si_std_a4x_g1, facecolor="blue", 
                        alpha=0.15)
axes[0, 1].fill_between(np.arange(150), si_avg_a4x_g2 - si_std_a4x_g2, si_avg_a4x_g2 + si_std_a4x_g2, facecolor="red", 
                        alpha=0.15)
axes[0, 2].fill_between(np.arange(150), had_avg_a4x_g1 - had_std_a4x_g1, had_avg_a4x_g1 + had_std_a4x_g1, 
                        facecolor="blue", alpha=0.15)
axes[0, 2].fill_between(np.arange(150), had_avg_a4x_g2 - had_std_a4x_g2, had_avg_a4x_g2 + had_std_a4x_g2, 
                        facecolor="red", alpha=0.15)

axes[0, 0].legend()

ax01.plot(ttest_a4x.pvalue, c="black", linewidth=0.5)
ax01.set_ylim((-0.01, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)
ax01.set_ylabel("p-value")

ax11.plot(si_ttest_a4x.pvalue, c="black", linewidth=0.5)
ax11.set_ylim((-0.01, 0.5))
ax11.axhline(y=0.05, c="gray", linewidth=0.5)
ax11.set_ylabel("p-value")

# ax21.plot(had_ttest_a4x.pvalue, c="black", linewidth=0.25)
# ax21.set_ylim((-0.01, 0.5))
# ax21.axhline(y=0.05, c="gray", linewidth=0.5)

axes[0, 0].set_title("AMOC index")
axes[0, 1].set_title("NH sea-ice area")
axes[0, 2].set_title("NH Hadley cell strength")
axes[0, 0].set_xlabel("Years since 4xCO$_2$")
axes[0, 1].set_xlabel("Years since 4xCO$_2$")
axes[0, 2].set_xlabel("Years since 4xCO$_2$")
axes[0, 0].set_ylabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[0, 1].set_ylabel("Sea-ice area in 10$^{6}$ km")
axes[0, 2].set_ylabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")

# second row --------------------------------------------------------
p1 = axes[1, 0].scatter(avg_a4x_g1[:elp1], had_avg_a4x_g1[:elp1], c="blue", marker="o", label="G1, early")
p2 = axes[1, 0].scatter(avg_a4x_g1[elp1:], had_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o", 
                        label="G1, late")
p3 = axes[1, 0].scatter(avg_a4x_g2[:elp2], had_avg_a4x_g2[:elp2], c="red", marker="o", label="G2, early")
p4 = axes[1, 0].scatter(avg_a4x_g2[elp2:], had_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o", 
                        label="G2, late")

g1e_range = np.array([np.min(avg_a4x_g1[:elp1]), np.max(avg_a4x_g1[:elp1])])
g1l_range = np.array([np.min(avg_a4x_g1[elp1:]), np.max(avg_a4x_g1[elp1:])])
g2e_range = np.array([np.min(avg_a4x_g2[:elp2]), np.max(avg_a4x_g2[:elp2])])
g2l_range = np.array([np.min(avg_a4x_g2[elp2:]), np.max(avg_a4x_g2[elp2:])])
g1e_h_range = np.array([np.min(had_avg_a4x_g1[:elp1]), np.max(had_avg_a4x_g1[:elp1])])
g1l_h_range = np.array([np.min(had_avg_a4x_g1[elp1:]), np.max(had_avg_a4x_g1[elp1:])])
g2e_h_range = np.array([np.min(had_avg_a4x_g2[:elp2]), np.max(had_avg_a4x_g2[:elp2])])
g2l_h_range = np.array([np.min(had_avg_a4x_g2[elp2:]), np.max(had_avg_a4x_g2[elp2:])])

p5, = axes[1, 0].plot(g1e_range, g1e_range * sle1 + yie1, c="black", linestyle="--", 
                      label=f"sl$_{{e1}}$={np.round(sle1, 2)}  p={np.round(p_e1, 2)}  R={np.round(r_e1, 2)}")
p6, = axes[1, 0].plot(g2e_range, g2e_range * sle2 + yie2, c="gray", linestyle="--",
                      label=f"sl$_{{e2}}$={np.round(sle2, 2)}  p={np.round(p_e2, 2)}  R={np.round(r_e2, 2)}")
p7, = axes[1, 0].plot(g1l_range, g1l_range * sll1 + yil1, c="black",
                      label=f"sl$_{{l1}}$={np.round(sll1, 2)}  p={np.round(p_l1, 2)}  R={np.round(r_l1, 2)}")
p8, = axes[1, 0].plot(g2l_range, g2l_range * sll2 + yil2, c="gray",
                      label=f"sl$_{{l2}}$={np.round(sll2, 2)}  p={np.round(p_l2, 2)}  R={np.round(r_l2, 2)}")

l1 = axes[1, 0].legend(handles=[p1, p2, p3, p4], loc="lower right")
axes[1, 0].legend(handles=[p5, p6, p7, p8], loc="upper left")
axes[1, 0].add_artist(l1)

axes[1, 1].scatter(avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1], c="blue", marker="o",)
axes[1, 1].scatter(avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o")
axes[1, 1].scatter(avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2], c="red", marker="o")
axes[1, 1].scatter(avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o")

axes[1, 1].plot(g1e_range, g1e_range * ssle1 + syie1, c="black", linestyle="--", 
                label=f"sl$_{{e1}}$={np.round(ssle1, 2)}  p={np.round(sp_e1, 2)}  R={np.round(sr_e1, 2)}")
axes[1, 1].plot(g2e_range, g2e_range * ssle2 + syie2, c="gray", linestyle="--",
                label=f"sl$_{{e2}}$={np.round(ssle2, 2)}  p={np.round(sp_e2, 2)}  R={np.round(sr_e2, 2)}")
axes[1, 1].plot(g1l_range, g1l_range * ssll1 + syil1, c="black",
                label=f"sl$_{{l1}}$={np.round(ssll1, 2)}  p={np.round(sp_l1, 2)}  R={np.round(sr_l1, 2)}")
axes[1, 1].plot(g2l_range, g2l_range * ssll2 + syil2, c="gray",
                label=f"sl$_{{l2}}$={np.round(ssll2, 2)}  p={np.round(sp_l2, 2)}  R={np.round(sr_l2, 2)}")

axes[1, 1].legend()

axes[1, 2].scatter(had_avg_a4x_g1[:elp1], si_avg_a4x_g1[:elp1], c="blue", marker="o",)
axes[1, 2].scatter(had_avg_a4x_g1[elp1:], si_avg_a4x_g1[elp1:], c="white", edgecolor="blue", marker="o")
axes[1, 2].scatter(had_avg_a4x_g2[:elp2], si_avg_a4x_g2[:elp2], c="red", marker="o")
axes[1, 2].scatter(had_avg_a4x_g2[elp2:], si_avg_a4x_g2[elp2:], c="white", edgecolor="red", marker="o")

axes[1, 2].plot(g1e_h_range, g1e_h_range * hssle1 + hsyie1, c="black", linestyle="--", 
                label=f"sl$_{{e1}}$={np.round(hssle1, 2)}  p={np.round(hsp_e1, 2)}  R={np.round(hsr_e1, 2)}")
axes[1, 2].plot(g2e_h_range, g2e_h_range * hssle2 + hsyie2, c="gray", linestyle="--",
                label=f"sl$_{{e2}}$={np.round(hssle2, 2)}  p={np.round(hsp_e2, 2)}  R={np.round(hsr_e2, 2)}")
axes[1, 2].plot(g1l_h_range, g1l_h_range * hssll1 + hsyil1, c="black",
                label=f"sl$_{{l1}}$={np.round(hssll1, 2)}  p={np.round(hsp_l1, 2)}  R={np.round(hsr_l1, 2)}")
axes[1, 2].plot(g2l_h_range, g2l_h_range * hssll2 + hsyil2, c="gray",
                label=f"sl$_{{l2}}$={np.round(hssll2, 2)}  p={np.round(hsp_l2, 2)}  R={np.round(hsr_l2, 2)}")

axes[1, 2].legend()

axes[1, 0].set_title("AMOC v. NH Hadley")
axes[1, 1].set_title("AMOC v. NH sea-ice area")
axes[1, 2].set_title("NH Hadley v. NH sea-ice area")

axes[1, 0].set_xlabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[1, 0].set_ylabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")

axes[1, 1].set_xlabel("AMOC index in 10$^{10}$ kg s$^{-1}$")
axes[1, 1].set_ylabel("Sea-ice area in 10$^{6}$ km")

axes[1, 2].set_xlabel("Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[1, 2].set_ylabel("Sea-ice area in 10$^{6}$ km")

pl.subplots_adjust(wspace=0.25)

pl.savefig(pl_path + "AMOC_SI_Had_G1_G2_Comparison.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
        
        
        