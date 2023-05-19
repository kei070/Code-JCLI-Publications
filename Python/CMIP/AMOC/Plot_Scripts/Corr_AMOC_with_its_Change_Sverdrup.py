"""
Correlate AMOC (piControl value, "pace of early change" in abrupt-4xCO2, absolute change under abrupt-4xCO2, relative 
change under abrupt-4xCO2) with feedback change and ECS change as derived from the Gregory method.

Generates Fig. 12, S23, and S24 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir and possibly data_path, amoc_path, greg_path, kern_fb_path, and pl_path.
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
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% Include kernel derived feedback? This will reduce the number of models...
inc_kfb = True
kl = "Sh08"
kfb = "LR"


#%% set model groups
models = ["ACCESS1_0", "ACCESS1_3", "GFDL_CM3", "GFDL_ESM2M", "GISS_E2_R", "MPI_ESM_LR", 
          "MPI_ESM_MR", "MPI_ESM_P", "MRI_CGCM3", "NorESM1_M", "ACCESS-CM2", "ACCESS-ESM1-5", "CanESM5", "CESM2", 
          "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CNRM-CM6-1", "E3SM-1-0", "EC-Earth3", 
          "EC-Earth3-Veg", "FGOALS-g3", "GFDL-ESM4", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "INM-CM4-8", 
          "INM-CM5", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HAM", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MIROC-ES2L",
          "MRI-ESM2-0", "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "UKESM1-0-LL", "CMCC-ESM2", "EC-Earth3-AerChem"]

# outliers GISS_E2_R, ACCESS-CM2, MIROC-ES2L

# "GISS-E2-2-G", "CMCC-CM2-SR5"

# models used in Eiselt and Graversen (2022); i.e. models that pass the CSLT AND do not have strong cloud feedback change
# models = ["GFDL_CM3", "GISS_E2_R", "MPI_ESM_LR", "MPI_ESM_MR", "MPI_ESM_P", "MRI_CGCM3", "NorESM1_M", 
#           "ACCESS-ESM1-5", "CanESM5", "CESM2-FV2", "CESM2-WACCM-FV2", "CNRM-CM6-1", "E3SM-1-0", "EC-Earth3", 
#           "FGOALS-g3", "INM-CM4-8", "INM-CM5", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MIROC-ES2L",
#           "UKESM1-0-LL", "CMCC-ESM2", "EC-Earth3-AerChem"]

models = np.array(models)


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""
data_dir6 = ""

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = "/MultModPlots/AMOC/Correlations/"


#%% kernel dictionaries
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% loop over group one
h2o_rho = 1035

amoc_a4x_dic = {}
amoc_pic_dic = {}
damoc_dic = {}

dtas_dic = {}

fb_dic = {}
ecs_dic = {}

ifb_dic = {}

sin_pic_dic = {}
sis_pic_dic = {}

cols = []

for g_n, group in zip(["models"], [models]):
    
    amoc_a4x = np.zeros((150, len(group)))
    amoc_pic = np.zeros((150, len(group)))
    dtas = np.zeros((150, len(group)))
    sin_pic = np.zeros((150, len(group)))
    sis_pic = np.zeros((150, len(group)))
    
    fb_dic[g_n] = {}
    ifb_dic[g_n] = {}
    ecs_dic[g_n] = {}
    fb_dic[g_n]["early"] = np.zeros(len(group))
    fb_dic[g_n]["late"] = np.zeros(len(group))
    fb_dic[g_n]["change"] = np.zeros(len(group))
    ifb_dic[g_n]["early"] = np.zeros(len(group))
    ifb_dic[g_n]["late"] = np.zeros(len(group))
    ifb_dic[g_n]["change"] = np.zeros(len(group))    
    ecs_dic[g_n]["early"] = np.zeros(len(group))
    ecs_dic[g_n]["late"] = np.zeros(len(group))
    ecs_dic[g_n]["change"] = np.zeros(len(group))
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            mod_p = nl5.models[mod]
            ens = cmip_dic.mod_ens_a4x["5"][mod_d]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
            cols.append("gray")
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_p = nl6.models[mod]
            ens = cmip_dic.mod_ens_a4x["6"][mod_d]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
            cols.append("black")
        # end if
        
        print(mod_d + "\n")
        
        # data path and name
        amoc_path = data_dir + f"/CMIP{cmip}/Data/{mod_d}/"
        f_name_amoc = f"amoc_{mod_d}_piControl_{a4x}_{ens}.nc"
        f_name_damoc = f"damoc_and_piControl_21YrRun_{mod_d}.nc"
        f_name_dtas = f"dtas_Amon_{mod_n}_{a4x}_piC21Run_*.nc"
        f_name_si = "si_ext_area_*piControl*.nc"
        
        # load the amoc file
        amoc_nc = Dataset(amoc_path + f_name_amoc)
        damoc_nc = Dataset(amoc_path + f_name_damoc)
        dtas_nc = Dataset(glob.glob(amoc_path + f_name_dtas)[0])
        si_nc = Dataset(glob.glob(amoc_path + f_name_si)[0])
        
        # load the data    
        amoc_a4x[:, i] = amoc_nc.variables["amoc_a4x"][:150] / h2o_rho /  1e6
        amoc_pic[:, i] = amoc_nc.variables["amoc_pic"][b_time:b_time+150] / h2o_rho /  1e6
        sin_pic[:, i] = si_nc.variables["si_area_n"][b_time:b_time+150]
        sis_pic[:, i] = si_nc.variables["si_area_s"][b_time:b_time+150]
        damoc = damoc_nc.variables["damoc"][:] / h2o_rho /  1e6
        lat = dtas_nc.variables["lat"][:]
        lon = dtas_nc.variables["lon"][:]
        dtas[:, i] = glob_mean(np.mean(dtas_nc.variables["tas_ch"], axis=0), lat, lon)[:150]
       
        # path to Gregory data
        greg_path = data_dir + f"/CMIP{cmip}/Outputs/TOA_Imbalance_piC_Run/{a4x}/tas_Based/"
        f_name_greg = f"TOA_Imbalance_GlobAnMean_and_tas_Based_TF_piC21Run_a4x_{mod_d}.nc"
        greg_nc = Dataset(greg_path + f_name_greg)
        
        # load the Greogry regression data
        fb_dic[g_n]["early"][i] = greg_nc.variables["fb_as_e"][0]
        fb_dic[g_n]["late"][i] = greg_nc.variables["fb_as_l"][0]
        fb_dic[g_n]["change"][i] = fb_dic[g_n]["late"][i] - fb_dic[g_n]["early"][i]
        ecs_dic[g_n]["early"][i] = greg_nc.variables["ecs_as_e"][0]
        ecs_dic[g_n]["late"][i] = greg_nc.variables["ecs_as_l"][0]
        ecs_dic[g_n]["change"][i] = ecs_dic[g_n]["late"][i] - ecs_dic[g_n]["early"][i]
        
        if inc_kfb:
            kern_fb_path = data_dir + f"/CMIP{cmip}/Outputs/Feedbacks/Kernel/{a4x}/{k_p[kl]}/tas_Based/"
            ind_fb_nc = Dataset(glob.glob(kern_fb_path + f"/*{mod_p}.nc")[0])
            ifb_dic[g_n]["early"][i] = ind_fb_nc.variables[kfb + "_fb_e"][0]
            ifb_dic[g_n]["late"][i] = ind_fb_nc.variables[kfb + "_fb_l"][0]
            ifb_dic[g_n]["change"][i] = ifb_dic[g_n]["late"][i] - ifb_dic[g_n]["early"][i]
        # end if
        
    # end for i, mod_d
    
    amoc_a4x_dic[g_n] = amoc_a4x
    amoc_pic_dic[g_n] = amoc_pic
    sin_pic_dic[g_n] = sin_pic
    sis_pic_dic[g_n] = sis_pic
    damoc_dic[g_n] = damoc

# end for g_n, group


#%% calculate the piControl mean AMOC
amoc_pic_mean = np.mean(amoc_pic_dic["models"], axis=0)


#%% calculate the piControl mean sea-ice area
sin_pic_mean = np.mean(sin_pic_dic["models"], axis=0)
sis_pic_mean = np.mean(sis_pic_dic["models"], axis=0)


#%% calculate the mean over the last 15 years of dtas
dtas_l15 = np.mean(dtas[-15:, :], axis=0)


#%% calculate the year-20 warming
dtas_20 = np.mean(dtas[15:25, :], axis=0)


#%% calculate the mean AMOC in the last 50 years of the abrupt-4xCO2 run
amoc_a4x_lmean = np.zeros(len(models))
amoc_a4x_emean = np.zeros(len(models))

for i in np.arange(len(models)):
    amoc_a4x_lmean[i] = np.mean(amoc_a4x_dic[g_n][48:53, i], axis=0)
    amoc_a4x_emean[i] = np.mean(amoc_a4x_dic[g_n][13:18, i], axis=0)
# end for i    


#%% calculate the difference between the piControl AMOC and the last 50-year mean 4xCO2 AMOC
damoc_y30 = amoc_a4x_lmean - amoc_pic_mean
damoc_y15 = amoc_a4x_emean - amoc_pic_mean


#%% calculate the "pace" of early AMOC decline in the first x years
ep = 15
lp = 50
amoc_decline = np.zeros(len(models))
amoc_recovery = np.zeros(len(models))
for i in np.arange(len(models)):
    amoc_decline[i] = lr(np.arange(ep), amoc_a4x_dic[g_n][:ep, i])[0]
    amoc_recovery[i] = lr(np.arange(150-lp), amoc_a4x_dic[g_n][lp:, i])[0]
# end for i


#%% --> extract the models with non-negative feedback change
p_dfb_ind = fb_dic[g_n]["change"] > 0
n_dfb_ind = fb_dic[g_n]["change"] < 0


#%% AMOC trend v indiv. feedback - Fig. S4
amoc_dtrend = amoc_recovery - amoc_decline

sle, yie, r_e, p_e = lr(amoc_decline, ifb_dic[g_n]["early"])[:4]
sll, yil, r_l, p_l = lr(amoc_recovery, ifb_dic[g_n]["late"])[:4]
sld, yid, r_d, p_d = lr(amoc_dtrend, ifb_dic[g_n]["change"])[:4]

fig, axes = pl.subplots(nrows=1, ncols=3, figsize=(11, 3), sharey=True)

for i in np.arange(len(models)):
    axes[0].scatter(amoc_decline[i], ifb_dic[g_n]["early"][i], c=cols[i], marker="o", s=10)
    axes[1].scatter(amoc_recovery[i], ifb_dic[g_n]["late"][i], c=cols[i], marker="o", s=10)
    axes[2].scatter(amoc_dtrend[i], ifb_dic[g_n]["change"][i], c=cols[i], marker="o", s=10)
# end for i

axes[0].plot(amoc_decline, amoc_decline * sle + yie, c="black",
             label=f"sl={np.round(sle, 2)},  R={np.round(r_e, 3)},  p={np.round(p_e, 4)}")
axes[1].plot(amoc_recovery, amoc_recovery * sll + yil, c="black", 
             label=f"sl={np.round(sll, 2)},  R={np.round(r_l, 3)},  p={np.round(p_l, 4)}")
axes[2].plot(amoc_dtrend, amoc_dtrend * sld + yid, c="black", 
             label=f"sl={np.round(sld, 2)},  R={np.round(r_d, 3)},  p={np.round(p_d, 4)}")

axes[0].axhline(y=0, c="gray", linewidth=0.5)
axes[0].axvline(x=0, c="gray", linewidth=0.5)
axes[1].axhline(y=0, c="gray", linewidth=0.5)
axes[1].axvline(x=0, c="gray", linewidth=0.5)
axes[2].axhline(y=0, c="gray", linewidth=0.5)

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")
axes[2].legend(loc="lower right")

axes[0].set_title(f"Early {kfb} feedback v.\nearly AMOC trend")
axes[1].set_title(f"Late {kfb} feedback v.\nlate AMOC trend")
axes[2].set_title(f"{kfb} feedback change v.\nAMOC trend change")

axes[0].set_ylabel(f"{kfb} feedback in Wm$^{{-2}}$K$^{{-1}}$")
axes[0].set_xlabel("Early AMOC trend in Sv a$^{-1}$")
axes[1].set_xlabel("Late AMOC trend in Sv a$^{-1}$")
axes[2].set_xlabel("AMOC trend change in Sv a$^{-1}$")

fig.subplots_adjust(top=0.73, wspace=0.05)

# pl.savefig(pl_path + f"/PNG/AMOC_trends_v_{kfb}_Fbs_CMIP56.png", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/AMOC_trends_v_{kfb}_Fbs_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% AMOC trend v total feedback - Fig. S5
"""
amoc_dtrend = amoc_recovery - amoc_decline

sle, yie, r_e, p_e = lr(amoc_decline / 1E8, fb_dic[g_n]["early"])[:4]
sll, yil, r_l, p_l = lr(amoc_recovery / 1E8, fb_dic[g_n]["late"])[:4]
sld, yid, r_d, p_d = lr(amoc_dtrend / 1E8, fb_dic[g_n]["change"])[:4]

fig, axes = pl.subplots(nrows=1, ncols=3, figsize=(11, 3), sharey=True)

for i in np.arange(len(models)):
    axes[0].scatter(amoc_decline[i] / 1E8, fb_dic[g_n]["early"][i], c=cols[i], marker="o", s=10)
    axes[1].scatter(amoc_recovery[i] / 1E8, fb_dic[g_n]["late"][i], c=cols[i], marker="o", s=10)
    axes[2].scatter(amoc_dtrend[i] / 1E8, fb_dic[g_n]["change"][i], c=cols[i], marker="o", s=10)
# end for i

axes[0].plot(amoc_decline / 1E8, amoc_decline / 1E8 * sle + yie, c="black",
             label=f"sl={np.round(sle, 2)},  R={np.round(r_e, 3)},  p={np.round(p_e, 4)}")
axes[1].plot(amoc_recovery / 1E8, amoc_recovery / 1E8 * sll + yil, c="black", 
             label=f"sl={np.round(sll, 2)},  R={np.round(r_l, 3)},  p={np.round(p_l, 4)}")
axes[2].plot(amoc_dtrend / 1E8, amoc_dtrend / 1E8 * sld + yid, c="black", 
             label=f"sl={np.round(sld, 2)},  R={np.round(r_d, 3)},  p={np.round(p_d, 4)}")

axes[0].axhline(y=0, c="gray", linewidth=0.5)
axes[0].axvline(x=0, c="gray", linewidth=0.5)
axes[1].axhline(y=0, c="gray", linewidth=0.5)
axes[1].axvline(x=0, c="gray", linewidth=0.5)
axes[2].axhline(y=0, c="gray", linewidth=0.5)

axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")
axes[2].legend(loc="lower right")

axes[0].set_title("Early feedback v.\nearly AMOC trend")
axes[1].set_title("Late feedback v.\nlate AMOC trend")
axes[2].set_title("Feedback change v.\nAMOC trend change")

axes[0].set_ylabel("Feedback in Wm$^{{-2}}$K$^{{-1}}$")
axes[0].set_xlabel("Early AMOC trend in 10$^{8}$ kg s$^{-1}$ a$^{-1}$")
axes[1].set_xlabel("Late AMOC trend in 10$^{8}$ kg s$^{-1}$ a$^{-1}$")
axes[2].set_xlabel("AMOC trend change in 10$^{8}$ kg s$^{-1}$ a$^{-1}$")

fig.subplots_adjust(top=0.73, wspace=0.05)

pl.savefig(pl_path + "/PNG/AMOC_trends_v_Fbs_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/AMOC_trends_v_Fbs_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
""" 

#%% linear regression
sl_p, yi_p, r_p, p_p, err_p = lr(amoc_pic_mean, amoc_decline * 10)
sl_ch, yi_ch, r_ch, p_ch, err_ch = lr(amoc_pic_mean, damoc_y30)
sl_fbe, yi_fbe, r_fbe, p_fbe, err_fbe = lr(amoc_pic_mean, fb_dic[g_n]["early"])
sl_dfb, yi_dfb, r_dfb, p_dfb, err_dfb = lr(amoc_pic_mean, fb_dic[g_n]["change"])
sl_tr, yi_tr, r_tr, p_tr, err_tr = lr(amoc_decline * 10, amoc_recovery * 10)
sl_prec, yi_prec, r_prec, p_prec, err_prec = lr(amoc_pic_mean, amoc_recovery * 10)

sl_pa_dfb, yi_pa_dfb, r_pa_dfb, p_pa_dfb, err_pa_dfb = lr(amoc_pic_mean,  fb_dic[g_n]["change"])
sl_pa_difb, yi_pa_difb, r_pa_difb, p_pa_difb, err_pa_difb = lr(amoc_pic_mean,  ifb_dic[g_n]["change"])

sl_ar_dfb, yi_ar_dfb, r_ar_dfb, p_ar_dfb, err_ar_dfb = lr(amoc_recovery * 10,  fb_dic[g_n]["change"])
sl_ar_difb, yi_ar_difb, r_ar_difb, p_ar_difb, err_ar_difb = lr(amoc_recovery * 10,  ifb_dic[g_n]["change"])

sl_ar_y15, yi_ar_y15, r_ar_y15, p_ar_y15, err_ar_y15 = lr(amoc_recovery * 10,  amoc_a4x_emean)
sl_ar_y30, yi_ar_y30, r_ar_y30, p_ar_y30, err_ar_y30 = lr(amoc_recovery * 10,  amoc_a4x_lmean)


#%% correlate piControl AMOC with early AMOC decline pace and early AMOC change
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

# for i in np.arange(len(models)):
#     axes.text(amoc_pic_mean[i] / 1E10, amoc_decline[i] / 1E8, models[i], c="black", horizontalalignment="center", 
#               fontsize=8)
# end for i
# axes[0].scatter(amoc_pic_mean / 1E10, amoc_decline / 1E8, c="black", marker="o")
for i in np.arange(len(models)):
    axes[0].scatter(amoc_pic_mean[i], amoc_decline[i] * 10, c=cols[i], marker="o", s=20)
# end for i    
axes[0].plot(amoc_pic_mean, amoc_pic_mean * sl_p + yi_p, c="black", label=f"R={r2(r_p)}, p={r4(p_p)}")
axes[0].legend()

axes[0].set_title("Early AMOC trend (years 1-15) v.\npiControl mean AMOC")
axes[0].set_xlabel("piControl mean AMOC in Sv")
axes[0].set_ylabel("Early AMOC trend in Sv dec$^{-1}$")

# axes[1].scatter(amoc_pic_mean / 1E10, damoc_y30 / 1E10, c="black", marker="o")
for i in np.arange(len(models)):
    axes[1].scatter(amoc_pic_mean[i], damoc_y30[i], c=cols[i], marker="o", s=20)
# end for i    
axes[1].plot(amoc_pic_mean, amoc_pic_mean * sl_ch + yi_ch, c="black", label=f"R={r2(r_ch)}, p={r4(p_ch)}")
axes[1].legend()

axes[1].set_title("AMOC change at year 30 v.\npiControl mean AMOC")
axes[1].set_xlabel("piControl mean AMOC in Sv")
axes[1].set_ylabel("AMOC change in Sv")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/dAMOC_v_piAMOC_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/dAMOC_v_piAMOC_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate piControl AMOC with early feedback and feedback change
"""
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

# for i in np.arange(len(models)):
#     axes.text(amoc_pic_mean[i] / 1E10, amoc_decline[i] / 1E8, models[i], c="black", horizontalalignment="center", 
#               fontsize=8)
# end for i
# axes[0].scatter(amoc_pic_mean / 1E10, fb_dic[g_n]["early"], c="black", marker="o")
for i in np.arange(len(models)):
    axes[0].scatter(amoc_pic_mean[i] / 1E10, fb_dic[g_n]["early"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(amoc_pic_mean / 1E10, amoc_pic_mean / 1E10 * sl_fbe + yi_fbe, c="black", 
             label=f"R={r2(r_fbe)}, p={r4(p_fbe)}")
axes[0].legend()

axes[0].set_title("Early climate feedback v.\npiControl mean AMOC")
axes[0].set_xlabel("piControl mean AMOC in 10$^{10}$ kg s$^{-1}$")
axes[0].set_ylabel("Early feedback in Wm$^{-2}$K$^{-1}$")

# axes[1].scatter(amoc_pic_mean / 1E10, fb_dic[g_n]["change"], c="black", marker="o")
for i in np.arange(len(models)):
    axes[1].scatter(amoc_pic_mean[i] / 1E10, fb_dic[g_n]["change"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(amoc_pic_mean / 1E10, amoc_pic_mean / 1E10 * sl_dfb + yi_dfb, c="black", 
             label=f"R={r2(r_dfb)}, p={r4(p_dfb)}")
axes[1].legend()

axes[1].set_title("Climate feedback change v.\npiControl mean AMOC")
axes[1].set_xlabel("piControl mean AMOC in 10$^{10}$ kg s$^{-1}$")
axes[1].set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/Feedback_v_piAMOC_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/Feedback_v_piAMOC_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
"""

#%% correlate piC AMOC with total and individual feedback change
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

for i in np.arange(len(models)):
    axes[0].scatter(amoc_pic_mean[i], fb_dic[g_n]["change"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(amoc_pic_mean, amoc_pic_mean * sl_pa_dfb + yi_pa_dfb, c="black", 
             label=f"R={r2(r_pa_dfb)}, p={r4(p_pa_dfb)}")
axes[0].legend()
axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[0].set_ylim((-1, 1.5))

axes[0].set_title("Total feedback change v.\npiControl mean AMOC")
axes[0].set_xlabel("piControl mean AMOC in Sv")
axes[0].set_ylabel("Total feedback change in Wm$^{-2}$K$^{-1}$")

for i in np.arange(len(models)):
    axes[1].scatter(amoc_pic_mean[i], ifb_dic[g_n]["change"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(amoc_pic_mean, amoc_pic_mean * sl_pa_difb + yi_pa_difb, c="black", 
             label=f"R={r2(r_pa_difb)}, p={r4(p_pa_difb)}")
axes[1].legend(loc="upper left")
axes[1].axhline(y=0, c="black", linewidth=0.5)
axes[1].set_ylim((-1, 1.5))

axes[1].set_title("Lapse-rate feedback change v.\npiControl mean AMOC")
axes[1].set_xlabel("piControl mean AMOC in Sv")
axes[1].set_ylabel("Lapse-rate feedback change in Wm$^{-2}$K$^{-1}$")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/Feedback_v_piAMOC_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/Feedback_v_piAMOC_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate AMOC recovery with total and individual feedback change
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

# for i in np.arange(len(models)):
#     axes[1].text(amoc_recovery[i] / 1E8, ifb_dic[g_n]["change"][i], models[i], c="black", horizontalalignment="center", 
#                  fontsize=8)
# end for i

for i in np.arange(len(models)):
    axes[0].scatter(amoc_recovery[i] * 10, fb_dic[g_n]["change"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(amoc_recovery * 10, amoc_recovery * 10 * sl_ar_dfb + yi_ar_dfb, c="black", 
             label=f"R={r2(r_ar_dfb)}, p={r4(p_ar_dfb)}")
axes[0].legend(loc="upper left")
axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[0].axvline(x=0, c="black", linewidth=0.5)
axes[0].set_ylim((-1, 1.5))

axes[0].set_title("Total feedback change v.\nlate AMOC trend")
axes[0].set_xlabel("Late AMOC trend in Sv dec$^{-1}$")
axes[0].set_ylabel("Total feedback change in Wm$^{-2}$K$^{-1}$")

for i in np.arange(len(models)):
    axes[1].scatter(amoc_recovery[i] * 10, ifb_dic[g_n]["change"][i], c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(amoc_recovery * 10, amoc_recovery * 10 * sl_ar_difb + yi_ar_difb, c="black", 
             label=f"R={r2(r_ar_difb)}, p={r4(p_ar_difb)}")
axes[1].legend(loc="upper left")
axes[1].axhline(y=0, c="black", linewidth=0.5)
axes[1].axvline(x=0, c="black", linewidth=0.5)
axes[1].set_ylim((-1, 1.5))

axes[1].set_title("Lapse-rate feedback change v.\nlate AMOC trend")
axes[1].set_xlabel("Late AMOC trend in Sv dec$^{-1}$")
axes[1].set_ylabel("Lapse-rate feedback change in Wm$^{-2}$K$^{-1}$")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/Feedback_v_AMOC_LateTrend_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/Feedback_v_AMOC_LateTrend_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate AMOC early with late trend and piC AMOC with AMOC recovery
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

for i in np.arange(len(models)):
    axes[0].scatter(amoc_decline[i] * 10, amoc_recovery[i] * 10, c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(amoc_decline * 10, amoc_decline * 10 * sl_tr + yi_tr, c="black", 
             label=f"R={r2(r_tr)}, p={r4(p_tr)}")
axes[0].legend()
axes[0].axhline(y=0, c="black", linewidth=0.5)

axes[0].set_title("Early AMOC trend v.\nlate AMOC trend")
axes[0].set_xlabel("Early AMOC trend in Sv dec$^{-1}$")
axes[0].set_ylabel("Late AMOC trend in Sv dec$^{-1}$")

for i in np.arange(len(models)):
    axes[1].scatter(amoc_pic_mean[i], amoc_recovery[i] * 10, c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(amoc_pic_mean, amoc_pic_mean * sl_prec + yi_prec, c="black", 
             label=f"R={r2(r_prec)}, p={r4(p_prec)}")
axes[1].legend(loc="lower right")
axes[1].axhline(y=0, c="black", linewidth=0.5)

axes[1].set_title("piControl mean AMOC v.\nlate AMOC trend")
axes[1].set_xlabel("piControl mean AMOC in Sv")
axes[1].set_ylabel("Late AMOC trend in Sv dec$^{-1}$")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/AMOC_Trends_and_piAMOC_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/AMOC_Trends_and_piAMOC_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate AMOC late trend with early change
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

for i in np.arange(len(models)):
    axes[0].scatter(amoc_recovery[i], amoc_a4x_emean[i], c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(amoc_recovery, amoc_recovery * sl_ar_y15 + yi_ar_y15, c="black", 
             label=f"R={r2(r_ar_y15)}, p={r4(p_ar_y15)}")
axes[0].legend()
axes[0].axvline(x=0, c="black", linewidth=0.5)

axes[0].set_title("AMOC change years 13-17 v.\nlate AMOC trend")
axes[0].set_xlabel("Late AMOC trend in Sv a$^{-1}$")
axes[0].set_ylabel("AMOC change yr15 in Sv")

for i in np.arange(len(models)):
    axes[1].scatter(amoc_recovery[i], amoc_a4x_lmean[i], c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(amoc_recovery, amoc_recovery * sl_ar_y30 + yi_ar_y30, c="black", 
             label=f"R={r2(r_ar_y30)}, p={r4(p_ar_y30)}")
axes[1].legend(loc="lower right")
axes[1].axvline(x=0, c="black", linewidth=0.5)

axes[1].set_title("AMOC change years 28-32 v.\nlate AMOC trend")
axes[1].set_xlabel("Late AMOC trend in Sv a$^{-1}$")
axes[1].set_ylabel("AMOC change yr30 in Sv")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/AMOC_Trend_and_AMOC_Change_CMIP56.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PDF/AMOC_Trend_and_AMOC_Change_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate sea-ice extent with AMOC
sl1, yi1, r_1, p_1 = lr(sin_pic_mean / 1E12, amoc_pic_mean)[:4]
sl2, yi2, r_2, p_2 = lr(sin_pic_mean / 1E12, amoc_decline * 10)[:4]


fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(11, 4))

for i in np.arange(len(models)):
    axes[0].scatter(sin_pic_mean[i] / 1E12, amoc_pic_mean[i], c=cols[i], marker="o", s=20)
# end for i   
axes[0].plot(sin_pic_mean / 1E12, sin_pic_mean / 1E12 * sl1 + yi1, c="black", label=f"R={r2(r_1)}, p={r4(p_1)}")
axes[0].legend()
axes[0].axhline(y=0, c="black", linewidth=0.5)
# axes[0].set_ylim((-1, 1.5))

axes[0].set_title("piControl NH sea-ice area v.\npiControl mean AMOC")
axes[0].set_xlabel("piControl NH sea-ice area in 10$^{6}$km$^2$")
axes[0].set_ylabel("piControl mean AMOC in Sv")

for i in np.arange(len(models)):
    axes[1].scatter(sin_pic_mean[i] / 1E12, amoc_decline[i] * 10, c=cols[i], marker="o", s=20)
# end for i   
axes[1].plot(sin_pic_mean / 1E12, sin_pic_mean / 1E12 * sl2 + yi2, c="black", label=f"R={r2(r_2)}, p={r4(p_2)}")
axes[1].legend(loc="lower left")
axes[1].axhline(y=0, c="black", linewidth=0.5)
# axes[1].set_ylim((-1, 1.5))

axes[1].set_title("piControl NH sea-ice area v.\nearly AMOC trend")
axes[1].set_xlabel("piControl NH sea-ice area in 10$^{6}$km$^2$")
axes[1].set_ylabel("Early AMOC trend in Sv dec$^{-1}$")

pl.subplots_adjust(wspace=0.3)

pl.savefig(pl_path + "/PNG/piSI_v_piAMOC_CMIP56.png", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PDF/Feedback_v_piAMOC_CMIP56.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
