"""
Global mean clear-sky linearity test (see Shell et al., 2008, and Ceppi and Gregory, 2017).

Be sure to set direc_data correctly; set list_path; set the path for float_thr_nc; 
and also check the code block "set paths"
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
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
script_direc = "/home/kei070/Documents/Python_Scripts_PhD_Publication1/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set up lists of models in a dictionary for both CMIPs
mod_lists = {"5":["ACCESS1_0", "ACCESS1_3", "BCC_CSM1_1", "BCC_CSM1_1_M", "BNU_ESM", "CanESM2", "CNRM_CM5", "GFDL_CM3", 
                  "GFDL_ESM2M", "GISS_E2_H", "GISS_E2_R", "HadGEM2_ES", "IPSL_CM5A_LR", "IPSL_CM5B_LR", "MIROC_ESM", 
                  "MIROC5", "MPI_ESM_LR", "MPI_ESM_MR", "MPI_ESM_P", "MRI_CGCM3", "NorESM1_M"], 
             
             "6":["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-ESM1", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5", 
                    "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-SR5", "CMCC-ESM2", 
                    "CNRM-CM6-1", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-Veg", 
                    "FGOALS-f3-L", "FGOALS-g3", "GFDL-CM4","GFDL-ESM4", "GISS-E2-1-G", "GISS-E2-1-H", "GISS-E2-2-G", 
                    "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "INM-CM4-8", "INM-CM5", "IPSL-CM6A-LR", "KIOST-ESM", "MIROC6", 
                    "MIROC-ES2L", "MPI-ESM1-2-HAM", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorCPM1", 
                    "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "TaiESM1", "UKESM1-0-LL"]}

mod_lists_n = {"5":["ACCESS1.0", "ACCESS1.3", "BCC-CSM1.1", "BCC-CSM1.1(m)", "BNU-ESM", "CanESM2", "CNRM-CM5", 
                    "GFDL-CM3", "GFDL-ESM2M", "GISS-E2-H", "GISS-E2-R", "HadGEM2-ES", "IPSL-CM5A-LR", "IPSL-CM5B-LR", 
                    "MIROC-ESM", "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P", "MRI-CGCM3", "NorESM1-M"], 

               "6":["ACCESS-CM2", "ACCESS-ESM1.5", "AWI-CM-1.1-MR", "BCC-ESM1", "BCC-CSM2-MR", "CAMS-CSM1.0", "CanESM5", 
                    "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-SR5", "CMCC-ESM2", 
                    "CNRM-CM6.1", "CNRM-ESM2.1", "E3SM-1.0", "EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-Veg", 
                    "FGOALS-f3-L", "FGOALS-g3", "GFDL-CM4","GFDL-ESM4", "GISS-E2.1-G", "GISS-E2.1-H", "GISS-E2.2-G", 
                    "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "INM-CM4.8", "INM-CM5", "IPSL-CM6A-LR", "KIOST-ESM", "MIROC6", 
                    "MIROC-ES2L", "MPI-ESM1.2-HAM", "MPI-ESM1.2-HR", "MPI-ESM1.2-LR", "MRI-ESM2.0", "NorCPM1", 
                    "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "TaiESM1", "UKESM1.0-LL"]}


#%% choose kernels
kl = "Sh08"

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]
k_k = [kl]


#%% which "CO2 forcing adjustment" should be used? Either "kernel" or "Gregory"
co2_adjs = "Gregory"  # for now leave this at Gregory since the kernel method is a little unclear at this point

# -> note that it acutally doesn't make so much sense to distinguish between these both here since we're in this script 
#    only concerned with feedbacks and hence the slopes of the regressions which won't be influenced by any constant
#    (in time) factor added to the data


#%% choose the surface temperature variable (tas or ts)
t_var = "tas"


#%% choose the threshold year to split early and late period
thr_yr = 20


#%% running mean piControl
run = 21


#%% number of years from the forced experiment
n_yr = 150


#%% set up dictionaries for both CMIPs
direc_data = {"5":"", "6":""}  # SETS BASIC DATA PATH
a4x = {"5":"abrupt4xCO2", "6":"abrupt-4xCO2"}


#%% load the model dictionary to relate the models to their index
import Namelists.CMIP_Dictionaries as di
mod_dict = di.mod_dict


#%% experiment (factor of abrupt CO2 increase)
exp = "4x"


#%% set up some initial dictionaries and lists

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# linestyles per kernel
l_sty = {"Sh08":":", "So08":"-.", "BM13":":", "H17":None, "P18":"--", "S18":"-."}

# marker style per kernel
m_sty = {"Sh08":"+", "So08":"s", "BM13":"^", "H17":"o", "P18":"x", "S18":"v"}

# kernel colour
k_c = {"Sh08":"green", "So08":"blue", "BM13":"magenta", "H17":"orange", "P18":"black", "S18":"gray"}

# response colours in the plots
re_c = {"T":"red", "Q":"green", "S":"orange", "C":"blue"}
re_more_c = {"Ta":"red", "Ts":"orange", "LR":"purple", "Pl":"violet", "Q_lw":"green", "Q_sw":"lightgreen", "C_lw":"blue", 
             "C_sw":"lightblue"}

# adjustment short and long names
res_s = ["T", "Q", "S", "C"]
res_l = {"T":"temperature", "Q":"water vapour", "S":"surface albedo", "C":"cloud"}
res_more_s = ["Ta", "Ts", "LR", "Pl", "Q_lw", "Q_sw", "C_lw", "C_sw"]
res_more_l = {"Ta":"air temperature", "Ts":"surface temperature", "LR":"lapse rate", "Pl":"Planck", 
              "Q_lw":"LW water vapour", "Q_sw":"SW water vapour", "C_lw":"LW cloud", "C_sw":"SW cloud"}

# sign of long wave kernels
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

# possibilites for CO2 adjustment: only BM13 and Sh08 have the CO2 "kernels"
co2_adjd = {"Sh08":co2_adjs, "So08":"Gregory", "BM13":co2_adjs, "H17":"Gregory", "P18":"Gregory", "S18":"Gregory"}


#%% set paths
# set up dictionaries for the paths to the different directories for the different kernels
re_paths = dict()  # adjustment paths
data_paths = dict()
pl_paths = dict()  # plot paths
png_paths = dict()  # plot paths
pl_paths_fb = dict()  # plot paths feedback decomposition
png_paths_fb = dict()  # plot paths feedback decomposition
out_paths_fb = dict()  # output paths feedback decomposition
list_paths = dict()  # store values for tables
for cmip_v in [5, 6]:
    data_path = (direc_data[str(cmip_v)])
    data_paths[str(cmip_v)] = data_path
        
    re_paths[str(cmip_v)] = (data_path + "Kernel_TOA_RadResp/" + a4x[str(cmip_v)] + "/" + k_p[kl] + "/")

# end for cmip


# set and create the list directory
list_path = f"/Model_Lists/ClearSkyLinearityTest/{t_var}_Based/"


#%% load the floating threshold Gregory regression data set to get the max dTF year
float_thr_nc = Dataset("" + f"Floating_ELP_Threshold_Range15_75_TF_Using_{t_var}.nc")
thr_str = str(thr_yr)
load_max_thr = False
# if the given threshold year is -2 use the max dTF magnitude, if it is -1 use the max dTF value
if thr_yr == -1:
    
    # load year of max dTF (not magnitude!) --> pair the values with the model name and convert this to a dictionary
    max_dfb_yr = dict(zip(float_thr_nc.variables["models"][:], float_thr_nc.variables["yr_max_dTF"][:]))
    
    # set the threshold string for the plot name
    thr_str = "MaxdTF"
    
    # maximum threshold yeear will be assigned in the for-loop below
    load_max_thr = True
    
    print("\nUsing the year of maximum dTF (not magnitude!).\n")
    
elif thr_yr == -2:

    # load year of max dTF magnitude (!) --> pair the values with the model name and convert this to a dictionary
    max_dfb_yr = dict(zip(float_thr_nc.variables["models"][:], float_thr_nc.variables["yr_abs_max_dTF"][:]))

    # set the threshold string for the plot name
    thr_str = "MaxAbsdTF"

    # maximum threshold yeear will be assigned in the for-loop below
    load_max_thr = True

    print("\nUsing the year of maximum dTF.\n")
# end if elif


#%% generate plot and output paths according to the "settings"
list_path = list_path + f"Thres_Year_{thr_str}/"


#%% set up dictionaries to contain all models
mod_fb = dict()
k_sum = dict()


#%% loop over all given models
print("Loop over all given models:\n")
for cmip_v in [5, 6]:
    
    for mod, mod_s in enumerate(mod_lists[str(cmip_v)]):
        
        # get the model name
        mod_n = mod_lists_n[str(cmip_v)][mod]
        
        print(mod_n)
        
        if load_max_thr:
            # extract the year and assign it to threshold year
            thr_yr = max_dfb_yr[mod_n]
            print(mod_n + f" Threshold Year {thr_yr}\n")
        # use the max (magnitude) dTF year
    
        # load the netcdf data ------------------------------------------------------------------------------------------
        t_nc = dict()
        q_nc = dict()
        s_nc = dict()
        c_nc = dict()
        
        for kl in k_k:
            t_nc[kl] = Dataset(re_paths[str(cmip_v)] + "T_Response/CS/TOA_RadResponse_ta_ts_lr_" + kl + "_Kernel_CS_" + 
                               mod_lists[str(cmip_v)][mod] + ".nc")
            q_nc[kl] = Dataset(re_paths[str(cmip_v)] + "Q_Response/CS/TOA_RadResponse_q_" + kl + "_Kernel_CS_" + 
                               mod_lists[str(cmip_v)][mod] + ".nc")
            s_nc[kl] = Dataset(re_paths[str(cmip_v)] + "SfcAlb_Response/CS/TOA_RadResponse_SfcAlb_" + kl + 
                               "_Kernel_CS_" + mod_lists[str(cmip_v)][mod] + ".nc")
        # end for kl    
        
        tas_nc = Dataset(data_paths[str(cmip_v)] + f"GlobMean_{t_var}_piC_Run/{a4x[str(cmip_v)]}" + 
                         f"/GlobalMean_a4xCO2_{t_var}_piC21run_{mod_lists[str(cmip_v)][mod]}.nc")
        toa_nc = Dataset(data_paths[str(cmip_v)] + f"TOA_Imbalance_piC_Run/{a4x[str(cmip_v)]}/{t_var}_Based/" + 
                         f"/TOA_Imbalance_GlobAnMean_and_{t_var}_Based_TF_piC{run}Run_" + 
                         f"a4x_{mod_lists[str(cmip_v)][mod]}.nc")
    
    
        # get the values ------------------------------------------------------------------------------------------------
        lat = t_nc[k_k[0]].variables["lat"][:]
        lon = t_nc[k_k[0]].variables["lon"][:]
        
        tas = tas_nc.variables[f"{t_var}_ch"][:]
        toa = toa_nc.variables["toa_imb_cs"][:]
        
        sl_toa_e, forcing_cs_e, r, p_toa_e = lr(tas[:thr_yr], toa[:thr_yr])[:4]
        sl_toa_l, forcing_cs_l, r, p_toa_l = lr(tas[thr_yr:], toa[thr_yr:])[:4]
        
        # gather the values in a dictionary
        sing_mod_fb = dict()
        sing_mod_fb["e"] = sl_toa_e
        sing_mod_fb["l"] = sl_toa_l
        
        # feedbacks -----------------------------------------------------------------------------------------------------
        res = dict()
        
        for kl in k_k:
            
            # global means
            res[kl] = dict()
        
            res[kl]["T"] = sig[kl] * glob_mean(an_mean(t_nc[kl].variables["t_resp"][:]), lat, lon)
            res[kl]["Q"] = glob_mean(an_mean(sig[kl] * q_nc[kl].variables["q_lw_resp"][:]), lat, lon) + \
                                                             glob_mean(an_mean(q_nc[kl].variables["q_sw_resp"][:]), lat, 
                                                                       lon)
            res[kl]["S"] = glob_mean(an_mean(s_nc[kl].variables["sa_resp"][:]), lat, lon)
        
        # end for kl        
    
        # linear regressions for the responses (use nested dictionaries) ------------------------------------------------
        #   nomenclature for better understanding: k iterates over the kernels (Huang, Pendergrass) and re iterates of
        #   the responses (temperature, water vapour, surface albedo, cloud)
        
        lr_r = dict()
        
        for kl in k_k:
            
            lr_r[kl] = dict()
        
            lr_res = dict()
        
            for re in res[kl].keys():
                
                lr_res[re] = dict()
                
                # set dictionary names        
                lr_res[re]["e"] = dict()
                lr_res[re]["l"] = dict()
                
                # early period
                lr_res[re]["e"]["s"], lr_res[re]["e"]["i"], lr_res[re]["e"]["r"], lr_res[re]["e"]["p"] = \
                                                                               lr(tas[:thr_yr], res[kl][re][:thr_yr])[:4]
                # late period
                lr_res[re]["l"]["s"], lr_res[re]["l"]["i"], lr_res[re]["l"]["r"], lr_res[re]["l"]["p"] = \
                                                                               lr(tas[thr_yr:], res[kl][re][thr_yr:])[:4]
            # end for i
            
            lr_r[kl] = lr_res        
        
        # end for re
    
        # sum the individual feedbacks --------------------------------------------------------------------------------------
        
        fb_sum = dict()
        
        for kl in k_k:
            
            fb_sum[kl] = dict()
        
            fb_e = 0
            fb_l = 0
        
            for re in res[kl].keys():
                fb_e += lr_r[kl][re]["e"]["s"]
                fb_l += lr_r[kl][re]["l"]["s"]
            # end for re
        
            fb_sum[kl]["e"] = fb_e
            fb_sum[kl]["l"] = fb_l
            
            # calculate the sum of the kernel-derived responses
            ksum = res[kl]["T"]+res[kl]["Q"]+res[kl]["S"] + forcing_cs_e
            ksum_s_e, ksum_i_e, r, ksum_p_e = lr(tas[:thr_yr], ksum[:thr_yr])[:4]
            ksum_s_l, ksum_i_l, r, ksum_p_l = lr(tas[thr_yr:], ksum[thr_yr:])[:4]
            
            # calculate the relative error
            re_err_e = np.round((sl_toa_e - ksum_s_e) / sl_toa_e * 100, decimals=1)
            re_err_l = np.round((sl_toa_l - ksum_s_l) / sl_toa_l * 100, decimals=1)
            pf_text = "pass"
            pf_color = "green"
            if ((np.abs(np.round(re_err_e)) > 15) | (np.abs(np.round(re_err_l)) > 15)):
                pf_text = "fail"
                pf_color = "red"
            # end if
            
            # generate a clear-sky Gregory plot
            fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))
            l11 = ax1.scatter(tas[:thr_yr], toa[:thr_yr], c="blue", marker="o", s=6, label=f"model years 1-{thr_yr}")
            l12 = ax1.scatter(tas[thr_yr:], toa[thr_yr:], c="red", marker="o", s=6, label=f"model years {thr_yr+1}-150")
            l13 = ax1.scatter(tas[:thr_yr], ksum[:thr_yr], c="magenta", marker="^", s=6, 
                              label=f"kernel years 1-{thr_yr}")
            l14 = ax1.scatter(tas[thr_yr:], ksum[thr_yr:], c="orange", marker="^", s=6, 
                              label=f"kernel years {thr_yr+1}-150")
            l21, = ax1.plot([0, np.max(tas)], [0 + forcing_cs_e, np.max(tas) * sl_toa_e + forcing_cs_e], c="black", 
                           linestyle="--", label="model early sl=" + str(np.round(sl_toa_e, decimals=2)) + 
                           " Wm$^{-2}$K$^{-1}$")
            l22, = ax1.plot([0, np.max(tas)], [0 + forcing_cs_l, np.max(tas) * sl_toa_l + forcing_cs_l], c="black", 
                           label="model late sl=" + str(np.round(sl_toa_l, decimals=2)) + " Wm$^{-2}$K$^{-1}$   " + 
                           "$\\Delta$=" + str(np.round(sl_toa_l-sl_toa_e, decimals=2)))
            l23, = ax1.plot([0, np.max(tas)], [0 + ksum_i_e, np.max(tas) * ksum_s_e + ksum_i_e], c="gray", 
                            linestyle="--", label="kernel early sl=" + str(np.round(fb_e, decimals=2)) + 
                            " Wm$^{-2}$K$^{-1}$   " + f"err={re_err_e}%")
            l24, = ax1.plot([0, np.max(tas)], [0 + ksum_i_l, np.max(tas) * ksum_s_l + ksum_i_l], c="gray", 
                            label="kernel late sl=" + str(np.round(fb_l, decimals=2)) + " Wm$^{-2}$K$^{-1}$   " + 
                            "$\\Delta$=" + str(np.round(fb_l-fb_e, decimals=2)) + f"   err={re_err_l}%")
            l1 = ax1.legend(handles=[l11, l12, l13, l14], loc="upper right")
            l2 = ax1.legend(handles=[l21, l22, l23, l24], loc="lower left")
            
            pf_l, = ax1.plot([], [], label=pf_text, linewidth=0)
            ax1.legend(handles=[pf_l], facecolor=pf_color, loc="center right")

            ax1.add_artist(l1)
            ax1.add_artist(l2)
            
            ax1.set_title("Clear-Sky Gregory Plot " + mod_n + " " + kl + " Kernels")
            ax1.set_xlabel(f"{t_var} in K")
            ax1.set_ylabel("TOA imbalance in Wm$^{-2}$")
            
            # ax1.set_xlim((-0.5, 9))
            # ax1.set_ylim((-1, 10))
            
            ax1.plot([0, np.max(tas)], [0, 0], linestyle="--", linewidth=0.75, c="gray")
            
            pl.show()
            pl.close()            
            
        # end for kl
        # raise Exception()
        
        # enter the values into the dictionaries ----------------------------------------------------------------------------
        mod_fb[mod_s] = sing_mod_fb
        k_sum[mod_s] = fb_sum
        
    # end for mod_s    


#%% generate a plot of the absolute and relative errors of the clear-sky kernel sum

n_mod = len(mod_lists["5"]) + len(mod_lists["6"])

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax2 = ax1.twinx()

rel_es = {}
rel_ls = {}

x = 0
for cmip_v in [5, 6]:
    rel_es[str(cmip_v)] = []
    rel_ls[str(cmip_v)] = []
    for i, mod_s in enumerate(mod_lists[str(cmip_v)]):
        
        mod = mod_dict[str(cmip_v)][mod_s]
        
        abs_e = mod_fb[mod_s]["e"] - k_sum[mod_s][kl]["e"]
        rel_e = np.abs(abs_e / mod_fb[mod_s]["e"] * 100)
        abs_l = mod_fb[mod_s]["l"] - k_sum[mod_s][kl]["l"]
        rel_l = np.abs(abs_l / mod_fb[mod_s]["l"] * 100)
        
        rel_es[str(cmip_v)].append(rel_e)
        rel_ls[str(cmip_v)].append(rel_l)
        
        ax1.scatter(x-0.1, abs_e, c="blue", marker="o", s=7.5)
        ax1.scatter(x-0.1, abs_l, c="red", marker="o", s=7.5)
        ax2.scatter(x+0.1, rel_e, c="blue", marker="^")
        ax2.scatter(x+0.1, rel_l, c="red", marker="^")    
        
        ax2.plot([x+0.1, x+0.1], [0, np.max([rel_e, rel_l])], c="gray", linestyle="--", linewidth=0.75)
        x += 1    
    # end for i, mod_s
    
    # convert the lists in the relative error dictionaries to numpy arrays
    rel_es[str(cmip_v)] = np.array(rel_es[str(cmip_v)])
    rel_ls[str(cmip_v)] = np.array(rel_ls[str(cmip_v)])
    
# end for cmip_v

ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [-15, -15], c="gray", linestyle="--", linewidth=0.75)
ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [0, 0], c="black", linestyle="--", linewidth=0.75)
ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [15, 15], c="red", linestyle="--", linewidth=1.75)
ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [30, 30], c="gray", linestyle="--", linewidth=0.75)
ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [45, 45], c="gray", linestyle="--", linewidth=0.75)
ax2.plot([0-n_mod*0.025, n_mod-1+n_mod*0.025], [60, 60], c="gray", linestyle="--", linewidth=0.75)

p1 = ax1.scatter([], [], c="blue", marker="o", label="early absolute")
p2 = ax1.scatter([], [], c="red", marker="o", label="late absolute")
p3 = ax1.scatter([], [], c="blue", marker="^", label="early relative")
p4 = ax1.scatter([], [], c="red", marker="^", label="late relative")

ax1.set_title("Clear-Sky Linearity Test " + kl + " Kernels")
ax1.set_ylabel("absolute error in Wm$^{-2}$K$^{-1}$")
ax2.set_ylabel("relative error in %")
ax1.set_xticks(np.arange(0, n_mod))
ax1.set_xticklabels(mod_lists_n["5"] + mod_lists_n["6"], rotation=90)

ax1.legend(loc="lower left", ncol=2)

ax1.set_xlim((0-n_mod*0.025, n_mod-1+n_mod*0.025))

ax1.set_ylim((-0.3, 0.9))
ax2.set_ylim((-30, 90))

pl.show()
pl.close()


#%% sort the models into the ones "passing" the clear-sky linearity test and into the ones that do not [and then sort
#   them along the sum of early and late error (?)]
cs_pass = dict()
n_cs_pass = dict()
cs_fail = dict()
n_cs_fail = dict()

for cmip_v in [5, 6]:
    cs_pass[str(cmip_v)] = ((np.round(rel_es[str(cmip_v)]) <= 15) & (np.round(rel_ls[str(cmip_v)]) <= 15))
    cs_fail[str(cmip_v)] = ((np.round(rel_es[str(cmip_v)]) > 15) | (np.round(rel_ls[str(cmip_v)]) > 15))
    
    n_cs_pass[str(cmip_v)] = len(rel_es[str(cmip_v)][cs_pass[str(cmip_v)]])
    n_cs_fail[str(cmip_v)] = len(rel_es[str(cmip_v)][cs_fail[str(cmip_v)]])
    
    print("\n\nPass (CMIP" + str(cmip_v) + ", " + str(n_cs_pass[str(cmip_v)]) + "/" + str(len(rel_es[str(cmip_v)])) + 
          "):")
    print(np.array(mod_lists_n[str(cmip_v)])[cs_pass[str(cmip_v)]])
    print(rel_es[str(cmip_v)][cs_pass[str(cmip_v)]])
    print("\n\nFail (CMIP" + str(cmip_v) + ", " + str(n_cs_fail[str(cmip_v)]) + "/" + str(len(rel_es[str(cmip_v)])) + 
          "):")
    print(np.array(mod_lists_n[str(cmip_v)])[cs_fail[str(cmip_v)]])
    print(rel_es[str(cmip_v)][cs_fail[str(cmip_v)]])
# end for cmip_v

# number of models the pass in total
n_pass = n_cs_pass["5"] + n_cs_pass["6"]


#%% plot the members that pass and the ones that do not separately in the same plot

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

x = 0
marker = {"5":"o", "6":"^"}
for cmip_v in [5, 6]:
    for i in np.arange(n_cs_pass[str(cmip_v)]):
        ax1.scatter(x, rel_es[str(cmip_v)][cs_pass[str(cmip_v)]][i], c="blue", zorder=101, marker=marker[str(cmip_v)])
        ax1.scatter(x, rel_ls[str(cmip_v)][cs_pass[str(cmip_v)]][i], c="red", zorder=101, marker=marker[str(cmip_v)])
        
        max_rel = np.max([rel_es[str(cmip_v)][cs_pass[str(cmip_v)]][i], rel_ls[str(cmip_v)][cs_pass[str(cmip_v)]][i]])
        
        ax1.plot([x, x], [0, max_rel], c="gray", linestyle="--", linewidth=0.75, zorder=100)

        x += 1
    # end for i
# end for cmip_v

for cmip_v in [5, 6]:
    for j in np.arange(n_cs_fail[str(cmip_v)]):
        ax1.scatter(x, rel_es[str(cmip_v)][cs_fail[str(cmip_v)]][j], c="blue", zorder=101, marker=marker[str(cmip_v)])
        ax1.scatter(x, rel_ls[str(cmip_v)][cs_fail[str(cmip_v)]][j], c="red", zorder=101, marker=marker[str(cmip_v)])
        
        max_rel = np.max([rel_es[str(cmip_v)][cs_fail[str(cmip_v)]][j], rel_ls[str(cmip_v)][cs_fail[str(cmip_v)]][j]])
        
        ax1.plot([x, x], [0, max_rel], c="gray", linestyle="--", linewidth=0.75, zorder=100)

        x += 1
    # end for i
# end for cmip_v

for y in np.arange(0, 51, 10):
    ax1.plot([0, n_mod - 1], [y, y], c="gray", linestyle="--", linewidth=0.75)
# end for y

ax1.plot([0, n_mod - 1], [15, 15], c="red", linestyle="--", linewidth=1.75)
ax1.plot([n_pass - 0.5, n_pass - 0.5], [0, 50], linewidth=0.75, c="black")

p1 = ax1.scatter([], [], marker=marker["5"], label="CMIP5, early", c="blue")
p2 = ax1.scatter([], [], marker=marker["5"], label="CMIP5, late", c="red")
p3 = ax1.scatter([], [], marker=marker["6"], label="CMIP6, early", c="blue")
p4 = ax1.scatter([], [], marker=marker["6"], label="CMIP6, late", c="red")
ax1.legend(handles=[p1, p2, p3, p4], loc="upper left")

ax1.set_ylim((-1, 51))

ax1.set_xticks(np.arange(0, n_mod, 1))
ax1.set_xticklabels(np.concatenate((np.array(mod_lists_n["5"])[cs_pass["5"]], np.array(mod_lists_n["6"])[cs_pass["6"]],
                                    np.array(mod_lists_n["5"])[cs_fail["5"]], np.array(mod_lists_n["6"])[cs_fail["6"]])),
                    rotation=90)

ax1.set_title(f"Clear-Sky Linearity Test {kl} Kernels\n{n_pass}/{n_mod} = {int(np.round(n_pass/n_mod * 100))}% Pass " +
              f" (CMIP5: {n_cs_pass['5']}/{len(rel_es['5'])}; CMIP6: {n_cs_pass['6']}/{len(rel_es['6'])})")
ax1.set_ylabel("relative error in %")

pl.savefig(pl_path + f"ClearSky_LinearityTest_{kl}_Kernels_{t_var}_Based_ThrYr{thr_str}.pdf", 
           dpi=300, bbox_inches="tight")
pl.savefig(pl_path + f"ClearSky_LinearityTest_{kl}_Kernels_{t_var}_Based_ThrYr{thr_str}.png", 
           dpi=300, bbox_inches="tight")

pl.show()
pl.close()
    

#%% store the lists of models that pass the test and that do not pass the test per kernel
pass_l_name = f"ClearSky_LinTest_Pass_{kl}_Kernels_{t_var}_Based_ThrYr{thr_str}.csv"
fail_l_name = f"ClearSky_LinTest_Fail_{kl}_Kernels_{t_var}_Based_ThrYr{thr_str}.csv"
pd.DataFrame(np.concatenate((np.array(mod_lists_n["5"])[cs_pass["5"]],
                             np.array(mod_lists_n["6"])[cs_pass["6"]]))).to_csv(list_path + pass_l_name)
pd.DataFrame(np.concatenate((np.array(mod_lists_n["5"])[cs_fail["5"]],
                             np.array(mod_lists_n["6"])[cs_fail["6"]]))).to_csv(list_path + fail_l_name)
