"""
Calculate the sea ice area (or extent) for G1 and G2 and generate the same plot as in panel (b) in Fig. 6 in Eiselt and
Graversen (2022), JCLI, and store that data for the figure.

Be sure to set the paths in the code block "set paths".
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


#%% choose if median with a certain quantile range shall be plotted or mean plus-minus standard deviation
avg = "mean"
quant = 0.75


#%% choose the variable (tas or ts)
fb_ts = "tas"


#%% set number of years to be investigated on the monthly basis
n_yrs = 150


#%% consider passing the clear-sky linearity test or not?
cslt = True
cslt_thr = 15  # --> CSLT relative error threshold in % --> 15% or 20% or... 


#%% remove models with strong cloud feedback?
rm_str_dcfb = True


#%% indicate which models should be excluded
excl = ""
excl = "outliers"
# excl_l = ["not high LR dFb", "not low LR dFb"]
# excl_l = ["not high LR dFb +", "not low LR dFb +"]
# excl_l = ["not high LR dFb -", "not low LR dFb -"]
# excl_l = ["not high LR dFb +", "not low LR dFb -"]
# excl_l = ["not high LR dFb -", "not low LR dFb +"]
# excl_l = ["not high LR dFb"]
# excl = "not low LR dFb"
# excl = "pos dECS"
# excl = "not high dECS"
# excl = "not low dECS"
# excl = "not high dTF"
# excl = "not low dTF"
# excl = "Fb not available"
# excl = "strong cloud dfb"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"


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


#%% set up some initial dictionaries and lists

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% add something to the file name concerning the CSLT
cslt_fadd = ""
if cslt:
    cslt_fadd = f"_CSLT{cslt_thr:02}{kl}"
# end if    


#%% set paths

# model data path
data_path5 = ""
data_path6 = ""

# clear sky linearity test path
cslt_path = "/Model_Lists/ClearSkyLinearityTest/{fb_ts}_Based/Thres_Year_20/"

# plot path
pl_path = ""

# generate the plot path
# os.makedirs(pl_path + "/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/PNG/", exist_ok=True)

out_path = "/G1_G2_Comp/"
os.makedirs(out_path, exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based_ThrYr20_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
sien_an_d = dict()
sian_an_d = dict()
sias_an_d = dict()
sian_mo_d = dict()
sias_mo_d = dict()

mods_d = dict()

pl_add = ""
tl_add = ""

for g12 in ["G1", "G2"]:
    
    #% select models to be excluded from the multi-model mean
    excl_fadd = ""
    excl_tadd = ""
    len_excl = 0
    excl_arr = np.array([""])
    
    if excl == "outliers":
        
        excl_arr = np.array(["GISS-E2-R", "MIROC-ES2L", "GISS-E2.2-G"])
    
        excl_fadd = "_WoOutl"
        len_excl = len(excl_arr)  # important!     
    
    # end if

    #% start a for-loop for the analysis
    i = 0
    mods = []
    # deg = 30  # latitude (in degree) north of which (and south of the negative of which) the mean is calculated
    cols = []
    
    # set up a counter for the models
    mod_count = 0
    
    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    sien_an_l = []
    sian_an_l = []
    sias_an_l = []
    sian_mo_l = []
    sias_mo_l = []
    
    # for CMIP5
    for mod, mod_n, mod_pl in zip(nl5.models, nl5.models_n, nl5.models_pl):
        
        if cslt:
            if not np.any(pass_cslt == mod_pl):
                print("\nJumping " + mod_pl + 
                      " because it fails the clear-sky linearity test (or data unavailable)...")
                continue
            # end if 
        # end if
        
        # exclude a chosen set of models
        if np.any(excl_arr == mod_pl):
            print("\nJumping " + mod_pl + " because it is in the exclude list (or data unavailable)...")
            continue
        # end if
        
        try:
            # load nc files
            # reg_nc = Dataset(glob.glob(data_path5 + f"/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + "*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
            si_nc = Dataset(glob.glob(data_path5 + f"/Data/{mod}/si_ext_area_*abrupt4xCO2*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/" + 
                                        f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])              
            # load the cloud feedback
            cl_fb_d = (i_fb_nc.variables["C_fb_l"][0] - i_fb_nc.variables["C_fb_e"][0]) 
            if (cl_fb_d > 0.5) & (rm_str_dcfb):
                print(f"\nJumping {mod_pl} because of large cloud dFb")
                continue
            # end if

            # load the lapse rate feedback
            lr_fb_d = (i_fb_nc.variables["LR_fb_l"][0] - i_fb_nc.variables["LR_fb_e"][0])
            
            # check the group (G1, G2) and add the data depending on that
            if (g12 == "G1") & (lr_fb_d >= 0.1):
                print(f"\nJumping {mod_pl} because of too large LR dFb for G1 ({lr_fb_d} Wm-2K-1)")
                continue
            if (g12 == "G2") & (lr_fb_d <= 0.5):
                print(f"\nJumping {mod_pl} because of too small LR dFb for G2 ({lr_fb_d} Wm-2K-1)")
                continue
            # end if            
            
            # load the values
            #sien_an_l.append(an_mean(si_nc.variables["si_ext_n"][:])[:n_yrs])
            sian_an_l.append(an_mean(si_nc.variables["si_area_n"][:])[:n_yrs])
            sias_an_l.append(an_mean(si_nc.variables["si_area_s"][:])[:n_yrs])
            sian_mo_l.append(si_nc.variables["si_area_n"][:n_yrs*12])
            sias_mo_l.append(si_nc.variables["si_area_s"][:n_yrs*12])
            
            i += 1
            mods.append(mod_pl)
            cols.append("blue")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except        
        
        mod_count += 1
    # end for mod
    
    # for CMIP6
    for mod, mod_n, mod_pl in zip(nl6.models, nl6.models_n, nl6.models_pl):
        
        if cslt:
            if not np.any(pass_cslt == mod_pl):
                print("\nJumping " + mod_pl + 
                      " because it fails the clear-sky linearity test (or data unavailable)...")
                continue
            # end if 
        # end if
        
        # exclude a chosen set of models
        if np.any(excl_arr == mod_pl):
            print("\nJumping " + mod_pl + " because it is in the exclude list (or data unavailable)...")
            continue
        # end if
        
        try:
            # load nc files
            # reg_nc = Dataset(glob.glob(data_path6 + f"/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + "*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
            si_nc = Dataset(glob.glob(data_path6 + f"/Data/{mod}/si_ext_area_*abrupt-4xCO2*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path6 + "/Outputs/" + 
                                        f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])
            
            # load the cloud feedback
            cl_fb_d = (i_fb_nc.variables["C_fb_l"][0] - i_fb_nc.variables["C_fb_e"][0]) 
            
            if (cl_fb_d > 0.5) & (rm_str_dcfb):
                print(f"\nJumping {mod_pl} because of large cloud dFb")
                continue
            # end if
            
            # load the lapse rate feedback
            lr_fb_d = (i_fb_nc.variables["LR_fb_l"][0] - i_fb_nc.variables["LR_fb_e"][0])
            
            # check the group (G1, G2) and add the data depending on that
            if (g12 == "G1") & (lr_fb_d >= 0.1):
                print(f"\nJumping {mod_pl} because of too large LR dFb for G1")
                continue
            if (g12 == "G2") & (lr_fb_d <= 0.5):
                print(f"\nJumping {mod_pl} because of too small LR dFb for G2")
                continue
            # end if                
        
            # load the values
            if mod == "AWI-CM-1-1-MR":
                # sien_an_l.append(si_nc.variables["si_ext_n"][:][:n_yrs])
                sian_an_l.append(si_nc.variables["si_area_n"][:][:n_yrs])
                sias_an_l.append(si_nc.variables["si_area_s"][:][:n_yrs])
            else:                    
                # sien_an_l.append(an_mean(si_nc.variables["si_ext_n"][:])[:n_yrs])
                sian_an_l.append(an_mean(si_nc.variables["si_area_n"][:])[:n_yrs])
                sias_an_l.append(an_mean(si_nc.variables["si_area_s"][:])[:n_yrs])
                sian_mo_l.append(si_nc.variables["si_area_n"][:n_yrs*12])
                sias_mo_l.append(si_nc.variables["si_area_s"][:n_yrs*12])
            # end if else               
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod
    
    # sien_an_d[g12] = np.array(sien_an_l)
    sian_an_d[g12] = np.array(sian_an_l)
    sias_an_d[g12] = np.array(sias_an_l)
    sian_mo_d[g12] = np.array(sian_mo_l)
    sias_mo_d[g12] = np.array(sias_mo_l)
    
    mods_d[g12] = mods
# end for g12


#%% calculate the means over the model groups
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


#%% calculate seasonal means for the G1 and G2 means
g1_avg_se = sea_mean(g1_avg_mo)
g2_avg_se = sea_mean(g2_avg_mo)
g1s_avg_se = sea_mean(g1s_avg_mo)
g2s_avg_se = sea_mean(g2s_avg_mo)


#%% compare seasonal means NH
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

pl.savefig(pl_path + f"/PNG/SIA_NH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/SIA_NH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

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

pl.savefig(pl_path + f"/PNG/SIA_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/SIA_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% NH and SH combined
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3), sharey=True, sharex=True)

axes[0].plot(g1_avg, c="blue", label="G1")
axes[0].plot(g2_avg, c="red", label="G2")
axes[0].fill_between(np.arange(len(g1_avg)), g1_avg - g1_lo, g1_avg + g1_hi, facecolor="blue", alpha=0.15)
axes[0].fill_between(np.arange(len(g2_avg)), g2_avg - g2_lo, g2_avg + g2_hi, facecolor="red", alpha=0.15)
axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[0].legend()

axes[1].plot(g1s_avg, c="blue")
axes[1].plot(g2s_avg, c="red")
axes[1].fill_between(np.arange(len(g1s_avg)), g1s_avg - g1s_lo, g1s_avg + g1s_hi, facecolor="blue", alpha=0.15)
axes[1].fill_between(np.arange(len(g2s_avg)), g2s_avg - g2s_lo, g2s_avg + g2s_hi, facecolor="red", alpha=0.15)
axes[1].axhline(y=0, c="black", linewidth=0.5)

axes[0].set_xlabel("Year of simulation")
axes[1].set_xlabel("Year of simulation")
axes[0].set_ylabel("Sea-ice area 10$^6$ km$^2$")

axes[0].set_title("Northern Hemisphere")
axes[1].set_title("Southern Hemisphere")

fig.suptitle("G1 and G2 sea-ice area")
fig.subplots_adjust(wspace=0.05, top=0.825)

axes[0].text(0, 1, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1].text(0, 1, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")

pl.savefig(pl_path + f"/PNG/SIA_NH_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/SIA_NH_SH_AnMean_G1_G2_{avg}{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store these data as netcdf file
f = Dataset(out_path + f"G1_G2_NH_SeaIce{cslt_fadd}.nc", "w", format="NETCDF4")

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

sia_g1_nc.description = "northern hemispheric mean sea ice area G1"
sia_g2_nc.description = "northern hemispheric mean sea ice area G2"

p_g1_g2_nc.description = "p-value for group mean difference Wald t-test between G1 and G2"

# date of file creation
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the file
f.close()



