"""
Calculate, plot, and store the polar amplification factor (PAF) for G1 and G2.

Used in the script for plotting Fig. 6 in Eiselt and Graversen (2022), JCLI.

Be sure to correctly set the paths in code block "set paths". Possibly also change the paths in the for-loop.
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
# from sklearn.linear_model import LinearRegression
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
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% choose the variable (tas or ts)
t_var = "ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


#%% set number of years to be investigated on the monthly basis
fyrs = 149


#%% set the region corrdinates
x1s, x2s = [0], [360]
y1s, y2s = [75], [90]


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

direc_5 = "/media/kei070/Work Disk 2"  # SETS BASIC DATA DIRECTORY
direc_6 = "/media/kei070/Seagate"  # SETS BASIC DATA DIRECTORY


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
kl = "P18"
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
# os.makedirs(out_path, exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based_ThrYr20_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")



#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
dts_glob_d = dict()
dts_arct_d = dict()
dts_yrs_glob_d = dict()
dts_yrs_arct_d = dict()
mods_d = dict()

# addendum for plot title and name for the group threshold sensitivity analysis
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
    dts_glob_l = []
    dts_arct_l = []
    dts_yrs_glob_l = []
    dts_yrs_arct_l = []
    
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
            ts_nc = Dataset(glob.glob(data_path5 + f"/Data/{mod}/d{t_var}_*abrupt4xCO2*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/"
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
            dts = np.mean(ts_nc.variables[t_var + "_ch"][:], axis=0)
            dts_yrs = ts_nc.variables[t_var + "_ch"][:, :fyrs, :, :]
            lat = ts_nc.variables["lat"][:]
            lon = ts_nc.variables["lon"][:]
            dts_glob = glob_mean(dts, lat, lon)
            dts_arct = region_mean(dts, x1s, x2s, y1s, y2s, lat, lon)
            dts_yrs_glob = glob_mean(dts_yrs, lat, lon)
            dts_yrs_arct = region_mean(dts_yrs, x1s, x2s, y1s, y2s, lat, lon)            
            dts_glob_l.append(dts_glob)
            dts_arct_l.append(dts_arct)
            dts_yrs_glob_l.append(dts_yrs_glob)
            dts_yrs_arct_l.append(dts_yrs_arct)
            
            i += 1
            mods.append(mod_pl)
            cols.append("blue")
        except:
            print("\n\nJumping " + mod_pl + " because data are not (yet) available...\n\n")
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
            ts_nc = Dataset(glob.glob(data_path6 + f"/Data/{mod}/d{t_var}_*abrupt-4xCO2*.nc")[0])
            
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
            dts = np.mean(ts_nc.variables[t_var + "_ch"][:], axis=0)
            dts_yrs = ts_nc.variables[t_var + "_ch"][:, :fyrs, :, :]
            lat = ts_nc.variables["lat"][:]
            lon = ts_nc.variables["lon"][:]
            dts_glob = glob_mean(dts, lat, lon)
            dts_arct = region_mean(dts, x1s, x2s, y1s, y2s, lat, lon)
            dts_yrs_glob = glob_mean(dts_yrs, lat, lon)
            dts_yrs_arct = region_mean(dts_yrs, x1s, x2s, y1s, y2s, lat, lon)            
            dts_glob_l.append(dts_glob)
            dts_arct_l.append(dts_arct)
            dts_yrs_glob_l.append(dts_yrs_glob)
            dts_yrs_arct_l.append(dts_yrs_arct)           
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod
    dts_glob = np.array(dts_glob_l)
    dts_arct = np.array(dts_arct_l)
    dts_glob_d[g12] = dts_glob
    dts_arct_d[g12] = dts_arct
    dts_yrs_glob = np.array(dts_yrs_glob_l)
    dts_yrs_arct = np.array(dts_yrs_arct_l)
    dts_yrs_glob_d[g12] = dts_yrs_glob
    dts_yrs_arct_d[g12] = dts_yrs_arct
    mods_d[g12] = mods
# end for excl


#%% calculate the means over the model groups
g1_glob_mean = np.mean(dts_glob_d["G1"], axis=0)
g2_glob_mean = np.mean(dts_glob_d["G2"], axis=0)
g1_arct_mean = np.mean(dts_arct_d["G1"], axis=0)
g2_arct_mean = np.mean(dts_arct_d["G2"], axis=0)
g1_yrs_glob_mean = np.mean(dts_yrs_glob_d["G1"], axis=0)
g2_yrs_glob_mean = np.mean(dts_yrs_glob_d["G2"], axis=0)
g1_yrs_arct_mean = np.mean(dts_yrs_arct_d["G1"], axis=0)
g2_yrs_arct_mean = np.mean(dts_yrs_arct_d["G2"], axis=0)

g1_pafs = dts_arct_d["G1"] / dts_glob_d["G1"]
g2_pafs = dts_arct_d["G2"] / dts_glob_d["G2"]
g1_pafs_mean = np.mean(g1_pafs, axis=0)
g1_pafs_std = np.std(g1_pafs, axis=0)
g2_pafs_mean = np.mean(g2_pafs, axis=0)
g2_pafs_std = np.std(g2_pafs, axis=0)


#%% perform the t-test to see if the means are different
ttest = ttest_ind(g1_pafs, g2_pafs, axis=0)


#%% reshape the monthly means for easier plotting
g1_yrs_glob_mean_rs = np.zeros(12*np.shape(g2_yrs_arct_mean)[-1])
g2_yrs_glob_mean_rs = np.zeros(12*np.shape(g2_yrs_arct_mean)[-1])
g1_yrs_arct_mean_rs = np.zeros(12*np.shape(g2_yrs_arct_mean)[-1])
g2_yrs_arct_mean_rs = np.zeros(12*np.shape(g2_yrs_arct_mean)[-1])

yr1 = 0
yr2 = 12
for yr in np.arange(np.shape(g2_yrs_arct_mean)[-1]):
    g1_yrs_glob_mean_rs[yr1:yr2] = g1_yrs_glob_mean[:, yr]
    g2_yrs_glob_mean_rs[yr1:yr2] = g2_yrs_glob_mean[:, yr]
    g1_yrs_arct_mean_rs[yr1:yr2] = g1_yrs_arct_mean[:, yr]
    g2_yrs_arct_mean_rs[yr1:yr2] = g2_yrs_arct_mean[:, yr]
    yr1 += 12
    yr2 += 12
# end for yr    


#%% plot the polar amplification factor for both groups (G1 and G2)
fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

# axes.plot(g1_arct_mean / g1_glob_mean, c="blue", label="G1")
# axes.plot(g2_arct_mean / g2_glob_mean, c="red", label="G2")

axr = axes.twinx()

axes.plot(g1_pafs_mean, c="blue", label="G1")
axes.fill_between(np.arange(len(g1_pafs_mean)), g1_pafs_mean - g1_pafs_std, g1_pafs_mean + g1_pafs_std, facecolor="blue",
                  alpha=0.25)
axes.plot(g2_pafs_mean, c="red", label="G2")
axes.fill_between(np.arange(len(g2_pafs_mean)), g2_pafs_mean - g2_pafs_std, g2_pafs_mean + g2_pafs_std, facecolor="red",
                  alpha=0.25)

axr.plot(ttest.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=1, c="gray", zorder=1)

# for the legend
p1, = axr.plot([], [], c="blue", label="G1")
p2, = axr.plot([], [], c="red", label="G2")
p3, = axr.plot([], [], c="black", linewidth=0.5, label="p-value")
l1 = axr.legend(handles=[p1, p2], fontsize=fsz, loc=(0.08, 0.015), framealpha=0.9)
axr.legend(handles=[p3], fontsize=fsz, loc=(0.32, 0.015), framealpha=0.9)
axr.add_artist(l1)

"""
for excl, col in zip(excl_l, ["blue", "red"]):
    for ar, gl in zip(dts_arct_d[excl], dts_glob_d[excl]):
        axes.plot(ar / gl, c=col, linewidth=0.3)
    # end for ar, gl
# end for excl, col    
"""
axes.set_title(f"PAF for G1 and G2{tl_add}", fontsize=fsz+1)
axes.set_xlabel("Years since branching of abrupt-4xCO2", fontsize=fsz)
axes.set_ylabel("PAF", fontsize=fsz)
axes.tick_params(labelsize=fsz)

axr.tick_params(labelsize=fsz)

pl.savefig(pl_path + f"/PNG/PAF_AnMean_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/PAF_AnMean_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the first three years of month mean Arctic and globel warming for G1 and G2
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(12, 6))

axes.plot(g1_yrs_glob_mean_rs, c="blue", label="G1 global", linestyle="--")
axes.plot(g1_yrs_arct_mean_rs, c="blue", label="G1 Arctic", linewidth=0.5)
axes.plot(g2_yrs_glob_mean_rs, c="red", label="G2 global", linestyle="--")
axes.plot(g2_yrs_arct_mean_rs, c="red", label="G2 Arctic", linewidth=0.5)

axes.legend()
axes.set_title(f"$\Delta${t_var}")
axes.set_xlabel("Mothns since branching of abrupt-4xCO2")
axes.set_ylabel(f"$\Delta${t_var} in K")

pl.show()
pl.close()


#%% plot the monthly PAF for G1 and G2
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(20, 6))

axes.plot(g1_yrs_arct_mean_rs / g1_yrs_glob_mean_rs, c="blue", label="G1 (weak $\Delta \\alpha _{LR}$)", linewidth=0.7)
axes.plot(g2_yrs_arct_mean_rs / g2_yrs_glob_mean_rs, c="red", label="G2 (strong $\Delta \\alpha _{LR}$)", linewidth=0.7)
axes.plot(np.arange(0, (fyrs+1)*12, 12), g1_arct_mean[:fyrs+1] / g1_glob_mean[:fyrs+1], c="blue", linestyle="--")
axes.plot(np.arange(0, (fyrs+1)*12, 12), g2_arct_mean[:fyrs+1] / g2_glob_mean[:fyrs+1], c="red", linestyle="--")

axes.legend()
axes.set_title(f"Polar Amplification Factor Based on $\Delta${t_var}")
axes.set_xlabel("Years since branching of abrupt-4xCO2")
axes.set_ylabel("PAF")

axes.set_xticks(np.arange(0, 12*fyrs, 12))
axes.set_xticklabels(np.arange(fyrs), rotation=90, fontsize=8)

for i in np.arange(0, (fyrs+1)*12, 12):
    axes.axvline(x=i, c="gray", linewidth=0.5)
# end for i    

axes.set_xlim((-1*12, 150*12))

pl.savefig(pl_path + f"/PDF/PAF_G1_G2_Monthly{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/PAF_G1_G2_Monthly{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the monthly PAF for G1 and G2 in the 1st year
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(20, 6))

axes.plot(g1_yrs_arct_mean_rs[:12] / g1_yrs_glob_mean_rs[:12], c="blue", label="G1 (weak $\Delta \\alpha _{LR}$)")
axes.plot(g2_yrs_arct_mean_rs[:12] / g2_yrs_glob_mean_rs[:12], c="red", label="G2 (strong $\Delta \\alpha _{LR}$)")

axes.legend()
axes.set_title(f"Polar Amplification Factor Based on $\Delta${t_var}")
axes.set_xlabel("Months since branching of abrupt-4xCO2")
axes.set_ylabel("PAF")

# pl.savefig(pl_path + "/PDF/PAF_G1_G2_Monthly.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/PAF_G1_G2_Monthly.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store these data as netcdf file
f = Dataset(out_path + f"G1_G2_PAF{cslt_fadd}.nc", "w", format="NETCDF4")

# create the dimensions
f.createDimension("years", 150)
f.createDimension("mods_G1", len(mods_d["G1"]))
f.createDimension("mods_G2", len(mods_d["G2"]))

# create the variables
mods_g1_nc = f.createVariable("mods_G1", "S25", "mods_G1")
mods_g2_nc = f.createVariable("mods_G2", "S25", "mods_G2")
paf_g1_nc = f.createVariable("PAF_G1", "f4", ("mods_G1", "years"))
paf_g2_nc = f.createVariable("PAF_G2", "f4", ("mods_G2", "years"))

p_g1_g2_nc = f.createVariable("p_G1_G2", "f4", "years")

# pass the data into the variables
mods_g1_nc[:] = np.array(mods_d["G1"])
mods_g2_nc[:] = np.array(mods_d["G2"])

paf_g1_nc[:] = g1_pafs
paf_g2_nc[:] = g2_pafs

p_g1_g2_nc[:] = ttest.pvalue

# descriptions of the variables
mods_g1_nc.description = "names of the model in G1 (weak lapse rate feedback change)"
mods_g2_nc.description = "names of the model in G2 (weak lapse rate feedback change)"

paf_g1_nc.description = f"Polar Amplification Index (Arctic/Global {t_var}) G1"
paf_g2_nc.description = f"Polar Amplification Index (Arctic/Global {t_var}) G2"

p_g1_g2_nc.description = "p-value for group mean difference Wald t-test between G1 and G2"

# date of file creation
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the file
f.close()


#%% plot the monthly PAF for individual models in the 1st year
"""
i = 8
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(20, 6))

for i in np.arange(9):
    axes.plot(dts_yrs_arct_d['not high LR dFb'][i, :, 0] / dts_yrs_glob_d['not high LR dFb'][i, :, 0], c="blue")
# end for i    

axes.legend()
axes.set_title(f"Polar Amplification Factor Based on $\Delta${t_var} for G1")
axes.set_xlabel("Months since branching of abrupt-4xCO2")
axes.set_ylabel("PAF")

# pl.savefig(pl_path + "/PDF/PAF_G1_G2_Monthly.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/PAF_G1_G2_Monthly.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the monthly global and Arctic warming for individual models in the 1st year
i = 5
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(20, 6))


axes.plot(dts_yrs_arct_d['not high LR dFb'][i, :, 0], c="black")
axes.plot(dts_yrs_glob_d['not high LR dFb'][i, :, 0], c="black", linestyle="--")

# axes.legend()
axes.set_title(f"Polar Amplification Factor Based on $\Delta${t_var} for {mods_d['not high LR dFb'][i]}")
axes.set_xlabel("Months since branching of abrupt-4xCO2")
axes.set_ylabel("PAF")

# pl.savefig(pl_path + "/PDF/PAF_G1_G2_Monthly.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/PAF_G1_G2_Monthly.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()

"""