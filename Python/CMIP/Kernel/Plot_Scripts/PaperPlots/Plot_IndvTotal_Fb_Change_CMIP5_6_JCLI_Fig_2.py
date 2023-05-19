"""
Generates Figure 2 in Eiselt and Graversen (2022), JCLI.

Scatters all individual kernel method derived feedback changes against the total Gregory method derived feedback changes
in two four-panel plots.

Furthermore plots the LR v. the SA feedback change.

Be sure to adjust the paths in code block "set paths".
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
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% choose the feedback
try:
    fb_s = sys.argv[1]
except:    
    fb_s = "LR"
    # fb_s = "C"
    # fb_s = "C_lw"
    # fb_s = "C_sw"
    # fb_s = "S"
    # fb_s = "Q"
    # fb_s = "Pl"
    # fb_s = "LR+Q"
# end try except    

# note that it is important that the cloud feedback is the first in the list!!!
fb_s_l = ["C", "LR", "S", "Q", "C_lw", "C_sw", "Pl", "LR+Q"]


#%% set the temperature variable that was used to calculate the feedbacks (tas or ts)
t_var = "tas"


#%% consider passing the clear-sky linearity test or not?
cslt = True
cslt_thr = 15


#%% indicate which models should be excluded
try:
    excl = sys.argv[2]
except:    
    excl = ""
    # excl = "strong cloud dfb and outliers"  # cloud feedback change > 0.5 and MIROC-ES2L
    # excl = "moderate cloud dfb"
    # excl = "CMIP6"
    # excl = "CMIP5"
    excl = "outliers"
# end try except


#%% set up some initial dictionaries and lists

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% load the namelist
import Namelists.Namelist_CMIP5 as nl5
a4x5 = "abrupt4xCO2"

import Namelists.Namelist_CMIP6 as nl6
a4x6 = "abrupt-4xCO2"


#%% set up some dictionaries and lists with names
fb_dic = {"LR":["Lapse Rate", "Lapse rate", "LapseRate"], 
          "C":["Cloud", "Cloud", "Cloud"], 
          "C_lw":["LW Cloud", "lw cloud", "CloudLW"], 
          "C_sw":["SW Cloud", "sw cloud", "CloudSW"], 
          "Pl":["Planck", "Planck", "Planck"], 
          "Q":["Water Vapour", "Water vapour", "WaterVapour"],
          "S":["Surface Albedo", "Surface albedo", "SurfaceAlbedo"],
          "LR+Q":["LR+WV", "LR+WV", "LRpWV"]}


fb_tn = fb_dic[fb_s][0] 
fb_labn = fb_dic[fb_s][1]
fb_fn = fb_dic[fb_s][2]

# do not change:
fb_col = "red"


#%% set the legend location depending on which feedback is chosen (only water vapour is different)
leg_loc1 = "upper left" 
leg_loc2 = "lower right"

if (fb_s == "Q") | (fb_s == "C_lw"):
    leg_loc1 = "lower left" 
    leg_loc2 = "upper right"
# end if    


#%% list of model names
all_mod = np.array(['ACCESS1.0', 'ACCESS1.3', 'BCC-CSM1.1', 'BCC-CSM1.1(m)', 'BNU-ESM', 'CanESM2', "CCSM4", 'CNRM-CM5', 
                    "FGOALS-s2", 'GFDL-CM3', "GFDL-ESM2G", 'GFDL-ESM2M', 'GISS-E2-H', 'GISS-E2-R', 'HadGEM2-ES', 
                    "INMCM4", 'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', "MIROC-ESM", 'MPI-ESM-LR', 'MPI-ESM-MR', 
                    'MPI-ESM-P', 'MRI-CGCM3', 'NorESM1-M',
                    'ACCESS-CM2', 'ACCESS-ESM1.5', 'AWI-CM-1.1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1.0', 'CanESM5', 
                    "CAS-ESM2.0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-ESM2", 
                    'CMCC-CM2-SR5', 'CNRM-CM6.1', "CNRM-CM6.1-HR", 'CNRM-ESM2.1', 'E3SM-1.0', 'EC-Earth3', 
                    'EC-Earth3-Veg', "EC-Earth3-AerChem", "FIO-ESM2.0", 'FGOALS-f3-L', 'FGOALS-g3', 'GFDL-CM4', 
                    'GFDL-ESM4', 'GISS-E2.1-G', 'GISS-E2.1-H', 'GISS-E2.2-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 
                    "IITM-ESM",  'INM-CM4.8', 'INM-CM5', 'IPSL-CM6A-LR', "KACE-1.0-G", "KIOST-ESM", 'MIROC-ES2L', 
                    'MIROC6', 'MPI-ESM1.2-HAM', 'MPI-ESM1.2-HR', 'MPI-ESM1.2-LR', 'MRI-ESM2.0', "NESM3", 'NorCPM1', 
                    "NorESM2-LM", "NorESM2-MM", 'SAM0-UNICON', 'TaiESM1', 'UKESM1.0-LL']) 


#%% select models to be excluded from the multi-model mean
excl_fadd = ""
excl_tadd = ""
len_excl = 0
excl_arr = np.array([""])

if excl == "neg abs dTF":
    excl_arr = np.array(["BNU-ESM", "GISS-E2.2-G", "MIROC6", "GFDL-ESM4", "MIROC-ES2L", "CAMS-CSM1.0", "GISS-E2.1-H", 
                         "FGOALS-s2", "CMCC-ESM2", "CNRM-ESM2.1", "CMCC-CM2-SR5", "FIO-ESM2.0", "CNRM-CM6.1", 
                         "CNRM-CM6.1-HR", "EC-Earth3-Veg"])

    excl_fadd = "_Pos_AbsdTF"
    len_excl = len(excl_arr)  # important!
if excl == "neg abs dTF & strong cloud":
    excl_arr = np.array(["BNU-ESM", "GISS-E2.2-G", "MIROC6", "GFDL-ESM4", "MIROC-ES2L", "CAMS-CSM1.0", "GISS-E2.1-H", 
                         "FGOALS-s2", "CMCC-ESM2", "CNRM-ESM2.1", "CMCC-CM2-SR5", "FIO-ESM2.0", "CNRM-CM6.1", 
                         "CNRM-CM6.1-HR", "EC-Earth3-Veg", "MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", 
                         "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'MIROC-ES2L', "CNRM-CM6.1-HR", 'GFDL-ESM4'])

    excl_fadd = "_Pos_AbsdTF_Excl_Strong_Cloud_dFb"
    len_excl = len(excl_arr)  # important!
    
if excl == "neg dTF":
    excl_arr = np.array(['MIROC-ES2L', "CNRM-CM6.1-HR", 'GFDL-ESM4'])

    excl_fadd = "_Pos_dTF"
    len_excl = len(excl_arr)  # important!
    
if excl == "<[3,4]K dTas<":
    excl_arr = np.array(['ACCESS1.0', 'BCC-CSM1.1', 'BCC-CSM1.1(m)', 'BNU-ESM', 'CanESM2', "CCSM4", 
                         'CNRM-CM5', "FGOALS-s2", 'GFDL-CM3', 'GISS-E2-R', 
                         'HadGEM2-ES', "INMCM4", 'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5', "MIROC-ESM", 'MPI-ESM-LR', 
                         'MPI-ESM-MR', 'MPI-ESM-P', 'MRI-CGCM3',
                         'ACCESS-CM2', 'ACCESS-ESM1.5', 'AWI-CM-1.1-MR', 'BCC-ESM1', 'CAMS-CSM1.0',
                         'CanESM5', "CAS-ESM2.0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", 
                         'CMCC-CM2-SR5', "CMCC-ESM2", 'CNRM-CM6.1', "CNRM-CM6.1-HR", 'CNRM-ESM2.1', 'E3SM-1.0', 
                         'EC-Earth3', 'EC-Earth3-Veg', "EC-Earth3-AerChem", "FIO-ESM2.0", 'GFDL-CM4', 
                         'GFDL-ESM4', 'GISS-E2.1-G', 'GISS-E2.1-H', 'GISS-E2.2-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 
                         "IITM-ESM",  'IPSL-CM6A-LR', "KACE-1.0-G", 'MIROC-ES2L', 
                         'MIROC6', 'MPI-ESM1.2-HAM', 'MPI-ESM1.2-LR', 'MRI-ESM2.0', "NESM3", 'NorCPM1',
                         'SAM0-UNICON', 'TaiESM1', 'UKESM1.0-LL'])

    excl_fadd = "_MidRange_dTas"
    len_excl = len(excl_arr)  # important!

if excl == "Fb not available":
    excl_arr = np.array(["FGOALS-s2", "GFDL-ESM2G", "INMCM4", "MIROC-ESM",
                         "CAS-ESM2.0", "CNRM-CM6.1-HR", "IITM-ESM", "KACE-1.0-G", "NESM3"])
    excl_fadd = "_Excl_FbNotAvail"
    len_excl = len(excl_arr)  # important!
    
if excl == "strong cloud dfb":
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
    excl_fadd = "_Excl_Strong_Cloud_dFb"
    len_excl = len(excl_arr)  # important!
    
if excl == "strong cloud dfb and outliers":
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2",
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4', "MIROC-ES2L", "GISS-E2.2-G"])
    excl_fadd = "_Excl_Strong_Cloud_dFb_WoOutl"
    len_excl = len(excl_arr)  # important!    
    
if excl == "moderate cloud dfb":
    excl_arr = np.array(['ACCESS1.0', 'ACCESS1.3', 'BCC-CSM1.1', 'BCC-CSM1.1(m)', 'BNU-ESM',
                         'CanESM2', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-H', 'GISS-E2-R', 'HadGEM2-ES',
                         'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC5',
                         'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM-P', 'MRI-CGCM3', 'NorESM1-M',
                         'ACCESS-CM2', 'ACCESS-ESM1.5', 'AWI-CM-1.1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1.0',
                         'CanESM5', 'CMCC-CM2-SR5', 'CNRM-CM6.1', 'CNRM-ESM2.1', 'E3SM-1.0', 'EC-Earth3', 
                         'EC-Earth3-Veg', 'FGOALS-f3-L', 'FGOALS-g3', 'GFDL-CM4', 'GFDL-ESM4', 'GISS-E2.1-G', 
                         'GISS-E2.1-H', 'GISS-E2.2-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'INM-CM4.8', 'INM-CM5',
                         'IPSL-CM6A-LR', 'MIROC-ES2L', 'MIROC6', 'MPI-ESM1.2-HR', 'MPI-ESM1.2-LR', 'MRI-ESM2.0',
                         'NorCPM1', 'SAM0-UNICON', 'TaiESM1', 'UKESM1.0-LL'])
    excl_fadd = "_Strong_Cloud_dFb"
    len_excl = len(excl_arr)  # important!    
    
elif excl == "CMIP5":
    excl_arr = np.array(['ACCESS1.0', 'ACCESS1.3', 'BCC-CSM1.1', 'BCC-CSM1.1(m)', 'BNU-ESM', 'CanESM2', "CCSM4", 
                         'CNRM-CM5', "FGOALS-s2", 'GFDL-CM3', "GFDL-ESM2G", 'GFDL-ESM2M', 'GISS-E2-H', 'GISS-E2-R', 
                         'HadGEM2-ES', "INMCM4", 'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC-ESM', 'MIROC5', 'MPI-ESM-LR', 
                         'MPI-ESM-MR', 'MPI-ESM-P', 'MRI-CGCM3', 'NorESM1-M'])
    excl_fadd = "_CMIP6"
    excl_tadd = " CMIP6"
    len_excl = len(excl_arr)  # important!    

elif excl == "CMIP6":
    excl_arr = np.array(['ACCESS-CM2', 'ACCESS-ESM1.5', 'AWI-CM-1.1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1.0', 
                         'CanESM5', "CAS-ESM2.0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", 
                         "CMCC-ESM2", 'CMCC-CM2-SR5', 'CNRM-CM6.1', "CNRM-CM6.1-HR", 'CNRM-ESM2.1', 'E3SM-1.0', 
                         'EC-Earth3', 'EC-Earth3-Veg', "EC-Earth3-AerChem", "FIO-ESM2.0", 'FGOALS-f3-L', 'FGOALS-g3', 
                         'GFDL-CM4', 'GFDL-ESM4', 'GISS-E2.1-G', 'GISS-E2.1-H', 'GISS-E2.2-G', 'HadGEM3-GC31-LL', 
                         'HadGEM3-GC31-MM', "IITM-ESM",  'INM-CM4.8', 'INM-CM5', 'IPSL-CM6A-LR', "KACE-1.0-G", 
                         "KIOST-ESM", 'MIROC-ES2L', 'MIROC6', 'MPI-ESM1.2-HAM', 'MPI-ESM1.2-HR', 'MPI-ESM1.2-LR', 
                         'MRI-ESM2.0', "NESM3", 'NorCPM1', "NorESM2-LM", "NorESM2-MM", 'SAM0-UNICON', 'TaiESM1', 
                         'UKESM1.0-LL'])
    excl_fadd = "_CMIP5"
    excl_tadd = " CMIP5"
    len_excl = len(excl_arr)  # important!    
    
elif excl == "low dFb":
    
    strong_dfb = np.array(['GFDL-CM3', 'BCC-CSM2-MR', 'ACCESS1.3', 'CCSM4', 'MPI-ESM1.2-HR', 'MRI-ESM2.0',
                           'ACCESS-ESM1.5', 'FGOALS-f3-L', 'CESM2', 'INM-CM4.8', 'CESM2-FV2',
                           'CAS-ESM2.0', 'NorESM2-MM', 'GFDL-CM4', 'NorESM1-M', 'GFDL-ESM2G', 'GISS-E2-R', 'NorESM2-LM'])
    excl_arr = np.setdiff1d(all_mod, strong_dfb)

    excl_fadd = "_ExWeakdFb"
    excl_tadd = " Moderate & Strong dFb"
    len_excl = len(excl_arr)  # important!

elif excl == "outliers":
    
    excl_arr = np.array(["GISS-E2-R", "MIROC-ES2L", "GISS-E2.2-G"])

    excl_fadd = "_WoOutl"
    len_excl = len(excl_arr)  # important!     
# end if elif


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
cslt_path = "/Model_Lists/ClearSkyLinearityTest/{t_var}_Based/Thres_Year_20/"

# plot path
pl_path = ""
pl_path2 = ""

# generate the plot path
os.makedirs(pl_path + "PDF/", exist_ok=True)
os.makedirs(pl_path + "PNG/", exist_ok=True)
os.makedirs(pl_path2 + "PDF/", exist_ok=True)
os.makedirs(pl_path2 + "PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{t_var}_Based_ThrYr20_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]


#%% start a for-loop for the analysis
i_fb_e = dict()
i_fb_l = dict()
di_fb = dict()

wea_dcfb = dict()
wea_ind_d_m = dict()
wea_ind_e_m = dict()
wea_ind_l_m = dict()
wea_fb_d_m = dict()

sl_all, yi_all, r_all, p_all = dict(), dict(), dict(), dict()
sl_wea, yi_wea, r_wea, p_wea = dict(), dict(), dict(), dict()

for fb_s in fb_s_l:

    fb_e = []
    fb_l = []
    i_fb_e[fb_s] = []
    i_fb_l[fb_s] = []
    t_fb_e = []
    t_fb_l = []
    
    # for Monte Carlo significance test
    i_fb_p = []
    t_fb_p = []
    fb_p = []
    
    i = 0
    mods = []
    # deg = 30  # latitude (in degree) north of which (and south of the negative of which) the mean is calculated
    cols = []
    
    # set up a counter for the models
    mod_count = 0
    
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
            ind_fb_nc = Dataset(glob.glob(data_path5 + 
                                          f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{t_var}_Based/*{mod}.nc")[0])
            greg_nc = Dataset(glob.glob(data_path5 + f"/TOA_Imbalance_piC_Run/{a4x5}/{t_var}_Based/TOA*{mod}.nc")[0])
            
            # load the feedback values
            i_fb_e[fb_s].append(ind_fb_nc.variables[fb_s + "_fb_e"][0])
            i_fb_l[fb_s].append(ind_fb_nc.variables[fb_s + "_fb_l"][0])
            t_fb_e.append(ind_fb_nc.variables["Total_fb_e"][0])
            t_fb_l.append(ind_fb_nc.variables["Total_fb_l"][0])
            fb_e.append(greg_nc.variables["fb_as_e"][0])
            fb_l.append(greg_nc.variables["fb_as_l"][0])
            
            # load the Monte Carlo test values
            i_fb_p.append(ind_fb_nc.variables[fb_s + "_p_dfb"][0])
            t_fb_p.append(ind_fb_nc.variables["Total_p_dfb"][0])
            fb_p.append(greg_nc.variables["p_as_dfb"][0])
            
            i += 1
            mods.append(mod_pl)
            cols.append("gray")
        except IndexError:
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
            ind_fb_nc = Dataset(glob.glob(data_path6 + 
                                          f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{t_var}_Based/*{mod}.nc")[0])
            greg_nc = Dataset(glob.glob(data_path6 + f"/TOA_Imbalance_piC_Run/{a4x6}/{t_var}_Based/TOA*{mod}.nc")[0])
            
            # load the feedback values
            i_fb_e[fb_s].append(ind_fb_nc.variables[fb_s + "_fb_e"][0])
            i_fb_l[fb_s].append(ind_fb_nc.variables[fb_s + "_fb_l"][0])
            t_fb_e.append(ind_fb_nc.variables["Total_fb_e"][0])
            t_fb_l.append(ind_fb_nc.variables["Total_fb_l"][0])
            fb_e.append(greg_nc.variables["fb_as_e"][0])
            fb_l.append(greg_nc.variables["fb_as_l"][0])
            
            # load the Monte Carlo test values
            i_fb_p.append(ind_fb_nc.variables[fb_s + "_p_dfb"][0])
            t_fb_p.append(ind_fb_nc.variables["Total_p_dfb"][0])
            fb_p.append(greg_nc.variables["p_as_dfb"][0])
            
            i += 1
            mods.append(mod_pl)
            cols.append("black")
        except IndexError:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod
    
    # convert from lists to arrays
    i_fb_e[fb_s] = np.array(i_fb_e[fb_s])
    i_fb_l[fb_s] = np.array(i_fb_l[fb_s])
    t_fb_e = np.array(t_fb_e)
    t_fb_l = np.array(t_fb_l)
    fb_e = np.array(fb_e)
    fb_l = np.array(fb_l)
    i_fb_p = np.array(i_fb_p)
    t_fb_p = np.array(t_fb_p)
    fb_p = np.array(t_fb_p)
    
    
    #% calculate the feedback changes
    di_fb[fb_s] = i_fb_l[fb_s] - i_fb_e[fb_s]
    dt_fb = t_fb_l - t_fb_e
    dfb = fb_l - fb_e
    
    
    #% perform linear regressions
    sl, yi, r, p = lr(di_fb[fb_s], dt_fb)[:4]  # kernel sum
    sl_g, yi_g, r_g, p_g = lr(di_fb[fb_s], dfb)[:4]  # total Gregory
    pcorr = pears_corr(di_fb[fb_s], dt_fb)[0]
    
    sl_r = np.round(sl, decimals=2)
    p_r = np.round(p, decimals=4)
    sl_g_r = np.round(sl_g, decimals=2)
    p_g_r = np.round(p_g, decimals=4)
    
    
    #% set up markers for the final scatter plot --> significance of the feedback change
    # markers = np.empty(mod_count, dtype=str)
    # markers[:] = "o"
    # markers[(t_fb_p > 0.05) & (i_fb_p > 0.05)] = "x"
    # markers[(t_fb_p > 0.05) & (i_fb_p <= 0.05)] = "v"
    # markers[(t_fb_p <= 0.05) & (i_fb_p > 0.05)] = "^"
    
    
    #% generate an array with models an values for the low to moderate cloud dFb models
    # str_dcfb = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2",
    #                      "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
    str_dcfb = np.array(mods)[di_fb["C"] > 0.5]
    
    wea_dcfb[fb_s] = []
    wea_ind_d_m[fb_s] = []
    wea_ind_e_m[fb_s] = []
    wea_ind_l_m[fb_s] = []
    wea_fb_d_m[fb_s] = []
    for m in np.arange(len(mods)):
        if len(np.intersect1d(mods[m], str_dcfb)) == 0:
            wea_dcfb[fb_s].append(mods[m])
            wea_ind_d_m[fb_s].append(di_fb[fb_s][m])
            wea_ind_e_m[fb_s].append(i_fb_e[fb_s][m])
            wea_ind_l_m[fb_s].append(i_fb_l[fb_s][m])
            wea_fb_d_m[fb_s].append(dt_fb[m])
        # end if
    # end for m
            
    wea_dcfb[fb_s] = np.array(wea_dcfb[fb_s])
    wea_ind_d_m[fb_s] = np.squeeze(np.array(wea_ind_d_m[fb_s]))
    wea_ind_e_m[fb_s] = np.squeeze(np.array(wea_ind_e_m[fb_s]))
    wea_ind_l_m[fb_s] = np.squeeze(np.array(wea_ind_l_m[fb_s]))
    wea_fb_d_m[fb_s] = np.squeeze(np.array(wea_fb_d_m[fb_s]))
    
    
    #% regress the EP warming on ECS and feedback
    sl_all[fb_s], yi_all[fb_s], r_all[fb_s], p_all[fb_s] = lr(di_fb[fb_s], dt_fb)[:4]
    sl_wea[fb_s], yi_wea[fb_s], r_wea[fb_s], p_wea[fb_s] = lr(wea_ind_d_m[fb_s], wea_fb_d_m[fb_s])[:4]
    
    sl_all_e, yi_all_e, r_all_e, p_all_e = lr(i_fb_e[fb_s], dt_fb)[:4]
    sl_wea_e, yi_wea_e, r_wea_e, p_wea_e = lr(wea_ind_e_m[fb_s], wea_fb_d_m[fb_s])[:4]
    
    sl_all_l, yi_all_l, r_all_l, p_all_l = lr(i_fb_l[fb_s], dt_fb)[:4]
    sl_wea_l, yi_wea_l, r_wea_l, p_wea_l = lr(wea_ind_l_m[fb_s], wea_fb_d_m[fb_s])[:4]

# end for fb_s
print("Models NOT loaded:")
print(np.setdiff1d(pass_cslt, np.array(mods)))
print("\nlarge cloud fb change models:")
print(str_dcfb)
print("\n")


#%% scatter individual against the total feedback change
fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(20, 13), sharex=False, sharey=True)
xlim = (-0.65, 1.3)
ylim = (-0.35, 1.25)

if excl == "":
    xlim = (-1.35, 1.65)
    ylim = (-0.65, 1.45)    
# end if

# set the position of th (a), (b), (c), (d) labels in the plots
x_ra = xlim[1] - xlim[0]
tex_x = xlim[1] - x_ra * 0.05
y_ra = ylim[1] - ylim[0]
tex_y = ylim[0] + y_ra * 0.925

for i in np.arange(len(dt_fb)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    axes[0, 0].scatter(di_fb["C"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[0, 1].scatter(di_fb["LR"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 0].scatter(di_fb["S"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 1].scatter(di_fb["Q"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    # axes[1, 1].text(di_fb["Q"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
    # axes[0, 0].text(di_fb["C"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
    # axes[0, 1].text(di_fb["LR"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
# end for i 

dc_mima = np.array([np.min(di_fb["C"]), np.max(di_fb["C"])])
dlr_mima = np.array([np.min(di_fb["LR"]), np.max(di_fb["LR"])])
ds_mima = np.array([np.min(di_fb["S"]), np.max(di_fb["S"])])
dq_mima = np.array([np.min(di_fb["Q"]), np.max(di_fb["Q"])])
p1, = axes[0, 0].plot(dc_mima, dc_mima * sl_all["C"] + yi_all["C"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['C'], decimals=2)}\n"+
                      f"R={np.round(r_all['C'], decimals=2)}\np={np.round(p_all['C'], decimals=4)}")
p2, = axes[0, 0].plot(wea_ind_d_m["C"], wea_ind_d_m["C"] * sl_wea["C"] + yi_wea["C"], c="black", 
                      label=f"slope={np.round(sl_wea['C'], decimals=2)}\n"+
                      f"R={np.round(r_wea['C'], decimals=2)}\np={np.round(p_wea['C'], decimals=4)}")
p3, = axes[0, 1].plot(dlr_mima, dlr_mima * sl_all["LR"] + yi_all["LR"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['LR'], decimals=2)}\n"+
                      f"R={np.round(r_all['LR'], decimals=2)}\np={np.round(p_all['LR'], decimals=4)}")
p4, = axes[0, 1].plot(wea_ind_d_m["LR"], wea_ind_d_m["LR"] * sl_wea["LR"] + yi_wea["LR"], c="black", 
                      label=f"slope={np.round(sl_wea['LR'], decimals=2)}\n"+
                      f"R={np.round(r_wea['LR'], decimals=2)}\np={np.round(p_wea['LR'], decimals=4)}")
p5, = axes[1, 0].plot(ds_mima, ds_mima * sl_all["S"] + yi_all["S"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['S'], decimals=2)}\n"+
                      f"R={np.round(r_all['S'], decimals=2)}\np={np.round(p_all['S'], decimals=4)}")
p6, = axes[1, 0].plot(wea_ind_d_m["S"], wea_ind_d_m["S"] * sl_wea["S"] + yi_wea["S"], c="black", 
                      label=f"slope={np.round(sl_wea['S'], decimals=2)}\n"+
                      f"R={np.round(r_wea['S'], decimals=2)}\np={np.round(p_wea['S'], decimals=4)}")
p7, = axes[1, 1].plot(dq_mima, dq_mima * sl_all["Q"] + yi_all["Q"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['Q'], decimals=2)}\n"+
                      f"R={np.round(r_all['Q'], decimals=2)}\np={np.round(p_all['Q'], decimals=4)}")
p8, = axes[1, 1].plot(wea_ind_d_m["Q"], wea_ind_d_m["Q"] * sl_wea["Q"] + yi_wea["Q"], c="black", 
                      label=f"slope={np.round(sl_wea['Q'], decimals=2)}\n"+
                      f"R={np.round(r_wea['Q'], decimals=2)}\np={np.round(p_wea['Q'], decimals=4)}")

axes[0, 0].axvline(x=0, linewidth=0.75, c="gray")
axes[0, 1].axvline(x=0, linewidth=0.75, c="gray")
axes[1, 0].axvline(x=0, linewidth=0.75, c="gray")
axes[1, 1].axvline(x=0, linewidth=0.75, c="gray")

# for the cloud feedback change
axes[0, 0].axvline(x=0.5, linewidth=2.75, linestyle="--", c="gray")

axes[0, 0].axhline(y=0, linewidth=0.75, c="gray")
axes[0, 1].axhline(y=0, linewidth=0.75, c="gray")
axes[1, 0].axhline(y=0, linewidth=0.75, c="gray")
axes[1, 1].axhline(y=0, linewidth=0.75, c="gray")

# ax1.legend(handles=[p1], loc="lower right", fontsize=13)
axes[0, 0].legend(handles=[p1, p2], loc="upper left", fontsize=19)
axes[0, 1].legend(handles=[p3, p4], loc="lower right", fontsize=19)
axes[1, 0].legend(handles=[p5, p6], loc="lower right", fontsize=19)
axes[1, 1].legend(handles=[p7, p8], loc="lower right", fontsize=19)
# ax1.add_artist(l1)

# ax1.set_title(fb_tn + " v. Total Feedback Change", fontsize=20)

axes[0, 0].set_xlabel("Cloud feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[0, 1].set_xlabel("Lapse-rate feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 0].set_xlabel("Surface-albedo feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 1].set_xlabel("Water-vapour feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[0, 0].set_ylabel("Total feedback change\nin Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 0].set_ylabel("Total feedback change\nin Wm$^{-2}$K$^{-1}$", fontsize=20)

axes[0, 0].tick_params(labelsize=20)
axes[1, 0].tick_params(labelsize=20)
axes[0, 1].tick_params(labelsize=20)
axes[1, 1].tick_params(labelsize=20)

axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)
axes[1, 0].set_xlim(xlim)
axes[1, 0].set_ylim(ylim)
axes[0, 1].set_xlim(xlim)
axes[0, 1].set_ylim(ylim)
axes[1, 1].set_xlim(xlim)
axes[1, 1].set_ylim(ylim)

axes[0, 0].text(tex_x, tex_y, "(a)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(tex_x, tex_y, "(b)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 0].text(tex_x, tex_y, "(c)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(tex_x, tex_y, "(d)", fontsize=21, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.05, hspace=0.25)

pl.savefig(pl_path + f"/PDF/Total_vs_Indiv_Fb_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Total_vs_Indiv_Fb_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter individual against the total feedback change
fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(20, 13), sharex=False, sharey=True)
xlim = (-0.65, 1.3)
ylim = (-0.35, 1.25)

if excl == "":
    xlim = (-1.35, 1.65)
    ylim = (-0.65, 1.45)    
# end if

# set the position of th (a), (b), (c), (d) labels in the plots
x_ra = xlim[1] - xlim[0]
tex_x = xlim[1] - x_ra * 0.05
y_ra = ylim[1] - ylim[0]
tex_y = ylim[0] + y_ra * 0.925

for i in np.arange(len(dt_fb)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    axes[0, 0].scatter(di_fb["C_lw"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[0, 1].scatter(di_fb["C_sw"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 0].scatter(di_fb["Pl"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 1].scatter(di_fb["LR+Q"][i], dt_fb[i], c=cols[i], marker=mark, s=50, linewidth=2)
    # axes[1, 1].text(di_fb["Q"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
    # axes[0, 0].text(di_fb["C"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
    # axes[0, 1].text(di_fb["LR"][i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
# end for i 

dc_mima = np.array([np.min(di_fb["C_lw"]), np.max(di_fb["C_lw"])])
dlr_mima = np.array([np.min(di_fb["C_sw"]), np.max(di_fb["C_sw"])])
ds_mima = np.array([np.min(di_fb["Pl"]), np.max(di_fb["Pl"])])
dq_mima = np.array([np.min(di_fb["LR+Q"]), np.max(di_fb["LR+Q"])])
p1, = axes[0, 0].plot(dc_mima, dc_mima * sl_all["C_lw"] + yi_all["C_lw"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['C_lw'], decimals=2)}\n"+
                      f"R={np.round(r_all['C_lw'], decimals=2)}\np={np.round(p_all['C_lw'], decimals=4)}")
p2, = axes[0, 0].plot(wea_ind_d_m["C_lw"], wea_ind_d_m["C_lw"] * sl_wea["C_lw"] + yi_wea["C_lw"], c="black", 
                      label=f"slope={np.round(sl_wea['C_lw'], decimals=2)}\n"+
                      f"R={np.round(r_wea['C_lw'], decimals=2)}\np={np.round(p_wea['C_lw'], decimals=4)}")
p3, = axes[0, 1].plot(dlr_mima, dlr_mima * sl_all["C_sw"] + yi_all["C_sw"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['C_sw'], decimals=2)}\n"+
                      f"R={np.round(r_all['C_sw'], decimals=2)}\np={np.round(p_all['C_sw'], decimals=4)}")
p4, = axes[0, 1].plot(wea_ind_d_m["C_sw"], wea_ind_d_m["C_sw"] * sl_wea["C_sw"] + yi_wea["C_sw"], c="black", 
                      label=f"slope={np.round(sl_wea['C_sw'], decimals=2)}\n"+
                      f"R={np.round(r_wea['C_sw'], decimals=2)}\np={np.round(p_wea['C_sw'], decimals=4)}")
p5, = axes[1, 0].plot(ds_mima, ds_mima * sl_all["Pl"] + yi_all["Pl"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['Pl'], decimals=2)}\n"+
                      f"R={np.round(r_all['Pl'], decimals=2)}\np={np.round(p_all['Pl'], decimals=4)}")
p6, = axes[1, 0].plot(wea_ind_d_m["Pl"], wea_ind_d_m["Pl"] * sl_wea["Pl"] + yi_wea["Pl"], c="black", 
                      label=f"slope={np.round(sl_wea['Pl'], decimals=2)}\n"+
                      f"R={np.round(r_wea['Pl'], decimals=2)}\np={np.round(p_wea['Pl'], decimals=4)}")
p7, = axes[1, 1].plot(dq_mima, dq_mima * sl_all["LR+Q"] + yi_all["LR+Q"], c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all['LR+Q'], decimals=2)}\n"+
                      f"R={np.round(r_all['LR+Q'], decimals=2)}\np={np.round(p_all['LR+Q'], decimals=4)}")
p8, = axes[1, 1].plot(wea_ind_d_m["LR+Q"], wea_ind_d_m["LR+Q"] * sl_wea["LR+Q"] + yi_wea["LR+Q"], c="black", 
                      label=f"slope={np.round(sl_wea['LR+Q'], decimals=2)}\n"+
                      f"R={np.round(r_wea['LR+Q'], decimals=2)}\np={np.round(p_wea['LR+Q'], decimals=4)}")

axes[0, 0].axvline(x=0, linewidth=0.75, c="gray")
axes[0, 1].axvline(x=0, linewidth=0.75, c="gray")
axes[1, 0].axvline(x=0, linewidth=0.75, c="gray")
axes[1, 1].axvline(x=0, linewidth=0.75, c="gray")

# for the cloud feedback change
axes[0, 0].axvline(x=0.5, linewidth=2.75, linestyle="--", c="gray")
axes[0, 1].axvline(x=0.5, linewidth=2.75, linestyle="--", c="gray")

axes[0, 0].axhline(y=0, linewidth=0.75, c="gray")
axes[0, 1].axhline(y=0, linewidth=0.75, c="gray")
axes[1, 0].axhline(y=0, linewidth=0.75, c="gray")
axes[1, 1].axhline(y=0, linewidth=0.75, c="gray")

# ax1.legend(handles=[p1], loc="lower right", fontsize=13)
axes[0, 0].legend(handles=[p1, p2], loc="lower right", fontsize=19)
axes[0, 1].legend(handles=[p3, p4], loc="upper left", fontsize=19)
axes[1, 0].legend(handles=[p5, p6], loc="lower right", fontsize=19)
axes[1, 1].legend(handles=[p7, p8], loc="lower right", fontsize=19)
# ax1.add_artist(l1)

# ax1.set_title(fb_tn + " v. Total Feedback Change", fontsize=20)

axes[0, 0].set_xlabel("LW Cloud feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[0, 1].set_xlabel("SW Cloud feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 0].set_xlabel("Planck feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 1].set_xlabel("LR+WV feedback change in Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[0, 0].set_ylabel("Total feedback change\nin Wm$^{-2}$K$^{-1}$", fontsize=20)
axes[1, 0].set_ylabel("Total feedback change\nin Wm$^{-2}$K$^{-1}$", fontsize=20)

axes[0, 0].tick_params(labelsize=20)
axes[1, 0].tick_params(labelsize=20)
axes[0, 1].tick_params(labelsize=20)
axes[1, 1].tick_params(labelsize=20)

axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)
axes[1, 0].set_xlim(xlim)
axes[1, 0].set_ylim(ylim)
axes[0, 1].set_xlim(xlim)
axes[0, 1].set_ylim(ylim)
axes[1, 1].set_xlim(xlim)
axes[1, 1].set_ylim(ylim)

axes[0, 0].text(tex_x, tex_y, "(a)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(tex_x, tex_y, "(b)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 0].text(tex_x, tex_y, "(c)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(tex_x, tex_y, "(d)", fontsize=21, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.05, hspace=0.25)

pl.savefig(pl_path + f"/PDF/Total_vs_Indiv_Fb_Part2_{t_var}_Based_{kl}Fb_Changes{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Total_vs_Indiv_Fb_Part2_{t_var}_Based_{kl}Fb_Changes{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% print the feedbacks 
# for i in np.arange(len(di_fb["LR"])):
#     print(f"{fb_labn} and total Fb change {mods[i]}:  {np.round(di_fb['LR'][i], 2)}   {np.round(dt_fb[i], 2)}")
# end for i

for i, j, k in zip(mods, di_fb['C'], dt_fb):
    print([i, np.round(j, 2), np.round(k, 2)])
# end for i, j



#%% plot LR v. SA feedback

sl, yi, r, p = lr(di_fb["LR"], di_fb["S"])[:4]

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i in np.arange(len(dt_fb)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    axes.scatter(di_fb["LR"][i], di_fb["S"][i], c=cols[i], marker=mark, s=50, linewidth=2)
    # axes.text(di_fb["LR"][i], di_fb["S"][i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
# end for i 

axes.plot(di_fb["LR"], di_fb["LR"] * sl + yi, c="black", label=f"sl={np.round(sl, 2)}\nR={np.round(r, 3)}" +
          f"\np={np.round(p, 4)}")

axes.axhline(y=0, c="gray", linewidth=0.5)
axes.axvline(x=0, c="gray", linewidth=0.5)

axes.legend(fontsize=15)

axes.set_title("LR v. SA feedback change", fontsize=20)
axes.set_xlabel("LR feedback change in Wm$^{-2}$K$^{-1}$", fontsize=15)
axes.set_ylabel("SA feedback change in Wm$^{-2}$K$^{-1}$", fontsize=15)

pl.savefig(pl_path2 + f"PDF/LR_v_SA_{t_var}_Based_{kl}Fb_Changes{cslt_fadd}{excl_fadd}.pdf",
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path2 + f"PNG/LR_v_SA_{t_var}_Based_{kl}Fb_Changes{cslt_fadd}{excl_fadd}.png",
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter early individual against the total feedback change
"""
fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(dt_fb)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    ax1.scatter(i_fb_e[i], dt_fb[i], c=cols[i], marker=mark, s=40, linewidth=0.5)
    # ax1.text(di_fb[i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
# end for i 

p1, = ax1.plot(i_fb_e, i_fb_e * sl_all_e + yi_all_e, c="black", 
               label=f"slope={np.round(sl_all_e, decimals=2)}\n"+
               f"R={np.round(r_all_e, decimals=2)}\np={np.round(p_all_e, decimals=4)}")
p2, = ax1.plot(wea_ind_e_m, wea_ind_e_m * sl_wea_e + yi_wea_e, c="grey", 
               label=f"slope={np.round(sl_wea_e, decimals=2)}\n"+
               f"R={np.round(r_wea_e, decimals=2)}\np={np.round(p_wea_e, decimals=4)}")

ax1.axvline(x=0, linewidth=0.75, c="gray")
# ax1.axvline(x=0.5, linewidth=0.75, linestyle="--", c="gray")
ax1.axhline(y=0, linewidth=0.75, c="gray")

# ax1.legend(handles=[p1], loc="lower right", fontsize=13)
ax1.legend(handles=[p1, p2], loc="lower left", fontsize=13)
# ax1.add_artist(l1)

ax1.set_title(fb_tn + " Early v. Total Feedback Change", fontsize=15)

ax1.set_xlabel(fb_labn + " feedback early in Wm$^{-2}$K$^{-1}$", fontsize=13)
ax1.set_ylabel("total feedback change in Wm$^{-2}$K$^{-1}$", fontsize=13)

ax1.tick_params(labelsize=13)

# ax1.set_xlim((-0.65, 1.3))
# ax1.set_ylim((-0.35, 1.25))

pl.savefig(pl_path + f"/PDF/dTF_vs_{fb_fn}_Early_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/dTF_vs_{fb_fn}_Early_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter late individual against the total feedback change
fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(dt_fb)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    ax1.scatter(i_fb_l[i], dt_fb[i], c=cols[i], marker=mark, s=40, linewidth=0.5)
    # ax1.text(di_fb[i], dt_fb[i], mods[i], fontsize=7, horizontalalignment="center", c=cols[i], alpha=0.7)
# end for i 

p1, = ax1.plot(i_fb_l, i_fb_l * sl_all_l + yi_all_l, c="black", 
               label=f"slope={np.round(sl_all_l, decimals=2)}\n"+
               f"R={np.round(r_all_l, decimals=2)}\np={np.round(p_all_l, decimals=4)}")
p2, = ax1.plot(wea_ind_l_m, wea_ind_l_m * sl_wea_l + yi_wea_l, c="grey", 
               label=f"slope={np.round(sl_wea_l, decimals=2)}\n"+
               f"R={np.round(r_wea_l, decimals=2)}\np={np.round(p_wea_l, decimals=4)}")

ax1.axvline(x=0, linewidth=0.75, c="gray")
# ax1.axvline(x=0.5, linewidth=0.75, linestyle="--", c="gray")
ax1.axhline(y=0, linewidth=0.75, c="gray")

# ax1.legend(handles=[p1], loc="lower right", fontsize=13)
ax1.legend(handles=[p1, p2], loc="lower left", fontsize=13)
# ax1.add_artist(l1)

ax1.set_title(fb_tn + " Late v. Total Feedback Change", fontsize=15)

ax1.set_xlabel(fb_labn + " feedback late in Wm$^{-2}$K$^{-1}$", fontsize=13)
ax1.set_ylabel("total feedback change in Wm$^{-2}$K$^{-1}$", fontsize=13)

ax1.tick_params(labelsize=13)

# ax1.set_xlim((-0.65, 1.3))
# ax1.set_ylim((-0.35, 1.25))

pl.savefig(pl_path + f"/PDF/dTF_vs_{fb_fn}_Early_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/dTF_vs_{fb_fn}_Early_{t_var}_Based_Fb_Changes{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


"""