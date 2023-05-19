"""
Generates Fig. 1 in Eiselt and Graversen (2022), JCLI: A histogram showing the feedback changes of multiple models.

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

from Functions.Func_Extract_Geographic_Region import extract_region
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
fb_s = "LR"
# fb_s = "C"
# fb_s = "C_lw"
# fb_s = "C_sw"
# fb_s = "S"
# fb_s = "Q"
# fb_s = "Pl"
# fb_s = "LR+Q"
fb_s = "Total"


#%% consider passing the clear-sky linearity test or not?
cslt = False
cslt_thr = 15  # --> CSLT relative error threshold in % --> 15% or 20% or... 


#%% indicate which models should be excluded
excl = ""
# excl = "outliers"
# excl = "strong cloud dfb and outliers"
# excl = "Fb not available"
# excl = "strong cloud dfb"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"


#%% choose the variable (tas or ts)
var = "tas"
var_s = "Tas"


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% set up some dictionaries and lists with names
fb_dic = {"LR":["Lapse Rate", "lapse rate", "LapseRate"], 
          "C":["Cloud", "cloud", "Cloud"], 
          "C_lw":["LW Cloud", "lw cloud", "CloudLW"], 
          "C_sw":["SW Cloud", "sw cloud", "CloudSW"], 
          "Pl":["Planck", "Planck", "Planck"], 
          "Q":["Water Vapour", "water vapour", "WaterVapour"],
          "S":["Surface Albedo", "surface albedo", "SurfaceAlbedo"],
          "LR+Q":["LR+WV", "LR+WV", "LRpWV"],
          "Total":["Total", "Total", "Total"]}

fb_tn = fb_dic[fb_s][0] 
fb_labn = fb_dic[fb_s][1]
fb_fn = fb_dic[fb_s][2]

# do not change:
fb_col = "red"


#%% load the namelist
import Namelists.Namelist_CMIP5 as nl5
a4x5 = "abrupt4xCO2"

import Namelists.Namelist_CMIP6 as nl6
a4x6 = "abrupt-4xCO2"

direc_5 = ""  # SETS BASIC DATA DIRECTORY
direc_6 = ""  # SETS BASIC DATA DIRECTORY


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

if excl == "strong cloud dfb":
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
    excl_fadd = "_Excl_Strong_Cloud_dFb"
    len_excl = len(excl_arr)  # important!
    
if excl == "strong cloud dfb and outliers":
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4', "MIROC-ES2L", "GISS-E2.2-G"])
    excl_fadd = "_Excl_Strong_Cloud_dFb_WoOutl"
    len_excl = len(excl_arr)  # important!  
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
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'MIROC-ES2L', "CNRM-CM6.1-HR", 'GFDL-ESM4'])
    excl_fadd = "_Excl_Strong_Cloud_dFb"
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


#%% set the kernel name --> NEEDED FOR CSLT !!!
# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"


#%% add something to the file name concerning the CSLT
cslt_fadd = ""
if cslt:
    cslt_fadd = f"_CSLT{cslt_thr:02}{kl}"
# end if


#%% load the floating threshold Gregory analysis

# load the nc file
float_greg_nc = Dataset(direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/Outputs/Floating_ELP_Threshold_" +
                        f"Range{thr_min}_{thr_max}_TF_Using_{var}.nc")

# set the threshold year for every to 20 if year of maximum dTF shall not be taken
thr_fn = "_ThrYrMaxdTF"
thr_plt = "Thres: Max. $\Delta$TF"
thr_str = "MaxdTF"
max_dfb_var = "yr_max_dTF"
if thr_yr == -2:
    thr_fn = "_ThrYrMaxAbsdTF"
    thr_plt = "Thres: Max. Abs. $\Delta$TF"
    max_dfb_var = "yr_abs_max_dTF"    
    thr_str = "MaxAbsdTF"
# end if    
    
# load the max dTF per model
yr_max_dfb = float_greg_nc.variables[max_dfb_var][:]
dfb_ind = yr_max_dfb - 15  # -15 because the nc-variable contains the year!
mods_max_dfb = float_greg_nc.variables["models"][:]

# make dictionary that assigns the year of max dTF to the corresponding model
max_dtf_thr = dict(zip(mods_max_dfb, yr_max_dfb))

# if threshold year is 20
if thr_yr > -1:
    thr_fn = f"_ThrYr{thr_yr}"
    thr_plt = f"Thres: Year {thr_yr}"
    thr_str = f"{thr_yr}"
    for key in max_dtf_thr.keys():
        max_dtf_thr[key] = thr_yr
    # end for
    
    dfb_ind[:] = thr_yr - thr_min
# end if


#%% load the floating threshold Gregory analysis

# load early and late slopes and y-intercepts
sls_e = float_greg_nc.variables["TF_early"][:]
sls_l = float_greg_nc.variables["TF_late"][:]
yis_e = float_greg_nc.variables["yi_early"][:]
yis_l = float_greg_nc.variables["yi_late"][:]

# calculate ECS
ecss_e = -yis_e / sls_e
ecss_l = -yis_l / sls_l
ecss_d = ecss_l - ecss_e


#%% set up some initial dictionaries and lists

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% set paths

# model data path
data_path5 = ""
data_path6 = ""

# clear sky linearity test path
cslt_path = "Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{var}_Based{thr_fn}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% start a for-loop for the analysis
i = 0
mods = []
# deg = 30  # latitude (in degree) north of which (and south of the negative of which) the mean is calculated
cols = []

# set up a counter for the models
mod_count = 0

# set up the dictionaries to store all feedbacks
lr_e = dict()
lr_l = dict()
lr_d = dict()

# set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
lr_e_l = []
lr_l_l = []
lr_d_l = []
i_fb_e_l = []
i_fb_l_l = []
i_dfb_l = []
fb_e_l = []
fb_l_l = []
fb_d_l = []
ecs_d = []

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
        reg_nc = Dataset(glob.glob(data_path5 + f"{var_s}on{var_s}_Reg/ThresYr20/*{mod}*.nc")[0])
        ind_fb_nc = Dataset(glob.glob(data_path5 + f"/Feedbacks_Local/Kernel/{a4x5}/{k_p[kl]}/{var}_Based/*{mod}" + 
                                      "*.nc")[0])

        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]        
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - feedback values
        i_fb_e_l.append(ind_fb_nc.variables[fb_s + "_fb_e_rg"][0, :, :])
        i_fb_l_l.append(ind_fb_nc.variables[fb_s + "_fb_l_rg"][0, :, :])
        i_dfb_l.append(ind_fb_nc.variables[fb_s + "_dfb_rg"][:])
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)        
        
        # load the ECS values
        ecs_d.append(ecss_d[mod_ind, dfb_ind[mod_ind]])
        
        i += 1
        mods.append(mod_pl)
        cols.append("blue")
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
        reg_nc = Dataset(glob.glob(data_path6 + f"{var_s}on{var_s}_Reg/ThresYr20/*{mod}*.nc")[0])
        ind_fb_nc = Dataset(glob.glob(data_path6 + f"/Feedbacks_Local/Kernel/{a4x6}/{k_p[kl]}/{var}_Based/*{mod}*" +
                                      ".nc")[0])
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]        
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - feedback values
        i_fb_e_l.append(ind_fb_nc.variables[fb_s + "_fb_e_rg"][0, :, :])
        i_fb_l_l.append(ind_fb_nc.variables[fb_s + "_fb_l_rg"][0, :, :])
        i_dfb_l.append(ind_fb_nc.variables[fb_s + "_dfb_rg"][:])
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)            
        
        # load the ECS values
        ecs_d.append(ecss_d[mod_ind, dfb_ind[mod_ind]])        
        
        i += 1
        mods.append(mod_pl)
        cols.append("red")
    except IndexError:
        print("\nJumping " + mod_pl + " because data are not (yet) available...")
        continue
    # end try except

    mod_count += 1     
# end for mod

mods = np.array(mods)


#%% get lat and lon
lat = reg_nc.variables["lat_rg"][:]
lon = reg_nc.variables["lon_rg"][:]


#%% convert the lists into numpy arrays and calculate the global means of the feedbacks
lr_e_a = np.array(lr_e_l)
lr_l_a = np.array(lr_l_l)
lr_d_a = np.array(lr_d_l)

i_fb_e_a = glob_mean(np.array(i_fb_e_l), lat, lon)
i_fb_l_a = glob_mean(np.array(i_fb_l_l), lat, lon)
i_fb_d_a = glob_mean(np.array(i_dfb_l), lat, lon)

fb_e_l = np.array(fb_e_l)
fb_l_l = np.array(fb_l_l)
fb_d_l = np.array(fb_d_l)

ecs_d = np.array(ecs_d)


#%% print all feedbacks
# for i, j in zip(mods, fb_d_l):
#     print([i, np.round(j, 2)])
# end for i, j    


#%% generate an array with models an values for the low to moderate cloud dFb models
str_dcfb = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                     "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])


#%% get the sorting permutation
perm_difb = np.argsort(i_fb_d_a)
perm_decs = np.argsort(ecs_d)


#%% feedback change
fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(i_fb_d_a)):
    if np.any(mods[perm_difb][i] == str_dcfb):
        mark = "x"
        mark2 = "+"
    else:
        mark = "o"
        mark2 = "s"        
    # end if else    
    ax1.scatter(i, i_fb_d_a[perm_difb][i], c=np.array(cols)[perm_difb][i], marker=mark, s=30, linewidth=0.5)
    ax1.scatter(i, fb_d_l[perm_difb][i], c=np.array(cols)[perm_difb][i], marker=mark2, s=30, linewidth=0.5)
# end for i    

# p1, = ax1.plot(reg_d_m, reg_d_m * sl_all + yi_all, c="black", 
#                label=f"slope={np.round(sl_all, decimals=2)}\nR$^{{2}}$={np.round(r_all**2, decimals=3)}\n"+
#                f"R={np.round(r_all, decimals=2)}\np={np.round(p_all, decimals=4)}")
# p2, = ax1.plot(wea_reg_d_m, wea_reg_d_m * sl_wea + yi_wea, c="gray", 
#                label=f"slope={np.round(sl_wea, decimals=2)}\nR$^{{2}}$={np.round(r_wea**2, decimals=3)}\n"+
#                f"R={np.round(r_wea, 2)}\np={np.round(p_wea, decimals=4)}")

# l1 = ax1.legend(handles=[p1], loc="lower right", fontsize=13)
# ax1.legend(handles=[p2], loc="center right", fontsize=13)
# ax1.add_artist(l1)

# ax1.set_ylim((-0.3, 1))

ax1.tick_params(labelsize=13)

ax1.set_xticks(np.arange(len(i_fb_d_a)))
ax1.set_xticklabels(mods[perm_difb], rotation=90)

ax1.axhline(y=0, c="gray", linewidth=0.75)
# ax1.axvline(x=0, c="gray", linewidth=0.75)

# ax1.set_xlabel(f"{acr} Warming change in KK$^{{-1}}$", fontsize=13)
ax1.set_ylabel("feedback change in Wm$^{-2}$K$^{-1}$", fontsize=13)

ax1.set_title(f"Global {fb_dic[fb_s][0]} Feedback Change", fontsize=15)

# pl.savefig(pl_path + f"/PDF/{acr}_Change_v_{fb_dic[fb_s][2]}_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.pdf", 
#            bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/{acr}_Change_v_{fb_dic[fb_s][2]}_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.png", 
#            bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter the dTF v dTFk
dfb_sl, dfb_yi, dfb_r, dfb_p = lr(fb_d_l, i_fb_d_a)[:4]

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(i_fb_d_a)):
    axes.scatter(fb_d_l[i], i_fb_d_a[i], marker="o", c=cols[i])
# end for i

axes.plot(fb_d_l, fb_d_l * dfb_sl + dfb_yi, c="black", 
          label=f"slope={np.round(dfb_sl, 2)}\nR={np.round(dfb_r, 3)}\np={np.round(dfb_p, 5)}")

axes.legend()

axes.set_xlabel("total feedback change in Wm$^{-2}$K$^{-1}$") 
axes.set_ylabel("kernel sum feedback change in Wm$^{-2}$K$^{-1}$") 
axes.set_title("Total Gregory Feedback v. Kernel Sum Feedback")

pl.show()
pl.close()


#%% ECS change
fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(ecs_d)):
    if np.any(mods[perm_decs][i] == str_dcfb):
        mark = "x"
        mark2 = "+"
    else:
        mark = "o"
        mark2 = "s"        
    # end if else    
    ax1.scatter(i, ecs_d[perm_decs][i], c=np.array(cols)[perm_decs][i], marker=mark, s=30, linewidth=0.5)
    # ax1.scatter(i, fb_d_l[perm_difb][i], c=cols[i], marker=mark2, s=30, linewidth=0.5)
# end for i    

# p1, = ax1.plot(reg_d_m, reg_d_m * sl_all + yi_all, c="black", 
#                label=f"slope={np.round(sl_all, decimals=2)}\nR$^{{2}}$={np.round(r_all**2, decimals=3)}\n"+
#                f"R={np.round(r_all, decimals=2)}\np={np.round(p_all, decimals=4)}")
# p2, = ax1.plot(wea_reg_d_m, wea_reg_d_m * sl_wea + yi_wea, c="gray", 
#                label=f"slope={np.round(sl_wea, decimals=2)}\nR$^{{2}}$={np.round(r_wea**2, decimals=3)}\n"+
#                f"R={np.round(r_wea, 2)}\np={np.round(p_wea, decimals=4)}")

# l1 = ax1.legend(handles=[p1], loc="lower right", fontsize=13)
# ax1.legend(handles=[p2], loc="center right", fontsize=13)
# ax1.add_artist(l1)

# ax1.set_ylim((-0.3, 1))

ax1.tick_params(labelsize=13)

ax1.set_xticks(np.arange(len(i_fb_d_a)))
ax1.set_xticklabels(mods[perm_decs], rotation=90)

ax1.axhline(y=0, c="gray", linewidth=0.75)
# ax1.axvline(x=0, c="gray", linewidth=0.75)

# ax1.set_xlabel(f"{acr} Warming change in KK$^{{-1}}$", fontsize=13)
ax1.set_ylabel("ECS change in K", fontsize=13)

ax1.set_title("Climate Sensitivity Change", fontsize=15)

# pl.savefig(pl_path + f"/PDF/{acr}_Change_v_{fb_dic[fb_s][2]}_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.pdf", 
#            bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/{acr}_Change_v_{fb_dic[fb_s][2]}_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.png", 
#            bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% assign the feedback change per model to bins
#   one set of boundaries: 0<TF<0.25, 0.25<=TF<0.75, 0.75<=TF

bins = {}
bins["-2"] = len(fb_d_l[((-0.25 > fb_d_l) & (fb_d_l > -0.5))])
bins["-1"] = len(fb_d_l[((0 > fb_d_l) & (fb_d_l > -0.25))])
bins["0"] = len(fb_d_l[((0 < fb_d_l) & (fb_d_l < 0.25))])
bins["1"] = len(fb_d_l[((0.25 <= fb_d_l) & (fb_d_l < 0.5))])
bins["2"] = len(fb_d_l[((0.5 <= fb_d_l) & (fb_d_l < 0.75))])
bins["3"] = len(fb_d_l[((0.75 <= fb_d_l) & (fb_d_l < 1.0))])
bins["4"] = len(fb_d_l[((1.0 <= fb_d_l) & (fb_d_l < 2.0))])


#%% generate a bar plot that shows the distribution
if not cslt:

    bwidth = 0.25
    fil_col = "lightgray"
    
    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(5, 3))
    
    axes.bar(0.125 - bwidth*2, bins["-2"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125 - bwidth, bins["-1"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125, bins["0"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125 + bwidth, bins["1"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125 + bwidth*2, bins["2"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125 + bwidth*3, bins["3"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.125 + bwidth*4, bins["4"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    
    axes.set_title("Total feedback change histogram")
    
    axes.set_xlabel("Total feedback change in Wm$^{-2}$K$^{-1}$")
    axes.set_ylabel("Percentage of models")
    
    axes.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25])
    axes.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25])
    
    axes.set_yticks(np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1))
    axes.set_yticklabels((np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1) / len(fb_d_l) * 100).astype(int))
    
    pl.savefig(pl_path + f"/PDF/TF_Change_Histogram{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)
    pl.savefig(pl_path + f"/PNG/TF_Change_Histogram{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
    
    pl.show()
    pl.close()
# end if
    

#%% generate a bar plot that shows the distribution for the models used in the paper
if cslt & (cslt_thr == 15):
    bins = {}
    bins["0"] = len(fb_d_l[((0 < fb_d_l) & (fb_d_l < 0.2))])
    bins["1"] = len(fb_d_l[((0.2 <= fb_d_l) & (fb_d_l < 0.4))])
    bins["2"] = len(fb_d_l[((0.4 <= fb_d_l) & (fb_d_l < 0.6))])
    bins["3"] = len(fb_d_l[((0.6 <= fb_d_l) & (fb_d_l < 0.8))])
    bins["4"] = len(fb_d_l[((0.8 <= fb_d_l) & (fb_d_l < 1.0))])
    
    bwidth = 0.2
    fil_col = "lightgray"
    
    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(5, 3))
    
    axes.bar(0.1, bins["0"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth, bins["1"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*2, bins["2"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*3, bins["3"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*4, bins["4"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    
    axes.set_title("Total feedback change histogram")
    
    axes.set_xlabel("Total feedback change in Wm$^{-2}$K$^{-1}$")
    axes.set_ylabel("Percentage of models")
    
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    axes.set_yticks(np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1))
    axes.set_yticklabels((np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1) / len(fb_d_l) * 100).astype(int))
    
    pl.savefig(pl_path + f"/PDF/TF_Change_Histogram{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)
    pl.savefig(pl_path + f"/PNG/TF_Change_Histogram{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
    
    pl.show()
    pl.close()
    
# end if    
    

#%% generate a bar plot that shows the distribution for the models used in the paper with higher CLST threshold
if cslt & (cslt_thr == 20):
    bins = {}
    bins["-1"] = len(fb_d_l[((-0.2 <= fb_d_l) & (fb_d_l < 0))])
    bins["0"] = len(fb_d_l[((0 <= fb_d_l) & (fb_d_l < 0.2))])
    bins["1"] = len(fb_d_l[((0.2 <= fb_d_l) & (fb_d_l < 0.4))])
    bins["2"] = len(fb_d_l[((0.4 <= fb_d_l) & (fb_d_l < 0.6))])
    bins["3"] = len(fb_d_l[((0.6 <= fb_d_l) & (fb_d_l < 0.8))])
    bins["4"] = len(fb_d_l[((0.8 <= fb_d_l) & (fb_d_l < 1.0))])
    bins["5"] = len(fb_d_l[((1 <= fb_d_l) & (fb_d_l < 1.2))])
    
    bwidth = 0.2
    fil_col = "lightgray"
    
    fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(5, 3))
    
    axes.bar(0.1 - bwidth, bins["-1"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1, bins["0"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth, bins["1"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*2, bins["2"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*3, bins["3"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*4, bins["4"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    axes.bar(0.1 + bwidth*5, bins["5"], bottom=0, width=bwidth, color=fil_col, edgecolor="black")
    
    axes.set_title("Total feedback change histogram")
    
    axes.set_xlabel("Total feedback change in Wm$^{-2}$K$^{-1}$")
    axes.set_ylabel("Percentage of models")
    
    axes.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    axes.set_xticklabels([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    
    axes.set_yticks(np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1))
    axes.set_yticklabels((np.arange(0, len(fb_d_l) * 0.5, len(fb_d_l) * 0.1) / len(fb_d_l) * 100).astype(int))
    
    pl.savefig(pl_path + f"/PDF/TF_Change_Histogram{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)
    pl.savefig(pl_path + f"/PNG/TF_Change_Histogram{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
    
    pl.show()
    pl.close()
    
# end if

