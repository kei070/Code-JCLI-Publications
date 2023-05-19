"""
Generates Figure 7 in Eiselt and Graversen (2022), JCLI.

Correlate global mean Gregory method derived feedback with warming (local on global mean tas/ts) in a given region.

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


#%% set some strings for region name and abbrviation
# acr = "EAR"

acr = "IPWP"

# acr = "EP"

# acr = "WP"

# acr = "EEP07"

# acr = "AR"

# acr = "AAR"


#%% choose the variable (tas or ts)
var = "ts"
var_s = "Ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


#%% consider passing the clear-sky linearity test or not?
cslt = True
cslt_thr = 15


#%% indicate which models should be excluded
excl = ""
excl = "outliers"
# excl = "strong cloud dfb and outliers"
# excl = "Fb not available"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"
# excl = "<[3,4]K dTas<"
# excl = "neg dTF"
# excl = "neg abs dTF"
# excl = "neg abs dTF & strong cloud"



#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the region
x1_reg1 = 0
x2_reg1 = 360
y1_reg1 = 75
y2_reg1 = 90
x1_reg2 = 50
x2_reg2 = 200
y1_reg2 = -30
y2_reg2 = 30
x1_reg3 = 360-100
x2_reg3 = 360-80
y1_reg3 = -30
y2_reg3 = 0
x1_reg4 = 150
x2_reg4 = 170
y1_reg4 = -15
y2_reg4 = 15

if acr == "AAR":
    x1_reg = 0
    x2_reg = 360
    y1_reg = -90
    y2_reg = -60
    titl_n = "Antarctic"
    pl_pn = "Antarctic_"    
elif acr == "AR":
    x1_reg = 0
    x2_reg = 360
    y1_reg = 75
    y2_reg = 90
    titl_n = "Arctic"
    pl_pn = "Arctic_"    
elif acr == "EAR":
    x1_reg = 0
    x2_reg = 360
    y1_reg = 75
    y2_reg = 90
    titl_n = "Extreme Arctic"
    pl_pn = "ExtrArctic_"    
elif acr == "EP":
    x1_reg = 360-100
    x2_reg = 360-80
    y1_reg = -30
    y2_reg = 0
    titl_n = "Eastern Pacific"
    pl_pn = "Eastern_Pacific_"
elif acr == "WP":
    x1_reg = 150
    x2_reg = 170
    y1_reg = -15
    y2_reg = 15
    titl_n = "Western Pacific"
    pl_pn = "Western_Pacific_"
elif acr == "IPWP":
    x1_reg = 50
    x2_reg = 200
    y1_reg = -30
    y2_reg = 30
    titl_n = "Indo-Pacific Warm Pool"
    pl_pn = "IPWP_"     
elif acr == "EEP01":
    x1_reg = 360-110
    x2_reg = 360-70
    y1_reg = -35
    y2_reg = -10
    titl_n = "Expanded [01] Eastern Pacific"
    pl_pn = "Exp01Eastern_Pacific_"

elif acr == "EEP02":
    x1_reg = 360-110
    x2_reg = 360-70
    y1_reg = -40
    y2_reg = -10
    titl_n = "Expanded [02] Eastern Pacific"
    pl_pn = "Exp02Eastern_Pacific_"

elif acr == "EEP03":
    x1_reg = 360-110
    x2_reg = 360-70
    y1_reg = -30
    y2_reg = 0
    titl_n = "Expanded [03] Eastern Pacific"
    pl_pn = "Exp03Eastern_Pacific_"

elif acr == "EEP04":
    x1_reg = 360-100
    x2_reg = 360-80
    y1_reg = -40
    y2_reg = -10
    titl_n = "Expanded [04] Eastern Pacific"
    pl_pn = "Exp04Eastern_Pacific_"
    
elif acr == "EEP05":
    x1_reg = 360-100
    x2_reg = 360-80
    y1_reg = -20
    y2_reg = 10
    titl_n = "Expanded [05] Eastern Pacific"
    pl_pn = "Exp05Eastern_Pacific_"

elif acr == "EEP06":
    x1_reg = 360-110
    x2_reg = 360-90
    y1_reg = -30
    y2_reg = 0
    titl_n = "Expanded [06] Eastern Pacific"
    pl_pn = "Exp06Eastern_Pacific_"

elif acr == "EEP07":
    x1_reg = 360-90
    x2_reg = 360-70
    y1_reg = -30
    y2_reg = 0
    titl_n = "Expanded [07] Eastern Pacific"
    pl_pn = "Exp07Eastern_Pacific_"

# end if elif    
    
#  x1,y2.....x2,y2  
#    .         .
#    .         .      East ->
#    .         .
#  x1,y1.....x2,y1


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
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
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


#%% load the floating threshold Gregory analysis

# load the nc file
float_greg_path = f"/Floating_ELP_Threshold_Range{thr_min}_{thr_max}_TF_Using_{fb_ts}.nc"
float_greg_nc = Dataset(float_greg_path)

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


#%% set paths

# model data path
data_path5 = ""
data_path6 = ""

# clear sky linearity test path
cslt_path = f"Model_Lists/ClearSkyLinearityTest/{fb_ts}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based{thr_fn}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]

print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


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
ecs_e_l = []
ecs_l_l = []
ecs_d_l = []
fb_e_l = []
fb_l_l = []
fb_d_l = []
cl_fb_d = []
yr_max_dfb_sort = []

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
        reg_nc = Dataset(glob.glob(data_path5 + f"{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
        
        # load cloud feedback files
        i_fb_nc = Dataset(glob.glob(data_path5 + 
                                          f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}*.nc")[0])
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
        # load the cloud feedback
        cl_fb_d.append(i_fb_nc.variables["C_fb_l"][0] - i_fb_nc.variables["C_fb_e"][0])
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - ECS values (focus here: get the right sorting!)
        ecs_e = ecss_e[mod_ind, dfb_ind[mod_ind]]
        ecs_l = ecss_l[mod_ind, dfb_ind[mod_ind]]
        ecs_e_l.append(ecs_e)
        ecs_l_l.append(ecs_l)
        ecs_d_l.append(ecs_l - ecs_e)
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)
        # fb_e = i_fb_nc.variables["Total_fb_e"][0]
        # fb_l = i_fb_nc.variables["Total_fb_l"][0]
        # fb_e_l.append(fb_e)
        # fb_l_l.append(fb_l)
        # fb_d_l.append(fb_l - fb_e)
        
        # get the year of max dTF (focus here: get the right sorting!)
        yr_max_dfb_sort.append(yr_max_dfb[mod_ind])
        
        i += 1
        mods.append(mod_pl)
        cols.append("gray")
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
        reg_nc = Dataset(glob.glob(data_path6 + f"{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
        
        i_fb_nc = Dataset(glob.glob(data_path6 + 
                                          f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])
    
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
        # load the cloud feedback
        cl_fb_d.append(i_fb_nc.variables["C_fb_l"][0] - i_fb_nc.variables["C_fb_e"][0])
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - ECS values (focus here: get the right sorting!)
        ecs_e = ecss_e[mod_ind, dfb_ind[mod_ind]]
        ecs_l = ecss_l[mod_ind, dfb_ind[mod_ind]]
        ecs_e_l.append(ecs_e)
        ecs_l_l.append(ecs_l)
        ecs_d_l.append(ecs_l - ecs_e)
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)
        # fb_e = i_fb_nc.variables["Total_fb_e"][0]
        # fb_l = i_fb_nc.variables["Total_fb_l"][0]
        # fb_e_l.append(fb_e)
        # fb_l_l.append(fb_l)
        # fb_d_l.append(fb_l - fb_e)

        # get the year of max dTF (focus here: get the right sorting!)
        yr_max_dfb_sort.append(yr_max_dfb[mod_ind])        
        
        i += 1
        mods.append(mod_pl)
        cols.append("black")
    except:
        print("\nJumping " + mod_pl + " because data are not (yet) available...")
        continue
    # end try except

    mod_count += 1     
# end for mod
print("Models NOT loaded:")
print(np.setdiff1d(pass_cslt, np.array(mods)))
print("\n")


#%% get lat and lon
lat = reg_nc.variables["lat_rg"][:]
lon = reg_nc.variables["lon_rg"][:]


#%% convert the lists into numpy arrays
lr_e_a = np.array(lr_e_l)
lr_l_a = np.array(lr_l_l)
lr_d_a = np.array(lr_d_l)

ecs_e_a = np.array(ecs_e_l)
ecs_l_a = np.array(ecs_l_l)
ecs_d_a = np.array(ecs_d_l)

fb_e_a = np.array(fb_e_l)
fb_l_a = np.array(fb_l_l)
fb_d_a = np.array(fb_d_l)

cl_fb_d = np.array(cl_fb_d)

yr_max_dfb_sort = np.array(yr_max_dfb_sort)


#%%
# test1, lon_eu = eu_centric(lr_e_a[0, :, :], lon)


#%% extract the region
# reg_e = extract_region(x1_reg, x2_reg, y1_reg, y2_reg, lat, lon, lr_e_a, test_plt=True, plot_title=titl_n)
# wp_e = extract_region(x1_wp, x2_wp, y1_wp, y2_wp, lat, lon, lr_e_a, test_plt=True, plot_title="WP Region")

# reg_l = extract_region(x1_reg, x2_reg, y1_reg, y2_reg, lat, lon, lr_l_a, test_plt=True, plot_title=titl_n)
# wp_l = extract_region(x1_wp, x2_wp, y1_wp, y2_wp, lat, lon, lr_l_a, test_plt=True, plot_title="WP Region")

reg1_d = extract_region(x1_reg1, x2_reg1, y1_reg1, y2_reg1, lat, lon, lr_d_a, test_plt=True, plot_title="Arctic")
reg2_d = extract_region(x1_reg2, x2_reg2, y1_reg2, y2_reg2, lat, lon, lr_d_a, test_plt=True, plot_title="IPWP")
reg3_d = extract_region(x1_reg3, x2_reg3, y1_reg3, y2_reg3, lat, lon, lr_d_a, test_plt=True, plot_title="EP")
reg4_d = extract_region(x1_reg4, x2_reg4, y1_reg4, y2_reg4, lat, lon, lr_d_a, test_plt=True, plot_title="WP")
# wp_d = extract_region(x1_wp, x2_wp, y1_wp, y2_wp, lat, lon, lr_d_a, test_plt=True, plot_title="WP Region")


#%% get the weights for averaging over the EP and WP regions
reg1_lons = lon[reg1_d[1][1][0, :]]
reg1_lats = lat[reg1_d[1][0][:, 0]]
reg1_lonm, reg1_latm = np.meshgrid(reg1_lons, reg1_lats)
reg1_latw = np.zeros(np.shape(reg1_d[0]))
reg1_latw[:] = np.cos(reg1_latm / 180 * np.pi)[None, :, :]

reg2_lons = lon[reg2_d[1][1][0, :]]
reg2_lats = lat[reg2_d[1][0][:, 0]]
reg2_lonm, reg2_latm = np.meshgrid(reg2_lons, reg2_lats)
reg2_latw = np.zeros(np.shape(reg2_d[0]))
reg2_latw[:] = np.cos(reg2_latm / 180 * np.pi)[None, :, :]

reg3_lons = lon[reg3_d[1][1][0, :]]
reg3_lats = lat[reg3_d[1][0][:, 0]]
reg3_lonm, reg3_latm = np.meshgrid(reg3_lons, reg3_lats)
reg3_latw = np.zeros(np.shape(reg3_d[0]))
reg3_latw[:] = np.cos(reg3_latm / 180 * np.pi)[None, :, :]

reg4_lons = lon[reg4_d[1][1][0, :]]
reg4_lats = lat[reg4_d[1][0][:, 0]]
reg4_lonm, reg4_latm = np.meshgrid(reg4_lons, reg4_lats)
reg4_latw = np.zeros(np.shape(reg4_d[0]))
reg4_latw[:] = np.cos(reg4_latm / 180 * np.pi)[None, :, :]


#%% calculate the mean over the regions
# reg_e_m = np.average(reg_e[0], weights=reg1_latw, axis=(1, 2))

# reg_l_m = np.average(reg_l[0], weights=reg1_latw, axis=(1, 2))

reg1_d_m = np.average(reg1_d[0], weights=reg1_latw, axis=(1, 2))
reg2_d_m = np.average(reg2_d[0], weights=reg2_latw, axis=(1, 2))
reg3_d_m = np.average(reg3_d[0], weights=reg3_latw, axis=(1, 2))
reg4_d_m = np.average(reg4_d[0], weights=reg4_latw, axis=(1, 2))


#%% generate an array with models an values for the low to moderate cloud dFb models
# str_dcfb = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
#                      "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
str_dcfb = np.array(mods)[cl_fb_d > 0.5]
print("large cloud fb change models:")
print(str_dcfb)

wea_dcfb = []
wea_reg1_d_m = []
wea_reg2_d_m = []
wea_reg3_d_m = []
wea_reg4_d_m = []
wea_fb_d_m = []
for m in np.arange(len(mods)):
    if len(np.intersect1d(mods[m], str_dcfb)) == 0:
        wea_dcfb.append(mods[m])
        wea_reg1_d_m.append(reg1_d_m[m])
        wea_reg2_d_m.append(reg2_d_m[m])
        wea_reg3_d_m.append(reg3_d_m[m])
        wea_reg4_d_m.append(reg4_d_m[m])
        wea_fb_d_m.append(fb_d_a[m])
    # end if
# end for m
        
wea_dcfb = np.array(wea_dcfb)
wea_reg1_d_m = np.array(wea_reg1_d_m)
wea_reg2_d_m = np.array(wea_reg2_d_m)
wea_reg3_d_m = np.array(wea_reg3_d_m)
wea_reg4_d_m = np.array(wea_reg4_d_m)
wea_fb_d_m = np.array(wea_fb_d_m)


#%% regress the EP warming on ECS and feedback
sl_all1, yi_all1, r_all1, p_all1 = lr(reg1_d_m, fb_d_a)[:4]
sl_wea1, yi_wea1, r_wea1, p_wea1 = lr(wea_reg1_d_m, wea_fb_d_m)[:4]
sl_all2, yi_all2, r_all2, p_all2 = lr(reg2_d_m, fb_d_a)[:4]
sl_wea2, yi_wea2, r_wea2, p_wea2 = lr(wea_reg2_d_m, wea_fb_d_m)[:4]
sl_all3, yi_all3, r_all3, p_all3 = lr(reg3_d_m, fb_d_a)[:4]
sl_wea3, yi_wea3, r_wea3, p_wea3 = lr(wea_reg3_d_m, wea_fb_d_m)[:4]
sl_all4, yi_all4, r_all4, p_all4 = lr(reg4_d_m, fb_d_a)[:4]
sl_wea4, yi_wea4, r_wea4, p_wea4 = lr(wea_reg4_d_m, wea_fb_d_m)[:4]


#%% TF change v. region warming change
fsize = 20
tfsize = 18
xlim = (-0.6, 1.4)
ylim = (-0.05, 1.25)
xlim_ar = (-3.1, 2.1)

if excl == "":
    ylim = (-0.5, 1.25)
# end if    

# set the position of th (a), (b), (c), (d) labels in the plots
x_ra = xlim[1] - xlim[0]
x_ra_ar = xlim_ar[1] - xlim_ar[0]
tex_x = xlim[0] + x_ra * 0.05
tex_x_ar = xlim_ar[0] + x_ra_ar * 0.05
y_ra = ylim[1] - ylim[0]
tex_y = ylim[0] + y_ra * 0.925

fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(20, 13), sharey=True)

for i in np.arange(len(reg1_d_m)):
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    axes[0, 0].scatter(reg1_d_m[i], fb_d_a[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[0, 1].scatter(reg2_d_m[i], fb_d_a[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 0].scatter(reg3_d_m[i], fb_d_a[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 1].scatter(reg4_d_m[i], fb_d_a[i], c=cols[i], marker=mark, s=50, linewidth=2)
    # axes[0, 0].text(reg1_d_m[i], fb_d_a[i], mods[i], c=cols[i], horizontalalignment="center", 
    #          verticalalignment="bottom", fontsize=8, alpha=0.6)    
# end for i    

axes[0, 0].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 0].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 1].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 1].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 0].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 0].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 1].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 1].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)

reg1_mima = np.array([np.min(reg1_d_m), np.max(reg1_d_m)])
reg2_mima = np.array([np.min(reg2_d_m), np.max(reg2_d_m)])
reg3_mima = np.array([np.min(reg3_d_m), np.max(reg3_d_m)])
reg4_mima = np.array([np.min(reg4_d_m), np.max(reg4_d_m)])
p1, = axes[0, 0].plot(reg1_mima, reg1_mima * sl_all1 + yi_all1, c="black", linestyle=":", 
                      label=f"slope={np.round(sl_all1, decimals=2)}\n"+
                      f"R={np.round(r_all1, 2)}\np={np.round(p_all1, decimals=4)}")
p2, = axes[0, 0].plot(wea_reg1_d_m, wea_reg1_d_m * sl_wea1 + yi_wea1, c="black", 
                      label=f"slope={np.round(sl_wea1, decimals=2)}\n"+
                      f"R={np.round(r_wea1, 2)}\np={np.round(p_wea1, decimals=4)}")
p3, = axes[0, 1].plot(reg2_mima, reg2_mima * sl_all2 + yi_all2, c="black", linestyle=":",
                      label=f"slope={np.round(sl_all2, decimals=2)}\n"+
                      f"R={np.round(r_all2, 2)}\np={np.round(p_all2, decimals=4)}")
p4, = axes[0, 1].plot(wea_reg2_d_m, wea_reg2_d_m * sl_wea2 + yi_wea2, c="black", 
                      label=f"slope={np.round(sl_wea2, decimals=2)}\n"+
                      f"R={np.round(r_wea2, 2)}\np={np.round(p_wea2, decimals=4)}")
p5, = axes[1, 0].plot(reg3_mima, reg3_mima * sl_all3 + yi_all3, c="black", linestyle=":",
                      label=f"slope={np.round(sl_all3, decimals=2)}\n"+
                      f"R={np.round(r_all3, 2)}\np={np.round(p_all3, decimals=4)}")
p6, = axes[1, 0].plot(wea_reg3_d_m, wea_reg3_d_m * sl_wea3 + yi_wea3, c="black", 
                      label=f"slope={np.round(sl_wea3, decimals=2)}\n"+
                      f"R={np.round(r_wea3, 2)}\np={np.round(p_wea3, decimals=4)}")
p7, = axes[1, 1].plot(reg4_mima, reg4_mima * sl_all4 + yi_all4, c="black", linestyle=":",
                      label=f"slope={np.round(sl_all4, decimals=2)}\n"+
                      f"R={np.round(r_all4, 2)}\np={np.round(p_all4, decimals=4)}")
p8, = axes[1, 1].plot(wea_reg4_d_m, wea_reg4_d_m * sl_wea4 + yi_wea4, c="black", 
                      label=f"slope={np.round(sl_wea4, decimals=2)}\n"+
                      f"R={np.round(r_wea4, 2)}\np={np.round(p_wea4, decimals=4)}")

leg_loc = "upper right"
if acr == "AR":
    leg_loc = "upper center"
# end if    
leg1 = axes[0, 0].legend(handles=[p1], loc="lower right", fontsize=fsize)
axes[0, 0].legend(handles=[p2], loc="upper center", fontsize=fsize)
axes[0, 0].add_artist(leg1)

leg2 = axes[0, 1].legend(handles=[p3], loc="lower right", fontsize=fsize)
axes[0, 1].legend(handles=[p4], loc="upper right", fontsize=fsize)
axes[0, 1].add_artist(leg2)

leg3 = axes[1, 0].legend(handles=[p5], loc="lower right", fontsize=fsize)
axes[1, 0].legend(handles=[p6], loc="upper right", fontsize=fsize)
axes[1, 0].add_artist(leg3)

leg4 = axes[1, 1].legend(handles=[p7], loc="lower right", fontsize=fsize)
axes[1, 1].legend(handles=[p8], loc="upper right", fontsize=fsize)
axes[1, 1].add_artist(leg4)

axes[0, 0].set_xlim(xlim_ar)
axes[0, 1].set_xlim(xlim)
axes[1, 0].set_xlim(xlim)
axes[1, 1].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)
axes[0, 1].set_ylim(ylim)
axes[1, 0].set_ylim(ylim)
axes[1, 1].set_ylim(ylim)

axes[0, 0].tick_params(labelsize=fsize)
axes[0, 1].tick_params(labelsize=fsize)
axes[1, 0].tick_params(labelsize=fsize)
axes[1, 1].tick_params(labelsize=fsize)

axes[0, 0].set_xlabel("Arctic warming change in KK$^{{-1}}$", fontsize=fsize)
axes[0, 1].set_xlabel("IPWP warming change in KK$^{{-1}}$", fontsize=fsize)
axes[1, 0].set_xlabel("EP warming change in KK$^{{-1}}$", fontsize=fsize)
axes[1, 1].set_xlabel("WP warming change in KK$^{{-1}}$", fontsize=fsize)

axes[0, 0].set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsize)
axes[1, 0].set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsize)

axes[0, 0].text(tex_x_ar, tex_y, "(a)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(tex_x, tex_y, "(b)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 0].text(tex_x, tex_y, "(c)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(tex_x, tex_y, "(d)", fontsize=21, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.05, hspace=0.25)

pl.savefig(pl_path + f"/PDF/Regional_WarmChange_v{thr_fn}_TF_Gregory_Change_{var_s}{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Regional_WarmChange_v{thr_fn}_TF_Gregory_Change_{var_s}{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()




