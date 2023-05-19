"""
Generates Figure S5 in the Supplementary Material of Eiselt and Graversen (2022), JCLI.

Correlate the global mean sum of all kernel derived individual feedbacks EXCEPT cloud feedback with the warming (local on 
global tas or ts mean regression) in a given region.

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


#%% set some strings for region name and abbrviation
# acr = "WP"  # "EP"  # "SO"
# titl_n = "Western Pacific"  # "Eastern Pacific"  # "Southern Ocean"
# pl_pn = "Western_Pacific_"  # "Southern_Ocean_"

# acr = "IPWP"
# titl_n = "IPWP"
# pl_pn = "IPWP_"

# acr = "EP"
# titl_n = "Eastern Pacific"
# pl_pn = "Eastern_Pacific_"

# acr = "EAR"
# titl_n = "Arctic"
# pl_pn = "Arctic_"

acr = "AR" 
titl_n = "Arctic"
pl_pn = "Arctic_"


#%% choose the feedback
fb_s = "LR"
# fb_s = "C"
# fb_s = "C_lw"
# fb_s = "C_sw"
# fb_s = "S"
# fb_s = "Q"
# fb_s = "Pl"
# fb_s = "LR+Q"


#%% consider passing the clear-sky linearity test or not?
cslt = True
cslt_thr = 20


#%% indicate which models should be excluded
excl = ""
excl = "outliers"
# excl = "strong cloud dfb and outliers"
# excl = "Fb not available"
# excl = "strong cloud dfb"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"


#%% choose the variable (tas or ts)
var = "ts"
var_s = "Ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the region
if acr == "WP":
    x1_wp = 150  # 360-100  # 0
    x2_wp = 170  # 360-80  # 360
    y1_wp = -15  # -30  # -70
    y2_wp = 15  # 0  # -40
elif acr == "IPWP":
    x1_ipwp = 50
    x2_ipwp = 200
    y1_ipwp = -30
    y2_ipwp = 30
elif acr == "EP":
    x1_ep = 360-100  # 0
    x2_ep = 360-80  # 360
    y1_ep = -30  # -70
    y2_ep = 0  # -40
elif acr == "EAR":
    x1_reg = 0
    x2_reg = 360
    y1_reg = 60
    y2_reg = 90
elif acr == "AR":
    x1_ar = 0
    x2_ar = 360
    y1_ar = 75
    y2_ar = 90
# end if elif

#  x1,y2.....x2,y2  
#    .         .
#    .         .      East ->
#    .         .
#  x1,y1.....x2,y1

x1_wp = [150]
x2_wp = [170]
y1_wp = [-15]
y2_wp = [15]
x1_ipwp = [50]
x2_ipwp = [200]
y1_ipwp = [-30]
y2_ipwp = [30]
x1_ep = [360-100]
x2_ep = [360-80]
y1_ep = [-30]
y2_ep = [0]
x1_ar = [0]
x2_ar = [360]
y1_ar = [75]
y2_ar = [90]


#%% set up some dictionaries and lists with names
fb_dic = {"LR":["Lapse Rate", "lapse rate", "LapseRate"], 
          "C":["Cloud", "cloud", "Cloud"], 
          "C_lw":["LW Cloud", "lw cloud", "CloudLW"], 
          "C_sw":["SW Cloud", "sw cloud", "CloudSW"], 
          "Pl":["Planck", "Planck", "Planck"], 
          "Q":["Water Vapour", "water vapour", "WaterVapour"],
          "S":["Surface Albedo", "surface albedo", "SurfaceAlbedo"],
          "LR+Q":["LR+Q", "LR+Q", "LRpQ"]}

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
cslt_path = f"/Model_Lists/ClearSkyLinearityTest/{fb_ts}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based{thr_fn}_" + 
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
t_dfb_l = []
q_dfb_l = []
s_dfb_l = []
c_dfb_l = []

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
        ind_fb_nc = Dataset(glob.glob(data_path5 + f"/Feedbacks_Local/Kernel/{a4x5}/{k_p[kl]}/{fb_ts}_Based/*{mod}" + 
                                      "*.nc")[0])
        # load cloud feedback files
        cl_fb_nc = Dataset(glob.glob(data_path5 + 
                                          f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}*.nc")[0])

        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - feedback values
        i_fb_e_l.append(ind_fb_nc.variables[fb_s + "_fb_e_rg"][0, :, :])
        i_fb_l_l.append(ind_fb_nc.variables[fb_s + "_fb_l_rg"][0, :, :])
        t_dfb_l.append(ind_fb_nc.variables["LR_dfb_rg"][:] + ind_fb_nc.variables["Pl_dfb_rg"][:])
        s_dfb_l.append(ind_fb_nc.variables["S_dfb_rg"][:])
        q_dfb_l.append(ind_fb_nc.variables["Q_dfb_rg"][:])
        c_dfb_l.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])
        
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
        reg_nc = Dataset(glob.glob(data_path6 + f"{var_s}on{var_s}_Reg/ThresYr20/*{mod}*.nc")[0])
        ind_fb_nc = Dataset(glob.glob(data_path6 + f"/Feedbacks_Local/Kernel/{a4x6}/{k_p[kl]}/{fb_ts}_Based/*{mod}*" +
                                      ".nc")[0])
        # load cloud feedback files
        cl_fb_nc = Dataset(glob.glob(data_path6 + 
                                          f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - feedback values
        i_fb_e_l.append(ind_fb_nc.variables[fb_s + "_fb_e_rg"][0, :, :])
        i_fb_l_l.append(ind_fb_nc.variables[fb_s + "_fb_l_rg"][0, :, :])
        i_dfb_l.append(ind_fb_nc.variables[fb_s + "_dfb_rg"][:])
        t_dfb_l.append(ind_fb_nc.variables["LR_dfb_rg"][:] + ind_fb_nc.variables["Pl_dfb_rg"][:])
        s_dfb_l.append(ind_fb_nc.variables["S_dfb_rg"][:])
        q_dfb_l.append(ind_fb_nc.variables["Q_dfb_rg"][:])        
        c_dfb_l.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])
        
        i += 1
        mods.append(mod_pl)
        cols.append("black")
    except IndexError:
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


#%% convert the lists into numpy arrays and calculate the global means of the feedbacks
lr_e_a = np.array(lr_e_l)
lr_l_a = np.array(lr_l_l)
lr_d_a = np.array(lr_d_l)

i_fb_e_a = glob_mean(np.array(i_fb_e_l), lat, lon)
i_fb_l_a = glob_mean(np.array(i_fb_l_l), lat, lon)
i_fb_d_a = glob_mean(np.array(i_dfb_l), lat, lon)

t_fb_d_a = glob_mean(np.array(t_dfb_l), lat, lon)
s_fb_d_a = glob_mean(np.array(s_dfb_l), lat, lon)
q_fb_d_a = glob_mean(np.array(q_dfb_l), lat, lon)

fb_d_sum = t_fb_d_a + s_fb_d_a + q_fb_d_a

c_dfb_a = np.array(c_dfb_l)


#%% calculate the mean over the regions
ar_d_m = region_mean(lr_d_a, x1_ar, x2_ar, y1_ar, y2_ar, lat, lon)
ep_d_m = region_mean(lr_d_a, x1_ep, x2_ep, y1_ep, y2_ep, lat, lon)
wp_d_m = region_mean(lr_d_a, x1_wp, x2_wp, y1_wp, y2_wp, lat, lon)
ipwp_d_m = region_mean(lr_d_a, x1_ipwp, x2_ipwp, y1_ipwp, y2_ipwp, lat, lon)


#%% regress the EP warming on ECS and feedback
# sl_rege_fbe, yi_rege_fbe, r_rege_fbe, p_rege_fbe = lr(reg_e_m, i_fb_e_a)[:4]
# sl_regl_fbl, yi_regl_fbl, r_regl_fbl, p_regl_fbl = lr(reg_l_m, i_fb_l_a)[:4]
# sl_regd_fbd, yi_regd_fbd, r_regd_fbd, p_regd_fbd = lr(reg_d_m, i_fb_d_a)[:4]

sl_ard_fbd, yi_ard_fbd, r_ard_fbd, p_ard_fbd = lr(ar_d_m, fb_d_sum)[:4]
sl_wpd_fbd, yi_wpd_fbd, r_wpd_fbd, p_wpd_fbd = lr(wp_d_m, fb_d_sum)[:4]
sl_epd_fbd, yi_epd_fbd, r_epd_fbd, p_epd_fbd = lr(ep_d_m, fb_d_sum)[:4]
sl_ipwpd_fbd, yi_ipwpd_fbd, r_ipwpd_fbd, p_ipwpd_fbd = lr(ipwp_d_m, fb_d_sum)[:4]


#%% generate an array with models an values for the low to moderate cloud dFb models
# str_dcfb = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
#                      "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
str_dcfb = np.array(mods)[c_dfb_a > 0.5]
print("large cloud fb change models:")
print(str_dcfb)

wea_dcfb = []
wea_ar_d_m = []
wea_wp_d_m = []
wea_ep_d_m = []
wea_ipwp_d_m = []
wea_fb_d_m = []
for m in np.arange(len(mods)):
    if len(np.intersect1d(mods[m], str_dcfb)) == 0:
        wea_dcfb.append(mods[m])
        wea_ar_d_m.append(ar_d_m[m])
        wea_wp_d_m.append(wp_d_m[m])
        wea_ep_d_m.append(ep_d_m[m])
        wea_ipwp_d_m.append(ipwp_d_m[m])
        wea_fb_d_m.append(fb_d_sum[m])
    # end if
# end for m
        
wea_dcfb = np.array(wea_dcfb)
wea_ar_d_m = np.array(wea_ar_d_m)
wea_wp_d_m = np.array(wea_wp_d_m)
wea_ep_d_m = np.array(wea_ep_d_m)
wea_ipwp_d_m = np.array(wea_ipwp_d_m)
wea_fb_d_m = np.array(wea_fb_d_m)


#%% regress the regional warming(s) on feedback
sl_ar_all, yi_ar_all, r_ar_all, p_ar_all = lr(ar_d_m, fb_d_sum)[:4]
sl_ep_all, yi_ep_all, r_ep_all, p_ep_all = lr(ep_d_m, fb_d_sum)[:4]
sl_wp_all, yi_wp_all, r_wp_all, p_wp_all = lr(wp_d_m, fb_d_sum)[:4]
sl_ipwp_all, yi_ipwp_all, r_ipwp_all, p_ipwp_all = lr(ipwp_d_m, fb_d_sum)[:4]
sl_ar_wea, yi_ar_wea, r_ar_wea, p_ar_wea = lr(wea_ar_d_m, wea_fb_d_m)[:4]
sl_ep_wea, yi_ep_wea, r_ep_wea, p_ep_wea = lr(wea_ep_d_m, wea_fb_d_m)[:4]
sl_wp_wea, yi_wp_wea, r_wp_wea, p_wp_wea = lr(wea_wp_d_m, wea_fb_d_m)[:4]
sl_ipwp_wea, yi_ipwp_wea, r_ipwp_wea, p_ipwp_wea = lr(wea_ipwp_d_m, wea_fb_d_m)[:4]


#%% feedback change v. EP change
fsz = 19

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in np.arange(len(ar_d_m)):
    # ax1.text(reg_d_m[i], fb_d_a[i], mods[i], c=cols[i], horizontalalignment="center", 
    #          verticalalignment="bottom", fontsize=5, alpha=0.6)
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    ax1.scatter(ar_d_m[i], fb_d_sum[i], c=cols[i], marker=mark, s=50, linewidth=0.5)
# end for i 

p1, = ax1.plot(ar_d_m, ar_d_m * sl_ar_all + yi_ar_all, c="gray", 
               label=f"slope={np.round(sl_ar_all, decimals=2)}\n"+
               f"R={np.round(r_ar_all, 2)}\np={np.round(p_ar_all, decimals=4)}")
p2, = ax1.plot(wea_ar_d_m, wea_ar_d_m * sl_ar_wea + yi_ar_wea, c="black", 
               label=f"slope={np.round(sl_ar_wea, decimals=2)}\n"+
               f"R={np.round(r_ar_wea, 2)}\np={np.round(p_ar_wea, decimals=4)}")

ax1.legend(loc="lower right", fontsize=fsz)

ax1.tick_params(labelsize=fsz)

ax1.axhline(y=0, c="gray", linewidth=0.75)
ax1.axvline(x=0, c="gray", linewidth=0.75)

ax1.set_xlabel("Surface warming change in KK$^{{-1}}$", fontsize=fsz)
ax1.set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsz)

ax1.set_title(f"Kernel feedback change sum without cloud v.\n{titl_n} surface warming change", fontsize=fsz+1)

# pl.savefig(pl_path + f"/PDF/{acr}_Change_v_Tot_wo_C_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.pdf", 
#           bbox_inches="tight", dpi=250)
#pl.savefig(pl_path + f"/PNG/{acr}_Change_v_Tot_wo_C_Fb_Change_{var_s}{cslt_fadd}{excl_fadd}.png", 
#           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% four-panel plot
fsize = 20
tfsize = 18
xlim = (-0.6, 1.4)
ylim = (-0.35, 0.8)
xlim_ar = (-3.1, 2.1)

# set the position of th (a), (b), (c), (d) labels in the plots
x_ra = xlim[1] - xlim[0]
x_ra_ar = xlim_ar[1] - xlim_ar[0]
tex_x = xlim[0] + x_ra * 0.05
tex_x_ar = xlim_ar[0] + x_ra_ar * 0.05
y_ra = ylim[1] - ylim[0]
tex_y = ylim[0] + y_ra * 0.925

fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(20, 13), sharey=True)

for i in np.arange(len(ar_d_m)):
    # ax1.text(reg_d_m[i], fb_d_a[i], mods[i], c=cols[i], horizontalalignment="center", 
    #          verticalalignment="bottom", fontsize=5, alpha=0.6)
    if np.any(mods[i] == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else        
    axes[0, 0].scatter(ar_d_m[i], fb_d_sum[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[0, 1].scatter(ipwp_d_m[i], fb_d_sum[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 0].scatter(ep_d_m[i], fb_d_sum[i], c=cols[i], marker=mark, s=50, linewidth=2)
    axes[1, 1].scatter(wp_d_m[i], fb_d_sum[i], c=cols[i], marker=mark, s=50, linewidth=2)
# end for i   

axes[0, 0].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 0].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 1].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[0, 1].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 0].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 0].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 1].axhline(y=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)
axes[1, 1].axvline(x=0, c="gray", linewidth=0.75, linestyle="-", zorder=0)

ar_mima = np.array([np.min(ar_d_m), np.max(ar_d_m)])
ipwp_mima = np.array([np.min(ipwp_d_m), np.max(ipwp_d_m)])
ep_mima = np.array([np.min(ep_d_m), np.max(ep_d_m)])
wp_mima = np.array([np.min(wp_d_m), np.max(wp_d_m)])

p1, = axes[0, 0].plot(ar_mima, ar_mima * sl_ar_all + yi_ar_all, c="black", linestyle=":", 
                      label=f"slope={np.round(sl_ar_all, decimals=2)}\n"+
                      f"R={np.round(r_ar_all, 2)}\np={np.round(p_ar_all, decimals=4)}")
p2, = axes[0, 0].plot(wea_ar_d_m, wea_ar_d_m * sl_ar_wea + yi_ar_wea, c="black", 
                      label=f"slope={np.round(sl_ar_wea, decimals=2)} "+
                      f"R={np.round(r_ar_wea, 2)} p={np.round(p_ar_wea, decimals=4)}")
p3, = axes[0, 1].plot(ipwp_mima, ipwp_mima * sl_ipwp_all + yi_ipwp_all, c="black", linestyle=":", 
                      label=f"slope={np.round(sl_ipwp_all, decimals=2)}\n"+
                      f"R={np.round(r_ipwp_all, 2)}\np={np.round(p_ipwp_all, decimals=4)}")
p4, = axes[0, 1].plot(wea_ipwp_d_m, wea_ipwp_d_m * sl_ipwp_wea + yi_ipwp_wea, c="black", 
                      label=f"slope={np.round(sl_ipwp_wea, decimals=2)}\n"+
                      f"R={np.round(r_ipwp_wea, 2)}\np={np.round(p_ipwp_wea, decimals=4)}")

p5, = axes[1, 0].plot(ep_mima, ep_mima * sl_ep_all + yi_ep_all, c="black", linestyle=":", 
                      label=f"slope={np.round(sl_ep_all, decimals=2)}\n"+
                      f"R={np.round(r_ep_all, 2)}\np={np.round(p_ep_all, decimals=4)}")
p6, = axes[1, 0].plot(wea_ep_d_m, wea_ep_d_m * sl_ep_wea + yi_ep_wea, c="black", 
                      label=f"slope={np.round(sl_ar_wea, decimals=2)}\n"+
                      f"R={np.round(r_ep_wea, 2)}\np={np.round(p_ep_wea, decimals=4)}")
p7, = axes[1, 1].plot(wp_mima, wp_mima * sl_wp_all + yi_wp_all, c="black", linestyle=":", 
                      label=f"slope={np.round(sl_wp_all, decimals=2)}\n"+
                      f"R={np.round(r_wp_all, 2)}\np={np.round(p_wp_all, decimals=4)}")
p8, = axes[1, 1].plot(wea_wp_d_m, wea_wp_d_m * sl_wp_wea + yi_wp_wea, c="black", 
                      label=f"slope={np.round(sl_wp_wea, decimals=2)}\n"+
                      f"R={np.round(r_wp_wea, 2)}\np={np.round(p_wp_wea, decimals=4)}")

 
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

pl.savefig(pl_path + f"/PDF/Regional_WarmChange_v{thr_fn}_TF_WOCloud_Change_{var_s}{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Regional_WarmChange_v{thr_fn}_TF_WOCloud_Change_{var_s}{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()



