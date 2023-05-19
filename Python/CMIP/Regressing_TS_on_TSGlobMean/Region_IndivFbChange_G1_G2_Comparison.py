"""
Somewhat of an early version of Fig. 5 in our paper.

Split the 29 models (see my document "Supposed Evidence and Hypothesis") into two groups: one with small positive or
negative lapse rate feedback change (<0.1 Wm-2K-1) and one with large positive lapse rate feedback change (>=0.5 
Wm-2K-1). For both model groups we calculate the mean relative warming (i.e., the local on global mean regression) change 
in the EP and the Arctic. We then compare the group averages. We expect that for the EP both groups only differ little, 
while for the Arctic there will be a larger difference. The EP is expected to be positive for both groups whereas the 
Arctic change should be less positive or more negative for the group with lapse rate feedback change < 0.1 Wm-2K-1 than
for the group with lapse rate feedback >= 0.5 Wm-2K-1.
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
script_direc = "/home/kei070/Documents/Python_Scripts_PhD_Publication1/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

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
try:
    fb_s = sys.argv[1]
    fb2_s = sys.argv[2]
except:    
    fb_s = "LR"
    
    # fb2_s = "LR" 
    # fb2_s = "C"
    # fb2_s = "C_lw"
    # fb2_s = "C_sw"
    fb2_s = "S"
    # fb2_s = "Q"
    # fb2_s = "Pl"
    # fb2_s = "LR+Q"
    # fb2_s = "Total"
# end try except


#%% set some strings for region name and abbrviation
# index = "TPWPI"  # "EPAA"  # "EQPNP"  #  "EPSO"  REPAA: Relative EP to Antarctica warming
acrs = ["AAR", "EAR"]  # "EQP"  # 
titl_ns = ["Antarctic", "Extreme Arctic"]
# pl_pn = f"{index}_"


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the index type: "add", "sub", "mul", "div"
ind_type = "sub"


#%% choose the two regions
x1s = [0, 0]  # 360-100
x2s = [360, 360]  # 360-80
y1s = [-90, 60]  # -30
y2s = [-60, 90]  # 0

#  x1,y2.....x2,y2  
#    .         .
#    .         .      East ->
#    .         .
#  x1,y1.....x2,y1


#%% choose the variable (tas or ts)
var = "tas"
var_s = "Tas"


#%% consider passing the clear-sky linearity test or not?
cslt = True


#%% indicate which models should be excluded
excl = ""
# excl = "Fb not available"
excl = "strong cloud dfb and outliers"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"
# excl = "<[3,4]K dTas<"
# excl = "neg abs dTF"


#%% select feedback name according to the acronyms
if fb2_s == "LR":
    fb2_n = "Lapse Rate"
elif fb2_s == "S":
    fb2_n = "Surface Albedo"
elif fb2_s == "C":
    fb2_n = "Cloud"
elif fb2_s == "C_lw":
    fb2_n = "LW Cloud"
elif fb2_s == "C_sw":
    fb2_n = "SW Cloud"    
elif fb2_s == "Pl":
    fb2_n = "Planck"
elif fb2_s == "Q":
    fb2_n = "Water Vapour"
elif fb2_s == "LR+Q":
    fb2_n = "LR+Q"
elif fb2_s == "Total":
    fb2_n = "Total"
# end if elif
    

#%% check if the number of regions and the index "type" fit
#   --> for subtraction and division as of now only exactly two regions are possible
#   --> for summation and multiplication 1 to arbitrary regions are possible
if (((ind_type == "sub") | (ind_type == "div")) & (len(x1s) != 2)):
    raise Exception("Index type 'sub' and 'div' work only for exactly two regions. Stopping execution.")
# end if    


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
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])
    excl_fadd = "_Excl_Strong_Cloud_dFb"
    len_excl = len(excl_arr)  # important!
    
if excl == "strong cloud dfb and outliers":
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
                         "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4', "MIROC-ES2L"])
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
    
    excl_arr = np.array(["GISS-E2-R", "MIROC-ES2L"])

    excl_fadd = "_WoOutl"
    len_excl = len(excl_arr)  # important!     
# end if elif


#%% add something to the file name concerning the CSLT
cslt_fadd = ""
if cslt:
    cslt_fadd = "_CSLT"
# end if    


#%% set up some initial dictionaries and lists --> NEEDED FOR CSLT !!!

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


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


#%% set paths

# model data path
data_path5 = (direc_5 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP5/Outputs/")
data_path6 = (direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP6/Outputs/")

# clear sky linearity test path
cslt_path = (direc_6 + f"/Uni/PhD/Tromsoe_UiT/Work/Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/")

# plot path
pl_path = direc_6 + f"/Uni/PhD/Tromsoe_UiT/Work/MultModPlots/RegionComparisonFeedbacks/{var}_Based/"

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{var}_Based{thr_fn}.csv"))[:, 1]


#%% load early and late slopes and y-intercepts
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
fb_e_l = []
fb_l_l = []
fb_d_l = []
i_fb_e = []
i_fb_l = []
max_dfbs = []
yr_max_dfb_sort = []
il_fb_e_l = []
il_fb_l_l = []
il_fb_d_l = []

# for CMIP5
for mod, mod_n, mod_pl in zip(nl5.models, nl5.models_n, nl5.models_pl):
    
    if cslt:
        if not np.any(pass_cslt == mod_pl):
            print("\n\nJumping " + mod_pl + 
                  " because it fails the clear-sky linearity test (or data unavailable)...\n\n")
            continue
        # end if 
    # end if
    
    # exclude a chosen set of models
    if np.any(excl_arr == mod_pl):
        print("\n\nJumping " + mod_pl + " because it is in the exclude list (or data unavailable)...\n\n")
        continue
    # end if
    
    try:
        # load nc files
        ind_fb_nc = Dataset(glob.glob(data_path5 + 
                                      f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}*.nc")[0])        
        # reg_nc = Dataset(glob.glob(data_path5 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
        ind_loc_fb_nc = Dataset(glob.glob(data_path5 + 
                                          f"/Feedbacks_Local/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}*.nc")[0])        

        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
        # load the values - tas or ts regressions
        # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # get the lapse rate feedback
        i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])               
                
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)
        
        # load the local feedbacks
        il_fb_e_l.append(ind_loc_fb_nc.variables[fb2_s + "_fb_e_rg"][0, :, :])
        il_fb_l_l.append(ind_loc_fb_nc.variables[fb2_s + "_fb_l_rg"][0, :, :])
        il_fb_d_l.append(ind_loc_fb_nc.variables[fb2_s + "_dfb_rg"][:])        

        # get the year of max dTF (focus here: get the right sorting!)
        yr_max_dfb_sort.append(yr_max_dfb[mod_ind])
        
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
            print("\n\nJumping " + mod_pl + 
                  " because it fails the clear-sky linearity test (or data unavailable)...\n\n")
            continue
        # end if 
    # end if
    
    # exclude a chosen set of models
    if np.any(excl_arr == mod_pl):
        print("\n\nJumping " + mod_pl + " because it is in the exclude list (or data unavailable)...\n\n")
        continue
    # end if
    
    try:
        # load nc files
        ind_fb_nc = Dataset(glob.glob(data_path6 + 
                                      f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])        
        # reg_nc = Dataset(glob.glob(data_path6 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
        ind_loc_fb_nc = Dataset(glob.glob(data_path6 + f"/Feedbacks_Local/Kernel/abrupt-4xCO2/{k_p[kl]}/" +
                                          f"{var}_Based/*{mod}*.nc")[0])          
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]        
    
        # load the values - tas or ts regressions
        # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # get the lapse rate feedback
        i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])        
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)
        
        # load the local feedbacks
        il_fb_e_l.append(ind_loc_fb_nc.variables[fb2_s + "_fb_e_rg"][0, :, :])
        il_fb_l_l.append(ind_loc_fb_nc.variables[fb2_s + "_fb_l_rg"][0, :, :])
        il_fb_d_l.append(ind_loc_fb_nc.variables[fb2_s + "_dfb_rg"][:])     
        
        # get the year of max dTF (focus here: get the right sorting!)
        yr_max_dfb_sort.append(yr_max_dfb[mod_ind])        
        
        i += 1
        mods.append(mod_pl)
        cols.append("red")
    except:
        print("\n\nJumping " + mod_pl + " because data are not (yet) available...\n\n")
        continue
    # end try except

    mod_count += 1     
# end for mod


#%% get lat and lon
lat = ind_loc_fb_nc.variables["lat_rg"][:]
lon = ind_loc_fb_nc.variables["lon_rg"][:]


#%% convert the lists into numpy arrays
lr_e_a = np.array(lr_e_l)
lr_l_a = np.array(lr_l_l)
lr_d_a = np.array(lr_d_l)

i_fb_e = np.array(i_fb_e)
i_fb_l = np.array(i_fb_l)
i_fb_d = i_fb_l - i_fb_e

fb_e_a = np.array(fb_e_l)
fb_l_a = np.array(fb_l_l)
fb_d_a = np.array(fb_d_l)

il_fb_e_l = np.array(il_fb_e_l)
il_fb_l_l = np.array(il_fb_l_l)
il_fb_d_l = np.array(il_fb_d_l)


#%% print info about the CSLT and the loaded data
print(f"\n\n{len(pass_cslt)} models pass the CSLT and data for {len(lr_e_a)} models is now loaded...\n\n")


#%%
# test1, lon_eu = eu_centric(lr_e_a[0, :, :], lon)


#%% extract the regions
r_e_m = []
r_l_m = []
r_d_m = []
for ri in np.arange(len(x1s)):
    r_e = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_e_l, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    r_l = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_l_l, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    r_d = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_d_l, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    
    # get the weights for averaging over the regions individually
    r_lons = lon[r_e[1][1][0, :]]
    r_lats = lat[r_e[1][0][:, 0]]
    r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
    r_latw = np.zeros(np.shape(r_e[0]))
    r_latw[:] = np.cos(r_latm / 180 * np.pi)[None, :, :]
    
    # calculate the mean over the regions
    r_e_m.append(np.average(r_e[0], weights=r_latw, axis=(1, 2)))
    r_l_m.append(np.average(r_l[0], weights=r_latw, axis=(1, 2)))
    r_d_m.append(np.average(r_d[0], weights=r_latw, axis=(1, 2)))
# end for ri


#%% convert the list to arrays  
r_e_m = np.array(r_e_m)
r_l_m = np.array(r_l_m)
r_d_m = np.array(r_d_m)


#%% get the indices for models with LR dFb < 0.1 Wm-2K-1 and the ones with LR dFB >= 0.5 Wm-2K-1
low_dfbi = i_fb_d < 0.1
high_dfbi = i_fb_d >= 0.5


#%% extract the model names as well as the data from their respective lists
low_mods = np.array(mods)[low_dfbi]
high_mods = np.array(mods)[high_dfbi]

low_regs = r_d_m[:, low_dfbi]
high_regs = r_d_m[:, high_dfbi]


#%% calculate the mean over the groups
low_regs_m = np.mean(low_regs, axis=1)
high_regs_m = np.mean(high_regs, axis=1)
low_regs_md = np.median(low_regs, axis=1)
high_regs_md = np.median(high_regs, axis=1)


#%% generate a bar plot showing the group means for both regions next to each other
fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

ax1.bar(0, low_regs_m[1], bottom=0, width=0.5, color="black", label=f"low global mean {fb_s} $\Delta$Fb")
ax1.bar(0.5, high_regs_m[1], bottom=0, width=0.5, color="gray", label=f"high global mean {fb_s} $\Delta$Fb")
ax1.bar(1.5, low_regs_m[0], bottom=0, width=0.5, color="black")
ax1.bar(2, high_regs_m[0], bottom=0, width=0.5, color="gray")

ax1.axhline(y=0, c="black")

ax1.legend(loc="lower right")

ax1.set_ylim((-3, 3))

ax1.set_ylabel(f"change of {fb2_n} Feedback in Wm$^{{-2}}$K$^{{-1}}$")

ax1.set_xticks([0.25, 1.75])
ax1.set_xticklabels([titl_ns[1], titl_ns[0]])

ax1.set_title(f"Comparison of {acrs[0]} and {acrs[1]} {fb2_n} Feedback Change\n" + 
              f"for Model Groups With High and Low {fb_s} $\Delta$Fb")

pl.savefig(pl_path + f"/PDF/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_{fb2_s}_Change.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_{fb2_s}_Change.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% generate a "circle" plot showing the group means AND individual values for both regions next to each other
fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(7, 7))

ax1.scatter(np.zeros(len(low_regs[1, :])), low_regs[1, :], edgecolor="blue", facecolor="white", 
            label=f"low global mean {fb_s} $\Delta$Fb")
ax1.scatter(np.zeros(len(high_regs[1, :]))+0.5, high_regs[1, :], edgecolor="red", facecolor="white", 
            label=f"high global mean {fb_s} $\Delta$Fb")
# for i in np.arange(len(high_mods)):
#    ax1.text(0.5, high_regs[1, i], high_mods[i], horizontalalignment="left", fontsize=8)
# end for i    
ax1.scatter(np.zeros(len(low_regs[0, :]))+1.5, low_regs[0, :], edgecolor="blue", facecolor="white")
ax1.scatter(np.zeros(len(high_regs[0, :]))+2, high_regs[0, :], edgecolor="red", facecolor="white")

ax1.scatter(0, low_regs_m[1], edgecolor="black", facecolor="white", s=80, zorder=0, linewidth=2, marker="o",
            label="mean")
ax1.scatter(0.5, high_regs_m[1], edgecolor="black", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")
ax1.scatter(1.5, low_regs_m[0], edgecolor="black", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")
ax1.scatter(2, high_regs_m[0], edgecolor="black", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")
ax1.scatter(0, low_regs_md[1], edgecolor="gray", facecolor="white", s=80, zorder=0, linewidth=2, marker="o",
            label="median")
ax1.scatter(0.5, high_regs_md[1], edgecolor="gray", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")
ax1.scatter(1.5, low_regs_md[0], edgecolor="gray", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")
ax1.scatter(2, high_regs_md[0], edgecolor="gray", facecolor="white", s=80, zorder=0, linewidth=2, marker="o")

ax1.axhline(y=0, c="black", zorder=-1)

ax1.legend(loc="lower right")

ax1.set_ylim((-4, 3.5))

ax1.set_ylabel(f"change of {fb2_n} Feedback in Wm$^{{-2}}$K$^{{-1}}$")

ax1.set_xticks([0.25, 1.75])
ax1.set_xticklabels([titl_ns[1], titl_ns[0]])

ax1.set_title(f"Comparison of {acrs[0]} and {acrs[1]} {fb2_n} Feedback Change\n" + 
              f"for Model Groups With High and Low {fb_s} $\Delta$Fb")

pl.savefig(pl_path + f"/PDF/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_{fb2_s}_Change_CirclePlot.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_{fb2_s}_Change_CirclePlot.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()



