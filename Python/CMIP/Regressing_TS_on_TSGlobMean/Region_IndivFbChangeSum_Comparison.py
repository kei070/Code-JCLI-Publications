"""
Generates Figure 5 in Eiselt and Graversen (2022), JCLI.

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
try:
    fb_s = sys.argv[1]
    fb2_s = sys.argv[2]
except:    
    fb_s = "LR"

    # fb_s = "C"
    # fb_s = "C_lw"
    # fb_s = "C_sw"
    # fb_s = "S"
    # fb_s = "Q"
    # fb_s = "Pl"
    # fb_s = "LR+Q"
    # fb_s = "Total"
# end try except


#%% set some strings for region name and abbrviation
# index = "TPWPI"  # "EPAA"  # "EQPNP"  #  "EPSO"  REPAA: Relative EP to Antarctica warming
# acrs = ["EWP", "WP", "EP", "AR", "GLExAR", "GL"]  # "EQP"  # 
# titl_ns = ["Ext. Western Pacific", "Western Pacific", "Eastern Pacific", "Arctic", "Excl. Arctic", "Global"]

# acrs = ["IPWP", "WP", "EP", "AR", "GLExAR", "GL"]  # "EQP"  # 
# titl_ns = ["Indo-Pacific\nWarm Pool", "Western\nPacific", "Eastern\nPacific", "Arctic", "Global\nwithout Arctic", 
#            "Global"]

acrs = ["Tropics", "NT", "ST", "NExT", "SExT", "GL"]  # "EQP"  # 
titl_ns = ["Tropics", "NH\nTropics", "SH\nTropics", "NH Extra-\nTropics", "SH Extra-\nTropics", "Global"]

# pl_pn = f"{index}_"


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the regions
"""
x1s = [50,   150, 260,   0,   0,   0]
x2s = [200,  170, 280, 360, 360, 360]
y1s = [-30,  -15, -30,  75, -90, -90]
y2s = [30,    15,   0,  90,  75,  90]
"""
x1s = [  0,   0,   0,   0,   0,   0]
x2s = [360, 360, 360, 360, 360, 360]
y1s = [-15,   0, -15,  15, -90, -90]
y2s = [ 15,  15,  0,   90, -15,  90]

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
cslt_thr = 15  # --> CSLT relative error threshold in % --> 15% or 20% or... 


#%% remove models with strong cloud feedback?
rm_str_dcfb = True


#%% indicate which models should be excluded
excl = ""
excl = "outliers"
# excl = "Fb not available"
# excl = "strong cloud dfb and outliers"
# excl = "moderate cloud dfb"
# excl = "CMIP6"
# excl = "CMIP5"
# excl = "low dFb"
# excl = "<[3,4]K dTas<"
# excl = "neg abs dTF"


#%% set up the feedback string list
fb_s_list = ["LR", "Q", "Q_lw", "Q_sw", "LR+Q", "C", "C_lw", "C_sw", "Pl", "S", "Total"]


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
    excl_arr = np.array(["MIROC-ESM", "GISS-E2-R", "NorESM1-M", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
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


#%% set up some initial dictionaries and lists --> NEEDED FOR CSLT !!!

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
float_greg_path = f"/Floating_ELP_Threshold_Range{thr_min}_{thr_max}_TF_Using_{var}.nc"
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
cslt_path = "/Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/Region_Fb_Comp/", exist_ok=True)
os.makedirs(pl_path + "/PNG/Region_Fb_Comp/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the models that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{var}_Based{thr_fn}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


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
il_fb_d_l = {}
il_fb_e_l = {}
il_fb_l_l = {}

# for CMIP5
for fb2_s in fb_s_list:
    
    i = 0
    mods = []

    # deg = 30  # latitude (in degree) north of which (and south of the negative of which) the mean is calculated
    cols = []

    # set up a counter for the models
    mod_count = 0


    # set up a list for the individual feedbacks
    il_fb_d_l[fb2_s] = []
    il_fb_e_l[fb2_s] = []
    il_fb_l_l[fb2_s] = []
    
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
                                          f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}*.nc")[0])        
            # reg_nc = Dataset(glob.glob(data_path5 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
            ind_loc_fb_nc = Dataset(glob.glob(data_path5 + f"/Feedbacks_Local/Kernel/abrupt4xCO2/{k_p[kl]}/" + 
                                              f"{var}_Based/*{mod}*.nc")[0])
            
            # load the cloud feedback to be able to jump over models with large cloud feedback change
            cl_fb_d = ind_fb_nc.variables["C_fb_l"][0] - ind_fb_nc.variables["C_fb_e"][0]
            if (cl_fb_d > 0.5) & (rm_str_dcfb):
                print(f"\nJumping {mod_pl} because of large cloud dFb")
                continue
            # end if            
    
            # get the model index in the max_dTF data
            mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
            
            # load the values - tas or ts regressions
            # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
            # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
            # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
            
            # get the lapse rate feedback
            if fb_s == fb2_s:
                i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
                i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])
            # end if
                
            # load the values - feedback values (focus here: get the right sorting!)
            fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
            fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
            fb_e_l.append(fb_e)
            fb_l_l.append(fb_l)
            fb_d_l.append(fb_l - fb_e)
            
            # load the local feedbacks
            il_fb_d_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_dfb_rg"][:])
            il_fb_e_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_fb_e_rg"][0, :, :])
            il_fb_l_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_fb_l_rg"][0, :, :])
            
    
            # get the year of max dTF (focus here: get the right sorting!)
            yr_max_dfb_sort.append(yr_max_dfb[mod_ind])

            mods.append(mod_pl)
            cols.append("blue")            
            i += 1
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
            ind_fb_nc = Dataset(glob.glob(data_path6 + 
                                          f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])        
            # reg_nc = Dataset(glob.glob(data_path6 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
            ind_loc_fb_nc = Dataset(glob.glob(data_path6 + f"/Feedbacks_Local/Kernel/abrupt-4xCO2/{k_p[kl]}/" +
                                              f"{var}_Based/*{mod}*.nc")[0])
            
            # load the cloud feedback to be able to jump over models with large cloud feedback change
            cl_fb_d = ind_fb_nc.variables["C_fb_l"][0] - ind_fb_nc.variables["C_fb_e"][0]
            if (cl_fb_d > 0.5) & (rm_str_dcfb):
                print(f"\nJumping {mod_pl} because of large cloud dFb")
                continue
            # end if
            
            # get the model index in the max_dTF data
            mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]        
        
            # load the values - tas or ts regressions
            # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
            # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
            # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
            
            if fb_s == fb2_s:
                i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
                i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])               
            # end if     
            
            # load the values - feedback values (focus here: get the right sorting!)
            fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
            fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
            fb_e_l.append(fb_e)
            fb_l_l.append(fb_l)
            fb_d_l.append(fb_l - fb_e)
            
            # load the local feedbacks
            il_fb_d_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_dfb_rg"][:])
            il_fb_e_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_fb_e_rg"][0, :, :])
            il_fb_l_l[fb2_s].append(ind_loc_fb_nc.variables[fb2_s + "_fb_l_rg"][0, :, :])
            
            # get the year of max dTF (focus here: get the right sorting!)
            yr_max_dfb_sort.append(yr_max_dfb[mod_ind])        
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
        
        print(f"\nData for {mod_pl} loaded...")
        
        mod_count += 1     
    # end for mod
    
    # convert the feedback lists in the dictionary to arrays
    il_fb_d_l[fb2_s] = np.array(il_fb_d_l[fb2_s])
    il_fb_e_l[fb2_s] = np.array(il_fb_e_l[fb2_s])
    il_fb_l_l[fb2_s] = np.array(il_fb_l_l[fb2_s])
    
# end for fb2_s

#%% print info about the CSLT and the loaded data
print(f"\n\n{len(pass_cslt)} models pass the CSLT and data for {mod_count} models is now loaded...\n\n")

print("Models NOT loaded:")
print(np.setdiff1d(pass_cslt, np.array(mods)))
print("\n")


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


#%%
# test1, lon_eu = eu_centric(lr_e_a[0, :, :], lon)


#%% extract the regions
r_d_m = {}
r_e_m = {}
r_l_m = {}

for fb2_s in fb_s_list:
    r_d_m[fb2_s] = []
    r_e_m[fb2_s] = []
    r_l_m[fb2_s] = []
    for ri in np.arange(len(x1s)):
        r_d = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_d_l[fb2_s], test_plt=False, 
                             plot_title=f"{fb2_s} Fb {acrs[ri]} Region")
        r_e = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_e_l[fb2_s], test_plt=False, 
                             plot_title=f"{fb2_s} Fb {acrs[ri]} Region")        
        r_l = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, il_fb_l_l[fb2_s], test_plt=False, 
                             plot_title=f"{fb2_s} Fb {acrs[ri]} Region")  
        
        # get the weights for averaging over the regions individually
        r_lons = lon[r_d[1][1][0, :]]
        r_lats = lat[r_d[1][0][:, 0]]
        r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
        r_latw = np.zeros(np.shape(r_d[0]))
        r_latw[:] = np.cos(r_latm / 180 * np.pi)[None, :, :]
        
        # calculate the mean over the regions
        r_d_m[fb2_s].append(np.average(r_d[0], weights=r_latw, axis=(1, 2)))
        r_e_m[fb2_s].append(np.average(r_e[0], weights=r_latw, axis=(1, 2)))
        r_l_m[fb2_s].append(np.average(r_l[0], weights=r_latw, axis=(1, 2)))
        
    # end for ri
    
    # convert the list to arrays  
    r_d_m[fb2_s] = np.array(r_d_m[fb2_s])
    r_e_m[fb2_s] = np.array(r_e_m[fb2_s])
    r_l_m[fb2_s] = np.array(r_l_m[fb2_s])
# end for fb2_s    


#%% get the indices for models with LR dFb < 0.1 Wm-2K-1 and the ones with LR dFB >= 0.5 Wm-2K-1
low_thr = 0.1
high_thr = 0.5

low_dfbi = i_fb_d < low_thr
high_dfbi = i_fb_d >= high_thr
ifb_d_lo = i_fb_d[low_dfbi]
ifb_d_hi = i_fb_d[high_dfbi]

pl_add = ""
tl_add = ""
if low_thr == 0.15:
    pl_add = pl_add + "_LP"
    tl_add = tl_add + " LP"
elif low_thr == 0.05:
    pl_add = pl_add + "_LM"
    tl_add = tl_add + " LM"
if high_thr == 0.55:
    pl_add = pl_add + "_HP"
    tl_add = tl_add + " HP"
elif high_thr == 0.45:
    pl_add = pl_add + "_HM"
    tl_add = tl_add + " HM"    
# end if elif


#%% extract the model names as well as the data from their respective lists
low_mods = np.array(mods)[low_dfbi]
high_mods = np.array(mods)[high_dfbi]

print("G1")
for i, j in zip(low_mods, ifb_d_lo): 
    print([i, j])
# end for i, j    
print("G2")
for i, j in zip(high_mods, ifb_d_hi): 
    print([i, j])
# end for i, j    

low_regs = {}
high_regs = {}
low_regs_m = {}
high_regs_m = {}

low_regs_e = {}
high_regs_e = {}
low_regs_m_e = {}
high_regs_m_e = {}

low_regs_l = {}
high_regs_l = {}
low_regs_m_l = {}
high_regs_m_l = {}

for fb2_s in fb_s_list:
    low_regs[fb2_s] = r_d_m[fb2_s][:, low_dfbi]
    high_regs[fb2_s] = r_d_m[fb2_s][:, high_dfbi]
    
    low_regs_e[fb2_s] = r_e_m[fb2_s][:, low_dfbi]
    high_regs_e[fb2_s] = r_e_m[fb2_s][:, high_dfbi]
    low_regs_l[fb2_s] = r_l_m[fb2_s][:, low_dfbi]
    high_regs_l[fb2_s] = r_l_m[fb2_s][:, high_dfbi]
    
    low_regs_m[fb2_s] = np.mean(low_regs[fb2_s], axis=1)
    high_regs_m[fb2_s] = np.mean(high_regs[fb2_s], axis=1)
    
    low_regs_m_e[fb2_s] = np.mean(low_regs_e[fb2_s], axis=1)
    high_regs_m_e[fb2_s] = np.mean(high_regs_e[fb2_s], axis=1)
    low_regs_m_l[fb2_s] = np.mean(low_regs_l[fb2_s], axis=1)
    high_regs_m_l[fb2_s] = np.mean(high_regs_l[fb2_s], axis=1)
# end for fb2_s


#%% show the feedback value
for fb2_s in fb_s_list: print(f"{fb2_s}: {high_regs_m[fb2_s][0]}")


#%% sort the feedbacks for plotting
low_fbs_r = {}
high_fbs_r = {}
low_efbs_r = {}
high_efbs_r = {}
low_lfbs_r = {}
high_lfbs_r = {}

fb_names = np.empty(len(high_regs_m.keys()), dtype=object)

low_fbs_r_n, low_fbs_r_nn = {}, {}
high_fbs_r_n, high_fbs_r_nn = {}, {}
low_fbs_r_p, low_fbs_r_pn = {}, {}
high_fbs_r_p, high_fbs_r_pn = {}, {}

low_efbs_r_n, low_efbs_r_nn = {}, {}
high_efbs_r_n, high_efbs_r_nn = {}, {}
low_lfbs_r_n, low_lfbs_r_nn = {}, {}
high_lfbs_r_n, high_lfbs_r_nn = {}, {}

low_efbs_r_p, low_efbs_r_pn = {}, {}
high_efbs_r_p, high_efbs_r_pn = {}, {}
low_lfbs_r_p, low_lfbs_r_pn = {}, {}
high_lfbs_r_p, high_lfbs_r_pn = {}, {}

for rind, regio in enumerate(acrs):
    low_fbs_r[regio] = np.zeros(len(low_regs_m.keys()))
    high_fbs_r[regio] = np.zeros(len(high_regs_m.keys()))
    low_efbs_r[regio] = np.zeros(len(low_regs_m_e.keys()))
    high_efbs_r[regio] = np.zeros(len(high_regs_m_e.keys()))
    low_lfbs_r[regio] = np.zeros(len(low_regs_m_l.keys()))
    high_lfbs_r[regio] = np.zeros(len(high_regs_m_l.keys()))    
    
    # for i, fb in enumerate(["S", "LR", "Pl", "Q_lw", "Q_sw", "C_lw", "C_sw"]):
    for i, fb in enumerate(["S", "LR", "Pl", "Q", "C"]):        
        low_fbs_r[regio][i] = low_regs_m[fb][rind]
        high_fbs_r[regio][i] = high_regs_m[fb][rind]

        low_efbs_r[regio][i] = low_regs_m_e[fb][rind]
        high_efbs_r[regio][i] = high_regs_m_e[fb][rind]
        
        low_lfbs_r[regio][i] = low_regs_m_l[fb][rind]
        high_lfbs_r[regio][i] = high_regs_m_l[fb][rind]        
        
        fb_names[i] = fb
    # end for i, fb
    
    # split into negative and positive - feedback change
    low_fbs_r_n[regio] = np.sort(low_fbs_r[regio][low_fbs_r[regio] < 0])
    low_fbs_r_nn[regio] = fb_names[low_fbs_r[regio] < 0][np.argsort(low_fbs_r[regio][low_fbs_r[regio] < 0])]
    high_fbs_r_n[regio] = np.sort(high_fbs_r[regio][high_fbs_r[regio] < 0])
    high_fbs_r_nn[regio] = fb_names[high_fbs_r[regio] < 0][np.argsort(high_fbs_r[regio][high_fbs_r[regio] < 0])]    
    
    low_fbs_r_p[regio] = np.sort(low_fbs_r[regio][low_fbs_r[regio] > 0])[::-1]
    low_fbs_r_pn[regio] = fb_names[low_fbs_r[regio] > 0][np.argsort(low_fbs_r[regio][low_fbs_r[regio] > 0])[::-1]]
    high_fbs_r_p[regio] = np.sort(high_fbs_r[regio][high_fbs_r[regio] > 0])[::-1]
    high_fbs_r_pn[regio] = fb_names[high_fbs_r[regio] > 0][np.argsort(high_fbs_r[regio][high_fbs_r[regio] > 0])[::-1]]
    
    # split into negative and positive - early feedback
    low_efbs_r_n[regio] = np.sort(low_efbs_r[regio][low_efbs_r[regio] < 0])
    low_efbs_r_nn[regio] = fb_names[low_efbs_r[regio] < 0][np.argsort(low_efbs_r[regio][low_efbs_r[regio] < 0])]    
    high_efbs_r_n[regio] = np.sort(high_efbs_r[regio][high_efbs_r[regio] < 0])
    high_efbs_r_nn[regio] = fb_names[high_efbs_r[regio] < 0][np.argsort(high_efbs_r[regio][high_efbs_r[regio] < 0])]
    
    low_efbs_r_p[regio] = np.sort(low_efbs_r[regio][low_efbs_r[regio] > 0])[::-1]
    low_efbs_r_pn[regio] = fb_names[low_efbs_r[regio] > 0][np.argsort(low_efbs_r[regio][low_efbs_r[regio] > 0])[::-1]]
    high_efbs_r_p[regio] = np.sort(high_efbs_r[regio][high_efbs_r[regio] > 0])[::-1]
    high_efbs_r_pn[regio] = fb_names[high_efbs_r[regio] > 0][np.argsort(high_efbs_r[regio][high_efbs_r[regio] > 
                                                                                                               0])[::-1]]
    
    # split into negative and positive - late feedback
    low_lfbs_r_n[regio] = np.sort(low_lfbs_r[regio][low_lfbs_r[regio] < 0])
    low_lfbs_r_nn[regio] = fb_names[low_lfbs_r[regio] < 0][np.argsort(low_lfbs_r[regio][low_lfbs_r[regio] < 0])]
    high_lfbs_r_n[regio] = np.sort(high_lfbs_r[regio][high_lfbs_r[regio] < 0])
    high_lfbs_r_nn[regio] = fb_names[high_lfbs_r[regio] < 0][np.argsort(high_lfbs_r[regio][high_lfbs_r[regio] < 0])]

    low_lfbs_r_p[regio] = np.sort(low_lfbs_r[regio][low_lfbs_r[regio] > 0])[::-1]
    low_lfbs_r_pn[regio] = fb_names[low_lfbs_r[regio] > 0][np.argsort(low_lfbs_r[regio][low_lfbs_r[regio] > 0])[::-1]]
    high_lfbs_r_p[regio] = np.sort(high_lfbs_r[regio][high_lfbs_r[regio] > 0])[::-1]
    high_lfbs_r_pn[regio] = fb_names[high_lfbs_r[regio] > 0][np.argsort(high_lfbs_r[regio][high_lfbs_r[regio] > 
                                                                                                               0])[::-1]]    
    
# end for rind, regio


#%% set up a dictionary for the colours corresponding to the individual feedbacks
fb_cols = {"S":"orange", "LR":"violet", "Pl":"firebrick", "Q":"green", "C":"blue", "Total":"black"}

"""
fb_cols = {"S":"orange", "LR":"violet", "Pl":"red", "Q":"green", "Q_lw":"green", "Q_sw":"lightgreen", "C":"blue", 
           "C_sw":"lightblue", "C_lw":"navy", "Total":"black"}

fb_n_long = {"S":"surface albedo", "LR":"lapse rate", "Pl":"Planck", "Q":"water vapour", "Q_lw":"LW water vapour", 
             "Q_sw":"SW water vapour", "C":"cloud", "C_sw":"SW Cloud", "C_lw":"LW Cloud", "Total":"total"}
"""

#%% generate a bar plot showing the group means for both regions next to each other
bar_space = 0.75

bwidth = 0.25

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

xtick_l = [bwidth/2]
for rind, regio in enumerate(acrs):
    
    # low dLR Fb - negative dFbs
    for i in np.arange(len(low_fbs_r_n[regio])):
        ax1.bar(rind*bar_space - 0.002, low_fbs_r_n[regio][i], bottom=np.sum(low_fbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[low_fbs_r_nn[regio][i]], edgecolor=fb_cols[low_fbs_r_nn[regio][i]])
    # end for i
    
    # low dLR Fb - positive dFbs
    for i in np.arange(len(low_fbs_r_p[regio])):
        ax1.bar(rind*bar_space - 0.002, low_fbs_r_p[regio][i], bottom=np.sum(low_fbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[low_fbs_r_pn[regio][i]], edgecolor=fb_cols[low_fbs_r_pn[regio][i]])
    # end for i

    ax1.text(rind*bar_space - 0.002, np.sum(low_fbs_r_p[regio]) + 0.2, "G1", horizontalalignment="center", fontsize=15)
    
    # low dLR Fb - total dFb
    # ax1.plot([rind*bar_space-bwidth/2 - 0.002, rind*bar_space+bwidth/2 - 0.002], 
    #          [low_regs_m["Total"][rind], low_regs_m["Total"][rind]], c="black", linewidth=1.75)
    ax1.scatter(rind*bar_space - 0.002, low_regs_m["Total"][rind], marker="D", c="black", zorder=101)

    # high dLR Fb - negative dFbs
    for i in np.arange(len(high_fbs_r_n[regio])):
        ax1.bar(rind*bar_space + 0.252, high_fbs_r_n[regio][i], bottom=np.sum(high_fbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[high_fbs_r_nn[regio][i]], edgecolor=fb_cols[high_fbs_r_nn[regio][i]])
    # end for i
    
    # high dLR Fb - positive dFbs
    for i in np.arange(len(high_fbs_r_p[regio])):
        ax1.bar(rind*bar_space + 0.252, high_fbs_r_p[regio][i], bottom=np.sum(high_fbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[high_fbs_r_pn[regio][i]], edgecolor=fb_cols[high_fbs_r_pn[regio][i]])
    # end for i
    
    ax1.text(rind*bar_space + 0.252, np.sum(high_fbs_r_p[regio]) + 0.2, "G2", horizontalalignment="center", fontsize=15)
    
    # high dLR Fb - total dFb
    # ax1.plot([rind*bar_space+0.252-bwidth/2, rind*bar_space+0.252+bwidth/2], 
    #          [high_regs_m["Total"][rind], high_regs_m["Total"][rind]], c="black", linewidth=1.5)
    ax1.scatter(rind*bar_space + 0.252, high_regs_m["Total"][rind], marker="D", c="black", zorder=101)    
   
    # add to the xticklabels
    xtick_l.append(xtick_l[rind] + bar_space)
    
# end for regio

# add for the legend
#for fb_leg, fb_col in zip(["LW cloud", "SW cloud", "LR", "Planck", "SA", "LW WV", "SW WV"], 
#                          ["C_lw", "C_sw", "LR", "Pl", "S", "Q_lw", "Q_sw"]):
for fb_leg, fb_col in zip(["Cloud", "LR", "Planck", "SA", "WV"], 
                          ["C", "LR", "Pl", "S", "Q"]):    
    ax1.bar(bar_space, 0, color=fb_cols[fb_col], label=fb_leg)
# end for fb_leg    
    
ax1.axhline(y=0, c="black", linewidth=0.75)

ax1.legend(loc="lower right", fontsize=15, title_fontsize=17, ncol=5)  # , title="Feedbacks"

ax1.tick_params(labelsize=16.5)

ax1.set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=16.5)

ax1.set_xticks(xtick_l[:-1])
ax1.set_xticklabels(titl_ns)

ax1.set_ylim((-2, 2))

# ax1.set_xticklabels([f"\nlow {fb_s} $\Delta$Fb", titl_ns[1], f"\nhigh {fb_s} $\Delta$Fb", 
#                      f"\nlow {fb_s} $\Delta$Fb", titl_ns[0], f"\nhigh {fb_s} $\Delta$Fb"])

# ax1.set_title("EWP, WP, EP, Arctic, ExAR, and Global Individual Feedback Changes\n" + 
#               f"for Model Groups With High and Low $\Delta \\alpha_{{{fb_s}}}$", fontsize=15)
ax1.set_title(f"Global and regional feedback changes for G1 and G2{tl_add}", fontsize=19)
"""
pl.savefig(pl_path + f"/PDF/Region_Fb_Comp/Comparison_Regional_{var}_Based_Fbs{pl_add}{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Fb_Comp/Comparison_Regional_{var}_Based_Fbs{pl_add}{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
"""
pl.show()
pl.close()


#%% generate a bar plot showing the group means for both regions next to each other - EARLY FEEDBACK
bar_space = 0.75

bwidth = 0.25

text_fsz = 14

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(12, 5))

xtick_l = [bwidth/2]
for rind, regio in enumerate(acrs):
    
    # low dLR Fb - negative dFbs
    for i in np.arange(len(low_efbs_r_n[regio])):
        ax1.bar(rind*bar_space - 0.002, low_efbs_r_n[regio][i], bottom=np.sum(low_efbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[low_efbs_r_nn[regio][i]], edgecolor=fb_cols[low_efbs_r_nn[regio][i]])
    # end for i
    
    # low dLR Fb - positive dFbs
    for i in np.arange(len(low_efbs_r_p[regio])):
        ax1.bar(rind*bar_space - 0.002, low_efbs_r_p[regio][i], bottom=np.sum(low_efbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[low_efbs_r_pn[regio][i]], edgecolor=fb_cols[low_efbs_r_pn[regio][i]])
    # end for i

    ax1.text(rind*bar_space - 0.002, np.sum(low_efbs_r_p[regio]) + 0.1, "G1", horizontalalignment="center",
             fontsize=text_fsz)
    
    # low dLR Fb - total dFb
    # ax1.plot([rind*bar_space-bwidth/2 - 0.002, rind*bar_space+bwidth/2 - 0.002], 
    #          [low_regs_m_e["Total"][rind], low_regs_m_e["Total"][rind]], c="black", linewidth=1.75)
    ax1.scatter(rind*bar_space - 0.002, low_regs_m_e["Total"][rind], marker="D", c="black", zorder=101)    

    # high dLR Fb - negative dFbs
    for i in np.arange(len(high_efbs_r_n[regio])):
        ax1.bar(rind*bar_space + 0.252, high_efbs_r_n[regio][i], bottom=np.sum(high_efbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[high_efbs_r_nn[regio][i]], edgecolor=fb_cols[high_efbs_r_nn[regio][i]])
    # end for i
    
    # high dLR Fb - positive dFbs
    for i in np.arange(len(high_efbs_r_p[regio])):
        ax1.bar(rind*bar_space + 0.252, high_efbs_r_p[regio][i], bottom=np.sum(high_efbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[high_efbs_r_pn[regio][i]], edgecolor=fb_cols[high_efbs_r_pn[regio][i]])
    # end for i
    
    ax1.text(rind*bar_space + 0.252, np.sum(high_efbs_r_p[regio]) + 0.1, "G2", horizontalalignment="center",
             fontsize=text_fsz)
    
    # high dLR Fb - total dFb
    # ax1.plot([rind*bar_space+0.252-bwidth/2, rind*bar_space+0.252+bwidth/2], 
    #          [high_regs_m_e["Total"][rind], high_regs_m_e["Total"][rind]], c="black", linewidth=1.5)
    ax1.scatter(rind*bar_space + 0.252, high_regs_m_e["Total"][rind], marker="D", c="black", zorder=101)
    
    # add to the xticklabels
    xtick_l.append(xtick_l[rind] + bar_space)
    
# end for regio

# add for the legend
# for fb_leg, fb_col in zip(["LW cloud", "SW cloud", "LR", "Planck", "SA", "LW WV", "SW WV"], 
#                           ["C_lw", "C_sw", "LR", "Pl", "S", "Q_lw", "Q_sw"]):
for fb_leg, fb_col in zip(["Cloud", "LR", "Planck", "SA", "WV"], 
                          ["C", "LR", "Pl", "S", "Q"]):
    
    ax1.bar(bar_space, 0, color=fb_cols[fb_col], label=fb_leg)
# end for fb_leg    
    
ax1.axhline(y=0, c="black", linewidth=0.75)
# ax1.axhline(y=0.5, c="gray", linewidth=0.75)

ax1.legend(loc="lower right", fontsize=14, title_fontsize=15, ncol=5)  # , title="Feedbacks"

# ax1.set_ylim((-1.5, 2))

ax1.tick_params(labelsize=15)

ax1.set_ylabel("Feedback in Wm$^{-2}$K$^{-1}$", fontsize=15)

ax1.set_xticks(xtick_l[:-1])
ax1.set_xticklabels(titl_ns)
# ax1.set_xticklabels([f"\nlow {fb_s} $\Delta$Fb", titl_ns[1], f"\nhigh {fb_s} $\Delta$Fb", 
#                      f"\nlow {fb_s} $\Delta$Fb", titl_ns[0], f"\nhigh {fb_s} $\Delta$Fb"])

# ax1.set_title("Individual Early Feedbacks in Several Regions\n" + 
#              f"for Model Groups With High and Low $\Delta \\alpha_{{{fb_s}}}$", fontsize=15)
ax1.set_title("Early feedbacks for G1 and G2", fontsize=17)

ax1.set_ylim(-6, 4.3)
"""
pl.savefig(pl_path + f"/PDF/Region_Fb_Comp/Comparison_Regional_{var}_Based_EarlyFbs{pl_add}{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Fb_Comp/Comparison_Regional_{var}_Based_EarlyFbs{pl_add}{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
"""
pl.show()
pl.close()


#%% generate a bar plot showing the group means for both regions next to each other - LATE FEEDBACK
bar_space = 0.75

bwidth = 0.25

text_fsz = 14

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(12, 5))

xtick_l = [bwidth/2]
for rind, regio in enumerate(acrs):
    
    # low dLR Fb - negative dFbs
    for i in np.arange(len(low_lfbs_r_n[regio])):
        ax1.bar(rind*bar_space - 0.002, low_lfbs_r_n[regio][i], bottom=np.sum(low_lfbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[low_lfbs_r_nn[regio][i]], edgecolor=fb_cols[low_lfbs_r_nn[regio][i]])
    # end for i
    
    # low dLR Fb - positive dFbs
    for i in np.arange(len(low_lfbs_r_p[regio])):
        ax1.bar(rind*bar_space - 0.002, low_lfbs_r_p[regio][i], bottom=np.sum(low_lfbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[low_lfbs_r_pn[regio][i]], edgecolor=fb_cols[low_lfbs_r_pn[regio][i]])
    # end for i

    ax1.text(rind*bar_space - 0.002, np.sum(low_lfbs_r_p[regio]) + 0.1, "G1", horizontalalignment="center",
             fontsize=text_fsz)    
    
    # low dLR Fb - total dFb
    # ax1.plot([rind*bar_space-bwidth/2 - 0.002, rind*bar_space+bwidth/2 - 0.002], 
    #          [low_regs_m_l["Total"][rind], low_regs_m_l["Total"][rind]], c="black", linewidth=1.75)
    ax1.scatter(rind*bar_space - 0.002, low_regs_m_l["Total"][rind], marker="D", c="black", zorder=101)
    
    # high dLR Fb - negative dFbs
    for i in np.arange(len(high_lfbs_r_n[regio])):
        ax1.bar(rind*bar_space + 0.252, high_lfbs_r_n[regio][i], bottom=np.sum(high_lfbs_r_n[regio][:i]), width=bwidth, 
                color=fb_cols[high_lfbs_r_nn[regio][i]], edgecolor=fb_cols[high_lfbs_r_nn[regio][i]])
    # end for i
    
    # high dLR Fb - positive dFbs
    for i in np.arange(len(high_lfbs_r_p[regio])):
        ax1.bar(rind*bar_space + 0.252, high_lfbs_r_p[regio][i], bottom=np.sum(high_lfbs_r_p[regio][:i]), width=bwidth, 
                color=fb_cols[high_lfbs_r_pn[regio][i]], edgecolor=fb_cols[high_lfbs_r_pn[regio][i]])
    # end for i
    
    ax1.text(rind*bar_space + 0.252, np.sum(high_lfbs_r_p[regio]) + 0.1, "G2", horizontalalignment="center",
             fontsize=text_fsz)
    
    # high dLR Fb - total dFb
    # ax1.plot([rind*bar_space+0.252-bwidth/2, rind*bar_space+0.252+bwidth/2], 
    #          [high_regs_m_l["Total"][rind], high_regs_m_l["Total"][rind]], c="black", linewidth=1.5)
    ax1.scatter(rind*bar_space + 0.252, high_regs_m_l["Total"][rind], marker="D", c="black", zorder=101)
    
    # add to the xticklabels
    xtick_l.append(xtick_l[rind] + bar_space)
    
# end for regio

# add for the legend
# for fb_leg, fb_col in zip(["LW cloud", "SW cloud", "LR", "Planck", "SA", "LW WV", "SW WV"], 
#                           ["C_lw", "C_sw", "LR", "Pl", "S", "Q_lw", "Q_sw"]):
for fb_leg, fb_col in zip(["Cloud", "LR", "Planck", "SA", "WV"], 
                          ["C", "LR", "Pl", "S", "Q"]):    
    ax1.bar(bar_space, 0, color=fb_cols[fb_col], label=fb_leg)
# end for fb_leg    
    
ax1.axhline(y=0, c="black", linewidth=0.75)
# ax1.axhline(y=0.5, c="gray", linewidth=0.75)

ax1.legend(loc="lower right", fontsize=14, title_fontsize=15, ncol=5)  # , title="Feedbacks"

# ax1.set_ylim((-1.5, 2))

ax1.tick_params(labelsize=15)

ax1.set_ylabel("Feedback in Wm$^{-2}$K$^{-1}$", fontsize=15)

ax1.set_xticks(xtick_l[:-1])
ax1.set_xticklabels(titl_ns)
# ax1.set_xticklabels([f"\nlow {fb_s} $\Delta$Fb", titl_ns[1], f"\nhigh {fb_s} $\Delta$Fb", 
#                      f"\nlow {fb_s} $\Delta$Fb", titl_ns[0], f"\nhigh {fb_s} $\Delta$Fb"])

# ax1.set_title("Individual Late Feedbacks in Several Regions\n" + 
#               f"for Model Groups With High and Low $\Delta \\alpha_{{{fb_s}}}$", fontsize=15)
ax1.set_title("Late feedbacks for G1 and G2", fontsize=17)

ax1.set_ylim(-6, 5.1)

"""
pl.savefig(pl_path + f"/PDF/Region_Fb_Comp/Comparison_Regional_{var}_Based_LateFbs{pl_add}{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Fb_Comp/Comparison_Regional_{var}_Based_LateFbs{pl_add}{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
"""

pl.show()
pl.close()
