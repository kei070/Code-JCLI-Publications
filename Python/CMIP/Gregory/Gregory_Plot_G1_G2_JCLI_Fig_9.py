"""
Generates Fig. 9 in our paper.

Comparing Gregory plots (clear-sky or all-sky) of the respective means over the two model groups G1 and G2 (see our
paper for details).
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
script_direc = "/home/kei070/Documents/Python_Scripts_PhD_Publication1/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

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
# acrs = ["EWP", "WP", "EP", "AR", "GLExAR", "GLExMPSH", "GLExTr", "GL"]  # "EQP"  # 
# titl_ns = ["Ext. Western Pacific", "Western Pacific", "Eastern Pacific", "Arctic", "Excl. Arctic", 
#            "Excl. MPSH", "Excl. Tropics", "Global"]

# acrs = ["GLExNHP", "GLExAR", "GLExMPSH", "GLExMPNH", "Tr", "Nof20N", "Sof20S", "GL"]  # "EQP"  # 
# titl_ns = ["Excl. NH Polar", "Excl. Arctic", "Excl. SH Mid&Pole", "Excl. NH Mid&Pole", "Tropics", "> 20N", "<20S", 
#            "Global"]
acrs = ["AR", "IPWP", "Tr", "ExTr", "Nof20N", "Sof20S", "GL"]  # "EQP"  # 
titl_ns = ["Arctic", "Warm Pool", "Tropics", "Extratropics", ">20N", "<20S", "Global"]


# pl_pn = f"{index}_"


#%% set the temperature variable (tas, ts)
t_var = "tas"


#%% all-sky (as) or clear-sky (cs)
sky = "as"
sky_ti = {"as":"", "cs":" (clear-sky)"}


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the two regions
# x1s = [50, 150, 260, 0, 0, 0, 0]
# x2s = [200, 170, 280, 360, 360, 360, 360]
# y1s = [-30, -15, -30, 75, -90, -45, -90]
# y2s = [30, 15, 0, 90, 75, 90, 90]

# x1s = [  0,   0,   0,   0,   0,   0,   0,   0]
# x2s = [360, 360, 360, 360, 360, 360, 360, 360]
# y1s = [-90, -90, -45, -90, -20,  20, -90, -90]
# y2s = [ 60,  75,  90,  45,  20,  90, -20,  90]
x1s = [  [0],  [50],   [0],     [0, 0],   [0],   [0],   [0]]
x2s = [[360], [200], [360], [360, 360], [360], [360], [360]]
y1s = [ [75], [-30], [-20],  [-90, 20],  [20], [-90], [-90]]
y2s = [ [90],  [30],  [20],  [-20, 90],  [90], [-20],  [90]]


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
data_path5 = (direc_5 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP5/")
data_path6 = (direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP6/")

# clear sky linearity test path
cslt_path = (direc_6 + f"/Uni/PhD/Tromsoe_UiT/Work/Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/")

# plot path
pl_path = direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/MultModPlots/Regional_vs_Global_TOA_Imb/"
pl_path2 = direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/MultModPlots/Regional_vs_Global_TOA_Imb_and_Warming/"
pl_path3 = direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/Plots_Paper_Proj1_New/"

# generate the plot paths
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)
os.makedirs(pl_path2 + "/PDF/", exist_ok=True)
os.makedirs(pl_path2 + "/PNG/", exist_ok=True)


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
cl_fb_d = []
max_dfbs = []
yr_max_dfb_sort = []
il_fb_d_l = {}
il_fb_e_l = {}
il_fb_l_l = {}

dts_l = []
dtoa_l = []

i = 0
mods = []

# deg = 30  # latitude (in degree) north of which (and south of the negative of which) the mean is calculated
cols = []

# set up a counter for the models
mod_count = 0


for mod, mod_n, mod_pl in zip(nl5.models, nl5.models_n, nl5.models_pl):
    
    if cslt:
        if not np.any(pass_cslt == mod_pl):
            print("\nJumping " + mod_pl + 
                  " because it fails the clear-sky linearity test...")
            continue
        # end if 
    # end if
    
    # exclude a chosen set of models
    if np.any(excl_arr == mod_pl):
        print("\nJumping " + mod_pl + " because it is in the exclude list...")
        continue
    # end if
    
    try:
        # load nc files
        ind_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/" +
                                      f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0]) 
        # reg_nc = Dataset(glob.glob(data_path5 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
        # ind_loc_fb_nc = Dataset(glob.glob(data_path5 + f"/Feedbacks_Local/Kernel/abrupt4xCO2/{k_p[kl]}/" + 
        #                                   f"{var}_Based/*{mod}*.nc")[0])
        # load cloud feedback files
        cl_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/" +
                                     f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])
        
        dts_nc = Dataset(glob.glob(data_path5 + f"/Data/{mod}/d{t_var}*_abrupt4xCO2*.nc")[0])
        dts_l.append(np.mean(dts_nc.variables[f"{t_var}_ch_rg"][:], axis=0))
        
        # print(f"\nLoading {mod} dTOA data...\n")
        dtoa_nc = Dataset(glob.glob(data_path5 + f"/Data/{mod}/dtoa*Regridded*")[0])
        dtoa_l.append(np.mean(dtoa_nc.variables["dtoa_" + sky][:], axis=0))
        # print(f"\n{mod} dTOA data successfully loaded...\n")
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
        # load the cloud feedback
        cl_fb_d.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])        
        
        # load the values - tas or ts regressions
        # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # get the lapse rate feedback
        i_fb_e.append(ind_fb_nc.variables["LR_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables["LR_fb_l"][0])
            
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e) 
    
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
                  " because it fails the clear-sky linearity test...")
            continue
        # end if 
    # end if
    
    # exclude a chosen set of models
    if np.any(excl_arr == mod_pl):
        print("\nJumping " + mod_pl + " because it is in the exclude list...")
        continue
    # end if
    
    try:
        # load nc files
        ind_fb_nc = Dataset(glob.glob(data_path6 + "/Outputs/" +
                                      f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])        
        # reg_nc = Dataset(glob.glob(data_path6 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}*.nc")[0])
        # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
        # ind_loc_fb_nc = Dataset(glob.glob(data_path6 + f"/Feedbacks_Local/Kernel/abrupt-4xCO2/{k_p[kl]}/" +
        #                                   f"{var}_Based/*{mod}*.nc")[0])          
        cl_fb_nc = Dataset(glob.glob(data_path6 + "/Outputs/" +
                                          f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]    
        
        dts_nc = Dataset(glob.glob(data_path6 + f"/Data/{mod}/d{t_var}*abrupt-4xCO2*")[0])
        dts_l.append(np.mean(dts_nc.variables[f"{t_var}_ch_rg"][:], axis=0))
        dtoa_nc = Dataset(glob.glob(data_path6 + f"/Data/{mod}/dtoa*Regridded*")[0])
        dtoa_l.append(np.mean(dtoa_nc.variables["dtoa_" + sky][:], axis=0))

        # load the cloud feedback
        cl_fb_d.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])        
        
        # load the values - tas or ts regressions
        # lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        # lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        # lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # get the lapse rate feedback
        i_fb_e.append(ind_fb_nc.variables["LR_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables["LR_fb_l"][0])
        
        # load the values - feedback values (focus here: get the right sorting!)
        fb_e = sls_e[mod_ind, dfb_ind[mod_ind]]
        fb_l = sls_l[mod_ind, dfb_ind[mod_ind]]
        fb_e_l.append(fb_e)
        fb_l_l.append(fb_l)
        fb_d_l.append(fb_l - fb_e)
        
        # get the year of max dTF (focus here: get the right sorting!)
        yr_max_dfb_sort.append(yr_max_dfb[mod_ind])        
        
        i += 1
        mods.append(mod_pl)
        cols.append("red")
    except:
        print("\nJumping " + mod_pl + " because data are not (yet) available...")
        continue
    # end try except

    mod_count += 1     
# end for mod
print("Models NOT loaded:")
print(np.setdiff1d(pass_cslt, np.array(mods)))
print("\n")


#%% convert the lists to arrays
dts_l = np.array(dts_l)
dtoa_l = np.array(dtoa_l)

i_fb_e = np.array(i_fb_e)
i_fb_l = np.array(i_fb_l)
i_fb_d = i_fb_l - i_fb_e

cl_fb_d = np.array(cl_fb_d)


#%% if requested remove the models with strong cloud feedback
if rm_str_dcfb:
    dts_l = dts_l[cl_fb_d <= 0.5, :, :, :]
    dtoa_l = dtoa_l[cl_fb_d <= 0.5, :, :, :]
    i_fb_d = i_fb_d[cl_fb_d <= 0.5]
    mods = np.array(mods)[cl_fb_d <= 0.5]
# end if    
    

#%% get the indices for models with LR dFb < 0.1 Wm-2K-1 and the ones with LR dFB > 0.5 Wm-2K-1
low_dfbi = i_fb_d < 0.1
high_dfbi = i_fb_d > 0.5


#%% print the models and their LR feedback changes
print("G1")
for i, j in zip(mods[low_dfbi], i_fb_d[low_dfbi]):
    print([i, j])
# end for i, j    
print("G2")
for i, j in zip(mods[high_dfbi], i_fb_d[high_dfbi]):
    print([i, j])
# end for i, j


#%% get the lat and lon for the target grid
targ_path = "/media/kei070/Work Disk 2/Uni/PhD/Tromsoe_UiT/Work/CMIP5/Data/CanESM2/"
lat = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lat"][:]
lon = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lon"][:]


#%% extract the regions
reg_dts = []
reg_dtoa = []

for ri in np.arange(len(x1s)):
    # calculate the mean over the regions
    reg_dts.append(region_mean(dts_l, x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon))
    reg_dtoa.append(region_mean(dtoa_l, x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon))
# end for ri
# gl_dts = np.array(reg_dts[-1])  # extract the global mean
# gl_dtoa = np.array(reg_dtoa[-1])  # extract the global mean


#%% calculate the global mean for all models
gl_dts = glob_mean(dts_l, lat, lon)
gl_dtoa = glob_mean(dtoa_l, lat, lon)


#%% extract groups G1 and G2
g1_gl_dts = np.mean(gl_dts[low_dfbi, :], axis=0)
g2_gl_dts = np.mean(gl_dts[high_dfbi, :], axis=0)
g1_gl_dtoa = np.mean(gl_dtoa[low_dfbi, :], axis=0)
g2_gl_dtoa = np.mean(gl_dtoa[high_dfbi, :], axis=0)


#%% extract the model names as well as the data from their respective lists
"""
low_mods = np.array(mods)[low_dfbi]
high_mods = np.array(mods)[high_dfbi]

dts_low_regs = reg_dts[:, low_dfbi, :]
dts_high_regs = reg_dts[:, high_dfbi, :]
dtoa_low_regs = reg_dtoa[:, low_dfbi, :]
dtoa_high_regs = reg_dtoa[:, high_dfbi, :]


#%% calculate the group means
dts_low_regs_m = np.mean(dts_low_regs, axis=1)
dts_high_regs_m = np.mean(dts_high_regs, axis=1)
dtoa_low_regs_m = np.mean(dtoa_low_regs, axis=1)
dtoa_high_regs_m = np.mean(dtoa_high_regs, axis=1)


#%% 21-150 regression for global means
ts_gsll_l, ts_gyil_l, ts_grl_l, ts_gpl_l = lr(np.arange(20, len(dts_low_regs_m[0, :])), dts_low_regs_m[-1, 20:])[:4]
ts_gsll_h, ts_gyil_h, ts_grl_h, ts_gpl_h = lr(np.arange(20, len(dts_low_regs_m[0, :])), dts_high_regs_m[-1, 20:])[:4]

toa_gsll_l, toa_gyil_l, toa_grl_l, toa_gpl_l = lr(np.arange(20, len(dtoa_low_regs_m[0, :])), 
                                                  dtoa_low_regs_m[-1, 20:])[:4]
toa_gsll_h, toa_gyil_h, toa_grl_h, toa_gpl_h = lr(np.arange(20, len(dtoa_low_regs_m[0, :])), 
                                                  dtoa_high_regs_m[-1, 20:])[:4]

#%% regress dTOA on dts
lwd = 3
reg_ind = 6
elp = 20
tt_sle_g1, tt_yie_g1, tt_re_g1, tt_pe_g1 = lr(g1_gl_dts[:elp], g1_gl_dtoa[:elp])[:4]
tt_sle_g2, tt_yie_g2, tt_re_g2, tt_pe_g2 = lr(g2_gl_dts[:elp], g2_gl_dtoa[:elp])[:4]
tt_sll_g1, tt_yil_g1, tt_rl_g1, tt_pl_g1 = lr(g1_gl_dts[elp:], g1_gl_dtoa[elp:])[:4]
tt_sll_g2, tt_yil_g2, tt_rl_g2, tt_pl_g2 = lr(g2_gl_dts[elp:], g2_gl_dtoa[elp:])[:4]

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.scatter(dts_low_regs_m[reg_ind, :elp], dtoa_low_regs_m[-1, :elp], marker="o", c="blue", edgecolor="blue")
axes.scatter(dts_low_regs_m[reg_ind, elp:], dtoa_low_regs_m[-1, elp:], marker="o", c="white",  edgecolor="blue",
             label="G1 ($\\alpha _{LR}$)")
axes.scatter(dts_high_regs_m[reg_ind, :elp], dtoa_high_regs_m[-1, :elp], marker="o", c="red", 
             label="G2 ($\\alpha _{LR}$)")
axes.scatter(dts_high_regs_m[reg_ind, elp:], dtoa_high_regs_m[-1, elp:], marker="o", c="white", edgecolor="red")

low_emax = np.max(dts_low_regs_m[reg_ind, :elp])
high_emax = np.max(dts_high_regs_m[reg_ind, :elp])

axes.plot(np.array([0, low_emax]), np.array([0, low_emax]) * tt_sle_g1 + tt_yie_g1, c="orange", linestyle="--",
          linewidth=lwd,)
axes.plot(np.array([0, high_emax]), np.array([0, high_emax]) * tt_sle_g2 + tt_yie_g2, c="gray", linestyle="--",
          linewidth=lwd,)
axes.scatter(0, tt_yie_g1, c="black", marker="^")
axes.scatter(0, tt_yie_g2, c="gray", marker="v")
axes.axvline(x=0, c="gray", linewidth=0.5)

axes.plot(dts_low_regs_m[reg_ind, elp:], dts_low_regs_m[reg_ind, elp:] * tt_sll_g1 + tt_yil_g1, c="orange", 
          linewidth=lwd,
          label=f"$\\Delta$= {np.round((tt_sll_g1-tt_sle_g1) / tt_sle_g1 * 100, 1)}%  " +
          f"({np.round((tt_sll_g1-tt_sle_g1), 2)} Wm$^{{-2}}$K$^{{-1}}$)")
axes.plot(dts_high_regs_m[reg_ind, elp:], dts_high_regs_m[reg_ind, elp:] * tt_sll_g2 + tt_yil_g2, c="gray",
          linewidth=lwd,
          label=f"$\\Delta$= {np.round((tt_sll_g2-tt_sle_g2) / tt_sle_g2 * 100, 1)}%  " + 
          f"({np.round((tt_sll_g2-tt_sle_g2), 2)} Wm$^{{-2}}$K$^{{-1}}$)")

axes.legend()

axes.set_xlabel(f"{titl_ns[reg_ind]} $\\Delta${t_var} in K")
axes.set_ylabel(f"{titl_ns[-1]} TOA Imbalance in Wm$^{{-2}}$")

axes.set_xlim((-0.05, 9.5))
axes.set_ylim((0.5, 7.75))

axes.set_title(f"{titl_ns[reg_ind]} $\\Delta${t_var} v. {titl_ns[-1]} TOA Imbalance")

pl.savefig(pl_path2 + f"/PNG/{acrs[reg_ind]}_v_Global_dTOA_d{t_var}_G1_G2.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path2 + f"/PDF/{acrs[reg_ind]}_v_Global_dTOA_d{t_var}_G1_G2.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
"""

#%% regress dTOA on dts
lwd = 3
c_g1 = "black"
c_g2 = "gray"
fsz = 19
reg_ind = 6
elp = 20
tt_sle_g1, tt_yie_g1, tt_re_g1, tt_pe_g1 = lr(g1_gl_dts[:elp], g1_gl_dtoa[:elp])[:4]
g1_ecs_e = - tt_yie_g1 / tt_sle_g1
tt_sle_g2, tt_yie_g2, tt_re_g2, tt_pe_g2 = lr(g2_gl_dts[:elp], g2_gl_dtoa[:elp])[:4]
g2_ecs_e = - tt_yie_g2 / tt_sle_g2
tt_sll_g1, tt_yil_g1, tt_rl_g1, tt_pl_g1 = lr(g1_gl_dts[elp:], g1_gl_dtoa[elp:])[:4]
g1_ecs_l = - tt_yil_g1 / tt_sll_g1
tt_sll_g2, tt_yil_g2, tt_rl_g2, tt_pl_g2 = lr(g2_gl_dts[elp:], g2_gl_dtoa[elp:])[:4]
g2_ecs_l = - tt_yil_g2 / tt_sll_g2

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

p1 = axes.scatter(g1_gl_dts[:elp], g1_gl_dtoa[:elp], marker="o", c="blue", edgecolor="blue",
                  label="G1")
axes.scatter(g1_gl_dts[elp:], g1_gl_dtoa[elp:], marker="o", c="white",  edgecolor="blue")
p2 = axes.scatter(g2_gl_dts[:elp], g2_gl_dtoa[:elp], marker="o", c="red", 
                  label="G2")
axes.scatter(g2_gl_dts[elp:], g2_gl_dtoa[elp:], marker="o", c="white", edgecolor="red")

# low_emax = np.max(dts_low_regs_m[reg_ind, :elp])
# high_emax = np.max(dts_high_regs_m[reg_ind, :elp])

p3, = axes.plot(np.array([0, g1_gl_dts[elp]]), np.array([0, g1_gl_dts[elp]]) * tt_sle_g1 + tt_yie_g1, c=c_g1, 
                linestyle="--", linewidth=lwd,
                label=f"slope={np.round(tt_sle_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$")
p4, = axes.plot(np.array([0, g2_gl_dts[elp]]), np.array([0, g2_gl_dts[elp]]) * tt_sle_g2 + tt_yie_g2, c=c_g2, 
                linestyle="--", linewidth=lwd,
                label=f"slope={np.round(tt_sle_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$")
axes.scatter(0, tt_yie_g1, c="blue", marker="^", s=50, zorder=101)
axes.scatter(0, tt_yie_g2, c="red", marker="v", s=50, zorder=101)
axes.axvline(x=0, c="gray", linewidth=0.5)
axes.axhline(y=0, c="gray", linewidth=0.5)

p5, = axes.plot(g1_gl_dts[elp:], g1_gl_dts[elp:] * tt_sll_g1 + tt_yil_g1, c=c_g1, linewidth=lwd+1,
                label=f"slope={np.round(tt_sll_g1, 2)} Wm$^{{-2}}$K$^{{-1}}$")
p6, = axes.plot(g2_gl_dts[elp:], g2_gl_dts[elp:] * tt_sll_g2 + tt_yil_g2, c=c_g2, linewidth=lwd+1,
                label=f"slope={np.round(tt_sll_g2, 2)} Wm$^{{-2}}$K$^{{-1}}$")


l1 = axes.legend(handles=[p1, p2], fontsize=fsz, loc="lower left")
axes.legend(handles=[p3, p4, p5, p6], fontsize=fsz, loc="upper right")
axes.add_artist(l1)

axes.set_xlabel("Global mean surface temperature in K", fontsize=fsz)
axes.set_ylabel("Global mean TOA imbalance in Wm$^{{-2}}$", fontsize=fsz)

axes.set_xlim((-0.075, 9.5))
axes.set_ylim((-0.075, 7.75))

axes.set_title("Gregory plot for G1 and G2" + sky_ti[sky], fontsize=fsz+1)

axes.tick_params(labelsize=fsz)

pl.savefig(pl_path3 + f"/PNG/Gregory_d{t_var}{cslt_fadd}{excl_fadd}_G1_G2_{sky}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path3 + f"/PDF/Gregory_d{t_var}{cslt_fadd}{excl_fadd}_G1_G2_{sky}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()

