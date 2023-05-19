"""
Generates Figure 8 in Eiselt and Graversen (2022), JCLI.

Scatter global mean total feedback change vs. (1) global mean surface temperature change at year 20 and (2) Arctic mean
surface temperature change at year 20. "At year 20" here means a 5-year average centered around year 20.

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

from Functions.Func_Region_Mean import region_mean
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


#%% set the year around which to center the dtas mean as well as the width of the interval
cent_yr = 20
yr_width = 2


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the variable (tas or ts)
var = "ts"
var_s = "Ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


#%% consider passing the clear-sky linearity test or not?
cslt = True
cslt_thr = 15


#%% set up the region for the correlation
acr = "AR"
rname = "Arctic"
x1s, x2s = [0], [360]
y1s, y2s = [75], [90]


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
yr_max_dfb_sort = []
dtas = []
dtas_nh = []
dtas_sh = []
dtas_rm = []
dtoa = []
forc_e = []
cl_fb_d = []

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
        reg_nc = Dataset(glob.glob(data_path5 + 
                                   f"Outputs/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}.nc")[0])
        greg_nc = Dataset(glob.glob(data_path5 + f"/Outputs/TOA_Imbalance_piC_Run/{a4x5}/{fb_ts}_Based/TOA*{mod}.nc")[0])
        dtas_nc = Dataset(glob.glob(data_path5 + f"/Data/{mod}/d{var}*{mod_n}_*abrupt4xCO2*.nc")[0])
        
        # load the dtas data
        dtas_mean = np.mean(np.mean(dtas_nc.variables[f"{var}_ch"][:], 
                                    axis=0)[(cent_yr-yr_width):(cent_yr+yr_width), :, :], axis=0)
        
        cl_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/" +
                                     f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])
        
        # load the cloud feedback
        cl_fb_d.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])        
        
        lat_dt = dtas_nc.variables["lat"][:]
        lon_dt = dtas_nc.variables["lon"][:]
        dtas.append(glob_mean(dtas_mean, lat_dt, lon_dt))
        dtas_hm = lat_mean(dtas_mean, lat_dt, lon_dt, lat=0)
        dtas_nh.append(dtas_hm["n_mean"])
        dtas_sh.append(dtas_hm["s_mean"])
        dtas_rm.append(region_mean(dtas_mean, x1s, x2s, y1s, y2s, lat_dt, lon_dt))
        
        dtoa.append(greg_nc.variables["toa_imb_as"][:])
        
        forc_e.append(greg_nc.variables["forcing_as_e"][0])
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
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
        reg_nc = Dataset(glob.glob(data_path6 + 
                                   f"Outputs/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}.nc")[0])
        greg_nc = Dataset(glob.glob(data_path6 + f"/Outputs/TOA_Imbalance_piC_Run/{a4x6}/{fb_ts}_Based/TOA*{mod}.nc")[0])
        dtas_nc = Dataset(glob.glob(data_path6 + f"/Data/{mod}/d{var}*{mod_n}_*abrupt-4xCO2*.nc")[0])
        
        # load the dtas data
        dtas_mean = np.mean(np.mean(dtas_nc.variables[f"{var}_ch"][:], 
                                    axis=0)[(cent_yr-yr_width):(cent_yr+yr_width), :, :], axis=0)
        
        cl_fb_nc = Dataset(glob.glob(data_path6 + "/Outputs/" +
                                     f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])        

        # load the cloud feedback
        cl_fb_d.append(cl_fb_nc.variables["C_fb_l"][0] - cl_fb_nc.variables["C_fb_e"][0])       
        
        lat_dt = dtas_nc.variables["lat"][:]
        lon_dt = dtas_nc.variables["lon"][:]
        dtas.append(glob_mean(dtas_mean, lat_dt, lon_dt))
        dtas_hm = lat_mean(dtas_mean, lat_dt, lon_dt, lat=0)
        dtas_nh.append(dtas_hm["n_mean"])
        dtas_sh.append(dtas_hm["s_mean"])
        dtas_rm.append(region_mean(dtas_mean, x1s, x2s, y1s, y2s, lat_dt, lon_dt))
        
        dtoa.append(greg_nc.variables["toa_imb_as"][:])
        
        forc_e.append(greg_nc.variables["forcing_as_e"][0])
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
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

dtas = np.array(dtas)
dtas_nh = np.array(dtas_nh)
dtas_sh = np.array(dtas_sh)
dtas_rm = np.array(dtas_rm)

dtoa = np.array(dtoa)

forc_e = np.array(forc_e)

ecs_e_a = np.array(ecs_e_l)
ecs_l_a = np.array(ecs_l_l)
ecs_d_a = np.array(ecs_d_l)

fb_e_a = np.array(fb_e_l)
fb_l_a = np.array(fb_l_l)
fb_d_a = np.array(fb_d_l)

cl_fb_d = np.array(cl_fb_d)

yr_max_dfb_sort = np.array(yr_max_dfb_sort)


#%% regress dTF on dts after 20 year
sl, yi, r, p = lr(dtas, fb_d_a)[:4]
sl_nh, yi_nh, r_nh, p_nh = lr(dtas_nh, fb_d_a)[:4]
sl_sh, yi_sh, r_sh, p_sh = lr(dtas_sh, fb_d_a)[:4]
sl_rm, yi_rm, r_rm, p_rm = lr(dtas_rm, fb_d_a)[:4]
sl_fe, yi_fe, r_fe, p_fe = lr(fb_e_a, fb_d_a)[:4]
sl_fl, yi_fl, r_fl, p_fl = lr(fb_l_a, fb_d_a)[:4]
sl_fot, yi_fot, r_fot, p_fot = lr(forc_e, dtas)[:4]


#%% generate an array with models an values for the low to moderate cloud dFb models
# str_dcfb = np.array(["MIROC-ESM", "GISS-E2-R", "CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2", 
#                      "NorESM2-LM", "NorESM2-MM", 'GFDL-CM4'])

str_dcfb = np.array(mods)[cl_fb_d > 0.5]
print("large cloud fb change models:")
print(str_dcfb)

wea_dcfb = []
wea_tas_d_m = []
wea_tas_rm_d_m = []
wea_fb_d_m = []
wea_fb_e_m = []
wea_fb_l_m = []
for m in np.arange(len(mods)):
    if len(np.intersect1d(mods[m], str_dcfb)) == 0:
        wea_dcfb.append(mods[m])
        wea_tas_d_m.append(dtas[m])
        wea_tas_rm_d_m.append(dtas_rm[m])
        wea_fb_d_m.append(fb_d_a[m])
        wea_fb_e_m.append(fb_e_a[m])
        wea_fb_l_m.append(fb_l_a[m])
    # end if
# end for m
        
wea_dcfb = np.array(wea_dcfb)
wea_tas_d_m = np.array(wea_tas_d_m)
wea_tas_rm_d_m = np.array(wea_tas_rm_d_m)
wea_fb_d_m = np.array(wea_fb_d_m)
wea_fb_e_m = np.array(wea_fb_e_m)
wea_fb_l_m = np.array(wea_fb_l_m)


#%% regress the EP warming on ECS and feedback
sl_wea, yi_wea, r_wea, p_wea = lr(wea_tas_d_m, wea_fb_d_m)[:4]
sl_wea_rm, yi_wea_rm, r_wea_rm, p_wea_rm = lr(wea_tas_rm_d_m, wea_fb_d_m)[:4]
sl_wea_e, yi_wea_e, r_wea_e, p_wea_e = lr(wea_fb_e_m, wea_fb_d_m)[:4]
sl_wea_l, yi_wea_l, r_wea_l, p_wea_l = lr(wea_fb_l_m, wea_fb_d_m)[:4]


#%% scatter dTF against dts centered at year 20
fsz = 18
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(dtas[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    # axes.text(dtas[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
p1, = axes.plot(dtas, dtas*sl + yi, c="gray", label=f"slope={np.round(sl, 2)}\nR={np.round(r, 2)}\np={np.round(p, 4)}")
p2, = axes.plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="black", 
                label=f"slope={np.round(sl_wea, 2)}\nR={np.round(r_wea, 2)}\np={np.round(p_wea, 4)}")

axes.tick_params(labelsize=fsz)
l1 = axes.legend(handles=[p1], fontsize=fsz, loc="upper right")
axes.legend(handles=[p2], fontsize=fsz, loc="lower left")
axes.add_artist(l1)

axes.set_xlabel(f"Surface warming at year {cent_yr} in K", fontsize=fsz)
axes.set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsz)

# axes.set_title(f"Global mean surface warming at year {cent_yr} v.\ntotal feedback change", fontsize=fsz+1)
axes.set_title("Global mean", fontsize=fsz+1)

axes.set_ylim((-0.1, 1.3))

# pl.savefig(pl_path + f"/PNG/d{var}_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.png", 
#            bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/d{var}_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.pdf", 
#            bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against regional mean dts centered at year 20
fsz = 18
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(dtas_rm[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    # axes.text(dtas[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
p1, = axes.plot(dtas_rm, dtas_rm*sl_rm + yi_rm, c="gray", 
                label=f"slope={np.round(sl_rm, 2)}\nR={np.round(r_rm, 2)}\np={np.round(p_rm, 4)}")
p2, = axes.plot(wea_tas_rm_d_m, wea_tas_rm_d_m*sl_wea_rm + yi_wea_rm, c="black", 
                label=f"slope={np.round(sl_wea_rm, 2)}  R={np.round(r_wea_rm, 2)}\np={np.round(p_wea_rm, 4)}")

axes.tick_params(labelsize=fsz)
l1 = axes.legend(handles=[p1], fontsize=fsz, loc="upper right")
axes.legend(handles=[p2], fontsize=fsz, loc="lower left")
axes.add_artist(l1)

axes.set_xlabel(f"Surface warming at year {cent_yr} in K", fontsize=fsz)
axes.set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsz)

# axes.set_title(f"{rname} surface warming at year {cent_yr} v.\ntotal feedback change", fontsize=fsz+1)
axes.set_title(f"{rname}", fontsize=fsz+1)

axes.set_ylim((-0.1, 1.3))

# pl.savefig(pl_path + f"/PNG/d{var}_{acr}_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.png", 
#            bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PDF/d{var}_{acr}_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.pdf", 
#            bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% double panel
fsz = 18

ylim = (-0.1, 1.26)
xlim_ar = (5.5, 18.35)
xlim_gl = (2.55, 6.0)

# set the position of th (a), (b) labels in the plots
x_ra_gl = xlim_gl[1] - xlim_gl[0]
tex_x_gl = xlim_gl[0] + x_ra_gl * 0.06
x_ra_ar = xlim_ar[1] - xlim_ar[0]
tex_x_ar = xlim_ar[0] + x_ra_ar * 0.06
y_ra = ylim[1] - ylim[0]
tex_y = ylim[0] + y_ra * 0.925

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(16, 5), sharey=True)

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes[0].scatter(dtas[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=2)
    axes[1].scatter(dtas_rm[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=2)
    # axes.text(dtas[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    

dtas_mima = np.array([np.min(dtas), np.max(dtas)])
dtas_rm_mima = np.array([np.min(dtas_rm), np.max(dtas_rm)])
p1, = axes[0].plot(dtas_mima, dtas_mima*sl + yi, c="black", linestyle=":", 
                   label=f"slope={np.round(sl, 2)}\nR={np.round(r, 2)}\np={np.round(p, 4)}")
p2, = axes[0].plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="black", 
                   label=f"slope={np.round(sl_wea, 2)}\n R={np.round(r_wea, 2)}\np={np.round(p_wea, 4)}")
p3, = axes[1].plot(dtas_rm_mima, dtas_rm_mima*sl_rm + yi_rm, c="black", linestyle=":",
                   label=f"slope={np.round(sl_rm, 2)}\nR={np.round(r_rm, 2)}\np={np.round(p_rm, 4)}")
p4, = axes[1].plot(wea_tas_rm_d_m, wea_tas_rm_d_m*sl_wea_rm + yi_wea_rm, c="black", 
                   label=f"slope={np.round(sl_wea_rm, 2)}  R={np.round(r_wea_rm, 2)}\np={np.round(p_wea_rm, 4)}")

axes[0].tick_params(labelsize=fsz)
axes[1].tick_params(labelsize=fsz)
l1 = axes[0].legend(handles=[p1], fontsize=fsz, loc="upper right")
axes[0].legend(handles=[p2], fontsize=fsz, loc="lower left")
axes[0].add_artist(l1)
l2 = axes[1].legend(handles=[p3], fontsize=fsz, loc="upper right")
axes[1].legend(handles=[p4], fontsize=fsz, loc="lower left")
axes[1].add_artist(l2)

axes[0].set_xlabel(f"Global surface warming at year {cent_yr} in K", fontsize=fsz)
axes[0].set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsz)
axes[1].set_xlabel(f"Arctic surface warming at year {cent_yr} in K", fontsize=fsz)
# axes[1].set_ylabel("Feedback change in Wm$^{-2}$K$^{-1}$", fontsize=fsz)

# axes.set_title(f"{rname} surface warming at year {cent_yr} v.\ntotal feedback change", fontsize=fsz+1)
# axes[0].set_title("Global mean", fontsize=fsz+1)
# axes[1].set_title("Arctic", fontsize=fsz+1)

axes[0].set_ylim(ylim)
axes[1].set_ylim(ylim)

axes[0].set_xlim(xlim_gl)
axes[1].set_xlim(xlim_ar)

axes[0].text(tex_x_gl, tex_y, "(a)", fontsize=21, horizontalalignment="center", verticalalignment="center")
axes[1].text(tex_x_ar, tex_y, "(b)", fontsize=21, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.05, hspace=None)

pl.savefig(pl_path + f"/PNG/d{var}_AR_GL_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/d{var}_AR_GL_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against NH dts centered at year 20
"""
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(dtas_nh[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(dtas_nh[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
axes.plot(dtas_nh, dtas_nh*sl_nh + yi_nh, c="black", 
          label=f"sl={np.round(sl_nh, 2)}  R={np.round(r_nh, 2)}  p={np.round(p_nh, 4)}")
# axes.plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="gray", 
#           label=f"sl={np.round(sl_wea, 2)}  R={np.round(r_wea, 2)}  p={np.round(p_wea, 4)}")

axes.legend()

axes.set_xlabel("$\Delta$ts in K")
axes.set_ylabel("$\Delta$TF in Wm$^{-2}$K$^{-1}$")

axes.set_title(f"NH $\Delta$ts at Year {cent_yr} v. $\Delta$TF")

pl.savefig(pl_path + f"/PNG/d{var}_NH_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/d{var}_NH_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against SH dts centered at year 20
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(dtas_sh[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(dtas_sh[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
axes.plot(dtas_sh, dtas_sh*sl_sh + yi_sh, c="black", 
          label=f"sl={np.round(sl_sh, 2)}  R={np.round(r_sh, 2)}  p={np.round(p_sh, 4)}")
# axes.plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="gray", 
#           label=f"sl={np.round(sl_wea, 2)}  R={np.round(r_wea, 2)}  p={np.round(p_wea, 4)}")

axes.legend()

axes.set_xlabel("$\Delta$ts in K")
axes.set_ylabel("$\Delta$TF in Wm$^{-2}$K$^{-1}$")

axes.set_title(f"SH $\Delta$ts at Year {cent_yr} v. $\Delta$TF")

pl.savefig(pl_path + f"/PNG/d{var}_SH_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/d{var}_SH_CenYr{cent_yr}_Width{yr_width}_v_dTF{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against early forcing
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(forc_e[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(forc_e[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
# axes.plot(dtas_sh, dtas_sh*sl_sh + yi_sh, c="black", 
#           label=f"sl={np.round(sl_sh, 2)}  R={np.round(r_sh, 2)}  p={np.round(p_sh, 4)}")
# axes.plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="gray", 
#           label=f"sl={np.round(sl_wea, 2)}  R={np.round(r_wea, 2)}  p={np.round(p_wea, 4)}")

axes.legend()

axes.set_xlabel("forcing in Wm$^{-2}$")
axes.set_ylabel("$\Delta$TF in Wm$^{-2}$K$^{-1}$")

axes.set_title("Early Forcing v. $\Delta$TF")

pl.savefig(pl_path + f"/PNG/ForcingEarly_v_dTF{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/ForcingEarly_v_dTF{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against early feedback
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(fb_e_a[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(fb_e_a[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
axes.plot(fb_e_a, fb_e_a*sl_fe + yi_fe, c="black", 
          label=f"sl={np.round(sl_fe, 2)}  R={np.round(r_fe, 2)}  p={np.round(p_fe, 4)}")
axes.plot(wea_fb_e_m, wea_fb_e_m*sl_wea_e + yi_wea_e, c="gray", 
          label=f"sl={np.round(sl_wea_e, 2)}  R={np.round(r_wea_e, 2)}  p={np.round(p_wea_e, 4)}")

axes.legend()

axes.set_xlabel("early feedback in Wm$^{-2}$K$^{-1}$")
axes.set_ylabel("$\Delta$TF in Wm$^{-2}$K$^{-1}$")

axes.set_title("Early Feedback v. $\Delta$TF")

pl.savefig(pl_path + f"/PNG/FeedbackEarly_v_dTF{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/FeedbackEarly_v_dTF{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter dTF against late feedback
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(fb_l_a[i], fb_d_a[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(fb_l_a[i], fb_d_a[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
axes.plot(fb_l_a, fb_l_a*sl_fl + yi_fl, c="black", 
          label=f"sl={np.round(sl_fl, 2)}  R={np.round(r_fl, 2)}  p={np.round(p_fl, 4)}")
axes.plot(wea_fb_l_m, wea_fb_l_m*sl_wea_l + yi_wea_l, c="gray", 
          label=f"sl={np.round(sl_wea_l, 2)}  R={np.round(r_wea_l, 2)}  p={np.round(p_wea_l, 4)}")

axes.legend()

axes.set_xlabel("late feedback in Wm$^{-2}$K$^{-1}$")
axes.set_ylabel("$\Delta$TF in Wm$^{-2}$K$^{-1}$")

axes.set_title("Late Feedback v. $\Delta$TF")

pl.savefig(pl_path + f"/PNG/FeedbackLate_v_dTF{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/FeedbackLate_v_dTF{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% scatter early forcing v. year-20 warming
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i, mod in enumerate(mods):
    if np.any(mod == str_dcfb):
        mark = "x"
    else:
        mark = "o"        
    # end if else           
    axes.scatter(forc_e[i], dtas[i], marker=mark, c=cols[i], s=40, linewidth=0.5)
    axes.text(forc_e[i], dtas[i], mod, horizontalalignment="center", fontsize=9, c=cols[i], alpha=0.5)
# end for i, mod    
axes.plot(forc_e, forc_e*sl_fot + yi_fot, c="black", 
          label=f"sl={np.round(sl_fot, 2)}  R={np.round(r_fot, 2)}  p={np.round(p_fot, 4)}")
# axes.plot(wea_tas_d_m, wea_tas_d_m*sl_wea + yi_wea, c="gray", 
#           label=f"sl={np.round(sl_wea, 2)}  R={np.round(r_wea, 2)}  p={np.round(p_wea, 4)}")

axes.legend()

axes.set_xlabel("forcing in Wm$^{-2}$")
axes.set_ylabel(f"$\Delta${var} in  K")

axes.set_title(f"Early Forcing v. $\Delta${var} at Year {cent_yr}")

pl.savefig(pl_path + f"/PNG/ForcEarly_v_dTF{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/ForcEarly_v_dTF{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()

"""
