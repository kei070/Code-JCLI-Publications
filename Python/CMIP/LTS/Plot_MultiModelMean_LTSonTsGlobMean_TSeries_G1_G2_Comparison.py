"""
Comparison og G1 and G2 LTS time series (see the paper for details).

Plots (among others) panels (e) and (f) of Fig. 6 in Eiselt and Graversen (2022), JCLI, and produces the data for that.

CORRECTION: The quantity called EIS here is actually LTS (lower tropospheric stability). We were following Ceppi and
Gregory (2018), PNAS, who also make this mistake.
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
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% choose the variable (tas or ts)
var = "tas"
var_s = "Tas"


#%% choose the threshold year; if this is set to 0, the "year of maximum feedback change" will be taken from the below
#   dictionary
thr_yr = 20

thr_min = 15
thr_max = 75


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

direc_5 = "/media/kei070/Work Disk 2"
direc_6 = "/media/kei070/Seagate"


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
# end if


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
data_path5 = (direc_5 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP5/Outputs/")  # SETS BASIC DATA DIRECTORY
data_path6 = (direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/CMIP6/Outputs/")  # SETS BASIC DATA DIRECTORY

# clear sky linearity test path
cslt_path = (direc_6 + f"/Uni/PhD/Tromsoe_UiT/Work/Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/")

# plot path
pl_path = direc_6 + "/Uni/PhD/Tromsoe_UiT/Work/MultModPlots/MMM_EIS/"

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)
# os.makedirs(pl_path + "/ZonalMeans/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/ZonalMeans/PNG/", exist_ok=True)

out_path = "/media/kei070/Seagate/Uni/PhD/Tromsoe_UiT/Work/Outputs/G1_G2_Comp/"
os.makedirs(out_path, exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{var}_Based_ThrYr20_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
mm_eis_pir_d = dict()
mm_eis_foa_d = dict()

eis_pir_d = dict()
eis_foa_d = dict()

eis_pir_zm_d = dict()
eis_foa_zm_d = dict()

mm_eis_pir_zm_d = dict()
mm_eis_foa_zm_d = dict()

mods_d = dict()

trop_band_mm_d = dict()

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
    
    # set up the dictionaries to store all feedbacks
    lr_e = dict()
    lr_l = dict()
    lr_d = dict()
    
    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    eis_pir = []
    eis_foa = []
    
    # for CMIP 5---------------------------------------------------------------------------------------------------------
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
            eis_nc = Dataset(glob.glob(data_path5 + f"/EIS/EIS_and_EIS_on_{var}_LR/*{mod}*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path5 +  
                                        f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])              
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
            eis_pir.append(eis_nc.variables["eis_pic_run_rg"][:])
            eis_foa.append(eis_nc.variables["eis_forced_rg"][:])
            
            i += 1
            mods.append(mod_pl)
            cols.append("blue")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except        
        
        mod_count += 1
    # end for mod -------------------------------------------------------------------------------------------------------
    
    # for CMIP6 ---------------------------------------------------------------------------------------------------------
    for mod, mod_n, mod_pl in zip(nl6.models, nl6.models_n, nl6.models_pl):
        # if mod_pl == "FGOALS-g3":
        #     print("Excluding FGOASL-g3 for now because of interpolation problems...")
        #     continue
        # end if
        
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
            eis_nc = Dataset(glob.glob(data_path6 + f"/EIS/EIS_and_EIS_on_{var}_LR/*{mod}*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path6 +
                                        f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{var}_Based/*{mod}.nc")[0])
            
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
            eis_pir.append(eis_nc.variables["eis_pic_run_rg"][:])
            eis_foa.append(eis_nc.variables["eis_forced_rg"][:])
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod -------------------------------------------------------------------------------------------------------


    #% get lat and lon
    lat = eis_nc.variables["lat_rg"][:]
    lon = eis_nc.variables["lon_rg"][:]


    #% convert the lists into numpy arrays
    eis_pir_a = np.array(eis_pir)
    eis_foa_a = np.array(eis_foa)
    
    
    eis_pir_a[eis_pir_a > 100] = 0
    eis_foa_a[eis_foa_a > 100] = 0
    eis_pir_a[eis_pir_a < 0] = 0
    eis_foa_a[eis_foa_a < 0] = 0


    #% calculate the individual zonal means
    eis_pir_zm = np.mean(eis_pir_a, axis=-1)
    eis_foa_zm = np.mean(eis_foa_a, axis=-1)
    

    #% calculate the multi-model means
    mm_eis_pir = np.mean(eis_pir_a, axis=0)
    mm_eis_foa = np.mean(eis_foa_a, axis=0)


    #% calculate the zonal means
    mm_eis_pir_zm = np.mean(mm_eis_pir, axis=-1)
    mm_eis_foa_zm = np.mean(mm_eis_foa, axis=-1)


    #% calculate the band means
    # trop_lat = 30
    # lat_means = lat_mean(mm_lr_d, lat, lon, lat=trop_lat, return_dic="new")
    
    """
    x1 = 0
    x2 = 360
    y1 = -30
    y2 = 0
    lonm, latm  = np.meshgrid(lon, lat)
    weights = np.zeros(np.shape(mm_eis_pir))
    weights[:, :] = np.cos(latm / 180 * np.pi)
    trop_reg = extract_region(x1, x2, y1, y2, lat, lon, mm_eis_pir, test_plt=True, plot_title="Southern Tropics")
    trop_latm = extract_region(x1, x2, y1, y2, lat, lon, latm, test_plt=True, plot_title="Southern Tropics")[0]
    trop_band_mm = np.average(trop_reg[0], weights=trop_latm, axis=0)
    """
    
    # store everything in dictionaries
    mm_eis_pir_d[g12] = mm_eis_pir
    mm_eis_foa_d[g12] = mm_eis_foa
    
    eis_pir_d[g12] = eis_pir_a
    eis_foa_d[g12] = eis_foa_a
    
    eis_pir_zm_d[g12] = eis_pir_zm
    eis_foa_zm_d[g12] = eis_foa_zm
    
    mm_eis_pir_zm_d[g12] = mm_eis_pir_zm
    mm_eis_foa_zm_d[g12] = mm_eis_foa_zm
    
    mods_d[g12] = mods
    
    # trop_band_mm_d[excl] = trop_band_mm

# end for excl


#%% remove unreasonable values
for g12 in ["G1", "G2"]:
    mm_eis_pir_d[g12][mm_eis_pir_d[g12] > 500] = np.nan
    mm_eis_foa_d[g12][mm_eis_foa_d[g12] > 500] = np.nan
    
    eis_pir_zm_d[g12][eis_pir_zm_d[g12] > 500] = np.nan
    eis_foa_zm_d[g12][eis_foa_zm_d[g12] > 500] = np.nan
    
    mm_eis_pir_zm_d[g12][mm_eis_pir_zm_d[g12] > 500] = np.nan
    mm_eis_foa_zm_d[g12][mm_eis_foa_zm_d[g12] > 500] = np.nan
# end for exc


#%% extract the Arctic region 75N-90N and look into the time development
x1s, x2s = [0], [360]
y1s, y2s = [75], [90]
eis_pir_regs = {}
eis_foa_regs = {}
eis_pir_regs["G1"] = region_mean(eis_pir_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_pir_regs["G2"] = region_mean(eis_pir_d["G2"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_regs["G1"] = region_mean(eis_foa_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_regs["G2"] = region_mean(eis_foa_d["G2"], x1s, x2s, y1s, y2s, lat, lon)

eis_pir_reg = {}
eis_foa_reg = {}
eis_pir_reg_std = {}
eis_foa_reg_std = {}

eis_pir_reg["G1"] = np.mean(eis_pir_regs["G1"], axis=0)
eis_pir_reg["G2"] = np.mean(eis_pir_regs["G2"], axis=0)
eis_foa_reg["G1"] = np.mean(eis_foa_regs["G1"], axis=0)
eis_foa_reg["G2"] = np.mean(eis_foa_regs["G2"], axis=0)
eis_pir_reg_std["G1"] = np.std(eis_pir_regs["G1"], axis=0)
eis_pir_reg_std["G2"] = np.std(eis_pir_regs["G2"], axis=0)
eis_foa_reg_std["G1"] = np.std(eis_foa_regs["G1"], axis=0)
eis_foa_reg_std["G2"] = np.std(eis_foa_regs["G2"], axis=0)

x1s, x2s = [260], [280]
y1s, y2s = [-30], [0]
eis_pir_ep = {}
eis_foa_ep = {}
eis_pir_ep["G1"] = region_mean(mm_eis_pir_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_pir_ep["G2"] = region_mean(mm_eis_pir_d["G2"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_ep["G1"] = region_mean(mm_eis_foa_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_ep["G2"] = region_mean(mm_eis_foa_d["G2"], x1s, x2s, y1s, y2s, lat, lon)

x1s, x2s = [50], [200]
y1s, y2s = [-15], [15]
eis_pir_ipwp = {}
eis_foa_ipwp = {}
eis_pir_ipwp["G1"] = region_mean(mm_eis_pir_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_pir_ipwp["G2"] = region_mean(mm_eis_pir_d["G2"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_ipwp["G1"] = region_mean(mm_eis_foa_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_ipwp["G2"] = region_mean(mm_eis_foa_d["G2"], x1s, x2s, y1s, y2s, lat, lon)

x1s, x2s = [150], [170]
y1s, y2s = [-30], [30]
eis_pir_wp = {}
eis_foa_wp = {}
eis_pir_wp["G1"] = region_mean(mm_eis_pir_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_pir_wp["G2"] = region_mean(mm_eis_pir_d["G2"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_wp["G1"] = region_mean(mm_eis_foa_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_wp["G2"] = region_mean(mm_eis_foa_d["G2"], x1s, x2s, y1s, y2s, lat, lon)

x1s, x2s = [0], [360]
y1s, y2s = [-50], [50]
eis_pir_gls = {}
eis_foa_gls = {}
eis_pir_gls["G1"] = region_mean(eis_pir_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_pir_gls["G2"] = region_mean(eis_pir_d["G2"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_gls["G1"] = region_mean(eis_foa_d["G1"], x1s, x2s, y1s, y2s, lat, lon)
eis_foa_gls["G2"] = region_mean(eis_foa_d["G2"], x1s, x2s, y1s, y2s, lat, lon)

eis_pir_gl = {}
eis_foa_gl = {}
eis_pir_gl_std = {}
eis_foa_gl_std = {}

eis_pir_gl["G1"] = np.mean(eis_pir_gls["G1"], axis=0)
eis_pir_gl["G2"] = np.mean(eis_pir_gls["G2"], axis=0)
eis_foa_gl["G1"] = np.mean(eis_foa_gls["G1"], axis=0)
eis_foa_gl["G2"] = np.mean(eis_foa_gls["G2"], axis=0)
eis_pir_gl_std["G1"] = np.std(eis_pir_gls["G1"], axis=0)
eis_pir_gl_std["G2"] = np.std(eis_pir_gls["G2"], axis=0)
eis_foa_gl_std["G1"] = np.std(eis_foa_gls["G1"], axis=0)
eis_foa_gl_std["G2"] = np.std(eis_foa_gls["G2"], axis=0)


#%% plot the time series of the absolute values
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].plot(eis_pir_reg["G1"], c="blue", label="G1 21-year running piControl", linestyle="--")
axes[0].plot(eis_pir_reg["G2"], c="red", label="G2 21-year running piControl", linestyle="--")
axes[0].plot(eis_foa_reg["G1"], c="blue", label="G1 abrupt4xCO2")
axes[0].plot(eis_foa_reg["G2"], c="red", label="G2 abrupt4xCO2")

axes[1].plot(eis_pir_gl["G1"], c="blue", label="G1 global 21-year running piControl", linestyle="--")
axes[1].plot(eis_pir_gl["G2"], c="red", label="G2 global 21-year running piControl", linestyle="--")
axes[1].plot(eis_foa_gl["G1"], c="blue", label="G1 abrupt4xCO2")
axes[1].plot(eis_foa_gl["G2"], c="red", label="G2 abrupt4xCO2")

axes[0].legend()
axes[1].legend()

axes[0].set_title("Arctic (75-90$\degree$N)")
axes[1].set_title("Global (50$\degree$S-50$\degree$N)")

fig.suptitle("EIS Evolution")

axes[0].set_xlabel("time since abrupt4xCO2 branching")
axes[0].set_ylabel("EIS in K")
axes[1].set_xlabel("time since abrupt4xCO2 branching")
axes[1].set_ylabel("EIS in K")

pl.show()
pl.close()


#%% plot the time series of the differences
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].plot(eis_pir_reg["G1"]-eis_pir_reg["G2"], c="black", label="21-year running piControl", 
             linestyle="--")
axes[0].plot(eis_foa_reg["G1"]-eis_foa_reg["G2"], c="gray", label="abrupt4xCO2")

axes[0].axhline(y=0, c="gray",linewidth=0.5)

axes[1].plot(eis_pir_gl["G1"]-eis_pir_gl["G2"], c="black", 
             linestyle="--")
axes[1].plot(eis_foa_gl["G1"]-eis_foa_gl["G2"], c="gray")

axes[1].axhline(y=0, c="gray",linewidth=0.5)

axes[0].legend()

axes[0].set_title("Arctic (75-90$\degree$N)")
axes[1].set_title("Global (50$\degree$S-50$\degree$N)")

fig.suptitle("EIS Evolution Group Difference (G2-G1)")

axes[0].set_xlabel("time since abrupt4xCO2 branching")
axes[0].set_ylabel("EIS in K")
axes[1].set_xlabel("time since abrupt4xCO2 branching")
axes[1].set_ylabel("EIS in K")

pl.show()
pl.close()


#%% plot global mean stability and Arctic stability in the forced simulation
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(12, 6))

axes[0].plot(eis_foa_reg["G1"], c="blue", label="G1 Arctic")
axes[0].plot(eis_foa_reg["G2"], c="red", label="G2 Arctic")
axes[0].plot(eis_foa_gl["G1"], c="blue", label="G1 global", linestyle="--")
axes[0].plot(eis_foa_gl["G2"], c="red", label="G2 global", linestyle="--")

axes[1].plot(eis_foa_reg["G2"] - eis_foa_reg["G1"], c="black", label="Arctic")
axes[1].plot(eis_foa_gl["G2"] - eis_foa_gl["G1"], c="black", label="global", linestyle="--")
axes[1].axhline(y=0, c="gray", linewidth=0.5)

axes[0].legend()
axes[1].legend()

axes[0].set_title("EIS Evolution abrupt4xCO2")
axes[1].set_title("Difference in EIS Evolution abrupt4xCO2 (G2-G1)")
axes[0].set_xlabel("time since abrupt4xCO2 branching")
axes[0].set_ylabel("EIS in K")
axes[1].set_xlabel("time since abrupt4xCO2 branching")
axes[1].set_ylabel("EIS in K")

pl.show()
pl.close()

axes[0].plot(eis_foa_reg["G1"], c="blue", label="G1 Arctic")
axes[0].plot(eis_foa_reg["G2"], c="red", label="G2 Arctic")
axes[0].plot(eis_foa_gl["G1"], c="blue", label="G1 global", linestyle="--")
axes[0].plot(eis_foa_gl["G2"], c="red", label="G2 global", linestyle="--")

axes[1].plot(eis_foa_reg["G2"] - eis_foa_reg["G1"], c="black", label="Arctic")
axes[1].plot(eis_foa_gl["G2"] - eis_foa_gl["G1"], c="black", label="global", linestyle="--")
axes[1].axhline(y=0, c="gray", linewidth=0.5)

axes[0].legend()
axes[1].legend()

axes[0].set_title("EIS Evolution abrupt4xCO2")
axes[1].set_title("Difference in EIS Evolution abrupt4xCO2 (G2-G1)")
axes[0].set_xlabel("time since abrupt4xCO2 branching")
axes[0].set_ylabel("EIS in K")
axes[1].set_xlabel("time since abrupt4xCO2 branching")
axes[1].set_ylabel("EIS in K")

pl.show()
pl.close()


#%% plot Arctic stability in the forced simulation

# perform the t-test for the mean difference
ttest = ttest_ind(eis_foa_regs["G1"], eis_foa_regs["G2"], axis=0)

fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

axes.plot(eis_foa_reg["G1"], c="blue", label="G1")
axes.fill_between(np.arange(len(eis_foa_reg["G1"])), eis_foa_reg["G1"] - eis_foa_reg_std["G1"], 
                  eis_foa_reg["G2"] + eis_foa_reg_std["G2"], facecolor="blue", alpha=0.25)
axes.plot(eis_foa_reg["G2"], c="red", label="G2")
axes.fill_between(np.arange(len(eis_foa_reg["G2"])), eis_foa_reg["G2"] - eis_foa_reg_std["G2"], 
                  eis_foa_reg["G2"] + eis_foa_reg_std["G2"], facecolor="red", alpha=0.25)

axr.plot(ttest.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=1, c="gray", zorder=1)
axes.plot([], [], c="black", linewidth=1, label="p-value")

# axes.plot(eis_foa_ipwp["G2"], c="blue", label="G1 IPWP", linestyle=":")
# axes.plot(eis_foa_ipwp["G1"], c="red", label="G2 IPWP", linestyle=":")
# axes.plot(eis_foa_ep["G2"], c="blue", label="G1 EP", linestyle="--")
# axes.plot(eis_foa_ep["G1"], c="red", label="G2 EP", linestyle="--")
# axes.plot(eis_foa_gl["G2"], c="blue", label="G1 global", linestyle="--")
# axes.plot(eis_foa_gl["G1"], c="red", label="G2 global", linestyle="--")

axes.legend(fontsize=fsz, loc="upper center", ncol=2)

axes.set_title(f"Arctic EIS for G1 and G2{tl_add}", fontsize=fsz)
axes.set_xlabel("Years since branching of abrupt-4xCO2", fontsize=fsz)
axes.set_ylabel("EIS in K", fontsize=fsz)
axes.tick_params(labelsize=fsz)

axr.tick_params(labelsize=fsz)

pl.savefig(pl_path + f"/PNG/EIS_Arctic_AnMean_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/EIS_Arctic_AnMean_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot global stability in the forced simulation

# perform the t-test for the mean difference
ttest = ttest_ind(eis_foa_gls["G1"], eis_foa_gls["G2"], axis=0)

fsz = 18

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(8, 5))

axr = axes.twinx()

# axes.plot(eis_foa_reg["G2"], c="blue", label="G1 Arctic")
# axes.plot(eis_foa_reg["G1"], c="red", label="G2 Arctic")
# axes.plot(eis_foa_ipwp["G2"], c="blue", label="G1 IPWP", linestyle=":")
# axes.plot(eis_foa_ipwp["G1"], c="red", label="G2 IPWP", linestyle=":")
# axes.plot(eis_foa_ep["G2"], c="blue", label="G1 EP", linestyle="--")
# axes.plot(eis_foa_ep["G1"], c="red", label="G2 EP", linestyle="--")
axes.plot(eis_foa_gl["G1"], c="blue", label="G1", linestyle="-")
axes.fill_between(np.arange(len(eis_foa_gl["G1"])), eis_foa_gl["G1"] - eis_foa_gl_std["G1"], 
                  eis_foa_gl["G2"] + eis_foa_gl_std["G2"], facecolor="blue", alpha=0.25)
axes.plot(eis_foa_gl["G2"], c="red", label="G2", linestyle="-")
axes.fill_between(np.arange(len(eis_foa_gl["G2"])), eis_foa_gl["G2"] - eis_foa_gl_std["G2"], 
                  eis_foa_gl["G2"] + eis_foa_gl_std["G2"], facecolor="red", alpha=0.25)

axr.plot(ttest.pvalue, c="black", linewidth=0.5)
axr.set_ylabel("p-value", fontsize=fsz)
axr.axhline(y=0.05, linewidth=1, c="gray", zorder=1)
axes.plot([], [], c="black", linewidth=1, label="p-value")

axes.legend(fontsize=fsz, loc=(0.155, 0.055), ncol=3)

axes.set_title(f"Global mean EIS for G1 and G2{tl_add}", fontsize=fsz)
axes.set_xlabel("Years since branching of abrupt-4xCO2", fontsize=fsz)
axes.set_ylabel("EIS in K", fontsize=fsz)
axes.tick_params(labelsize=fsz)

axr.tick_params(labelsize=fsz)

pl.savefig(pl_path + f"/PNG/EIS_Global_AnMean_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/EIS_Global_AnMean_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store these data as netcdf file
f = Dataset(out_path + f"G1_G2_Arctic_and_Gl_Mean_EIS{cslt_fadd}.nc", "w", format="NETCDF4")

# create the dimensions
f.createDimension("years", 150)
f.createDimension("mods_G1", len(mods_d["G1"]))
f.createDimension("mods_G2", len(mods_d["G2"]))

# create the variables
mods_g1_nc = f.createVariable("mods_G1", "S25", "mods_G1")
mods_g2_nc = f.createVariable("mods_G2", "S25", "mods_G2")
eis_g1_nc = f.createVariable("eis_G1", "f4", ("mods_G1", "years"))
eis_g2_nc = f.createVariable("eis_G2", "f4", ("mods_G2", "years"))
eis_ar_g1_nc = f.createVariable("eis_Ar_G1", "f4", ("mods_G1", "years"))
eis_ar_g2_nc = f.createVariable("eis_Ar_G2", "f4", ("mods_G2", "years"))

p_g1_g2_nc = f.createVariable("p_G1_G2", "f4", "years")
p_ar_g1_g2_nc = f.createVariable("p_Ar_G1_G2", "f4", "years")

# pass the data into the variables
mods_g1_nc[:] = np.array(mods_d["G1"])
mods_g2_nc[:] = np.array(mods_d["G2"])

eis_g1_nc[:] = eis_foa_gls["G1"]
eis_g2_nc[:] = eis_foa_gls["G2"]
eis_ar_g1_nc[:] = eis_foa_regs["G1"]
eis_ar_g2_nc[:] = eis_foa_regs["G2"]

p_g1_g2_nc[:] = ttest_ind(eis_foa_gls["G1"], eis_foa_gls["G2"], axis=0).pvalue
p_ar_g1_g2_nc[:] = ttest_ind(eis_foa_regs["G1"], eis_foa_regs["G2"], axis=0).pvalue

# set units of the variables
eis_g1_nc.units = "K"
eis_g2_nc.units = "K"

# descriptions of the variables
mods_g1_nc.description = "names of the model in G1 (weak lapse rate feedback change)"
mods_g2_nc.description = "names of the model in G2 (weak lapse rate feedback change)"

eis_g1_nc.description = "global mean EIS G1"
eis_g2_nc.description = "global mean EIS G2"
eis_ar_g1_nc.description = "Arctic mean EIS G1"
eis_ar_g2_nc.description = "Arctic mean EIS G2"

p_g1_g2_nc.description = "p-value for group mean difference Wald t-test between G1 and G2 for global mean EIS"
p_ar_g1_g2_nc.description = "p-value for group mean difference Wald t-test between G1 and G2 Arctic mean EIS"

# date of file creation
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the file
f.close()


