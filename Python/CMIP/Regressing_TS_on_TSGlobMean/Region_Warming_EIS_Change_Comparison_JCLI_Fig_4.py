"""
Generates Fig. 4 in Eiselt and Graversen (2022), JCLI. 

A bar plot comparing the relative surface warming and stability changes in four different regions.

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
# index = "TPWPI"  # "EPAA"  # "EQPNP"  #  "EPSO"  REPAA: Relative EP to Antarctica warming
acrs = ["IPWP", "WP", "EP", "AR"]  # "EQP"  # 
titl_ns = ["Indo-Pacific\nWarm Pool", "Western\nPacific", "Eastern\nPacific", "Arctic"]
# pl_pn = f"{index}_"


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


#%% choose the threshold year; if this is set to -1, the "year of maximum feedback change" will be used; if this is set 
#   to -2, the "year of maximum feedback change magnitude" (i.e., positive OR negative) will be used
thr_yr = 20


#%% set the threshold range
thr_min = 15
thr_max = 75


#%% choose the two regions
x1s = [50, 150, 260, 0]
x2s = [200, 170, 280, 360]
y1s = [-30, -15, -30, 75]
y2s = [30, 15, 0, 90]

#  x1,y2.....x2,y2  
#    .         .
#    .         .      East ->
#    .         .
#  x1,y1.....x2,y1


#%% choose the variable (tas or ts)
var = "ts"
var_s = "Ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


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

if excl == "outliers":
    
    excl_arr = np.array(["GISS-E2-R", "MIROC-ES2L", "GISS-E2.2-G"])

    excl_fadd = "_WoOutl"
    len_excl = len(excl_arr)  # important!     
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
os.makedirs(pl_path + "/PDF/Region_Warm_EIS_Comp/", exist_ok=True)
os.makedirs(pl_path + "/PNG/Region_Warm_EIS_Comp/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based{thr_fn}_" + 
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
eis_e_l = []
eis_l_l = []
fb_e_l = []
fb_l_l = []
fb_d_l = []
i_fb_e = []
i_fb_l = []
max_dfbs = []
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
        ind_fb_nc = Dataset(glob.glob(data_path5 + 
                                      f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}*.nc")[0])        
        reg_nc = Dataset(glob.glob(data_path5 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}.nc")[0])
        eis_nc = Dataset(glob.glob(data_path5 + "/EIS/" +
                                   f"/EIS_and_EIS_on_tas_LR/Thres_Year_{max_dtf_thr[mod_pl]}/*{mod_n}_a*.nc")[0])        
        # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
        
        # load the cloud feedback to be able to jump over models with large cloud feedback change
        cl_fb_d = ind_fb_nc.variables["C_fb_l"][0] - ind_fb_nc.variables["C_fb_e"][0]
        if (cl_fb_d > 0.5) & (rm_str_dcfb):
            print(f"\nJumping {mod_pl} because of large cloud dFb")
            continue
        # end if        

        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]
        
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # load the values - EIS on tas (or ts) regressions
        eis_e_l.append(eis_nc.variables["lr_e_rg"][:])
        eis_l_l.append(eis_nc.variables["lr_l_rg"][:])
        
        # get the lapse rate feedback to late separate G1 and G2
        i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])               
                
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
        cols.append("blue")
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
                                      f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])        
        reg_nc = Dataset(glob.glob(data_path6 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/*{mod}.nc")[0])
        eis_nc = Dataset(glob.glob(data_path6 + "/EIS/" +
                                   f"/EIS_and_EIS_on_tas_LR/Thres_Year_{max_dtf_thr[mod_pl]}/*{mod}_a*.nc")[0])        
        # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])

        # load the cloud feedback to be able to jump over models with large cloud feedback change
        cl_fb_d = ind_fb_nc.variables["C_fb_l"][0] - ind_fb_nc.variables["C_fb_e"][0]
        if (cl_fb_d > 0.5) & (rm_str_dcfb):
            print(f"\nJumping {mod_pl} because of large cloud dFb")
            continue
        # end if
        
        # get the model index in the max_dTF data
        mod_ind = np.where(mod_pl == mods_max_dfb)[0][0]        
    
        # load the values - tas or ts regressions
        lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
        lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
        lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
        
        # get the lapse rate feedback to late separate G1 and G2
        i_fb_e.append(ind_fb_nc.variables[f"{fb_s}_fb_e"][0])
        i_fb_l.append(ind_fb_nc.variables[f"{fb_s}_fb_l"][0])        
        
        # load the values - EIS on tas (or ts) regressions
        eis_e_l.append(eis_nc.variables["lr_e_rg"][:])
        eis_l_l.append(eis_nc.variables["lr_l_rg"][:])        
        
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


#%% get lat and lon
lat = reg_nc.variables["lat_rg"][:]
lon = reg_nc.variables["lon_rg"][:]


#%% convert the lists into numpy arrays
lr_e_a = np.array(lr_e_l)
lr_l_a = np.array(lr_l_l)
lr_d_a = np.array(lr_d_l)

i_fb_e = np.array(i_fb_e)
i_fb_l = np.array(i_fb_l)
i_fb_d = i_fb_l - i_fb_e

eis_e_a = np.array(eis_e_l)
eis_l_a = np.array(eis_l_l)
eis_d_a = eis_l_a - eis_e_a

fb_e_a = np.array(fb_e_l)
fb_l_a = np.array(fb_l_l)
fb_d_a = np.array(fb_d_l)


#%% print info about the CSLT and the loaded data
print(f"\n\n{len(pass_cslt)} models pass the CSLT and data for {len(lr_e_a)} models is now loaded...\n\n")


#%%
# test1, lon_eu = eu_centric(lr_e_a[0, :, :], lon)


#%% extract the regions
r_e_m = []
r_l_m = []
r_d_m = []
eisr_e_m = []
eisr_l_m = []
eisr_d_m = []
for ri in np.arange(len(x1s)):
    r_e = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, lr_e_a, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    r_l = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, lr_l_a, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    r_d = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, lr_d_a, test_plt=True, 
                         plot_title=f"{acrs[ri]} Region")
    eisr_e = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, eis_e_a, test_plt=True, 
                            plot_title=f"LTS on {acrs[ri]} Region")
    eisr_l = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, eis_l_a, test_plt=True, 
                            plot_title=f"LTS on {acrs[ri]} Region")
    eisr_d = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, eis_d_a, test_plt=True, 
                            plot_title=f"LTS on {acrs[ri]} Region")      
    
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
    eisr_e_m.append(np.average(eisr_e[0], weights=r_latw, axis=(1, 2)))
    eisr_l_m.append(np.average(eisr_l[0], weights=r_latw, axis=(1, 2)))
    eisr_d_m.append(np.average(eisr_d[0], weights=r_latw, axis=(1, 2)))     
# end for ri


#%% convert the list to arrays  
r_e_m = np.array(r_e_m)
r_l_m = np.array(r_l_m)
r_d_m = np.array(r_d_m)
eisr_e_m = np.array(eisr_e_m)
eisr_l_m = np.array(eisr_l_m)
eisr_d_m = np.array(eisr_d_m)


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
    tl_add = tl_add + "_HM"
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
    
low_regs = r_d_m[:, low_dfbi]
high_regs = r_d_m[:, high_dfbi]
low_eiss = eisr_d_m[:, low_dfbi]
high_eiss = eisr_d_m[:, high_dfbi]

low_regs_e = r_e_m[:, low_dfbi]
high_regs_e = r_e_m[:, high_dfbi]
low_eiss_e = eisr_e_m[:, low_dfbi]
high_eiss_e = eisr_e_m[:, high_dfbi]

low_regs_l = r_l_m[:, low_dfbi]
high_regs_l = r_l_m[:, high_dfbi]
low_eiss_l = eisr_l_m[:, low_dfbi]
high_eiss_l = eisr_l_m[:, high_dfbi]


#%% calculate the mean over the groups
low_regs_m = np.mean(low_regs, axis=1)
high_regs_m = np.mean(high_regs, axis=1)
low_eiss_m = np.mean(low_eiss, axis=1)
high_eiss_m = np.mean(high_eiss, axis=1)

low_regs_e_m = np.mean(low_regs_e, axis=1)
high_regs_e_m = np.mean(high_regs_e, axis=1)
low_eiss_e_m = np.mean(low_eiss_e, axis=1)
high_eiss_e_m = np.mean(high_eiss_e, axis=1)

low_regs_l_m = np.mean(low_regs_l, axis=1)
high_regs_l_m = np.mean(high_regs_l, axis=1)
low_eiss_l_m = np.mean(low_eiss_l, axis=1)
high_eiss_l_m = np.mean(high_eiss_l, axis=1)


#%% generate a bar plot showing the group means for both regions next to each other
bar_space = 0.75

bwidth = 0.25

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i in np.arange(len(low_regs_m)):
    ax1.bar(i * bar_space + 0, low_regs_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0.25, high_regs_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0, low_eiss_m[i], bottom=0, width=bwidth, color="gray")
    ax1.bar(i * bar_space + 0.25, high_eiss_m[i], bottom=0, width=bwidth, color="gray")
    
    text_low_y = np.max([low_regs_m[i], low_eiss_m[i]])
    text_high_y = np.max([high_regs_m[i], high_eiss_m[i]])
    if text_low_y < 0:
        text_low_y = 0
    if text_high_y < 0:
        text_high_y = 0
    # end if    
    ax1.text(i * bar_space + 0, text_low_y + 0.05, "G1", horizontalalignment="center", fontsize=15)
    ax1.text(i * bar_space + 0.25, text_high_y + 0.05, "G2", horizontalalignment="center", fontsize=15)
# end for i

ax1.bar(0, 0, bottom=0, width=bwidth, color="black", label="Surface warming change")
ax1.bar(0, 0, bottom=0, width=bwidth, color="gray", label="LTS change")

ax1.axhline(y=0, c="black")

ax1.legend(loc="lower left", fontsize=17)

ax1.tick_params(labelsize=16.5)

ax1.set_ylim((-2, 1.55))

ax1.set_ylabel("Change of surface warming and LTS in KK$^{{-1}}$", fontsize=15.5)

ax1.set_xticks([bwidth/2, bar_space+bwidth/2, 2*bar_space+bwidth/2, 3*bar_space+bwidth/2])
ax1.set_xticklabels(titl_ns)

ax1.set_title(f"Warming and LTS changes for G1 and G2{tl_add}", fontsize=19)

pl.savefig(pl_path + f"/PDF/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Change{pl_add}" +
           f"{cslt_fadd}{excl_fadd}.pdf", 
           bbox_inches="tight", 
           dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Change{pl_add}" + 
           f"{cslt_fadd}{excl_fadd}.png", 
           bbox_inches="tight", 
           dpi=250)

pl.show()
pl.close()


#%% generate a bar plot showing the group means for both regions next to each other for the early period
bar_space = 0.75

bwidth = 0.25

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i in np.arange(len(low_regs_m)):
    ax1.bar(i * bar_space + 0, low_regs_e_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0.25, high_regs_e_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0, low_eiss_e_m[i], bottom=0, width=bwidth, color="gray")
    ax1.bar(i * bar_space + 0.25, high_eiss_e_m[i], bottom=0, width=bwidth, color="gray")
    
    text_low_y = np.max([low_regs_e_m[i], low_eiss_e_m[i]])
    text_high_y = np.max([high_regs_e_m[i], high_eiss_e_m[i]])
    if text_low_y < 0:
        text_low_y = 0
    if text_high_y < 0:
        text_high_y = 0
    # end if    
    ax1.text(i * bar_space + 0, text_low_y + 0.05, "G1", horizontalalignment="center")
    ax1.text(i * bar_space + 0.25, text_high_y + 0.05, "G2", horizontalalignment="center")
# end for i

ax1.bar(0, 0, bottom=0, width=bwidth, color="black", label="Surface warming")
ax1.bar(0, 0, bottom=0, width=bwidth, color="gray", label="LTS")

ax1.axhline(y=0, c="black")

ax1.legend(loc="lower left", fontsize=13)

ax1.tick_params(labelsize=13)

# ax1.set_ylim((-2, 1.55))

ax1.set_ylabel("Early surface warming and LTS in KK$^{{-1}}$", fontsize=13)

ax1.set_xticks([bwidth/2, bar_space+bwidth/2, 2*bar_space+bwidth/2, 3*bar_space+bwidth/2])
ax1.set_xticklabels(titl_ns)

ax1.set_title(f"Early Warming and LTS for G1 and G2{tl_add}", fontsize=15)

pl.savefig(pl_path + f"/PDF/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Early{pl_add}" +
           f"{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Early{pl_add}" + 
           f"{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% generate a bar plot showing the group means for both regions next to each other for the late period
bar_space = 0.75

bwidth = 0.25

fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i in np.arange(len(low_regs_m)):
    ax1.bar(i * bar_space + 0, low_regs_l_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0.25, high_regs_l_m[i], bottom=0, width=bwidth, color="black")
    ax1.bar(i * bar_space + 0, low_eiss_l_m[i], bottom=0, width=bwidth, color="gray")
    ax1.bar(i * bar_space + 0.25, high_eiss_l_m[i], bottom=0, width=bwidth, color="gray")
    
    text_low_y = np.max([low_regs_l_m[i], low_eiss_l_m[i]])
    text_high_y = np.max([high_regs_l_m[i], high_eiss_l_m[i]])
    if text_low_y < 0:
        text_low_y = 0
    if text_high_y < 0:
        text_high_y = 0
    # end if    
    ax1.text(i * bar_space + 0, text_low_y + 0.05, "G1", horizontalalignment="center")
    ax1.text(i * bar_space + 0.25, text_high_y + 0.05, "G2", horizontalalignment="center")
# end for i

ax1.bar(0, 0, bottom=0, width=bwidth, color="black", label="Surface warming")
ax1.bar(0, 0, bottom=0, width=bwidth, color="gray", label="LTS")

ax1.axhline(y=0, c="black")

ax1.legend(loc="lower left", fontsize=13)

ax1.tick_params(labelsize=13)

# ax1.set_ylim((-2, 1.55))

ax1.set_ylabel("Late surface warming and LTS in KK$^{{-1}}$", fontsize=13)

ax1.set_xticks([bwidth/2, bar_space+bwidth/2, 2*bar_space+bwidth/2, 3*bar_space+bwidth/2])
ax1.set_xticklabels(titl_ns)

ax1.set_title(f"Late Warming and LTS for G1 and G2{tl_add}", fontsize=15)

pl.savefig(pl_path + f"/PDF/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Late{pl_add}" + 
           f"{cslt_fadd}{excl_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Region_Warm_EIS_Comp/Comparison_{acrs[0]}_{acrs[1]}_{var}_Warming_Late{pl_add}" +
           f"{cslt_fadd}{excl_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()
