"""
Generates Fig. S4 in Eiselt and Graversen (2022), JCLI.

Compare model groups G1 and G2 (for details see our paper) Hadley cell.

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
import progressbar as pg
import dask.array as da
import time as ti
import xarray as xr


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
# excl_l = ["not high LR dFb"]
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
data_path5 = ""
data_path6 = ""

# clear sky linearity test path
cslt_path = f"/Model_Lists/ClearSkyLinearityTest/{var}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{var}_Based{thr_fn}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% loop over the groups and collect their data to compare later

# set up the dictionaries to store the Hadley strengths
had_n_d = dict()
had_s_d = dict()

thad_n_d = dict()
thad_s_d = dict()

dhad_n_d = dict()
dhad_s_d = dict()

had_pi_n_d = dict()

dtas_d = dict()

# set up model dictionary
mods_d = dict()

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
    
    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    had_n_l = []
    had_s_l = []
    thad_n_l = []
    thad_s_l = []
    dhad_n_l = []
    dhad_s_l = []
    had_pi_n_l = []
    
    dtas_l = []
    
    
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
            # reg_nc = Dataset(glob.glob(data_path5 + f"/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + "*.nc")[0])        
            mpsi_nc = Dataset(glob.glob(data_path5 + "/Data/" + mod + "/abrupt4xCO2_mpsi_Files/*.nc")[0])
            mpsi_pi_nc = Dataset(glob.glob(data_path5 + "/Data/" + mod + "/piControl_mpsi_Files/*.nc")[0])
            dtas_nc = Dataset(glob.glob(data_path5 + "/Data/" + mod + "/dtas_Amon*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])

            # load cloud feedback files
            i_fb_nc = Dataset(glob.glob(data_path5 + "/Outputs/" +
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
            
            # get the streamfunction values
            mpsi_mon = np.array(xr.open_mfdataset(glob.glob(data_path5 + "/Data/" + mod + 
                                                            "/abrupt4xCO2_mpsi_Files*/*.nc")).mpsi)
            mpsi_mon = mpsi_mon[:150*12, :, :]

            mpsi_pi_mon = np.array(xr.open_mfdataset(glob.glob(data_path5 + "/Data/" + mod + 
                                                               "/mpsi_piControl_21YrRunMean_Files*/*.nc")).mpsi)
            
            # get the grid info for the regridded dtas data
            lat_rg = dtas_nc.variables["lat_rg"][:]
            lon_rg = dtas_nc.variables["lon_rg"][:]
            
            # load dtas and calculate the annual global mean
            dtas_ag = an_mean(glob_mean(dtas_nc.variables["tas_ch_rg"], lat_rg, lon_rg))
            dtas_l.append(dtas_ag)

            # aggregate to annual means
            mpsi = an_mean(mpsi_mon)
            mpsi_pi = np.mean(mpsi_pi_mon, axis=0)
            
            lat =  mpsi_nc.variables["lat"][:]
            
            # extract the streamfunction for the latitudes between 15 and 25 N and S
            hadley_n = mpsi[:, :, (lat > 15) & (lat < 25)]
            hadley_s = mpsi[:, :,  (lat > -25) & (lat < -15)]
            hadley_pi_n = mpsi_pi[:, :, (lat > 15) & (lat < 25)]
            hadley_pi_s = mpsi_pi[:, :,  (lat > -25) & (lat < -15)]

            # get the time series via extracting the maximum streamfunction from the Hadley values
            had_n_tser = np.max(hadley_n, axis=(1, 2))
            had_s_tser = np.min(hadley_s, axis=(1, 2))
            had_pi_n_tser = np.max(hadley_pi_n, axis=(1, 2))
            had_pi_s_tser = np.min(hadley_pi_s, axis=(1, 2))

            # perform linear regression on the piControl
            # tser = np.arange(150)
            # sl_n, yi_n = lr(tser, had_pi_n_tser)[:2]
            # sl_s, yi_s = lr(tser, had_pi_s_tser)[:2]

            # append to the Hadley list
            thad_n_l.append(had_n_tser)
            thad_s_l.append(had_s_tser)            
            had_n_l.append((had_n_tser - had_pi_n_tser) / np.mean(had_pi_n_tser))
            had_s_l.append((had_s_tser - had_pi_s_tser) / np.mean(had_pi_s_tser))
            dhad_n_l.append(had_n_tser - had_pi_n_tser)
            dhad_s_l.append(had_s_tser - had_pi_s_tser)            
            had_pi_n_l.append(had_pi_n_tser)
            
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
            # reg_nc = Dataset(glob.glob(data_path6 + f"/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + "*.nc")[0])
            mpsi_nc = Dataset(glob.glob(data_path6 + "/Data/" + mod + "/abrupt-4xCO2_mpsi_Files*/*.nc")[0])
            mpsi_pi_nc = Dataset(glob.glob(data_path6 + "/Data/" + mod + "/piControl_mpsi_Files*/*.nc")[0])
            dtas_nc = Dataset(glob.glob(data_path6 + "/Data/" + mod + "/dtas_Amon*.nc")[0])
            # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path6 + "/Outputs/"
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
            
            # get the streamfunction values
            mpsi_mon = np.array(xr.open_mfdataset(glob.glob(data_path6 + "/Data/" + mod + 
                                                            "/abrupt-4xCO2_mpsi_Files*/*.nc")).mpsi)
            mpsi_mon = mpsi_mon[:150*12, :, :]

            # mpsi_pi_mon = np.array(xr.open_mfdataset(glob.glob(data_path6 + mod + "/piControl_mpsi_Files*/*.nc")).mpsi)
            # mpsi_pi_mon = mpsi_pi_mon[nl6.b_times[mod]*12:nl6.b_times[mod]*12+150*12, :, :]
            mpsi_pi_mon = np.array(xr.open_mfdataset(glob.glob(data_path6 + "/Data/" + mod + 
                                                               "/mpsi_piControl_21YrRunMean_Files*/*.nc")).mpsi)
            # mpsi_pi_mon = mpsi_pi_mon[:]

            # get the grid info for the regridded dtas data
            lat_rg = dtas_nc.variables["lat_rg"][:]
            lon_rg = dtas_nc.variables["lon_rg"][:]
            
            # load dtas and calculate the annual global mean
            dtas_ag = an_mean(glob_mean(dtas_nc.variables["tas_ch_rg"], lat_rg, lon_rg))
            dtas_l.append(dtas_ag)
            
            # aggregate to annual means
            mpsi = an_mean(mpsi_mon)
            mpsi_pi = np.mean(mpsi_pi_mon, axis=0)
            
            lat =  mpsi_nc.variables["lat"][:]
            
            # extract the streamfunction for the latitudes between 15 and 25 N and S
            hadley_n = mpsi[:, :, (lat > 15) & (lat < 25)]
            hadley_s = mpsi[:, :,  (lat > -25) & (lat < -15)]
            hadley_pi_n = mpsi_pi[:, :, (lat > 15) & (lat < 25)]
            hadley_pi_s = mpsi_pi[:, :,  (lat > -25) & (lat < -15)]

            # get the time series via extracting the maximum streamfunction from the Hadley values
            had_n_tser = np.max(hadley_n, axis=(1, 2))  # --> northward transport
            had_s_tser = np.min(hadley_s, axis=(1, 2))  # --> northward transport
            had_pi_n_tser = np.max(hadley_pi_n, axis=(1, 2))  # --> northward transport
            had_pi_s_tser = np.min(hadley_pi_s, axis=(1, 2))  # --> northward transport

            # perform linear regression on the piControl
            tser = np.arange(150)
            sl_n, yi_n = lr(tser, had_pi_n_tser)[:2]
            sl_s, yi_s = lr(tser, had_pi_s_tser)[:2]
            

            # append to the Hadley list
            thad_n_l.append(had_n_tser)
            thad_s_l.append(had_s_tser)
            had_n_l.append((had_n_tser - had_pi_n_tser))
            had_s_l.append((had_s_tser - had_pi_s_tser))
            dhad_n_l.append(had_n_tser - had_pi_n_tser)
            dhad_s_l.append(had_s_tser - had_pi_s_tser)
            had_pi_n_l.append(had_pi_n_tser)
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod
    
    print(mods)
    
    thad_n_d[g12] = np.array(thad_n_l)
    thad_s_d[g12] = np.array(thad_s_l)    
    had_n_d[g12] = np.array(had_n_l)
    had_s_d[g12] = np.array(had_s_l)
    dhad_n_d[g12] = np.array(dhad_n_l)
    dhad_s_d[g12] = np.array(dhad_s_l)
    had_pi_n_d[g12] = np.array(had_pi_n_l)
    dtas_d[g12] = np.array(dtas_l)
    mods_d[g12] = np.array(mods)

# end for excl


#%% calculate the means over G1 and over G2
scal = 1E-2

# Hadley cell strength data - relative change
g1_had_nh = np.mean(had_n_d["G1"]/scal, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_had_nh = np.mean(had_n_d["G2"]/scal, axis=0)  # remember, this is "exclude not high LR dFb" => G2
g1_had_nh_std = np.std(had_n_d["G1"]/scal, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_had_nh_std = np.std(had_n_d["G2"]/scal, axis=0)  # remember, this is "exclude not high LR dFb" => G2

g1_had_sh = np.mean(had_s_d["G1"]/scal, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_had_sh = np.mean(had_s_d["G2"]/scal, axis=0)  # remember, this is "exclude not high LR dFb" => G2
g1_had_sh_std = np.std(had_s_d["G1"]/scal, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_had_sh_std = np.std(had_s_d["G2"]/scal, axis=0)  # remember, this is "exclude not high LR dFb" => G2

# Hadley cell strength data - absolute change
g1_dhad_nh = np.mean(dhad_n_d["G1"] / 1e10, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_dhad_nh = np.mean(dhad_n_d["G2"] / 1e10, axis=0)  # remember, this is "exclude not high LR dFb" => G2
g1_dhad_sh = np.mean(dhad_s_d["G1"] / 1e10, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_dhad_sh = np.mean(dhad_s_d["G2"] / 1e10, axis=0)  # remember, this is "exclude not high LR dFb" => G2

g1_std_dhad_nh = np.std(dhad_n_d["G1"] / 1e10, axis=0)
g2_std_dhad_nh = np.std(dhad_n_d["G2"] / 1e10, axis=0)
g1_std_dhad_sh = np.std(dhad_s_d["G1"] / 1e10, axis=0)
g2_std_dhad_sh = np.std(dhad_s_d["G2"] / 1e10, axis=0)

g1_thad_nh = np.mean(thad_n_d["G1"] / 1e10, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_thad_nh = np.mean(thad_n_d["G2"] / 1e10, axis=0)  # remember, this is "exclude not high LR dFb" => G2
g1_thad_sh = np.mean(thad_s_d["G1"] / 1e10, axis=0)  # remember, this is "exclude not low LR dFb" => G1
g2_thad_sh = np.mean(thad_s_d["G2"] / 1e10, axis=0)  # remember, this is "exclude not high LR dFb" => G2

g1_std_thad_nh = np.std(thad_n_d["G1"] / 1e10, axis=0)
g2_std_thad_nh = np.std(thad_n_d["G2"] / 1e10, axis=0)
g1_std_thad_sh = np.std(thad_s_d["G1"] / 1e10, axis=0)
g2_std_thad_sh = np.std(thad_s_d["G2"] / 1e10, axis=0)


# dtas data
g1_dtas = np.mean(dtas_d["G1"], axis=0)
g2_dtas = np.mean(dtas_d["G2"], axis=0)


#%% calculate the values for the model for which we have AMOC
g1_amoc = np.array(['CanESM5', 'CNRM-CM6.1', 'E3SM-1.0', 'MPI-ESM1.2-LR', 'UKESM1.0-LL', 'CMCC-ESM2'])
g2_amoc = np.array(['GFDL-CM3', 'NorESM1-M', 'ACCESS-ESM1.5', 'EC-Earth3', 'FGOALS-g3', 'INM-CM4.8'])

# loop over the models and exclude the ones that are not on the AMOC lists
dhad_n_d["G1_AMOC"] = []
dhad_n_d["G2_AMOC"] = []
dhad_s_d["G1_AMOC"] = []
dhad_s_d["G2_AMOC"] = []
thad_n_d["G1_AMOC"] = []
thad_n_d["G2_AMOC"] = []
thad_s_d["G1_AMOC"] = []
thad_s_d["G2_AMOC"] = []

for mod_amoc in g1_amoc:
    mod_ind = np.where(mod_amoc == mods_d["G1"])[0][0]
    dhad_n_d["G1_AMOC"].append(dhad_n_d["G1"][mod_ind])
    dhad_s_d["G1_AMOC"].append(dhad_s_d["G1"][mod_ind])
    thad_n_d["G1_AMOC"].append(thad_n_d["G1"][mod_ind])
    thad_s_d["G1_AMOC"].append(thad_s_d["G1"][mod_ind])
# end for mod_amoc
for mod_amoc in g2_amoc:
    mod_ind = np.where(mod_amoc == mods_d["G2"])[0][0]
    dhad_n_d["G2_AMOC"].append(dhad_n_d["G2"][mod_ind])
    dhad_s_d["G2_AMOC"].append(dhad_s_d["G2"][mod_ind])
    thad_n_d["G2_AMOC"].append(thad_n_d["G2"][mod_ind])
    thad_s_d["G2_AMOC"].append(thad_s_d["G2"][mod_ind])    
# end for mod_amoc

# convert the data lists to array
dhad_n_d["G1_AMOC"] = np.array(dhad_n_d["G1_AMOC"])
dhad_s_d["G1_AMOC"] = np.array(dhad_s_d["G1_AMOC"])
dhad_n_d["G2_AMOC"] = np.array(dhad_n_d["G2_AMOC"])
dhad_s_d["G2_AMOC"] = np.array(dhad_s_d["G2_AMOC"])
thad_n_d["G1_AMOC"] = np.array(thad_n_d["G1_AMOC"])
thad_s_d["G1_AMOC"] = np.array(thad_s_d["G1_AMOC"])
thad_n_d["G2_AMOC"] = np.array(thad_n_d["G2_AMOC"])
thad_s_d["G2_AMOC"] = np.array(thad_s_d["G2_AMOC"])

# calculate the G1 and G2 means
g1_dhad_nh_amoc = np.mean(dhad_n_d["G1_AMOC"] / 1e10, axis=0)
g1_dhad_sh_amoc = np.mean(dhad_s_d["G1_AMOC"] / 1e10, axis=0)
g2_dhad_nh_amoc = np.mean(dhad_n_d["G2_AMOC"] / 1e10, axis=0)
g2_dhad_sh_amoc = np.mean(dhad_s_d["G2_AMOC"] / 1e10, axis=0)
g1_thad_nh_amoc = np.mean(thad_n_d["G1_AMOC"] / 1e10, axis=0)
g1_thad_sh_amoc = np.mean(thad_s_d["G1_AMOC"] / 1e10, axis=0)
g2_thad_nh_amoc = np.mean(thad_n_d["G2_AMOC"] / 1e10, axis=0)
g2_thad_sh_amoc = np.mean(thad_s_d["G2_AMOC"] / 1e10, axis=0)

g1_std_dhad_nh_amoc = np.std(dhad_n_d["G1_AMOC"] / 1e10, axis=0)
g1_std_dhad_sh_amoc = np.std(dhad_s_d["G1_AMOC"] / 1e10, axis=0)
g2_std_dhad_nh_amoc = np.std(dhad_n_d["G2_AMOC"] / 1e10, axis=0)
g2_std_dhad_sh_amoc = np.std(dhad_s_d["G2_AMOC"] / 1e10, axis=0)
g1_std_thad_nh_amoc = np.std(thad_n_d["G1_AMOC"] / 1e10, axis=0)
g1_std_thad_sh_amoc = np.std(thad_s_d["G1_AMOC"] / 1e10, axis=0)
g2_std_thad_nh_amoc = np.std(thad_n_d["G2_AMOC"] / 1e10, axis=0)
g2_std_thad_sh_amoc = np.std(thad_s_d["G2_AMOC"] / 1e10, axis=0)


#%% liner regressions
sle_g1, yie_g1, r, p, sle_g1_u = lr(np.arange(20), g1_had_nh[:20])
sll_g1, yil_g1, r, p, sll_g1_u = lr(np.arange(20, 150), g1_had_nh[20:150])
sle_g2, yie_g2, r, p, sle_g2_u = lr(np.arange(20), g2_had_nh[:20])
sll_g2, yil_g2, r, p, sll_g2_u = lr(np.arange(20, 150), g2_had_nh[20:150])

sle_sg1, yie_sg1, r, p, sle_sg1_u = lr(np.arange(20), g1_had_sh[:20])
sll_sg1, yil_sg1, r, p, sll_sg1_u = lr(np.arange(20, 150), g1_had_sh[20:150])
sle_sg2, yie_sg2, r, p, sle_sg2_u = lr(np.arange(20), g2_had_sh[:20])
sll_sg2, yil_sg2, r, p, sll_sg2_u = lr(np.arange(20, 150), g2_had_sh[20:150])


#%% plot the values in one plot - northern hemisphere
fsz = 15

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_had_nh, c="blue", label="G1")
p2, = axes.plot(g2_had_nh, c="red", label="G2")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_had_nh)), g1_had_nh - g1_had_nh_std, g1_had_nh + g1_had_nh_std, 
                  facecolor="blue", alpha=0.15)
axes.fill_between(np.arange(len(g2_had_nh)), g2_had_nh - g2_had_nh_std, g2_had_nh + g2_had_nh_std, 
                  facecolor="red", alpha=0.15)

p3, = axes.plot(np.arange(20) * sle_g1 + yie_g1, c="black", 
                label=f"G1 early slope={np.round(sle_g1, 2)}$\pm${np.round(sle_g1_u, 2)}")
p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g1 + yil_g1, c="black", linestyle="--",
                label=f"G1 late slope={np.round(sll_g1, 2)}$\pm${np.round(sll_g1_u, 2)}")
p5, = axes.plot(np.arange(20) * sle_g2 + yie_g2, c="gray", 
                label=f"G2 early slope={np.round(sle_g2, 2)}$\pm${np.round(sle_g2_u, 2)}")
p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g2 + yil_g2, c="gray", linestyle="--",
                label=f"G2 late slope={np.round(sll_g2, 2)}$\pm${np.round(sll_g2_u, 2)}")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-0.5, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-0.5, loc="lower right")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Northward Hadley mass flux\nchange in % of piControl mean", fontsize=fsz+1)

axes.set_title(f"Northern hemispheric Hadley circulation for G1 and G2{tl_add}", fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.set_ylim((52, 83))

# pl.savefig(pl_path + f"/PDF/Hadley_delta_piC21Yr_RunMean_NH_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/Hadley_delta_piC21Yr_RunMean_NH_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)


pl.show()
pl.close()


#%% plot the values in one plot - southern hemisphere
fsz = 15

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_had_sh, c="blue", label="G1")
p2, = axes.plot(g2_had_sh, c="red", label="G2")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_had_sh)), g1_had_sh - g1_had_sh_std, g1_had_sh + g1_had_sh_std, 
                  facecolor="blue", alpha=0.15)
axes.fill_between(np.arange(len(g2_had_sh)), g2_had_sh - g2_had_sh_std, g2_had_sh + g2_had_sh_std, 
                  facecolor="red", alpha=0.15)

p3, = axes.plot(np.arange(20) * sle_sg1 + yie_sg1, c="black", 
                label=f"G1 early slope={np.round(sle_sg1, 2)}$\pm${np.round(sle_sg1_u, 2)}")
p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_sg1 + yil_sg1, c="black", linestyle="--",
                label=f"G1 late slope={np.round(sll_sg1, 2)}$\pm${np.round(sll_sg1_u, 2)}")
p5, = axes.plot(np.arange(20) * sle_sg2 + yie_sg2, c="gray", 
                label=f"G2 early slope={np.round(sle_sg2, 2)}$\pm${np.round(sle_sg2_u, 2)}")
p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_sg2 + yil_sg2, c="gray", linestyle="--",
                label=f"G2 late slope={np.round(sll_sg2, 2)}$\pm${np.round(sll_sg2_u, 2)}")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-0.5, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-0.5, loc="lower right")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Southward Hadley mass flux\nchange in % of piControl mean", fontsize=fsz+1)

axes.set_title(f"Southern hemispheric Hadley circulation for G1 and G2{tl_add}", fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.invert_yaxis()

# axes.set_ylim((52, 83))

pl.savefig(pl_path + f"/PDF/Hadley_delta_piC21Yr_RunMean_SH_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Hadley_delta_piC21Yr_RunMean_SH_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot dtas v. NH Hadley cell strength for G1 and G2

# linear regression
elp = 20
sl_e_g1, yi_e_g1, r_e_g1, p_e_g1 = lr(g1_dtas[:elp], g1_dhad_nh[:elp])[:4]
sl_l_g1, yi_l_g1, r_l_g1, p_l_g1 = lr(g1_dtas[elp:], g1_dhad_nh[elp:])[:4]
sl_e_g2, yi_e_g2, r_e_g2, p_e_g2 = lr(g2_dtas[:elp], g2_dhad_nh[:elp])[:4]
sl_l_g2, yi_l_g2, r_l_g2, p_l_g2 = lr(g2_dtas[elp:], g2_dhad_nh[elp:])[:4]

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

axes.scatter(g1_dtas[:elp], g1_dhad_nh[:elp], c="white", edgecolor="blue", marker="o", 
             label=f"G1 early slope={np.round(sl_e_g1, 2)} $\\times$ 10$^{{10}}$ kg K$^{{-1}}$ s$^{{-1}}$")
axes.scatter(g2_dtas[:elp], g2_dhad_nh[:elp], c="white", edgecolor="red", marker="o",
             label=f"G2 early slope={np.round(sl_e_g2, 2)} $\\times$ 10$^{{10}}$ kg K$^{{-1}}$ s$^{{-1}}$")
axes.scatter(g1_dtas[elp:], g1_dhad_nh[elp:], c="blue", marker="o",
             label=f"G1 late slope={np.round(sl_l_g1, 2)} $\\times$ 10$^{{10}}$ kg K$^{{-1}}$ s$^{{-1}}$")
axes.scatter(g2_dtas[elp:], g2_dhad_nh[elp:], c="red", marker="o",
             label=f"G2 late slope={np.round(sl_l_g2, 2)} $\\times$ 10$^{{10}}$ kg K$^{{-1}}$ s$^{{-1}}$")

axes.legend()

axes.plot(g1_dtas[:elp], g1_dtas[:elp] * sl_e_g1 + yi_e_g1, c="black", linestyle="--")
axes.plot(g1_dtas[elp:], g1_dtas[elp:] * sl_l_g1 + yi_l_g1, c="black", linestyle="-")
axes.plot(g2_dtas[:elp], g2_dtas[:elp] * sl_e_g2 + yi_e_g2, c="gray", linestyle="--")
axes.plot(g2_dtas[elp:], g2_dtas[elp:] * sl_l_g2 + yi_l_g2, c="gray", linestyle="-")

axes.set_xlabel("SAT change in K")
axes.set_ylabel("NH Hadley cell strength change in 10$^10$ kg s$^{-1}$")

axes.set_title("NH Hadley strength v. global mean SAT change $-$ G1 and G2")

pl.savefig(pl_path + f"/PDF/Hadley_v_dtas_NH_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Hadley_v_dtas_NH_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot dtas v. SH Hadley cell strength for G1 and G2

# linear regression
elp = 20
sl_e_g1, yi_e_g1, r_e_g1, p_e_g1 = lr(g1_dtas[:elp], g1_dhad_sh[:elp])[:4]
sl_l_g1, yi_l_g1, r_l_g1, p_l_g1 = lr(g1_dtas[elp:], g1_dhad_sh[elp:])[:4]
sl_e_g2, yi_e_g2, r_e_g2, p_e_g2 = lr(g2_dtas[:elp], g2_dhad_sh[:elp])[:4]
sl_l_g2, yi_l_g2, r_l_g2, p_l_g2 = lr(g2_dtas[elp:], g2_dhad_sh[elp:])[:4]

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

axes.scatter(g1_dtas, g1_dhad_sh, c="blue", label="G1", marker="o")
axes.scatter(g2_dtas, g2_dhad_sh, c="red", label="G2", marker="o")

axes.legend()

axes.plot(g1_dtas[:elp], g1_dtas[:elp] * sl_e_g1 + yi_e_g1, c="black", linestyle="--")
axes.plot(g1_dtas[elp:], g1_dtas[elp:] * sl_l_g1 + yi_l_g1, c="black", linestyle="-")
axes.plot(g2_dtas[:elp], g2_dtas[:elp] * sl_e_g2 + yi_e_g2, c="gray", linestyle="--")
axes.plot(g2_dtas[elp:], g2_dtas[elp:] * sl_l_g2 + yi_l_g2, c="gray", linestyle="-")

axes.set_xlabel("SAT change in K")
axes.set_ylabel("SH Hadley cell strength change in 10$^10$ kg s$^{-1}$")

axes.set_title("SH Hadley strength v. global mean SAT change $-$ G1 and G2")

pl.savefig(pl_path + f"/PDF/Hadley_v_dtas_SH_G1_G2{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Hadley_v_dtas_SH_G1_G2{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% G1 and G2 Hadley cell strength change for models with AMOC - northern hemisphere
elp = 20
sle_g1, yie_g1, r, p, sle_g1_u = lr(np.arange(elp), g1_dhad_nh_amoc[:elp])
sll_g1, yil_g1, r, p, sll_g1_u = lr(np.arange(elp, 150), g1_dhad_nh_amoc[elp:150])
sle_g2, yie_g2, r, p, sle_g2_u = lr(np.arange(elp), g2_dhad_nh_amoc[:elp])
sll_g2, yil_g2, r, p, sll_g2_u = lr(np.arange(elp, 150), g2_dhad_nh_amoc[elp:150])

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_dhad_nh_amoc, c="blue", label="G1")
p2, = axes.plot(g2_dhad_nh_amoc, c="red", label="G2")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_dhad_nh_amoc)), g1_dhad_nh_amoc - g1_std_dhad_nh_amoc, g1_dhad_nh_amoc + 
                  g1_std_dhad_nh_amoc, facecolor="blue", alpha=0.15)
axes.fill_between(np.arange(len(g2_dhad_nh_amoc)), g2_dhad_nh_amoc - g2_std_dhad_nh_amoc, g2_dhad_nh_amoc + 
                  g2_std_dhad_nh_amoc, facecolor="red", alpha=0.15)

p3, = axes.plot(np.arange(20) * sle_g1 + yie_g1, c="black", linewidth=3,
                label=f"G1 sl$_e$={np.round(sle_g1*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g1 + yil_g1, c="black", linestyle="--", linewidth=3,
                label=f"G1 sl$_l$={np.round(sll_g1*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p5, = axes.plot(np.arange(20) * sle_g2 + yie_g2, c="gray",  linewidth=3,
                label=f"G2 sl$_e$={np.round(sle_g2*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g2 + yil_g2, c="gray", linestyle="--",  linewidth=3,
                label=f"G2 sl$_l$={np.round(sll_g2*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-2, ncol=2, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-2, loc="lower center")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Northward Hadley cell change in 10$^{10}$ kg s$^{-1}$", fontsize=fsz+1)

# axes.set_title("Northern hemispheric Hadley circulation for G1 and G2\n only for members with AMOC data available", 
#                fontsize=fsz+1)
axes.set_title("Northern hemispheric Hadley circulation G1$-$G2 comparison", 
               fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.set_ylim((52, 83))

pl.savefig(pl_path + f"/PDF/Hadley_delta_NH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Hadley_delta_NH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% G1 and G2 Hadley cell strength change for models with AMOC - southern hemisphere
elp = 20
sle_g1, yie_g1, r, p, sle_g1_u = lr(np.arange(elp), g1_dhad_sh_amoc[:elp])
sll_g1, yil_g1, r, p, sll_g1_u = lr(np.arange(elp, 150), g1_dhad_sh_amoc[elp:150])
sle_g2, yie_g2, r, p, sle_g2_u = lr(np.arange(elp), g2_dhad_sh_amoc[:elp])
sll_g2, yil_g2, r, p, sll_g2_u = lr(np.arange(elp, 150), g2_dhad_sh_amoc[elp:150])

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_dhad_sh_amoc, c="blue", label="G1")
p2, = axes.plot(g2_dhad_sh_amoc, c="red", label="G2")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_dhad_sh_amoc)), g1_dhad_sh_amoc - g1_std_dhad_sh_amoc, g1_dhad_sh_amoc + 
                  g1_std_dhad_sh_amoc, facecolor="blue", alpha=0.15)
axes.fill_between(np.arange(len(g2_dhad_sh_amoc)), g2_dhad_sh_amoc - g2_std_dhad_sh_amoc, g2_dhad_sh_amoc + 
                  g2_std_dhad_sh_amoc, facecolor="red", alpha=0.15)

p3, = axes.plot(np.arange(20) * sle_g1 + yie_g1, c="black", 
                label=f"G1 early slope={np.round(sle_g1, 2)}$\pm${np.round(sle_g1_u, 2)}")
p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g1 + yil_g1, c="black", linestyle="--",
                label=f"G1 late slope={np.round(sll_g1, 2)}$\pm${np.round(sll_g1_u, 2)}")
p5, = axes.plot(np.arange(20) * sle_g2 + yie_g2, c="gray", 
                label=f"G2 early slope={np.round(sle_g2, 2)}$\pm${np.round(sle_g2_u, 2)}")
p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g2 + yil_g2, c="gray", linestyle="--",
                label=f"G2 late slope={np.round(sll_g2, 2)}$\pm${np.round(sll_g2_u, 2)}")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-0.5, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-0.5, loc="lower right")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Northward Hadley cell change in 10$^{10}$ kg s$^{-1}$", fontsize=fsz+1)

axes.set_title("Southern hemispheric Hadley circulation for G1 and G2\n only for members with AMOC data available", 
               fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.set_ylim((52, 83))

pl.savefig(pl_path + f"/PDF/Hadley_delta_SH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/Hadley_delta_SH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% G1 and G2 Hadley cell strength change for models - southern hemisphere (all G1 and G2 members)
elp = 20
sle_g1, yie_g1, r, p, sle_g1_u = lr(np.arange(elp), g1_dhad_sh[:elp])
sll_g1, yil_g1, r, p, sll_g1_u = lr(np.arange(elp, 150), g1_dhad_sh[elp:150])
sle_g2, yie_g2, r, p, sle_g2_u = lr(np.arange(elp), g2_dhad_sh[:elp])
sll_g2, yil_g2, r, p, sll_g2_u = lr(np.arange(elp, 150), g2_dhad_sh[elp:150])

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_dhad_sh, c="blue", label="G1")
p2, = axes.plot(g2_dhad_sh, c="red", label="G2")
axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_dhad_sh)), g1_dhad_sh - g1_std_dhad_sh, g1_dhad_sh + 
                  g1_std_dhad_sh, facecolor="blue", alpha=0.15)
axes.fill_between(np.arange(len(g2_dhad_sh)), g2_dhad_sh - g2_std_dhad_sh, g2_dhad_sh + 
                  g2_std_dhad_sh, facecolor="red", alpha=0.15)

p3, = axes.plot(np.arange(20) * sle_g1 + yie_g1, c="black", 
                label=f"G1 early slope={np.round(sle_g1, 2)}$\pm${np.round(sle_g1_u, 2)}")
p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g1 + yil_g1, c="black", linestyle="--",
                label=f"G1 late slope={np.round(sll_g1, 2)}$\pm${np.round(sll_g1_u, 2)}")
p5, = axes.plot(np.arange(20) * sle_g2 + yie_g2, c="gray", 
                label=f"G2 early slope={np.round(sle_g2, 2)}$\pm${np.round(sle_g2_u, 2)}")
p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g2 + yil_g2, c="gray", linestyle="--",
                label=f"G2 late slope={np.round(sll_g2, 2)}$\pm${np.round(sll_g2_u, 2)}")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-0.5, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-0.5, loc="lower right")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Northward Hadley cell change in 10$^{10}$ kg s$^{-1}$", fontsize=fsz+1)

axes.set_title("Southern hemispheric Hadley circulation for G1 and G2", 
               fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.set_ylim((52, 83))

pl.show()
pl.close()


#%% G1 and G2 Hadley cell strength change for models with AMOC - northern hemisphere - Patter effect workshop
elpg1 = 15
elpg2 = 15
sle_g1, yie_g1, r, p, sle_g1_u = lr(np.arange(elpg1), g1_thad_nh_amoc[:elpg1])
sll_g1, yil_g1, r, p, sll_g1_u = lr(np.arange(elpg1, 150), g1_thad_nh_amoc[elpg1:150])
sle_g2, yie_g2, r, p, sle_g2_u = lr(np.arange(elpg2), g2_thad_nh_amoc[:elpg2])
sll_g2, yil_g2, r, p, sll_g2_u = lr(np.arange(elpg2, 150), g2_thad_nh_amoc[elpg2:150])

slll_g1, yill_g1 = lr(np.arange(30, len(g1_thad_nh_amoc)), g1_thad_nh_amoc[30:])[:2]
slll_g2, yill_g2 = lr(np.arange(30, len(g2_thad_nh_amoc)), g2_thad_nh_amoc[30:])[:2]

fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

p1, = axes.plot(g1_thad_nh_amoc, c="blue", label="G1")
p2, = axes.plot(g2_thad_nh_amoc, c="red", label="G2")
# axes.axhline(y=0, c="gray", linewidth=0.5)

axes.fill_between(np.arange(len(g1_thad_nh_amoc)), g1_thad_nh_amoc - g1_std_thad_nh_amoc, g1_thad_nh_amoc + 
                  g1_std_thad_nh_amoc, facecolor="blue", alpha=0.25)
axes.fill_between(np.arange(len(g2_thad_nh_amoc)), g2_thad_nh_amoc - g2_std_thad_nh_amoc, g2_thad_nh_amoc + 
                  g2_std_thad_nh_amoc, facecolor="red", alpha=0.25)

p3, = axes.plot(np.arange(elpg1) * sle_g1 + yie_g1, c="black", linewidth=3,
                label=f"G1 sl$_{{-15}}$={np.round(sle_g1*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p4, = axes.plot(np.arange(30, 150), np.arange(30, 150) * slll_g1 + yill_g1, c="black", linewidth=3, linestyle="--",
                label=f"G1 sl$_{{31+}}$={np.round(slll_g1*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
# p4, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g1 + yil_g1, c="black", linestyle="--", linewidth=3,
#                 label=f"G1 sl$_l$={np.round(sll_g1*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p5, = axes.plot(np.arange(elpg2) * sle_g2 + yie_g2, c="gray",  linewidth=3,
                label=f"G2 sl$_{{-15}}$={np.round(sle_g2*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
p6, = axes.plot(np.arange(30, 150), np.arange(30, 150) * slll_g2 + yill_g2, c="gray", linewidth=3, linestyle="--",
                label=f"G2 sl$_{{31+}}$={np.round(slll_g2*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")
# p6, = axes.plot(np.arange(20, 150), np.arange(20, 150) * sll_g2 + yil_g2, c="gray", linestyle="--",  linewidth=3,
#                 label=f"G2 sl$_l$={np.round(sll_g2*10, 2)} $\\times$ 10$^{{10}}$ kg s$^{{-1}}$ dec$^{{-1}}$")

l1 = axes.legend(handles=[p1, p2], fontsize=fsz-2.75, ncol=2, loc="upper center")
axes.legend(handles=[p3, p4, p5, p6], ncol=2, fontsize=fsz-2.75, loc="lower center")
axes.add_artist(l1)

axes.set_xlabel("Years since branching of abrupt4xCO2", fontsize=fsz+1)
axes.set_ylabel("Northward Hadley cell in 10$^{10}$ kg s$^{-1}$", fontsize=fsz+1)

# axes.set_title("Northern hemispheric Hadley circulation for G1 and G2\n only for members with AMOC data available", 
#                fontsize=fsz+1)
axes.set_title("Northern hemispheric Hadley circulation G1$-$G2 comparison", 
               fontsize=fsz+1)

axes.tick_params(labelsize=fsz+1)

# axes.set_ylim((52, 83))

# pl.savefig(pl_path + f"/PDF/Hadley_NH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/Hadley_NH_G1_G2_AMOC_models{pl_add}{cslt_fadd}.png", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PDF/Hadley_NH_G1_G2_AMOC_models{pl_add}{cslt_fadd}_V2.pdf", bbox_inches="tight", dpi=250)
pl.show()
pl.close()



#%% test plot the piControl running means
i = 0
g = "G1"

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

for i in np.arange(len(had_pi_n_d[g])):
    axes.plot(had_pi_n_d[g][i], label=mods_d[g][i])
# end for i
axes.legend()
pl.show()
pl.close()


