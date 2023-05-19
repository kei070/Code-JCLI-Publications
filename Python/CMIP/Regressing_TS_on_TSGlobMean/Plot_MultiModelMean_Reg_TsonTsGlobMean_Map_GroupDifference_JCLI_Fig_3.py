"""
Generates Fig. 3 as well as Figs. S2 and S3 in Eiselt and Graversen (2022), JCLI.

Generates three-panel plots for the ts on global mean ts regressions for groups G1 and G2 (for details see our paper).
Shows G1 (upper panel), G2 (middle panel), and their difference (lower panel) for early period, late period, and the
late minus early change.

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
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy import signal
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


#%% choose the variable (tas or ts)
var = "tas"
var_s = "Tas"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


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

direc_5 = "/media/kei070/Work Disk 2"  # SETS BASIC DATA DIRECTORY
direc_6 = "/media/kei070/Seagate"  # SETS BASIC DATA DIRECTORY


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
cslt_path = f"/Model_Lists/ClearSkyLinearityTest/{fb_ts}_Based/Thres_Year_{thr_str}/"

# plot path
pl_path = ""

# generate the plot path
# os.makedirs(pl_path + "/Maps/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/Maps/PNG/", exist_ok=True)
# os.makedirs(pl_path + "/ZonalMeans/PDF/", exist_ok=True)
# os.makedirs(pl_path + "/ZonalMeans/PNG/", exist_ok=True)
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based{thr_fn}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
mm_lr_d_d = dict()
lr_d_d = dict()
mm_lr_e_d = dict()
mm_lr_l_d = dict()
lr_d_zm_d = dict()
mm_lr_d_zm_d = dict()
mm_lr_e_zm_d = dict()
mm_lr_l_zm_d = dict()
trop_band_mm_d = dict()
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
    
    # set up the dictionaries to store all feedbacks
    lr_e = dict()
    lr_l = dict()
    lr_d = dict()
    
    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    lr_e_l = []
    lr_l_l = []
    lr_d_l = []
    
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
            reg_nc = Dataset(glob.glob(data_path5 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + 
                                       ".nc")[0])
            # greg_nc = Dataset(glob.glob(data_path5 + "/TOA_Imbalance_piC_Run/" + a4x5 + "/TOA*" + mod + ".nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path5 + 
                                        f"/Feedbacks/Kernel/abrupt4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])  
            
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
            lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
            lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
            lr_d_l.append(reg_nc.variables["lr_d_rg"][:])
            
            i += 1
            mods.append(mod_pl)
            cols.append("blue")
        except:
            print("\n\nJumping " + mod_pl + " because data are not (yet) available...")
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
            reg_nc = Dataset(glob.glob(data_path6 + f"/{var_s}on{var_s}_Reg/ThresYr{max_dtf_thr[mod_pl]}/" + "*" + mod + 
                                       ".nc")[0])
            # greg_nc = Dataset(glob.glob(data_path6 + "/TOA_Imbalance_piC_Run/" + a4x6 + "/TOA*" + mod + ".nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(data_path6 +
                                         f"/Feedbacks/Kernel/abrupt-4xCO2/{k_p[kl]}/{fb_ts}_Based/*{mod}.nc")[0])
            
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
            lr_e_l.append(reg_nc.variables["lr_e_rg"][0, :, :])
            lr_l_l.append(reg_nc.variables["lr_l_rg"][0, :, :])
            lr_d_l.append(reg_nc.variables["lr_d_rg"][:])     
            
            i += 1
            mods.append(mod_pl)
            cols.append("red")
        except:
            print("\n\nJumping " + mod_pl + " because data are not (yet) available...")
            continue
        # end try except
    
        mod_count += 1     
    # end for mod
    
    print("\n\n")

    #% get lat and lon
    lat = reg_nc.variables["lat_rg"][:]
    lon = reg_nc.variables["lon_rg"][:]


    #% convert the lists into numpy arrays
    lr_e_a = np.array(lr_e_l)
    lr_l_a = np.array(lr_l_l)
    lr_d_a = np.array(lr_d_l)


    #% calculate the individual zonal means
    lr_d_zm = np.mean(lr_d_a, axis=-1)


    #% calculate the multi-model means
    mm_lr_e = np.mean(lr_e_a, axis=0)
    mm_lr_l = np.mean(lr_l_a, axis=0)
    mm_lr_d = np.mean(lr_d_a, axis=0)
    
    print(f"{g12} shape: {np.shape(lr_d_a)}")

    #% calculate the zonal means
    mm_lr_e_zm = np.mean(mm_lr_e, axis=1)
    mm_lr_l_zm = np.mean(mm_lr_l, axis=1)
    mm_lr_d_zm = np.mean(mm_lr_d, axis=1)


    #% calculate the band means
    # trop_lat = 30
    # lat_means = lat_mean(mm_lr_d, lat, lon, lat=trop_lat, return_dic="new")
    
    x1 = 0
    x2 = 360
    y1 = -30
    y2 = 0
    lonm, latm  = np.meshgrid(lon, lat)
    weights = np.zeros(np.shape(mm_lr_d))
    weights[:, :] = np.cos(latm / 180 * np.pi)
    trop_reg = extract_region(x1, x2, y1, y2, lat, lon, mm_lr_d, test_plt=True, plot_title="Southern Tropics")
    trop_latm = extract_region(x1, x2, y1, y2, lat, lon, latm, test_plt=True, plot_title="Southern Tropics")[0]
    trop_band_mm = np.average(trop_reg[0], weights=trop_latm, axis=0)
    
    # store everything in dictionaries
    mm_lr_d_d[g12] = mm_lr_d
    lr_d_d[g12] = lr_d_a
    mm_lr_e_d[g12] = mm_lr_e
    mm_lr_l_d[g12] = mm_lr_l
    lr_d_zm_d[g12] = lr_d_zm
    mm_lr_d_zm_d[g12] = mm_lr_d_zm
    mm_lr_e_zm_d[g12] = mm_lr_e_zm
    mm_lr_l_zm_d[g12] = mm_lr_l_zm
    trop_band_mm_d[g12] = trop_band_mm
    
    mods_d[g12] = np.array(mods)
    
# end for g12


#%% perform a robustness test: per grid cell check if ALL models have the same sign as the model mean
test_g1 = np.zeros(np.shape(mm_lr_d_d["G1"])).astype(int)
for i in np.arange(np.shape(lr_d_d["G1"])[0]):
    test_g1 = test_g1 + (np.sign(mm_lr_d_d["G1"]) == np.sign(lr_d_d["G1"][i])).astype(int)
# end for i

pl.imshow(test_g1, origin="lower")
pl.colorbar()
pl.show()
pl.close()

mask_robust_g1 = test_g1 == 7

pl.imshow(mask_robust_g1, origin="lower")
pl.colorbar()
pl.show()
pl.close()

test_g2 = np.zeros(np.shape(mm_lr_d_d["G2"])).astype(int)
for i in np.arange(np.shape(lr_d_d["G2"])[0]):
    test_g2 = test_g2 + (np.sign(mm_lr_d_d["G2"]) == np.sign(lr_d_d["G2"][i])).astype(int)
# end for i

pl.imshow(test_g2, origin="lower")
pl.colorbar()
pl.show()
pl.close()

mask_robust_g2 = test_g2 >= 7
mask_extra_g2 = test_g2 == 9

pl.imshow(mask_robust_g2, origin="lower")
pl.colorbar()
pl.show()
pl.close()


#%% add the cyclic point
g1_cy = cu.add_cyclic_point(mm_lr_d_d["G1"])
g2_cy = cu.add_cyclic_point(mm_lr_d_d["G2"])

g1m_cy = cu.add_cyclic_point(mask_robust_g1.astype(float))
g2m_cy = cu.add_cyclic_point(mask_robust_g2.astype(float))
g2m_cy_extra = cu.add_cyclic_point(mask_extra_g2.astype(float))

g1m_cy[g1m_cy == 0] = np.nan
g2m_cy[g2m_cy == 0] = np.nan
g2m_cy_extra[g2m_cy_extra == 0] = np.nan

g1e_cy = cu.add_cyclic_point(mm_lr_e_d["G1"])
g2e_cy = cu.add_cyclic_point(mm_lr_e_d["G2"])

g1l_cy = cu.add_cyclic_point(mm_lr_l_d["G1"])
g2l_cy = cu.add_cyclic_point(mm_lr_l_d["G2"])

lon_cy = np.concatenate([lon, np.array([360])])


#%% plot slope changes - map and zonal mean - SINGLE PANEL
"""
xlim = (-3.5, 3.5)

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
xticks = ["-2.5", "0", "2.5"]

clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
proj = ccrs.PlateCarree(central_longitude=0)
# proj = ccrs.Robinson(central_longitude=200)

# plot change
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(g1_cy, l_num, c_min=c_min, c_max=c_max, quant=0.99)

levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
# levels = np.arange(-1, 1.1, 0.25)

fig = pl.figure(figsize=(11.5, 7))
gs = gridspec.GridSpec(nrows=1, ncols=7, wspace=0.5) 

ax1 = pl.subplot(gs[0, :], projection=proj)
    
p1 = ax1.contourf(x, y, g1_cy, levels=levels, norm=norm, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

ax1.contourf(x, y, g1m_cy, hatches=["///"], extend="lower", colors="none", transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope changes in KK$^{-1}$")

ax1.set_title("Surface warming change, G1")

pl.show()
pl.close()
"""


#%% plot slope changes - map and zonal mean
xlim = (-3.5, 3.5)

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
xticks = ["-2.5", "0", "2.5"]

clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot change
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(g1_cy, l_num, c_min=c_min, c_max=c_max, quant=0.99)

ticks = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
levels = np.arange(-1, 1.01, 0.1)

fig = pl.figure(figsize=(9.5, 11.5))
gs = gridspec.GridSpec(nrows=3, ncols=7, wspace=0.5) 


# upper panel -----------------------------------------------------------------------------------------------------------    
ax1 = pl.subplot(gs[0, 1:], projection=proj)
    
p1 = ax1.contourf(x, y, g1_cy, levels=levels, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

# ax1.contourf(x, y, g1m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=ticks)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope changes in KK$^{-1}$")

ax1.set_title("Surface warming change, G1")

ax2 = pl.subplot(gs[0, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax2.plot(mm_lr_d_zm_d["G1"], clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim(xlim)
ax2.set_xticks([-2.5, 0, 2.5])
ax2.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal mean")

ax2.set_yticks(np.sin(yticks / 180 * np.pi))
ax2.set_yticklabels(yticks)
ax2.set_xlabel("Slope change in KK$^{-1}$")


# middel panel ----------------------------------------------------------------------------------------------------------
ax3 = pl.subplot(gs[1, 1:], projection=proj)
    
p3 = ax3.contourf(x, y, g2_cy, levels=levels, norm=norm, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

# ax3.contourf(x, y, g2m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())
# ax3.contourf(x, y, g2m_cy_extra, hatches=["\\\\"], colors="none", transform=ccrs.PlateCarree())
ax3.set_global()

cb3 = fig.colorbar(p3, ax=ax3, shrink=cb_shrink, ticks=ticks)
       
ax3.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb3.set_label("Regression slope changes in KK$^{-1}$")

ax3.set_title("Surface warming change, G2")

ax4 = pl.subplot(gs[1, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax4.plot(mm_lr_d_zm_d["G2"], clat, c="black")
ax4.axvline(x=0, c="gray", linewidth=0.75)
ax4.axhline(y=0, c="gray", linewidth=0.75)

ax4.set_xlim(xlim)
ax4.set_xticks([-2.5, 0, 2.5])
ax4.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax4.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax4.set_title("Zonal mean")

ax4.set_yticks(np.sin(yticks / 180 * np.pi))
ax4.set_yticklabels(yticks)
ax4.set_xlabel("Slope change in KK$^{-1}$")


# lower panel -----------------------------------------------------------------------------------------------------------
ax5 = pl.subplot(gs[2, 1:], projection=proj)
    
p5 = ax5.contourf(x, y, g2_cy - g1_cy, levels=levels, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

cb5 = fig.colorbar(p5, ax=ax5, shrink=cb_shrink, ticks=ticks)
       
ax5.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax5.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb5.set_label("Regression slope changes in KK$^{-1}$")

ax5.set_title("Difference (G2-G1)")

ax6 = pl.subplot(gs[2, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax6.plot(mm_lr_d_zm_d["G2"] - mm_lr_d_zm_d["G1"], clat, c="black")
ax6.axvline(x=0, c="gray", linewidth=0.75)
ax6.axhline(y=0, c="gray", linewidth=0.75)

ax6.set_xlim(xlim)
ax6.set_xticks([-2.5, 0, 2.5])
ax6.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax6.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax6.set_title("Zonal mean")

ax6.set_yticks(np.sin(yticks / 180 * np.pi))
ax6.set_yticklabels(yticks)
ax6.set_xlabel("Slope change in KK$^{-1}$")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
           f"{cslt_fadd}{excl_fadd}.pdf", 
           dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
           f"{cslt_fadd}{excl_fadd}.png", 
           dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% plot early slopes - map and zonal mean
xlim1 = (-0.5, 4)
xticks1 = ["0", "2", "4"]
tick_locs1 = [0, 2, 4]

xlim2 = (-1, 1)
xticks2 = ["1", "0", "1"]
tick_locs2 = [-1, 0, 1]

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])

clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot change
l_num = 14
# levels, norm, cmp = set_levs_norm_colmap(g1_cy, l_num, c_min=c_min, c_max=c_max, quant=0.99)

# levels1 = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
levels1 = np.arange(0, 3.51, 0.1)
ticks1 = np.arange(0, 3.51, 0.5)
levels2 = np.arange(-1, 1.01, 0.05)
ticks2 = np.arange(-1, 1.01, 0.25)
# ticks2 = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
# levels = np.arange(-1, 1.1, 0.25)

fig = pl.figure(figsize=(9.5, 11.5))
gs = gridspec.GridSpec(nrows=3, ncols=7, wspace=0.5) 


# upper panel -----------------------------------------------------------------------------------------------------------    
ax1 = pl.subplot(gs[0, 1:], projection=proj)
    
p1 = ax1.contourf(x, y, g1e_cy, levels=levels1, cmap=cm.Reds, extend="both", 
                  transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=ticks1)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope in KK$^{-1}$")

ax1.set_title("Early surface warming, G1")

ax2 = pl.subplot(gs[0, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax2.plot(mm_lr_e_zm_d["G1"], clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim(xlim1)
ax2.set_xticks(tick_locs1)
ax2.set_xticklabels(xticks1)

clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal mean")

ax2.set_yticks(np.sin(yticks / 180 * np.pi))
ax2.set_yticklabels(yticks)
ax2.set_xlabel("Slope in KK$^{-1}$")


# middel panel ----------------------------------------------------------------------------------------------------------
ax3 = pl.subplot(gs[1, 1:], projection=proj)
    
p3 = ax3.contourf(x, y, g2e_cy, levels=levels1, cmap=cm.Reds, extend="both", 
                  transform=ccrs.PlateCarree())

cb3 = fig.colorbar(p3, ax=ax3, shrink=cb_shrink, ticks=ticks1)
       
ax3.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb3.set_label("Regression slope in KK$^{-1}$")

ax3.set_title("Early surface warming, G2")

ax4 = pl.subplot(gs[1, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax4.plot(mm_lr_e_zm_d["G2"], clat, c="black")
ax4.axvline(x=0, c="gray", linewidth=0.75)
ax4.axhline(y=0, c="gray", linewidth=0.75)

ax4.set_xlim(xlim1)
ax4.set_xticks(tick_locs1)
ax4.set_xticklabels(xticks1)

clat_span = clat[-1] - clat[0]
ax4.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax4.set_title("Zonal mean")

ax4.set_yticks(np.sin(yticks / 180 * np.pi))
ax4.set_yticklabels(yticks)
ax4.set_xlabel("Slope in KK$^{-1}$")


# lower panel -----------------------------------------------------------------------------------------------------------
ax5 = pl.subplot(gs[2, 1:], projection=proj)
    
p5 = ax5.contourf(x, y, g2e_cy - g1e_cy, levels=levels2, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

cb5 = fig.colorbar(p5, ax=ax5, shrink=cb_shrink, ticks=ticks2)
       
ax5.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax5.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb5.set_label("Regression slope in KK$^{-1}$")

ax5.set_title("Difference (G2-G1)")

ax6 = pl.subplot(gs[2, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax6.plot(mm_lr_e_zm_d["G2"] - mm_lr_e_zm_d["G1"], clat, c="black")
ax6.axvline(x=0, c="gray", linewidth=0.75)
ax6.axhline(y=0, c="gray", linewidth=0.75)

ax6.set_xlim(xlim2)
ax6.set_xticks(tick_locs2)
ax6.set_xticklabels(xticks2)

clat_span = clat[-1] - clat[0]
ax6.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax6.set_title("Zonal mean")

ax6.set_yticks(np.sin(yticks / 180 * np.pi))
ax6.set_yticklabels(yticks)
ax6.set_xlabel("Slope in KK$^{-1}$")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Early_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
           f"{cslt_fadd}{excl_fadd}.pdf", dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Early_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" + 
           f"{cslt_fadd}{excl_fadd}.png", dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% plot late slopes - map and zonal mean
xlim1 = (-0.5, 4)
xticks1 = ["0", "2", "4"]
tick_locs1 = [0, 2, 4]

xlim2 = (-1, 2.5)
xticks2 = ["1", "0", "2"]
tick_locs2 = [-1, 0, 2]

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot change
l_num = 14
# levels, norm, cmp = set_levs_norm_colmap(g1_cy, l_num, c_min=c_min, c_max=c_max, quant=0.99)

levels1 = np.arange(0, 3.51, 0.1)
ticks1 = np.arange(0, 3.51, 0.5)
levels2 = np.arange(-1, 1.01, 0.05)
ticks2 = np.arange(-1, 1.01, 0.25)

fig = pl.figure(figsize=(9.5, 11.5))
gs = gridspec.GridSpec(nrows=3, ncols=7, wspace=0.5) 


# upper panel -----------------------------------------------------------------------------------------------------------    
ax1 = pl.subplot(gs[0, 1:], projection=proj)
    
p1 = ax1.contourf(x, y, g1l_cy, levels=levels1, cmap=cm.Reds, extend="both", 
                  transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=ticks1)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope in KK$^{-1}$")

ax1.set_title("Late surface warming, G1")

ax2 = pl.subplot(gs[0, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax2.plot(mm_lr_l_zm_d["G1"], clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim(xlim1)
ax2.set_xticks(tick_locs1)
ax2.set_xticklabels(xticks1)

clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal mean")

ax2.set_yticks(np.sin(yticks / 180 * np.pi))
ax2.set_yticklabels(yticks)
ax2.set_xlabel("Slope in KK$^{-1}$")


# middel panel ----------------------------------------------------------------------------------------------------------
ax3 = pl.subplot(gs[1, 1:], projection=proj)
    
p3 = ax3.contourf(x, y, g2l_cy, levels=levels1, cmap=cm.Reds, extend="both", 
                  transform=ccrs.PlateCarree())

cb3 = fig.colorbar(p3, ax=ax3, shrink=cb_shrink, ticks=ticks1)
       
ax3.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb3.set_label("Regression slope in KK$^{-1}$")

ax3.set_title("Late surface warming, G2")

ax4 = pl.subplot(gs[1, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax4.plot(mm_lr_l_zm_d["G2"], clat, c="black")
ax4.axvline(x=0, c="gray", linewidth=0.75)
ax4.axhline(y=0, c="gray", linewidth=0.75)

ax4.set_xlim(xlim1)
ax4.set_xticks(tick_locs1)
ax4.set_xticklabels(xticks1)

clat_span = clat[-1] - clat[0]
ax4.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax4.set_title("Zonal mean")

ax4.set_yticks(np.sin(yticks / 180 * np.pi))
ax4.set_yticklabels(yticks)
ax4.set_xlabel("Slope in KK$^{-1}$")


# lower panel -----------------------------------------------------------------------------------------------------------
ax5 = pl.subplot(gs[2, 1:], projection=proj)
    
p5 = ax5.contourf(x, y, g2l_cy - g1l_cy, levels=levels2, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

cb5 = fig.colorbar(p5, ax=ax5, shrink=cb_shrink, ticks=ticks2)
       
ax5.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax5.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb5.set_label("Regression slope in KK$^{-1}$")

ax5.set_title("Difference (G2-G1)")

ax6 = pl.subplot(gs[2, 0])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax6.plot(mm_lr_l_zm_d["G2"] - mm_lr_l_zm_d["G1"], clat, c="black")
ax6.axvline(x=0, c="gray", linewidth=0.75)
ax6.axhline(y=0, c="gray", linewidth=0.75)

ax6.set_xlim(xlim2)
ax6.set_xticks(tick_locs2)
ax6.set_xticklabels(xticks2)

clat_span = clat[-1] - clat[0]
ax6.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax6.set_title("Zonal mean")

ax6.set_yticks(np.sin(yticks / 180 * np.pi))
ax6.set_yticklabels(yticks)
ax6.set_xlabel("Slope in KK$^{-1}$")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Late_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" + 
           f"{cslt_fadd}{excl_fadd}.pdf", dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Late_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" + 
           f"{cslt_fadd}{excl_fadd}.png", dpi=250, bbox_inches="tight")

pl.show()
pl.close()



#%% plot the early and late zonal mean slopes and slope change in one plot
"""
xticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])

clat = np.sin(lat / 180 * np.pi)

# find max and min y-values
lr_min = np.min([mm_lr_e_zm, mm_lr_l_zm, mm_lr_d_zm])
lr_max = np.max([mm_lr_e_zm, mm_lr_l_zm, mm_lr_d_zm])

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))

# plot the zero lines
ax1.axhline(y=0, linewidth=0.75, c="gray", linestyle="--")
ax1.axvline(x=0, linewidth=0.75, c="gray", linestyle="--")

# plot feedbacks
ax1.plot(clat, mm_lr_e_zm, c="blue", label="early", linewidth=0.7)
ax1.plot(clat, mm_lr_l_zm, c="red", label="late", linewidth=0.7)
ax1.plot(clat, mm_lr_d_zm, c="black", label="change", linewidth=1.5)

ax1.legend()

ax1.set_xlabel("latitude")
ax1.set_ylabel("slope (change) in KK$^{-1}$")

ax1.set_ylim((-1.5, 4))

ax1.set_title(f"Local {var_s} on Global {var_s} Regression Multi-Model Mean Zonal Mean{excl_tadd}\n({thr_plt}) " +
              f"({np.shape(lr_e_a)[0]} Models)")

ax1.set_xticks(np.sin(xticks / 180 * np.pi))
ax1.set_xticklabels(xticks)

pl.savefig(pl_path + f"/ZonalMeans/PDF/{var_s}on{var_s}_Reg_ZonalMean_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", 
           bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/ZonalMeans/PNG/{var_s}on{var_s}_Reg_ZonalMean_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", 
           bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot EARLY slopes
c_min = -3
c_max = 3

cb_shrink = 0.75

# set up a projection
x, y = np.meshgrid(lon, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot EARLY feedbacks
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(mm_lr_e, l_num, c_min=c_min, c_max=c_max, quant=0.99)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6), subplot_kw=dict(projection=proj))
    
p1 = ax1.contourf(x, y, mm_lr_e, levels=levels, norm=norm, cmap=cmp, extend="both", transform=ccrs.PlateCarree())
cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
#p1 = ax1.contourf(x, y, fb_e, extend="both", cmap=cm.RdBu_r, transform=ccrs.PlateCarree())
#cb1 = fig.colorbar(p1, ax=ax1)                  
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("regression slopes in KK$^{-1}$")

ax1.set_title(f'Local on Global Mean {var_s} Multi-Model Mean{excl_tadd} Early Period ({thr_plt})' + 
              f' ({np.shape(lr_e_a)[0]} Models)')

pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_Reg_Map_EarlyPeriod_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", dpi=250, 
           bbox_inches="tight")
pl.savefig(pl_path + f"/Maps/PNG/{var_s}on{var_s}_Reg_Map_EarlyPeriod_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", dpi=250,
           bbox_inches="tight")

pl.show()
pl.close()    


#%% plot LATE slopes
c_min = -3
c_max = 3

cb_shrink = 0.75

# set up a projection
x, y = np.meshgrid(lon, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot EARLY feedbacks
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(mm_lr_l, l_num, c_min=c_min, c_max=c_max, quant=0.99)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6), subplot_kw=dict(projection=proj))
    
p1 = ax1.contourf(x, y, mm_lr_l, levels=levels, norm=norm, cmap=cmp, extend="both", transform=ccrs.PlateCarree())
cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
#p1 = ax1.contourf(x, y, fb_e, extend="both", cmap=cm.RdBu_r, transform=ccrs.PlateCarree())
#cb1 = fig.colorbar(p1, ax=ax1)                  
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("regression slopes in KK$^{-1}$")

ax1.set_title(f'Local on Global Mean {var_s} Multi-Model Mean{excl_tadd} Late Period ({thr_plt})' + 
              f' ({np.shape(lr_e_a)[0]} Models)')

pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_Reg_Map_LatePeriod_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", dpi=250, 
           bbox_inches="tight")
pl.savefig(pl_path + f"/Maps/PNG/{var_s}on{var_s}_Reg_Map_LatePeriod_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", dpi=250,
           bbox_inches="tight")

pl.show()
pl.close()      


#%% plot slope changes
c_min = -1
c_max = 1

cb_shrink = 0.75

# set up a projection
x, y = np.meshgrid(lon, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot EARLY feedbacks
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(mm_lr_d, l_num, c_min=c_min, c_max=c_max, quant=0.99)

levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]  # 
# levels = np.arange(-1, 1.1, 0.25)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(10, 6), subplot_kw=dict(projection=proj))
    
p1 = ax1.contourf(x, y, mm_lr_d, levels=levels, norm=norm, cmap=cmp, extend="both", transform=ccrs.PlateCarree())
cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
#p1 = ax1.contourf(x, y, fb_e, extend="both", cmap=cm.RdBu_r, transform=ccrs.PlateCarree())
#cb1 = fig.colorbar(p1, ax=ax1)                  
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("regression slopes in KK$^{-1}$")

ax1.set_title(f'Local on Global Mean {var_s} Multi-Model Mean{excl_tadd} Change ({thr_plt})' + 
              f' ({np.shape(lr_e_a)[0]} Models)')

pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_Reg_Map_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", dpi=250, 
           bbox_inches="tight")
pl.savefig(pl_path + f"/Maps/PNG/{var_s}on{var_s}_Reg_Map_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", dpi=250,
           bbox_inches="tight")

pl.show()
pl.close()


#%% plot slope changes - map and zonal mean
c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
proj = ccrs.Robinson(central_longitude=200)

# plot change
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(mm_lr_d, l_num, c_min=c_min, c_max=c_max, quant=0.99)

levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
# levels = np.arange(-1, 1.1, 0.25)

fig = pl.figure(figsize=(20, 7))
gs = gridspec.GridSpec(nrows=1, ncols=7, wspace=0.5) 
    
ax1 = pl.subplot(gs[0, 1:], projection=proj)
    
p1 = ax1.contourf(x, y, mm_lr_d, levels=levels, norm=norm, cmap=cmp, extend="both", transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("regression slope changes in KK$^{-1}$")

ax1.set_title(f'Local on Global Mean {var_s} Multi-Model Mean{excl_tadd} Change ({thr_plt})' + 
              f' ({np.shape(lr_e_a)[0]} Models)')

ax2 = pl.subplot(gs[0, 0])

for i in np.arange(np.shape(lr_d_zm)[0]):
    ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# end for i    
ax2.plot(mm_lr_d_zm, clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim((-3, 3))
clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal Mean")

ax2.set_yticks(np.sin(xticks / 180 * np.pi))
ax2.set_yticklabels(xticks)
ax2.set_xlabel("slope change in KK$^{-1}$")

# pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", 
#            dpi=250, bbox_inches="tight")
# pl.savefig(pl_path + f"/Maps/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", 
#            dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% multi-panel plot

if excl == "not high LR dFb":
    model_list = np.array(["BCC-CSM1.1", "BCC-CSM1.1(m)", "GFDL-CM3", "GISS-E2-H", "MPI-ESM-P", "ACCESS-ESM1.5", 
                           "EC-Earth3", "FGOALS-g3", "INM-CM4.8"])
elif excl == "not low LR dFb":    
    model_list = np.array(["IPSL-CM5A-LR", "CanESM5", "CNRM-CM6.1", "E3SM-1.0", "UKESM1.0-LL", "CMCC-ESM2"])
# end if elif

mods_a = np.array(mods)

l_num = 14
levels, norm, cmp = set_levs_norm_colmap(mm_lr_d, l_num, c_min=c_min, c_max=c_max, quant=0.99)
levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]

# set the number of rows and columns to be plotted, respectively
nrows = 3
ncols = 2

cb_shrink = 0.75

# set up projection and generate the meshgrid of lat and lon
proj = ccrs.Robinson(central_longitude=200)
x, y = np.meshgrid(lon, lat)

fig, axes = pl.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12), subplot_kw=dict(projection=proj))

ind1 = 0
for ro in np.arange(nrows):
    for co in np.arange(ncols):
    
        # get the index
        ind2 = np.where(model_list[ind1] == mods_a)[0][0]
        
        p1 = axes[ro, co].contourf(x, y, lr_d_a[ind2, :, :], levels=levels, norm=norm, cmap=cmp, extend="both", 
                                   transform=ccrs.PlateCarree())
        cb = fig.colorbar(p1, ax=axes[ro, co], shrink=cb_shrink, ticks=levels)
        axes[ro, co].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
        axes[ro, co].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        cb.set_label("regression slope changes in KK$^{-1}$")
        
        axes[ro, co].set_title(model_list[ind1])
        
        ind1 += 1
    # end for co
# end for mod ro

pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_MultiPanel_Reg_Map_Change_HighLRdFb_d{var_s}_Mods_{thr_fn}.pdf", 
           dpi=250, bbox_inches="tight")
    
pl.show()
pl.close()


#%% generate the plots on maps and zonal band mean - CHANGE
xticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])

clat = np.sin(lat / 180 * np.pi)

c_min = -2.5
c_max = 2.5

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
# proj = ccrs.Mollweide(central_longitude=180)
# proj = ccrs.PlateCarree(central_longitude=180)
proj = ccrs.Robinson(central_longitude=180)

l_num = 14
# levels, norm, cmp = set_levs_norm_colmap(lr_d, l_num, c_min=c_min, c_max=c_max, quant=0.99)
# levels = [-5, -4, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5]
levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]

fig = pl.figure(figsize=(17, 15))
gs = gridspec.GridSpec(nrows=14, ncols=1, wspace=0.5) 

ax1 = pl.subplot(gs[:9, 0], projection=proj)
    
p1 = ax1.contourf(x, y, mm_lr_d, levels=levels, cmap=cm.RdBu_r, extend="both", transform=ccrs.PlateCarree())
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax1.set_title(f'Local on Global Mean {var_s} Multi-Model Mean{excl_tadd} Change ({thr_plt})' + 
              f' ({np.shape(lr_e_a)[0]} Models)')

ax2 = pl.subplot(gs[9, 0])
cb1 = fig.colorbar(p1, cax=ax2, orientation="horizontal", ticks=levels)
cb1.set_label("regression slope difference KK$^{-1}$")

ax3 = pl.subplot(gs[11:, 0])
ax3.axhline(y=0, c="gray", linewidth=0.75)
ax3.axvline(x=180, c="gray", linewidth=0.75)

ax3.plot(lon, trop_band_mean, c="black")
ax3.set_ylim((-0.5, 0.5))
ax3.set_xlim((-2.5, 362.5))
ax3.set_xlabel("longitude")
ax3.set_ylabel("regression slope difference KK$^{-1}$")
ax3.set_title(f"Zonal Band Mean {np.abs(y1)}$\degree$S to {y2}$\degree$")

pl.savefig(pl_path + f"/Maps/PDF/{var_s}on{var_s}_Reg_Map_and_TropMean_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.pdf", 
           dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/Maps/PNG/{var_s}on{var_s}_Reg_Map_and_TropMean_Change_MMMean{cslt_fadd}{excl_fadd}{thr_fn}.png", 
           dpi=250, bbox_inches="tight")
pl.show()
pl.close()


"""


#%% plot slope changes - map and zonal mean  --> DIFFERENT projecton
ax_w = 4
ncol_gr = 15

xlim = (-3.5, 3.5)

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
xticks = ["-2.5", "0", "2.5"]

clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
# proj = ccrs.PlateCarree(central_longitude=0)
# proj = ccrs.Robinson(central_longitude=200)
proj = ccrs.LambertAzimuthalEqualArea(central_longitude=-179, central_latitude=0.0, false_easting=0.0, 
                                      false_northing=0.0, globe=None)
# proj = ccrs.Orthographic(central_longitude=180, central_latitude=45, globe=None)

# plot change
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(g1_cy, l_num, c_min=c_min, c_max=c_max, quant=0.99)

levels = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]
# levels = np.arange(-1, 1.1, 0.25)

fig = pl.figure(figsize=(6.5, 11.5))
gs = gridspec.GridSpec(nrows=3, ncols=ncol_gr, wspace=0.5) 


# upper panel -----------------------------------------------------------------------------------------------------------    
ax1 = pl.subplot(gs[0, ax_w:], projection=proj)
    
p1 = ax1.contourf(x, y, g1_cy, levels=levels, norm=norm, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

# ax1.contourf(x, y, g1m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, ax=ax1, shrink=cb_shrink, ticks=levels)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope changes in KK$^{-1}$")

ax1.set_title("Surface warming change, G1")

ax2 = pl.subplot(gs[0, :ax_w])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax2.plot(mm_lr_d_zm_d["G1"], clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim(xlim)
ax2.set_xticks([-2.5, 0, 2.5])
ax2.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal mean")

ax2.set_yticks(np.sin(yticks / 180 * np.pi))
ax2.set_yticklabels(yticks)
ax2.set_xlabel("Slope change in KK$^{-1}$")


# middel panel ----------------------------------------------------------------------------------------------------------
ax3 = pl.subplot(gs[1, ax_w:], projection=proj)
    
p3 = ax3.contourf(x, y, g2_cy, levels=levels, norm=norm, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

# ax3.contourf(x, y, g2m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())
# ax3.contourf(x, y, g2m_cy_extra, hatches=["\\\\"], colors="none", transform=ccrs.PlateCarree())
ax3.set_global()

cb3 = fig.colorbar(p3, ax=ax3, shrink=cb_shrink, ticks=levels)
       
ax3.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb3.set_label("Regression slope changes in KK$^{-1}$")

ax3.set_title("Surface warming change, G2")

ax4 = pl.subplot(gs[1, :ax_w])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax4.plot(mm_lr_d_zm_d["G2"], clat, c="black")
ax4.axvline(x=0, c="gray", linewidth=0.75)
ax4.axhline(y=0, c="gray", linewidth=0.75)

ax4.set_xlim(xlim)
ax4.set_xticks([-2.5, 0, 2.5])
ax4.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax4.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax4.set_title("Zonal mean")

ax4.set_yticks(np.sin(yticks / 180 * np.pi))
ax4.set_yticklabels(yticks)
ax4.set_xlabel("Slope change in KK$^{-1}$")


# lower panel -----------------------------------------------------------------------------------------------------------
ax5 = pl.subplot(gs[2, ax_w:], projection=proj)
    
p5 = ax5.contourf(x, y, g2_cy - g1_cy, levels=levels, norm=norm, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())

cb5 = fig.colorbar(p5, ax=ax5, shrink=cb_shrink, ticks=levels)
       
ax5.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax5.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb5.set_label("Regression slope changes in KK$^{-1}$")

ax5.set_title("Difference (G2-G1)")

ax6 = pl.subplot(gs[2, :ax_w])

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax6.plot(mm_lr_d_zm_d["G2"] - mm_lr_d_zm_d["G1"], clat, c="black")
ax6.axvline(x=0, c="gray", linewidth=0.75)
ax6.axhline(y=0, c="gray", linewidth=0.75)

ax6.set_xlim(xlim)
ax6.set_xticks([-2.5, 0, 2.5])
ax6.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax6.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax6.set_title("Zonal mean")

ax6.set_yticks(np.sin(yticks / 180 * np.pi))
ax6.set_yticklabels(yticks)
ax6.set_xlabel("Slope change in KK$^{-1}$")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
           f"{cslt_fadd}{excl_fadd}_Lambert.pdf", 
           dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
           f"{cslt_fadd}{excl_fadd}_Lambert.png", 
           dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% plot slope changes - map and zonal mean  --> ONE MORE DIFFERENT projecton
ax_w = 4
ncol_gr = 15

xlim = (-3.5, 3.5)

yticks = np.array([-90, -50, -30, -15, 0, 15, 30, 50, 90])
xticks = ["-2.5", "0", "2.5"]

clat = np.sin(lat / 180 * np.pi)

c_min = -1
c_max = 1

cb_shrink = 1

# set up a projection
x, y = np.meshgrid(lon_cy, lat)
proj = ccrs.Orthographic(central_longitude=335, central_latitude=90)

# plot change
l_num = 14
levels = np.arange(-2, 2.01, 0.1)
ticks = np.arange(-2, 2.01, 0.5)

fig = pl.figure(figsize=(6.5, 11.5))
gs = gridspec.GridSpec(nrows=3, ncols=20, wspace=0.5) 


# upper panel -----------------------------------------------------------------------------------------------------------    
ax1 = pl.subplot(gs[0, 4:19], projection=proj)
ax2 = pl.subplot(gs[0, :4])
ax11 = pl.subplot(gs[0, 19])
    
p1 = ax1.contourf(x, y, g1_cy, levels=levels, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())
ax1.set_global()
# ax1.contourf(x, y, g1m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())

cb1 = fig.colorbar(p1, cax=ax11, ticks=ticks)
       
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb1.set_label("Regression slope changes in KK$^{-1}$")

ax1.set_title("Surface warming change, G1")

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax2.plot(mm_lr_d_zm_d["G1"], clat, c="black")
ax2.axvline(x=0, c="gray", linewidth=0.75)
ax2.axhline(y=0, c="gray", linewidth=0.75)

ax2.set_xlim(xlim)
ax2.set_xticks([-2.5, 0, 2.5])
ax2.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax2.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax2.set_title("Zonal mean")

ax2.set_yticks(np.sin(yticks / 180 * np.pi))
ax2.set_yticklabels(yticks)
ax2.set_xlabel("Slope change in KK$^{-1}$")


# middel panel ----------------------------------------------------------------------------------------------------------
ax3 = pl.subplot(gs[1, 4:19], projection=proj)
ax4 = pl.subplot(gs[1, :4])
ax31 = pl.subplot(gs[1, 19])
    
p3 = ax3.contourf(x, y, g2_cy, levels=levels, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())
# ax3.contourf(x, y, g2m_cy, hatches=["///"], colors="none", transform=ccrs.PlateCarree())
# ax3.contourf(x, y, g2m_cy_extra, hatches=["\\\\"], colors="none", transform=ccrs.PlateCarree())
ax3.set_global()

cb3 = fig.colorbar(p3, cax=ax31, shrink=cb_shrink, ticks=ticks)
       
ax3.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb3.set_label("Regression slope changes in KK$^{-1}$")

ax3.set_title("Surface warming change, G2")

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax4.plot(mm_lr_d_zm_d["G2"], clat, c="black")
ax4.axvline(x=0, c="gray", linewidth=0.75)
ax4.axhline(y=0, c="gray", linewidth=0.75)

ax4.set_xlim(xlim)
ax4.set_xticks([-2.5, 0, 2.5])
ax4.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax4.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax4.set_title("Zonal mean")

ax4.set_yticks(np.sin(yticks / 180 * np.pi))
ax4.set_yticklabels(yticks)
ax4.set_xlabel("Slope change in KK$^{-1}$")


# lower panel -----------------------------------------------------------------------------------------------------------
ax5 = pl.subplot(gs[2, 4:19], projection=proj)
ax6 = pl.subplot(gs[2, :4])
ax51 = pl.subplot(gs[2, 19])
    
p5 = ax5.contourf(x, y, g2_cy - g1_cy, levels=levels, cmap=cm.RdBu_r, extend="both", 
                  transform=ccrs.PlateCarree())
ax5.set_global()
cb5 = fig.colorbar(p5, cax=ax51, shrink=cb_shrink, ticks=ticks)
       
ax5.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax5.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cb5.set_label("Regression slope changes in KK$^{-1}$")

ax5.set_title("Difference (G2-G1)")

# individual model zonal means
# for i in np.arange(np.shape(lr_d_zm)[0]):
#     ax2.plot(lr_d_zm[i, :], clat, c="gray", linewidth=0.5)
# # end for i    

ax6.plot(mm_lr_d_zm_d["G2"] - mm_lr_d_zm_d["G1"], clat, c="black")
ax6.axvline(x=0, c="gray", linewidth=0.75)
ax6.axhline(y=0, c="gray", linewidth=0.75)

ax6.set_xlim(xlim)
ax6.set_xticks([-2.5, 0, 2.5])
ax6.set_xticklabels(xticks)

clat_span = clat[-1] - clat[0]
ax6.set_ylim((clat[0] - clat_span*0.01, clat[-1] + clat_span*0.01))

ax6.set_title("Zonal mean")

ax6.set_yticks(np.sin(yticks / 180 * np.pi))
ax6.set_yticklabels(yticks)
ax6.set_xlabel("Slope change in KK$^{-1}$")

pl.subplots_adjust(hspace=0.4)

# pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
#            f"{cslt_fadd}{excl_fadd}_Lambert.pdf", 
#            dpi=250, bbox_inches="tight")
# pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_Reg_Map_and_ZM_Change_High_and_Low_GM_LR_dFb{pl_add}_GroupComp{thr_fn}" +
#            f"{cslt_fadd}{excl_fadd}_Lambert.png", 
#            dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% plot the differences - WITHOUT zonal means
levels = np.arange(-2, 2.1, 0.1)
ticks = np.arange(-2, 2.1, 0.5)

x, y = np.meshgrid(lon_cy, lat)

proj = ccrs.Orthographic(central_longitude=335, central_latitude=90)

fig, axes = pl.subplots(ncols=1, nrows=3, figsize=(12, 12), subplot_kw={'projection':proj})
# axes[0].quiver(x, y, exp_u_levd, exp_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p1 = axes[0].contourf(x, y, g1_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb1 = fig.colorbar(p1, ax=axes[0], ticks=ticks)
cb1.set_label("Relative warming change in K K$^{-1}$")
axes[0].set_global()
# axes[1].quiver(x, y, com_u_levd, com_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p2 = axes[1].contourf(x, y, g2_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb2 = fig.colorbar(p2, ax=axes[1], ticks=ticks)
cb2.set_label("Relative warming change in K K$^{-1}$")
axes[1].set_global()

p3 = axes[2].contourf(x, y, g2_cy-g1_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", 
                      levels=levels)
cb3 = fig.colorbar(p3, ax=axes[2], ticks=ticks)
cb3.set_label("Difference in K K$^{-1}$")
axes[2].set_global()
# axes[2].quiver(x, y, (exp_u_levd-com_u_levd), (exp_v_levd-com_v_levd), transform=ccrs.PlateCarree(), regrid_shape=25)

axes[0].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axes[1].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axes[2].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

axes[0].set_title("G1, years 21-150 minus 1-20")
axes[1].set_title("G2 years 21-150 minus 1-20")
axes[2].set_title("G2 minus G1 years 21-150 minus 1-20")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_RegMap_Arctic_G1_G2_with_Diff.pdf", 
           dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_RegMap_Arctic_G1_G2_with_Diff.png", 
           dpi=250, bbox_inches="tight")

pl.show()
pl.close()


#%% plot the differences - WITHOUT zonal means --> AMOC meeting (London 2022) poster
levels = np.arange(-2, 2.1, 0.1)
ticks = np.arange(-2, 2.1, 0.5)

x, y = np.meshgrid(lon_cy, lat)

# proj = ccrs.Orthographic(central_longitude=335, central_latitude=45)
proj = ccrs.Robinson(central_longitude=280)

fig, axes = pl.subplots(ncols=1, nrows=2, figsize=(8, 8), subplot_kw={'projection':proj})
# axes[0].quiver(x, y, exp_u_levd, exp_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p1 = axes[0].contourf(x, y, g1_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb1 = fig.colorbar(p1, ax=axes[0], ticks=ticks)
cb1.set_label("Relative warming change in K K$^{-1}$")
axes[0].set_global()
# axes[1].quiver(x, y, com_u_levd, com_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p2 = axes[1].contourf(x, y, g2_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb2 = fig.colorbar(p2, ax=axes[1], ticks=ticks)
cb2.set_label("Relative warming change in K K$^{-1}$")
axes[1].set_global()

axes[0].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axes[1].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

axes[0].set_title("G1, years 21-150 minus 1-20")
axes[1].set_title("G2 years 21-150 minus 1-20")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_RegMap_Arctic_G1_G2.pdf", dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_RegMap_Arctic_G1_G2.png", dpi=250, bbox_inches="tight")

pl.show()
pl.close() 


#%% plot the differences - WITHOUT zonal means --> EGU23 poster
levels = np.arange(-2, 2.1, 0.1)
ticks = np.arange(-2, 2.1, 0.5)

x, y = np.meshgrid(lon_cy, lat)

# proj = ccrs.Orthographic(central_longitude=335, central_latitude=45)
proj = ccrs.Robinson(central_longitude=280)

fig, axes = pl.subplots(ncols=1, nrows=3, figsize=(8, 12), subplot_kw={'projection':proj})
# axes[0].quiver(x, y, exp_u_levd, exp_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p1 = axes[0].contourf(x, y, g1_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb1 = fig.colorbar(p1, ax=axes[0], ticks=ticks)
cb1.set_label("Relative warming change in K K$^{-1}$")
axes[0].set_global()
# axes[1].quiver(x, y, com_u_levd, com_v_levd, transform=ccrs.PlateCarree(), regrid_shape=25)

p2 = axes[1].contourf(x, y, g2_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb2 = fig.colorbar(p2, ax=axes[1], ticks=ticks)
cb2.set_label("Relative warming change in K K$^{-1}$")
axes[1].set_global()

p3 = axes[2].contourf(x, y, g2_cy - g1_cy, transform=ccrs.PlateCarree(), cmap=cm.RdBu_r, extend="both", levels=levels)
cb3 = fig.colorbar(p3, ax=axes[2], ticks=ticks)
cb3.set_label("Difference in K K$^{-1}$")
axes[2].set_global()

axes[0].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axes[1].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
axes[2].coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
axes[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

axes[0].set_title("G1, years 21-150 minus 1-20")
axes[1].set_title("G2 years 21-150 minus 1-20")
axes[2].set_title("G2 minus G1")

pl.subplots_adjust(hspace=0.4)

pl.savefig(pl_path + f"/PDF/{var_s}on{var_s}_RegMap_G1_G2.pdf", dpi=250, bbox_inches="tight")
pl.savefig(pl_path + f"/PNG/{var_s}on{var_s}_RegMap_G1_G2.png", dpi=250, bbox_inches="tight")

pl.show()
pl.close() 



#%% store the slopes
"""
out_path = "/media/kei070/One Touch/Uni/PhD/Tromsoe_UiT/Work/Running_CESM2/G1_G2_Data/"

# set up an output name
out_name = "G1_G2_ts_Loc_on_Glob_Regression_Slopes.nc"

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")


# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))
f.createDimension("group", 2)

# build the variables
latitude = f.createVariable("lat", "f8", "lat")
longitude = f.createVariable("lon", "f8", "lon")
group_nc = f.createVariable("group", "S2", "group")


# create the wind speed variable
dsl_nc = f.createVariable("delta_slope", "f4", ("group", "lat", "lon"))
esl_nc = f.createVariable("early_slope", "f4", ("group", "lat", "lon"))
lsl_nc = f.createVariable("late_slope", "f4", ("group", "lat", "lon"))

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat

group_nc[:] = np.array(["G1", "G2"])
dsl_nc[0, :, :] = mm_lr_d_d["G1"]
dsl_nc[1, :, :] = mm_lr_d_d["G2"]
esl_nc[0, :, :] = mm_lr_e_d["G1"]
esl_nc[1, :, :] = mm_lr_e_d["G2"]
lsl_nc[0, :, :] = mm_lr_l_d["G1"]
lsl_nc[1, :, :] = mm_lr_l_d["G2"]

# add attributes
f.description = ("")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees_east"
latitude.units = "degrees_north"

# close the dataset
f.close()
"""

