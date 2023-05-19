"""
Compare regional LTS and CRE.

For now Figs. 3 and S3 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_dir5 and data_dir6 as well as pl_path. Potentially change the paths in the for-loop.
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
from scipy.stats import ttest_ind
from scipy.stats import linregress as lr
from scipy.stats import pearsonr as pears_corr
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

from Functions.Func_Region_Mean import region_mean
from Functions.Func_Extract_Geographic_Region import extract_region
from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Climatology import climatology
from Functions.Func_SeasonalMean import sea_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Regrid_Data import remap as remap2
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set model groups
# models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
# models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]
models1 = ["UKESM1-0-LL", "IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "NorESM1_M", "BCC-CSM2-MR", "EC-Earth3", "FGOALS-g3", 
           "INM-CM4-8"]
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR"]
models2 = ["BCC-CSM2-MR", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models = {"G1":models1, "G2":models2}
g_cols = {"G1":"blue", "G2":"red"}


#%% how many years
n_months = 60


#%% set the region
### --> Subpolar North Atlantic (SPNA) in Bellomo et al. (2021):  50-70N, 80W-10E

# MLNA
x1 = [300]
x2 = [350]
y1 = [40]
y2 = [60]

# MLNP
# x1 = [150]
# x2 = [220]
# y1 = [30]
# y2 = [60]

# SO
# x1 = [0]
# x2 = [360]
# y1 = [-60]
#  = [-30]


# Bellomo et al. (2021)
# x1 = [280]
# x2 = [10]
# y1 = [50]
# y2 = [70]

# set the region name
reg_name = "2"
reg_name_title = {"1":"Barents-Norwegian-Greenland Sea", "2":"mid-latitude North Atlantic", 
                  "3":"Western Pacific", "4":"Central Mid-Latitude North Pacific", 
                  "5":"Mid-Latitude North Pacific", "6":"Southern Ocean"}
reg_name_fn = {"1":"BNG_Sea", "2":"MLNA", "3":"NWP", "4":"CMLP", "5":"MLNP", "6":"SO"}


#%% choose the threshold year; if this is set to 0, the "year of maximum feedback change" will be taken from the below
#   dictionary
thr_yr = 20

thr_min = 15
thr_max = 75


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


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = "/CMIP5/"  # SETS BASIC DATA DIRECTORY
data_dir6 = "/CMIP6/"  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""

# generate the plot path
os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% loop over the groups and collect their data to compare later

# set up dictionaries to collect the data
mods_d = dict()

pl_add = ""
tl_add = ""

lts_pic_run_d = dict()
lts_pic_d = dict()
lts_a4x_d = dict()
dlts_d = dict()

t700_pic_d = dict()
t700_a4x_d = dict()
dt700_d = dict()

tas_pic_d = dict()
tas_a4x_d = dict()
dtas_d = dict()

cre_pic_d = dict()
cre_a4x_d = dict()
dcre_d = dict()

cre_lw_pic_d = dict()
cre_lw_a4x_d = dict()
dcre_lw_d = dict()

cre_sw_pic_d = dict()
cre_sw_a4x_d = dict()
dcre_sw_d = dict()

for g12, group in zip(["G1", "G2"], [models1, models2]):
    
    mods = []
    mod_count = 0

    # set up arrays to store all the feedback values --> we need this to easily calculate the multi-model mean
    lts_pic_run_l = []
    lts_pic_l = []
    lts_a4x_l = []
    dlts_l = []

    t700_pic_l = []
    t700_a4x_l = []
    dt700_l = []
    
    tas_pic_l = []
    tas_a4x_l = []
    dtas_l = []
    
    cre_pic_l = []
    cre_a4x_l = []
    dcre_l = []
    
    cre_lw_pic_l = []
    cre_lw_a4x_l = []
    dcre_lw_l = []
    
    cre_sw_pic_l = []
    cre_sw_a4x_l = []
    dcre_sw_l = []
    
    for i, mod_d in enumerate(group):
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            mod_dir = nl5.models[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
            si_varn = "sic"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_dir = nl6.models[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
            si_varn = "siconc"
        # end if        
        
        # set start and end index for the piControl experiment depending on the branch year
        bst = b_time * 12
        ben = bst + 150*12
        
        # load nc files
        lts_nc = Dataset(glob.glob(data_dir + f"Outputs/LTS/LTS_and_LTS_on_tas_LR/Thres_Year_20/*{mod_n}*.nc")[0])
        lts_pic_nc = Dataset(glob.glob(data_dir + f"Outputs/LTS/piControl/*{mod_n}*.nc")[0])
        tas_pic_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/tas_*piControl*.nc")[0])
        tas_a4x_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/tas_*{a4x}*.nc")[0])
        dtas_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/dtas_*.nc")[0])
        cre_pic_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/cre_*piControl*.nc")[0])
        cre_a4x_nc = Dataset(glob.glob(data_dir + f"Data/{mod_d}/cre_*{a4x}*.nc")[0])
        dcre_nc = Dataset(glob.glob(data_dir + f"Outputs/CRE/{a4x}/*{mod_d}.nc")[0])
        
        #% get lat and lon
        lat = lts_nc.variables["lat"][:]
        lon = lts_nc.variables["lon"][:]
        olat = lat + 90
        olon = lon + 0
        
        # extract region
        lts_pic = region_mean(lts_pic_nc.variables["lts"][:], x1, x2, y1, y2, lat, lon, test_plt=False, 
                              plot_title="LTS piC")
        lts_pic_run = region_mean(lts_nc.variables["eis_pic"][:], x1, x2, y1, y2, lat, lon, test_plt=False, 
                                  plot_title="LTS piC running")
        lts_a4x = region_mean(lts_nc.variables["eis_a4x"][:], x1, x2, y1, y2, lat, lon, test_plt=False, 
                               plot_title="LTS a4x")
        dlts = region_mean(lts_nc.variables["deis"][:], x1, x2, y1, y2, lat, lon)
        
        t700_pic = region_mean(np.mean(lts_nc.variables["ta_pic_700"][:], axis=0), x1, x2, y1, y2, lat, lon)
        t700_a4x = region_mean(an_mean(lts_nc.variables["ta_a4x_700"][:]), x1, x2, y1, y2, lat, lon)        
        tas_pic = region_mean(an_mean(tas_pic_nc.variables["tas"][bst:ben, :, :]), x1, x2, y1, y2, lat, lon)
        tas_a4x = region_mean(an_mean(tas_a4x_nc.variables["tas"][:150*12, :, :]), x1, x2, y1, y2, lat, lon)
        dtas = region_mean(np.mean(dtas_nc.variables["tas_ch"][:], axis=0), x1, x2, y1, y2, lat, lon)
        
        cre_lw_pic = region_mean(an_mean(cre_pic_nc.variables["cre_lw"][bst:ben, :, :]), x1, x2, y1, y2, lat, lon)
        cre_sw_pic = region_mean(an_mean(cre_pic_nc.variables["cre_sw"][bst:ben, :, :]), x1, x2, y1, y2, lat, lon)
        cre_lw_a4x = region_mean(an_mean(cre_a4x_nc.variables["cre_lw"][:150*12, :, :]), x1, x2, y1, y2, lat, lon)
        cre_sw_a4x = region_mean(an_mean(cre_a4x_nc.variables["cre_sw"][:150*12, :, :]), x1, x2, y1, y2, lat, lon)
        dcre_lw = region_mean(an_mean(dcre_nc.variables["dcre_lw"][:]), x1, x2, y1, y2, lat, lon)
        dcre_sw = region_mean(an_mean(dcre_nc.variables["dcre_sw"][:]), x1, x2, y1, y2, lat, lon)
        
        lts_pic_run_l.append(lts_pic_run)
        lts_pic_l.append(lts_pic)
        lts_a4x_l.append(lts_a4x)
        dlts_l.append(dlts)
        
        t700_pic_l.append(t700_pic)
        t700_a4x_l.append(t700_a4x)
        dt700_l.append(t700_a4x - t700_pic)
        
        tas_pic_l.append(tas_pic)
        tas_a4x_l.append(tas_a4x)
        dtas_l.append(dtas)
        
        cre_pic_l.append(cre_lw_pic + cre_sw_pic)
        cre_a4x_l.append(cre_lw_a4x + cre_sw_a4x)
        dcre_l.append(dcre_lw + dcre_sw)
        
        cre_lw_pic_l.append(cre_lw_pic)
        cre_lw_a4x_l.append(cre_lw_a4x)
        dcre_lw_l.append(dcre_lw)
        
        cre_sw_pic_l.append(cre_sw_pic)
        cre_sw_a4x_l.append(cre_sw_a4x)
        dcre_sw_l.append(dcre_sw)
        
        print(g12 + " " + str(i) + ": " + mod_pl)
        
        mods.append(mod_pl)
        
        mod_count += 1
    # end for i, mod_d
    
    print("\n")
    
    # print(f"{g12} shape: {np.shape(cre_lw_pic)}")

    print("\n\n")
    # store everything in dictionaries
    lts_pic_run_d[g12] = np.array(lts_pic_run_l)
    lts_pic_d[g12] = np.array(lts_pic_l)
    lts_a4x_d[g12] = np.array(lts_a4x_l)
    dlts_d[g12] = np.array(dlts_l)
    
    t700_pic_d[g12] = np.array(t700_pic_l)
    t700_a4x_d[g12] = np.array(t700_a4x_l)
    dt700_d[g12] = np.array(dt700_l)
    
    tas_pic_d[g12] = np.array(tas_pic_l)
    tas_a4x_d[g12] = np.array(tas_a4x_l)
    dtas_d[g12] = np.array(dtas_l)
    
    cre_pic_d[g12] = np.array(cre_pic_l)
    cre_a4x_d[g12] = np.array(cre_a4x_l)
    dcre_d[g12] = np.array(dcre_l)
    
    cre_lw_pic_d[g12] = np.array(cre_lw_pic_l)
    cre_lw_a4x_d[g12] = np.array(cre_lw_a4x_l)
    dcre_lw_d[g12] = np.array(dcre_lw_l)
    
    cre_sw_pic_d[g12] = np.array(cre_sw_pic_l)
    cre_sw_a4x_d[g12] = np.array(cre_sw_a4x_l)
    dcre_sw_d[g12] = np.array(dcre_sw_l)
    
    mods_d[g12] = np.array(mods)
    
# end for g12


#%% calculate the group averages

# annual means
lts_pic_run_g1 = np.mean(lts_pic_run_d["G1"], axis=0)
lts_pic_run_g2 = np.mean(lts_pic_run_d["G2"], axis=0)
lts_pic_g1 = np.mean(lts_pic_d["G1"], axis=0)
lts_pic_g2 = np.mean(lts_pic_d["G2"], axis=0)
lts_a4x_g1 = np.mean(lts_a4x_d["G1"], axis=0)
lts_a4x_g2 = np.mean(lts_a4x_d["G2"], axis=0)
dlts_g1 = np.mean(dlts_d["G1"], axis=0)
dlts_g2 = np.mean(dlts_d["G2"], axis=0)

t700_pic_g1 = np.mean(t700_pic_d["G1"], axis=0)
t700_pic_g2 = np.mean(t700_pic_d["G2"], axis=0)
t700_a4x_g1 = np.mean(t700_a4x_d["G1"], axis=0)
t700_a4x_g2 = np.mean(t700_a4x_d["G2"], axis=0)
dt700_g1 = np.mean(dt700_d["G1"], axis=0)
dt700_g2 = np.mean(dt700_d["G2"], axis=0)

tas_pic_g1 = np.mean(tas_pic_d["G1"], axis=0)
tas_pic_g2 = np.mean(tas_pic_d["G2"], axis=0)
tas_a4x_g1 = np.mean(tas_a4x_d["G1"], axis=0)
tas_a4x_g2 = np.mean(tas_a4x_d["G2"], axis=0)
dtas_g1 = np.mean(dtas_d["G1"], axis=0)
dtas_g2 = np.mean(dtas_d["G2"], axis=0)

cre_pic_g1 = np.mean(cre_pic_d["G1"], axis=0)
cre_pic_g2 = np.mean(cre_pic_d["G2"], axis=0)
cre_a4x_g1 = np.mean(cre_a4x_d["G1"], axis=0)
cre_a4x_g2 = np.mean(cre_a4x_d["G2"], axis=0)
dcre_g1 = np.mean(dcre_d["G1"], axis=0)
dcre_g2 = np.mean(dcre_d["G2"], axis=0)

cre_lw_pic_g1 = np.mean(cre_lw_pic_d["G1"], axis=0)
cre_lw_pic_g2 = np.mean(cre_lw_pic_d["G2"], axis=0)
cre_lw_a4x_g1 = np.mean(cre_lw_a4x_d["G1"], axis=0)
cre_lw_a4x_g2 = np.mean(cre_lw_a4x_d["G2"], axis=0)
dcre_lw_g1 = np.mean(dcre_lw_d["G1"], axis=0)
dcre_lw_g2 = np.mean(dcre_lw_d["G2"], axis=0)

cre_sw_pic_g1 = np.mean(cre_sw_pic_d["G1"], axis=0)
cre_sw_pic_g2 = np.mean(cre_sw_pic_d["G2"], axis=0)
cre_sw_a4x_g1 = np.mean(cre_sw_a4x_d["G1"], axis=0)
cre_sw_a4x_g2 = np.mean(cre_sw_a4x_d["G2"], axis=0)
dcre_sw_g1 = np.mean(dcre_sw_d["G1"], axis=0)
dcre_sw_g2 = np.mean(dcre_sw_d["G2"], axis=0)

lts_pic_g1_std = np.std(lts_pic_d["G1"], axis=0)
lts_pic_g2_std = np.std(lts_pic_d["G2"], axis=0)
lts_a4x_g1_std = np.std(lts_a4x_d["G1"], axis=0)
lts_a4x_g2_std = np.std(lts_a4x_d["G2"], axis=0)
dlts_g1_std = np.std(dlts_d["G1"], axis=0)
dlts_g2_std = np.std(dlts_d["G2"], axis=0)

t700_pic_g1_std = np.std(t700_pic_d["G1"], axis=0)
t700_pic_g2_std = np.std(t700_pic_d["G2"], axis=0)
t700_a4x_g1_std = np.std(t700_a4x_d["G1"], axis=0)
t700_a4x_g2_std = np.std(t700_a4x_d["G2"], axis=0)
dt700_g1_std = np.std(dt700_d["G1"], axis=0)
dt700_g2_std = np.std(dt700_d["G2"], axis=0)

tas_pic_g1_std = np.std(tas_pic_d["G1"], axis=0)
tas_pic_g2_std = np.std(tas_pic_d["G2"], axis=0)
tas_a4x_g1_std = np.std(tas_a4x_d["G1"], axis=0)
tas_a4x_g2_std = np.std(tas_a4x_d["G2"], axis=0)
dtas_g1_std = np.std(dtas_d["G1"], axis=0)
dtas_g2_std = np.std(dtas_d["G2"], axis=0)

cre_pic_g1_std = np.std(cre_pic_d["G1"], axis=0)
cre_pic_g2_std = np.std(cre_pic_d["G2"], axis=0)
cre_a4x_g1_std = np.std(cre_a4x_d["G1"], axis=0)
cre_a4x_g2_std = np.std(cre_a4x_d["G2"], axis=0)
dcre_g1_std = np.std(dcre_d["G1"], axis=0)
dcre_g2_std = np.std(dcre_d["G2"], axis=0)

cre_lw_pic_g1_std = np.std(cre_lw_pic_d["G1"], axis=0)
cre_lw_pic_g2_std = np.std(cre_lw_pic_d["G2"], axis=0)
cre_lw_a4x_g1_std = np.std(cre_lw_a4x_d["G1"], axis=0)
cre_lw_a4x_g2_std = np.std(cre_lw_a4x_d["G2"], axis=0)
dcre_lw_g1_std = np.std(dcre_lw_d["G1"], axis=0)
dcre_lw_g2_std = np.std(dcre_lw_d["G2"], axis=0)

cre_sw_pic_g1_std = np.std(cre_sw_pic_d["G1"], axis=0)
cre_sw_pic_g2_std = np.std(cre_sw_pic_d["G2"], axis=0)
cre_sw_a4x_g1_std = np.std(cre_sw_a4x_d["G1"], axis=0)
cre_sw_a4x_g2_std = np.std(cre_sw_a4x_d["G2"], axis=0)
dcre_sw_g1_std = np.std(dcre_sw_d["G1"], axis=0)
dcre_sw_g2_std = np.std(dcre_sw_d["G2"], axis=0)

lts_pic_ttest = ttest_ind(lts_pic_d["G1"], lts_pic_d["G2"], axis=0)
lts_a4x_ttest = ttest_ind(lts_a4x_d["G1"], lts_a4x_d["G2"], axis=0)
dlts_ttest = ttest_ind(dlts_d["G1"], dlts_d["G2"], axis=0)

t700_pic_ttest = ttest_ind(t700_pic_d["G1"], t700_pic_d["G2"], axis=0)
t700_a4x_ttest = ttest_ind(t700_a4x_d["G1"], t700_a4x_d["G2"], axis=0)
dt700_ttest = ttest_ind(dt700_d["G1"], dt700_d["G2"], axis=0)

tas_pic_ttest = ttest_ind(tas_pic_d["G1"], tas_pic_d["G2"], axis=0)
tas_a4x_ttest = ttest_ind(tas_a4x_d["G1"], tas_a4x_d["G2"], axis=0)
dtas_ttest = ttest_ind(dtas_d["G1"], dtas_d["G2"], axis=0)

cre_pic_ttest = ttest_ind(cre_pic_d["G1"], cre_pic_d["G2"], axis=0)
cre_a4x_ttest = ttest_ind(cre_a4x_d["G1"], cre_a4x_d["G2"], axis=0)
dcre_ttest = ttest_ind(dcre_d["G1"], dcre_d["G2"], axis=0)

cre_lw_pic_ttest = ttest_ind(cre_lw_pic_d["G1"], cre_lw_pic_d["G2"], axis=0)
cre_lw_a4x_ttest = ttest_ind(cre_lw_a4x_d["G1"], cre_lw_a4x_d["G2"], axis=0)
dcre_lw_ttest = ttest_ind(dcre_lw_d["G1"], dcre_lw_d["G2"], axis=0)

cre_sw_pic_ttest = ttest_ind(cre_sw_pic_d["G1"], cre_sw_pic_d["G2"], axis=0)
cre_sw_a4x_ttest = ttest_ind(cre_sw_a4x_d["G1"], cre_sw_a4x_d["G2"], axis=0)
dcre_sw_ttest = ttest_ind(dcre_sw_d["G1"], dcre_sw_d["G2"], axis=0)

cre_lw_pic_ttest = ttest_ind(cre_lw_pic_d["G1"], cre_lw_pic_d["G2"], axis=0)
cre_lw_a4x_ttest = ttest_ind(cre_lw_a4x_d["G1"], cre_lw_a4x_d["G2"], axis=0)
dcre_lw_ttest = ttest_ind(dcre_lw_d["G1"], dcre_lw_d["G2"], axis=0)

cre_sw_pic_ttest = ttest_ind(cre_sw_pic_d["G1"], cre_sw_pic_d["G2"], axis=0)
cre_sw_a4x_ttest = ttest_ind(cre_sw_a4x_d["G1"], cre_sw_a4x_d["G2"], axis=0)
dcre_sw_ttest = ttest_ind(dcre_sw_d["G1"], dcre_sw_d["G2"], axis=0)


#%% plot the time series - LTS (Fig. 3 in the new paper)
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(21.3, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(lts_pic_g1, c="blue")
axes[0].plot(lts_pic_g2, c="red")

axes[0].fill_between(np.arange(len(lts_pic_g1)), lts_pic_g1 - lts_pic_g1_std, lts_pic_g1 + lts_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(len(lts_pic_g2)), lts_pic_g2 - lts_pic_g2_std, lts_pic_g2 + lts_pic_g2_std, 
                     facecolor="red", alpha=0.25)

ax01.plot(lts_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0].set_xlabel("Year of simulation", fontsize=14)
axes[0].set_ylabel("LTS in K", fontsize=14)

axes[0].set_title("piControl (21-year running mean)", fontsize=16)


axes[1].plot(lts_a4x_g1, c="blue", label="G1")
axes[1].plot(lts_a4x_g2, c="red", label="G2")

axes[1].fill_between(np.arange(len(lts_a4x_g1)), lts_a4x_g1 - lts_a4x_g1_std, lts_a4x_g1 + lts_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(len(lts_a4x_g2)), lts_a4x_g2 - lts_a4x_g2_std, lts_a4x_g2 + lts_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

axes[1].legend(loc="upper center", fontsize=14)

ax11.plot(lts_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1].set_xlabel("Year of simulation", fontsize=14)
axes[1].set_ylabel("LTS in K", fontsize=14)

axes[1].set_title("abrupt4xCO2", fontsize=16)


axes[2].plot(dlts_g1, c="blue")
axes[2].plot(dlts_g2, c="red")

axes[2].fill_between(np.arange(len(dlts_g1)), dlts_g1 - dlts_g1_std, dlts_g1 + dlts_g1_std, facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(len(dlts_g2)), dlts_g2 - dlts_g2_std, dlts_g2 + dlts_g2_std, facecolor="red", alpha=0.25)

# ax21.plot(dlts_ttest.pvalue, linewidth=0.5, c="black")
# ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))
ax21.set_yticks([])
ax21.set_yticklabels(labels=[])

axes[2].set_xlabel("Year of simulation", fontsize=14)
axes[2].set_ylabel("LTS in K", fontsize=14)

axes[2].set_title("Change", fontsize=16)

axes[0].tick_params(axis='x', labelsize=13)
axes[0].tick_params(axis='y', labelsize=13)
axes[1].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)
axes[2].tick_params(axis='x', labelsize=13)
axes[2].tick_params(axis='y', labelsize=13)
ax01.tick_params(axis='x', labelsize=13)
ax01.tick_params(axis='y', labelsize=13)
ax11.tick_params(axis='x', labelsize=13)
ax11.tick_params(axis='y', labelsize=13)

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)

ax01.text(5, 0.68, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.68, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax21.text(5, 0.68, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"LTS {reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)", fontsize=17)
fig.subplots_adjust(wspace=0.35, top=0.85)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_LTS_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_LTS_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - temperature
fig, axes = pl.subplots(ncols=3, nrows=2, figsize=(21.3, 12))

ax00 = axes[0, 0].twinx()
ax01 = axes[0, 1].twinx()
ax02 = axes[0, 2].twinx()
ax10 = axes[1, 0].twinx()
ax11 = axes[1, 1].twinx()
ax12 = axes[1, 2].twinx()

axes[0, 0].plot(tas_pic_g1, c="blue")
axes[0, 0].plot(tas_pic_g2, c="red")

axes[0, 0].fill_between(np.arange(len(tas_pic_g1)), tas_pic_g1 - tas_pic_g1_std, tas_pic_g1 + tas_pic_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 0].fill_between(np.arange(len(tas_pic_g2)), tas_pic_g2 - tas_pic_g2_std, tas_pic_g2 + tas_pic_g2_std, 
                        facecolor="red", alpha=0.25)

ax00.plot(tas_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax00.axhline(y=0.05, c="grey", linewidth=0.5)
ax00.set_ylim((-0.01, 0.7))

axes[0, 0].set_xlabel("Year of simulation", fontsize=14)
axes[0, 0].set_ylabel("Tas in K", fontsize=14)

axes[0, 0].set_title("Tas piControl", fontsize=16)


axes[0, 1].plot(tas_a4x_g1, c="blue", label="G1")
axes[0, 1].plot(tas_a4x_g2, c="red", label="G2")

axes[0, 1].fill_between(np.arange(len(tas_a4x_g1)), tas_a4x_g1 - tas_a4x_g1_std, tas_a4x_g1 + tas_a4x_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 1].fill_between(np.arange(len(tas_a4x_g2)), tas_a4x_g2 - tas_a4x_g2_std, tas_a4x_g2 + tas_a4x_g2_std, 
                        facecolor="red", alpha=0.25)

ax01.plot(tas_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0, 1].legend(loc="lower center", fontsize=14)

axes[0, 1].set_xlabel("Year of simulation", fontsize=14)
axes[0, 1].set_ylabel("Tas in K", fontsize=14)

axes[0, 1].set_title("Tas abrupt4xCO2", fontsize=16)


axes[0, 2].plot(dtas_g1, c="blue", label="G1")
axes[0, 2].plot(dtas_g2, c="red", label="G2")

axes[0, 2].fill_between(np.arange(len(dtas_g1)), dtas_g1 - dtas_g1_std, dtas_g1 + dtas_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 2].fill_between(np.arange(len(dtas_g2)), dtas_g2 - dtas_g2_std, dtas_g2 + dtas_g2_std, 
                        facecolor="red", alpha=0.25)

ax02.plot(dtas_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax02.axhline(y=0.05, c="grey", linewidth=0.5)
ax02.set_ylim((-0.01, 0.7))

axes[0, 2].set_xlabel("Year of simulation", fontsize=14)
axes[0, 2].set_ylabel("dTas in K", fontsize=14)

axes[0, 2].set_title("dTas abrupt4xCO2", fontsize=16)


axes[1, 0].plot(t700_pic_g1, c="blue")
axes[1, 0].plot(t700_pic_g2, c="red")

axes[1, 0].fill_between(np.arange(len(t700_pic_g1)), t700_pic_g1 - t700_pic_g1_std, t700_pic_g1 + t700_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1, 0].fill_between(np.arange(len(t700_pic_g2)), t700_pic_g2 - t700_pic_g2_std, t700_pic_g2 + t700_pic_g2_std, 
                     facecolor="red", alpha=0.25)

ax10.plot(t700_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax10.axhline(y=0.05, c="grey", linewidth=0.5)
ax10.set_ylim((-0.01, 0.7))

axes[1, 0].set_xlabel("Year of simulation", fontsize=14)
axes[1, 0].set_ylabel("Ta 700 hPa in K", fontsize=14)

axes[1, 0].set_title("Ta 700 hPa piControl (21-year running)", fontsize=16)


axes[1, 1].plot(t700_a4x_g1, c="blue", label="G1")
axes[1, 1].plot(t700_a4x_g2, c="red", label="G2")

axes[1, 1].fill_between(np.arange(len(t700_a4x_g1)), t700_a4x_g1 - t700_a4x_g1_std, t700_a4x_g1 + t700_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1, 1].fill_between(np.arange(len(t700_a4x_g2)), t700_a4x_g2 - t700_a4x_g2_std, t700_a4x_g2 + t700_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

ax11.plot(t700_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1, 1].set_xlabel("Year of simulation", fontsize=14)
axes[1, 1].set_ylabel("Ta 700 hPa in K", fontsize=14)

axes[1, 1].set_title("Ta 700 hPa abrupt4xCO2", fontsize=16)


axes[1, 2].plot(dt700_g1, c="blue")
axes[1, 2].plot(dt700_g2, c="red")

axes[1, 2].fill_between(np.arange(len(dt700_g1)), dt700_g1 - dt700_g1_std, dt700_g1 + dt700_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1, 2].fill_between(np.arange(len(dt700_g2)), dt700_g2 - dt700_g2_std, dt700_g2 + dt700_g2_std, 
                     facecolor="red", alpha=0.25)

ax12.plot(dt700_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax12.axhline(y=0.05, c="grey", linewidth=0.5)
ax12.set_ylim((-0.01, 0.7))

axes[1, 2].set_xlabel("Year of simulation", fontsize=14)
axes[1, 2].set_ylabel("dTa 700 hPa in K", fontsize=14)

axes[1, 2].set_title("dTa 700 hPa abrupt4xCO2", fontsize=16)

axes[0, 0].tick_params(axis='x', labelsize=13)
axes[0, 0].tick_params(axis='y', labelsize=13)
axes[0, 1].tick_params(axis='x', labelsize=13)
axes[0, 1].tick_params(axis='y', labelsize=13)
axes[0, 2].tick_params(axis='x', labelsize=13)
axes[0, 2].tick_params(axis='y', labelsize=13)
axes[1, 0].tick_params(axis='x', labelsize=13)
axes[1, 0].tick_params(axis='y', labelsize=13)
axes[1, 1].tick_params(axis='x', labelsize=13)
axes[1, 1].tick_params(axis='y', labelsize=13)
axes[1, 2].tick_params(axis='x', labelsize=13)
axes[1, 2].tick_params(axis='y', labelsize=13)
ax00.tick_params(axis='x', labelsize=13)
ax00.tick_params(axis='y', labelsize=13)
ax01.tick_params(axis='x', labelsize=13)
ax01.tick_params(axis='y', labelsize=13)
ax02.tick_params(axis='x', labelsize=13)
ax02.tick_params(axis='y', labelsize=13)
ax10.tick_params(axis='x', labelsize=13)
ax10.tick_params(axis='y', labelsize=13)
ax11.tick_params(axis='x', labelsize=13)
ax11.tick_params(axis='y', labelsize=13)
ax12.tick_params(axis='x', labelsize=13)
ax12.tick_params(axis='y', labelsize=13)

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)

ax00.text(5, 0.68, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax01.text(5, 0.68, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax02.text(5, 0.68, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax10.text(5, 0.68, "(d)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.68, "(e)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax12.text(5, 0.68, "(f)", fontsize=14, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)", fontsize=17)
fig.subplots_adjust(top=0.92, wspace=0.35)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_tas_t700_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_tas_t700_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - CRE
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(18, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(cre_pic_g1, c="blue")
axes[0].plot(cre_pic_g2, c="red")

axes[0].fill_between(np.arange(len(cre_pic_g1)), cre_pic_g1 - cre_pic_g1_std, cre_pic_g1 + cre_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(len(cre_pic_g2)), cre_pic_g2 - cre_pic_g2_std, cre_pic_g2 + cre_pic_g2_std, 
                     facecolor="red", alpha=0.25)

ax01.plot(cre_pic_ttest.pvalue, linewidth=0.5, c="black")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0].set_xlabel("Year of simulation")
axes[0].set_ylabel("CRE in Wm$^{-2}$")

axes[0].set_title("CRE piControl")


axes[1].plot(cre_a4x_g1, c="blue", label="G1")
axes[1].plot(cre_a4x_g2, c="red", label="G2")

axes[1].fill_between(np.arange(len(cre_a4x_g1)), cre_a4x_g1 - cre_a4x_g1_std, cre_a4x_g1 + cre_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(len(cre_a4x_g2)), cre_a4x_g2 - cre_a4x_g2_std, cre_a4x_g2 + cre_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

axes[1].legend(loc="upper center")

ax11.plot(cre_a4x_ttest.pvalue, linewidth=0.5, c="black")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1].set_xlabel("Year of simulation")
axes[1].set_ylabel("CRE in Wm$^{-2}$")

axes[1].set_title("CRE abrupt4xCO2")


axes[2].plot(dcre_g1, c="blue")
axes[2].plot(dcre_g2, c="red")

axes[2].fill_between(np.arange(len(dcre_g1)), dcre_g1 - dcre_g1_std, dcre_g1 + dcre_g1_std, facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(len(dcre_g2)), dcre_g2 - dcre_g2_std, dcre_g2 + dcre_g2_std, facecolor="red", alpha=0.25)

axes[2].axhline(y=0.05, c="grey", linewidth=0.5)

ax21.plot(dcre_ttest.pvalue, linewidth=0.5, c="black")
ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))

axes[2].set_xlabel("Year of simulation")
axes[2].set_ylabel("CRE in Wm$^{-2}$")

axes[2].set_title("Change of CRE")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)")
fig.subplots_adjust(top=0.9, wspace=0.3)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_CRE_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_CRE_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - CRE LW
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(18, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(cre_lw_pic_g1, c="blue")
axes[0].plot(cre_lw_pic_g2, c="red")

axes[0].fill_between(np.arange(len(cre_lw_pic_g1)), cre_lw_pic_g1 - cre_lw_pic_g1_std, cre_lw_pic_g1 + cre_lw_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(len(cre_lw_pic_g2)), cre_lw_pic_g2 - cre_lw_pic_g2_std, cre_lw_pic_g2 + cre_lw_pic_g2_std, 
                     facecolor="red", alpha=0.25)

ax01.plot(cre_lw_pic_ttest.pvalue, linewidth=0.5, c="black")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0].set_xlabel("Year of simulation")
axes[0].set_ylabel("LW CRE in Wm$^{-2}$")

axes[0].set_title("LW CRE piControl")


axes[1].plot(cre_lw_a4x_g1, c="blue", label="G1")
axes[1].plot(cre_lw_a4x_g2, c="red", label="G2")

axes[1].fill_between(np.arange(len(cre_lw_a4x_g1)), cre_lw_a4x_g1 - cre_lw_a4x_g1_std, cre_lw_a4x_g1 + cre_lw_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(len(cre_lw_a4x_g2)), cre_lw_a4x_g2 - cre_lw_a4x_g2_std, cre_lw_a4x_g2 + cre_lw_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

axes[1].legend(loc="upper center")

ax11.plot(cre_lw_a4x_ttest.pvalue, linewidth=0.5, c="black")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1].set_xlabel("Year of simulation")
axes[1].set_ylabel("LW CRE in Wm$^{-2}$")

axes[1].set_title("LW CRE abrupt4xCO2")


axes[2].plot(dcre_lw_g1, c="blue")
axes[2].plot(dcre_lw_g2, c="red")

axes[2].fill_between(np.arange(len(dcre_lw_g1)), dcre_lw_g1 - dcre_lw_g1_std, dcre_lw_g1 + dcre_lw_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(len(dcre_lw_g2)), dcre_lw_g2 - dcre_lw_g2_std, dcre_lw_g2 + dcre_lw_g2_std, 
                     facecolor="red", alpha=0.25)

axes[2].axhline(y=0.05, c="grey", linewidth=0.5)

ax21.plot(dcre_lw_ttest.pvalue, linewidth=0.5, c="black")
ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))

axes[2].set_xlabel("Year of simulation")
axes[2].set_ylabel("LW CRE in Wm$^{-2}$")

axes[2].set_title("Change of LW CRE")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)")
fig.subplots_adjust(top=0.9, wspace=0.3)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_CRE_LW_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_CRE_LW_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the time series - CRE SW
fig, axes = pl.subplots(ncols=3, nrows=1, figsize=(18, 6))

ax01 = axes[0].twinx()
ax11 = axes[1].twinx()
ax21 = axes[2].twinx()

axes[0].plot(cre_sw_pic_g1, c="blue")
axes[0].plot(cre_sw_pic_g2, c="red")

axes[0].fill_between(np.arange(len(cre_sw_pic_g1)), cre_sw_pic_g1 - cre_sw_pic_g1_std, cre_sw_pic_g1 + cre_sw_pic_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[0].fill_between(np.arange(len(cre_sw_pic_g2)), cre_sw_pic_g2 - cre_sw_pic_g2_std, cre_sw_pic_g2 + cre_sw_pic_g2_std, 
                     facecolor="red", alpha=0.25)

ax01.plot(cre_sw_pic_ttest.pvalue, linewidth=0.5, c="black")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0].set_xlabel("Year of simulation")
axes[0].set_ylabel("SW CRE in Wm$^{-2}$")

axes[0].set_title("SW CRE piControl")


axes[1].plot(cre_sw_a4x_g1, c="blue", label="G1")
axes[1].plot(cre_sw_a4x_g2, c="red", label="G2")

axes[1].fill_between(np.arange(len(cre_sw_a4x_g1)), cre_sw_a4x_g1 - cre_sw_a4x_g1_std, cre_sw_a4x_g1 + cre_sw_a4x_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[1].fill_between(np.arange(len(cre_sw_a4x_g2)), cre_sw_a4x_g2 - cre_sw_a4x_g2_std, cre_sw_a4x_g2 + cre_sw_a4x_g2_std, 
                     facecolor="red", alpha=0.25)

axes[1].legend(loc="upper center")

ax11.plot(cre_sw_a4x_ttest.pvalue, linewidth=0.5, c="black")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[1].set_xlabel("Year of simulation")
axes[1].set_ylabel("SW CRE in Wm$^{-2}$")

axes[1].set_title("SW CRE abrupt4xCO2")


axes[2].plot(dcre_sw_g1, c="blue")
axes[2].plot(dcre_sw_g2, c="red")

axes[2].fill_between(np.arange(len(dcre_sw_g1)), dcre_sw_g1 - dcre_sw_g1_std, dcre_sw_g1 + dcre_sw_g1_std, 
                     facecolor="blue", alpha=0.25)
axes[2].fill_between(np.arange(len(dcre_sw_g2)), dcre_sw_g2 - dcre_sw_g2_std, dcre_sw_g2 + dcre_sw_g2_std, 
                     facecolor="red", alpha=0.25)

axes[2].axhline(y=0.05, c="grey", linewidth=0.5)

ax21.plot(dcre_sw_ttest.pvalue, linewidth=0.5, c="black")
ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))

axes[2].set_xlabel("Year of simulation")
axes[2].set_ylabel("SW CRE in Wm$^{-2}$")

axes[2].set_title("Change of SW CRE")

fig.suptitle(f"{reg_name_title[reg_name]} " + 
             f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)")
fig.subplots_adjust(top=0.9, wspace=0.3)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_CRE_SW_G1_G2_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_CRE_SW_G1_G2_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% correlate stability with (SW) CRE
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.scatter(lts_a4x_g1, cre_a4x_g1, marker="o", c="blue")
axes.scatter(lts_a4x_g2, cre_a4x_g2, marker="o", c="red")

axes.set_xlabel("LTS in K")
axes.set_ylabel("CRE in Wm$^{-2}$ (positive-down)")
axes.set_title("CRE v. LTS $-$ G1 and G2")

pl.show()
pl.close()


#%% correlate stability with (SW) CRE - members
g12 = "G2"
mem = 8
colours = {"G1":"blue", "G2":"red"}

sl, yi, r_v, p_v, err = lr(lts_a4x_d[g12][mem], cre_sw_a4x_d[g12][mem])

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

# axes.scatter(lts_a4x_d[g12][mem], cre_a4x_d[g12][mem], marker="o", c=colours[g12])
axes.scatter(lts_a4x_d[g12][mem], cre_sw_a4x_d[g12][mem], marker="o", c=colours[g12])
axes.plot(lts_a4x_d[g12][mem], lts_a4x_d[g12][mem] * sl + yi, c="black", label=f"sl={r2(sl)} $\pm$ {r2(err)}, " +
          f"R={r3(r_v)}, p={r4(p_v)}")
axes.legend()

axes.set_xlabel("LTS in K")
axes.set_ylabel("CRE in Wm$^{-2}$ (positive-down)")
axes.set_title(f"CRE v. LTS $-$ {mods_d[g12][mem]} ({g12})")

pl.show()
pl.close()


#%% plot the time series - LTS (Fig. 3 in the new paper) - Version 2 (4 panels)
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(13, 9))

ax01 = axes[0, 0].twinx()
ax11 = axes[0, 1].twinx()
ax21 = axes[1, 0].twinx()
ax31 = axes[1, 1].twinx()


axes[0, 0].plot(lts_pic_g1, c="blue")
axes[0, 0].plot(lts_pic_g2, c="red")

axes[0, 0].fill_between(np.arange(len(lts_pic_g1)), lts_pic_g1 - lts_pic_g1_std, lts_pic_g1 + lts_pic_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 0].fill_between(np.arange(len(lts_pic_g2)), lts_pic_g2 - lts_pic_g2_std, lts_pic_g2 + lts_pic_g2_std, 
                        facecolor="red", alpha=0.25)

ax01.plot(lts_pic_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax01.axhline(y=0.05, c="grey", linewidth=0.5)
ax01.set_ylim((-0.01, 0.7))

axes[0, 0].set_ylabel("LTS in K", fontsize=14)

axes[0, 0].set_title("piControl (21-year running mean)", fontsize=16)


axes[0, 1].plot(lts_a4x_g1, c="blue", label="G1")
axes[0, 1].plot(lts_a4x_g2, c="red", label="G2")

axes[0, 1].fill_between(np.arange(len(lts_a4x_g1)), lts_a4x_g1 - lts_a4x_g1_std, lts_a4x_g1 + lts_a4x_g1_std, 
                        facecolor="blue", alpha=0.25)
axes[0, 1].fill_between(np.arange(len(lts_a4x_g2)), lts_a4x_g2 - lts_a4x_g2_std, lts_a4x_g2 + lts_a4x_g2_std, 
                        facecolor="red", alpha=0.25)

axes[0, 1].legend(loc="upper right", fontsize=13, ncol=2)

ax11.plot(lts_a4x_ttest.pvalue, linewidth=0.5, c="black", linestyle=":")
ax11.axhline(y=0.05, c="grey", linewidth=0.5)
ax11.set_ylim((-0.01, 0.7))

axes[0, 1].set_ylabel("LTS in K", fontsize=14)

axes[0, 1].set_title("abrupt4xCO2", fontsize=16)


axes[1, 0].plot(dlts_g1, c="blue")
axes[1, 0].plot(dlts_g2, c="red")

axes[1, 0].fill_between(np.arange(len(dlts_g1)), dlts_g1 - dlts_g1_std, dlts_g1 + dlts_g1_std, facecolor="blue", 
                        alpha=0.25)
axes[1, 0].fill_between(np.arange(len(dlts_g2)), dlts_g2 - dlts_g2_std, dlts_g2 + dlts_g2_std, facecolor="red", 
                        alpha=0.25)

# ax21.plot(dlts_ttest.pvalue, linewidth=0.5, c="black")
# ax21.axhline(y=0.05, c="grey", linewidth=0.5)
ax21.set_ylim((-0.01, 0.7))
ax21.set_yticks([])
ax21.set_yticklabels(labels=[])

ax31.set_ylim((-0.01, 0.7))
ax31.set_yticks([])
ax31.set_yticklabels(labels=[])

axes[1, 0].set_xlabel("Year of simulation", fontsize=14)
axes[1, 0].set_ylabel("LTS in K", fontsize=14)

axes[1, 0].set_title("Change", fontsize=16)


axes[1, 1].plot(dlts_g2 - dlts_g1, c="black", label="change")
axes[1, 1].plot(lts_a4x_g2 - lts_a4x_g1, c="black", linestyle="--", label="abrupt4xCO2")
axes[1, 1].plot(lts_pic_g2 - lts_pic_g1, c="black", linestyle=":", label="piControl")
axes[1, 1].axhline(y=0.05, c="grey", linewidth=0.5)

axes[1, 1].legend(fontsize=13, ncol=1)

axes[1, 1].set_xlabel("Year of simulation", fontsize=14)
axes[1, 1].set_title("Group difference (G2 $-$ G1)", fontsize=16)

axes[0, 0].tick_params(axis='x', labelsize=13)
axes[0, 0].tick_params(axis='y', labelsize=13)
axes[0, 1].tick_params(axis='x', labelsize=13)
axes[0, 1].tick_params(axis='y', labelsize=13)
axes[1, 0].tick_params(axis='x', labelsize=13)
axes[1, 0].tick_params(axis='y', labelsize=13)
axes[1, 1].tick_params(axis='x', labelsize=13)
axes[1, 1].tick_params(axis='y', labelsize=13)
ax01.tick_params(axis='x', labelsize=13)
ax01.tick_params(axis='y', labelsize=13)
ax11.tick_params(axis='x', labelsize=13)
ax11.tick_params(axis='y', labelsize=13)

ax01.set_ylabel("$p$ value", fontsize=14)
ax11.set_ylabel("$p$ value", fontsize=14)

ax01.text(5, 0.68, "(a)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax11.text(5, 0.68, "(b)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax21.text(5, 0.68, "(c)", fontsize=14, horizontalalignment="center", verticalalignment="center")
ax31.text(5, 0.68, "(d)", fontsize=14, horizontalalignment="center", verticalalignment="center")

# fig.suptitle(f"LTS G1$-$G2 comparison {reg_name_title[reg_name]}\n" + 
#              f"({y1[0]}$-${y2[0]}$\degree$N, {x1[0]}$-${x2[0]}$\degree$E)", fontsize=17)
fig.suptitle(f"LTS G1$-$G2 comparison {reg_name_title[reg_name]}\n" + 
             f"({y1[0]}$-${y2[0]}$\degree$N, 60$-$10$\degree$W)", fontsize=17)
fig.subplots_adjust(wspace=0.35, top=0.88)

pl.savefig(pl_path + f"/PDF/{reg_name_fn[reg_name]}_LTS_G1_G2_Comp_V2.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + f"/PNG/{reg_name_fn[reg_name]}_LTS_G1_G2_Comp_V2.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()