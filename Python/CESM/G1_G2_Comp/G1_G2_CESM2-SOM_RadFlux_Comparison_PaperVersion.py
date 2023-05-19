"""
Compare the differences of multiple "sub-parts" of the Northern Hemisphere between G1 and G2 for kernel-derived radiative
fluxes and feedbacks.

Generates Fig. 6, 7, 8, 9, and 10 in Eiselt and Graversen (2023), JCLI.

Be sure to set data_dir5 and data_dir6 correctly and set and pl_path. Also set data_path and flux_path further below.
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
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Region_Mean import region_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set feedback
fb = "WV"
cs = 0  # 1: True  0: False


#%% set the region
### --> Subpolar North Atlantic (SPNA) in Bellomo et al. (2021):  50-70N, 80W-10E

# set the region name
reg_name = "2"

reg_name_title = {"1":"Mid-latitude North Atlantic", "2":"Arctic", "3":"Northern Polar", "4":"Northern Extratropics", 
                  "5":"Northern Hemisphere"}
reg_name_fn = {"1":"MLNA", "2":"Arctic", "3":"NHP", "4":"NHExT", "5":"NH"}


# MLNA
x1, x2 = [300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [350, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360]
y1, y2 = [40, 75, 60, 30, 0, 0, -90, -30, -90, -90, -90], [60, 90, 90, 90, 30, 90, 0, 0, -30, -60, -75]

# Arctic
x12, x22 = [0], [360]
y12, y22 = [75], [90]

# Northern Hemisphere Polar
x13, x23 = [0], [360]
y13, y23 = [60], [90]

# Northern Hemisphere Extratropics
x14, x24 = [0], [360]
y14, y24 = [30], [90]

# Northern Tropics
x15, x25 = [0], [360]
y15, y25 = [0], [30]

# Northern Hemisphere
x16, x26 = [0], [360]
y16, y26 = [0], [90]

# Southern Hemisphere
x17, x27 = [0], [360]
y17, y27 = [-90], [0]

# Southern Tropics
x18, x28 = [0], [360]
y18, y28 = [-30], [0]

# Southern Extratropics
x19, x29 = [0], [360]
y19, y29 = [-90], [-30]

# Southern Polar
x110, x210 = [0], [360]
y110, y210 = [-90], [-60]

# Antarctica
x111, x211 = [0], [360]
y111, y211 = [-90], [-75]



#%% set model groups
models1 = ["CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]  # AMOC models
models2 = ["NorESM1_M", "ACCESS-ESM1-5", "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]               # AMOC models   

models1 = ["IPSL_CM5A_LR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1", "E3SM-1-0", "MPI-ESM1-2-LR", "UKESM1-0-LL"]
models2 = ["BCC_CSM1_1", "BCC_CSM1_1_M", "GFDL_CM3", "GISS_E2_H", "ACCESS-ESM1-5", "NorESM1_M", "BCC-CSM2-MR", 
           "EC-Earth3", "FGOALS-g3", "INM-CM4-8"]

models = {"G1":models1, "G2":models2}


#%% radiative kernels
kl = "Sh08"


#%% feedback dictionaries
fb_pn = {"SA":"SfcAlb", "T":"T", "Pl":"T", "LR":"T", "WVLW":"Q", "WVSW":"Q", "WV":"Q", "C":"Cloud", "CLW":"Cloud", 
         "CSW":"Cloud"}
fb_fn = {"SA":"SfcAlb", "T":"ta_ts_lr", "Pl":"ta_ts_lr", "LR":"ta_ts_lr", "WVLW":"q", "WVSW":"q", "WV":"q", "C":"Cloud", 
         "CLW":"Cloud", "CSW":"Cloud"}
adj_fn = {"SA":"sa", "Pl":"ta_ts", "T":"ta_ts", "WVLW":"q", "WVSW":"q", "WV":"Q"}
fb_varn = {"SA":"sa_resp", "T":"t_resp", "Pl":["t_resp", "lr_resp"], "LR":"lr_resp", "WVLW":"q_lw_resp", 
           "WVSW":"q_sw_resp", "WV":["q_lw_resp", "q_sw_resp"], "C":"c_resp", "CLW":"c_lw_resp", "CSW":"c_sw_resp"}
adj_varn = {"SA":"sa_adj", "T":"t_adj", "WVLW":"q_lw_adj", "WVSW":"q_sw_adj", "WV":["q_lw_adj", "q_sw_adj"]}

fb_title = {"SA":"surface-albedo", "Pl":"Planck", "LR":"lapse-rate", "WVLW":"long-wave water-vapour", 
            "WVSW":"short-wave water-vapour", "WV":"water-vapour", "T":"temperature",
            "C":"cloud", "CLW":"long-wave cloud", "CSW":"short-wave cloud"}

# introduce a sign dictionary (only for the test plots) --> -1 for the positive down LW kernels, +1 for positive up
sig = {"Sh08":1, "So08":1, "BM13":1, "H17":1, "P18":1, "S18":1}
if fb in ["LR", "Pl", "T", "WVLW", "WV"]:
    sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}
# end if

# clear-sky path add
cs_path = ""
cs_fn = ""
if cs:
    cs_path = "/CS/"
    cs_fn = "_CS"
# end if    

cs_add = ""
cs_title = ""
if cs:
    cs_add = "_CS"
    cs_title = " clear-sky"
# end if


#%% load model lists to check if a particular model is a member of CMIP5 or 6
import Namelists.Namelist_CMIP5 as nl5
import Namelists.Namelist_CMIP6 as nl6
import Namelists.CMIP_Dictionaries as cmip_dic
data_dir5 = ""  # SETS BASIC DATA DIRECTORY
data_dir6 = ""  # SETS BASIC DATA DIRECTORY

cmip5_list = nl5.models
cmip6_list = nl6.models

pl_path = ""

os.makedirs(pl_path + "/PDF/", exist_ok=True)
os.makedirs(pl_path + "/PNG/", exist_ok=True)


#%% kernel dictionaries
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018"}


#%% load the CanESM5 grid as target for regridding
can_nc = Dataset(glob.glob(data_dir6 + "/CMIP6/Data/CanESM5/ts_*4xCO2*.nc")[0])
tlat = can_nc.variables["lat"][:] + 90 
tlon = can_nc.variables["lon"][:]
pllat = can_nc.variables["lat"][:]
pllon = can_nc.variables["lon"][:]


#%% loop over group one
fl_regi_dic = {}
fl_glob_dic = {}  # global

dtas_d = dict()
dtas_gl_d = dict()


for g_n, group in zip(["G1", "G2"], [models1, models2]):
    
    fl_regi_l = []
    fl_glob_l = []  # global
    dtas_gl_l = []
    
    for i, mod_d in enumerate(group):
        
        print(f"\n{g_n} {mod_d}\n")
        
        if mod_d in cmip5_list:
            data_dir = data_dir5
            cmip = "5"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl5.models_pl[mod]
            mod_n = nl5.models_n[mod]
            mod_p = nl5.models[mod]
            b_time = nl5.b_times[mod_d]
            a4x = "abrupt4xCO2"
        elif mod_d in cmip6_list:
            data_dir = data_dir6
            cmip = "6"
            mod = cmip_dic.mod_dict[cmip][mod_d]
            mod_pl = nl6.models_pl[mod]
            mod_n = nl6.models_n[mod]
            mod_p = nl6.models[mod]
            b_time = nl6.b_times[mod_d]        
            a4x = "abrupt-4xCO2"
        # end if
        
        # data path and name
        flux_path = data_dir + f"/CMIP{cmip}/Outputs/Kernel_TOA_RadResp/{a4x}/{k_p[kl]}/{fb_pn[fb]}_Response/" + cs_path
        f_name_rflx = f"TOA_RadResponse_{fb_fn[fb]}_{kl}_Kernel*_{mod_p}.nc"
        
        # load the files
        rflx_nc = Dataset(glob.glob(flux_path + f_name_rflx)[0])
        dtas_nc = Dataset(glob.glob(data_dir + f"/CMIP{cmip}/Data/{mod_d}/dtas_*{a4x}*.nc")[0])
        
        # get lat and lon
        lat = rflx_nc.variables["lat"][:]
        lon = rflx_nc.variables["lon"][:]
        olat = lat + 90
        olon = lon + 0

        # load the ts grid
        lat2d = dtas_nc.variables["lat"][:]
        lon2d = dtas_nc.variables["lon"][:]
        
        # load the data
        if fb == "WV":
            rflx_temp = (sig[kl] * an_mean(rflx_nc.variables[fb_varn[fb][0]][:150*12, :, :]) + 
                         an_mean(rflx_nc.variables[fb_varn[fb][1]][:150*12, :, :]))
        elif fb == "Pl":
            rflx_temp = sig[kl] * (an_mean(rflx_nc.variables[fb_varn[fb][0]][:150*12, :, :]) - 
                                   an_mean(rflx_nc.variables[fb_varn[fb][1]][:150*12, :, :]))
        else:    
            rflx_temp = sig[kl] * an_mean(rflx_nc.variables[fb_varn[fb]][:150*12, :, :])
        # end if elif else
        
        fl_regi_l.append(region_mean(rflx_temp, x1, x2, y1, y2, lat, lon, test_plt=False, plot_title="fl", 
                                     multi_reg_mean=False)) 
        fl_glob_l.append(glob_mean(rflx_temp, lat, lon)) 
        
        dtas_gl = glob_mean(np.mean(dtas_nc.variables["tas_ch"][:], axis=0), lat2d, lon2d)

        dtas_gl_l.append(dtas_gl)
                
    # end for i, mod_d
    
    fl_regi_dic[g_n] = np.array(fl_regi_l)
    fl_glob_dic[g_n] = np.array(fl_glob_l)
    dtas_gl_d[g_n] = np.array(dtas_gl_l)

    
# end for g_n, group


#%% calculate the group mean

# MLNA
g1_fl_regi = np.mean(fl_regi_dic["G1"], axis=0)
g2_fl_regi = np.mean(fl_regi_dic["G2"], axis=0)

g1_fl_regi_std = np.std(fl_regi_dic["G1"], axis=0)
g2_fl_regi_std = np.std(fl_regi_dic["G2"], axis=0)

dg12_fl_regi = g2_fl_regi - g1_fl_regi

fl_regi_ttest = []
for i in np.arange(len(x1)):
    fl_regi_ttest.append(ttest_ind(fl_regi_dic["G1"][:, i, :], fl_regi_dic["G2"][:, i, :], axis=0))
# end for i

# global
g1_fl_glob = np.mean(fl_glob_dic["G1"], axis=0)
g2_fl_glob = np.mean(fl_glob_dic["G2"], axis=0)

g1_fl_glob_std = np.std(fl_glob_dic["G1"], axis=0)
g2_fl_glob_std = np.std(fl_glob_dic["G2"], axis=0)

dg12_fl_glob = g2_fl_glob - g1_fl_glob

fl_glob_ttest = ttest_ind(fl_glob_dic["G1"], fl_glob_dic["G2"], axis=0)


# global tas
dtas_gl_g1 = np.mean(dtas_gl_d["G1"], axis=0)
dtas_gl_g2 = np.mean(dtas_gl_d["G2"], axis=0)



#%% LOAD CESM2-SOM data: setup case name --------------------------------------------------------------------------------
case_4x51 = "Proj2_KUE_4xCO2"
case_4x41 = "Yr41_4xCO2"
case_4x61 = "Proj2_KUE_4xCO2_61"

case_dq4x51 = "dQ01yr_4xCO2"
case_dq4x41 = "Y41_dQ01"
case_dq4x61 = "Y61_dQ01"

# case_dq4x51 = "dQ01_r30_4xCO2"
# case_dq4x41 = "dQ01_r30_4xCO2_41"
# case_dq4x61 = "dQ01_r30_4xCO2_61"


#%% clear-sky?
cs = 0


#%% set the region
reg_name_title = {"1":"Mid-latitude North Atlantic", 
                  "2":"Arctic", 
                  "3":"Northern Polar", 
                  "4":"Northern Extratropics", 
                  "5":"Northern Tropics", 
                  "6":"Northern Hemisphere", 
                  "7":"Southern Hemisphere", 
                  "8":"Southern Tropics", 
                  "9":"Southern Extratropics", 
                  "10":"Southern Polar",
                  "11":"Antarctica",
                  "12":"Global"}
reg_name_fn = {"1":"MLNA", 
               "2":"Arctic", 
               "3":"NP", 
               "4":"NExT",
               "5":"NT",
               "6":"NH", 
               "7":"SH", 
               "8":"ST", 
               "9":"SExT", 
               "10":"SP",
               "11":"Antarctica", 
               "12":"Global"}

region_n = reg_name_fn[reg_name]


#%% set paths
data_path = ""
flux_path = f"/Rad_Kernel/Kernel_TOA_RadResp/{k_p[kl]}/{fb_pn[fb]}_Response/"


#%% load the kernel-derived radiative fluxes
flux_4x51_nc = Dataset(data_path + f"/{case_4x51}/" + flux_path + 
                       f"TOA_RadResponse_{fb_fn[fb]}_{case_4x51}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")
flux_4x41_nc = Dataset(data_path + f"/{case_4x41}/" + flux_path + 
                       f"TOA_RadResponse_{fb_fn[fb]}_{case_4x41}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")
flux_4x61_nc = Dataset(data_path + f"/{case_4x61}/" + flux_path + 
                       f"TOA_RadResponse_{fb_fn[fb]}_{case_4x61}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")

flux_dq4x51_nc = Dataset(data_path + f"/{case_dq4x51}/" + flux_path + 
                         f"TOA_RadResponse_{fb_fn[fb]}_{case_dq4x51}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")
flux_dq4x41_nc = Dataset(data_path + f"/{case_dq4x41}/" + flux_path + 
                         f"TOA_RadResponse_{fb_fn[fb]}_{case_dq4x41}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")
flux_dq4x61_nc = Dataset(data_path + f"/{case_dq4x61}/" + flux_path + 
                         f"TOA_RadResponse_{fb_fn[fb]}_{case_dq4x61}_{kl}_Kernel{cs_add}_CESM2-SOM.nc")


#%% load lat and lon
lat = flux_4x51_nc.variables["lat"][:]
lon = flux_4x51_nc.variables["lon"][:]


#%% loop over the 5 regions
nyrs = 40

flux_dic = {}
avflux_dic = {}
dtas_dic = {}
avdtas_dic = {}
flux_dq_dic = {}
avflux_dq_dic = {}
dtas_dq_dic = {}
avdtas_dq_dic = {}

for reg_name in reg_name_title.keys():
    # MLNA
    if reg_name == "1":
        x1, x2 = [300], [350]
        y1, y2 = [40], [60]
    
    # Arctic
    elif reg_name == "2":
        x1, x2 = [0], [360]
        y1, y2 = [75], [90]
    
    # North Hemisphere Polar
    elif reg_name == "3":
        x1, x2 = [0], [360]
        y1, y2 = [60], [90]
    
    # North Hemisphere Extratropics
    elif reg_name == "4":
        x1, x2 = [0], [360]
        y1, y2 = [30], [90]
    
    # Northern Tropics
    elif reg_name == "5":
        x1, x2 = [0], [360]
        y1, y2 = [0], [30]
        
    # North Hemisphere
    elif reg_name == "6":
        x1, x2 = [0], [360]
        y1, y2 = [0], [90]    
    
    # South Hemisphere
    elif reg_name == "7":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [0]
    
    # Southern Tropics
    elif reg_name == "8":
        x1, x2 = [0], [360]
        y1, y2 = [-30], [0]
    
    # Southern Extratropocs
    elif reg_name == "9":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-30]
    
    # Southern Polar
    elif reg_name == "10":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-60]        
        
    # Antarctica    
    elif reg_name == "11":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [-75]            
    
    # Global
    elif reg_name == "12":
        x1, x2 = [0], [360]
        y1, y2 = [-90], [90]            
    # end if elif
    
    flux_dic[reg_name_fn[reg_name]] = np.zeros((3, nyrs))
    flux_dq_dic[reg_name_fn[reg_name]] = np.zeros((3, nyrs))
   

    # load the radiative fluxes
    if fb == "WV":
        flux_4x51 = (sig[kl] * np.mean(flux_4x51_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                     np.mean(flux_4x51_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_4x41 = (sig[kl] * np.mean(flux_4x41_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                     np.mean(flux_4x41_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_4x61 = (sig[kl] * np.mean(flux_4x61_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                     np.mean(flux_4x61_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        
        flux_dq4x51 = (sig[kl] * np.mean(flux_dq4x51_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                       np.mean(flux_dq4x51_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_dq4x41 = (sig[kl] * np.mean(flux_dq4x41_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                       np.mean(flux_dq4x41_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_dq4x61 = (sig[kl] * np.mean(flux_dq4x61_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) +
                       np.mean(flux_dq4x61_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
    elif fb == "Pl":
        flux_4x51 = (sig[kl] * np.mean(flux_4x51_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                     sig[kl] * np.mean(flux_4x51_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_4x41 = (sig[kl] * np.mean(flux_4x41_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                     sig[kl] * np.mean(flux_4x41_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_4x61 = (sig[kl] * np.mean(flux_4x61_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                     sig[kl] * np.mean(flux_4x61_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        
        flux_dq4x51 = (sig[kl] * np.mean(flux_dq4x51_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                       sig[kl] * np.mean(flux_dq4x51_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_dq4x41 = (sig[kl] * np.mean(flux_dq4x41_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                       sig[kl] * np.mean(flux_dq4x41_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
        flux_dq4x61 = (sig[kl] * np.mean(flux_dq4x61_nc.variables[fb_varn[fb][0]][:nyrs, :, :, :], axis=1) -
                       sig[kl] * np.mean(flux_dq4x61_nc.variables[fb_varn[fb][1]][:nyrs, :, :, :], axis=1))
    else:    
        flux_4x51 = sig[kl] * np.mean(flux_4x51_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means
        flux_4x41 = sig[kl] * np.mean(flux_4x41_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means
        flux_4x61 = sig[kl] * np.mean(flux_4x61_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means
        
        flux_dq4x51 = sig[kl] * np.mean(flux_dq4x51_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means
        flux_dq4x41 = sig[kl] * np.mean(flux_dq4x41_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means
        flux_dq4x61 = sig[kl] * np.mean(flux_dq4x61_nc.variables[fb_varn[fb]][:nyrs, :, :, :], axis=1)  # annual means

    # end if else
        
    # calculate global and regional fluxes
    flux_dic[reg_name_fn[reg_name]][0, :] = region_mean(flux_4x51, x1, x2, y1, y2, lat, lon)
    flux_dic[reg_name_fn[reg_name]][1, :] = region_mean(flux_4x41, x1, x2, y1, y2, lat, lon)
    flux_dic[reg_name_fn[reg_name]][2, :] = region_mean(flux_4x61, x1, x2, y1, y2, lat, lon)
    avflux_dic[reg_name_fn[reg_name]] = np.mean(flux_dic[reg_name_fn[reg_name]], axis=0)
    
    flux_dq_dic[reg_name_fn[reg_name]][0, :] = region_mean(flux_dq4x51, x1, x2, y1, y2, lat, lon)
    flux_dq_dic[reg_name_fn[reg_name]][1, :] = region_mean(flux_dq4x41, x1, x2, y1, y2, lat, lon)
    flux_dq_dic[reg_name_fn[reg_name]][2, :] = region_mean(flux_dq4x61, x1, x2, y1, y2, lat, lon)
    avflux_dq_dic[reg_name_fn[reg_name]] = np.mean(flux_dq_dic[reg_name_fn[reg_name]], axis=0)

# end for reg_name


#%% calculate the p-values for the CESM2-SOM experiments (only possible if multiple ensembles exist)
fl_dq_ttest = dict()

for name in flux_dq_dic.keys():
    fl_dq_ttest[name] = ttest_ind(flux_dq_dic[name], flux_dic[name], axis=0)
# end for name    


#%% plot the inter-group differences in radiative flux in one panel
fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=False)

p75n0, = axes[0, 0].plot(dg12_fl_regi[1, :], c="blue", label="R>75N")
p60n0, = axes[0, 0].plot(dg12_fl_regi[2, :], c="orange", label="R>60N")
p30n0, = axes[0, 0].plot(dg12_fl_regi[3, :], c="red", label="R>30N")
ps30n0, = axes[0, 0].plot(dg12_fl_regi[4, :], c="violet", label="R<30N")
p0n0, = axes[0, 0].plot(dg12_fl_regi[5, :], c="gray", label="R>0N")

p75s0, = axes[0, 1].plot(dg12_fl_regi[10, :], c="blue", linestyle="-", label="R>75S")
p60s0, = axes[0, 1].plot(dg12_fl_regi[9, :], c="orange", linestyle="-", label="R>60S")
p30s0, = axes[0, 1].plot(dg12_fl_regi[8, :], c="red", linestyle="-", label="R>30S")
ps30s0, = axes[0, 1].plot(dg12_fl_regi[7, :], c="violet", linestyle="-", label="R<30S")
p0s0, = axes[0, 1].plot(dg12_fl_regi[6, :], c="gray", linestyle="-", label="R>0S")

pgln0, = axes[0, 0].plot(dg12_fl_glob, c="black", linestyle="-", label="global")
pgls0, = axes[0, 1].plot(dg12_fl_glob, c="black", linestyle="-", label="global")

axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)

axes[0, 0].set_ylabel("Radiative flux in Wm$^{-2}$")

axes[0, 0].set_title("Northern Hemisphere $-$ G2 minus G1")
axes[0, 1].set_title("Southern Hemisphere $-$ G2 minus G1")


p75n1, = axes[1, 0].plot(avflux_dq_dic["Arctic"] - avflux_dic["Arctic"], c="blue", label="R>75N")
p60n1, = axes[1, 0].plot(avflux_dq_dic["NP"] - avflux_dic["NP"], c="orange", label="R>60N")
p30n1, = axes[1, 0].plot(avflux_dq_dic["NExT"] - avflux_dic["NExT"], c="red", label="R>30N")
ps30n1, = axes[1, 0].plot(avflux_dq_dic["NT"] - avflux_dic["NT"], c="violet", label="R<30N")
p0n1, = axes[1, 0].plot(avflux_dq_dic["NH"] - avflux_dic["NH"], c="gray", label="R>0N")

p75s1, = axes[1, 1].plot(avflux_dq_dic["Antarctica"] - avflux_dic["Antarctica"], c="blue", label="R>75S")
p60s1, = axes[1, 1].plot(avflux_dq_dic["SP"] - avflux_dic["SP"], c="orange", label="R>60S")
p30s1, = axes[1, 1].plot(avflux_dq_dic["SExT"] - avflux_dic["SExT"], c="red", label="R>30S")
ps30s1, = axes[1, 1].plot(avflux_dq_dic["ST"] - avflux_dic["ST"], c="violet", label="R<30S")
p0s1, = axes[1, 1].plot(avflux_dq_dic["SH"] - avflux_dic["SH"], c="gray", label="R<0S")

pgln1, = axes[1, 0].plot(avflux_dq_dic["Global"] - avflux_dic["Global"], c="black", linestyle="-", label="global")
pgls1, = axes[1, 1].plot(avflux_dq_dic["Global"] - avflux_dic["Global"], c="black", linestyle="-", label="global")

axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)


axes[1, 0].set_ylabel("Radiative flux in Wm$^{-2}$")

axes[1, 0].set_title("Northern Hemisphere $-$ dQ minus no-dQ")
axes[1, 1].set_title("Southern Hemisphere $-$ dQ minus no-dQ")

axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")

axes[0, 1].set_yticklabels([])
axes[1, 1].set_yticklabels([])

if fb == "LR":
    axes[0, 0].set_ylim([-5, 3.5])
    axes[0, 1].set_ylim([-5, 3.5])    
    axes[0, 0].text(0, -4.5, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -4.5, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-7.25, 2.5])
    axes[1, 1].set_ylim([-7.25, 2.5])    
    axes[1, 0].text(0, -6.75, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -6.75, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    l1 = axes[0, 0].legend(handles=[p75n0, p60n0, p30n0], ncol=1, loc="lower right")
    l2 = axes[1, 0].legend(handles=[ps30n0, p0n0, pgln0], ncol=1, loc="lower right")
    axes[1, 1].legend(ncol=3, loc="lower right")
    
elif fb == "SA":
    axes[0, 0].set_ylim([-9.5, 0.5])
    axes[0, 1].set_ylim([-9.5, 0.5])    
    axes[0, 0].text(0, -9, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -9, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-15.75, 1])
    axes[1, 1].set_ylim([-15.75, 1])    
    axes[1, 0].text(0, -15., "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -15., "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    l1 = axes[0, 0].legend(handles=[p75n0, p60n0, p30n0], ncol=1, loc="lower right")
    l2 = axes[1, 0].legend(handles=[ps30n0, p0n0, pgln0], ncol=1, loc="lower right")
    axes[1, 1].legend(ncol=3, loc="lower right")

elif fb == "WV":
    axes[0, 0].set_ylim([-6, 0.5])
    axes[0, 1].set_ylim([-6, 0.5])    
    axes[0, 0].text(0, -5.7, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -5.7, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-13, 7])
    axes[1, 1].set_ylim([-13, 7])    
    axes[1, 0].text(0, -12., "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -12., "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].legend(ncol=3, loc="upper right")
    axes[1, 1].legend(ncol=3, loc="lower right")

elif fb == "CLW":
    axes[0, 0].set_ylim([-2, 1.35])
    axes[0, 1].set_ylim([-2, 1.35])    
    axes[0, 0].text(0, -1.85, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -1.85, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-13, 10])
    axes[1, 1].set_ylim([-13, 10])    
    axes[1, 0].text(0, -12., "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -12., "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

    axes[1, 0].legend(ncol=3, loc="upper right")
    axes[1, 1].legend(ncol=3, loc="lower right")
    
elif fb == "CSW":
    axes[0, 0].set_ylim([-5, 0.35])
    axes[0, 1].set_ylim([-5, 0.35])    
    axes[0, 0].text(0, -4.75, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -4.75, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-10, 7])
    axes[1, 1].set_ylim([-10, 7])    
    axes[1, 0].text(0, -9., "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -9., "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center") 
    
    axes[0, 0].legend(ncol=3, loc="upper right")
    axes[1, 1].legend(ncol=3, loc="upper right")

elif fb == "C":
    axes[0, 0].set_ylim([-6, 1.5])
    axes[0, 1].set_ylim([-6, 1.5])    
    axes[0, 0].text(0, -5.65, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[0, 1].text(0, -5.65, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    
    axes[1, 0].set_ylim([-6, 2.5])
    axes[1, 1].set_ylim([-6, 2.5])    
    axes[1, 0].text(0, -5.65, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
    axes[1, 1].text(0, -5.65, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

    axes[0, 0].legend(ncol=3, loc="upper right")
    axes[1, 1].legend(ncol=3, loc="lower right")
# end if    

fig.suptitle(f"{fb} radiative flux")
fig.subplots_adjust(wspace=0.05, hspace=0.3, top=0.9)

pl.savefig(pl_path + f"/PDF/MultiRegion_{fb}{cs_fn}_RadFlux_G1_G2_CESM2_SOM_Diff.pdf", bbox_inches="tight", 
           dpi=250)
pl.savefig(pl_path + f"/PNG/MultiRegion_{fb}{cs_fn}_RadFlux_G1_G2_CESM2_SOM_Diff.png", bbox_inches="tight", 
           dpi=250)
pl.show()
pl.close()


#%% plot the p-values of the inter-group differences in radiative flux in one panel
lwd = 0.75

fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(10, 6.5), sharey=True, sharex=False)

axes[0, 0].plot(fl_regi_ttest[1].pvalue, c="blue", label="R>75N", linewidth=lwd)
axes[0, 0].plot(fl_regi_ttest[2].pvalue, c="orange", label="R>60N", linewidth=lwd)
axes[0, 0].plot(fl_regi_ttest[3].pvalue, c="red", label="R>30N", linewidth=lwd)
axes[0, 0].plot(fl_regi_ttest[4].pvalue, c="violet", label="R<30N", linewidth=lwd)
axes[0, 0].plot(fl_regi_ttest[5].pvalue, c="gray", label="R>0N", linewidth=lwd)

axes[0, 1].plot(fl_regi_ttest[10].pvalue, c="blue", linestyle="-", label="R>75S", linewidth=lwd)
axes[0, 1].plot(fl_regi_ttest[9].pvalue, c="orange", linestyle="-", label="R>60S", linewidth=lwd)
axes[0, 1].plot(fl_regi_ttest[8].pvalue, c="red", linestyle="-", label="R>30S", linewidth=lwd)
axes[0, 1].plot(fl_regi_ttest[7].pvalue, c="violet", linestyle="-", label="R<30S", linewidth=lwd)
axes[0, 1].plot(fl_regi_ttest[6].pvalue, c="gray", linestyle="-", label="R>0S", linewidth=lwd)

axes[0, 0].plot(fl_glob_ttest.pvalue, c="black", linestyle="-", label="global", linewidth=lwd)
axes[0, 1].plot(fl_glob_ttest.pvalue, c="black", linestyle="-", label="global", linewidth=lwd)

axes[0, 0].axhline(y=0, c="black", linewidth=0.5)
axes[0, 1].axhline(y=0, c="black", linewidth=0.5)

axes[0, 0].axhline(y=0.05, c="red", linewidth=1, linestyle="--")
axes[0, 1].axhline(y=0.05, c="red", linewidth=1, linestyle="--")

axes[0, 0].legend(ncol=2, loc="upper center")
axes[0, 1].legend(ncol=2, loc="upper center")

# axes[0, 0].set_xlabel("Year of simulation")
# axes[0, 1].set_xlabel("Year of simulation")
axes[0, 0].set_ylabel("$p$ value")

axes[0, 0].set_title("Northern Hemisphere $-$ G2 minus G1")
axes[0, 1].set_title("Southern Hemisphere $-$ G2 minus G1")

# axes[0, 1].set_yticklabels([])
# axes[1, 1].set_yticklabels([])
# axes[0, 0].set_ylim((-0.05, 0.5))
axes[0, 1].set_ylim((-0.05, 0.5))

axes[0, 0].text(140, 0.47, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[0, 1].text(140, 0.47, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")

# lower panels
axes[1, 0].plot(fl_dq_ttest["Arctic"].pvalue, c="blue", label="R>75N", linewidth=lwd)
axes[1, 0].plot(fl_dq_ttest["NP"].pvalue, c="orange", label="R>60N", linewidth=lwd)
axes[1, 0].plot(fl_dq_ttest["NExT"].pvalue, c="red", label="R>30N", linewidth=lwd)
axes[1, 0].plot(fl_dq_ttest["NT"].pvalue, c="violet", label="R<30N", linewidth=lwd)
axes[1, 0].plot(fl_dq_ttest["NH"].pvalue, c="gray", label="R>0N", linewidth=lwd)
axes[1, 0].plot(fl_dq_ttest["Global"].pvalue, c="black", label="global", linewidth=lwd)

axes[1, 1].plot(fl_dq_ttest["Antarctica"].pvalue, c="blue", label="R>75S", linewidth=lwd)
axes[1, 1].plot(fl_dq_ttest["SP"].pvalue, c="orange", label="R>60S", linewidth=lwd)
axes[1, 1].plot(fl_dq_ttest["SExT"].pvalue, c="red", label="R>30S", linewidth=lwd)
axes[1, 1].plot(fl_dq_ttest["ST"].pvalue, c="violet", label="R<30S", linewidth=lwd)
axes[1, 1].plot(fl_dq_ttest["SH"].pvalue, c="gray", label="R>0S", linewidth=lwd)
axes[1, 1].plot(fl_dq_ttest["Global"].pvalue, c="black", label="global", linewidth=lwd)

axes[1, 0].axhline(y=0, c="black", linewidth=0.5)
axes[1, 1].axhline(y=0, c="black", linewidth=0.5)

axes[1, 0].axhline(y=0.05, c="red", linewidth=1, linestyle="--")
axes[1, 1].axhline(y=0.05, c="red", linewidth=1, linestyle="--")

# axes[1, 0].legend(ncol=2, loc="upper center")

axes[1, 0].set_ylabel("$p$ value")
axes[1, 0].set_xlabel("Year of simulation")
axes[1, 1].set_xlabel("Year of simulation")

axes[1, 0].set_title("Northern Hemisphere $-$ dQ minus no-dQ")
axes[1, 1].set_title("Southern Hemisphere $-$ dQ minus no-dQ")

axes[1, 0].text(36.5, 0.47, "(c)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1, 1].text(36.5, 0.47, "(d)", fontsize=12, horizontalalignment="center", verticalalignment="center")

fig.suptitle(f"{fb} radiative flux")
fig.subplots_adjust(wspace=0.05, hspace=0.3, top=0.9)

pl.savefig(pl_path + f"/PDF/MultiRegion_{fb}{cs_fn}_RadFlux_G1_G2_CESM2_SOM_Diff_p_values.pdf", bbox_inches="tight", 
           dpi=250)
pl.savefig(pl_path + f"/PNG/MultiRegion_{fb}{cs_fn}_RadFlux_G1_G2_CESM2_SOM_Diff_p_values.png", bbox_inches="tight", 
           dpi=250)

pl.show()
pl.close()