"""
Plot and store global mean and Arctic tas or ts as well as global mean TOA imbalance for G1 and G2 in a netCDF file.

Generates data for Fig. 6 in Eiselt and Graversen (2022), JCLI.

Be sure to set cslt_path, data_path, toa_path, and fb_path correctly.
"""


#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import pandas as pd
from numpy.random import permutation as perm
import numexpr as ne
import pylab as pl
import matplotlib.colors as colors
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
from scipy.stats import norm as sci_norm
from scipy.stats import percentileofscore as perc_of_score
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Z_Score_P_Value import z_score
from Functions.Func_Region_Mean import region_mean
from Functions.Func_MonteCarlo_SignificanceTest_V2 import monte_carlo
from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean, an_mean_verb
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Classes.Class_MidpointNormalize import MidpointNormalize
from dask.distributed import Client


#%% establish connection to the client
c = Client("localhost:8786")


#%% choose the kernels
kl = "P18"


#%% set the surface temperature variable that will be used to perform the Gregory regression (tas or ts)
t_var = "ts"

# feedbacks calculated by regressing on tas or ts?
fb_ts = "tas"


#%% set early and late period threshold year
elp = 20


#%% set up the region
x1s, x2s = [0], [360]
y1s, y2s = [75], [90]


#%% set the interval over which to average the TOA imbalance at the elp threshold
av_w = 5
dav_w = int(av_w/2)


#%% set the runnning mean
run = 21


#%% length of the experiment in years
n_yr = 150


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


#%% set paths
out_path = "/G1_G2_Comp/"


#%% set the experiment - either "4x" or "2x"
try:
    exp = sys.argv[3]
except:
    exp = "4x"
# end try except


#%% set up some initial dictionaries and lists

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# sign of long wave kernels
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

# set up a response list
respl = ["Cloud_Response", "Q_Response", "SfcAlb_Response", "T_Response"]


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)

# clear sky linearity test path
cslt_path = f"//Model_Lists/ClearSkyLinearityTest/{fb_ts}_Based/Thres_Year_{elp}/"


#%% load in the clear-sky linearity test result (i.e. the modes that pass the test)
pass_cslt = np.array(pd.read_csv(cslt_path + f"ClearSky_LinTest_Pass_{kl}_Kernels_{fb_ts}_Based_ThrYr{elp}_" + 
                                 f"DiffThr{cslt_thr:02}.csv"))[:, 1]
print(f"\n{len(pass_cslt)} models pass the CSLT in the given configuration...\n")


#%% add something to the file name concerning the CSLT
cslt_fadd = ""
if cslt:
    cslt_fadd = f"_CSLT{cslt_thr:02}{kl}"
# end if    


#%% select models to be excluded from the multi-model mean
excl_fadd = ""
excl_tadd = ""
len_excl = 0
excl_arr = np.array([""])


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
elif excl == "outliers":
    
    excl_arr = np.array(["GISS-E2-R", "MIROC-ES2L", "GISS-E2.2-G"])

    excl_fadd = "_WoOutl"
    len_excl = len(excl_arr)  # important!     
# end if elif    


#%% loop over all models
sle_an_li, sll_an_li, sld_an_li = [], [], []
sle_djf_li, sll_djf_li, sld_djf_li = [], [], []
sle_mam_li, sll_mam_li, sld_mam_li = [], [], []
sle_jja_li, sll_jja_li, sld_jja_li = [], [], []
sle_son_li, sll_son_li, sld_son_li = [], [], []

mods = []
dtas_all = {}
dtoa_all = {}
dtas_rm_all = {}
fbe = {}
fbl = {}
lr_fb_d = []

for cmip_v in [5, 6]:
    
    if cmip_v == 5:
        import Namelists.Namelist_CMIP5 as nl
        cmip = "CMIP5"
        a4x = "abrupt" + exp + "CO2"
    elif cmip_v ==  6:
        import Namelists.Namelist_CMIP6 as nl
        cmip = "CMIP6"
        a4x = "abrupt-" + exp + "CO2"
    # end if elif

    for mod, mod_n in enumerate(nl.models_pl):
                
        if cslt:
            if not np.any(pass_cslt == mod_n):
                print("\nJumping " + mod_n + 
                      " because it fails the clear-sky linearity test (or data unavailable)...")
                continue
            # end if 
        # end if
        
        # exclude a chosen set of models
        if np.any(excl_arr == mod_n):
            print("\nJumping " + mod_n + " because it is in the exclude list (or data unavailable)...")
            continue
        # end if        
        
        # print(nl.models_pl[mod])
        
        try:

            #% set paths
            data_path = f"//{cmip}/Data/{nl.models[mod]}/"
            toa_path = (f"//{cmip}/Outputs/Kernel_TOA_RadResp/{a4x}/{k_p[kl]}/")
            fb_path = f"/{cmip}/"
    
    
            #% load dtas and dtoa
            dtas_nc = Dataset(glob.glob(data_path + f"d{t_var}*.nc")[0])
            dtoa_nc = Dataset(glob.glob(data_path + "dtoa_as_cs_Amon*.nc")[0])
            
            # load the files with the different responses and get the flux values
            dtoa_s_nc = Dataset(glob.glob(toa_path + f"SfcAlb_Response/*{nl.models[mod]}*.nc")[0])
            
            # load feedback files
            i_fb_nc = Dataset(glob.glob(fb_path + "/Outputs/" + 
                                        f"/Feedbacks/Kernel/{a4x}/{k_p[kl]}/{fb_ts}_Based/*{nl.models[mod]}.nc")[0])              
            # load the cloud feedback
            cl_fb_d = (i_fb_nc.variables["C_fb_l"][0] - i_fb_nc.variables["C_fb_e"][0]) 
            if (cl_fb_d > 0.5) & (rm_str_dcfb):
                print(f"\nJumping {mod_n} because of large cloud dFb")
                continue
            # end if

            # load the lapse rate feedback
            lr_fb_d.append(i_fb_nc.variables["LR_fb_l"][0] - i_fb_nc.variables["LR_fb_e"][0])             
            
        except:
            print(f"\nAnalysis not working for {nl.models_pl[mod]}. Continuing...")
            continue
        # end try except            

        #% load the lats and lon
        lat = dtoa_s_nc.variables["lat"][:]
        lon = dtoa_s_nc.variables["lon"][:]
        lat2d = dtas_nc.variables["lat"][:]
        lon2d = dtas_nc.variables["lon"][:]
            
        dtas_gm = glob_mean(np.mean(dtas_nc.variables[f"{t_var}_ch"][:], axis=0), lat2d, lon2d)
        dtas_rm = region_mean(np.mean(dtas_nc.variables[f"{t_var}_ch"][:], axis=0), x1s, x2s, y1s, y2s, lat2d, lon2d)
        dtoa_gm = glob_mean(np.mean(dtoa_nc.variables["dtoa_as"][:], axis=0), lat2d, lon2d)
        
        sle = lr(dtas_gm[:elp], dtoa_gm[:elp])[0]
        sll = lr(dtas_gm[elp:], dtoa_gm[elp:])[0]
        
        #% generate the seasonal means
        alb_gm_sm = {}
        alb_hm_sm = {}
        alb_rm_sm = {}
        #for sea, inds in zip(["an", "djf", "mam", "jja", "son"], [np.arange(12), (11, 0, 1), 
        #                                                          (2, 3, 4), (5, 6, 7), (8, 9, 10)]):
        dtas_all[mod_n] = dtas_gm
        dtas_rm_all[mod_n] = dtas_rm
        dtoa_all[mod_n] = dtoa_gm
        fbe[mod_n] = sle
        fbl[mod_n] = sll
        
        mods.append(mod_n)
        
    # end for mod
# end for cmip_v

mods = np.array(mods)
lr_fb_d = np.array(lr_fb_d)


#%% set up the two model groups separated by global mean lapse rate feedback change
g1_add = ""
g2_add = ""

pl_add = ""
tl_add = ""


#%% get the indices for models with LR dFb < 0.1 Wm-2K-1 and the ones with LR dFB > 0.5 Wm-2K-1
low_dfbi = lr_fb_d < 0.1
high_dfbi = lr_fb_d > 0.5


#%% print the models and their LR feedback changes
print("G1")
for i, j in zip(mods[low_dfbi], lr_fb_d[low_dfbi]):
    print([i, j])
# end for i, j    
print("G2")
for i, j in zip(mods[high_dfbi], lr_fb_d[high_dfbi]):
    print([i, j])
# end for i, j

str_lr_dfb = mods[high_dfbi]
wea_lr_dfb = mods[low_dfbi]

#% calculate the mean over the two groups individually
str_lr_dtas = []
str_lr_treg = []
str_lr_dtoa = []
for mstr in str_lr_dfb:
    str_lr_dtas.append(dtas_all[mstr])
    str_lr_treg.append(dtas_rm_all[mstr])
    str_lr_dtoa.append(dtoa_all[mstr])
# end for mstr
wea_lr_dtas = []
wea_lr_treg = []
wea_lr_dtoa = []
for mstr in wea_lr_dfb:
    wea_lr_dtas.append(dtas_all[mstr])
    wea_lr_treg.append(dtas_rm_all[mstr])
    wea_lr_dtoa.append(dtoa_all[mstr])
# end for mstr
str_lr_dtas = np.array(str_lr_dtas)
wea_lr_dtas = np.array(wea_lr_dtas)
str_lr_treg = np.array(str_lr_treg)
wea_lr_treg = np.array(wea_lr_treg)
str_lr_dtoa = np.array(str_lr_dtoa)
wea_lr_dtoa = np.array(wea_lr_dtoa)

mean_str_lr_dtas = np.mean(str_lr_dtas, axis=0)
mean_wea_lr_dtas = np.mean(wea_lr_dtas, axis=0)

mean_str_lr_treg = np.mean(str_lr_treg, axis=0)
mean_wea_lr_treg = np.mean(wea_lr_treg, axis=0)
std_str_lr_treg = np.std(str_lr_treg, axis=0)
std_wea_lr_treg = np.std(wea_lr_treg, axis=0)

mean_str_lr_dtoa = np.mean(str_lr_dtoa, axis=0)
mean_wea_lr_dtoa = np.mean(wea_lr_dtoa, axis=0)


#%% store these data as netcdf file
f = Dataset(out_path + f"G1_G2_Arctic_and_Gl_Mean_d{t_var}{cslt_fadd}.nc", "w", format="NETCDF4")

# create the dimensions
f.createDimension("years", n_yr)
f.createDimension("mods_G1", len(wea_lr_dfb))
f.createDimension("mods_G2", len(str_lr_dfb))

# create the variables
mods_g1_nc = f.createVariable("mods_G1", "S25", "mods_G1")
mods_g2_nc = f.createVariable("mods_G2", "S25", "mods_G2")
dtas_g1_nc = f.createVariable(f"d{t_var}_G1", "f4", ("mods_G1", "years"))
dtas_g2_nc = f.createVariable(f"d{t_var}_G2", "f4", ("mods_G2", "years"))
dtas_ar_g1_nc = f.createVariable(f"d{t_var}_Ar_G1", "f4", ("mods_G1", "years"))
dtas_ar_g2_nc = f.createVariable(f"d{t_var}_Ar_G2", "f4", ("mods_G2", "years"))
dtoa_g1_nc = f.createVariable("dtoa_G1", "f4", ("mods_G1", "years"))
dtoa_g2_nc = f.createVariable("dtoa_G2", "f4", ("mods_G2", "years"))

p_g1_g2_nc = f.createVariable("p_G1_G2", "f4", "years")
p_ar_g1_g2_nc = f.createVariable("p_Ar_G1_G2", "f4", "years")

# pass the data into the variables
mods_g1_nc[:] = np.array(wea_lr_dfb)
mods_g2_nc[:] = np.array(str_lr_dfb)

for i, mstr in enumerate(wea_lr_dfb):
    dtas_g1_nc[i, :] = dtas_all[mstr]
    dtas_ar_g1_nc[i, :] = dtas_rm_all[mstr]
    dtoa_g1_nc[i, :] = dtoa_all[mstr]
for i, mstr in enumerate(str_lr_dfb):
    dtas_g2_nc[i, :] = dtas_all[mstr]
    dtas_ar_g2_nc[i, :] = dtas_rm_all[mstr]
    dtoa_g2_nc[i, :] = dtoa_all[mstr]
# end for i, mstr

p_g1_g2_nc[:] = ttest_ind(str_lr_dtas, wea_lr_dtas, axis=0).pvalue
p_ar_g1_g2_nc[:] = ttest_ind(str_lr_treg, wea_lr_treg, axis=0).pvalue

# set units of the variables
dtoa_g1_nc.units = "W m^-2"
dtoa_g2_nc.units = "W m^-2"
dtas_g1_nc.units = "K"
dtas_g2_nc.units = "K"
dtas_ar_g1_nc.units = "K"
dtas_ar_g2_nc.units = "K"

# descriptions of the variables
mods_g1_nc.description = "names of the model in G1 (weak lapse rate feedback change)"
mods_g2_nc.description = "names of the model in G2 (weak lapse rate feedback change)"

dtoa_g1_nc.description = "global mean TOA imbalance (positive down) G1"
dtoa_g2_nc.description = "global mean TOA imbalance (positive down) G2"
dtas_g1_nc.description = f"global mean d{t_var} G1"
dtas_g2_nc.description = f"global mean d{t_var} G2"
dtas_ar_g1_nc.description = f"Arctic mean d{t_var} G1"
dtas_ar_g2_nc.description = f"Arctic mean d{t_var} G2"

p_g1_g2_nc.description = f"p-value for group mean difference Wald t-test between G1 and G2 for global mean {t_var}"
p_ar_g1_g2_nc.description = f"p-value for group mean difference Wald t-test between G1 and G2 Arctic mean {t_var}"

# date of file creation
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the file
f.close()
