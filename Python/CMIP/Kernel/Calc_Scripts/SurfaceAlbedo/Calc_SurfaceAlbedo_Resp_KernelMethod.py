"""
Calculate the surface albedo radiative response for CMIP members via the kernel method.

This script uses the piControl 21-year running means as unperturbed state. The 21-year running means are computed here.

Be sure to set data_path, kern_path, and out_path.
"""

#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import pylab as pl
import matplotlib.colors as colors
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
import progressbar as pg
import dask.array as da
import time as ti
import Ngl as ngl
import geocat.ncomp as geoc
# from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Flip import flip
from Functions.Func_Print_In_One_Line import print_one_line


#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[9]
except:
    toa_sfc = "TOA"
# end try except


#%% choose a kernel: "Sh08", "So08", "BM13", "H17", "P18"
try:
    kl = sys.argv[4]
except:
    kl = "Sh08"
# end try except


#%% set the kernel state (as of now only applicable for the Block & Mauritsen, 2013 kernels)
#   possible values: "pi", "2x", "4x", "8x"
try:
    k_st = sys.argv[5]
except:
    k_st = "pi"
# end try except

# addend for file names regarding the climate state under which the kernels are derived
state_fn = {"pi":"CTRL", "2x":"2xCO2", "4x":"4xCO2", "8x":"8xCO2"}
state_ofn = {"pi":"_Kernel", "2x":"_2xCO2Kernel", "4x":"_4xCO2Kernel", "8x":"_8xCO2Kernel"}  # out file name

# note that the reasoning behind not giving an addend to the CTRL kernels is that the climate state in which they were
# calculated should more or less corrspond to the ones in which the others were calculated


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 18
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except


#%% set clear-sky (1) or all-sky (0) --> standard is all-sky
try:
    sky = int(sys.argv[3])
except:
    sky = 0
# end try except


#%% set the experiment
try:
    exp = sys.argv[8]
except:    
    exp = "a4x"
    exp = "1pct"
# end try except


#%% set some parameters for the names and variables accordingly
cs_var = ""
cs_n = ""
cs_pa = ""
sky_desc = "all-sky"
if sky == 1:
    cs_var = "clr"
    cs_n = "CS_"
    cs_pa = "CS/"
    sky_desc = "clear-sky"
# end if


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[6]
except:
    ensemble_b = "r1i1p1f1"    
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[7]
except:
    ensemble_f = "r3i1p1f1"    
# end try except    


#%% total number years
try:
    n_yr = int(sys.argv[10])
except:    
    n_yr = 150
# end try except      


#%% handle the ensemble
ensemble_b_d = ""
ensemble_f_d = ""

# handle the ensemble in the file name    
if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # base experiment ensemble (usually piControl)
    ensemble_b_d = "_" + ensemble_b
# end if
if (ensemble_f != "r1i1p1f1") & (ensemble_f != "r1i1p1"):  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f_d = "_" + ensemble_f
# end if


#%% set the running mean 
run = 21


#%% check availablity in case of surface kernels
if (kl in ["So08", "Sh08"]) & (toa_sfc == "SFC"):
    print("\nError: Surface kernels not available. Aborting.\n")
    raise Exception
# end if    


#%% load the namelist
exp_s = ""
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    if exp == "a2x":
        exp_s = "_a2x"
        exp_n = "abrupt2xCO2"
    elif exp == "a4x":        
        exp_n = "abrupt4xCO2"
    elif exp == "1pct":        
        exp_s = "_1pct"
        exp_n = "1pctCO2"
    # end if elif  
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    if exp == "a2x":
        exp_s = "_a2x"
        exp_n = "abrupt-2xCO2"
    elif exp == "a4x":        
        exp_n = "abrupt-4xCO2"
    elif exp == "1pct":        
        exp_s = "_1pct"
        exp_n = "1pctCO2"
    # end if elif       
# end if elif


#%% get the indices for the time interval to be chosen
exp_sta = int(nl.b_times[nl.models[mod]] * 12)
if nl.b_times[nl.models[mod]] < 10:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)    
    b_end_ind = int(n_yr * 12) + b_sta_ind
    print("\npiControl does not start at least 10 year before forcing experiment. " + 
          "Taking start index from namelist: " + str(b_sta_ind) + ".\n")
else:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)
    b_sta_ind = int(b_sta_ind - ((run - 1) / 2) * 12)
    print("\npiControl does start at least 10 year before forcing experiment. " + 
          "Adjusting start index from namelist: " + str(b_sta_ind) + ".\n")
# end if else


#%% set up dictionaries for the kernel choice

# kernels names for keys
k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]

# kernel names for plot names/titles, texts, paths/directories etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_t = {"Sh08":"Shell et al. (2008)", "So08":"Soden et al. (2008)", "BM13":"Block and Mauritsen (2013)", 
       "H17":"Huang et al. (2017)", "P18":"Pendergrass et al. (2018)", "S18":"Smith et al. (2018)"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", 
       "H17":"Huang_etal_2017", "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# set the regrid method per kernel
k_rg = {"Sh08":"scipy", "So08":"scipy", "BM13":"scipy", "H17":"scipy", "P18":"scipy", "S18":"scipy"}

# set kernel directory addend
k_ad = {"Sh08":"kernels/", "So08":"", "BM13":"", "H17":"kernel-highR/toa/", "P18":"cam5-kernels/kernels/", "S18":""}

# clear-sky addend
k_cs_ad = {"Sh08":"", "So08":"", "BM13":"d","H17":"cld", "P18":"", "P18":"", "S18":""}
if sky == 1:
    k_cs_ad = {"Sh08":"_clr", "So08":"clr", "BM13":"f", "H17":"clr", "P18":"C", "S18":"_cs"}
# end if

# set kernel file names
k_fn = {"Sh08":"CAM3_albedo_sw" + k_cs_ad[kl] + "_kernel.nc", "So08":"TOA_GFDL_Kerns.nc", 
        "BM13":state_fn[k_st] + "_kernel_mm_1950.nc", "H17":"RRTMG_alb_toa_" + k_cs_ad[kl] + "_highR.nc", 
        "P18":"alb.kernel.nc", "S18":"HadGEM2_sw_TOA_L38.nc"}

# set kernel variable names
if toa_sfc == "TOA":
    k_vn = {"Sh08":"monkernel", "So08":"sw" + k_cs_ad[kl] + "_a", "BM13":"A_sra" + k_cs_ad[kl] + "0", "H17":"swkernel", 
            "P18":"FSNT" + k_cs_ad[kl], "S18":"albedo_sw" + k_cs_ad[kl]}
elif toa_sfc == "SFC":
    k_vn = {"Sh08":"monkernel", "So08":"sw" + k_cs_ad[kl] + "_a", "BM13":"A_sra" + k_cs_ad[kl] + "0", "H17":"swkernel", 
            "P18":"FSNS" + k_cs_ad[kl], "S18":"albedo_sw" + k_cs_ad[kl]}
# end if elif    

# set kernel grid variable names
k_gvn = {"Sh08":["lat", "lon"], "So08":["latitude", "longitude"], "BM13":["lat", "lon"], "H17":["lat", "lon"], 
         "P18":["lat", "lon"], "S18":["lat", "lon"]}

# set kernel grid "transform" --> flip the given axis
k_flip = {"Sh08":"NA", "So08":"NA", "BM13":"lat", "H17":"lat", "P18":"NA", "S18":"NA"}

# output name addend in case there is a special kernel property; as of now (26.06.2020) the only addend is the kernel
# state in case of the Block and Mauritsen (2013) kernels
out_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}

# add a directory for all states except the control state; this only affects the BM13 kernels because the other
# kernels exist only for the control state
out_path_ad_bm13 = {"pi":"", "2x":"/State_2xCO2/", "4x":"/State_4xCO2/", "8x":"/State_8xCO2/"}
out_path_ad_rest = {"pi":"", "2x":"", "4x":"", "8x":""}
out_path_ad = {"Sh08":out_path_ad_rest, "So08":out_path_ad_rest, "BM13":out_path_ad_bm13, "H17":out_path_ad_rest, 
               "P18":out_path_ad_rest, "S18":out_path_ad_rest}


#%% print the kernel name
print(k_t[kl] + " kernels...")


#%% set the regridding method - either "scipy" or "geocat"
regrid_meth = k_rg[kl]


#%% set paths

# model data path
# --> must contain the variables loaded in the next code block
data_path = ""

# kernel path
# --> must contain the radiative kernels as set above
kern_path = f"/{k_p[kl]}_Kernels/{k_ad[kl]}"

# output path
out_path = f"/Kernel_{toa_sfc}_RadResp/{exp_n}/{k_p[kl]}/SfcAlb_Response/" + cs_pa + out_path_ad[kl][k_st]


#%% load the nc files
alke_nc = Dataset(kern_path + k_fn[kl])

rsds_4x_nc = Dataset(glob.glob(data_path + "rsds_Amon_" + nl.models_n[mod] + "_" + exp_n + "_" + ensemble_f + "*")[0])
rsus_4x_nc = Dataset(glob.glob(data_path + "rsus_Amon_" + nl.models_n[mod] + "_" + exp_n + "_" + ensemble_f + "*")[0])

rsds_pi_nc = Dataset(glob.glob(data_path + "rsds_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])
rsus_pi_nc = Dataset(glob.glob(data_path + "rsus_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])

# load one air temperature file to get the 3d coordinates
try:
    dqdt_nc = Dataset(sorted(glob.glob(data_path + "dlogQdT_Files" + exp_s + ensemble_f_d + "/*.nc"), 
                             key=str.casefold)[0])
except:
    dqdt_nc = Dataset(sorted(glob.glob(data_path + exp_s + "_ta_Files" + ensemble_f_d + "/*.nc"), 
                             key=str.casefold)[0])    
# end try except


#%% get the values

# kernel latitudes (note that the "NA" part seems like rather bad programming...)
grid_d = {"lat":alke_nc.variables[k_gvn[kl][0]][:], "lon":alke_nc.variables[k_gvn[kl][1]][:], "NA":np.arange(1)}

# flip the necessary part of the grid info
grid_d[k_flip[kl]] = flip(grid_d[k_flip[kl]])

lat_ke = grid_d["lat"]
lon_ke = grid_d["lon"]

# model latitudes
lat2d = rsds_4x_nc.variables["lat"][:]
lon2d = rsds_4x_nc.variables["lon"][:]
lat3d = dqdt_nc.variables["lat"][:]
lon3d = dqdt_nc.variables["lon"][:]

# test if 2d and 3d lats are of same length
if len(lat2d) == len(lat3d):
    lat = lat2d
    lon = lon2d
else:
    lat = lat3d
    lon = lon3d
# end if else        

tim = rsds_4x_nc.variables["time"][:int(n_yr*12)]

# NOTE THE POSSIBLE FLIPPING ALONG THE LATITUDE
alke = flip(np.ma.masked_invalid(alke_nc.variables[k_vn[kl]][:]), axis=k_flip[kl])

rsds_4x = da.from_array(rsds_4x_nc.variables["rsds"][:int(n_yr*12), :, :])
rsus_4x = da.from_array(rsus_4x_nc.variables["rsus"][:int(n_yr*12), :, :])
rsds_pi = da.from_array(rsds_pi_nc.variables["rsds"])
rsus_pi = da.from_array(rsus_pi_nc.variables["rsus"])


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
if ((da.shape(rsds_pi)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print(f"\npiControl run after branch year is at least 10 years longer than {exp_n}. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print(f"\npiControl run after branch year is less than 10 years longer than {exp_n}. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")    
# end if

rsds_pi = rsds_pi[b_sta_ind:b_end_ind, :, :]
rsus_pi = rsus_pi[b_sta_ind:b_end_ind, :, :]
# raise Exception()


#%% test the values
"""
ds4x_glan = glob_mean(an_mean(rsds_4x), lat2d, lon2d)
us4x_glan = glob_mean(an_mean(rsus_4x), lat2d, lon2d)
dspi_glan = glob_mean(an_mean(rsds_pi), lat2d, lon2d)
uspi_glan = glob_mean(an_mean(rsus_pi), lat2d, lon2d)

# plot
fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(10, 6))

ax1.plot(ds4x_glan, c="red", label="rsds 4x")
ax1.plot(dspi_glan, c="blue", label="rsds pi")

ax2.plot(us4x_glan, c="orange", label="rsus 4x")
ax2.plot(uspi_glan, c="green", label="rsus pi")

ax1.legend()
ax2.legend()

pl.show()
pl.close()
"""


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    f_rsds = sorted(glob.glob(data_path + "/AddFiles_2d_piC/rsds_*" + ensemble_b_d + "*_AddBe.nc"), key=str.casefold)
    f_rsus = sorted(glob.glob(data_path + "/AddFiles_2d_piC/rsus_*" + ensemble_b_d + "*_AddBe.nc"), key=str.casefold)
    
    rsds_pi = da.concatenate([da.ma.masked_array(Dataset(f_rsds[-1]).variables["rsds"], lock=True), rsds_pi], axis=0)
    rsus_pi = da.concatenate([da.ma.masked_array(Dataset(f_rsus[-1]).variables["rsus"], lock=True), rsus_pi], axis=0)

    print("\nAdded AddBe file. New piControl shape: " + str(da.shape(rsds_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_rsds = sorted(glob.glob(data_path + "/AddFiles_2d_piC/rsds_*" + ensemble_b_d + "*_AddAf.nc"), key=str.casefold)
    f_rsus = sorted(glob.glob(data_path + "/AddFiles_2d_piC/rsus_*" + ensemble_b_d + "*_AddAf.nc"), key=str.casefold)
    
    rsds_pi = da.concatenate([rsds_pi, da.ma.masked_array(Dataset(f_rsds[-1]).variables["rsds"], lock=True)], axis=0)
    rsus_pi = da.concatenate([rsus_pi, da.ma.masked_array(Dataset(f_rsus[-1]).variables["rsus"], lock=True)], axis=0)

    print("\nAdded AddAf file. New piControl shape: " + str(da.shape(rsds_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except   


#%% regrid the kernels to the model grid if necessary
if ~np.array_equal(lat_ke, lat) | ~np.array_equal(lon_ke, lon):
    
    # first transform the lats and lons to increase monotonically
    lon_tr = lon
    lat_tr = lat + 90
    lon_o_tr = lon_ke
    lat_o_tr = lat_ke + 90
     
    if regrid_meth == "scipy":
        # regrid the field
        rem_alke = np.zeros((12, len(lat), len(lon)))

        print("\nRegridding kernels to " + nl.models_pl[mod] + " grid via scipy...\n")
        for t in range(12):
            
            print(str(t+1) + " of 12")
            
            rem_alke[t, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, alke[t, :, :])
        # end for z
    elif regrid_meth == "geocat":
        print("\nRegridding kernels to " + nl.models_pl[mod] + " grid via geocat...\n")
        rem_alke = geoc.linint2(alke, xo=lon_tr, yo=lat_tr, icycx=True, xi=lon_o_tr, yi=lat_o_tr)
    # end if elif
    
# end if
# raise Exception

#%% calculate the surface albedo
alb_4x = np.array(rsus_4x / rsds_4x)

if ((cmip_v == 6) & np.any(mod == np.array([3, 20, 52]))) | ((cmip_v == 5) & np.any(mod == np.array([2, 3, 6, 15, 23]))):
    rsds_pi_np = np.zeros(da.shape(rsds_pi))
    rsus_pi_np = np.zeros(da.shape(rsus_pi))
    for i in np.arange(np.shape(rsds_pi_np)[0]):
        rsds_pi_np[i, :, :] = np.array(rsds_pi[i, :, :])
        rsus_pi_np[i, :, :] = np.array(rsus_pi[i, :, :])
    # end for i
    alb_pi = rsus_pi_np / rsds_pi_np
else:
    alb_pi = np.array(rsus_pi / rsds_pi)  # this does not seem to work for KIOST-ESM
# end if else

# the following seems to be necessary for KIOST-ESM
# alb_pi = np.zeros((int(n_yr*12), len(lat), len(lon)))
# for i in np.arange(int(n_yr*12)):
#     alb_pi[i, :, :] = rsus_pi[i, :, :] / rsds_pi[i, :, :]
# end for i    


#%% albedo should not be smaller than zero --> convert values < 0 to 0
#   albedo should not be larger than 1 --> convert values > 1 to 1
a4x_n_l1 = np.sum(alb_4x > 1)
pic_n_l1 = np.sum(alb_pi > 1)
a4x_n_s0 = np.sum(alb_4x < 0)
pic_n_s0 = np.sum(alb_pi < 0)

print("\nNumber of cells a4x > 1: " + str(a4x_n_l1))
print("\nNumber of cells piC > 1: " + str(pic_n_l1))
print("\nNumber of cells a4x < 0: " + str(a4x_n_s0))
print("\nNumber of cells piC < 0: " + str(pic_n_s0) + "\n")

alb_4x[alb_4x > 1] = 1
alb_pi[alb_pi > 1] = 1
alb_4x[alb_4x < 0] = 0 
alb_pi[alb_pi < 0] = 0

print("\nValues > 1 replaced with 1.")
print("\nValues < 0 replaced with 0.\n")


#%% calculate the surface albedo piControl 21-year running mean

# NOTE: This n_add_yrs stuff still works even after I switched to producing the dedicated AddBe and AddAf files but it is
#       basically unnecessary now. I'll leave it in for now as a check: n_add_yrs HAS to be 0 since the values that 
#       originally had to be filled here WERE (in the current implementation) already filled via the extend files. 

# how many "additional years" are needed?
n_add_yrs = n_yr - (int(np.shape(alb_pi)[0] / 12) - (run-1))
print("\n\nadd_yrs = " + str(n_add_yrs) + "\n\n")

if n_add_yrs > 0:
    add_b = int(n_add_yrs/2)  # add before
    add_a = -(int(n_add_yrs/2) + (n_add_yrs % 2))  # add after
else:
    add_b = 0
    add_a = None
# end if else


alb_pi_run = np.zeros((12, n_yr, len(lat2d), len(lon2d)))
print("\nCalculating running means...\n")
for mon in np.arange(12):
    print(str(mon+1) + "/12")
    alb_pi_run[mon, add_b:add_a, :, :] = run_mean_da(alb_pi[mon::12, :, :], running=run)
# end for mon

"""
desc_add_ayr = ""
if add_yrs > 0:
    desc_add_ayr = ("\nFurther, the first and/or last year of the running means has used to fill up the time slots" +
                    "\nfor which no running mean could be calculated since because the time series is not long " +
                    "enough.")
    print("\nFilling first years...\n")
    for i in np.arange(add_b):
        print(i)
        for mon in np.arange(12):        
            alb_pi_run[mon, i, :, :] = run_mean_da(alb_pi[mon::12, :, :], running=run)[0, :, :]
        # end for mon
    # end for i
    print("\nFilling last years...\n")
    for i in np.arange(1, -add_a+1, 1):
        print(i)
        for mon in np.arange(12):        
            alb_pi_run[mon, -i, :, :] = run_mean_da(alb_pi[mon::12, :, :], running=run)[-1, :, :]
        # end for mon
    # end for i
# end if
"""


#%% calculate the surface albedo change
alb_ch = np.zeros((n_yr*12, len(lat2d), len(lon2d)))
for yr, i in enumerate(np.arange(0, n_yr*12, 12)):
    alb_ch[i:i+12, :, :] = (alb_4x[i:i+12, :, :] - alb_pi_run[:, yr, :, :]) * 100  # apparently required in percent
# end for yr, i

print("\nNaN count: " + str(np.sum(np.isnan(alb_ch))))
print("\nInf count: " + str(np.sum(np.isinf(alb_ch))))

# set the NaN values to 0
alb_ch[np.isnan(alb_ch)] = 0
alb_ch[np.isinf(alb_ch)] = 0


#%% if necessary (as e.g. for ACCESS-ESM1.5) regrid the surface pressure climatology to the air temperature grid
if len(lat2d) != len(lat):
    alb_ch_re = np.zeros((12*n_yr, len(lat), len(lon)))
    
    lat_o = lat2d + 90
    lat_t = lat + 90
    
    print("\n\nRegridding surfaces albedo because of different latitudes from atmospheric variables...\n\n")
    for mon in np.arange(n_yr*12):
        alb_ch_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, alb_ch[mon, :, :], verbose=True)
    # end for mon
    
    # replace the variabels with the regridded ones
    alb_ch = alb_ch_re
# end if

print("\nNaNs replaced by 0. NaN count: " + str(np.sum(np.isnan(alb_ch))))
print("\nInfs replaced by 0. Inf count: " + str(np.sum(np.isinf(alb_ch))))


#%% calculate the TOA radiation change due to the surface albedo change

alb_flux = np.zeros(np.shape(alb_ch))

count = 0
for i in np.arange(0, int(n_yr*12), 12):
    
    alb_flux[i:i+12, :, :] = alb_ch[i:i+12, :, :] * rem_alke
    
    count += 1
# end for i


#%% calculate and print the global mean of the mean over the last 30 years

# calculate the last 30 years mean
alb_flux_30y = np.mean(alb_flux[-(30*12):, :, :], axis=0)

print("\n" + str(np.sum(np.isnan(alb_flux_30y))) + " NaN values in the last 30-year mean\n")
alb_flux_30y[np.isnan(alb_flux_30y)] = 0
# print(np.sum(np.isnan(alb_flux_30y)))

# generate the weights for the global average
weights2d = np.zeros((len(lat), len(lon)))
weights2d[:, :] = np.cos(lat[:, None] / 180 * np.pi)

alb_flux_30y_gl = np.average(alb_flux_30y, weights=weights2d, axis=(-2, -1))

print("global mean surface albedo response last 30-year mean " + k_t[kl] + ": " + str(alb_flux_30y_gl) + " W/m²")


#%% plot the last 30-year mean response

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

# colorbar ticks
c_ticks = np.arange(0, 106, 15)

# surface albedo kernel
fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 4.5), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, alb_flux_30y, transform=ccrs.PlateCarree(), cmap=cm.Reds, 
                  extend="both", levels=c_ticks)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1 = pl.colorbar(p1, ax=ax1, ticks=c_ticks)
cb1.set_label("kernel in Wm$^{-2}$")
ax1.set_title("Surface Albedo Response " + nl.models_pl[mod] + " " + k_t[kl] + "\n" + "global mean " + 
              str(alb_flux_30y_gl) + " Wm$^{-2}$")

pl.show()
pl.close()


#%% plot test
pl.imshow(alb_ch[0, :, :] * rem_alke[0, :, :], origin="lower")
pl.colorbar()
pl.title(f"First Month TOA Contribution {nl.models_pl[mod]}")

pl.show()
pl.close()


#%% plot the time series
toa_an = np.zeros(n_yr)
yrind = 0
for i in np.arange(0, int(n_yr*12), 12):
    
    toa_an[yrind] = glob_mean(np.mean(alb_ch[i:i+12, :, :] * rem_alke, axis=0), lat, lon)
    yrind += 1
# end for i

pl.plot(toa_an)
pl.title(f"Global Mean Albedo Evolution {nl.models_pl[mod]}")
pl.show()
pl.close()    


#%% store the result in a netcdf file

print("\nGenerating nc file...\n")
out_name = toa_sfc + "_RadResponse_SfcAlb" + exp_s + "_" + kl + out_ad[kl] + "_" + cs_n + nl.models[mod] + ".nc"
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", n_yr*12)
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
lon_nc = f.createVariable("lon", "f8", "lon")
tim_nc = f.createVariable("time", "f4", "time")
a_re_nc = f.createVariable("sa_resp", "f4", ("time", "lat", "lon"))
da_nc = f.createVariable("dsa", "f4", ("time", "lat", "lon"))

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
tim_nc[:] = np.arange(n_yr*12)
count = 0
for i in np.arange(0, int(n_yr*12), 12):
    
    a_re_nc[i:i+12, :, :] = alb_ch[i:i+12, :, :] * rem_alke
    da_nc[i:i+12, :, :] = alb_ch[i:i+12, :, :]
    
    count += 1
    print_one_line(str(count) + f"/{n_yr}  ")
# end for i
# a_re_nc[:] = alb_flux

a_re_nc.units = "W m^-2"

lat_nc.units = rsds_4x_nc.variables["lat"].units
lon_nc.units = rsds_4x_nc.variables["lon"].units
tim_nc.units = rsds_4x_nc.variables["time"].units

a_re_nc.description = (f"surface albedo {toa_sfc} radiative response")

# add attributes
desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values corresponding to the whole duration of the forcing experiment which is why they were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = (f"This file contains the surface albedo {sky_desc} {toa_sfc} radiative response in the " + exp_n +
                 "\nexperiment for " + nl.models_pl[mod] + 
                 ". The values correspond to monthly means.\nThe kernels are taken from " + k_t[kl] + ".\n" + 
                 f"They where derived in a {state_fn[k_st]} climate state.\n" + 
                 "The piControl values were averaged to a 21-year running mean." + desc_add)

f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


#%% check if the file has been produced
try:
    test = Dataset(out_path + out_name)
    print("\n\nFile successfully produced. Directory:")
    print(out_path)
    print("\nName: " + out_name + "\n")
except:
    print("Error. File could not be stored.\n")
# end try except


