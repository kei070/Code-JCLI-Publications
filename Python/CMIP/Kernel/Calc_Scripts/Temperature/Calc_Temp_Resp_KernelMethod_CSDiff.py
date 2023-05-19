"""
Calculate the temperature radiative "cloud adjustment" for CMIP members via the kernel method.

Note that there are two kinds of kernels with regards to the vertical coordinate: 
    1. Pressure levels (Sh08, So08, H17)
        Here we simply interpolate the kernel values linearly to the CMIP member's vertical coordinates (which are also
        pressure levels) and then multiply them by the pressure level thickness divided by 100 hPa. This last step 
        basically corresponds to a weighting along the vertical, making possible the summation along this coordinate.
    2. Hybrid (sigma) levels (BM13, P18)
        Here we first have to calculate the individual pressure of each grid cell from the hyam, hybm, and surface 
        pressure variables as well as p0 in case of the P18 kernels. We then divide the kernels by the pressure level
        tickness (in hPa) so that we have the kernels in hPa^-1. These are then interpolated linearly to the pressure
        levels of the CMIP member. Finally, the interpolated values are multiplyed again by the pressure level thickness
        of the CMIP member's pressure coordinates so that, again, the summation over the vertical is made possible.
        
    Note that as the surface pressure field for calculating the pressure level thicknesses we use a mean climatology over
    the last 30 years of the piControl run of the CMIP member.
    
Be sure to set data_path, kern_path, and out_path.    
"""

#%% imports
import os
import sys
import copy
import time
import glob
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
import timeit
import Ngl as ngl
import geocat.ncomp as geoc
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Interp4d import interp4d, interp3d
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Flip import flip
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Functions.Func_Closest_Ind import closest_ind


#%% set the surface temperature varaible: either ts or tas
ts_var = "ts"


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 31
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except


#%% choose a kernel: "Sh08", "So08", "BM13", "H17", "P18"
try:
    kl = sys.argv[3]
except:
    kl = "Sh08"
# end try except


#%% set the kernel state (as of now only applicable for the Block & Mauritsen, 2013 kernels)
#   possible values: "pi", "2x", "4x", "8x"
try:
    k_st = sys.argv[4]
except:
    k_st = "pi"
# end try except

# addend for file names regarding the climate state under which the kernels are derived
state_fn = {"pi":"CTRL", "2x":"2xCO2", "4x":"4xCO2", "8x":"8xCO2"}
state_ofn = {"pi":"_Kernel", "2x":"_2xCO2Kernel", "4x":"_4xCO2Kernel", "8x":"_8xCO2Kernel"}  # out file name

# note that the reasoning behind not giving an addend to the CTRL kernels is that the climate state in which they were
# calculated should more or less corrspond to the ones in which the others were calculated


#%% apply stratosphere mask? (1 == yes, 0 == no)
try:
    strat_mask = int(sys.argv[5])
except:
    strat_mask = 1
# end try except

# adjust the output path accordingly
if strat_mask:
    strat_path = ""
else:
    strat_path = "/WO_Strat_Mask/"
# end


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[6]
except:
    ensemble_b = "r1i1p1f1"    
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[7]
except:
    ensemble_f = "r1i1p1f1"    
# end try except 


#%% set the experiment
try:
    exp = sys.argv[8]
except:    
    exp = "1pct"
# end try except



#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[9]
except:
    toa_sfc = "TOA"
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



#%% set clear-sky (1) or all-sky (0) --> standard is all-sky
sky = 0

# set some parameters for the names and variables accordingly
cs_var = ""
cs_n = ""
cs_p = ""
sky_desc = "all-sky"
if sky == 1:
    cs_var = "clr"
    cs_n = "CS_"
    cs_p = "CS "
    sky_desc = "clear-sky"
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


#%% set the running mean 
run = 21


#%% file name suffix for at-surface
sfc_add = ""
if toa_sfc == "SFC":
    sfc_add = "_SFC"
# end if


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

# all- and clear-sky addend
k_as_ad = {"Sh08":"", "So08":"", "BM13":"", "H17":"cld", "P18":"", "S18":""}
k_cs_ad = {"Sh08":"_clr", "So08":"clr", "BM13":"", "H17":"clr", "P18":"C", "S18":""}
k_as_vad = {"Sh08":"", "So08":"", "BM13":"d", "H17":"cld", "P18":"", "S18":""}
k_cs_vad = {"Sh08":"C", "So08":"clr", "BM13":"f", "H17":"clr", "P18":"C", "S18":"_cs"}

# set kernel file names (maybe this is not the most economic way...)
k_as_fn = {"Sh08":{"t":"CAM3_planck_lw" + k_as_ad[kl] + "_kernel.nc", "ts":"CAM3_surft_lw" + k_as_ad[kl] + "_kernel.nc"}, 
           "So08":{"t":"TOA_GFDL_Kerns.nc", "ts":"TOA_GFDL_Kerns.nc"},
           "BM13":{"t":state_fn[k_st] + "_kernel_mm_1950.nc", "ts":state_fn[k_st] + "_kernel_mm_1950.nc"}, 
           "H17":{"t":"RRTMG_t_toa_" + k_as_ad[kl] + "_highR.nc", "ts":"RRTMG_ts_toa_" + k_as_ad[kl] + "_highR.nc"}, 
           "P18":{"t":"t.kernel.nc", "ts":"ts.kernel.nc"},
           "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}
k_cs_fn = {"Sh08":{"t":"CAM3_planck_lw" + k_cs_ad[kl] + "_kernel.nc", "ts":"CAM3_surft_lw" + k_cs_ad[kl] + "_kernel.nc"}, 
           "So08":{"t":"TOA_GFDL_Kerns.nc", "ts":"TOA_GFDL_Kerns.nc"}, 
           "BM13":{"t":state_fn[k_st] + "_kernel_mm_1950.nc", "ts":state_fn[k_st] + "_kernel_mm_1950.nc"}, 
           "H17":{"t":"RRTMG_t_toa_" + k_cs_ad[kl] + "_highR.nc", "ts":"RRTMG_ts_toa_" + k_cs_ad[kl] + "_highR.nc"}, 
           "P18":{"t":"t.kernel.nc", "ts":"ts.kernel.nc"},
           "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}

# set kernel variable names (maybe this is not the most economic way...)
if toa_sfc == "TOA":
    k_as_vn = {"Sh08":{"t":"kernel_p", "ts":"FLNT" + k_as_vad[kl] + "A_FLNT" + k_as_vad[kl]}, 
               "So08":{"t":"lw" + k_as_vad[kl] + "_t", "ts":"lw" + k_as_vad[kl] + "_ts"},
               "BM13":{"t":"Ta_tra" + k_as_vad[kl] + "0", "ts":"Ts_tra" + k_as_vad[kl] + "0"},
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNT" + k_as_vad[kl], "ts":"FLNT" + k_as_vad[kl]},
               "S18":{"t":"ta_lw" + k_as_vad[kl], "ts":"tsurf_lw" + k_as_vad[kl]}}
    k_cs_vn = {"Sh08":{"t":"kernel_p", "ts":"FLNT" + k_cs_vad[kl] + "A_FLNT" + k_cs_vad[kl]}, 
               "So08":{"t":"lw" + k_cs_vad[kl] + "_t", "ts":"lw" + k_cs_vad[kl] + "_ts"}, 
               "BM13":{"t":"Ta_tra" + k_cs_vad[kl] + "0", "ts":"Ts_tra" + k_cs_vad[kl] + "0"},
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNT" + k_cs_vad[kl], "ts":"FLNT" + k_cs_vad[kl]},
               "S18":{"t":"ta_lw" + k_cs_vad[kl], "ts":"tsurf_lw" + k_cs_vad[kl]}}
elif toa_sfc == "SFC":
    k_as_vn = {"Sh08":{"t":"kernel_p", "ts":"FLNT" + k_as_vad[kl] + "A_FLNT" + k_as_vad[kl]}, 
               "So08":{"t":"lw" + k_as_vad[kl] + "_t", "ts":"lw" + k_as_vad[kl] + "_ts"},
               "BM13":{"t":"Ta_tra" + k_as_vad[kl] + "0", "ts":"Ts_tra" + k_as_vad[kl] + "0"},
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNS" + k_as_vad[kl], "ts":"FLNS" + k_as_vad[kl]},
               "S18":{"t":"ta_lw" + k_as_vad[kl], "ts":"tsurf_lw" + k_as_vad[kl]}}
    k_cs_vn = {"Sh08":{"t":"kernel_p", "ts":"FLNT" + k_cs_vad[kl] + "A_FLNT" + k_cs_vad[kl]}, 
               "So08":{"t":"lw" + k_cs_vad[kl] + "_t", "ts":"lw" + k_cs_vad[kl] + "_ts"}, 
               "BM13":{"t":"Ta_tra" + k_cs_vad[kl] + "0", "ts":"Ts_tra" + k_cs_vad[kl] + "0"},
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNS" + k_cs_vad[kl], "ts":"FLNS" + k_cs_vad[kl]},
               "S18":{"t":"ta_lw" + k_cs_vad[kl], "ts":"tsurf_lw" + k_cs_vad[kl]}}
# end if elif

# set kernel grid variable names
k_gvn = {"Sh08":["lev", "lat", "lon"], 
         "So08":["plev", "latitude", "longitude"], 
         "BM13":["mlev", "lat", "lon"], 
         "H17":["player", "lat", "lon"], 
         "P18":["lev", "lat", "lon"],
         "S18":["plev", "lat", "lon"]}

# set the kernel surface pressure variable name (if existent)
k_ps_vn = {"Sh08":"", "So08":"PS", "BM13":"aps", "H17":"", "P18":"PS", "S18":""}

# set kernel grid "transform" --> flip the given axis
k_flip = {"Sh08":"lev", "So08":"NA", "BM13":"lat", "H17":"lat", "P18":"NA", "S18":"lev"}

# vertical interpolation method
k_vi = {"Sh08":"np", "So08":"np", "BM13":"np", "H17":"np", "P18":"np", "S18":"np"}

# the positive direction of the long-wave flux (positive "up" or "down")
k_pud = {"Sh08":"down", "So08":"up", "BM13":"down", "H17":"up", "P18":"down", "S18":"up"}

# output name addend in case there is a special kernel property; as of now (26.06.2020) the only addend is the kernel
# state in case of the Block and Mauritsen (2013) kernels
out_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}

# introduce a sign dictionary (only for the test plots) --> -1 for the positive down kernels, +1 for positive up
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

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
# --> see code block "get the lists of and load files" for required files
data_path = ""

# kernel path
kern_path = f"/{k_p[kl]}_Kernels/{k_ad[kl]}"

# output path
out_path = (f"/Kernel_{toa_sfc}_RadResp/{exp_n}/{k_p[kl]}/01_Diff_to_CS" + 
            f"/{nl.models[mod]}/{strat_path}/{out_path_ad[kl][k_st]}/")


#%% load the nc files
take_nc = Dataset(kern_path + k_as_fn[kl]["t"])
take_cs_nc = Dataset(kern_path + k_cs_fn[kl]["t"])
tske_nc = Dataset(kern_path + k_as_fn[kl]["ts"])
tske_cs_nc = Dataset(kern_path + k_cs_fn[kl]["ts"])


#%% load the kernel values

take = (flip(np.ma.masked_invalid(take_cs_nc.variables[k_cs_vn[kl]["t"]][:]), axis=k_flip[kl]) - 
        flip(np.ma.masked_invalid(take_nc.variables[k_as_vn[kl]["t"]][:]), axis=k_flip[kl]))
tske = (flip(np.ma.masked_invalid(tske_cs_nc.variables[k_cs_vn[kl]["ts"]][:]), axis=k_flip[kl]) -
        flip(np.ma.masked_invalid(tske_nc.variables[k_as_vn[kl]["ts"]][:]), axis=k_flip[kl]))

# kernel latitudes (note that the "NA" part seems like rather bad programming...)
grid_d = {"lev":take_nc.variables[k_gvn[kl][0]][:],
          "lat":take_nc.variables[k_gvn[kl][1]][:], 
          "lon":take_nc.variables[k_gvn[kl][2]][:], 
          "NA":np.arange(1)}

# flip the necessary part of the grid info
grid_d[k_flip[kl]] = flip(grid_d[k_flip[kl]])

# check if the pressure units are hPa and if not convert
pres_fac = 1
if np.max(grid_d["lev"]) > 1E4:
    pres_fac = 100
# end if

levs_k = grid_d["lev"] / pres_fac
lat_k = grid_d["lat"]
lon_k = grid_d["lon"]


#%% plot some values
"""
z = 0

# vmax = np.nanmax([take[0, z, :, :], take_gf[0, z, :, :], take_vf[0, z, :, :]])
# vmin = np.nanmin([take[0, z, :, :], take_gf[0, z, :, :], take_vf[0, z, :, :]])
vmax = 0
vmin = -2

pl.imshow(take[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.show()
pl.close()

pl.imshow(take_gf[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.show()
pl.close()

pl.imshow(take_vf[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.show()
pl.close()
"""

#%% get the lists of and load files
dqdt_f_list = sorted(glob.glob(data_path + "dlogQdT_Files" + exp_s + ensemble_f_d + "/*.nc"), key=str.casefold)

# ts files
ts_a4x_nc = Dataset(glob.glob(data_path + ts_var + "_Amon_" + nl.models_n[mod] + "_" + exp_n + "_" + ensemble_f + 
                              "*")[0])
ts_pic_nc = Dataset(glob.glob(data_path + ts_var + "_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])

# ps files
ps_a4x_nc = Dataset(glob.glob(data_path + "ps_Amon_" + nl.models_n[mod] + "_" + exp_n + "_" + ensemble_f + "*")[0])
ps_pic_nc = Dataset(glob.glob(data_path + "ps_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])


#%%  loop over the files and concatenate the values via dask
"""
if len(ta_a4x_f_list) > 1:
    for i in np.arange(1, len(ta_a4x_f_list)):
        ta_a4x = da.concatenate([ta_a4x, da.ma.masked_array(Dataset(ta_a4x_f_list[i]).variables["ta"], lock=True)])
    # end for i
# end if
    
if len(ta_pic_f_list) > 1:
    for i in np.arange(1, len(ta_pic_f_list)):
        ta_pic = da.concatenate([ta_pic, da.ma.masked_array(Dataset(ta_pic_f_list[i]).variables["ta"], lock=True)])
    # end for i
# end if

# reduce the arrays to the 150 years of the experiment and the corresponding control run
ta_a4x = ta_a4x[:(n_yr*12), :, :, :]
ta_pic = ta_pic[b_sta_ind:b_end_ind, :, :, :]

# test plot of the lowest level
# pl.imshow(ta_a4x[100, 0, :, :], origin="lower"), pl.colorbar()

# rechunk the dask arrays
ta_a4x = da.rechunk(ta_a4x, chunks=(30, len(levs), len(lat), len(lon)))
ta_pic = da.rechunk(ta_pic, chunks=(30, len(levs), len(lat), len(lon)))
"""


#%% load in the respective first dQ/dT file
dqdt_nc = Dataset(dqdt_f_list[0])

# load the first dQ/dT file
dta = da.ma.masked_array(dqdt_nc.variables["dT"], lock=True)


#%% get the grid coordinates (vertical and horizontal)

# get the levels
levs = dqdt_nc.variables["plev"][:]

# get lat, lon, and levels for the model data
lat = dqdt_nc.variables["lat"][:]
lon = dqdt_nc.variables["lon"][:]
tim = ps_a4x_nc.variables["time"][:n_yr*12]

# load the (possibly different) lat and lon for the surface quantities
lat2d = ts_a4x_nc.variables["lat"][:]
lon2d = ts_a4x_nc.variables["lon"][:]


#%%  loop over the dlogQ and dQdT files and concatenate the values via dask
if len(dqdt_f_list) > 1:
    for i in np.arange(1, len(dqdt_f_list)):
        dta = da.concatenate([dta, da.ma.masked_array(Dataset(dqdt_f_list[i]).variables["dT"], lock=True)])
    # end for i
# end if

# print the dataset history of the last dataset loaded
print("\n\ndT, dQ, dQ/dT data set " + Dataset(dqdt_f_list[i]).history + "\n\n")

# reduce the arrays to the n_yr years of the experiment and the corresponding control run
dta = dta[:(n_yr*12), :, :, :]

# rechunk the dask arrays
dta = da.rechunk(dta, chunks=(30, len(levs), len(lat), len(lon)))


#%% load the 2d-variable
ts_a4x = da.from_array(ts_a4x_nc.variables[ts_var], lock=True, chunks=(30, len(lat), len(lon)))[:(n_yr*12), :, :]

ts_pic = da.from_array(ts_pic_nc.variables[ts_var], lock=True, chunks=(30, len(lat), len(lon)))

# load the ps fields as numpy arrays
ps_a4x = ps_a4x_nc.variables["ps"][:(n_yr*12), :, :]
ps_pic = ps_pic_nc.variables["ps"]


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
if ((da.shape(ts_pic)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print(f"\npiControl run after branch year is at least 10 years longer than {exp_n}. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print(f"\npiControl run after branch year is less than 10 years longer than {exp_n}. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")    
# end if

ts_pic = ts_pic[b_sta_ind:b_end_ind, :, :]
ps_pic = ps_pic[b_sta_ind:b_end_ind, :, :]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    # f_ps = sorted(glob.glob(data_path + "/AddFiles_2d_piC/ps_*AddBe.nc"), key=str.casefold)
    f_ts = sorted(glob.glob(data_path + f"/AddFiles_2d_piC/{ts_var}_Amon_{nl.models_n[mod]}_piControl{ensemble_b_d}" + 
                            "*AddBe.nc"), key=str.casefold)    
    
    # ps_pic = da.concatenate([da.ma.masked_array(Dataset(f_ps[-1]).variables["ps"], lock=True), ps_pic], axis=0)
    ts_pic = da.concatenate([da.ma.masked_array(Dataset(f_ts[-1]).variables[ts_var], lock=True), ts_pic], axis=0)

    print("\nAdded AddBe file. New piControl shape: " + str(da.shape(ts_pic)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    # f_ps = sorted(glob.glob(data_path + "/AddFiles_2d_piC/ps_*AddAf.nc"), key=str.casefold)
    f_ts = sorted(glob.glob(data_path + f"/AddFiles_2d_piC/{ts_var}_Amon_{nl.models_n[mod]}_piControl{ensemble_b_d}" + 
                            "*AddAf.nc"), key=str.casefold)
    
    # ps_pic = da.concatenate([ps_pic, da.ma.masked_array(Dataset(f_ps[-1]).variables["ps"], lock=True)], axis=0)
    ts_pic = da.concatenate([ts_pic, da.ma.masked_array(Dataset(f_ts[-1]).variables[ts_var], lock=True)], axis=0)

    print("\nAdded AddAf file. New piControl shape: " + str(da.shape(ts_pic)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except 


#%% calculate the surface albedo piControl 21-year running mean

# NOTE: This n_add_yrs stuff still works even after I switched to producing the dedicated AddBe and AddAf files but it is
#       basically unnecessary now. I'll leave it in for now as a check: n_add_yrs HAS to be 0 since the values that 
#       originally had to be filled here WERE (in the current implementation) already filled via the extend files. 

n_add_yrs = n_yr - (int(np.shape(ts_pic)[0] / 12) - (run-1))
print("\n\nadd_yrs = " + str(n_add_yrs) + "\n\n")

if n_add_yrs > 0:
    add_b = int(n_add_yrs/2)  # add before
    add_a = -(int(n_add_yrs/2) + (n_add_yrs % 2))  # add after
else:
    add_b = 0
    add_a = None
# end if else


ts_pic_run = np.zeros((12, n_yr, len(lat2d), len(lon2d)))
print("\nCalculating running means...\n")
for mon in np.arange(12):
    print(mon)
    ts_pic_run[mon, add_b:add_a, :, :] = run_mean_da(ts_pic[mon::12, :, :], running=run)
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
            ts_pic_run[mon, i, :, :] = run_mean_da(ts_pic[mon::12, :, :], running=run)[0, :, :]
        # end for mon
    # end for i
    print("\nFilling last years...\n")
    for i in np.arange(1, -add_a+1, 1):
        print(i)
        for mon in np.arange(12):        
            ts_pic_run[mon, -i, :, :] = run_mean_da(ts_pic[mon::12, :, :], running=run)[-1, :, :]
        # end for mon
    # end for i
# end if
"""


#%% calculate a last 30 year climatology of surface pressure from the piControl experiment (the choice of time interval
#   is rather arbitrary)
ps_clim = np.zeros((12, len(lat2d), len(lon2d)))
for i in np.arange(12):
    ps_clim[i, :, :] = np.mean(ps_pic[(-30*12)+i::12, :, :], axis=0)
# end for i


#%% if necessary (as e.g. for ACCESS-ESM1.5) regrid the surface pressure climatology to the air temperature grid
if len(lat2d) != len(lat):
    ps_clim_re = np.zeros((12, len(lat), len(lon)))
    
    lat_o = lat2d + 90
    lat_t = lat + 90
    
    print("\n\nRegridding surfaces pressure because of different latitudes from atmospheric variables...\n\n")
    for mon in np.arange(12):
        ps_clim_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, ps_clim[mon, :, :], verbose=True)
    # end for mon
    
    # replace the variabels with the regridded ones
    ps_clim = ps_clim_re
# end if


#%% regrid the ps climatology to the kernel grid
ps_ke = np.zeros((12, len(lat_k), len(lon_k)))

# first transform the lats and lons
lon_tr = lon_k
lat_tr = lat_k + 90
lon_o_tr = lon
lat_o_tr = lat + 90

print("\n\nRegridding the surface pressure field to the kernel grid...\n\n")
for t in range(12):
    
    print_one_line(str(t+1) + "   ")
    
    ps_ke[t, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, ps_clim[t, :, :])

# end for t


#%% calculate the model grid pressure level thickness using the piControl surface pressure climatology field (!)
try:
    pdiff_vi = np.array(geoc.dpres_plevel(plev=levs/100, psfc=ps_clim/100, ptop=np.min(levs)/100, msg=np.nan))
except ValueError:  # apparently, this is necesseray because of python's round problem (?)
    pdiff_vi = np.array(geoc.dpres_plevel(plev=levs/100, psfc=ps_clim/100, 
                                              ptop=np.min(levs)/100-np.min(levs)/100 * 0.001, msg=np.nan))   
# end try except  


#%% get the kernel pressure variables
if kl == "Sh08":
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke/100, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100

elif kl == "So08":
    ps_ke_nc = take_nc
    
    ps_ke = flip(np.ma.masked_invalid(ps_ke_nc.variables[k_ps_vn[kl]][:]), axis=k_flip[kl])
    ptop = np.min(levs_k)
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100    

elif kl == "BM13":  # BM13 kernels
    ps_ke_nc = take_nc
    pmid_nc = take_nc
    
    pdiff = flip(np.ma.masked_invalid(pmid_nc.variables["pdiff"][:]), axis=k_flip[kl]) / 100
    # pdiff = pmid_nc.variables["pdiff"][:, :, ::-1, :] / 100  # directly convert to hPa
    ps_ke = flip(np.ma.masked_invalid(ps_ke_nc.variables[k_ps_vn[kl]][:]), axis=k_flip[kl])
    
    hyam = take_nc.variables["hyam"][:]
    hybm = take_nc.variables["hybm"][:] 
    
    levs_k = np.zeros((12, len(levs_k), len(lat_k), len(lon_k)))
    for t in np.arange(12):
        for i in np.arange(len(lat_k)):
            for j in np.arange(len(lon_k)):
                levs_k[t, :, i, j] = hyam + hybm * ps_ke[t, i, j]
            # end for j
        # end for i
    # end for t
    levs_k = levs_k / 100

elif kl == "H17":
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke/100, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100

elif kl == "P18":
    ps_ke_nc = Dataset(kern_path + "PS.nc")
    pmid_nc = Dataset(kern_path + "p_sigma.nc")
    
    p0 = take_nc.variables["P0"][:]
    hyam = take_nc.variables["hyam"][:]
    hybm = take_nc.variables["hybm"][:] 
    ps_ke = ps_ke_nc.variables["PS"][:]
    
    ps_ke = flip(np.ma.masked_invalid(ps_ke_nc.variables[k_ps_vn[kl]][:]), axis=k_flip[kl])
    # pdiff = pmid_nc.variables["pdiff"][:] / 100  # directly convert to hPa
    pdiff = flip(np.ma.masked_invalid(pmid_nc.variables["pdiff"][:]), axis=k_flip[kl]) / 100
    
    levs_k = np.zeros((12, len(levs_k), len(lat_k), len(lon_k)))
    for t in np.arange(12):
        for i in np.arange(len(lat_k)):
            for j in np.arange(len(lon_k)):
                levs_k[t, :, i, j] = hyam * p0 + hybm * ps_ke[t, i, j]
            # end for j
        # end for i
    # end for t
    levs_k = levs_k / 100
    
elif kl == "S18":
    # levs_k = take_nc.variables["p_mean"][:] / 100
    # pdiff = take_nc.variables["p_thickness"][:] / 100
    # pdiff_vi = pdiff_vi / 100
    pdiff = 1
    pdiff_vi = pdiff_vi / 100
    
# end if elif


#%% vertical kernel interpolation

# "grid fill"
take_gf = copy.deepcopy(take)
print("\nFill the masked values of the ta kernel via lat-lon grid filling using scipy's interpolate.griddata...\n")
for m in range(12):
    print_one_line(str(m + 1) + "   ")
    for z in range(np.shape(take)[-3]):
        if np.sum(take[m, z, :, :].mask) > 0:  # Any masked values? If not jump to the next loop...
        
            xx, yy = np.meshgrid(lon_k, lat_k)
            
            # get only the valid values
            x1 = xx[~take[m, z, :, :].mask]
            y1 = yy[~take[m, z, :, :].mask]
            newarr = take[m, z, :, :][~take[m, z, :, :].mask]
            
            take_gf[m, z, :, :] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
                       
        else:
            continue
        # end if else
   # end for z
# end for m

# print("\n\n\nFill the remaining values with 0...\n")
# take_gf[np.isnan(take)] = 0

# interpolate the kernel to the CMIP pressure levels
take_vi = np.zeros((12, len(levs), len(lat_k), len(lon_k)))
print("\n\nInterpolate the kernel vertically to the CMIP pressure levels using my interp4d...\n")
for t in range(12):
    print_one_line(str(t + 1) + "   ")
    take_vi[t, :, :, :] = interp4d(t, levs_k, take_gf/pdiff, levs/100)
# end for t


#%% calculate the pressure level level-thickness ON THE MODEL GRID --> rem_pske 
"""
ptop = np.min(levs/100)

pdiff_vi_l = []
print("\n\nCalculating pressure level thickness...\n")
for i in np.arange(0, 1800, 180):
    print(str(i) + "   ")
    pdiff_vi_l.append(da.from_array(np.array(geoc.dpres_plevel(levs/100, ps_a4x[i:i+180, :, :]/100, ptop, 
                                                               msg=np.nan))))
# end for i

pdiff_vi = da.concatenate(pdiff_vi_l, axis=0) / 100  # directly calculate the fraction of 100 hPa level thickness
"""

#%% plot some values
"""
xi = 60
yi = 60
pl.plot(take_vi[0, :, yi, xi], levs/100, c="red", label="ta kernel vert. int.", marker="x")
pl.plot(take[0, :, yi, xi], levs_k, c="blue", label="ta kernel original", marker="+")
pl.legend()
pl.gca().invert_yaxis()
pl.show()
pl.close()
"""

"""
fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(18, 14))

z = 5

z_k = np.argmin(np.abs(levs_k - levs[z]/100))

p1 = ax1.imshow(take_vi[0, z, :, :], origin="lower", vmin=-2, vmax=0)
fig.colorbar(p1, ax=ax1)
ax1.set_title(str(levs[z]/100) + " hPa $-$ CMIP Pressure Levels (Index " + str(z) + ")")

p2 = ax2.imshow(take_vf[0, z_k, :, :], origin="lower", vmin=-2, vmax=0)
fig.colorbar(p2, ax=ax2)
ax2.set_title(str(levs_k[z_k]) + " hPa $-$ Kernel Pressure Levels (Index " + str(z_k) + ")")
pl.show()
pl.close()
"""

#%% interpolate the kernels and the surface pressure field to the horizontal model grid

# first transform the lats and lons
lon_tr = lon
lat_tr = lat + 90
lon_o_tr = lon_k
lat_o_tr = lat_k + 90

# regrid the field
print("\n\n\nRegridding kernels to " + nl.models_pl[mod] + " grid via " + regrid_meth + "...\n")
if regrid_meth == "scipy":

    rem_take = np.zeros((12, len(levs), len(lat), len(lon)))
    rem_tske = np.zeros((12, len(lat), len(lon)))
    
    for t in range(12):
        
        print_one_line(str(t+1) + "   ")
        
        for z in range(len(levs)):
            rem_take[t, z, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, take_vi[t, z, :, :])
        # end for z
        rem_tske[t, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, tske[t, :, :])
    
    # end for t
# end if


#%% multiply the remapped pressure differences onto the kernels
rem_take = rem_take * pdiff_vi


#%% plot the remapped kernel
"""
mon = 0
v_ind = -1


# rem  and vint take
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(rem_take[mon, v_ind, :, :], l_num, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, rem_take[mon, v_ind, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("Ta Kernel in Wm$^{-2}$K$^{-1}$")
ax1.set_title("rem and vint Ta Kernel Month " + str(mon) + " level: " + str(levs[v_ind]/100) + " hPa")
pl.show()
pl.close()

# pdiff_vi
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(pdiff_vi[mon, v_ind, :, :], l_num, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, pdiff_vi[mon, v_ind, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("pdiff in hPa")
ax1.set_title("Pdiff Month " + str(mon) + " level: " + str(levs[v_ind] / 100) + " hPa")
pl.show()
pl.close()


#%% take_vi
# take_vi[take_vi > 0.0] = 0

l_num = 14
levels, norm, cmp = set_levs_norm_colmap(take_vi[mon, v_ind, :, :], l_num, quant=0.99)

x, y = np.meshgrid(lon_k, lat_k)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, take_vi[mon, v_ind, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("Ta Kernel in Wm$^{-2}$K$^{-1}$")
ax1.set_title("vint Ta Kernel Month " + str(mon) + " level: " + str(levs[v_ind]/100) + " hPa")
pl.show()
pl.close()


#%% take
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(take[mon, v_ind, :, :], l_num, quant=0.99)

x, y = np.meshgrid(lon_k, lat_k)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, take[mon, v_ind, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("Ta Kernel in Wm$^{-2}$K$^{-1}$")
ax1.set_title("Ta Kernel Month " + str(mon) + " level: " + str(v_ind))
pl.show()
pl.close()


#%% take_gf
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(take_gf[mon, v_ind, :, :], l_num, quant=0.99)

x, y = np.meshgrid(lon_k, lat_k)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, take_gf[mon, v_ind, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("Ta Kernel in Wm$^{-2}$K$^{-1}$")
ax1.set_title("gf Ta Kernel Month " + str(mon) + " level: " + str(v_ind))
pl.show()
pl.close()
"""


#%%
"""
la1 = 0
lo1 = 350
xi = closest_ind(lo1, lon_k)
yi = closest_ind(la1, lat_k)
xi2 = closest_ind(lo1, lon)
yi2 = closest_ind(la1, lat)

pl.plot(take_vi[6, :, yi, xi], levs/100, label="take_vi")
pl.plot(take[6, :, yi, xi], levs_k[6, :, yi, xi], label="take_vi")
pl.gca().invert_yaxis()
pl.title("lon: " + str(lon_k[xi]) + "  lat: " + str(lat_k[yi]))
pl.show()
pl.close()
"""

#%% plot some values (before stratosphere masking)
"""
fac = 1
if np.max(levs_k) > 5000:
    fac = 100
# end if
la1 = 30
lo1 = 50
xi = closest_ind(lo1, lon_k)
yi = closest_ind(la1, lat_k)
xi2 = closest_ind(lo1, lon)
yi2 = closest_ind(la1, lat)

try:
    vert_k = levs_k[0, :, yi, xi] / fac
except:
    vert_k = levs_k / fac
# end try except

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(14, 8))
ax1.plot(rem_take[0, :, yi2, xi2], levs/100, c="red", label="ta kernel vert. int. & remapped", marker="x")
ax1.plot(take_vi[0, :, yi, xi], levs/100, c="blue", label="ta kernel vert. int.", marker="+")
ax1.plot(take[0, :, yi, xi], vert_k, c="black", label="ta kernel original", marker=".")
ax1.legend()
pl.gca().invert_yaxis()
ax1.set_title(kl + " Ta Kernels\n" + "lat_k: " + str(lat_k[yi]) + "  lon_k: " + str(lon_k[xi]) + 
              "\nlat_m: " + str(lat[yi2]) + "  lon_m: " + str(lon[xi2]))
pl.show()
pl.close()

fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, figsize=(18, 14))

z = 7

p1 = ax1.imshow(take_vi[0, z, :, :], origin="lower", vmin=-2, vmax=0)
fig.colorbar(p1, ax=ax1)
ax1.set_title(str(levs[z]/100) + " hPa $-$ CMIP Pressure Levels (Index " + str(z) + ")")

p2 = ax2.imshow(rem_take[0, z, :, :], origin="lower", vmin=-2, vmax=0)
fig.colorbar(p2, ax=ax2)
ax2.set_title(str(levs[z]/100) + " hPa $-$ Remapped Kernel Pressure Levels (Index " + str(z_k) + ")")
pl.show()
pl.close()
"""

#%% mask the cells below surface level
"""
# only mask the cells if they are not already masked in model output
if np.ma.count_masked(ta_a4x[0, 0, :, :].compute()) == 0:

    print("\n\nGenerating the mask for the cells below surface pressure...\n")    
    
    # threshold of lowest allowed pressure level above surface pressure
    p_thres = 0
    
    p3d = np.zeros((1, len(levs), len(lat), len(lon)))
    p3d[:, :, :, :] = levs[None, :, None, None]
    
    # for abrupt4xCO2
    m_a4x = np.ones(np.shape(p3d))
    m_ind = np.where(p3d - ps_a4x[0, :, :][None, None, :, :] > p_thres)
    m_a4x[m_ind] = np.nan
    
    # for piControl
    m_pic = np.ones(np.shape(p3d))
    m_ind = np.where(p3d - ps_pic[0, :, :][None, None, :, :] > p_thres)
    m_pic[m_ind] = np.nan
    
    for i in np.arange(1, np.shape(ta_a4x)[0], 1):
        
        # for abrupt4xCO2
        m_a4x_new = np.ones(np.shape(p3d))
        m_ind = np.where(p3d - ps_a4x[i, :, :][None, None, :, :] > p_thres)
        m_a4x_new[m_ind] = np.nan
        
        m_a4x = da.concatenate([m_a4x, m_a4x_new], axis=0)
        
        # for piControl
        m_pic_new = np.ones(np.shape(p3d))
        m_ind = np.where(p3d - ps_pic[i, :, :][None, None, :, :] > p_thres)
        m_pic_new[m_ind] = np.nan
        
        m_pic = da.concatenate([m_pic, m_pic_new], axis=0)
        
    # end for i
    
    # rechunk the dask arrays
    m_a4x = da.rechunk(m_a4x, chunks=(30, len(levs), len(lat), len(lon)))
    m_pic = da.rechunk(m_pic, chunks=(30, len(levs), len(lat), len(lon)))
    
    
    # multiply the masks with the respective temperature arrays
    ta_a4x = ta_a4x * m_a4x
    ta_pic = ta_pic * m_pic

else:
    print("\n\nValues below surface pressure already masked...\n")    
# end if else
"""


#%% calculate the tas change
dts = np.zeros(np.shape(ts_a4x))
for yr, i in enumerate(np.arange(0, n_yr*12, 12)):
    dts[i:i+12, :, :] = ts_a4x[i:i+12, :, :] - ts_pic_run[:, yr, :, :]
# end for yr, i


#%% if necessary (as e.g. for ACCESS-ESM1.5) regrid the surface temperature to the air temperature grid
if len(lat2d) != len(lat):
    dts_re = np.zeros((12*n_yr, len(lat), len(lon)))
    
    lat_o = lat2d + 90
    lat_t = lat + 90
    
    print("\n\nRegridding surfaces temperature because of different latitudes from atmospheric variables...\n\n")
    for mon in np.arange(12*n_yr):
        dts_re[mon, :, :] = remap(lon, lat_t, lon, lat_o, dts[mon, :, :], verbose=True)
    # end for mon
    
    # replace the variabels with the regridded ones
    dts = dts_re
# end if


#%% generate a rough troposphere mask
if strat_mask:
    print("\n\nGenerating a stratosphere mask...\n")
    
    # the troposphere is assumed to be at 100 hPa at the equator and at 300 hPa at the poles
    weights = np.cos(lat / 180 * np.pi)
    
    p_tropo_zonal = 300 - 200*weights
    
    # set up the 4d pressure field
    p4d = np.zeros((12, len(levs), len(lat), len(lon)))
    p4d[:, :, :, :] = levs[None, :, None, None] / 100
    
    strat_mask = p4d > p_tropo_zonal[None, None, :, None]
    
    print("\nMultiplying the stratosphere mask on the kernels...\n")
    rem_take = np.array(rem_take) * np.array(strat_mask)
else:
    print("\nNo stratosphere mask applied.\n")
# end if else


#%% plot after startosphere mask
"""
fac = 1
if np.max(levs_k) > 5000:
    fac = 100
# end if
xi = 50
yi = 50
xi2 = closest_ind(lon_k[xi], lon)
yi2 = closest_ind(lat_k[yi], lat)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(14, 8))
ax1.plot(rem_take[0, :, yi2, xi2], levs/100, c="red", label="ta kernel vert. int. & remapped", marker="x")
ax1.plot(take_vi[0, :, yi, xi], levs/100, c="blue", label="ta kernel vert. int.", marker="+")
ax1.plot(take[0, :, yi, xi], levs_k / fac, c="black", label="ta kernel original", marker=".")
ax1.legend()
pl.gca().invert_yaxis()
ax1.set_title(kl + " Ta Kernels\n" + "lat_k: " + str(lat_k[yi]) + "  lon_k: " + str(lon_k[yi]) + 
              "\nlat_m: " + str(lat[yi2]) + "  lon_m: " + str(lon[xi2]))
pl.show()
pl.close()
"""


#%% plot annual and zonal means or slices of the kernel and the temperature values
"""
# ta
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(dta[-12:, :, :, :].compute(), l_num, quant=0.975)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(np.nanmean(np.nanmean(dta[-12:, :, :, :].compute(), axis=0), axis=-1), levels=levels, norm=norm, 
                  cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("ta change in K")
ax1.set_title("Ta Change Last Year Mean Zonal Mean " + nl.models_pl[mod])
pl.show()
pl.close()


#%% ta kernel orig
x, y = pl.meshgrid(lat_k, levs_k[0, :, 50, 50])
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(np.nanmean(take, axis=0), axis=-1), l_num, quant=0.99)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(x, y, np.nanmean(np.nanmean(take, axis=0), axis=-1), levels=levels, norm=norm, cmap=cmp, 
                  extend="both")
pl.gca().invert_yaxis()
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("kernel in Wm$^{-2}$K$^{-1}$100hPa$^{-1}$")
ax1.set_title(cs_p + "Original Ta Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()


#%% ta kernel vi
x, y = pl.meshgrid(lat_k, levs/100)
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(np.nanmean(take_vi, axis=0), axis=-1), l_num, quant=1)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(x, y, np.nanmean(np.nanmean(take_vi, axis=0), axis=-1), levels=levels, norm=norm, cmap=cmp, 
                  extend="both")
pl.gca().invert_yaxis()
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("kernel in Wm$^{-2}$K$^{-1}$100hPa$^{-1}$")
ax1.set_title(cs_p + "Vert. Int. Ta Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()


#%% ta kernel vi & remapped
x, y = pl.meshgrid(lat, levs/100)
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(np.nanmean(rem_take, axis=0), axis=-1), l_num, quant=1)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(x, y, np.nanmean(np.nanmean(rem_take, axis=0), axis=-1), levels=levels, norm=norm, cmap=cmp, 
                  extend="both")
pl.gca().invert_yaxis()
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("kernel in Wm$^{-2}$K$^{-1}$100hPa$^{-1}$")
ax1.set_title(cs_p + "Vert. Int. & Remapped Ta Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()


#%% ts
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(dts[-12:, :, :], axis=0), l_num, quant=0.975)

x, y = pl.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(11, 5), subplot_kw=dict(projection=proj))
p1 = ax1.contourf(x, y, np.nanmean(dts[-12:, :, :], axis=0), transform=ccrs.PlateCarree(), levels=levels, 
                  norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("ts change in K")
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.set_title("Ts Change Last Year Mean " + nl.models_pl[mod])
pl.show()
pl.close()

# ts kernel
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(rem_tske, axis=0), l_num, quant=0.975)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(11, 5), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, np.nanmean(rem_tske, axis=0), transform=ccrs.PlateCarree(), levels=levels, norm=norm, 
                  cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("ts kernel in Wm$^{-2}$K$^{-1}$")
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
ax1.set_title(cs_p + "Ts Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()
"""
# if pl_only:
#    raise Exception("Stop.")
# end if
#raise Exception("")

#%% repeat the ta and ts kernels along the month axis
take_rep = da.tile(da.from_array(rem_take), [n_yr, 1, 1, 1])
take_rep = da.rechunk(take_rep, chunks=(30, len(levs), len(lat), len(lon)))

tske_rep = da.tile(da.from_array(rem_tske), [n_yr, 1, 1])
tske_rep = da.rechunk(tske_rep, chunks=(30, len(lat), len(lon)))


#%% multiply the kernel on the ta change and on the ts change
ta_adj = take_rep * dta
ts_adj = tske_rep * dts


#%% calculate the Planck response by expanding tas to all heights and then multiplying on it the ta kernels
pl_adj = take_rep * dts[:, None, :, :]


#%% calculate the vertically integrated Planck response
pl_resp = np.nansum(pl_adj, axis=1)


#%% calculate the lapse rate radiative response
laps_resp = da.nansum(take_rep * (dta - dts[:, None, :, :]), axis=1)


#%% sum up the vertical levels
ta_adj_tot = da.nansum(ta_adj, axis=1)


#%% calculate the total temperature and Planck response
t_adj_tot = ta_adj_tot + ts_adj
pl_resp_tot = pl_resp + ts_adj


#%% a final test plot
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(sig[kl]*t_adj_tot[6, :, :].compute(), l_num, c_min=None, c_max=None, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, sig[kl]*t_adj_tot[6, :, :].compute(), levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("T adjustment in Wm$^{-2}$K$^{-1}$")
ax1.set_title(cs_p + "T Adjustment First Time Step " + kl + " Kernels\nGlobal Mean: " + 
              str(glob_mean(sig[kl]*t_adj_tot[6, :, :].compute(), lat, lon)) + " Wm$^{-2}$")
pl.show()
pl.close()

print(glob_mean(sig[kl]*t_adj_tot[6, :, :].compute(), lat, lon))
# raise Exception("")


#%% store the temperature, Planck and lapse rate radiative response in a netcdf file

print("\nGenerating nc file...\n")
out_name = f"Diff_to_CS{sfc_add}_ta_ts" + exp_s + "_" + kl + out_ad[kl] + "_" + nl.models[mod] + ".nc"
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", n_yr*12)
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
lon_nc = f.createVariable("lon", "f8", "lon")
tim_nc = f.createVariable("time", "f8", "time")
t_adj_nc = f.createVariable("t_adj", "f4", ("time", "lat", "lon"))
ts_adj_nc = f.createVariable("ts_adj", "f4", ("time", "lat", "lon"))
ta_adj_nc = f.createVariable("ta_adj", "f4", ("time", "lat", "lon"))

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
tim_nc[:] = np.arange(n_yr*12)
print("\nTry to store the adjustment (clear-sky minus all-sky) to temperature change...\n")
start = timeit.default_timer()
n = 100
for i in np.arange(0, n_yr * 12, n):
    t_adj_nc[i:(i+n), :, :] = t_adj_tot[i:(i+n), :, :].compute(scheduler='single-threaded')
    ts_adj_nc[i:(i+n), :, :] = ts_adj[i:(i+n), :, :].compute(scheduler='single-threaded')
    ta_adj_nc[i:(i+n), :, :] = ta_adj_tot[i:(i+n), :, :].compute(scheduler='single-threaded')
    ti_temp = timeit.default_timer()
    del_t = ti_temp - start
    frac = (i+n) / (n_yr * 12)
    remaining = np.round((del_t / frac - del_t) / 60, decimals=1)
    print_one_line(str(np.round((i+n) / (n_yr * 12) * 100, decimals=1)) + " %  " + str(remaining) + " min remain   ")
# end for i
tot_time = np.round((timeit.default_timer() - start) / 60, decimals=1)    
print("\nTime needed for storage: " + str(tot_time) + " min.")

t_adj_nc.units = "W m^-2"
ta_adj_nc.units = "W m^-2"
ts_adj_nc.units = "W m^-2"

lat_nc.units = "degrees_north"
lon_nc.units = "degrees_east"
tim_nc.units = ps_a4x_nc.variables["time"].units

t_adj_nc.description = (f"difference between long wave clear- and all-sky radiative response change at {toa_sfc} due " +
                        "to ta and ts change, i.e. temperature adjustment; positive-" + k_pud[kl])
ts_adj_nc.description = (f"difference between long wave clear and total sky radiative response change at {toa_sfc} " + 
                         "due to ts change, i.e. surface temperature adjustment; positive-" + k_pud[kl])
ta_adj_nc.description = (f"difference between long wave clear and total sky radiative response change at {toa_sfc} " +
                         "due to ta change, i.e. air temperature adjustment; positive-" + k_pud[kl])

# add attributes
desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values corresponding to the whole duration of the forcing experiment which is why they were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = (f"This file contains the {toa_sfc} radiative response difference " + 
                 f"between clear and total sky due to temperature change in the {exp_n} experiment for " +
                 nl.models_pl[mod] + ". This corresponds to the temperature adjustment for the cloud radiative\n" + 
                 "response as calculated via the Soden et al. (2008) adjustment method.\n" + 
                 "The values correspond to monthly means. Kernel reference: " + k_t[kl] + "." + desc_add)
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