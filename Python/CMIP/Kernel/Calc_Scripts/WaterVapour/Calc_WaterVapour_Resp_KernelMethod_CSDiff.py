"""
Calculate the water vapour radiative "cloud adjustment" for CMIP members via the kernel method.

NOTE: To execute this script (status: 25.06.2020) first execute the script Calc_dQdT.py for the respective model and 
      the respective Q properties (dQ or dlogQ).

Approach: "Ignore" the NaN values at levels below surface pressure and in
          case the respective model has temperature values at levels below
          surface pressure convert them to NaN.
          
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
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti
import timeit
import Ngl as ngl
import geocat.ncomp as geoc
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Interp4d import interp4d, interp3d
from Functions.Func_Flip import flip
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap


#%% dlog(Q)/dT or dQ/dT?
dlogq = True

# change some parameters accordingly
dq_str = "dQ"
if dlogq:
    dq_str = "dlogQ"
# end if


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
    exp = "a4x"
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
k_as_fn = {"Sh08":{"lw":"CAM3_wv_lw" + k_as_ad[kl] + "_kernel.nc", "sw":"CAM3_wv_sw" + k_as_ad[kl] + "_kernel.nc"}, 
           "So08":{"lw":"TOA_GFDL_Kerns.nc", "sw":"TOA_GFDL_Kerns.nc"}, 
           "BM13":{"lw":state_fn[k_st] + "_kernel_mm_1950.nc", "sw":state_fn[k_st] + "_kernel_mm_1950.nc"},
           "H17":{"lw":"RRTMG_wv_lw_toa_" + k_as_ad[kl] + "_highR.nc", 
                  "sw":"RRTMG_wv_sw_toa_" + k_as_ad[kl] + "_highR.nc"}, 
           "P18":{"lw":"q.kernel.nc", "sw":"q.kernel.nc"},
           "S18":{"lw":"HadGEM2_lw_TOA_L17.nc", "sw":"HadGEM2_sw_TOA_L17.nc"}}
k_cs_fn = {"Sh08":{"lw":"CAM3_wv_lw" + k_cs_ad[kl] + "_kernel.nc", "sw":"CAM3_wv_sw" + k_cs_ad[kl] + "_kernel.nc"}, 
           "So08":{"lw":"TOA_GFDL_Kerns.nc", "sw":"TOA_GFDL_Kerns.nc"}, 
           "BM13":{"lw":state_fn[k_st] + "_kernel_mm_1950.nc", "sw":state_fn[k_st] + "_kernel_mm_1950.nc"},
           "H17":{"lw":"RRTMG_wv_lw_toa_" + k_cs_ad[kl] + "_highR.nc", 
                  "sw":"RRTMG_wv_sw_toa_" + k_cs_ad[kl] + "_highR.nc"}, 
           "P18":{"lw":"q.kernel.nc", "sw":"q.kernel.nc"},
           "S18":{"lw":"HadGEM2_lw_TOA_L17.nc", "sw":"HadGEM2_sw_TOA_L17.nc"}}

# set kernel variable names (maybe this is not the most economic way...)
if toa_sfc == "TOA":
    k_as_vn = {"Sh08":{"lw":"kernel_p", "sw":"kernel_p"}, 
               "So08":{"lw":"lw" + k_as_vad[kl] + "_q", "sw":"sw" + k_as_vad[kl] + "_q"}, 
               "BM13":{"lw":"dRq_tra" + k_as_vad[kl] + "0", "sw":"dRq_sra" + k_as_vad[kl] + "0"},
               "H17":{"lw":"lwkernel", "sw":"swkernel"}, 
               "P18":{"lw":"FLNT" + k_as_vad[kl], "sw":"FSNT" + k_as_vad[kl]},
               "S18":{"lw":"q_lw" + k_as_vad[kl], "sw":"q_sw" + k_as_vad[kl]}}
    k_cs_vn = {"Sh08":{"lw":"kernel_p", "sw":"kernel_p"}, 
               "So08":{"lw":"lw" + k_cs_vad[kl] + "_q", "sw":"sw" + k_cs_vad[kl] + "_q"}, 
               "BM13":{"lw":"dRq_tra" + k_cs_vad[kl] + "0", "sw":"dRq_sra" + k_cs_vad[kl] + "0"},
               "H17":{"lw":"lwkernel", "sw":"swkernel"}, 
               "P18":{"lw":"FLNT" + k_cs_vad[kl], "sw":"FSNT" + k_cs_vad[kl]},
               "S18":{"lw":"q_lw" + k_cs_vad[kl], "sw":"q_sw" + k_cs_vad[kl]}}
if toa_sfc == "SFC":
    k_as_vn = {"Sh08":{"lw":"kernel_p", "sw":"kernel_p"}, 
               "So08":{"lw":"lw" + k_as_vad[kl] + "_q", "sw":"sw" + k_as_vad[kl] + "_q"}, 
               "BM13":{"lw":"dRq_tra" + k_as_vad[kl] + "0", "sw":"dRq_sra" + k_as_vad[kl] + "0"},
               "H17":{"lw":"lwkernel", "sw":"swkernel"}, 
               "P18":{"lw":"FLNT" + k_as_vad[kl], "sw":"FSNT" + k_as_vad[kl]},
               "S18":{"lw":"q_lw" + k_as_vad[kl], "sw":"q_sw" + k_as_vad[kl]}}
    k_cs_vn = {"Sh08":{"lw":"kernel_p", "sw":"kernel_p"}, 
               "So08":{"lw":"lw" + k_cs_vad[kl] + "_q", "sw":"sw" + k_cs_vad[kl] + "_q"}, 
               "BM13":{"lw":"dRq_tra" + k_cs_vad[kl] + "0", "sw":"dRq_sra" + k_cs_vad[kl] + "0"},
               "H17":{"lw":"lwkernel", "sw":"swkernel"}, 
               "P18":{"lw":"FLNS" + k_cs_vad[kl], "sw":"FSNS" + k_cs_vad[kl]},
               "S18":{"lw":"q_lw" + k_cs_vad[kl], "sw":"q_sw" + k_cs_vad[kl]}}
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


#%% running mean years
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


#%% set paths

# model data path
# --> see code block "get the lists of and load files" for required files
data_path = ""

# kernel path
kern_path = f"{k_p[kl]}_Kernels/{k_ad[kl]}"

# output path
out_path = (f"Kernel_{toa_sfc}_RadResp/{exp_n}/{k_p[kl]}" +
            f"/01_Diff_to_CS/{nl.models[mod]}/{strat_path}/{out_path_ad[kl][k_st]}/")


#%% get the lists of and load files
dqdt_f_list = sorted(glob.glob(data_path + dq_str + "dT_Files" + exp_s + ensemble_f_d + "/*.nc"), key=str.casefold)

# ps files
ps_a4x_nc = Dataset(glob.glob(data_path + "ps_Amon_" + nl.models_n[mod] + "_" + exp_n + "_" + ensemble_f + "*")[0])
ps_pic_nc = Dataset(glob.glob(data_path + "ps_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])

# load in the respective first dQ/dT file
dqdt_nc = Dataset(dqdt_f_list[0])

# load the first dQ/dT file
dqdt = da.ma.masked_array(dqdt_nc.variables["dlogQdT"], lock=True)
dq = da.ma.masked_array(dqdt_nc.variables[dq_str], lock=True)


#%% load the kernel values

# kernels
q_lw_ke_nc = Dataset(kern_path + k_as_fn[kl]["lw"])
q_sw_ke_nc = Dataset(kern_path + k_as_fn[kl]["sw"])
q_lw_ke_cs_nc = Dataset(kern_path + k_cs_fn[kl]["lw"])
q_sw_ke_cs_nc = Dataset(kern_path + k_cs_fn[kl]["sw"])

# kernel latitudes (note that the "NA" part seems like rather bad programming...)
grid_d = {"lev":q_lw_ke_nc.variables[k_gvn[kl][0]][:],
          "lat":q_lw_ke_nc.variables[k_gvn[kl][1]][:], 
          "lon":q_lw_ke_nc.variables[k_gvn[kl][2]][:], 
          "NA":np.arange(1)}

# flip the necessary part of the grid info
grid_d[k_flip[kl]] = flip(grid_d[k_flip[kl]])

levs_k = grid_d["lev"]
lat_k = grid_d["lat"]
lon_k = grid_d["lon"]


#%% get the grid coordinates (vertical and horizontal)

# get the levels
levs = dqdt_nc.variables["plev"][:]

# get lat, lon, and levels for the model data
lat = dqdt_nc.variables["lat"][:]
lon = dqdt_nc.variables["lon"][:]
tim = ps_a4x_nc.variables["time"][:n_yr*12]

# load the (possibly different) lat and lon for the surface quantities
lat2d = ps_a4x_nc.variables["lat"][:]
lon2d = ps_a4x_nc.variables["lon"][:]


#%%  loop over the dlogQ and dQdT files and concatenate the values via dask
if len(dqdt_f_list) > 1:
    for i in np.arange(1, len(dqdt_f_list)):
        dqdt = da.concatenate([dqdt, da.ma.masked_array(Dataset(dqdt_f_list[i]).variables["dlogQdT"], lock=True)])
        dq = da.concatenate([dq, da.ma.masked_array(Dataset(dqdt_f_list[i]).variables[dq_str], lock=True)])
    # end for i
# end if

# print the dataset history of the last dataset loaded
print("\n\ndT, dQ, dQ/dT data set " + Dataset(dqdt_f_list[i]).history + "\n\n")

# reduce the arrays to the 150 years of the experiment and the corresponding control run
dqdt = dqdt[:(n_yr*12), :, :, :]
dq = dq[:(n_yr*12), :, :, :]

# rechunk the dask arrays
dqdt = da.rechunk(dqdt, chunks=(30, len(levs), len(lat), len(lon)))
dq = da.rechunk(dq, chunks=(30, len(levs), len(lat), len(lon)))


#%% load the ps fields as numpy arrays
ps_a4x = ps_a4x_nc.variables["ps"][:(n_yr*12), :, :]
ps_pic = ps_pic_nc.variables["ps"]


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
add_yrs = False  # this is used to determine if the piControl run (used in the Calc_dQdT script) had to be extended
#                  note that this is a rather quick-and-dirty solution...  
if ((da.shape(ps_pic)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print(f"\npiControl run after branch year is at least 10 years longer than {exp_n}. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print(f"\npiControl run after branch year is less than 10 years longer than {exp_n}. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")
    add_yrs = True
# end if

ps_pic = ps_pic[b_sta_ind:b_end_ind, :, :]


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

    
#%% load the kernels and the additional kernel data
qke_lw = (flip(np.ma.masked_invalid(q_lw_ke_cs_nc.variables[k_cs_vn[kl]["lw"]][:]), axis=k_flip[kl]) - 
          flip(np.ma.masked_invalid(q_lw_ke_nc.variables[k_as_vn[kl]["lw"]][:]), axis=k_flip[kl]))
qke_sw = (flip(np.ma.masked_invalid(q_sw_ke_cs_nc.variables[k_cs_vn[kl]["sw"]][:]), axis=k_flip[kl]) - 
          flip(np.ma.masked_invalid(q_sw_ke_nc.variables[k_as_vn[kl]["sw"]][:]), axis=k_flip[kl]))


#%% get the kernel pressure variables
if kl == "Sh08":
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke/100, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100

elif kl == "So08":
    ps_ke_nc = q_lw_ke_nc
    
    ps_ke = flip(np.ma.masked_invalid(ps_ke_nc.variables[k_ps_vn[kl]][:]), axis=k_flip[kl])
    ptop = np.min(levs_k)
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100    

elif kl == "BM13":  # BM13 kernels
    ps_ke_nc = q_lw_ke_nc
    pmid_nc = q_lw_ke_nc
    
    pdiff = flip(np.ma.masked_invalid(pmid_nc.variables["pdiff"][:]), axis=k_flip[kl]) / 100
    # pdiff = pmid_nc.variables["pdiff"][:, :, ::-1, :] / 100  # directly convert to hPa
    ps_ke = flip(np.ma.masked_invalid(ps_ke_nc.variables[k_ps_vn[kl]][:]), axis=k_flip[kl])
    
    hyam = q_lw_ke_nc.variables["hyam"][:]
    hybm = q_lw_ke_nc.variables["hybm"][:] 
    
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
    
    p0 = q_lw_ke_nc.variables["P0"][:]
    hyam = q_lw_ke_nc.variables["hyam"][:]
    hybm = q_lw_ke_nc.variables["hybm"][:] 
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
    # pdiff = np.array(geoc.dpres_plevel(levs_k, ps_ke/100, np.min(levs_k), msg=np.nan))
    pdiff = 1
    pdiff_vi = pdiff_vi / 100    
    levs_k = levs_k / 100
    
# end if elif


#%% interpolate the missing values of the ta kernel

# "grid fill"
qke_lw_gf = copy.deepcopy(qke_lw)
qke_sw_gf = copy.deepcopy(qke_sw)
print("\nFill the masked values of the ta kernel via lat-lon grid filling using scipy's interpolate.griddata...\n")
for m in range(12):
    print_one_line(str(m + 1) + "   ")
    for z in range(len(levs_k)):
        if np.sum(qke_lw[m, z, :, :].mask) > 0:  # Any masked values? If not jump to the next loop...
        
            xx, yy = np.meshgrid(lon_k, lat_k)
            
            # get only the valid values
            x1 = xx[~qke_sw[m, z, :, :].mask]
            y1 = yy[~qke_sw[m, z, :, :].mask]
            x2 = xx[~qke_lw[m, z, :, :].mask]
            y2 = yy[~qke_lw[m, z, :, :].mask]
            newarr = qke_sw[m, z, :, :][~qke_sw[m, z, :, :].mask]
            newarr2 = qke_lw[m, z, :, :][~qke_lw[m, z, :, :].mask]
            
            qke_sw_gf[m, z, :, :] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
            qke_lw_gf[m, z, :, :] = interpolate.griddata((x2, y2), newarr2.ravel(), (xx, yy), method='linear')
                       
        else:
            continue
        # end if else
   # end for z
# end for m

# interpolate the kernel to the CMIP pressure levels
qke_sw_vi = np.zeros((12, len(levs), len(lat_k), len(lon_k)))
qke_lw_vi = np.zeros((12, len(levs), len(lat_k), len(lon_k)))
print("\n\n\nInterpolate the kernel vertically to the CMIP pressure levels using my interp4d...\n")
for t in range(12):
    print_one_line(str(t + 1) + "   ")
    qke_sw_vi[t, :, :, :] = interp4d(t, levs_k, qke_sw_gf/pdiff, levs/100)
    qke_lw_vi[t, :, :, :] = interp4d(t, levs_k, qke_lw_gf/pdiff, levs/100)
# end for t


#%% plot some values
"""
z = 0

#vmax = np.nanmax([qke_sw[0, z, :, :], qke_sw_gf[0, z, :, :], qke_sw_vf[0, z, :, :]])
#vmin = np.nanmin([qke_sw[0, z, :, :], qke_sw_gf[0, z, :, :], qke_sw_vf[0, z, :, :]])
#vmax = 0
#vmin = -2
vmin = None
vmax = None

pl.imshow(qke_sw[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.show()
pl.close()

#pl.imshow(qke_sw_gf[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
#pl.colorbar()
#pl.show()
#pl.close()

pl.imshow(qke_sw_vi[0, z, :, :], origin="lower", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.show()
pl.close()
raise Exception("")
"""


#%% interpolate the kernels to the horizontal model grid

# first transform the lats and lons
lon_tr = lon
lat_tr = lat + 90
lon_o_tr = lon_k
lat_o_tr = lat_k + 90

# regrid the field
print("\n\nRegridding kernels to " + nl.models_pl[mod] + " grid using " + regrid_meth + "...\n")

rem_qke_lw = np.zeros((12, len(levs), len(lat), len(lon)))
rem_qke_sw = np.zeros((12, len(levs), len(lat), len(lon)))
# rem_pske = np.zeros((12, len(lat), len(lon)))

for t in range(12):

    print_one_line(str(t + 1) + "   ")

    for z in range(len(levs)):
        rem_qke_lw[t, z, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, qke_lw_vi[t, z, :, :])
        rem_qke_sw[t, z, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, qke_sw_vi[t, z, :, :])
    # end for z
# end for t


#%% multiply the remapped pressure differences onto the kernels
rem_qke_lw = rem_qke_lw * pdiff_vi
rem_qke_sw = rem_qke_sw * pdiff_vi


#%% mask the cells below surface level
"""
# only mask the cells if they are not already masked in model output
if (np.ma.count_masked(dq[0, 0, :, :].compute()) == 0) & (not np.any(np.isnan(dq[0, 0, :, :].compute()))):
    print("\n\nGenerating the mask for the cells below surface pressure...\n")    
    
    p3d = np.zeros((1, len(levs), len(lat), len(lon)))
    p3d[:, :, :, :] = levs[None, :, None, None]
    
    # for abrupt4xCO2
    m_a4x = np.ones(np.shape(p3d))
    m_ind = np.where(p3d - ps_a4x[0, :, :][None, None, :, :] > 0)
    m_a4x[m_ind] = np.nan
    
    # for piControl
    m_pic = np.ones(np.shape(p3d))
    m_ind = np.where(p3d - ps_pic[0, :, :][None, None, :, :] > 0)
    m_pic[m_ind] = np.nan
    
    for i in np.arange(1, np.shape(dq)[0], 1):
        
        # for abrupt4xCO2
        m_a4x_new = np.ones(np.shape(p3d))
        m_ind = np.where(p3d - ps_a4x[i, :, :][None, None, :, :] > 0)
        m_a4x_new[m_ind] = np.nan
        
        m_a4x = da.concatenate([m_a4x, m_a4x_new], axis=0)
        
        # for piControl
        m_pic_new = np.ones(np.shape(p3d))
        m_ind = np.where(p3d - ps_pic[i, :, :][None, None, :, :] > 0)
        m_pic_new[m_ind] = np.nan
        
        m_pic = da.concatenate([m_pic, m_pic_new], axis=0)
        
    # end for i
    
    # rechunk the dask arrays
    m_a4x = da.rechunk(m_a4x, chunks=(30, len(levs), len(lat), len(lon)))
    m_pic = da.rechunk(m_pic, chunks=(30, len(levs), len(lat), len(lon)))
    
    # multiply the masks with the respective humidity arrays
    # q_a4x = q_a4x * m_a4x
    # q_pic = q_pic * m_pic
    q_ch = dq * m_a4x * m_pic
    
else:
    print("\n\nValues below surface pressure already masked...\n")
    q_ch = dq
# end if else
"""
q_ch = dq


#%% generate a rough troposphere mask
if strat_mask:
    print("\nGenerating a stratosphere mask...\n")
    
    # the troposphere is assumed to be at 100 hPa at the equator and at 300 hPa at the poles
    weights = np.cos(lat / 180 * np.pi)
    
    p_tropo_zonal = 300 - 200*weights
    
    # set up the 4d pressure field
    p4d = np.zeros((12, len(levs), len(lat), len(lon)))
    p4d[:, :, :, :] = levs[None, :, None, None] / 100
    
    strat_mask = p4d > p_tropo_zonal[None, None, :, None]
    
    print("\nMultiplying the stratosphere mask on the kernels...\n")
    rem_qke_lw = np.array(rem_qke_lw) * np.array(strat_mask)
    rem_qke_sw = np.array(rem_qke_sw) * np.array(strat_mask)
else:
    print("\nNo stratosphere mask applied.\n")
# end if else


#%% plot annual and zonal means or slices of the kernel and the temperature values
"""
# dQ
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(dq[-12:, :, :, :].compute(), l_num, quant=0.99)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(np.nanmean(np.nanmean(dq[-12:, :, :, :].compute(), axis=0), axis=-1), levels=levels, norm=norm, 
                  cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("Q change")
ax1.set_title("Q Change Last Year Mean Zonal Mean " + nl.models_pl[mod])
pl.show()
pl.close()

 
#%% dQ/dT
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(dqdt[-12:, :, :, :].compute(), l_num, quant=0.99)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(np.nanmean(np.nanmean(dqdt[-12:, :, :, :].compute(), axis=0), axis=-1), levels=levels, norm=norm, 
                  cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("dQ/dT change in K$^{-1}$")
ax1.set_title("Q Change Last Year Mean Zonal Mean " + nl.models_pl[mod])
pl.show()
pl.close()


#%% Q LW kernel
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(np.nanmean(rem_qke_lw, axis=0), axis=-1), l_num, quant=1)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(np.nanmean(np.nanmean(rem_qke_lw, axis=0), axis=-1), levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("Q LW kernel in Wm$^{-2}$K$^{-1}$100hPa$^{-1}$")
ax1.set_title(cs_p + "Q LW Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()


#%% Q SW kernel
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(np.nanmean(np.nanmean(rem_qke_sw, axis=0), axis=-1), l_num, quant=0.975)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6))
p1 = ax1.contourf(np.nanmean(np.nanmean(rem_qke_sw, axis=0), axis=-1), levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
cb1.set_label("Q SW kernel in Wm$^{-2}$K$^{-1}$100hPa$^{-1}$")
ax1.set_title(cs_p + "Q SW Kernel Annual Mean Zonal Mean")
pl.show()
pl.close()
"""

#%% repeat the lw and sw kernels along the month axis
qke_lw_rep = da.tile(da.from_array(rem_qke_lw), [n_yr, 1, 1, 1])
qke_lw_rep = da.rechunk(qke_lw_rep, chunks=(30, len(levs), len(lat), len(lon)))

qke_sw_rep = da.tile(da.from_array(rem_qke_sw), [n_yr, 1, 1, 1])
qke_sw_rep = da.rechunk(qke_sw_rep, chunks=(30, len(levs), len(lat), len(lon)))


#%% multiply the kernel on the ta change and on the ts change
#   --> NEW METHODOLOGY: multiply the separately calculated dQ/dT here
q_lw_toa = qke_lw_rep / dqdt * q_ch
q_sw_toa = qke_sw_rep / dqdt * q_ch


#%% sum up the vertical levels
q_lw_toa_tot = da.nansum(q_lw_toa, axis=1)
q_sw_toa_tot = da.nansum(q_sw_toa, axis=1)


#%% a final test plot
l_num = 14
levels, norm, cmp = set_levs_norm_colmap(sig[kl]*q_lw_toa_tot[6, :, :].compute(), l_num, c_min=None, c_max=None, 
                                         quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, sig[kl]*q_lw_toa_tot[6, :, :].compute(), levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label("Q LW adjustment in Wm$^{-2}$K$^{-1}$")
ax1.set_title("Q LW Adjustment First Time Step " + kl + " Kernels\nGlobal Mean: " + 
              str(glob_mean(sig[kl]*q_lw_toa_tot[6, :, :].compute(), lat, lon)) + " Wm$^{-2}$")
pl.show()
pl.close()

print(glob_mean(sig[kl]*q_lw_toa_tot[6, :, :].compute(), lat, lon))
# raise Exception("")


#%% store the water vapour radiative response in a netcdf file

print("\nGenerating nc file...\n")
out_name = f"Diff_to_CS{sfc_add}_q" + exp_s + "_" + kl + out_ad[kl] + "_" + nl.models[mod] + ".nc"
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", n_yr*12)
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
lon_nc = f.createVariable("lon", "f8", "lon")
tim_nc = f.createVariable("time", "f4", "time")
q_lw_resp_nc = f.createVariable("q_lw_adj", "f4", ("time", "lat", "lon"))
q_sw_resp_nc = f.createVariable("q_sw_adj", "f4", ("time", "lat", "lon"))

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
tim_nc[:] = np.arange(n_yr*12)
print("\nTrying to store the clear-sky all-sky difference in response to water vapour change...\n")
start = timeit.default_timer()
n = 90
for i in np.arange(0, n_yr*12, n):
    q_lw_resp_nc[i:(i+n), :, :] = q_lw_toa_tot[i:(i+n), :, :].compute(scheduler='single-threaded')
    q_sw_resp_nc[i:(i+n), :, :] = q_sw_toa_tot[i:(i+n), :, :].compute(scheduler='single-threaded')
    # print("Months " + str(i+1) + " to " + str(i + n_yr) + f" of {str(n_yr*12)} stored")
    ti_temp = timeit.default_timer()
    del_t = ti_temp - start
    frac = (i+n) / (n_yr*12)
    remaining = np.round((del_t / frac - del_t) / 60, decimals=1)
    print_one_line(str(np.round((i+n) / (n_yr*12) * 100, decimals=1)) + " %  " + str(remaining) + " min remain   ")
# end for i
tot_time = np.round((timeit.default_timer() - start) / 60, decimals=1)    
print("\nTime needed for storage: " + str(tot_time) + " min.")

q_lw_resp_nc.units = "W m^-2"
q_sw_resp_nc.units = "W m^-2"

lat_nc.units = "degrees_north"
lon_nc.units = "degrees_east"
tim_nc.units = ps_a4x_nc.variables["time"].units

q_lw_resp_nc.description = (f"difference between long wave clear and total sky radiative response change at {toa_sfc}" +
                            " due to " + dq_str + ", i.e. the lw " + dq_str + " adjustment; positive-" + k_pud[kl])
q_sw_resp_nc.description = (f"difference between short wave clear and total sky radiative response change at {toa_sfc}"+
                            " due to " + dq_str + ", i.e. the sw " + dq_str + " adjustment; positive-down")

# add attributes
desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values corresponding to the whole duration of the forcing experiment which is why they were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = (f"This file contains the {toa_sfc} long- and short-wave radiative response difference " + 
                 f"between clear and total sky due to water vapour change in the {exp_n} experiment for " +
                 nl.models_pl[mod] + ". This corresponds to the water vapour adjustment for the cloud radiative\n" + 
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

