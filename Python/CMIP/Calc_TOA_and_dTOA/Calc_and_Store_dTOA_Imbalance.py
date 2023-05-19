"""
Calculate and store monthly local delta (i.e., forced minus control experiment) TOA (all-sky and clear-sky) imbalance 
(dtoa).

Be sure to change data_path and out_path.
"""


#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
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
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti


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
from Classes.Class_MidpointNormalize import MidpointNormalize
from dask.distributed import Client


#%% establish connection to the client
c = Client("localhost:8786")


#%% Include clear-sky? Should be True if clear-sky data are available and False if the are not.
with_cs = True


#%% set the runnning mean
run = 21


#%% length of the experiment in years
n_yr = 150


#%% set the experiment - either "4x" or "2x"
try:
    exp = sys.argv[3]
except:
    # exp = "1pctCO2"
    exp = "abrupt-4xCO2"
# end try except


#%% CMIP version and data path
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except

if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
# end if elif


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 18
# end try except

print(nl.models_pl[mod])

try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[4]
except:
    ensemble_b = "r1i1p1f1"  
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[5]
except:
    ensemble_f = "r1i1p1f1"    
# end try except

ensemble_b_d = ""
ensemble_f_d = ""
gr_lab = ""
# grid label only exists in CMIP6, if CMIP is 5 this will be replaced with an empty string
if cmip == "CMIP6":
    gr_lab = "_gn"
    if nl.models[mod] in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", 
                          "EC-Earth3-AerChem", "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
        gr_lab = "_gr"
    elif nl.models[mod] in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5", "KIOST-ESM"]:
        gr_lab = "_gr1"
    # end if

    # handle the ensemble in the file name    
    if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # base experiment ensemble (usually piControl)
        ensemble_b_d = "_" + ensemble_b
    # end if
    if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # forced experiments ensemble (usually abrupt-4xCO2)
        ensemble_f_d = "_" + ensemble_f
    # end if
    
# end if


#%% generate a suffix to the decsription if the clear-sky data are not available
cs_add = ""
if with_cs:
    cs_add = " NOTE THAT CLEAR-SKY IS NOT AVAILABLE YET FOR THIS MODEL!"
# end if    


#%% select the forced experiment file name
#if exp == "4x":
#    f_n = nl.a4x_f[mod]
#elif exp == "2x":
#    f_n = nl.a2x_f[mod]
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


#%%  set paths
# --> must contain toa files for piControl as well as a forcing experiment (e.g., abrupt-4xCO2)
# --> must also contain a directory AddFiles_2d_piC wit the respective files as created by Extend_piControl_2d.py if
#     necessary
data_path = ""

# output path
out_path = ""


#%% load nc files
f_a4x_add = nl.models_n[mod] + "_" + exp + ensemble_f_d
f_piC_add = nl.models_n[mod] + "_piControl" + ensemble_b_d

toa_4x_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + "/toa_Amon_" + f_a4x_add + "*")[0])
toa_pi_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + "/toa_Amon_" + f_piC_add + "*")[0])

if with_cs:
    toacs_4x_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + "/toacs_Amon_" + f_a4x_add + "*")[0])
    toacs_pi_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + "/toacs_Amon_" + f_piC_add + "*")[0])
# end if


#%% extract the period from the files
per_sta = glob.glob(data_path + "Data/" + nl.models[mod] + "/toa_Amon_" + f_a4x_add + "*")[0][-16:-9]
per_sto = f'{int(per_sta[:4]) + n_yr - 1:04}12'

period = f'{per_sta}{per_sto}'


#%% get the branch year
if cmip == "CMIP6":
    bt_chi = toa_4x_nc.getncattr("branch_time_in_child")
    bt_par = toa_4x_nc.getncattr("branch_time_in_parent")
    par_exp = toa_4x_nc.getncattr("parent_experiment_id")
    par_tu = toa_4x_nc.getncattr("parent_time_units")
    # var_lab = toa_4x_nc.getncattr("variant_label")
elif cmip == "CMIP5":
    bt = toa_4x_nc.getncattr("branch_time")
# end if elif


#%% get the variant label/ensemble
if cmip == "CMIP6":
    try:
        var_lab_b = toa_pi_nc.getncattr("variant_label")
        var_lab_f = toa_4x_nc.getncattr("variant_label")
    except:
        var_lab_b = "r1i1p1f1"
        var_lab_f = "r1i1p1f1"
    # end try except
# end if


#%% get the values and calculate the TOA net imbalance changes
toa_4x = da.ma.masked_array(toa_4x_nc.variables["toa"])[0:(12*n_yr), :, :]
toa_pi = da.ma.masked_array(toa_pi_nc.variables["toa"])

if with_cs:
    toa_cs_4x = da.ma.masked_array(toacs_4x_nc.variables["toacs"])[0:(12*n_yr), :, :]
    toa_cs_pi = da.ma.masked_array(toacs_pi_nc.variables["toacs"])
# end if


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
add_yrs_b = False
if ((da.shape(toa_pi)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print("\npiControl run after branch year is at least 10 years longer than abrupt4xCO2. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
    add_yrs_b = True
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print("\npiControl run after branch year is fewer than 10 years longer than abrupt4xCO2. Using original" + 
          " b_end_ind: " + str(b_end_ind) + "\n")
    add_yrs_b = True
# end if

toa_pi = toa_pi[b_sta_ind:b_end_ind, :, :]

if with_cs:
    toa_cs_pi = toa_cs_pi[b_sta_ind:b_end_ind, :, :]
# end if    


#%%  load lat and lon
lat = toa_4x_nc.variables["lat"][:]
lon = toa_4x_nc.variables["lon"][:]

# lat_rg = toa_regrid_nc.variables["lat"][:]
# lon_rg = toa_regrid_nc.variables["lon"][:]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
try:
    f_toa = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toa_*AddBe.nc"), key=str.casefold)
    toa_pi = da.concatenate([da.ma.masked_array(Dataset(f_toa[-1]).variables["toa"], lock=True), toa_pi], axis=0)

    if with_cs:
        f_toacs = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toacs_*AddBe.nc"), 
                         key=str.casefold)
        toa_cs_pi = da.concatenate([da.ma.masked_array(Dataset(f_toacs[-1]).variables["toacs"], lock=True), toa_cs_pi], 
                                   axis=0)
    # end if        
    
    print("\nAdded AddBe file. New piControl shape: " + str(da.shape(toa_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_toa = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toa_*AddAf.nc"), key=str.casefold)
    toa_pi = da.concatenate([toa_pi, da.ma.masked_array(Dataset(f_toa[-1]).variables["toa"], lock=True)], axis=0)

    if with_cs:
        f_toacs = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toacs_*AddAf.nc"), 
                         key=str.casefold)
        toa_cs_pi = da.concatenate([toa_cs_pi, da.ma.masked_array(Dataset(f_toacs[-1]).variables["toacs"], lock=True)], 
                                   axis=0)
    # end if
    
    print("\nAdded AddAf file. New piControl shape: " + str(da.shape(toa_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except


#%% calculate the running means of the piControl

# NOTE: This add_yrs stuff still works even after I switched to producing the dedicated AddBe and AddAf files but it is
#       basically unnecessary now. I'll leave it in for now as a check: add_yrs HAS to be 0. 

# how many "additional years" are needed?
add_yrs = n_yr - (int(np.shape(toa_pi)[0] / 12) - (run-1))
print("add_yrs = " + str(add_yrs))

if add_yrs > 0:
    add_b = int(add_yrs/2)  # add before
    add_a = -(int(add_yrs/2) + (add_yrs % 2))  # add after
else:
    add_b = 0
    add_a = None
# end if else    


#%% store the result as a netcdf file
out_name = ("dtoa_as_cs_Amon_" + nl.models_n[mod] + "_" + exp + "_piC21Run_" + ensemble_f + gr_lab + "_" + 
            period + ".nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
if cmip == "CMIP6":
    f.setncattr("branch_time_in_child", bt_chi)
    f.setncattr("branch_time_in_parent", bt_par)
    f.setncattr("parent_experiment_id", par_exp)
    f.setncattr("parent_time_units", par_tu)
    f.setncattr("variant_label_base", var_lab_b)
    f.setncattr("variant_label_forced", var_lab_f)
elif cmip == "CMIP5":
    f.setncattr("branch_time", bt)
# end if elif

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("years", n_yr)
f.createDimension("months", 12)

# create the wind speed variable
dtoa_as_nc = f.createVariable("dtoa_as", "f4", ("months", "years", "lat", "lon"))
toa_pi_as_nc = f.createVariable("toa_as_pi_run", "f4", ("months", "years", "lat", "lon"))

if with_cs:
    dtoa_cs_nc = f.createVariable("dtoa_cs", "f4", ("months", "years", "lat", "lon"))
    toa_pi_cs_nc = f.createVariable("toa_cs_pi_run", "f4", ("months", "years", "lat", "lon"))
# end if    

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat

print("\nCalculating running means...\n")
test = np.zeros((12, n_yr, len(lat), len(lon)))
for mon in np.arange(12):
    print("Starting month " + str(mon) + " all-sky...")
    toa_pi_as_temp = run_mean_da(toa_pi[mon::12, :, :], running=run)
    toa_pi_as_nc[mon, add_b:add_a, :, :] = toa_pi_as_temp
    
    print("Running means calculated and stored. Continuing to calculate and store the changes all-sky...\n")
    test[mon, add_b:add_a, :, :] = (toa_4x[mon::12, :, :][add_b:add_a, :, :] - toa_pi_as_temp)
    dtoa_as_nc[mon, add_b:add_a, :, :] = (toa_4x[mon::12, :, :][add_b:add_a, :, :] - toa_pi_as_temp)
    
    if with_cs:
        print("Starting month " + str(mon) + " clear-sky...")
        toa_pi_cs_temp = run_mean_da(toa_cs_pi[mon::12, :, :], running=run)
        toa_pi_cs_nc[mon, add_b:add_a, :, :] = toa_pi_cs_temp
        
        print("Running means calculated and stored. Continuing to calculate and store the changes clear-sky...\n")
        dtoa_cs_nc[mon, add_b:add_a, :, :] = (toa_cs_4x[mon::12, :, :][add_b:add_a, :, :] - toa_pi_cs_temp)
    # end if
        
# end for mon

add_desc = ""
if add_yrs_b:
    add_desc = (" Because the piControl experiment did not extend at least 10 years before or after (or both) beyond\n" +
                "the " + exp + " experiment it was extended via a linear trend calculated from the last/first 10\n" +
                "years of the piControl run (per grid cell!).")
# end if

dtoa_as_nc.description = ("all-sky TOA imbalance change from 21-year running mean piControl to " + exp)
toa_pi_as_nc.description = "21-year running mean piControl all-sky TOA imbalance"

if with_cs:
    dtoa_cs_nc.description = ("clear-sky TOA imbalance change from 21-year running mean piControl to " + exp)
    toa_pi_cs_nc.description = "21-year running mean piControl clear-sky TOA imbalance"
# end if

# add attributes
f.description = ("All-sky and clear-sky TOA imbalance change file of the " + exp + " experiment calculated with\n" +
                 "respect to a 21-year running mean of the piControl simulation. The file also contains this " +
                 "21-year running mean of the piControl simulation." + add_desc + cs_add + " Model: " + 
                 nl.models_pl[mod])
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"

# close the dataset
f.close()