"""
Script for calculating and storing the tas change in the abrupt-4xCO2 (or abrupt-2xCO2) experiment for the CMIP models.
The change is calculated with respect to (1) a 21-year running mean over the piControl run and (2) a linear trend over
the piControl run (NOT the 21-year running mean of that run but annual means)

Be sure to change data_path and out_path as well as the targ_path.
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
from Functions.Func_AnMean import an_mean, an_mean_verb
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Regrid_Data import remap as remap2
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_Print_In_One_Line import print_one_line


#%% select the variable: ts or tas
try:
    var = sys.argv[6]
except:    
    var = "tas"
# end try except    

if var == "ts":    
    var_p = "Ts"
if var == "tas":
    var_p = "Tas"
# end if


#%% set running mean
run = 21


#%% number of years to take from the abrupt4xCO2 experiment (usually 150)
n_yr = 150  #  --> 150 for abrupt-4xCO2; 140 for 1pctCO2


#%% set the experiment - either "4x" or "2x"
try:
    exp = sys.argv[3]
except:    
    # exp = "1pctCO2"
    exp = "abrupt-4xCO2"
# end try except


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[4]
except:
    ensemble_b = "r1i1p1f2"    
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[5]
except:
    ensemble_f = "r1i1p1f2"
# end try except


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 48
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
# end if elif

print(nl.models_pl[mod])


#%%  grid label and ensemble
# grid label only exists in CMIP6, if CMIP is 5 this will be replaced with an empty string
ensemble_b_d = ""
ensemble_f_d = ""
gr_lab = ""
if cmip == "CMIP6":
    gr_lab = "_gn"
    if nl.models[mod] in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3-AerChem",
                          "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
        gr_lab = "_gr"
    elif nl.models[mod] in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5"]:
        gr_lab = "_gr1"
    # end if

    # handle the ensemble in the file name    
    if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # base experiment ensemble (usually piControl)
        ensemble_b_d = "_" + ensemble_b
    # end if
    if (ensemble_f != "r1i1p1f1") & (ensemble_f != "r1i1p1"):  # forced experiments ensemble (usually abrupt-4xCO2)
        ensemble_f_d = "_" + ensemble_f
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
# --> must contain t(a)s files for piControl as well as a forcing experiment (e.g., abrupt-4xCO2) 
# --> must also contain a directory AddFiles_2d_piC wit the respective files as created by Extend_piControl_2d.py if
#     necessary
data_path = ""

# output path
out_path = "/GlobMean_{var}_piC_Run/" + exp + "/"


#%% get the lat and lon for the target grid
targ_path = ""  # path where the target grid can be taken from
lat_rg = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lat"][:]
lon_rg = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lon"][:]


#%% load the netcdf files

f_a4x_add = nl.models_n[mod] + "_" + exp + ensemble_f_d
f_piC_add = nl.models_n[mod] + "_piControl" + ensemble_b_d

# tas files
ts_a4x_nc = Dataset(glob.glob(data_path + var + "_Amon_" + f_a4x_add + "*")[0])
ts_pic_nc = Dataset(glob.glob(data_path + var + "_Amon_" + f_piC_add + "*")[0])


#%% extract the period from the files
period = glob.glob(data_path + var + "_Amon_" + nl.models_n[mod] + "_" + exp + "_" + ensemble_f + "*")[0][-16:-9] + \
         glob.glob(data_path + var + "_Amon_" + nl.models_n[mod] + "_" + exp + "_" + ensemble_f + "*")[0][-9:-3]

per_sta = glob.glob(data_path + f"/{var}_Amon_" + f_a4x_add + "*")[0][-16:-9]
per_sto = f'{int(per_sta[:4]) + n_yr - 1:04}12'

period = f'{per_sta}{per_sto}'


#%% get the values

# get lat, lon, and levels for the model data
lat = ts_a4x_nc.variables["lat"][:]
lon = ts_a4x_nc.variables["lon"][:]

# tas values
ts_a4x = da.from_array(ts_a4x_nc.variables[var], lock=True)[:(n_yr*12), :, :]

ts_pic = da.from_array(ts_pic_nc.variables[var], lock=True)


#%% get the variant label/ensemble
if cmip == "CMIP6":
    try:
        var_lab_b = ts_pic_nc.getncattr("variant_label")
        var_lab_f = ts_a4x_nc.getncattr("variant_label")
    except:
        var_lab_b = "r1i1p1f1"
        var_lab_f = "r1i1p1f1"
    # end try except        
# end if


#%% get the branch year
if cmip == "CMIP6":
    bt_chi = ts_a4x_nc.getncattr("branch_time_in_child")
    bt_par = ts_a4x_nc.getncattr("branch_time_in_parent")
    par_exp = ts_a4x_nc.getncattr("parent_experiment_id")
    par_tu = ts_a4x_nc.getncattr("parent_time_units")
    # var_lab = toa_4x_nc.getncattr("variant_label")
elif cmip == "CMIP5":
    bt = ts_a4x_nc.getncattr("branch_time")
# end if elif


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
if ((da.shape(ts_pic)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print("\npiControl run after branch year is at least 10 years longer than abrupt4xCO2. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print("\npiControl run after branch year is less than 10 years longer than abrupt4xCO2. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")    
# end if

ts_pic = ts_pic[b_sta_ind:b_end_ind, :, :]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    f_add = sorted(glob.glob(data_path + "/AddFiles_2d_piC/" + var + f"*{ensemble_b_d}_AddBe.nc"), key=str.casefold)
    ts_pic = da.concatenate([da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True), ts_pic], axis=0)
    print("\nAdded AddBe file. New piControl shape:" + str(da.shape(ts_pic)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_add = sorted(glob.glob(data_path + "/AddFiles_2d_piC/" + var + f"*{ensemble_b_d}_AddAf.nc"), key=str.casefold)
    ts_pic = da.concatenate([ts_pic, da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True)], axis=0)
    print("\nAdded AddAf file. New piControl shape:" + str(da.shape(ts_pic)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except    


#%% calculate the running mean
ts_a4x_rea = np.zeros((12, n_yr, len(lat), len(lon)))
ts_pic_run = np.zeros((12, int((np.shape(ts_pic)[0])/12 - (run-1)), len(lat), len(lon)))
print("\nCalculating running means...\n")
for mon in np.arange(12):
    print(mon)
    ts_pic_run[mon, :, :, :] = run_mean_da(ts_pic[mon::12, :, :], running=run)
    ts_a4x_rea[mon, :, :, :] = ts_a4x[mon::12, :, :]  # "rearranged" a4x
# end for i


#%% calculate the annual means
print("\nCalculating annual means...\n")
ts_a4x_an = da.mean(ts_a4x_rea, axis=0)
ts_pic_an = da.mean(ts_pic_run, axis=0)
#ts_ch_an = da.mean(ts_ch, axis=0)

# calculate the annual means of the original piControl values
ts_pic_an_orig = an_mean_verb(ts_pic)

    
#%% calculate global means
print("\nCalculating global means...\n")
ts_a4x_gl = glob_mean(ts_a4x_an, lat, lon)
ts_pic_gl = glob_mean(ts_pic_an, lat, lon)
ts_ch_gl = ts_a4x_gl - ts_pic_gl

# calculate the annual means of the original piControl values
ts_pic_gl_orig = glob_mean(ts_pic_an_orig, lat, lon)


#%% calculate the linear trend over the 150 years of piControl corresponding to the abrupt-4xCO2
sl, yi, r, p = lr(np.arange(len(ts_pic_gl_orig[10:-10])), ts_pic_gl_orig[10:-10])[:4]

# calculate the change with respect to the linear piControl trend
dts_lt = (ts_a4x_gl - (np.arange(len(ts_pic_gl_orig[10:-10])) * sl + yi)).compute()


#%% store the global mean values in a netcdf file
print("\nGenerating nc file...\n")
f = Dataset(out_path + "GlobalMean_" + exp + "_" + var + "_piC" + str(run) + "run_" + nl.models[mod] + ".nc", 
            "w", format="NETCDF4")

# set the variant label
if cmip == "CMIP6":
    f.setncattr("variant_label_base", var_lab_b)
    f.setncattr("variant_label_forced", var_lab_f)
# end if

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", np.shape(ts_a4x_an)[0])
f.createDimension("time170", len(ts_pic_gl_orig))

# create the variables
tas_ch_gl_nc = f.createVariable(var + "_ch", "f4", "time")
tas_ch_lt_gl_nc = f.createVariable(var + "_ch_lt", "f4", "time")
tas_a4x_gl_nc = f.createVariable(var + "_" + exp, "f4", "time")
tas_pic_run_nc = f.createVariable(var + "_pic_run", "f4", "time")
tas_pic_gl_nc = f.createVariable(var + "_pic", "f4", "time170")

# pass the data into the variables
tas_ch_gl_nc[:] = ts_ch_gl
tas_ch_lt_gl_nc[:] = dts_lt
tas_a4x_gl_nc[:] = ts_a4x_gl
tas_pic_run_nc[:] = ts_pic_gl
tas_pic_gl_nc[:] = ts_pic_gl_orig

tas_ch_gl_nc.units = "K"
tas_ch_lt_gl_nc.units = "K"
tas_a4x_gl_nc.units = "K"
tas_pic_gl_nc.units = "K"

tas_ch_gl_nc.description = ("global mean annual mean " + var + " change calculated " + 
                            "by subtracting " + str(run) + "-year run mean monthly " + var + " values " +
                            "on the model grid of the piControl run from " + 
                            "the corresponding values of the " + exp +
                            " run; subsequently the values were averaged " +
                            "to annual means and finally to global means")
tas_ch_lt_gl_nc.description = ("global mean annual mean " + var + " change calculated by subtracting a linear trend " + 
                               "of the " + "piControl run from the corresponding values of the " + exp + 
                               " run.")
tas_a4x_gl_nc.description = ("global mean annual mean " + var + " values of the " +
                             exp + " experiment")
tas_pic_run_nc.description = ("global mean " + str(run) + "-year running mean of the piControl experiment")
tas_pic_gl_nc.description = ("global mean annual mean " + var + " values of the piControl experiment " + 
                             "(170 years, i.e. the " + 
                             exp + " run corresponds to " + var + "_pic[10:-10] in python code)")

# add attributes
desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values completely corresponding to the forcing experiment which is why the were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = ("This file contains the " + var + " values of the " + exp +
                 " and the piControl experiments as well as their difference " +
                 " for " + nl.models_pl[mod] + 
                 ". The values correspond to globally averaged annual means. Note that the piControl value have been " +
                 "climatologically averaged over a running period of " + str(run) + " years." + desc_add)
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


#%% store the local monthly values in a netcdf file
print("\nGenerating nc file...\n")
out_name = ("d" + var + "_Amon_" + nl.models_n[mod] + "_" + exp + "_piC21Run_" + ensemble_f + gr_lab + "_" + period + 
            ".nc")
f = Dataset(data_path + out_name, "w", format="NETCDF4")

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
f.createDimension("years", n_yr)
f.createDimension("months", 12)
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))
f.createDimension("lon_rg", len(lon_rg))
f.createDimension("lat_rg", len(lat_rg))

# create the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")
lon_rg_nc = f.createVariable("lon_rg", "f8", "lon_rg")
lat_rg_nc = f.createVariable("lat_rg", "f8", "lat_rg")

dtas_nc = f.createVariable(var + "_ch", "f4", ("months", "years", "lat", "lon"))
dtas_rg_nc = f.createVariable(var + "_ch_rg", "f4", ("months", "years", "lat_rg", "lon_rg"))

# pass the grid data into the variables
longitude[:] = lon
latitude[:] = lat
lon_rg_nc[:] = lon_rg
lat_rg_nc[:] = lat_rg 

# pass the data into the variables
if nl.models[mod] != "FGOALS-g3":
    lat_o = lat + 90
    lat_t = lat_rg + 90 
    for yr in np.arange(n_yr):
        print_one_line(str(yr+1) + f"/{n_yr}   ")
        dtas_temp = ts_a4x_rea[:, yr, :, :] - ts_pic_run[:, yr, :, :]
        dtas_nc[:, yr, :, :] = dtas_temp
        for i in np.arange(12):
            dtas_rg_nc[i, yr, :, :] = remap(lon_rg, lat_t, lon, lat_o, dtas_temp[i, :, :])
        # end for i        
    # end for yr    
else:
    print("\n\nSpecial case for FGOALS-g3\n\n")
    lat_o = lat
    lat_t = lat_rg
    lon_o = lon - 180
    lon_t = lon_rg - 180     
    for yr in np.arange(n_yr):
        print_one_line(str(yr+1) + "/150   ")
        dtas_temp = ts_a4x_rea[:, yr, :, :] - ts_pic_run[:, yr, :, :]
        dtas_nc[:, yr, :, :] = dtas_temp
        for i in np.arange(12):
            dtas_rg_nc[i, yr, :, :] = remap2(lon_rg, lat_t, lon, lat_o, dtas_temp[i, :, :])
        # end for i        
    # end for yr    
# variable units
dtas_nc.units = "K"

# variable description
dtas_nc.description = ("local monthly mean d" + var + " values")
dtas_nc.description = ("local monthly mean d" + var + " values regridded to CanESM2 grid")

# add attributes
desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values completely corresponding to the forcing experiment which is why the were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = ("This file contains the local monthly delta " + var + " values of the " + exp +
                 " and the piControl experiments for " + nl.models_pl[mod] + 
                 ". Note that the piControl value have been " +
                 "climatologically averaged over a running period of " + str(run) + " years." + desc_add)
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()
