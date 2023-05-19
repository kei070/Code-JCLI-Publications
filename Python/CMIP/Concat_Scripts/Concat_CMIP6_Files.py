"""
Concatenate CMIP6 netCDF files. This is so far only used for 2D (+time) files such as tas, ts, rsdt, rsut, ... to 
expedite late calculations.

Be sure to change data_path and out_path.
"""

#%% imports
import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset
import time as ti


#%% set script and data paths
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

    

#%% set the required parameters
try:
    exp = int(sys.argv[1])
except:
    exp = 1
try:
    var_name = sys.argv[2]
except:
    var_name = "hfls"
try:    
    ensemble = sys.argv[3]
except:    
    ensemble = "r1i1p1f1"
try:
    model = sys.argv[4]
except:    
    model = "E3SM-2-0" 
try:
    c_tabl = sys.argv[5]
except:        
    c_tabl = "Amon"
# end try except 

exp_name = ["piControl", "abrupt-4xCO2", "1pctCO2", "abrupt-2xCO2", "ssp585", "amip-piForcing", "amip-4xCO2", 
            "amip-hist", "abrupt-0p5xCO2", "historical", "amip"]

print(model)
print("\n" + var_name + "\n")
print(exp_name[exp] + "\n")


#%% handle the grid 
gr_lab = "gn"

if model in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "E3SM-2-0", "EC-Earth3", 
             "EC-Earth3-AerChem", "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
    gr_lab = "gr"
elif model in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5", "KIOST-ESM"]:
    gr_lab = "gr1"
# end if


#%% handle the ensemble
ensemble_d = ""

if ensemble != "r1i1p1f1":
    ensemble_d = "_" + ensemble
# end if


#%% paths and names
# --> path to given model data; must contain a folder with the given variable files
# --> folder name: [experiment name]_[variable name_Files]_[ensemble; not if ensemble = r1i1p1f1] 
data_path = "" + exp_name[exp] + "_" + var_name + "_Files" + ensemble_d + "/"


#%% get the files
f_list = np.sort(glob.glob(data_path + "/*.nc"))


#%% check if there are any files to concatenate and stop the execution if not
if len(f_list) == 0:
    raise Exception("Stopping concatenation because there are no " + var_name + " " + exp_name[exp] + 
                    " files to concatenate for " + model + ".")
# end if   


#%% extract the period from the files
#   !!make sure that you have all files in between!!
period = f_list[0][-16:-9] + f_list[-1][-9:-3]


#%% load the first file and values as well as lat and lon

# load nc file
nc = Dataset(f_list[0])
# print(nc.variables[var_name])

# get the branch year
if exp != 10:
    bt_chi = nc.getncattr("branch_time_in_child")
    bt_par = nc.getncattr("branch_time_in_parent")
    par_exp = nc.getncattr("parent_experiment_id")
    par_tu = nc.getncattr("parent_time_units")
# end if    
var_lab = nc.getncattr("variant_label")

# get the values
var = nc.variables[var_name][:]

# load the first time array
tim = nc.variables["time"][:]

# get the time units
tim_units = nc.variables["time"].units

# load lats and lons
try:
    lat = nc.variables["lat"][:]
    lon = nc.variables["lon"][:]
except:
    lat = nc.variables["latitude"][:]
    lon = nc.variables["longitude"][:]
# end try


#%% load all files and concatenate them
for i in np.arange(1, len(f_list), 1):
    
    var_add = Dataset(f_list[i]).variables[var_name][:]
    tim_add = Dataset(f_list[i]).variables["time"][:]
    
    var = np.concatenate((var, var_add), axis=0)
    tim = np.concatenate((tim, tim_add), axis=0)
    
# end for i
    
    
#%% store the concatenated data as nc file

# generate the output name
out_path = ""
out_name = (var_name + "_" + c_tabl + "_" + model + "_" + exp_name[exp] + "_" + ensemble + "_" + gr_lab + "_" + period + 
            ".nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
if exp != 10:
    f.setncattr("branch_time_in_child", bt_chi)
    f.setncattr("branch_time_in_parent", bt_par)
    f.setncattr("parent_experiment_id", par_exp)
    f.setncattr("parent_time_units", par_tu)
# end if
f.setncattr("variant_label", var_lab)

# set up the wind speed createVariable list, Note that it's the wrong way
# round due to the appending and will later be flipped and then converted
# to a tuple
create_var = ["lon", "lat", "time"]


# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("lon", len(lon))
f.createDimension("lat", len(lat))

# build the variables
longitude = f.createVariable("lon", "f8", "lon")
latitude = f.createVariable("lat", "f8", "lat")

f.createDimension("time", len(tim))
time_nc = f.createVariable("time", "i4", "time")
time_nc[:] = tim

# flip the list and turn it into a tuple
create_var = tuple(create_var[::-1])

# create the variable
var_nc = f.createVariable(var_name, "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
var_nc[:] = var

# add attributes
f.description = (var_name + " file of the " + exp_name[exp] + " experiment concatenated from the " + str(len(f_list)) + 
                 " files:\n" + f_list[0] + "\n.\n.\n.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"
time_nc.units = tim_units

# close the dataset
f.close()