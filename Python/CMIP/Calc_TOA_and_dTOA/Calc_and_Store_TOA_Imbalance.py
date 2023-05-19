"""
Script for calculating the TOA imbalance from the variables rlut(cs), rsut(cs), rsdt.

Be sure to change data_path and out_path.
"""


#%% imports
import os
import sys
import glob
import numpy as np
from netCDF4 import Dataset
import time as ti
import dask.array as da
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Print_In_One_Line import print_one_line


#%% establish connection to the client
c = Client("localhost:8786")
    

#%% set the required parameters
try:
    cmip = sys.argv[1]
except:
    cmip = "CMIP6"
# end try except
try:
    model = sys.argv[2]
except:
    model = "UKESM1-1-LL"
# end try except
try:
    exp = int(sys.argv[3])
except:
    exp = 1
# end try except
try:
    cs = int(sys.argv[4])
except:
    cs = 1
try:
    ensemble = sys.argv[5]
except:
    ensemble = "r1i1p1f2"
# end try except

# grid label only exists in CMIP6, if CMIP is 5 this will be replaced with an empty string
gr_lab = "_gn"
if model in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "E3SM-2-0", "EC-Earth3", 
             "EC-Earth3-AerChem", "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
    gr_lab = "_gr"
elif model in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5", "KIOST-ESM"]:
    gr_lab = "_gr1"
# end if

if cmip == "CMIP5":
    import Namelists.Namelist_CMIP5 as nl
    exp_name = ["piControl", "abrupt4xCO2", "1pctCO2", "rcp85", "historical"]
    # ensemble = "r1i1p1"
    gr_lab = ""
    direc_data = "/media/kei070/Work Disk 2"  # SETS DATA DIRECTORY
elif cmip == "CMIP6":
    import Namelists.Namelist_CMIP6 as nl
    exp_name = ["piControl", "abrupt-4xCO2", "1pctCO2", "abrupt-2xCO2", "ssp585", "amip-piForcing", "amip-4xCO2",
                "historical", "amip"]
    # ensemble = "r1i1p1f1"
    direc_data = "/media/kei070/Seagate"  # SETS DATA DIRECTORY
# end if elif

cs_n = "all-sky"
if cs:
    cs_n = "clear-sky"
# end if
    
print("\n" + model + " " + exp_name[exp] + " " + cs_n + "\n")


#%% set up the model number dictionary
import Namelists.CMIP_Dictionaries as di
mod_dict = di.mod_dict

# model number
mod = mod_dict[cmip[-1]][model]


#%% hand all-sky/clear-sky
cs_n = ""
cs_desc = ""
if cs:
    cs_n = "cs"
    cs_desc = "Clear-sky "
# end if


#%% handle the ensemble
ensemble_d = ""
if (ensemble != "r1i1p1f1") & (ensemble != "r1i1p1"):
    ensemble_d = "_" + ensemble
# end if


#%% paths and names
# --> data_path directory must contain rsdt, rsut, and rlut file for the given experiment
data_path = "" + cmip + "/Data/" + model + "/"

out_path = "" + cmip + "/Data/" + model + "/"


#%% get the files
rsdt_f = glob.glob(data_path + "/rsdt_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + "*.nc")[0]
rsut_f = glob.glob(data_path + "/rsut" + cs_n + "_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                   "*.nc")[0]
rlut_f = glob.glob(data_path + "/rlut" + cs_n + "_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                   "*.nc")[0]


#%% extract the period from the files
period = rsdt_f[-16:-9] + rsdt_f[-9:-3]


#%% load the files and values as well as lat and lon

# load nc file
rlut_nc = Dataset(rlut_f)
rsdt_nc = Dataset(rsdt_f)
rsut_nc = Dataset(rsut_f)

# get the branch year
if (cmip == "CMIP6") & (exp != 8):
    bt_chi = rlut_nc.getncattr("branch_time_in_child")
    bt_par = rlut_nc.getncattr("branch_time_in_parent")
    par_exp = rlut_nc.getncattr("parent_experiment_id")
    par_tu = rlut_nc.getncattr("parent_time_units")
    var_lab = rlut_nc.getncattr("variant_label")
elif cmip == "CMIP5":
    bt = rlut_nc.getncattr("branch_time")
# end if elif

# get the values
rlut = da.ma.masked_array(rlut_nc.variables["rlut" + cs_n], lock=True)
rsdt = da.ma.masked_array(rsdt_nc.variables["rsdt"], lock=True)
rsut = da.ma.masked_array(rsut_nc.variables["rsut" + cs_n], lock=True)

# load the first time array
tim = rlut_nc.variables["time"][:]

# get the time units
tim_units = rlut_nc.variables["time"].units

# load lats and lons
lat = rlut_nc.variables["lat"][:]
lon = rlut_nc.variables["lon"][:]


#%% calculate the TOA imbalance
# calculation: incoming shortwave - outgoing longwave - outgoing shortwave
# toa_imb is positive downward
#toa_imb = ne.evaluate("rsdt - rlut - rsut")

#del rlut, rsdt, rsut 


#%% store the result as a netcdf file

out_name = ("toa" + cs_n + "_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + gr_lab + "_" + 
            period + ".nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
if (cmip == "CMIP6") & (exp != 8):
    f.setncattr("branch_time_in_child", bt_chi)
    f.setncattr("branch_time_in_parent", bt_par)
    f.setncattr("parent_experiment_id", par_exp)
    f.setncattr("parent_time_units", par_tu)
    f.setncattr("variant_label", var_lab)
elif cmip == "CMIP5":
    f.setncattr("branch_time", bt)
# end if elif

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

# create the TOA variable
toa_nc = f.createVariable("toa" + cs_n, "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
t_step = 100  # "time step" in years
for i in np.arange(0, int(len(tim)/12), t_step):
    print_one_line(f"{i} /{int(len(tim)/12)}   ")
    toa_nc[i*12:(i+t_step)*12, :, :] = (rsdt[i*12:(i+t_step)*12, :, :] - rlut[i*12:(i+t_step)*12, :, :] - 
                                        rsut[i*12:(i+t_step)*12, :, :])
# end for i    

# add attributes
f.description = (cs_desc + "TOA imbalance file of the " + exp_name[exp] + " experiment calculated from rsdt-(rlut+rsut)")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"
time_nc.units = tim_units

# close the dataset
f.close()
