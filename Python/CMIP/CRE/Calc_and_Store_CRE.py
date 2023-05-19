"""
Script for calculating the CRE at TOA from the variables rlut(cs), rsut(cs), rsdt.

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


#%% establish connection to the client
c = Client("localhost:8786")


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Print_In_One_Line import print_one_line
    

#%% set the required parameters
try:
    cmip = sys.argv[1]
except:
    cmip = "CMIP6"
# end try except
try:
    model = sys.argv[2]
except:
    model = "UKESM1-0-LL"
# end try except
try:
    exp = int(sys.argv[3])
except:
    exp = 1
try:
    ensemble = sys.argv[4]
except:
    ensemble = "r1i1p1f2"    
# end try exceptz

# grid label only exists in CMIP6, if CMIP is 5 this will be replaced with an empty string
gr_lab = "_gn"
if model in ["CIESM", "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", "EC-Earth3-AerChem", 
             "EC-Earth3-CC", "EC-Earth3-Veg", "FGOALS-f3-L", "IPSL-CM6A-LR"]:
    gr_lab = "_gr"
elif model in ["GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5", "KIOST-ESM"]:
    gr_lab = "_gr1"
# end if

if cmip == "CMIP5":
    import Namelists.Namelist_CMIP5 as nl
    exp_name = ["piControl", "abrupt4xCO2", "1pctCO2"]
    # ensemble = "r1i1p1"
    gr_lab = ""
    direc_data = "/media/kei070/Work Disk 2"  # SETS DATA DIRECTORY
elif cmip == "CMIP6":
    import Namelists.Namelist_CMIP6 as nl
    exp_name = ["piControl", "abrupt-4xCO2", "1pctCO2", "abrupt-2xCO2", "ssp585", "amip-piForcing", "amip-4xCO2"]
    # ensemble = "r1i1p1f1"
    direc_data = "/media/kei070/Seagate"  # SETS DATA DIRECTORY
# end if elif


#%% set up the model number dictionary
import Namelists.CMIP_Dictionaries as di
mod_dict = di.mod_dict

# model number
mod = mod_dict[cmip[-1]][model]


#%% handle the ensemble
ensemble_d = ""
if (ensemble != "r1i1p1f1") & (ensemble != "r1i1p1"):
    ensemble_d = "_" + ensemble
# end if


#%% paths and names
data_path = model + "/"

out_path = model + "/"


#%% get the files
rlut_f = glob.glob(data_path + "/rlut_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + "*")[0]

rlut_nc = Dataset(glob.glob(data_path + "/rlut_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                            "*")[0])
rsut_nc = Dataset(glob.glob(data_path + "/rsut_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                            "*")[0])
rlutcs_nc = Dataset(glob.glob(data_path + "rlutcs_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                              "*")[0])
rsutcs_nc = Dataset(glob.glob(data_path + "rsutcs_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + 
                              "*")[0])


#%% extract the period from the files
period = rlut_f[-16:-9] + rlut_f[-9:-3]


#%% load the files and values as well as lat and lon

# get the branch year
if cmip == "CMIP6":
    bt_chi = rlut_nc.getncattr("branch_time_in_child")
    bt_par = rlut_nc.getncattr("branch_time_in_parent")
    par_exp = rlut_nc.getncattr("parent_experiment_id")
    par_tu = rlut_nc.getncattr("parent_time_units")
    #var_lab = rlut_nc.getncattr("variant_label")
    var_lab = ensemble
elif cmip == "CMIP5":
    bt = rlut_nc.getncattr("branch_time")
# end if elif

# get the values
rlut = da.ma.masked_array(rlut_nc.variables["rlut"], lock=True)
rsut = da.ma.masked_array(rsut_nc.variables["rsut"], lock=True)
rlutcs = da.ma.masked_array(rlutcs_nc.variables["rlutcs"], lock=True)
rsutcs = da.ma.masked_array(rsutcs_nc.variables["rsutcs"], lock=True)

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

out_name = ("cre_Amon_" + nl.models_n[mod] + "_" + exp_name[exp] + "_" + ensemble + gr_lab + "_" + period + ".nc")

# generate the file
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
if cmip == "CMIP6":
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

# create the wind speed variable
cre_lw_nc = f.createVariable("cre_lw", "f4", create_var)
cre_sw_nc = f.createVariable("cre_sw", "f4", create_var)

# pass the remaining data into the variables
longitude[:] = lon 
latitude[:] = lat
t_step = 100  # "time step" in years
for i in np.arange(0, int(len(tim)/12), t_step):
    print_one_line(f"{i} /{int(len(tim)/12)}   ")
    cre_lw_nc[i*12:(i+t_step)*12, :, :] = (rlutcs[i*12:(i+t_step)*12, :, :] - rlut[i*12:(i+t_step)*12, :, :])
    cre_sw_nc[i*12:(i+t_step)*12, :, :] = (rsutcs[i*12:(i+t_step)*12, :, :] - rsut[i*12:(i+t_step)*12, :, :])
# end for i    

# add attributes
f.description = ("Cloud radiative effect (CRE) file of the " + exp_name[exp] + " experiment containing " + 
                 "long-wave CRE and short-wave CRE calculated from rlut-rlutcs and rsut-rsutcs, respectively.")
f.history = "Created " + ti.strftime("%d/%m/%y")

longitude.units = "degrees east"
latitude.units = "degrees north"
time_nc.units = tim_units

# close the dataset
f.close()
