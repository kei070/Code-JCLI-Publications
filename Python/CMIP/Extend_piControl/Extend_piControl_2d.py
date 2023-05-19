"""
Extend the 2d piControl files (if necessary).

Be sure to change data_path and out_path.
"""


"""
Extend piControl runs for 21-year running mean calculations:
    For some models the piControl runs start in the same year as the branching experiment (e.g. EC-Earth3-Veg) and for
    some models they even end in the same year (e.g. GISS-E2.2-G) so that the 21-year running means of the piControl run
    actually cannot be calculated for the corresponding period. Similar to Grose et al. (2018, Geophys. Res. L.) but 
    maybe a little different, extend the piControl run for the model for which it is necessary by fitting linear trends
    to the first and last 21 year, respectively, (per month!) and add this as a new file to the piControl run data.
    
    The file to be added before the piControl run has the file name suffix AddBe ("add before") and the file added after
    it has the suffix AddAf ("add after").
    
    NOTE: We assume here that one of to cases occurs here:
        1. The piControl run starts in the same year as the forcing experiment but runs at least 10 years longer.
        2. The piControl run starts in the same year as the forcing experiment and runs exactly as long.
    This means in effect the we do not handle cases here where the piControl run starts >=10 years before the forcing
    experiment but ends in the same year.    
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
from scipy import interpolate
import progressbar as pg
import dask.array as da
import time as ti
import timeit
import smtplib, ssl
from dask.distributed import Client


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
from Functions.Func_Print_In_One_Line import print_one_line
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% establish connection to the client
c = Client("localhost:8786")


#%% variable, either "ta", or "hus"
try:
    var = sys.argv[3]
except:
    var = "toa"
# end try except

# var_unit = "Pa"
# var_unit = "%"
# var_unit = "m"
# var_unit = "K"
# var_unit = "kg/kg"
var_unit = "W m-2"

"""
add_toa = ""
if (var == "rsut") | (var == "rsdt") | (var == "rlut"):
    add_toa = "/piControl_toa_Files/"
# end if
"""

#%% set the runnning mean
run = 21


#%% length of the experiment in years
n_yr = 150


#%% set the experiment - either "4x" or "2x"
"""
try:
    exp = sys.argv[3]
except:
    exp = "4x"
# end try except
"""


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 5
# end try except

if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    # a4x = "abrupt" + exp + "CO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    # a4x = "abrupt-" + exp + "CO2"
# end if elif


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 5
# end try except
    
print(nl.models_pl[mod])

# set the branch year for CESM2 to 100
# nl.b_times["CAMS-CSM1-0"] = 25
# nl.b_times["CESM2"] = 100


#%% set the ensemble
try:    
    ensemble = sys.argv[4]
except:    
    ensemble = "r1i1p1"
# end try ensemble    


#%% handle the ensemble
ensemble_d = ""

if (ensemble != "r1i1p1f1") & (ensemble != "r1i1p1"):
    ensemble_d = "_" + ensemble
# end if


#%% get the branching index
signs = []
add_be = True
if nl.b_times[nl.models[mod]] >= 10:
    add_be = False
    # raise Exception("Stopping exection since the piControl run starts >=10 years before the forcing experiment.")
    print("\nNo Add_Be file necessary since the the piControl run starts >=10 years before the forcing experiment.\n")
else:     
    print("\nAdd_Be file necessary since the the piControl run starts <10 years before the forcing experiment.\n")
    signs.append(1)
# end if else

b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)
b_end_ind = int(n_yr * 12) + b_sta_ind


#%% set paths

# model data path
# --> must contain the piControl file of the given variable for the given model
data_path = ""

# output path
out_path = "/AddFiles_2d_piC/"
os.makedirs(out_path, exist_ok=True)


#%% get the lists files
f_list = sorted(glob.glob(data_path + "/" + var + f"_*_piControl_{ensemble}_*.nc"), key=str.casefold)


#%% load first file

# load in the respective first hus file
nc = Dataset(f_list[0])


#%% grid variables

# get lat, lon, and levels for the model data
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]


#%% load all files and concatenate them
if len(f_list) > 1:
    nc_l = []
    for i in np.arange(0, len(f_list)):
        nc_l.append(da.ma.masked_array(Dataset(f_list[i]).variables[var], lock=True))
    # end for i
    
    # concatenate the loaded files
    vals_da = da.concatenate(nc_l, axis=0)
else:
    print("\nOnly one " + var + " file...\n")
    vals_da = da.ma.masked_array(Dataset(f_list[0]).variables[var], lock=True)
# end if else


#%% get the variant label/ensemble
if cmip == "CMIP6":
    try:
        var_lab = Dataset(f_list[0]).getncattr("variant_label")
    except:
        var_lab = "r1i1p1f1"
    # end try except        
# end if    


#%% check if the control run ends in the same year as the forcing run (duration = 150 years = 1800 months)
if (da.shape(vals_da)[0] - b_sta_ind) < (n_yr + int((run-1)/2))*12:
    add_af = True
    print("\npiControl duration after branching year < 160 years => AddAf file necessary...")      
    signs.append(-1)
else:
    add_af = False
    print("\npiControl duration >= 160 years => no AddAf file necessary...")
    # b_end_ind =  int(n_yr * 12 + b_sta_ind + ((run - 1) / 2) * 12)  # branch end index
# end if else


#%% check if any of the Add_Be or Add_Af files are necessary and if not stop the execution
if (not add_be) & (not add_af):
    raise Exception("Stopping exection since neither Add_Be nor Add_Af are necessary.")
# end if    


#%% reduce the value array to the start and end indices
vals_da = vals_da[b_sta_ind:b_end_ind, :, :]


#%% extract the first and last 21 years from the data
startt = timeit.default_timer()
for sign in signs:
    
    # set up a target array for the first/last 10 years
    t_arr = np.zeros((10*12, len(lat), len(lon)))
    
    # set up the origin coords
    if sign > 0:  # iteration for first 21 years
        
        print("\nIteration for first 21 years...\n")
        
        # first 21 years if sign is +1 one and last 21 years if sign is -1
        vals_21 = np.array(np.ma.filled(vals_da[:sign*21*12].compute(), np.nan))
        vals_21[vals_21 == -1.00000006e+27] = np.nan  # special treatment for GISS
       
        orig_coords = np.arange(sign*21)
        targ_coords = np.arange(-sign*10, 0)
        
        add = "_AddBe"
        desc = ("This file contains the linearly in time extrapolated " + var + " values for 'before' the " + 
                nl.models_pl[mod] + " piControl run. The values were calculated more or less in accordance with\n" + 
                "Grose et al. (2018, Geophys. Rev. Let.) to be able to generate 21-year running mean control runs\n" + 
                "for the corrsponding 150-year monthly values forced experiments (e.g. abrupt4xCO2). Some models\n" + 
                "(notably this one) have control runs starting in the same year as the forced experiment so that\n" + 
                "this extrapolation is necessary to get a (more or less) corresponding piControl run for the forced\n" + 
                "experiment.")
        
    elif sign < 0:  # iteration for last 21 years
    
        print("\nIteration for last 21 years...\n")    
    
        # first 21 years if sign is +1 one and last 21 years if sign is -1
        vals_21 = np.array(np.ma.filled(vals_da[sign*21*12:].compute(), np.nan))
        vals_21[vals_21 == -1.00000006e+27] = np.nan  # special treatment for GISS
        
        orig_coords = np.arange(sign*21, 0)
        targ_coords = np.arange(-sign*10)
        
        add = "_AddAf"
        desc = ("This file contains the linearly in time extrapolated " + var + " values for 'after' the " + 
                nl.models_pl[mod] + " piControl run. The values were calculated more or less in accordance with\n" + 
                "Grose et al. (2018, Geophys. Rev. Let.) to be able to generate 21-year running mean control runs\n" + 
                "for the corrsponding 150-year monthly values forced experiments (e.g. abrupt4xCO2). Some models\n" + 
                "(notably this one) have control runs ending in the same year as the forced experiment so that\n" + 
                "this extrapolation is necessary to get a (more or less) corresponding piControl run for the forced\n" + 
                "experiment.")        
    # end if elif
    
    # iterate over the months
    for mon in np.arange(12):
        print("\n " + var + " month " +  str(mon))
        for la in np.arange(len(lat)):
            for lo in np.arange(len(lon)):
                """
                interp_f = interpolate.interp1d(orig_coords,
                            vals_21[mon::12, la, lo],
                            kind="linear", 
                            fill_value="extrapolate")
                t_arr[mon::12, la, lo] = interp_f(targ_coords)
                """
                sl, yi = lr(orig_coords, vals_21[mon::12, la, lo])[:2]
                t_arr[mon::12, la, lo] = sl * targ_coords + yi
            # end for lo
        # end for la
    # end for mon
    
    print("\nOpening the nc file...\n")
    # f = Dataset(out_path + var + "_Omon_" + nl.models_n[mod] + "_piControl" + ensemble_d + "_n80" + add + ".nc", "w", 
    #             format="NETCDF4")
    f = Dataset(out_path + var + "_Amon_" + nl.models_n[mod] + "_piControl" + ensemble_d + add + ".nc", "w", 
                format="NETCDF4")
    
    # set the variant label
    if cmip == "CMIP6":
        f.setncattr("variant_label", var_lab)
    # end if
    
    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("mon", 12*10)
    f.createDimension("lat", len(lat))
    f.createDimension("lon", len(lon))
    
    # create the variables
    lat_nc = f.createVariable("lat", "f8", "lat")
    lon_nc = f.createVariable("lon", "f8", "lon")
    var_nc = f.createVariable(var, "f4", ("mon", "lat", "lon"))    
    
    # fill in the values to the nc variables
    lat_nc[:] = lat
    lon_nc[:] = lon
    var_nc[:] = t_arr
    
    # set the variable units
    var_nc.units = var_unit
    lat_nc.units = "degrees_north"
    lon_nc.units = "degrees_east"
    
    # enter variable descriptions
    var_nc.description = "extention of " + var 
    
    # enter a file description
    f.description = desc
    
    # enter the file creation date
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    # close the dataset
    f.close()    
    """
    pl.imshow(t_arr[0, :, :], origin="lower"), pl.colorbar()
    pl.title(var)
    pl.show()
    pl.close()
    
    pl_la = int(len(lat)/2)
    pl_lo = int(len(lon)/2)
    pl.plot(np.arange(21, 31), t_arr[::12, pl_la, pl_lo]), pl.plot(np.arange(0, 21), vals_21[::12, pl_la, pl_lo])
    pl.title(var)
    pl.show()
    pl.close()
    """
# end if

print("\nTime required: " + str((timeit.default_timer()-startt) / 60) + " min")

    
    
    


