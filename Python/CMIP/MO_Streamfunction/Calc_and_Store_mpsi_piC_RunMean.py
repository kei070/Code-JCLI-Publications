"""
Calculate and store the change of the meridional overturing stream function (atmosphere) with respect to 21-year running 
mean piControl run.

Be sure to set data_path and possibly out_path.
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


#%% set the variable name
var = "mpsi"


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


#%% set the experiment
try:
    exp = sys.argv[8]
except:    
    exp = "a4x"
# end try except


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[6]
except:
    ensemble_b = "r1i1p1f2"
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[7]
except:
    ensemble_f = "r1i1p1f2"
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


#%% total number years
n_yr = 150


#%% load the namelist
exp_s = ""

if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    if exp == "a2x":
        aXx = "abrupt2xCO2"
        exp_s = "_a2x"
    elif exp == "a4x":        
        aXx = "abrupt4xCO2"
    # end if elif    
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    if exp == "a2x":
        aXx = "abrupt-2xCO2"
        exp_s = "_a2x"
    elif exp == "a4x":        
        aXx = "abrupt-4xCO2"
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


#%% set paths

# model data path
data_path = cmip + "/Data/" + nl.models[mod] + "/"

# output path
out_path = data_path + f"/mpsi_piControl_{run}YrRunMean_Files{ensemble_b_d}/"

os.makedirs(out_path, exist_ok=True)

# plot path
# pl_path = (direc + "/Uni/PhD/Tromsoe_UiT/Work/" + cmip + "/Plots/" + nl.models[mod] + "/Kernel/SfcAlbedo/")
# os.makedirs(pl_path, exist_ok=True)


#%% load the nc files
# mo_aXx_f_list = sorted(glob.glob(data_path + aXx + "_mpsi_Files" + ensemble_f_d + "/*.nc"), key=str.casefold)
mo_pic_f_list = sorted(glob.glob(data_path + "piControl_mpsi_Files" + ensemble_b_d + "/*.nc"), key=str.casefold)


#%% load first files

# load in the respective first hus file
mo_pic_nc = Dataset(mo_pic_f_list[0])
# mo_aXx_nc = Dataset(mo_aXx_f_list[0])

# load the first hus file data
mo_pic = da.ma.masked_array(mo_pic_nc.variables["mpsi"], lock=True)
# mo_aXx = da.ma.masked_array(mo_aXx_nc.variables["mpsi"], lock=True)


#%% get the grid coordinates (vertical and horizontal)

# get the levels
plev = mo_pic_nc.variables["plev"][:]

# get lat, lon, and levels for the model data
lat = mo_pic_nc.variables["lat"][:]


#%% load all files and concatenate them
if len(mo_pic_f_list) > 1:
    nc_l = []
    for i in np.arange(0, len(mo_pic_f_list)):
        nc_l.append(da.ma.masked_array(Dataset(mo_pic_f_list[i]).variables[var], lock=True))
    # end for i
    
    # concatenate the loaded files
    vals_da = da.concatenate(nc_l, axis=0)
else:
    print("\nOnly one " + var + " file...\n")
    vals_da = da.ma.masked_array(Dataset(mo_pic_f_list[0]).variables[var], lock=True)
# end if else


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
if ((da.shape(vals_da)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print("\npiControl run after branch year is at least 10 years longer than abrupt4xCO2. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print("\npiControl run after branch year is less than 10 years longer than abrupt4xCO2. Using original b_end_ind: " + 
          str(b_end_ind) + "\n")
# end if

vals_da = vals_da[b_sta_ind:b_end_ind, :, :]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    f_add = sorted(glob.glob(data_path + f"/AddFiles_mpsi_piC{ensemble_b_d}/*AddBe.nc"), 
                   key=str.casefold)
    vals_da = da.concatenate([da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True), vals_da], axis=0)
    print("\nAdded AddBe file. New piControl shape:" + str(da.shape(vals_da)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_add = sorted(glob.glob(data_path + f"/AddFiles_mpsi_piC{ensemble_b_d}/*AddAf.nc"), 
                   key=str.casefold)
    vals_da = da.concatenate([vals_da, da.ma.masked_array(Dataset(f_add[-1]).variables[var], lock=True)], axis=0)
    print("\nAdded AddAf file. New piControl shape:" + str(da.shape(vals_da)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except


#%% save as nc file
print("\nOpening the nc file...\n")
f = Dataset(out_path + f"{var}_{nl.models_n[mod]}_piControl_{run}Yr_RunMean_0001-015012.nc", 
            "w", format="NETCDF4")

# add the ensemble as an attribute
f.setncattr("variant_label", ensemble_b)

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("mon", 12)
f.createDimension("year", n_yr)
f.createDimension("lat", len(lat))
f.createDimension("plev", len(plev))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
plev_nc = f.createVariable("plev", "f8", "plev")
var_nc = f.createVariable(var, "f4", ("mon", "year", "plev", "lat"))    

# calculate the running means of the piControl    
vals_run = np.zeros((12, n_yr, len(plev), len(lat)))
print("\nCalculating running means...\n")
for mon in np.arange(12):
    print(mon)
    vals_run[mon, :, :, :] = run_mean_da(vals_da[mon::12, :, :], running=run)
# end for mon

desc_add = ""
if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values completely corresponding to the forcing experiment which is why the were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

# fill in the values to the nc variables
lat_nc[:] = lat
plev_nc[:] = plev
var_nc[:] = vals_run

# set the variable units
var_nc.units = "kg / s"
lat_nc.units = "degrees_north"
plev_nc.units = "Pa"

# enter variable descriptions
var_nc.description = var + " " + str(run) + "-year running mean"

# enter a file description
f.description = (str(run) + "-year running mean of " + var + " of the piControl run of the " + nl.models[mod] + 
                 " model." + desc_add)

# enter the file creation date
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()