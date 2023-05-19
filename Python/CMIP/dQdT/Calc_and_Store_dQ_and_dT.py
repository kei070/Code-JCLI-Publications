"""
Script for calculating monthly dQ and dT.

Be sure to change data_path and out_path.
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
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
import dask.array as da
import time as ti
import timeit
import Ngl as ngl
import geocat.ncomp as geoc
import smtplib, ssl
from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_calcsatspechum import calcsatspechum, da_calcsatspechum


#%% establish connection to the client
c = Client("localhost:8786")


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


#%% set the experiment (probably only applicable for CMIP6)
# --> for abrupt-4xCO2 set exp = "a4x"
# --> for abrupt-2xCO2 set exp = "a2x"
exp = "a4x"


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[3]
except:
    ensemble_b = "r1i1p1f2"
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[4]
except:
    ensemble_f = "r1i1p1f2"
# end try except


#%% handle the ensemble in the file path
ensemble_b_d = ""
ensemble_f_d = ""
if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # base experiment ensemble (usually piControl)
    ensemble_b_d = "_" + ensemble_b
# end if
if (ensemble_f != "r1i1p1f1") & (ensemble_f != "r1i1p1"):  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f_d = "_" + ensemble_f
# end if


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    aXx = "abrupt4xCO2"
    exp = ""
    exp_n = ""
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    if exp == "a4x":
        aXx = "abrupt-4xCO2"
        exp_n = ""
    elif exp == "a2x":
        aXx = "abrupt-2xCO2"
        exp_n = "_a2x"
    # end if elif        
# end if elif


#%% print the model
print(nl.models_pl[mod])


#%% number of years
n_yr = 150


#%% branch time indices --> should no longer be necessary because of the running mean piControl
"""
# set the branch year for CESM2 to 0
nl.b_times["CESM2"] = 0
nl.b_times["CAMS-CSM1-0"] = 0

b_sta_ind = nl.b_times[nl.models[mod]]  # int(nl.b_times[nl.models[mod]] * 12)
b_end_ind = n_yr + b_sta_ind  # branch end index
"""


#%% set paths

# model data path
data_path = (cmip + "/Data/" + nl.models[mod] + "/")

# output path
out_path = (cmip + "/Data/" + nl.models[mod] + "/" + "dQ_and_dT_Files" + ensemble_f_d + exp_n + "/")
os.makedirs(out_path, exist_ok=True)


#%% get the lists of hus ta files
q_aXx_f_list = sorted(glob.glob(data_path + aXx + "_hus_Files" + ensemble_f_d + "/*.nc"), key=str.casefold)
q_pic_f_list = sorted(glob.glob(data_path + "hus_piControl_21YrRunMean_Files" + ensemble_b_d + "/*.nc"), 
                      key=str.casefold)
ta_aXx_f_list = sorted(glob.glob(data_path + aXx + "_ta_Files" + ensemble_f_d + "/*.nc"), key=str.casefold)
ta_pic_f_list = sorted(glob.glob(data_path + "ta_piControl_21YrRunMean_Files" + ensemble_b_d + "/*.nc"), 
                       key=str.casefold)


#%% load first files

# load in the respective first hus file
q_pic_nc = Dataset(q_pic_f_list[0])
q_aXx_nc = Dataset(q_aXx_f_list[0])

# load the first hus file data
q_pic = da.ma.masked_array(q_pic_nc.variables["hus"], lock=True)
q_aXx = da.ma.masked_array(q_aXx_nc.variables["hus"], lock=True)

# load in the respective first ta file
ta_aXx_nc = Dataset(ta_aXx_f_list[0])
ta_pic_nc = Dataset(ta_pic_f_list[0])

# load the first ta file data
ta_aXx = da.ma.masked_array(ta_aXx_nc.variables["ta"], lock=True)
ta_pic = da.ma.masked_array(ta_pic_nc.variables["ta"], lock=True)


#%% get the grid coordinates (vertical and horizontal)

# get the levels
levs = ta_aXx_nc.variables["plev"][:]

# get lat, lon, and levels for the model data
lat = ta_aXx_nc.variables["lat"][:]
lon = ta_aXx_nc.variables["lon"][:]


#%%  loop over the files and concatenate the values via dask
if (len(ta_aXx_f_list) > 1) & (len(q_aXx_f_list) > 1):
    for i in np.arange(1, len(ta_aXx_f_list)):
        q_aXx = da.concatenate([q_aXx, da.ma.masked_array(Dataset(q_aXx_f_list[i]).variables["hus"], lock=True)])
        ta_aXx = da.concatenate([ta_aXx, da.ma.masked_array(Dataset(ta_aXx_f_list[i]).variables["ta"], lock=True)])
    # end for i
else:
    print("\nOnly one ta and q " + exp + " file...\n")
# end if else

q_pic_l = []
ta_pic_l = []
if (len(ta_pic_f_list) > 1) & (len(q_pic_f_list) > 1):
    for i in np.arange(0, len(ta_pic_f_list)):
        q_pic_l.append(da.ma.masked_array(Dataset(q_pic_f_list[i]).variables["hus"], lock=True))
        ta_pic_l.append(da.ma.masked_array(Dataset(ta_pic_f_list[i]).variables["ta"], lock=True))
    # end for i
else:
    print("\nOnly one ta and q " + exp + " file...\n")
# end if else

# print the dataset history of the last dataset loaded
print("\n\nQ piControl Running Mean data set " + Dataset(q_pic_f_list[i]).history)
print("\n\nT piControl Running Mean data set " + Dataset(ta_pic_f_list[i]).history + "\n\n")

q_pic = da.stack(q_pic_l, axis=2)
ta_pic = da.stack(ta_pic_l, axis=2)

# reduce the arrays to the 150 years of the experiment and the corresponding control run
q_aXx = q_aXx[:(n_yr*12), :, :, :]
ta_aXx = ta_aXx[:(n_yr*12), :, :, :]

# q_pic = q_pic[b_sta_ind:b_end_ind, :, :, :]
# ta_pic = ta_pic[b_sta_ind:b_end_ind, :, :, :]

# test plot of the lowest level
# pl.imshow(ta_aXx[100, 0, :, :], origin="lower"), pl.colorbar()

# rechunk the dask arrays
# q_aXx = da.rechunk(q_aXx, chunks=(30, len(levs), len(lat), len(lon)))
# q_pic = da.rechunk(q_pic, chunks=(30, len(levs), len(lat), len(lon)))
# ta_aXx = da.rechunk(ta_aXx, chunks=(30, len(levs), len(lat), len(lon)))
# ta_pic = da.rechunk(ta_pic, chunks=(30, len(levs), len(lat), len(lon)))


#%% set up the 4d pressure field
p4d = np.zeros((12, len(levs), len(lat), len(lon)))
p4d[:, :, :, :] = levs[None, :, None, None] / 100


#%% calculate dqdt
print("\nCalculating dQ and dT...\n")

sta_yr = 0

yr = sta_yr + 1  # year-counter for the file name
start = timeit.default_timer()
ti_old = copy.deepcopy(start)
for yri, i in enumerate(np.arange(sta_yr, int(n_yr*12), 12)):
    
    yri = yri + sta_yr

    # set output name
    out_name =  "dQ_and_dT_" + nl.models_n[mod] + exp_n + "_" + f'{yr:04}' + "01" + "-" + f'{yr:04}' + "12" + ".nc"
    
    # test if the file already exists and jump the loop if so
    if os.path.isfile(out_path + out_name):
        print("\nFile for year " + str(yri) + " already produced. Continuing...")
        # update the year-counter
        yr += 1        
        continue
    # end if
    
    print("\nDask calculations " + str(yr) + " of " + str(n_yr) + "...\n")           
    
    # change the masks to NaN directly
    # furthermore, for some reason the array function is necessary to make the array writable (EC-Earth3-Veg)...
    ta1 = np.array(np.ma.filled(ta_pic[:, yri, :, :, :].compute(), np.nan))
    ta2 = np.array(np.ma.filled(ta_aXx[i:(i+12), :, :, :].compute(), np.nan))
    q1 = np.array(np.ma.filled(q_pic[:, yri, :, :, :].compute(), np.nan))
    q2 = np.array(np.ma.filled(q_aXx[i:(i+12), :, :, :].compute(), np.nan))
    
    qs1 = da_calcsatspechum(ta1, p4d)
    qs2 = da_calcsatspechum(ta2, p4d)

       
    print("\nGenerating nc file " + str(yr) + " of " + str(n_yr) + "...\n")
    f = Dataset(out_path + out_name, 
                "w", format="NETCDF4")

    # add the ensemble as an attribute
    f.setncattr("variant_label_base", ensemble_b)
    f.setncattr("variant_label_forced", ensemble_f)
    
    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("time", 12)
    f.createDimension("plev", len(levs))
    f.createDimension("lat", len(lat))
    f.createDimension("lon", len(lon))
    
    # create the variables
    lat_nc = f.createVariable("lat", "f8", "lat")
    lon_nc = f.createVariable("lon", "f8", "lon")
    lev_nc = f.createVariable("plev", "f4", "plev")
    dq_nc = f.createVariable("dQ", "f4", ("time", "plev", "lat", "lon"))
    dt_nc = f.createVariable("dT", "f4", ("time", "plev", "lat", "lon"))
    drh_nc = f.createVariable("dRH", "f4", ("time", "plev", "lat", "lon"))
    
    # pass the data into the variables
    lat_nc[:] = lat
    lon_nc[:] = lon
    lev_nc[:] = levs
    
    dq_nc[:] = q2 - q1  # .compute()
    dt_nc[:] = ta2 - ta1  # .compute()
    
    drh_nc[:] = q2/qs2 - q1/qs1  # .compute()
    
    # set the variable units
    dt_nc.units = "K"
    lat_nc.units = "degrees_north"
    lon_nc.units = "degrees_east"
    lev_nc.units = "Pa"
    
    # enter variable descriptions
    dq_nc.description = "dQ"
    dt_nc.description = "dT"
    drh_nc.description = "dRH"
    
    # enter a file description
    f.description = ("This file contains the monthly Q and T change as well as RH. Note that the " +
                     "piControl values of ta and hus have previously been averaged over a 21-year running period." )
    
    # enter the file creation date
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    # close the dataset
    f.close()
    
    print("\nFile generated.\n")
    
    ti_temp = timeit.default_timer()
    del_t = ti_temp - start
    frac = yr / 150
    iter_time = (ti_temp-ti_old) / 60
    
    # remaining = np.round((del_t / frac - del_t) / 60, decimals=1)
    remaining = np.round(iter_time * (150 - yr), decimals=2)
    
    print_one_line(str(np.round(yr / 150 * 100, decimals=1)) + " %  " + str(remaining) + " min remain\n\n")
    
    # update the year-counter
    yr += 1
    # raise Exception("First file generated.")
    
    # set ti_old to the value of ti_temp
    ti_old = copy.deepcopy(ti_temp)
    
# end for i
    
    
    
    
