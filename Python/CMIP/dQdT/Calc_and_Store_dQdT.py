"""
Script for calculating monthly dQ/dT for a given model. This is needed for the calculation of monthly water vapour 
and temperature radiative-kernel response. This script uses the piControl 21-year running means of ta and hus
calculated via the script Calc_ta_hus_RunningMean.py.

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


#%% number of years
n_yr = 150


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
    mod = 18
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
# --> for 1pctCO2 set      exp = "1pct"
exp = "a4x"
# exp = "1pct"


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[3]
except:
    ensemble_b = "r1i1p1f1"
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[4]
except:
    ensemble_f = "r3i1p1f1"
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
    if exp == "a4x":
        aXx = "abrupt4xCO2"
        exp_n = ""
    elif exp == "a2x":
        aXx = "abrupt2xCO2"
        exp_n = "_a2x"
    elif exp == "1pct":
        aXx = "1pctCO2"
        exp_n = "_1pct"
    # end if elif
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    if exp == "a4x":
        aXx = "abrupt-4xCO2"
        exp_n = ""
    elif exp == "a2x":
        aXx = "abrupt-2xCO2"
        exp_n = "_a2x"
    elif exp == "1pct":
        aXx = "1pctCO2"
        exp_n = "_1pct"
    # end if elif        
# end if elif


#%% print the model
print(nl.models_pl[mod])


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
# --> directory must contain the running forcing experiment ta and hus files in folders
# abrupt-4xCO2_ta_Files and abrupt-4xCO2_hus_Files (no hyphen in case of CMIP5) as well as the 21-year running mean 
# piControl files in ta_piControl_21YrRunMean_Files and hus_piControl_21YrRunMean_Files
# model data path
data_path = (cmip + "/Data/" + nl.models[mod] + "/")

# output path
out_path = (cmip + "/Data/" + nl.models[mod] + "/" + "dQdT_Files" + ensemble_f_d + exp_n + "/")
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
print("\nCalculating " + dq_str + "/dT...\n")

sta_yr = 0

yr = sta_yr + 1  # year-counter for the file name
start = timeit.default_timer()
ti_old = copy.deepcopy(start)
for yri, i in enumerate(np.arange(sta_yr, int(n_yr*12), 12)):
    
    yri = yri + sta_yr

    # set output name
    out_name =  dq_str + "dT_" + nl.models_n[mod] + exp_n + "_" + f'{yr:04}' + "01" + "-" + f'{yr:04}' + "12" + ".nc"
    
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
    
    """
    if yri == 149:
        pl.imshow(ta2[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta abrupt-4xC02 " + nl.models_pl[mod])
        pl.show()
        pl.close()
        
        pl.imshow(ta1[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta piControl Running " + nl.models_pl[mod])
        pl.show()
        pl.close()
        
        pl.imshow(ta2[3, 0, :, :] - ta1[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta Difference " + nl.models_pl[mod])
        pl.show()
        pl.close()
    # end if
    """

    #q1[q1 == 1e+20] = np.nan  # should not be necessary!
    #q2[q2 == 9.96921e+36] = np.nan  # probably only for CESM2
    #q2[q2 == 1e+20] = np.nan
    #q1[q1 == -1.00000006e+27] = np.nan  # special treatment for the GISS models
    #q2[q2 == -1.00000006e+27] = np.nan  # special treatment for the GISS modelsnan
    #ta1[ta1 == 1e+20] = np.nan  # should not be necessary!
    #ta2[ta2 == 9.96921e+36] = np.nan  # probably only for CESM2
    #ta2[ta2 == 1e+20] = np.nan

    
    qs1 = da_calcsatspechum(ta1, p4d)
    qs2 = da_calcsatspechum(ta2, p4d)
    dqsdt = (qs2 - qs1) / (ta2 - ta1)
    
    #% test plot ---------------------------------------------
    """
    q1_pl = np.mean(q1, axis=(0, -1))
    qs1_pl = np.nanmean(qs1, axis=(0, -1))
    ta1_pl = np.mean(ta1, axis=(0, -1))
    p4d_pl = np.mean(p4d, axis=(0, -1))
    
    fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)
    
    p1 = axes[0, 0].pcolormesh(q1_pl)
    fig.colorbar(p1, ax=axes[0, 0])
    axes[0, 0].set_title("Relative humidity")
    
    axes[0, 0].invert_yaxis()
    
    p2 = axes[0, 1].pcolormesh(qs1_pl, vmin=0, vmax=0.01)
    fig.colorbar(p2, ax=axes[0, 1])
    axes[0, 1].set_title("Specific humidity")
    
    p3 = axes[1, 0].pcolormesh(ta1_pl)
    fig.colorbar(p3, ax=axes[1, 0])
    axes[1, 0].set_title("Air temperature")
    
    p4 = axes[1, 1].pcolormesh(p4d_pl)
    fig.colorbar(p4, ax=axes[1, 1])
    axes[1, 1].set_title("Air pressure")
    
    pl.show()
    pl.close()
    """
    # --------------------------------------------------------
    
    if dlogq:
        rh = 1000 * q1 / qs1 
        dqdt = (rh * dqsdt) / (1000 * q1)  # calculate dlogQ/dT
        dq_or_dlogq = (q2 - q1) / q1
    else:
        rh = q1 / qs1 
        dqdt = rh * dqsdt  # calculate dQ/dT
        dq_or_dlogq = q2 - q1
    # end if else
       
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
    dqdt_nc = f.createVariable(dq_str + "dT", "f4", ("time", "plev", "lat", "lon"))
    dq_or_dlogq_nc = f.createVariable(dq_str, "f4", ("time", "plev", "lat", "lon"))
    dt_nc = f.createVariable("dT", "f4", ("time", "plev", "lat", "lon"))
    
    # pass the data into the variables
    lat_nc[:] = lat
    lon_nc[:] = lon
    lev_nc[:] = levs
    
    dqdt_nc[:] = dqdt  # .compute()
    dq_or_dlogq_nc[:] = dq_or_dlogq  # .compute()
    dt_nc[:] = ta2 - ta1  # .compute()
    
    # set the variable units
    dqdt_nc.units = "K^-1"
    dt_nc.units = "K"
    lat_nc.units = "degrees_north"
    lon_nc.units = "degrees_east"
    lev_nc.units = "Pa"
    
    # enter variable descriptions
    dqdt_nc.description = (dq_str + "/dT assuming constant relative humidity")
    dq_or_dlogq_nc.description = dq_str
    dt_nc.description = "dT"
    
    # enter a file description
    f.description = ("This file contains the monthly " + dq_str + "/dT change in " + aXx + 
                     " assuming constant relative humidity " + 
                     "calculated using the methodology in the kernel_demo_pressure.m from Pendergrass et al. (2018)." + 
                     "\nIt should be possible to use these values to calculate the water vapour radiative response " + 
                     "on a monthly basis.\nThe file further contains the " + dq_str + " and dT. Note that the " +
                     "piControl values of ta and hus have previously been averaged over a 21-year running period." )
    
    # enter the file creation date
    f.history = "Created " + ti.strftime("%d/%m/%y")
    
    # close the dataset
    f.close()
    
    print("\nFile generated.\n")
    
    ti_temp = timeit.default_timer()
    del_t = ti_temp - start
    frac = yr / n_yr
    iter_time = (ti_temp-ti_old) / 60
    
    # remaining = np.round((del_t / frac - del_t) / 60, decimals=1)
    remaining = np.round(iter_time * (n_yr - yr), decimals=2)
    
    print_one_line(str(np.round(yr / n_yr * 100, decimals=1)) + " %  " + str(remaining) + " min remain\n\n")
    
    # update the year-counter
    yr += 1
    # raise Exception("First file generated.")
    
    # set ti_old to the value of ti_temp
    ti_old = copy.deepcopy(ti_temp)
    
# end for i
    
    
    
    
