"""
Script for calculating monthly dQ/dT. The results are used in the kernel calculations.

Be sure to set the data_path correctly.
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


#%% setup case name

# BE SURE TO SET UP THE CORRECT CONTROL CASE!
case_con = "Proj2_KUE"
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% dlog(Q)/dT or dQ/dT?
dlogq = True

# change some parameters accordingly
dq_str = "dQ"
if dlogq:
    dq_str = "dlogQ"
# end if


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
data_path = "/Data/"

# output path
out_path = data_path + case + "/" + dq_str + "dT_Files/"
    
os.makedirs(out_path, exist_ok=True)


#%% get the lists of hus ta files
case_f_list = sorted(glob.glob(data_path + f"{case}/CAM_Files/*.nc"), key=str.casefold)
# con_f_list = sorted(glob.glob(data_path + f"{case_con}/CAM_Files/*.nc"), key=str.casefold)


#%% number of years
n_yr = int(len(case_f_list) / 12)


#%% load first files

# load in the respective first file
case_nc = Dataset(case_f_list[0])
con_q_nc = Dataset(glob.glob(data_path + case_con + f"/Q_Climatology_{case_con}_*.nc")[0])
con_t_nc = Dataset(glob.glob(data_path + case_con + f"/T_Climatology_{case_con}_*.nc")[0])

# load the first hus file data
q_case = da.ma.masked_array(case_nc.variables["Q"], lock=True)
t_case = da.ma.masked_array(case_nc.variables["T"], lock=True)

# load the control run climatology of Q and T
q_con = da.ma.masked_array(con_q_nc.variables["Q"], lock=True)
t_con = da.ma.masked_array(con_t_nc.variables["T"], lock=True)


#%% get the grid coordinates (vertical and horizontal)

# get the levels
levs = case_nc.variables["lev"][:]

# get lat, lon, and levels for the model data
lat = case_nc.variables["lat"][:]
lon = case_nc.variables["lon"][:]


#%%  loop over the files and concatenate the values via dask
if len(case_f_list) > 1:
    for i in np.arange(1, len(case_f_list)):
        q_case = da.concatenate([q_case, da.ma.masked_array(Dataset(case_f_list[i]).variables["Q"], lock=True)])
        t_case = da.concatenate([t_case, da.ma.masked_array(Dataset(case_f_list[i]).variables["T"], lock=True)])
    # end for i
else:
    print("\nOnly one ta and q " + case + " file...\n")
# end if else

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
p4d[:, :, :, :] = levs[None, :, None, None]


#%% execute the calculations for ta1 and q1 as well as qs1 for the dQ/dT calculations later
q1 = np.array(np.ma.filled(q_con[:, :, :, :].compute(), np.nan))
ta1 = np.array(np.ma.filled(t_con[:, :, :, :].compute(), np.nan))
qs1 = da_calcsatspechum(ta1, p4d)


#%% test plot

q1_pl = np.mean(q1, axis=(0, -1))
qs1_pl = np.nanmean(qs1, axis=(0, -1))
ta1_pl = np.mean(ta1, axis=(0, -1))
p4d_pl = np.mean(p4d, axis=(0, -1))

fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)

p1 = axes[0, 0].pcolormesh(q1_pl)
fig.colorbar(p1, ax=axes[0, 0])
axes[0, 0].set_title("Specific humidity")

axes[0, 0].invert_yaxis()

p2 = axes[0, 1].pcolormesh(qs1_pl, vmin=0, vmax=0.01)
fig.colorbar(p2, ax=axes[0, 1])
axes[0, 1].set_title("Saturation humidity")

p3 = axes[1, 0].pcolormesh(ta1_pl)
fig.colorbar(p3, ax=axes[1, 0])
axes[1, 0].set_title("Air temperature")

p4 = axes[1, 1].pcolormesh(p4d_pl)
fig.colorbar(p4, ax=axes[1, 1])
axes[1, 1].set_title("Air pressure")

pl.show()
pl.close()


#%% calculate dqdt
print("\nCalculating " + dq_str + "/dT...\n")

sta_yr = 0

yr = sta_yr + 1  # year-counter for the file name
start = timeit.default_timer()
ti_old = copy.deepcopy(start)
for yri, i in enumerate(np.arange(sta_yr, int(n_yr*12), 12)):
    
    yri = yri + sta_yr

    # set output name
    out_name =  dq_str + "dT_CESM2-SOM_" + case + "_" + f'{yr:04}' + "01" + "-" + f'{yr:04}' + "12" + ".nc"
    
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
    ta2 = np.array(np.ma.filled(t_case[i:(i+12), :, :, :].compute(), np.nan))
    q2 = np.array(np.ma.filled(q_case[i:(i+12), :, :, :].compute(), np.nan))
    
    """
    if yri == 50:
        pl.imshow(ta2[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta abrupt-4xC02 CESM2-SOM")
        pl.show()
        pl.close()
        
        pl.imshow(ta1[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta piControl Running CESM2-SOM")
        pl.show()
        pl.close()
        
        pl.imshow(ta2[3, 0, :, :] - ta1[3, 0, :, :], origin="lower")
        pl.colorbar()
        pl.title("ta Difference CESM2-SOM")
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

    
    qs2 = da_calcsatspechum(ta2, p4d)
    dqsdt = (qs2 - qs1) / (ta2 - ta1)
    
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
    f = Dataset(out_path + out_name, "w", format="NETCDF4")
    
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
    f.description = ("This file contains the monthly CESM2-SOM " + dq_str + "/dT change in the case " + case + 
                     " assuming constant relative humidity " + 
                     "calculated using the methodology in the kernel_demo_pressure.m from Pendergrass et al. (2018)." + 
                     "\nIt should be possible to use these values to calculate the water vapour radiative response " + 
                     "on a monthly basis.\nThe file further contains the " + dq_str + " and dT. Note that the " +
                     "dT and dQ are calculated with respect to an average over a 28-year control run of CESM2-SOM." )
    
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
    
    
    
    
