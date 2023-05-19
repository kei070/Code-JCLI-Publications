"""
Calculate and store abrupt4xCO2 and piControl AMOC for a given model...

Be sure to adjust data_dir and out_path.
Possibly adjust the paths set in code block "load level and lat".
"""

#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
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


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Round import r2
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% AMOC or "PMOC"
var_out = "AMOC"


#%% set CMIP
try:
    cmip_v = int(sys.argv[1])
except:
    cmip_v = 6
# end try except


#%% set model number
try:
    mod = int(sys.argv[2])
except:
    mod = 17
# end try except


#%% set the variable name: msftmz, msftyz 
if (cmip_v == 6) & (mod in [0, 1, 6, 8, 9, 10, 11, 17, 20, 21, 25, 27, 31, 32, 36, 37, 38, 39, 40, 44, 45, 46]):
    var = "msftmz"
elif cmip_v == 6:   
    var = "msftyz"
elif (cmip_v == 5) & (mod in [0, 1, 9, 11]):
    var = "msftyyz"
elif cmip_v == 5:
    var = "msftmyz"    
# end if else
  

#%% set control experiment name
exp = "piControl"


#%% set ensemble names
ens_b = "_r1i1p1f1"
ens_f = "_r1i1p1f1"


#%% load the namelist and the model name
import Namelists.CMIP_Dictionaries as cmip_dic
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    a4x = "abrupt4xCO2"
    # a4x = "1pctCO2"
    cmip = "CMIP5"
elif cmip_v == 6:
    import Namelists.Namelist_CMIP6 as nl
    a4x = "abrupt-4xCO2"
    # a4x = "abrupt-2xCO2"
    # a4x = "1pctCO2"
    cmip = "CMIP6"
# end if elif

mod_d = nl.models[mod]
mod_n = nl.models_n[mod]
mod_pl = nl.models_pl[mod]


#%% set data paths
data_dir = ""
out_path = data_dir + f"/{cmip}/Data/{mod_d}/"


#%% set basin
if var_out == "AMOC":
    bas_ind = 0
    
    if (mod_pl == "GFDL-CM3") | (mod_pl == "GFDL-ESM2M"):
        bas_ind = 3
    # end if
    
    if ((mod_pl == "CNRM-CM6.1") | (mod_pl == "CNRM-ESM2.1") | (mod_pl == "E3SM-1.0") | (mod_pl == "EC-Earth3") | 
        (mod_pl == "EC-Earth3-AerChem") | (mod_pl == "EC-Earth3-Veg") | (mod_pl == "IPSL-CM6A-LR") | 
        (mod_pl == "MPI-ESM1.2-LR") | (mod_pl == "MPI-ESM1.2-HAM") | (mod_pl == "MPI-ESM1.2-HR") | 
        (mod_pl == "FGOALS-g3")):
        bas_ind = 1
    # end if
    
elif var_out == "PMOC":
    bas_ind = 1

    if ((mod_pl == "CNRM-CM6.1") | (mod_pl == "CNRM-ESM2.1") | (mod_pl == "EC-Earth3") | (mod_pl == "MPI-ESM1.2-LR") | 
        (mod_pl == "UKESM1.0-LL")):
        bas_ind = 2
    # end if
# end if elif    


#%% get the branch year
b_yr = nl.b_times[mod_d]


#%% set the model data path
if (ens_b == "_r1i1p1f1") | (ens_b == "_r1i1p1"):
    data_p_exp = (data_dir + f"/{cmip}/Data/{mod_d}/piControl_{var}_Files/" + 
                 f"{var}_*_piControl_*.nc")
else:
    data_p_exp = (data_dir + f"/{cmip}/Data/{mod_d}/piControl_{var}_Files{ens_b}/" + 
                 f"{var}_*_piControl_*.nc")
if (ens_f == "_r1i1p1f1") | (ens_f == "_r1i1p1"):
    data_p_a4x = (data_dir + f"/{cmip}/Data/{mod_d}/{a4x}_{var}_Files/" + 
                 f"{var}_*_{a4x}_*.nc")
else:
    data_p_a4x = (data_dir + f"/{cmip}/Data/{mod_d}/{a4x}_{var}_Files{ens_f}/" + 
                 f"{var}_*_{a4x}_*.nc")
# end if else

pl_path = f"/Plots/{var}_Plots/"


#%% load the file(s)
f_l_exp = sorted(glob.glob(data_p_exp), key=str.casefold)
f_l_a4x = sorted(glob.glob(data_p_a4x), key=str.casefold)

nc_exp = Dataset(f_l_exp[0])
nc_a4x = Dataset(f_l_a4x[0])


#%% for CESM2-FV2 level units are centimeters => introduce a factor that corrects this
lev_fac = 1
if mod_pl[:5] == "CESM2":
    lev_fac = 100
# end if


#%% load level and lat
try:
    lev = nc_exp.variables["lev"][:] / lev_fac  # in meters
except:    
    lev = nc_exp.variables["olevel"][:]  # in meters
# end try except

if mod_pl not in ["CNRM-CM6.1", "CNRM-ESM2.1", "IPSL-CM6A-LR"]:
    if (var == "msftyz") & (mod_pl != "GFDL-ESM4"):
        lat = nc_exp.variables["rlat"][:]  # in degrees North
    elif var == "msftmz":
        lat = nc_exp.variables["lat"][:]  # in degrees North    
    elif mod_pl == "GFDL-ESM4":        
        lat = nc_exp.variables["y"][:]  # in degrees North
    elif (var == "msftyyz") | (var == "msftmyz"):
        try:
            lat = nc_exp.variables["rlat"][:]  # in degrees North     
        except:
            lat = nc_exp.variables["lat"][:]  # in degrees North    
    # end if else     
elif mod_pl == "CNRM-CM6.1":
    sic_nc = Dataset(glob.glob(data_dir + f"/{cmip}/Data/{mod_d}/piControl_siconc_Files_r1i1p1f2/*")[0])
    lat = sic_nc.variables["lat"][:, 250]
elif mod_pl == "CNRM-ESM2.1":
    sic_nc = Dataset(glob.glob(data_dir + f"/{cmip}/Data/{mod_d}/piControl_siconc_Files_r1i1p1f2/*")[0])
    lat = sic_nc.variables["lat"][:, 250]    
elif mod_pl == "IPSL-CM6A-LR":
    lat = nc_exp.variables["nav_lat"][:, 0]  # in degrees North
# end if elif


#%% set a depth threshold (see Lin et al., 2019) and get the corresponding indices
### --> msftmyz or msftyyz north of 30 degrees N EXCLUDING the first 500m
z_thr = 500  # in meters
z_ind = lev > z_thr


#%% set a latitude threshold (see Lin et al., 2019) and get the corresponding indices
lat_thr = 30
lat_ind = lat > lat_thr


#%% load the data --> the first entry along the basin dimension corresponds to the Atlantic
amoc_raw_exp = np.array(nc_exp.variables[var][:, bas_ind, z_ind, lat_ind])
amoc_raw_a4x = np.array(nc_a4x.variables[var][:, bas_ind, z_ind, lat_ind])


#%% mask the array where value == 1E20
amoc_raw_exp = np.ma.masked_where(amoc_raw_exp == 1E20, amoc_raw_exp)
amoc_raw_a4x = np.ma.masked_where(amoc_raw_a4x == 1E20, amoc_raw_a4x)


#%% set the axis or axes over which the max shall be taken
max_ax_exp = (-2, -1)
max_ax_a4x = (-2, -1)
if np.shape(amoc_raw_exp)[0] == 12:
    max_ax_exp = None
# end if    
if np.shape(amoc_raw_a4x)[0] == 12:
    max_ax_a4x = None
# end if    


#%% calculate the annual means
amoc_raw_exp_an = an_mean(amoc_raw_exp)
amoc_raw_a4x_an = an_mean(amoc_raw_a4x)

amoc_raw_exp_m = np.mean(amoc_raw_exp_an, axis=0)
if np.ndim(amoc_raw_exp_m) > 1:
    damoc_raw_an = amoc_raw_a4x_an - amoc_raw_exp_m[None, :, :]
else:    
    damoc_raw_an = amoc_raw_a4x_an - amoc_raw_exp_m[None, :]
# end if else    
    

#%% generate the AMOC as described by Lin et al. (2019): take the maximum
amoc_exp = np.array(np.max(amoc_raw_exp_an, axis=max_ax_exp))
amoc_a4x = np.array(np.max(amoc_raw_a4x_an, axis=max_ax_a4x))

if np.shape(amoc_raw_exp)[0] == 12:
    amoc_exp = amoc_exp[None]
# end if    
if np.shape(amoc_raw_a4x)[0] == 12:
    amoc_a4x = amoc_a4x[None]
# end if    

"""
# get the index with the maximum AMOC and AMOC change
amoc_max_ind = np.argmax(np.max(amoc_raw_a4x_an, axis=1), axis=-1)
damoc_min_ind = np.argmin(np.max(damoc_raw_an, axis=1), axis=-1)
print(lat[lat_ind][amoc_max_ind[0]])
print(lat[lat_ind][damoc_min_ind[0]])

# loop over the max (d)AMOC index arrays to get the latitude of maximum (d)AMOC per year
amoc_max = np.zeros(len(amoc_max_ind))
damoc_min = np.zeros(len(damoc_min_ind))

for i in np.arange(len(amoc_max)):
    amoc_max[i] = lat[lat_ind][amoc_max_ind[i]]
    damoc_min[i] = lat[lat_ind][damoc_min_ind[i]]
# end for i


#%% test plot of the dAMOC as a pcolormesh
t_step = 0  # np.shape(amoc_raw_a4x_an)[0] - 1

x, y = np.meshgrid(lat[lat_ind], lev[z_ind])

fig = pl.figure(figsize=(12, 5))
gs = gridspec.GridSpec(nrows=1, ncols=2)

ax1 = pl.subplot(gs[0, 0])
ax2 = pl.subplot(gs[0, 1])

p1 = ax1.pcolormesh(x, y, amoc_raw_a4x_an[t_step, :, :] / 1E10, vmin=-2, vmax=2, cmap=cm.RdBu_r, shading="auto")
ax1.axvline(amoc_max[t_step], c="black")
cb1 = fig.colorbar(p1, ax=ax1)
cb1.set_label("AMOC in 10$^{10}$ kg s $^{-1}$")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Depth in m")
ax1.invert_yaxis()
ax1.set_title(mod_pl + f" AMOC abrupt-4xCO2 Year {t_step}")

p2 = ax2.pcolormesh(x, y, damoc_raw_an[t_step, :, :] / 1E10, vmin=-2, vmax=2, cmap=cm.RdBu_r, shading="auto")
ax2.axvline(damoc_min[t_step], c="black")
cb2 = fig.colorbar(p2, ax=ax2)
cb2.set_label("$\Delta$AMOC in 10$^{9}$ kg s $^{-1}$")
ax2.set_yticklabels([])
ax2.set_xlabel("Latitude")
ax2.invert_yaxis()
ax2.set_title(mod_pl + f" $\Delta$AMOC Year {t_step}")

fig.subplots_adjust(wspace=0.1)

pl.show()
pl.close()


#%% test plot of the max (d)AMOC latitude per year
fig = pl.figure(figsize=(8, 5))
gs = gridspec.GridSpec(nrows=1, ncols=1)

ax1 = pl.subplot(gs[0, 0])

ax1.plot(amoc_max, c="gray", linewidth=0.5)
ax1.axhline(np.mean(amoc_max), c="gray", label=f"abrupt4xCO2; mean={r2(np.mean(amoc_max))}")

ax1.plot(damoc_min, c="black", linewidth=0.5)
ax1.axhline(np.mean(damoc_min), c="black", label=f"abrupt4xCO2 minus piControl; mean={r2(np.mean(damoc_min))}")

ax1.legend()

ax1.set_xlabel("Years since 4xCO$_2$")
ax1.set_ylabel("Latitude")

pl.show()
pl.close()
"""

#%% if the msftmz is stored in multiple files, load the other files - control experiment
if len(f_l_exp) > 1:
    for i in np.arange(1, len(f_l_exp)):
        nc_add = Dataset(f_l_exp[i])
        
        amoc_add = np.max(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind]), axis=max_ax_exp)
        if np.shape(nc_add.variables[var][:, bas_ind, z_ind, lat_ind])[0] == 12:
            amoc_add = amoc_add[None]
        # end if            
        
        amoc_exp = np.concatenate((amoc_exp, amoc_add), axis=0)
    # end for i
# end if    


#%% if the msftmz is stored in multiple files, load the other files - forced experiment
if len(f_l_a4x) > 1:
    for i in np.arange(1, len(f_l_a4x)):
        nc_add = Dataset(f_l_a4x[i])
        
        amoc_add = np.max(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind]), axis=max_ax_a4x)
        if np.shape(nc_add.variables[var][:, bas_ind, z_ind, lat_ind])[0] == 12:
            amoc_add = amoc_add[None]
        # end if            
        
        amoc_a4x = np.concatenate((amoc_a4x, amoc_add), axis=0)
    # end for i
# end if    


#%% plot the AMOC
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(amoc_exp / 1E10, c="blue", label=f"{var_out} piC", linewidth=0.7)
axes.plot(np.arange(b_yr, b_yr + len(amoc_a4x)), amoc_a4x / 1E10, c="red", label=f"{var_out} {a4x}", linewidth=0.7)

axes.legend()

axes.set_title(f"{var_out} strength {mod_pl}")
axes.set_xlabel("Years since piControl start")
axes.set_ylabel(f"{var_out} strength in 10$^{{10}}$ kg s$^{{-1}}$")

# pl.savefig(pl_path + f"{var}_{mod_d}_{exp}_{a4x}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store the annual values in a netcdf file
print("\nGenerating nc file...\n")
out_name = (f"{var_out.lower()}_{mod_d}_{exp}_{a4x}{ens_f}.nc")

f = Dataset(out_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
f.setncattr("branch_time", b_yr)
if cmip_v == 6:
    f.setncattr("variant_label_pic", nc_exp.variant_label)
    f.setncattr("variant_label_a4x", nc_a4x.variant_label)
# end if    

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("years_piC", len(amoc_exp))
f.createDimension("years_a4x", len(amoc_a4x))

# create the variables
amoc_pic_nc = f.createVariable(f"{var_out.lower()}_pic", "f4", "years_piC")
amoc_a4x_nc = f.createVariable(f"{var_out.lower()}_a4x", "f4", "years_a4x")

# pass the data into the variables
amoc_pic_nc[:] = amoc_exp
amoc_a4x_nc[:] = amoc_a4x

# variable units
amoc_pic_nc.units = "kg s-1"
amoc_a4x_nc.units = "kg s-1"

# variable description
amoc_pic_nc.description = f"annual mean piControl {var_out} ({ens_b}) as calculated according to Lin et al. (2019)"
amoc_a4x_nc.description = f"annual mean {a4x} {var_out} ({ens_f}) as calculated according to Lin et al. (2019)"

# add some attributes
f.description = (f"This file contains the annual mean {var_out} for the {exp} and {a4x} experiments for {mod_pl}.\n" +
                 "The calculation of the {var_out} follows Lin, Y.-J., Y.-T. Hwang, P. Ceppi, and J. M. Gregory " + 
                 "(2019)\nin Geophysical Research Letters, 46, https://doi.org/10.1029/2019GL083084")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()





