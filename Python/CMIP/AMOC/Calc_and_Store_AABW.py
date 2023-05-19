"""
Calculate and store abrupt4xCO2 and piControl Antarctic Bottom Water (AABW) formation index for a given model...

See He et al. (2017)

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
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% AABW or "PMOC"
var_out = "AABW"


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
    mod = 48
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
ens_b = "_r1i1p1f2"
ens_f = "_r1i1p1f2"


#%% load the namelist and the model name
import Namelists.CMIP_Dictionaries as cmip_dic
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    a4x = "abrupt4xCO2"
    cmip = "CMIP5"
elif cmip_v == 6:
    import Namelists.Namelist_CMIP6 as nl
    a4x = "abrupt-4xCO2"
    cmip = "CMIP6"
# end if elif

mod_d = nl.models[mod]
mod_n = nl.models_n[mod]
mod_pl = nl.models_pl[mod]


#%% set basin
if var_out == "AABW":
    bas_ind = 2

    if ((mod_pl == "CNRM-CM6.1") | (mod_pl == "CNRM-ESM2.1") | 
        (mod_pl == "E3SM-1.0") | (mod_pl == "EC-Earth3") | (mod_pl == "EC-Earth3-AerChem") | 
        (mod_pl == "EC-Earth3-Veg") | (mod_pl == "FGOALS-g3") | (mod_pl == "IPSL-CM6A-LR") | 
        (mod_pl == "MPI-ESM1.2-HAM") | (mod_pl == "MPI-ESM1.2-HR") | (mod_pl == "MPI-ESM1.2-LR")):
        bas_ind = 0
    elif ((mod_pl == "UKESM1.0-LL") | (mod_pl == "HadGEM3-GC31-LL") | (mod_pl == "HadGEM3-GC31-MM")):
        bas_ind = 1
    elif ((mod_pl == "INM-CM4.8") | (mod_pl == "INM-CM5") | (mod_pl == "NorESM2-LM") | (mod_pl == "NorESM2-MM")):
        bas_ind = 3
    elif (mod_pl == "GFDL-CM3") | (mod_pl == "GFDL-ESM2M"):
        bas_ind = 4        
    # end if elif        
# end if elif    


#%% get the branch year
b_yr = nl.b_times[mod_d]


#%% set data paths
data_dir = ""
out_path = data_dir + f"/{cmip}/Data/{mod_d}/"


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
    # elif var == "msftmyz":
    # try:    
    #     lat = nc_exp.variables["lat"][:]  # in degrees North    
    elif (var == "msftyyz") | (var == "msftmyz"):
    # except:
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


#%% set a depth threshold (see He et al., 2017) and get the corresponding indices  --> include?
### --> msftmyz or msftyyz north of 30 degrees N EXCLUDING the first 500m
z_thr = 0  # in meters
z_ind = lev > z_thr


#%% set a latitude threshold (see He et al., 2017) and get the corresponding indices
lat_thr0 = -69
lat_ind0 = lat < lat_thr0
lat_thr1 = -67
lat_ind1 = lat < lat_thr1
lat_thr2 = -60
lat_ind2 = lat < lat_thr2
lat_thr3 = -40
lat_ind3 = lat < lat_thr3


#%% load the data --> the first entry along the basin dimension corresponds to the Atlantic
aabw_raw_exp0 = np.array(nc_exp.variables[var][:, bas_ind, z_ind, lat_ind0])
aabw_raw_a4x0 = np.array(nc_a4x.variables[var][:, bas_ind, z_ind, lat_ind0])

aabw_raw_exp1 = np.array(nc_exp.variables[var][:, bas_ind, z_ind, lat_ind1])
aabw_raw_a4x1 = np.array(nc_a4x.variables[var][:, bas_ind, z_ind, lat_ind1])

aabw_raw_exp2 = np.array(nc_exp.variables[var][:, bas_ind, z_ind, lat_ind2])
aabw_raw_a4x2 = np.array(nc_a4x.variables[var][:, bas_ind, z_ind, lat_ind2])

aabw_raw_exp3 = np.array(nc_exp.variables[var][:, bas_ind, z_ind, lat_ind3])
aabw_raw_a4x3 = np.array(nc_a4x.variables[var][:, bas_ind, z_ind, lat_ind3])


#%% mask the array where value == 1E20
aabw_raw_exp0 = np.ma.masked_where(aabw_raw_exp0 == 1E20, aabw_raw_exp0)
aabw_raw_a4x0 = np.ma.masked_where(aabw_raw_a4x0 == 1E20, aabw_raw_a4x0)

aabw_raw_exp1 = np.ma.masked_where(aabw_raw_exp1 == 1E20, aabw_raw_exp1)
aabw_raw_a4x1 = np.ma.masked_where(aabw_raw_a4x1 == 1E20, aabw_raw_a4x1)

aabw_raw_exp2 = np.ma.masked_where(aabw_raw_exp2 == 1E20, aabw_raw_exp2)
aabw_raw_a4x2 = np.ma.masked_where(aabw_raw_a4x2 == 1E20, aabw_raw_a4x2)

aabw_raw_exp3 = np.ma.masked_where(aabw_raw_exp3 == 1E20, aabw_raw_exp3)
aabw_raw_a4x3 = np.ma.masked_where(aabw_raw_a4x3 == 1E20, aabw_raw_a4x3)


#%% set the axis or axes over which the max shall be taken
min_ax_exp = (-2, -1)
min_ax_a4x = (-2, -1)
if np.shape(aabw_raw_exp1)[0] == 12:
    min_ax_exp = None
# end if    
if np.shape(aabw_raw_a4x1)[0] == 12:
    min_ax_a4x = None
# end if    


#%% calculate the annual means
aabw_raw_exp_an0 = an_mean(aabw_raw_exp0)
aabw_raw_a4x_an0 = an_mean(aabw_raw_a4x0)

aabw_raw_exp_an1 = an_mean(aabw_raw_exp1)
aabw_raw_a4x_an1 = an_mean(aabw_raw_a4x1)

aabw_raw_exp_an2 = an_mean(aabw_raw_exp2)
aabw_raw_a4x_an2 = an_mean(aabw_raw_a4x2)

aabw_raw_exp_an3 = an_mean(aabw_raw_exp3)
aabw_raw_a4x_an3 = an_mean(aabw_raw_a4x3)


#%% generate the AABW as described by He et al. (2017): take the minimum
aabw_exp0 = np.array(np.min(aabw_raw_exp_an0, axis=min_ax_exp))
aabw_a4x0 = np.array(np.min(aabw_raw_a4x_an0, axis=min_ax_a4x))

aabw_exp1 = np.array(np.min(aabw_raw_exp_an1, axis=min_ax_exp))
aabw_a4x1 = np.array(np.min(aabw_raw_a4x_an1, axis=min_ax_a4x))

aabw_exp2 = np.array(np.min(aabw_raw_exp_an2, axis=min_ax_exp))
aabw_a4x2 = np.array(np.min(aabw_raw_a4x_an2, axis=min_ax_a4x))

aabw_exp3 = np.array(np.min(aabw_raw_exp_an3, axis=min_ax_exp))
aabw_a4x3 = np.array(np.min(aabw_raw_a4x_an3, axis=min_ax_a4x))

if np.shape(aabw_raw_exp1)[0] == 12:
    aabw_exp0 = aabw_exp0[None]
    aabw_exp1 = aabw_exp1[None]
    aabw_exp2 = aabw_exp2[None]
    aabw_exp3 = aabw_exp3[None]
# end if    
if np.shape(aabw_raw_a4x1)[0] == 12:
    aabw_a4x0 = aabw_a4x0[None]
    aabw_a4x1 = aabw_a4x1[None]
    aabw_a4x2 = aabw_a4x2[None]
    aabw_a4x3 = aabw_a4x3[None]
# end if    


#%% if the msftmz is stored in multiple files, load the other files - control experiment
if len(f_l_exp) > 1:
    for i in np.arange(1, len(f_l_exp)):
        nc_add = Dataset(f_l_exp[i])
        
        aabw_add0 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind0]), axis=min_ax_exp)
        aabw_add1 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind1]), axis=min_ax_exp)
        aabw_add2 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind2]), axis=min_ax_exp)
        aabw_add3 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind3]), axis=min_ax_exp)
        
        if np.shape(nc_add.variables[var][:, bas_ind, z_ind, lat_ind1])[0] == 12:
            aabw_add0 = aabw_add0[None]
            aabw_add1 = aabw_add1[None]
            aabw_add2 = aabw_add2[None]
            aabw_add3 = aabw_add3[None]
        # end if            
        aabw_exp0 = np.concatenate((aabw_exp0, aabw_add0), axis=0)
        aabw_exp1 = np.concatenate((aabw_exp1, aabw_add1), axis=0)
        aabw_exp2 = np.concatenate((aabw_exp2, aabw_add2), axis=0)
        aabw_exp3 = np.concatenate((aabw_exp3, aabw_add3), axis=0)
    # end for i
# end if    


#%% if the msftmz is stored in multiple files, load the other files - forced experiment
if len(f_l_a4x) > 1:
    for i in np.arange(1, len(f_l_a4x)):
        nc_add = Dataset(f_l_a4x[i])
        
        aabw_add0 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind0]), axis=min_ax_a4x)
        aabw_add1 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind1]), axis=min_ax_a4x)
        aabw_add2 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind2]), axis=min_ax_a4x)
        aabw_add3 = np.min(an_mean(nc_add.variables[var][:, bas_ind, z_ind, lat_ind3]), axis=min_ax_a4x)
        
        if np.shape(nc_add.variables[var][:, bas_ind, z_ind, lat_ind1])[0] == 12:
            aabw_add0 = aabw_add0[None]
            aabw_add1 = aabw_add1[None]
            aabw_add2 = aabw_add2[None]
            aabw_add3 = aabw_add3[None]
        # end if            
        aabw_a4x0 = np.concatenate((aabw_a4x0, aabw_add0), axis=0)
        aabw_a4x1 = np.concatenate((aabw_a4x1, aabw_add1), axis=0)
        aabw_a4x2 = np.concatenate((aabw_a4x2, aabw_add2), axis=0)
        aabw_a4x3 = np.concatenate((aabw_a4x3, aabw_add3), axis=0)
    # end for i
# end if    


#%% multiply be -1 as in He et al. (2017)
aabw_a4x0 = aabw_a4x0 * -1
aabw_exp0 = aabw_exp0 * -1
aabw_a4x1 = aabw_a4x1 * -1
aabw_exp1 = aabw_exp1 * -1
aabw_a4x2 = aabw_a4x2 * -1
aabw_exp2 = aabw_exp2 * -1
aabw_a4x3 = aabw_a4x3 * -1
aabw_exp3 = aabw_exp3 * -1


#%% plot the AABW
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(aabw_exp0 / 1E10, c="blue", label=f"{var_out} piC - south of 69$\degree$S", linewidth=0.7)
axes.plot(aabw_exp1 / 1E10, c="blue", label=f"{var_out} piC - south of 67$\degree$S", linewidth=0.7, linestyle="-.")
axes.plot(aabw_exp2 / 1E10, c="blue", label=f"{var_out} piC - south of 60$\degree$S", linewidth=0.7, linestyle="--")
axes.plot(aabw_exp3 / 1E10, c="blue", label=f"{var_out} piC - south of 40$\degree$S", linewidth=0.7, linestyle=":")

axes.plot(np.arange(b_yr, b_yr + len(aabw_a4x0)), aabw_a4x0 / 1E10, c="red", linewidth=0.7, 
          label=f"{var_out} abrupt4xCO2 - south of 69$\degree$S")
axes.plot(np.arange(b_yr, b_yr + len(aabw_a4x1)), aabw_a4x1 / 1E10, c="red", linestyle="-.", linewidth=0.7, 
          label=f"{var_out} abrupt4xCO2 - south of 67$\degree$S")
axes.plot(np.arange(b_yr, b_yr + len(aabw_a4x2)), aabw_a4x2 / 1E10, c="red", linestyle="--", linewidth=0.7, 
          label=f"{var_out} abrupt4xCO2 - south of 60$\degree$S")
axes.plot(np.arange(b_yr, b_yr + len(aabw_a4x3)), aabw_a4x3 / 1E10, c="red", linestyle=":", linewidth=0.7, 
          label=f"{var_out} abrupt4xCO2 - south of 40$\degree$S")

axes.legend(ncol=2)

axes.set_title(f"{var_out} strength {mod_pl}")
axes.set_xlabel("Years since piControl start")
axes.set_ylabel(f"{var_out} strength in 10$^{{10}}$ kg s$^{{-1}}$")

# pl.savefig(pl_path + f"{var}_{mod_d}_{exp}_{a4x}.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store the annual values in a netcdf file
print("\nGenerating nc file...\n")

data_path = data_dir + f"/{cmip}/Data/{mod_d}/"
out_name = (f"{var_out.lower()}_{mod_d}_{exp}_{a4x}_{ens_f}.nc")

f = Dataset(data_path + out_name, "w", format="NETCDF4")

# set the branch_time attribute
f.setncattr("branch_time", b_yr)
if cmip_v == 6:
    f.setncattr("variant_label_pic", nc_exp.variant_label)
    f.setncattr("variant_label_a4x", nc_a4x.variant_label)
# end if    

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("years_piC", len(aabw_exp1))
f.createDimension("years_a4x", len(aabw_a4x1))


# create the variables
aabw0_pic_nc = f.createVariable(f"{var_out.lower()}_pic_69S", "f4", "years_piC")
aabw0_a4x_nc = f.createVariable(f"{var_out.lower()}_a4x_69S", "f4", "years_a4x")

aabw1_pic_nc = f.createVariable(f"{var_out.lower()}_pic_67S", "f4", "years_piC")
aabw1_a4x_nc = f.createVariable(f"{var_out.lower()}_a4x_67S", "f4", "years_a4x")

aabw2_pic_nc = f.createVariable(f"{var_out.lower()}_pic_60S", "f4", "years_piC")
aabw2_a4x_nc = f.createVariable(f"{var_out.lower()}_a4x_60S", "f4", "years_a4x")

aabw3_pic_nc = f.createVariable(f"{var_out.lower()}_pic_40S", "f4", "years_piC")
aabw3_a4x_nc = f.createVariable(f"{var_out.lower()}_a4x_40S", "f4", "years_a4x")

# pass the data into the variables
aabw0_pic_nc[:] = aabw_exp0
aabw0_a4x_nc[:] = aabw_a4x0

aabw1_pic_nc[:] = aabw_exp1
aabw1_a4x_nc[:] = aabw_a4x1

aabw2_pic_nc[:] = aabw_exp2
aabw2_a4x_nc[:] = aabw_a4x2

aabw3_pic_nc[:] = aabw_exp3
aabw3_a4x_nc[:] = aabw_a4x3

# variable units
aabw0_pic_nc.units = "kg s-1"
aabw0_a4x_nc.units = "kg s-1"

aabw1_pic_nc.units = "kg s-1"
aabw1_a4x_nc.units = "kg s-1"

aabw2_pic_nc.units = "kg s-1"
aabw2_a4x_nc.units = "kg s-1"

aabw3_pic_nc.units = "kg s-1"
aabw3_a4x_nc.units = "kg s-1"

# variable description
aabw0_pic_nc.description = f"annual mean piControl {var_out} calculated  as min of psi south of 69S"
aabw0_a4x_nc.description = f"annual mean {a4x} {var_out} calculated as min of psi south of 69S "

aabw1_pic_nc.description = f"annual mean piControl {var_out} calculated  as min of psi south of 67S"
aabw1_a4x_nc.description = f"annual mean {a4x} {var_out} calculated as min of psi south of 67S "

aabw2_pic_nc.description = f"annual mean piControl {var_out} calculated  as min of psi south of 60S"
aabw2_a4x_nc.description = f"annual mean {a4x} {var_out} calculated as min of psi south of 60S "

aabw3_pic_nc.description = f"annual mean piControl {var_out} calculated  as min of psi south of 40S"
aabw3_a4x_nc.description = f"annual mean {a4x} {var_out} calculated as min of psi south of 40S "

# add some attributes
f.description = (f"This file contains the annual mean {var_out} for the {exp} and {a4x} experiments for {mod_pl}.\n" +
                 "The calculation of the {var_out} follows (roughly) He et al. (2017), doi: 10.1175/JCLI-D-160581.1. " +
                 "\nSimilar to He et al. (2017) we here take the minimum of the streamfunction at a latitude\n" +
                 "south of 69S but also 67S, south of 60S, and south of 40S.")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()





