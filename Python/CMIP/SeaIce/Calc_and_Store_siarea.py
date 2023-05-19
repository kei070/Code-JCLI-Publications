"""
Calculate and store northern and southern hemisphere sea-ice area from siconc and areacello.

Note that there is a special case for AWI-CM-1.1-MR.
"""


#%% imports
import os
import sys
import glob
import numpy as np
import dask.array as da
import pylab as pl
from netCDF4 import Dataset
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_AnMean import an_mean


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 41
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except


#%% set the experiment; if abrupt4xCO2 enter "a4x"
# exp = "piControl"
exp = "a4x"
# exp = "a2x"


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    var = "sic"

    if exp == "a4x":
        exp = "abrupt4xCO2"
    # end if    
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    var = "siconc"
    
    if exp == "a4x":
        exp = "abrupt-4xCO2"
    if exp == "a2x":
        exp = "abrupt-2xCO2"        
    # end if
# end if elif

ens =  ""  # "_r13i1p1f1"
# ens = "_r2i1p1f1"


#%% get the branch year
n_yr = 150
b_sta_ind = nl.b_times[nl.models[mod]]
b_end_ind = n_yr + b_sta_ind


#%% set path
# --> must contain the folder loaded in the code block "list the files" below
data_path = ""


#%% list the files
f_list = sorted(glob.glob(data_path + f"/{exp}_{var}_Files{ens}/{var}_*{exp}*.nc"), key=str.casefold)


#%% load first file and from that lat and lon
first_file = Dataset(f_list[0])
try:
    lat = first_file.variables["latitude"][:]
    lon = first_file.variables["longitude"][:]
except:
    try:
        lat = first_file.variables["lat"][:]
        lon = first_file.variables["lon"][:]
    except:
        lat = first_file.variables["nav_lat"][:]
        lon = first_file.variables["nav_lon"][:]
    # end try except
# end try except


#%% get the total grid grid cell number
nx = np.shape(lat)[1]
ny = np.shape(lat)[0]
ngrid = int(nx*ny)


#%% extract the ensemble and the branch time
if cmip_v == 5:
    # ens = first_file.parent_experiment_rip
    # if ens == "N/A":
    #     ens = "r1i1p1"
    # end if
    ens = "r1i1p1"
    bt = first_file.branch_time
elif cmip_v == 6:
    ens = first_file.variant_label
    bt = first_file.branch_time_in_parent
# end if elif    


#%% loop over all files and perform the calculations
si_area_n_sum = []
si_area_s_sum = []

ar_var = "areacello"
if nl.models_pl[mod] == "NESM3":
    ar_var = "areacelli"
# end if   

for i in np.arange(len(f_list)):
    # load the siconc nc file as well as the areacello file
    sic_nc = Dataset(f_list[i])
    aro_nc = Dataset(glob.glob(data_path + f"{ar_var}*.nc")[0])
    # aro_nc = Dataset(glob.glob(data_path + f"{ar_var}*rcp45*.nc")[0])
    
    # load the siconc variable
    sic = sic_nc.variables[var][:]   
    sic = da.ma.masked_array(sic_nc.variables[var], lock=True)
    
    # if nl.models_pl[mod] == "KIOST-ESM":
    #     sic = sic / 100
    # end if    

    aro = aro_nc.variables[ar_var][:np.shape(sic)[-2], :np.shape(sic)[-1]]
    
    if (nl.models_pl[mod] == "FGOALS-g3") | (nl.models_pl[mod] == "FGOALS-f3-L"): 
        aro = aro[::-1, :]
        aro[aro > 1E20] = np.nan
    # end if        
    
    # if latitude is a 1d variable expand it to 2d
    if np.ndim(lat) < 2:
        lon, lat = np.meshgrid(lon, lat)
    # end if   
    
    # divide areacello by 100 (to convert percent to fraction) and multiply this field onto the siconc variable
    si_area = sic * aro[None, :, :] / 100
    
    # reshape
    si_temp = np.reshape(si_area, (np.shape(si_area)[0], ngrid))
    
    # get lat index
    latn = np.reshape((lat > 0), (ngrid))
    lats = np.reshape((lat < 0), (ngrid))
    
    # calculate the sum over the northern hemisphere sea ice area
    si_area_n_sum.append(np.nansum(si_temp[:, latn], axis=-1).compute())
    si_area_s_sum.append(np.nansum(si_temp[:, lats], axis=-1).compute())
    
    print(f"{i+1} / {len(f_list)}")
# end for i


#%% test plot the sic(conc) as a map
mon = 10

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(12, 6))

p1 = axes.imshow(sic[mon, :, :], origin="lower")
fig.colorbar(p1, ax=axes)

axes.set_title(nl.models_pl[mod] + " " + var + f" (month {mon})") 

pl.show()
pl.close()


#%% test plot the sea-ice area as a map
mon = 10

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(12, 6))

p1 = axes.imshow(si_area[mon, :, :], origin="lower")
fig.colorbar(p1, ax=axes)

axes.set_title(nl.models_pl[mod] + f" sea-ice area (month {mon})") 

pl.show()
pl.close()


#%% test plot the area size data as a map
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(12, 6))

p1 = axes.imshow(aro, origin="lower")
fig.colorbar(p1, ax=axes)

axes.set_title(nl.models_pl[mod] + " areacello") 

pl.show()
pl.close()


#%% concatenate the individual time series
si_area_n_sum = np.concatenate(si_area_n_sum)
si_area_s_sum = np.concatenate(si_area_s_sum)


#%% calculate the hemispheric internal variability
nh_int_var = np.std(an_mean(si_area_n_sum))
sh_int_var = np.std(an_mean(si_area_s_sum))


#%% test plot
"""
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(si_area_n_sum[:12*125]/1E12, c="blue", label="area")
# axes.plot(si_ext_n_sum/1E12, c="red", label="extent")
axes.legend()
axes.set_title(nl.models_pl[mod])
axes.set_ylabel("10$^6$ km$^2$")
axes.set_xlabel("months since branching of 4$\\times$CO$_2$")

pl.show()
pl.close()


#%% special for GFDL-CM3
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.plot(np.arange(12*115, 12*125), si_area_n_sum[12*115:12*125]/1E12, c="blue")
axes.plot(np.arange(12*125, 12*135), np.tile(si_area_n_sum[12*124:12*125]/1E12, 10), c="red")
axes.plot(np.arange(12*135, 12*145), si_area_n_sum[12*125:12*135]/1E12, c="blue")
# axes.plot(si_ext_n_sum/1E12, c="red", label="extent")
axes.set_title(nl.models_pl[mod])
axes.set_ylabel("10$^6$ km$^2$")
axes.set_xlabel("months since branching of 4$\\times$CO$_2$")

pl.show()
pl.close()
"""

#%% insert the 10 missing years for GFDL-CM3
if (exp == "abrupt4xCO2") & (nl.models_pl[mod] == "GFDL-CM3"):
    si_area_n_temp = np.zeros(12*150)
    si_area_s_temp = np.zeros(12*150)
    
    # fill the arrays
    si_area_n_temp[:12*125] = si_area_n_sum[:12*125]
    si_area_n_temp[12*125:12*135] = np.tile(si_area_n_sum[12*124:12*125], 10)
    si_area_n_temp[12*135:] = si_area_n_sum[12*125:]
    si_area_n_sum = si_area_n_temp
    
    si_area_s_temp[:12*125] = si_area_s_sum[:12*125]
    si_area_s_temp[12*125:12*135] = np.tile(si_area_s_sum[12*124:12*125], 10)
    si_area_s_temp[12*135:] = si_area_s_sum[12*125:]
    si_area_s_sum = si_area_s_temp
# end if    


#%% get the number of years in for the output file
n_yr = str(int(len(si_area_n_sum) / 12))


#%% test plot

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(12, 6))

axes.plot(an_mean(si_area_n_sum)/1E12, c="blue", 
             label=f"NH; int. var.: {np.round(nh_int_var/1E12, 3)} $\\times$ 10$^6$ km$^2$")
axes.plot(an_mean(si_area_s_sum)/1E12, c="red", 
             label=f"SH; int. var.: {np.round(sh_int_var/1E12, 3)} $\\times$ 10$^6$ km$^2$")
axes.axvline(x=b_sta_ind, c="gray")
axes.axvline(x=b_end_ind, c="gray")
# axes.plot(an_mean(si_ext_n_sum)/1E12, c="red", label="extent")
axes.legend()
axes.set_title(nl.models_pl[mod] + " annual mean sea-ice area")
axes.set_ylabel("10$^6$ km$^2$")
# axes.set_xlabel("years since branching of 4$\\times$CO$_2$")
axes.set_xlabel("years")
# axes.set_ylim((0, 18))
# axes.set_xlim((-5, 150))

pl.show()
pl.close()


#%% store sea ice area and extent as calculated above for the model in question
out_name = f"si_ext_area_{nl.models[mod]}_{exp}_{ens}_{n_yr}yrs.nc"

f = Dataset(data_path + out_name, "w", format="NETCDF4")

# add attributes
if cmip == "CMIP6":
    f.setncattr("branch_time", bt)
    f.setncattr("variante_label", ens)
elif cmip == "CMIP5":
    f.setncattr("branch_time", bt)
    f.setncattr("ensemble", ens)    
# end if elif

# create dimensions
f.createDimension("time", len(si_area_n_sum))

# create variables
si_area_n_sum_nc = f.createVariable("si_area_n", "f8", "time")
si_area_s_sum_nc = f.createVariable("si_area_s", "f8", "time")

# fill the variables
si_area_n_sum_nc[:] = si_area_n_sum
si_area_s_sum_nc[:] = si_area_s_sum

# variable description
si_area_n_sum_nc.description = (f"northern hemispheric sea ice area calculated by multiplying {var}/100 on areacello" +
                                " and summing over cells with latitude > 0 degrees north")
si_area_s_sum_nc.description = (f"southern hemispheric sea ice area calculated by multiplying {var}/100 on areacello" +
                                " and summing over cells with latitude < 0 degrees north")


# file description
desc_add = ""
if nl.models_pl[mod] == "GFDL-CM3":
    desc_add = (" Note the for this model (GFDL-CM3) the years 126-135 are missing and here the year 125 has been" +
                " repeated ten times to fill these ten years.")
if nl.models_pl[mod] == "HadGEM2-ES":    
    desc_add = (" Note the for this model (HadGEM2-ES) the data start in December and not in January. Hence there are" +
                " 1813 data point (151 year plus December of the year before the first year included here).")    
# end if    
f.description = (f"{nl.models_pl[mod]} {exp} northern and southern hemisphere sea ice area and extent as monthly mean " +
                 "time series." + desc_add)

f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()
