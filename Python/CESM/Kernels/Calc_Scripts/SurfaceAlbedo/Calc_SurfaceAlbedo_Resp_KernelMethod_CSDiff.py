"""
Calculate the surface albedo radiative "cloud adjustment" for CESM via the kernel method.

This script uses the piControl 21-year running means as unperturbed state. The 21-year running means are computed here.

Be sure to adjust paths in code block "set paths".
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
# from dask.distributed import Client


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Flip import flip
from Functions.Func_Print_In_One_Line import print_one_line


#%% establish connection to the client
# c = Client("localhost:8786")


#%% setup case name
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[4]
except:
    toa_sfc = "TOA"
# end try except


#%% choose a kernel: "Sh08", "So08", "BM13", "H17", "P18"
try:
    kl = sys.argv[2]
except:
    kl = "So08"
# end try except


#%% set the kernel state (as of now only applicable for the Block & Mauritsen, 2013 kernels)
#   possible values: "pi", "2x", "4x", "8x"
try:
    k_st = sys.argv[3]
except:
    k_st = "pi"
# end try except

# addend for file names regarding the climate state under which the kernels are derived
state_fn = {"pi":"CTRL", "2x":"2xCO2", "4x":"4xCO2", "8x":"8xCO2"}
state_ofn = {"pi":"_Kernel", "2x":"_2xCO2Kernel", "4x":"_4xCO2Kernel", "8x":"_8xCO2Kernel"}  # out file name

# note that the reasoning behind not giving an addend to the CTRL kernels is that the climate state in which they were
# calculated should more or less corrspond to the ones in which the others were calculated


#%% file name suffix for at-surface
sfc_add = ""
if toa_sfc == "SFC":
    sfc_add = "_SFC"
# end if


#%% set up dictionaries for the kernel choice

# kernels names for keys
k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]

# kernel names for plot names/titles, texts, paths/directories etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_t = {"Sh08":"Shell et al. (2008)", "So08":"Soden et al. (2008)", "BM13":"Block and Mauritsen (2013)", 
       "H17":"Huang et al. (2017)", "P18":"Pendergrass et al. (2018)", "S18":"Smith et al. (2018)"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", 
       "H17":"Huang_etal_2017", "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# set the regrid method per kernel
k_rg = {"Sh08":"scipy", "So08":"scipy", "BM13":"scipy", "H17":"scipy", "P18":"scipy", "S18":"scipy"}

# set kernel directory addend
k_ad = {"Sh08":"kernels/", "So08":"", "BM13":"", "H17":"kernel-highR/toa/", "P18":"cam5-kernels/kernels/", "S18":""}

# clear-sky addend
k_as_ad = {"Sh08":"", "So08":"", "BM13":"d", "H17":"cld", "P18":"", "S18":""}
k_cs_ad = {"Sh08":"_clr", "So08":"clr", "BM13":"f", "H17":"clr", "P18":"C", "S18":"_cs"}

# set kernel file names
k_as_fn = {"Sh08":"CAM3_albedo_sw" + k_as_ad[kl] + "_kernel.nc", "So08":"TOA_GFDL_Kerns.nc", 
           "BM13":state_fn[k_st] + "_kernel_mm_1950.nc", "H17":"RRTMG_alb_toa_" + k_as_ad[kl] + "_highR.nc", 
           "P18":"alb.kernel.nc", "S18":"HadGEM2_sw_TOA_L38.nc"}
k_cs_fn = {"Sh08":"CAM3_albedo_sw" + k_cs_ad[kl] + "_kernel.nc", "So08":"TOA_GFDL_Kerns.nc", 
           "BM13":state_fn[k_st] + "_kernel_mm_1950.nc", "H17":"RRTMG_alb_toa_" + k_cs_ad[kl] + "_highR.nc", 
           "P18":"alb.kernel.nc", "S18":"HadGEM2_sw_TOA_L38.nc"}

# set kernel variable names
if toa_sfc == "TOA":
    k_as_vn = {"Sh08":"monkernel", "So08":"sw" + k_as_ad[kl] + "_a", "BM13":"A_sra" + k_as_ad[kl] + "0", 
               "H17":"swkernel", "P18":"FSNT" + k_as_ad[kl], "S18":"albedo_sw" + k_as_ad[kl]} 
    k_cs_vn = {"Sh08":"monkernel", "So08":"sw" + k_cs_ad[kl] + "_a", "BM13":"A_sra" + k_cs_ad[kl] + "0", 
               "H17":"swkernel", "P18":"FSNT" + k_cs_ad[kl], "S18":"albedo_sw" + k_cs_ad[kl]}    
elif toa_sfc == "SFC":
    k_as_vn = {"Sh08":"monkernel", "So08":"sw" + k_as_ad[kl] + "_a", "BM13":"A_sra" + k_as_ad[kl] + "0", 
               "H17":"swkernel", "P18":"FSNS" + k_as_ad[kl], "S18":"albedo_sw" + k_as_ad[kl]}
    k_cs_vn = {"Sh08":"monkernel", "So08":"sw" + k_cs_ad[kl] + "_a", "BM13":"A_sra" + k_cs_ad[kl] + "0", 
               "H17":"swkernel", "P18":"FSNS" + k_cs_ad[kl], "S18":"albedo_sw" + k_cs_ad[kl]}
# end if elif

# set kernel grid variable names
k_gvn = {"Sh08":["lat", "lon"], "So08":["latitude", "longitude"], "BM13":["lat", "lon"], 
         "H17":["lat", "lon"], "P18":["lat", "lon"], "S18":["lat", "lon"]}

# set kernel grid "transform" --> flip the given axis
k_flip = {"Sh08":"NA", "So08":"NA", "BM13":"lat", "H17":"lat", "P18":"NA", "S18":"NA"}

# output name addend in case there is a special kernel property; as of now (26.06.2020) the only addend is the kernel
# state in case of the Block and Mauritsen (2013) kernels
out_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}

# add a directory for all states except the control state; this only affects the BM13 kernels because the other
# kernels exist only for the control state
out_path_ad_bm13 = {"pi":"", "2x":"/State_2xCO2/", "4x":"/State_4xCO2/", "8x":"/State_8xCO2/"}
out_path_ad_rest = {"pi":"", "2x":"", "4x":"", "8x":""}
out_path_ad = {"Sh08":out_path_ad_rest, "So08":out_path_ad_rest, "BM13":out_path_ad_bm13, "H17":out_path_ad_rest, 
               "P18":out_path_ad_rest, "S18":out_path_ad_rest}


#%% print the kernel name
print(k_t[kl] + " kernels...")


#%% set the regridding method - either "scipy" or "geocat"
regrid_meth = k_rg[kl]


#%% set paths

# model data path
data_path = f"/Data/{case}/"

# kernel path
kern_path = ("/Kernels/" + k_p[kl] + "_Kernels/" + k_ad[kl])

# output path
out_path = (data_path + f"/Rad_Kernel/Kernel_{toa_sfc}_RadResp/{k_p[kl]}" + "/01_Diff_to_CS/" + 
            out_path_ad[kl][k_st])

os.makedirs(out_path, exist_ok=True)

# plot path
# pl_path = (direc + "/Uni/PhD/Tromsoe_UiT/Work/" + cmip + "/Plots/" + nl.models[mod] + "/Kernel/SfcAlbedo/")
# os.makedirs(pl_path, exist_ok=True)


#%% load the nc files
alke_nc = Dataset(kern_path + k_as_fn[kl])
alke_cs_nc = Dataset(kern_path + k_cs_fn[kl])

dsfcalb_nc = Dataset(glob.glob(data_path + "dSfcAlb_Mon_*")[0])


#%% get the values

# kernel latitudes (note that the "NA" part seems like rather bad programming...)
grid_d = {"lat":alke_nc.variables[k_gvn[kl][0]][:], "lon":alke_nc.variables[k_gvn[kl][1]][:], "NA":np.arange(1)}

# flip the necessary part of the grid info
grid_d[k_flip[kl]] = flip(grid_d[k_flip[kl]])

lat_ke = grid_d["lat"]
lon_ke = grid_d["lon"]

# model latitudes
lat2d = dsfcalb_nc.variables["lat"][:]
lon2d = dsfcalb_nc.variables["lon"][:]
lat3d = dsfcalb_nc.variables["lat"][:]
lon3d = dsfcalb_nc.variables["lon"][:]

# test if 2d and 3d lats are of same length
if len(lat2d) == len(lat3d):
    lat = lat2d
    lon = lon2d
else:
    lat = lat3d
    lon = lon3d
# end if else   

# NOTE THE POSSIBLE FLIPPING ALONG THE LATITUDE -> subtract all-sky from clear-sky
alke = (flip(np.ma.masked_invalid(alke_cs_nc.variables[k_cs_vn[kl]][:]), axis=k_flip[kl]) - 
        flip(np.ma.masked_invalid(alke_nc.variables[k_as_vn[kl]][:]), axis=k_flip[kl]))

dsfcalb = da.from_array(dsfcalb_nc.variables["dSfcAlb_clim"][:]) * 100


#%% total number years
n_yr = np.shape(dsfcalb)[0]  


#%% regrid the kernels to the model grid if necessary
if ~np.array_equal(lat_ke, lat) | ~np.array_equal(lon_ke, lon):
    
    # first transform the lats and lons to increase monotonically
    lon_tr = lon
    lat_tr = lat + 90
    lon_o_tr = lon_ke
    lat_o_tr = lat_ke + 90
     
    if regrid_meth == "scipy":
        # regrid the field
        rem_alke = np.zeros((12, len(lat), len(lon)))

        print("\nRegridding kernels to CESM2-SOM grid via scipy...\n")
        for t in range(12):
            
            print(str(t+1) + " of 12")
            
            rem_alke[t, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, alke[t, :, :])
        # end for z
    elif regrid_meth == "geocat":
        print("\nRegridding kernels to CESM2-SOM grid via geocat...\n")
        rem_alke = geoc.linint2(alke, xo=lon_tr, yo=lat_tr, icycx=True, xi=lon_o_tr, yi=lat_o_tr)
    # end if elif
    
# end if


#%% calculate the TOA radiation change due to the surface albedo change
alb_adj = rem_alke[None, :, :, :] * dsfcalb
alb_adj.compute()


#%% calculate and print the global mean of the mean over the last 30 years

# calculate the last 30 years mean
toa_ch_30y = np.nanmean(alb_adj[-30:, 0, :, :], axis=0)

print("\n" + str(np.sum(np.isnan(toa_ch_30y))) + " NaN values in the last 30-year mean\n")
toa_ch_30y[np.isnan(toa_ch_30y)] = 0
# print(np.sum(np.isnan(toa_ch_30y)))

# generate the weights for the global average
weights2d = np.zeros((len(lat), len(lon)))
weights2d[:, :] = np.cos(lat[:, None] / 180 * np.pi)

toa_ch_30y_gl = np.average(toa_ch_30y, weights=weights2d, axis=(-2, -1))

print("global mean surface albedo response last 30-year mean " + k_t[kl] + ": " + str(toa_ch_30y_gl) + " W/m²")


#%% plot the last 30-year mean response

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

# colorbar ticks
c_ticks = np.arange(0, 106, 15)

# surface albedo kernel
fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(10, 4.5), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, toa_ch_30y, transform=ccrs.PlateCarree(), cmap=cm.Reds, 
                  extend="both", levels=c_ticks)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1 = pl.colorbar(p1, ax=ax1, ticks=c_ticks)
cb1.set_label("kernel in Wm$^{-2}$")
ax1.set_title("Surface Albedo Adjustment CESM2-SOM " + k_t[kl] + "\n" + "global mean " + 
              str(toa_ch_30y_gl) + " Wm$^{-2}$")

pl.show()
pl.close()


#%% store the result in a netcdf file

print("\nGenerating nc file...\n")
out_name = f"Diff_to_CS{sfc_add}_sa_" + case + "_" + kl + out_ad[kl] + "_CESM2-SOM.nc"
f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("year", n_yr)
f.createDimension("month", 12)
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
lon_nc = f.createVariable("lon", "f8", "lon")
year_nc = f.createVariable("year", "f4", "year")
month_nc = f.createVariable("month", "f4", "month")
a_ad_nc = f.createVariable("sa_adj", "f4", ("year", "month", "lat", "lon"))

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
year_nc[:] = np.arange(n_yr)
month_nc[:] = np.arange(12)

a_ad_nc[:] = alb_adj

a_ad_nc.units = "W m^-2"

a_ad_nc.description = ("difference between clear and total sky radiative response change due to surface albedo " + 
                       f"change at {toa_sfc}, i.e. surface albedo adjustment")

# add attributes
f.description = (f"This file contains the {toa_sfc} radiative response difference " + 
                 "between clear- and all-sky due to surface albedo change in the abrupt4xCO2 experiment for " +
                 "CESM2-SOM. This corresponds to the surface albedo adjustment for the cloud radiative\n" + 
                 "response as calculated via the Soden et al. (2008) adjustment method.\n" + 
                 "The values correspond to monthly means. Kernel reference: " + k_t[kl] +
                 "\nAnomalies are with respect to a control-run climatology over 28 years with CESM2-SOM.")

f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


#%% check if the file has been produced
try:
    test = Dataset(out_path + out_name)
    print("\n\nFile successfully produced. Directory:")
    print(out_path)
    print("\nName: " + out_name + "\n")
except:
    print("Error. File could not be stored.\n")
# end try except


