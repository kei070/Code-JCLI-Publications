"""
Calculate cloud radiative response via the Soden et al. (2008) kernel adjustment method.
Note that the cloud masking of the forcing can only be calculated for the Sh08 and BM13 kernels. For the other kernels,
the variable in the output file contains the value 0. Furthermore note, that the adjusted CRE as calculated here lacks
the CO2 forcing masking since this is unavailable for most kernels. But since this is unnecessary in the feedback 
calculations (since they are the -->slopes<-- of linear regressions on the surface temperature) we do not need to do this
for the purposes here. Remember however, that the -->radiative fluxes<-- themselves as calculated here should NOT be used
to represent the radiative flux change due to cloud since, again, they lack the CO2 forcing adjustment.

Be sure to adjust the paths in code block "set paths".
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
# from dask.distributed import Client


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
from Functions.Func_Flip import flip
from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Functions.Func_Regrid_Data_RectBiVSpline import remap


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


#%% set the temperature variable
t_var = "TREFHT"


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


#%% set the regridding method
regrid_meth = "scipy"


#%% set up dictionaries for the kernel choice

# kernels names for keys
k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]

# kernel names for plot names/titles, texts, paths/directories etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_t = {"Sh08":"Shell et al. (2008)", "So08":"Soden et al. (2008)", "BM13":"Block and Mauritsen (2013)", 
       "H17":"Huang et al. (2017)", "P18":"Pendergrass et al. (2018)", "S18":"Smith et al. (2018)"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", 
       "H17":"Huang_etal_2017", "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# output name addend in case there is a special kernel property; as of now (26.06.2020) the only addend is the kernel
# state in case of the Block and Mauritsen (2013) kernels
fn_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}
out_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}

# set kernel directory addend
k_ad = {"Sh08":"kernels/", "So08":"", "BM13":"", "H17":"kernel-highR/toa/", "P18":"cam5-kernels/forcing/", "S18":""}

# all- and clear-sky addend
k_as_ad = {"Sh08":"", "So08":"", "BM13":"", "H17":"cld", "P18":"", "S18":""}
k_cs_ad = {"Sh08":"_clr", "So08":"clr", "BM13":"", "H17":"clr", "P18":"C", "S18":""}
k_as_vad = {"Sh08":"", "So08":"", "BM13":"d", "H17":"cld", "P18":"", "S18":""}
k_cs_vad = {"Sh08":"C", "So08":"clr", "BM13":"f", "H17":"clr", "P18":"C", "S18":"_cs"}

# set kernel file names (maybe this is not the most economic way...)
k_as_fn = {"Sh08":"CAM3_CO2_lw" + k_as_ad[kl] + "_kernel.nc", 
           "So08":{"t":"TOA_GFDL_Kerns.nc", "ts":"TOA_GFDL_Kerns.nc"},
           "BM13":state_fn[k_st] + "_kernel_mm_1950.nc", 
           "H17":{"t":"RRTMG_t_toa_" + k_as_ad[kl] + "_highR.nc", "ts":"RRTMG_ts_toa_" + k_as_ad[kl] + "_highR.nc"}, 
           "P18":"ghg.forcing.nc", "S18":"HadGEM2_lw_TOA_L17.nc"}
k_cs_fn = {"Sh08":"CAM3_CO2_lw" + k_cs_ad[kl] + "_kernel.nc", 
           "So08":{"t":"TOA_GFDL_Kerns.nc", "ts":"TOA_GFDL_Kerns.nc"}, 
           "BM13":state_fn[k_st] + "_kernel_mm_1950.nc", 
           "H17":{"t":"RRTMG_t_toa_" + k_cs_ad[kl] + "_highR.nc", "ts":"RRTMG_ts_toa_" + k_cs_ad[kl] + "_highR.nc"}, 
           "P18":"ghg.forcing.nc", "S18":"HadGEM2_lw_TOA_L17.nc"}

# set kernel variable names (maybe this is not the most economic way...)
if toa_sfc == "TOA":
    k_as_vn = {"Sh08":"toaeffect", 
               "So08":{"t":"lw" + k_as_vad[kl] + "_t", "ts":"lw" + k_as_vad[kl] + "_ts"},
               "BM13":"CO2_tra" + k_as_vad[kl] + "0",
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNT" + k_as_vad[kl], "ts":"FLNT" + k_as_vad[kl]},
               "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}
    k_cs_vn = {"Sh08":"toaeffect", 
               "So08":{"t":"lw" + k_cs_vad[kl] + "_t", "ts":"lw" + k_cs_vad[kl] + "_ts"}, 
               "BM13":"CO2_tra" + k_cs_vad[kl] + "0",
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNT" + k_cs_vad[kl], "ts":"FLNT" + k_cs_vad[kl]},
               "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}
elif toa_sfc == "SFC":
    k_as_vn = {"Sh08":"toaeffect", 
               "So08":{"t":"lw" + k_as_vad[kl] + "_t", "ts":"lw" + k_as_vad[kl] + "_ts"},
               "BM13":"CO2_tra" + k_as_vad[kl] + "0",
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNS" + k_as_vad[kl], "ts":"FLNS" + k_as_vad[kl]},
               "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}
    k_cs_vn = {"Sh08":"toaeffect", 
               "So08":{"t":"lw" + k_cs_vad[kl] + "_t", "ts":"lw" + k_cs_vad[kl] + "_ts"}, 
               "BM13":"CO2_tra" + k_cs_vad[kl] + "0",
               "H17":{"t":"lwkernel", "ts":"lwkernel"}, 
               "P18":{"t":"FLNS" + k_cs_vad[kl], "ts":"FLNS" + k_cs_vad[kl]},
               "S18":{"t":"HadGEM2_lw_TOA_L17.nc", "ts":"HadGEM2_lw_TOA_L38.nc"}}
# end if elif

# set kernel grid variable names
k_gvn = {"Sh08":["lev", "lat", "lon"], 
         "So08":["plev", "latitude", "longitude"], 
         "BM13":["mlev", "lat", "lon"], 
         "H17":["player", "lat", "lon"], 
         "P18":["lev", "lat", "lon"],
         "S18":["plev", "lat", "lon"]}

# set the kernel surface pressure variable name (if existent)
k_ps_vn = {"Sh08":"", "So08":"PS", "BM13":"aps", "H17":"", "P18":"PS", "S18":""}

# set kernel grid "transform" --> flip the given axis
k_flip = {"Sh08":"lev", "So08":"NA", "BM13":"lat", "H17":"lat", "P18":"NA", "S18":"NA"}

# vertical interpolation method
k_vi = {"Sh08":"np", "So08":"np", "BM13":"np", "H17":"np", "P18":"np", "S18":"np"}

# the positive direction of the long-wave flux (positive "up" or "down")
k_pud = {"Sh08":"down", "So08":"up", "BM13":"down", "H17":"up", "P18":"down", "S18":"up"}

# output name addend in case there is a special kernel property; as of now (26.06.2020) the only addend is the kernel
# state in case of the Block and Mauritsen (2013) kernels
out_ad = {"Sh08":"_Kernel", "So08":"_Kernel", "BM13":state_ofn[k_st], "H17":"_Kernel", "P18":"_Kernel", "S18":"_Kernel"}

# introduce a sign dictionary (only for the test plots)
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

# add a directory for all states except the control state; this only affects the BM13 kernels because the other
# kernels exist only for the control state
out_path_ad_bm13 = {"pi":"", "2x":"/State_2xCO2/", "4x":"/State_4xCO2/", "8x":"/State_8xCO2/"}
out_path_ad_rest = {"pi":"", "2x":"", "4x":"", "8x":""}
out_path_ad = {"Sh08":out_path_ad_rest, "So08":out_path_ad_rest, "BM13":out_path_ad_bm13, "H17":out_path_ad_rest, 
               "P18":out_path_ad_rest, "S18":out_path_ad_rest}


#%% print the kernel name
print(k_t[kl] + " kernels...")


#%% file name suffix for at-surface
sfc_add = ""
if toa_sfc == "SFC":
    sfc_add = "_SFC"
# end if


#%% set paths

# model data path
data_path = f"/Data/{case}/"

# adjustment path
adj_path = (data_path + f"/Rad_Kernel/Kernel_{toa_sfc}_RadResp/{k_p[kl]}" + "/01_Diff_to_CS/" + 
            out_path_ad[kl][k_st])

# CRE path
cre_path = data_path

# forcing path
# f4x_path = (direc_data + f"/Uni/PhD/Tromsoe_UiT/Work/{cmip}/Outputs/TOA_Imbalance_piC_Run/{a4x}/{t_var}_Based/")

# kernel path
kern_path = "/Kernels/" + k_p[kl] + "_Kernels/" + k_ad[kl]

# output path
out_path = (data_path + f"/Rad_Kernel/Kernel_{toa_sfc}_RadResp/{k_p[kl]}" + "/Cloud_Response/" + 
            out_path_ad[kl][k_st])

os.makedirs(out_path, exist_ok=True)


#%% load nc files

# kernel calculate quantities
t_adj_nc = Dataset(adj_path + f"Diff_to_CS{sfc_add}_ta_ts_{case}_{kl}{fn_ad[kl]}_CESM2-SOM.nc")
q_adj_nc = Dataset(adj_path + f"Diff_to_CS{sfc_add}_q_{case}_{kl}{fn_ad[kl]}_CESM2-SOM.nc")
a_adj_nc = Dataset(adj_path + f"Diff_to_CS{sfc_add}_sa_{case}_{kl}{fn_ad[kl]}_CESM2-SOM.nc")

# cloud radiative effect
cre_nc = Dataset(glob.glob(cre_path + f"dCRE_Mon_{case}_*.nc")[0])

# forcing estimate
# f4x_nc = Dataset(glob.glob(f4x_path + f"TOA_Imbalance_GlobAnMean_*_{nl.models[mod]}.nc")[0])


#%% get the values - directly change the sign of the long-wave responses where necessary
t_re = sig[kl] * t_adj_nc.variables["t_adj"][:]
q_lw_re = sig[kl] * q_adj_nc.variables["q_lw_adj"][:]
q_sw_re = q_adj_nc.variables["q_sw_adj"][:]
a_re = a_adj_nc.variables["sa_adj"][:]
cre_sw = cre_nc.variables["dcre_sw"][:]
cre_lw = cre_nc.variables["dcre_lw"][:]


#%% total number years
n_yr = np.shape(a_re)[0]


#%% get the all-sky and clear-sky forcing estimates
# f4x_as = f4x_nc.variables["forcing_as"][:][0]
# f4x_cs = f4x_nc.variables["forcing_cs"][:][0]


#%% get the grid quantities
lat = t_adj_nc.variables["lat"][:]
lon = t_adj_nc.variables["lon"][:]


#%% handle CO2 forcing from kernels if applicable

if (kl == "BM13") | (kl == "Sh08"):
    
    print("\nCO2 'kernels' exist. Calculating cloud masking of CO2 forcing from them...\n")
    
    # add descriptions for the nc file
    co2_mask_desc = ". Note that in the total and the long-" +\
                 "wave part the cloud masking of the forcing is not considered yet. The cloud masking of the forcing " +\
                 "is included as its own dedicated variable: 'co2_masking'. (The reasoning behind this is that, on " +\
                 "the one hand not all kernels include the CO2 forcing and, on the other hand in the global mean the " +\
                 "cloud masking of the CO2 forcing can also be estimated via the Gregory method.)\n"
    co2_mask_vdesc = "cloud masking of CO2 forcing for 4xCO2 per month"
    
    # CO2 forcing from kernels
    co2ke_nc = Dataset(kern_path + k_as_fn[kl])
    co2ke_cs_nc = Dataset(kern_path + k_cs_fn[kl])
    
    co2ke = (flip(np.ma.masked_invalid(co2ke_cs_nc.variables[k_cs_vn[kl]][:]), axis=k_flip[kl]) - 
             flip(np.ma.masked_invalid(co2ke_nc.variables[k_as_vn[kl]][:]), axis=k_flip[kl])) * 2  # x2 for 4xCO2
    
    # kernel latitudes (note that the "NA" part seems like rather bad programming...)
    grid_d = {"lev":0,
              "lat":co2ke_nc.variables[k_gvn[kl][1]][:], 
              "lon":co2ke_nc.variables[k_gvn[kl][2]][:], 
              "NA":np.arange(1)}
    
    # flip the necessary part of the grid info
    grid_d[k_flip[kl]] = flip(grid_d[k_flip[kl]])
    
    lat_k = grid_d["lat"]
    lon_k = grid_d["lon"]
    
    
    #% interpolate the kernels and the surface pressure field to the horizontal model grid
    
    # first transform the lats and lons
    lon_tr = lon
    lat_tr = lat + 90
    lon_o_tr = lon_k
    lat_o_tr = lat_k + 90
    
    # regrid the field
    print("\n\n\nRegridding kernels to CESM2-SOM grid via " + regrid_meth + "...\n")
    if regrid_meth == "scipy":
    
        rem_co2ke = np.zeros((12, len(lat), len(lon)))
        
        for t in range(12):
            
            print_one_line(str(t+1) + "   ")
            
            rem_co2ke[t, :, :] = remap(lon_tr, lat_tr, lon_o_tr, lat_o_tr, co2ke[t, :, :])
        
        # end for t
    # end if
    print("")
else:

    print("\nCO2 'kernels' do not exist. String an empty variable\n")
    rem_co2ke = 0
    
    # add descriptions for the nc file
    co2_mask_desc = ". Note that since the " + kl + " kernels do not provide a CO2 'kernel', the cloud masking of CO2" +\
                    " forcing here is an empty variable and should be ignored. The BM13 and Sh08 kernels have" +\
                    " meaningful values for this variable. Note that global mean values for this quantity can be" + \
                    " estimated via the Gregory method.\n"
    co2_mask_vdesc = "empty variable for the present kernels"
    
# end if else


#%% take a peak into the individual parts
"""
t_an_gl = glob_mean(an_mean(t_re), lat, lon)
pl.plot(t_an_gl)
pl.title("Temperature Adjustment")
pl.show()
pl.close()

q_lw_an_gl = glob_mean(an_mean(q_lw_re), lat, lon)
pl.plot(q_lw_an_gl)
pl.title("Q LW Adjustment")
pl.show()
pl.close()

q_sw_an_gl = glob_mean(an_mean(q_sw_re), lat, lon)
pl.plot(q_sw_an_gl)
pl.title("Q SW Adjustment")
pl.show()
pl.close()

a_an_gl = glob_mean(an_mean(a_re), lat, lon)
pl.plot(a_an_gl)
pl.title("Surface Albedo Adjustment")
pl.show()
pl.close()

cre_lw_an_gl = glob_mean(an_mean(cre_lw), lat, lon)
pl.plot(cre_lw_an_gl)
pl.title("CRE LW")
pl.show()
pl.close()

cre_sw_an_gl = glob_mean(an_mean(cre_sw), lat, lon)
pl.plot(cre_sw_an_gl)
pl.title("CRE SW")
pl.show()
pl.close()


#%% a test map plot
if (kl == "BM13") | (kl == "Sh08"):
    l_num = 14
    levels, norm, cmp = set_levs_norm_colmap(sig[kl]*rem_co2ke[0, :, :], l_num, c_min=None, c_max=None, quant=0.99)
    
    x, y = np.meshgrid(lon, lat)
    proj = ccrs.PlateCarree(central_longitude=0)
    
    fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))
    
    p1 = ax1.contourf(x, y, sig[kl]*rem_co2ke[0, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
    cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
    ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
    cb1.set_label("adjustment in Wm$^{-2}$K$^{-1}$")
    ax1.set_title("CO$_2$ Forcing Adjustment First Time Step from " + kl + " Kernels " + nl.models_pl[mod] + 
                  "\nGlobal Mean: " + str(glob_mean(sig[kl]*rem_co2ke[0, :, :], lat, lon)) + " Wm$^{-2}$")
    pl.show()
    pl.close()
    
    print(glob_mean(sig[kl]*rem_co2ke[0, :, :], lat, lon))
# end if
    """

#%% calculate the full q response
# q_re = -q_lw_re + q_sw_re


#%% calculate the adjusted cloud radiative response
# c_re_sw = ne.evaluate("cre_sw + a_re + q_sw_re")
# c_re_lw = ne.evaluate("cre_lw + t_re + q_lw_re")
# c_re = ne.evaluate("c_re_lw + c_re_sw")  # + (f4x_cs - f4x_as)  --> adding the forcing-masking to every gridcell does 
#                                                                     not seem right...

# note that from my calculations the CRE is POSITIVE DOWN (i.e. where the CRE values are positive, more energy is going
# into the climate system than is leaving it due to the effect of clouds and vice versa; compare this e.g. to 
# corresponding plots of clwvi: where clwvi is DECREASING the SW CRE should give positive values while the LW CRE 
# should give negative values)
# further note that for the short-wave kernels the differences (here a_re and q_sw_re) are POSITIVE UP, hence they are
# ADDED to the CRE, while for the long-wave kernels the differences (here t_re and q_lw_re) are POSITIVE DOWN, hence
# they are subtracted from the CRE


#%% calculate the global means
# c_re_gl = glob_mean(c_re, lat, lon)


#%% calculate the annual means of the global means
# c_re_gl_an = an_mean(c_re_gl)


#%% plot the global annual means 
"""
pl.figure(figsize=(10, 6))
pl.plot(c_re_gl_an)
pl.plot([0, 150], [0, 0], linewidth=0.75, c="gray", linestyle="--")
pl.title("Cloud Radiative Response Global & Annual Mean " + models_pl[mod])
pl.xlabel("years since branching")
pl.ylabel("radiative response in Wm$^{-2}$")
pl.show()
pl.close()
"""


#%% store the result in a netcdf file

print("\nGenerating nc file...\n")
out_name = f"{toa_sfc}_RadResponse_Cloud_" + case + "_" + kl + out_ad[kl] + "_CESM2-SOM.nc"
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
c_re_nc = f.createVariable("c_resp", "f4", ("year", "month", "lat", "lon"))
c_sw_re_nc = f.createVariable("c_sw_resp", "f4", ("year", "month", "lat", "lon"))
c_lw_re_nc = f.createVariable("c_lw_resp", "f4", ("year", "month", "lat", "lon"))
co2_c_masking_nc = f.createVariable("co2_masking", "f4", ("month", "lat", "lon"))

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
year_nc[:] = np.arange(n_yr)
month_nc[:] = np.arange(12)
c_re_nc[:] = ne.evaluate("cre_sw + a_re + q_sw_re") + ne.evaluate("-cre_lw + t_re + q_lw_re")  # here cre_lw is pos.-up!
c_sw_re_nc[:] = ne.evaluate("cre_sw + a_re + q_sw_re")
c_lw_re_nc[:] = ne.evaluate("-cre_lw + t_re + q_lw_re")
co2_c_masking_nc[:] = rem_co2ke

c_re_nc.units = "W m^-2"
c_sw_re_nc.units = "W m^-2"
c_lw_re_nc.units = "W m^-2"
co2_c_masking_nc.units = "W m^-2"

lat_nc.units = t_adj_nc.variables["lat"].units
lon_nc.units = t_adj_nc.variables["lon"].units

c_re_nc.description = (f"adjusted cloud radiative {toa_sfc} response (WITHOUT cloud masking of the CO2 forcing!)")
c_sw_re_nc.description = (f"adjusted short-wave cloud radiative {toa_sfc} response")
c_lw_re_nc.description = (f"adjusted long-wave cloud radiative {toa_sfc} response (WITHOUT cloud masking of the " + 
                          "CO2 forcing!)")
co2_c_masking_nc.description = co2_mask_vdesc

f.description = (f"This file contains the adjusted cloud radiative {toa_sfc} response calculated following " + 
                 "Soden et al. (2008) in the abrupt4xCO2 experiment for CESM2-SOM " + co2_mask_desc +
                 f"The values correspond to monthly means.\nThe kernels are taken from {k_t[kl]}")
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


#%% plot the global mean time series
# pl.plot(glob_mean(np.mean(c_resp, axis=1), lat, lon))