"""
Calculate and store LOCAL individual feedbacks for arbitrary kernels. That is local TOA/SFC flux on LOCAL and ZONAL mean 
tas/ts.

Be sure to adjust the data_path and the targ_path.
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
# from sklearn.linear_model import LinearRegression
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
from Functions.Func_Regrid_Data import remap as remap2
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_MonteCarlo_SignificanceTest import monte_carlo
from Classes.Class_MidpointNormalize import MidpointNormalize
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap


#%% which "CO2 forcing adjustment" should be used? Either "kernel" or "Gregory"
co2_adjs = "Gregory"  # for now leave this at Gregory since the kernel method is a little unclear at this point

# -> note that it acutally doesn't make so much sense to distinguish between these both here since we're in this script 
#    only concerned with feedbacks and hence the slopes of the regressions which won't be influenced by any constant
#    (in time) factor added to the data


#%% temperature variable
t_var = "tas"


#%% running mean piControl
run = 21


#%% number of years from the forced experiment
n_yr = 150


#%% Monte Carlo test permutation number
perm = 5000


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 8
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except


#%% kernel
try:
    kl = sys.argv[3]
except:
    kl = "Sh08"
# end try except
print("\nCalculating feedbacks for " + kl + " kernels")


#%% experiment (factor of abrupt CO2 increase)
exp = "4x"


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    a4x = f"abrupt{exp}CO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    a4x = f"abrupt-{exp}CO2"
# end if elif


#%% set up some initial dictionaries and lists

# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]
k_k = [kl]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}

# linestyles per kernel
l_sty = {"Sh08":":", "So08":"-.", "BM13":":", "H17":None, "P18":"--", "S18":"-."}

# marker style per kernel
m_sty = {"Sh08":"+", "So08":"s", "BM13":"^", "H17":"o", "P18":"x", "S18":"v"}

# kernel colour
k_c = {"Sh08":"green", "So08":"blue", "BM13":"magenta", "H17":"orange", "P18":"black", "S18":"gray"}

# response colours in the plots
re_c = {"T":"red", "Q":"green", "S":"orange", "C":"blue"}
re_more_c = {"Ta":"red", "Ts":"orange", "LR":"purple", "Pl":"violet", "Q_lw":"green", "Q_sw":"lightgreen", "C_lw":"blue", 
             "C_sw":"lightblue"}

# adjustment short and long names
res_s = ["T", "Q", "S", "C"]
res_l = {"T":"temperature", "Q":"water vapour", "S":"surface albedo", "C":"cloud"}
res_more_s = ["Ta", "Ts", "LR", "Pl", "Q_lw", "Q_sw", "C_lw", "C_sw"]
res_more_l = {"Ta":"air temperature", "Ts":"surface temperature", "LR":"lapse rate", "Pl":"Planck", 
              "Q_lw":"LW water vapour", "Q_sw":"SW water vapour", "C_lw":"LW cloud", "C_sw":"SW cloud"}

# sign of long wave kernels
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

# possibilites for CO2 adjustment: only BM13 and Sh08 have the CO2 "kernels"
co2_adjd = {"Sh08":co2_adjs, "So08":"Gregory", "BM13":co2_adjs, "H17":"Gregory", "P18":"Gregory", "S18":"Gregory"}


#%% set paths
data_path = cmip + "/Outputs/"

# set up dictionaries for the paths to the different directories for the different kernels
re_paths = dict()  # adjustment paths
re_paths[kl] = (data_path + f"Kernel_TOA_RadResp/{a4x}/{k_p[kl]}/")
out_path = data_path + f"/Feedbacks_Local/Kernel/{a4x}/{k_p[kl]}/Local_{t_var}_Based/"

os.makedirs(out_path, exist_ok=True)


#%% get the lat and lon for the target grid
targ_path = "/CMIP5/Data/CanESM2/"
lat_rg = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lat"][:]
lon_rg = Dataset(glob.glob(targ_path + "dtas*CanESM2*.nc")[0]).variables["lon"][:]


#%% load the netcdf data
t_nc = dict()
q_nc = dict()
s_nc = dict()
c_nc = dict()

for kl in k_k:
    t_nc[kl] = Dataset(re_paths[kl] + "T_Response/TOA_RadResponse_ta_ts_lr_" + kl + "_Kernel_" + nl.models[mod] + 
                       ".nc")
    q_nc[kl] = Dataset(re_paths[kl] + "Q_Response/TOA_RadResponse_q_" + kl + "_Kernel_" + nl.models[mod] + ".nc")
    s_nc[kl] = Dataset(re_paths[kl] + "SfcAlb_Response/TOA_RadResponse_SfcAlb_" + kl + "_Kernel_" + nl.models[mod] + 
                       ".nc")
    c_nc[kl] = Dataset(re_paths[kl] + "Cloud_Response/TOA_RadResponse_Cloud_" + kl + "_Kernel_" + nl.models[mod] + ".nc")
# end for kl    

# tas_nc = Dataset(data_path + f"GlobMean_{t_var}_piC_Run/{a4x}/GlobalMean_a4xCO2_{t_var}_piC21run_{nl.models[mod]}.nc")
toa_nc = Dataset(data_path + f"TOA_Imbalance_piC_Run/{a4x}/{t_var}_Based/TOA_Imbalance_" + 
                 f"GlobAnMean_and_{t_var}_Based_TF_piC21Run_a4x_{nl.models[mod]}.nc")

tas_nc = Dataset(glob.glob(direc_data + f"/Uni/PhD/Tromsoe_UiT/Work/{cmip}/Data/{nl.models[mod]}/dtas_*")[0])


#%% get the values
lat = t_nc[k_k[0]].variables["lat"][:]
lon = t_nc[k_k[0]].variables["lon"][:]

tas = np.mean(tas_nc.variables[t_var + "_ch"][:], axis=0)  # calculate annual mean
tas_zm = np.mean(tas, axis=-1)  # calculate annual mean
toa = toa_nc.variables["toa_imb_as"][:]
f4x_as = toa_nc.variables["forcing_as_e"][:][0]
f4x_as_l = toa_nc.variables["forcing_as_l"][:][0]
f4x_cs = toa_nc.variables["forcing_cs_e"][:][0]
sl_toa_e = toa_nc.variables["fb_as_e"][:][0]
sl_toa_l = toa_nc.variables["fb_as_l"][:][0]
p_toa_e = toa_nc.variables["p_as_e"][:][0]
p_toa_l = toa_nc.variables["p_as_l"][:][0]  
    

#%% feedbacks - treat the individual responses individually
res = dict()
res["LR"] = sig[kl] * an_mean(t_nc[kl].variables["lr_resp"][:])
res["Pl"] = sig[kl] * an_mean(t_nc[kl].variables["t_resp"][:]) - res["LR"]
res["C_lw"] = an_mean(c_nc[kl].variables["c_lw_resp"][:])
res["C_sw"] = an_mean(c_nc[kl].variables["c_sw_resp"][:])
res["C"] = an_mean(c_nc[kl].variables["c_resp"][:])
res["Q_lw"] = an_mean(sig[kl] * q_nc[kl].variables["q_lw_resp"][:])
res["Q_sw"] = an_mean(q_nc[kl].variables["q_sw_resp"][:])
res["Q"] = res["Q_lw"] + res["Q_sw"]
res["S"] = an_mean(s_nc[kl].variables["sa_resp"][:])
res["LR+Q"] = res["LR"] + res["Q"]
res["Total"] = res["S"] + res["Q"] + res["C"] + res["LR"] + res["Pl"]

# regress the local values on the global mean tas change - once for early (1-20) and once for late (21-150) period
fbs = dict()
for fb in res.keys():
    
    fbs[fb] = dict()
    
    fbs[fb]["fb_e"] = np.zeros((4, len(lat), len(lon)))  # dimension 0 has length 2 times 2 for slope and p-value for 
    fbs[fb]["fb_l"] = np.zeros((4, len(lat), len(lon)))  # the regression on local SAT and zonal mean SAT
    fbs[fb]["fb_t"] = np.zeros((4, len(lat), len(lon)))

    
    print("\nPerforming the linear regressions for " + fb + "...")
    for la in np.arange(len(lat)):
        for lo in np.arange(len(lon)):
            # local SAT
            fbs[fb]["fb_e"][0, la, lo], yi, r, fbs[fb]["fb_e"][1, la, lo] = lr(tas[:20, la, lo], 
                                                                               res[fb][:20, la, lo])[:4]
            fbs[fb]["fb_l"][0, la, lo], yi, r, fbs[fb]["fb_l"][1, la, lo] = lr(tas[20:, la, lo], 
                                                                               res[fb][20:, la, lo])[:4]
            fbs[fb]["fb_t"][0, la, lo], yi, r, fbs[fb]["fb_t"][1, la, lo] = lr(tas[:, la, lo], res[fb][:, la, lo])[:4]
            
            # zonal mean SAT
            fbs[fb]["fb_e"][2, la, lo], yi, r, fbs[fb]["fb_e"][3, la, lo] = lr(tas_zm[:20, la], 
                                                                               res[fb][:20, la, lo])[:4]
            fbs[fb]["fb_l"][2, la, lo], yi, r, fbs[fb]["fb_l"][3, la, lo] = lr(tas_zm[20:, la], 
                                                                               res[fb][20:, la, lo])[:4]
            fbs[fb]["fb_t"][2, la, lo], yi, r, fbs[fb]["fb_t"][3, la, lo] = lr(tas_zm[:, la], res[fb][:, la, lo])[:4]            
        # end for lo
    # end for la
    
    # calculate the change in lapse rate feedback from early to late period, ignoring the p-value for now
    fbs[fb]["dfb"] = np.zeros((2, len(lat), len(lon)))
    fbs[fb]["dfb"][0, :, :] = fbs[fb]["fb_l"][0, :, :] - fbs[fb]["fb_e"][0, :, :]  # local SAT regression
    fbs[fb]["dfb"][1, :, :] = fbs[fb]["fb_l"][2, :, :] - fbs[fb]["fb_e"][2, :, :]  # zonal mean SAT regression
    
# end for fb


#%% regrid the local feedbacks to a given model grid, ignoring the p-value for now
print("\nRegrid the feedbacks to the CanESM2 grid...\n")
fbs_rg = dict()
for fb in fbs.keys():
    
    fbs_rg[fb] = dict()
    
    fbs_rg[fb]["fb_e"] = np.zeros((2, len(lat_rg), len(lon_rg)))
    fbs_rg[fb]["fb_l"] = np.zeros((2, len(lat_rg), len(lon_rg)))
    fbs_rg[fb]["fb_t"] = np.zeros((2, len(lat_rg), len(lon_rg)))

    # regrid the feedbacks
    if ((mod != 21) & (cmip_v == 6)) | (cmip_v == 5):
        lat_o = lat + 90
        lat_t = lat_rg + 90
        
        # local SAT regression
        fbs_rg[fb]["fb_e"][0, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_e"][0, :, :], verbose=True)
        fbs_rg[fb]["fb_l"][0, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_l"][0, :, :])
        fbs_rg[fb]["fb_t"][0, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_t"][0, :, :])
        
        # zonal mean SAT regression
        fbs_rg[fb]["fb_e"][1, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_e"][2, :, :], verbose=True)
        fbs_rg[fb]["fb_l"][1, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_l"][2, :, :])
        fbs_rg[fb]["fb_t"][1, :, :] = remap(lon_rg, lat_t, lon, lat_o, fbs[fb]["fb_t"][2, :, :])
    else:
        print("\n\nSpecial case for FGOALS-g3\n\n")
        lat_o = lat
        lat_t = lat_rg
        lon_o = lon - 180
        lon_t = lon_rg - 180
        
        # local SAT regression
        fbs_rg[fb]["fb_e"][0, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_e"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)
        fbs_rg[fb]["fb_l"][0, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_l"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)
        fbs_rg[fb]["fb_t"][0, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_t"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)
        
        # zonal mean SAT regression
        fbs_rg[fb]["fb_e"][1, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_e"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)
        fbs_rg[fb]["fb_l"][1, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_l"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)
        fbs_rg[fb]["fb_t"][1, :, :] = remap2(lon_t, lat_t, lon_o, lat_o, fbs[fb]["fb_t"][0, :, :], rad_of_i=1500000, 
                                             epsilon=0.5, fill_value=None)         
    # end if else
        
    # calculate the change in lapse rate feedback from early to late period
    fbs_rg[fb]["dfb"] = np.zeros((2, len(lat_rg), len(lon_rg)))
    fbs_rg[fb]["dfb"][0, :, :] = fbs_rg[fb]["fb_l"][0, :, :] - fbs_rg[fb]["fb_e"][0, :, :]  # local SAT regression
    fbs_rg[fb]["dfb"][1, :, :] = fbs_rg[fb]["fb_l"][1, :, :] - fbs_rg[fb]["fb_e"][1, :, :]  # zonal mean SAT regression
    
# end for fb


#%% make test plots
"""
fb = "C"
fb_kind = "fb_e"

l_num = 14
levels, norm, cmp = set_levs_norm_colmap(fbs[fb][fb_kind][0, :, :], l_num, c_min=-5, c_max=5, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, fbs[fb][fb_kind][0, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label(fb_kind + " in Wm$^{-2}$K$^{-1}$")
ax1.set_title(fb + " " + fb_kind + " " + nl.models[mod])
pl.show()
pl.close()


l_num = 14
levels, norm, cmp = set_levs_norm_colmap(fbs_rg[fb][fb_kind][0, :, :], l_num, c_min=-5, c_max=5, quant=0.99)

x_rg, y_rg = np.meshgrid(lon_rg, lat_rg)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x_rg, y_rg, fbs_rg[fb][fb_kind][0, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label(fb_kind + " in Wm$^{-2}$K$^{-1}$")
ax1.set_title("Regridded " + fb + " " + fb_kind + " " + nl.models[mod])
pl.show()
pl.close()
"""
"""
fb_kind = "fb_l"

l_num = 14
levels, norm, cmp = set_levs_norm_colmap(fbs[fb][fb_kind][0, :, :], l_num, c_min=-20, c_max=10, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, fbs[fb][fb_kind][0, :, :], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label(fb_kind + " in Wm$^{-2}$K$^{-1}$")
ax1.set_title(fb + " " + fb_kind + " " + nl.models[mod])
pl.show()
pl.close()

fb_kind = "dfb"

l_num = 14
levels, norm, cmp = set_levs_norm_colmap(fbs[fb][fb_kind], l_num, c_min=-20, c_max=10, quant=0.99)

x, y = np.meshgrid(lon, lat)
proj = ccrs.PlateCarree(central_longitude=0)

fig, ax1 = pl.subplots(nrows=1, ncols=1, figsize=(12, 6), subplot_kw=dict(projection=proj))

p1 = ax1.contourf(x, y, fbs[fb][fb_kind], levels=levels, norm=norm, cmap=cmp, extend="both")
cb1 = fig.colorbar(p1, ax=ax1, ticks=levels)
ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
cb1.set_label(fb_kind + " in Wm$^{-2}$K$^{-1}$")
ax1.set_title(fb + " " + fb_kind + " " + nl.models[mod])
pl.show()
pl.close()
"""


#%% store the local feedbacks
print("\nStoring data...\n")
    
f = Dataset(out_path + f"Local_Feedbacks_Local_{t_var}_Based_{kl}Kernel_piC{run}Run_a{exp}_{nl.models[mod]}.nc", "w", 
            format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("2", 2)
f.createDimension("4", 4)
f.createDimension("lat", len(lat))
f.createDimension("lon", len(lon))
f.createDimension("lat_rg", len(lat_rg))
f.createDimension("lon_rg", len(lon_rg))

# create the variables
lat_nc = f.createVariable("lat", "f8", "lat")
lon_nc = f.createVariable("lon", "f8", "lon")
lat_rg_nc = f.createVariable("lat_rg", "f8", "lat_rg")
lon_rg_nc = f.createVariable("lon_rg", "f8", "lon_rg")

for fb in res.keys():
    fb_e_nc = f.createVariable(fb + "_fb_e", "f4", ("4", "lat", "lon"))
    fb_l_nc = f.createVariable(fb + "_fb_l", "f4", ("4", "lat", "lon"))
    fb_t_nc = f.createVariable(fb + "_fb_t", "f4", ("4", "lat", "lon"))
    dfb_nc = f.createVariable(fb + "_dfb", "f4", ("2", "lat", "lon"))
    fb_e_rg_nc = f.createVariable(fb + "_fb_e_rg", "f4", ("2", "lat_rg", "lon_rg"))
    fb_l_rg_nc = f.createVariable(fb + "_fb_l_rg", "f4", ("2", "lat_rg", "lon_rg"))
    fb_t_rg_nc = f.createVariable(fb + "_fb_t_rg", "f4", ("2", "lat_rg", "lon_rg"))
    dfb_rg_nc = f.createVariable(fb + "_dfb_rg", "f4", ("2", "lat_rg", "lon_rg"))    
    
    fb_e_nc[:] = fbs[fb]["fb_e"]
    fb_l_nc[:] = fbs[fb]["fb_l"]
    fb_t_nc[:] = fbs[fb]["fb_t"]
    dfb_nc[:] = fbs[fb]["dfb"]

    fb_e_rg_nc[:] = fbs_rg[fb]["fb_e"]
    fb_l_rg_nc[:] = fbs_rg[fb]["fb_l"]
    fb_t_rg_nc[:] = fbs_rg[fb]["fb_t"]
    dfb_rg_nc[:] = fbs_rg[fb]["dfb"]    
    
    fb_e_nc.units = "W m^-2 K^-1"
    fb_l_nc.units = "W m^-2 K^-1"
    fb_t_nc.units = "W m^-2 K^-1"
    dfb_nc.units = "W m^-2 K^-1"
    fb_e_rg_nc.units = "W m^-2 K^-1"
    fb_l_rg_nc.units = "W m^-2 K^-1"
    fb_t_rg_nc.units = "W m^-2 K^-1"
    dfb_rg_nc.units = "W m^-2 K^-1"
# end for fb

# pass the data into the variables
lat_nc[:] = lat
lon_nc[:] = lon
lat_rg_nc[:] = lat_rg
lon_rg_nc[:] = lon_rg

lat_nc.units = t_nc[k_k[0]].variables["lat"].units
lon_nc.units = t_nc[k_k[0]].variables["lon"].units
lat_rg_nc.units = t_nc[k_k[0]].variables["lat"].units
lon_rg_nc.units = t_nc[k_k[0]].variables["lon"].units

f.description = ("This file contains local annual mean radiative feedbacks (early, late, and total period as well as\n" + 
                 "the change from early to late) for " + nl.models[mod] +
                 ". The feedbacks have been calculated in two different ways: By (1) regressing local TOA radiative " +
                 f"flux on local (i.e., grid cell) {t_var} and by (2) regressing local TOA radiative flux on zonal " +
                 f"mean {t_var}.\n" +
                 f"The first dimension contains the local {t_var} based feedback values (slopes) in [0, :, :] as well " +
                 f"as the corresponding p-values in [1, :, :]. In [2, :, :] it contains the zonal mean {t_var} based " +
                 "feedbacks with the corresponding p-values in [3, :, :].\n" + 
                 f"Experiments: piControl and {a4x}")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()


#%% test
"""
f = Dataset(out_path + f"Local_Feedbacks_Local_{t_var}_Based_{kl}Kernel_piC{run}Run_a{exp}_{nl.models[mod]}.nc")
test = f.variables["LR_fb_l"][:]
pl.imshow(test[0, :, :], origin="lower")
pl.colorbar()
pl.show()
pl.close()

pl.imshow(test[2, :, :], origin="lower")
pl.colorbar()
pl.show()
pl.close()
"""