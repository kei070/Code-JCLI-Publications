"""
Calculate and store individual feedbacks for arbitrary kernels.
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
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_MonteCarlo_SignificanceTest import monte_carlo
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% which "CO2 forcing adjustment" should be used? Either "kernel" or "Gregory"
co2_adjs = "Gregory"  # for now leave this at Gregory since the kernel method is a little unclear at this point

# -> note that it acutally doesn't make so much sense to distinguish between these both here since we're in this script 
#    only concerned with feedbacks and hence the slopes of the regressions which won't be influenced by any constant
#    (in time) factor added to the data


#%% TOA or SFC
toa_sfc = "TOA"


#%% all-sky or clear-sky
try:
    cs = eval(sys.argv[4])
except:
    cs = True
# end try except


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
    mod = 6
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


#%% load the namelist
if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    a4x = "abrupt4xCO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    a4x = "abrupt-4xCO2"
# end if elif


#%% experiment (factor of abrupt CO2 increase)
exp = "4x"


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

res_all = ["T", "Q", "S", "C", "Ta", "Ts", "LR", "Pl", "Q_lw", "Q_sw", "C_lw", "C_sw"]

# sign of long wave kernels
sig = {"Sh08":-1, "So08":1, "BM13":1, "H17":1, "P18":-1, "S18":1}

# possibilites for CO2 adjustment: only BM13 and Sh08 have the CO2 "kernels"
co2_adjd = {"Sh08":co2_adjs, "So08":"Gregory", "BM13":co2_adjs, "H17":"Gregory", "P18":"Gregory", "S18":"Gregory"}


#%% set paths
data_path = ""

# set up dictionaries for the paths to the different directories for the different kernels
re_paths = dict()  # adjustment paths
pl_paths = dict()  # plot paths
png_paths = dict()  # plot paths
pl_paths_fb = dict()  # plot paths feedback decomposition
png_paths_fb = dict()  # plot paths feedback decomposition
out_paths_fb = dict()  # output paths feedback decomposition
list_paths = dict()  # store values for tables

sfc_path = ""
if toa_sfc == "SFC":
    sfc_path = "SFC_Feedback"
# end if    

cs_path = ""
ascs = "as"
ascs_n = ""
ascs_desc = "all-sky"
if cs:
    cs_path = "CS/"
    ascs = "cs"
    ascs_n = "CS_"
    ascs_desc = "clear-sky"
# end if

for kl in k_k:
    
    re_paths[kl] = (data_path + f"Kernel_{toa_sfc}_RadResp/{a4x}/" + k_p[kl] + "/")
    
    out_paths_fb[kl] = f"/Feedbacks/Kernel/{a4x}/{k_p[kl]}/{t_var}_Based/{sfc_path}/{cs_path}"
        
    # create the list path directory
    os.makedirs(list_paths[kl], exist_ok=True)
    
# end for kl


#%% load the netcdf data
t_nc = dict()
q_nc = dict()
s_nc = dict()
if not cs:
    c_nc = dict()
# end if  

for kl in k_k:
    t_nc[kl] = Dataset(re_paths[kl] + 
                       f"T_Response/{cs_path}{toa_sfc}_RadResponse_ta_ts_lr_{kl}_Kernel_{ascs_n}{nl.models[mod]}.nc")
    q_nc[kl] = Dataset(re_paths[kl] + 
                       f"Q_Response/{cs_path}{toa_sfc}_RadResponse_q_{kl}_Kernel_{ascs_n}{nl.models[mod]}.nc")
    s_nc[kl] = Dataset(re_paths[kl] + 
                       f"SfcAlb_Response/{cs_path}{toa_sfc}_RadResponse_SfcAlb_{kl}_Kernel_{ascs_n}{nl.models[mod]}.nc")
    if not cs:
        c_nc[kl] = Dataset(re_paths[kl] + f"Cloud_Response/{toa_sfc}_RadResponse_Cloud_{kl}_Kernel_{nl.models[mod]}.nc")
    # end if   
# end for kl    

tas_nc = Dataset(data_path + f"GlobMean_{t_var}_piC_Run/{a4x}/GlobalMean_a4xCO2_{t_var}_piC21run_{nl.models[mod]}.nc")
toa_nc = Dataset(data_path + f"TOA_Imbalance_piC_Run/{a4x}/{t_var}_Based/TOA_Imbalance_GlobAnMean_and_{t_var}_Based_" + 
                 f"TF_piC21Run_a4x_{nl.models[mod]}.nc")


#%% get the values
lat = t_nc[k_k[0]].variables["lat"][:]
lon = t_nc[k_k[0]].variables["lon"][:]

tas = tas_nc.variables[t_var + "_ch"][:]
toa = toa_nc.variables["toa_imb_" + ascs][:]
f4x_as = toa_nc.variables[f"forcing_{ascs}_e"][:][0]
f4x_as_l = toa_nc.variables[f"forcing_{ascs}_l"][:][0]
f4x_cs = toa_nc.variables["forcing_cs_e"][:][0]
sl_toa_e = toa_nc.variables[f"fb_{ascs}_e"][:][0]
sl_toa_l = toa_nc.variables[f"fb_{ascs}_l"][:][0]
p_toa_e = toa_nc.variables[f"p_{ascs}_e"][:][0]
p_toa_l = toa_nc.variables[f"p_{ascs}_l"][:][0]  


#%% get the regridded forcing values
fco2_as = dict()
if co2_adjd[kl] == "kernel":
    fco2_nc = Dataset(re_paths[kl] + "/CO2_Forcing/TOA_CO2_Forcing_" + kl + "_Kernel_" + nl.models[mod] + ".nc")
    fco2_as[kl] = glob_mean(np.mean(fco2_nc.variables[f"co2_{ascs}_f"][:], axis=0), lat, lon) * 2  # x2 for 4xCO2
else:
    fco2_as[kl] = f4x_as
# end if else    
    

#%% feedbacks
res = dict()
res_hem = dict()
res_more = dict()
res_hem_more = dict()
co2_adj = dict()

for kl in k_k:
    
    """    
    sig = 1
    # for the Pendergrass kernels set sig to -1 because their LW kernels are positive-up
    if (kl == "P18") | (kl == "Sh08"):
        sig = -1
    # end if
    """
    if co2_adjd[kl] == "Gregory":
        co2_adj[kl] = f4x_cs - f4x_as
    elif co2_adjd[kl] == "kernel":
        co2_adj[kl] = glob_mean(np.mean(c_nc[kl].variables["co2_masking"][:], axis=0), lat, lon)
    # end if elif      
    
    # global means
    res[kl] = dict()
    res[kl] = dict()

    res[kl]["T"] = sig[kl] * glob_mean(an_mean(t_nc[kl].variables["t_resp"][:]), lat, lon)
    res[kl]["Ts"] = sig[kl] * glob_mean(an_mean(t_nc[kl].variables["ts_resp"][:]), lat, lon)
    res[kl]["LR"] = sig[kl] * glob_mean(an_mean(t_nc[kl].variables["lr_resp"][:]), lat, lon)
    res[kl]["Ta"] = res[kl]["T"] - res[kl]["Ts"]
    res[kl]["Q_lw"] = glob_mean(an_mean(sig[kl] * q_nc[kl].variables["q_lw_resp"][:]), lat, lon)
    res[kl]["Q_sw"] = glob_mean(an_mean(q_nc[kl].variables["q_sw_resp"][:]), lat, lon)
    res[kl]["Q"] = res[kl]["Q_lw"] + res[kl]["Q_sw"]
    res[kl]["S"] = glob_mean(an_mean(s_nc[kl].variables["sa_resp"][:]), lat, lon)
    res[kl]["LR+Q"] = res[kl]["LR"] + res[kl]["Q"]
    # -> include the "CO2 adjustment" (f4x_cs - f4x_as) for the cloud radiative response
    if co2_adjd[kl] == "kernel":
        res[kl]["C_ke"] = glob_mean(an_mean(c_nc[kl].variables["c_resp"][:]), lat, lon) + co2_adj[kl]
    # end if
       
    if not cs:
        res[kl]["C"] = glob_mean(an_mean(c_nc[kl].variables["c_resp"][:]), lat, lon) + (f4x_cs - f4x_as)
        res[kl]["C_lw"] = glob_mean(an_mean(c_nc[kl].variables["c_lw_resp"][:]), lat, lon) + (f4x_cs - f4x_as)
        res[kl]["C_sw"] = glob_mean(an_mean(c_nc[kl].variables["c_sw_resp"][:]), lat, lon)
        res[kl]["Total"] = res[kl]["S"] + res[kl]["Q"] + res[kl]["C"] + res[kl]["T"]

        # include the total response (add f4x_as as the initial forcing)
        if co2_adjd[kl] == "kernel":
            res[kl]["Total"] = res[kl]["T"] + res[kl]["Q"] + res[kl]["S"] + res[kl]["C"] + fco2_as[kl]
        else:
            res[kl]["Total"] = res[kl]["T"] + res[kl]["Q"] + res[kl]["S"] + res[kl]["C"] + f4x_as
        # end if        
    else:
        res[kl]["Total"] = res[kl]["S"] + res[kl]["Q"] + res[kl]["T"] + f4x_cs
    # end 

    # hemispheric means
    res_hem[kl] = dict()
    res_hem[kl] = dict()
    res_hem[kl]["NH"] = dict()
    res_hem[kl]["SH"] = dict()
    res_hem[kl]["NH"] = dict()
    res_hem[kl]["SH"] = dict()
    
    res_hem[kl]["NH"]["T"], res_hem[kl]["SH"]["T"], r = \
                                       lat_mean(sig[kl] * an_mean(t_nc[kl].variables["t_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["Ts"], res_hem[kl]["SH"]["Ts"], r = \
                                      lat_mean(sig[kl] * an_mean(t_nc[kl].variables["ts_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["LR"], res_hem[kl]["SH"]["LR"], r = \
                                      lat_mean(sig[kl] * an_mean(t_nc[kl].variables["lr_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["Ta"] = res_hem[kl]["NH"]["T"] - res_hem[kl]["NH"]["Ts"]
    res_hem[kl]["SH"]["Ta"] = res_hem[kl]["SH"]["T"] - res_hem[kl]["SH"]["Ts"]
    res_hem[kl]["NH"]["Q_lw"], res_hem[kl]["SH"]["Q_lw"], r = \
                                    lat_mean(an_mean(sig[kl] * q_nc[kl].variables["q_lw_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["Q_sw"], res_hem[kl]["SH"]["Q_sw"], r = \
                                              lat_mean(an_mean(q_nc[kl].variables["q_sw_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["Q"] = res_hem[kl]["NH"]["Q_lw"] + res_hem[kl]["NH"]["Q_sw"]
    res_hem[kl]["SH"]["Q"] = res_hem[kl]["SH"]["Q_lw"] + res_hem[kl]["SH"]["Q_sw"]
    res_hem[kl]["NH"]["S"], res_hem[kl]["SH"]["S"], r = \
                                                lat_mean(an_mean(s_nc[kl].variables["sa_resp"][:]), lat, lon, 0).values()
    res_hem[kl]["NH"]["LR+Q"] = res_hem[kl]["NH"]["LR"] + res_hem[kl]["NH"]["Q"]
    res_hem[kl]["SH"]["LR+Q"] = res_hem[kl]["SH"]["LR"] + res_hem[kl]["SH"]["Q"]

    if not cs:
        res_hem[kl]["NH"]["C"], res_hem[kl]["SH"]["C"], r = \
                                               lat_mean(an_mean(c_nc[kl].variables["c_resp"][:]), lat, lon, 0).values()
        
        res_hem[kl]["NH"]["C_lw"], res_hem[kl]["SH"]["C_lw"], r = \
                                            lat_mean(an_mean(c_nc[kl].variables["c_lw_resp"][:]), lat, lon, 0).values()
        res_hem[kl]["NH"]["C_sw"], res_hem[kl]["SH"]["C_sw"], r = \
                                            lat_mean(an_mean(c_nc[kl].variables["c_sw_resp"][:]), lat, lon, 0).values()

        res_hem[kl]["NH"]["Total"] = res_hem[kl]["NH"]["S"] + res_hem[kl]["NH"]["Q"] + res_hem[kl]["NH"]["C"] + \
                                                                                                res_hem[kl]["NH"]["T"]
        res_hem[kl]["SH"]["Total"] = res_hem[kl]["SH"]["S"] + res_hem[kl]["SH"]["Q"] + res_hem[kl]["SH"]["C"] + \
                                                                                                res_hem[kl]["SH"]["T"]
    else:
        res_hem[kl]["NH"]["Total"] = res_hem[kl]["NH"]["S"] + res_hem[kl]["NH"]["Q"] + res_hem[kl]["NH"]["T"]
        res_hem[kl]["SH"]["Total"] = res_hem[kl]["SH"]["S"] + res_hem[kl]["SH"]["Q"] + res_hem[kl]["SH"]["T"]
    # end if else
                                                 
    # note that since we are only interested in the feedback (=regression slope) there is no need to add the cloud 
    # masking of the CO2 forcing (which is just a constant value)
 
# end for kl

# add the Planck response (-> T_resp = LR_resp + Planck_resp)
res[kl]["Pl"] = res[kl]["T"] - res[kl]["LR"]
res_hem[kl]["NH"]["Pl"] = res_hem[kl]["NH"]["T"] - res_hem[kl]["NH"]["LR"]
res_hem[kl]["SH"]["Pl"] = res_hem[kl]["SH"]["T"] - res_hem[kl]["SH"]["LR"]

print("\nData loaded. Proceeding with feedback calculations...\n")


#%% linear regressions for the responses (use nested dictionaries)
#   nomenclature for better understanding: k iterates over the kernels (Huang, Pendergrass) and re iterates of the
#   responses (temperature, water vapour, surface albedo, cloud)

lr_r = dict()
mc_st = dict()

for kl in k_k:
    
    lr_r[kl] = dict()
    lr_res = dict()
    
    mc_st[kl] = dict()
    mc_st_r = dict()

    for re in res[kl].keys():
        
        print(f"\nGlobal {re} {toa_sfc} feedback ({kl} kernels)...\n")
        
        lr_res[re] = dict()
        
        # set dictionary names        
        lr_res[re]["e"] = dict()
        lr_res[re]["l"] = dict()
        lr_res[re]["t"] = dict()
        
        # early period
        lr_res[re]["e"]["s"], lr_res[re]["e"]["i"], lr_res[re]["e"]["r"], lr_res[re]["e"]["p"] = \
                                                                                       lr(tas[:20], res[kl][re][:20])[:4]
        # late period
        lr_res[re]["l"]["s"], lr_res[re]["l"]["i"], lr_res[re]["l"]["r"], lr_res[re]["l"]["p"] = \
                                                                                       lr(tas[20:], res[kl][re][20:])[:4]
        # full period
        lr_res[re]["t"]["s"], lr_res[re]["t"]["i"], lr_res[re]["t"]["r"], lr_res[re]["t"]["p"] = \
                                                                                       lr(tas[:], res[kl][re][:])[:4]
                                                                                       
        # use the Monte Carlo significance test to calculate the significance of the change of slope between early and
        # late feedback
        mc_st_r[re] = monte_carlo(tas, res[kl][re], perm=perm)
    # end for i
    
    lr_r[kl] = lr_res
    mc_st[kl] = mc_st_r

# end for re

print("\n" + nl.models[mod] + " Monte-Carlo Test 'p-values'\n")
print(mc_st)
print("")


#%% linear regressions for the hemispheric mean responses (use nested dictionaries)
#   nomenclature for better understanding: k iterates over the kernels (Huang, Pendergrass) and re iterates of the
#   responses (temperature, water vapour, surface albedo, cloud)

lr_h_r = dict()

for h in ["NH", "SH"]:
    
    lr_h_r[h] = dict()
    
    for kl in k_k:
        
        lr_h_r[kl] = dict()
        lr_res = dict()

        for re in res[kl].keys():
            
            print(f"\n{h} {re} {toa_sfc} feedback ({kl} kernels)...\n")
            
            lr_res[re] = dict()
            
            # set dictionary names        
            lr_res[re]["e"] = dict()
            lr_res[re]["l"] = dict()
            lr_res[re]["t"] = dict()
            
            # early period
            lr_res[re]["e"]["s"], lr_res[re]["e"]["i"], lr_res[re]["e"]["r"], lr_res[re]["e"]["p"] = \
                                                                                lr(tas[:20], res_hem[kl][h][re][:20])[:4]
            # late period
            lr_res[re]["l"]["s"], lr_res[re]["l"]["i"], lr_res[re]["l"]["r"], lr_res[re]["l"]["p"] = \
                                                                                lr(tas[20:], res_hem[kl][h][re][20:])[:4]
            # full period
            lr_res[re]["t"]["s"], lr_res[re]["t"]["i"], lr_res[re]["t"]["r"], lr_res[re]["t"]["p"] = \
                                                                                lr(tas[:], res_hem[kl][h][re][:])[:4]        
        
        lr_h_r[h][kl] = lr_res
        # end for re
    # end for kl    
        
# end for h


#%% storing the feedbacks
print("\nGenerating nc file...\n\n")

# create the variables
for kl in k_k:
    
    f = Dataset(out_paths_fb[kl] + 
                f"/GlobalMeanRespAndFeedbacks_{ascs_n}{kl}Kernel_{toa_sfc}_and_{t_var}_piC{run}Run_a{exp}_" + 
                f"{nl.models[mod]}.nc", "w", format="NETCDF4")

    # specify dimensions, create, and fill the variables; also pass the units
    f.createDimension("1", 1)
    f.createDimension("time", n_yr)
    
    # global mean 
    for re in res[kl].keys():
        res_nc = f.createVariable(re + "_resp", "f4", "time")
        fb_e_nc = f.createVariable(re + "_fb_e", "f4", "1")
        fb_l_nc = f.createVariable(re + "_fb_l", "f4", "1")
        fb_t_nc = f.createVariable(re + "_fb_t", "f4", "1")
        ad_e_nc = f.createVariable(re + "_adj_e", "f4", "1")
        ad_l_nc = f.createVariable(re + "_adj_l", "f4", "1")
        ad_t_nc = f.createVariable(re + "_adj_t", "f4", "1")
        p_e_nc = f.createVariable(re + "_p_e", "f4", "1")
        p_l_nc = f.createVariable(re + "_p_l", "f4", "1")
        p_t_nc = f.createVariable(re + "_p_t", "f4", "1")
        p_dfb_nc = f.createVariable(re + "_p_dfb", "f4", "1")
        
        res_nc[:] = res[kl][re]
        fb_e_nc[:] = lr_r[kl][re]["e"]["s"]
        fb_l_nc[:] = lr_r[kl][re]["l"]["s"]
        fb_t_nc[:] = lr_r[kl][re]["t"]["s"]
        ad_e_nc[:] = lr_r[kl][re]["e"]["i"]
        ad_l_nc[:] = lr_r[kl][re]["l"]["i"]
        ad_t_nc[:] = lr_r[kl][re]["t"]["i"]
        p_e_nc[:] = lr_r[kl][re]["e"]["p"]
        p_l_nc[:] = lr_r[kl][re]["l"]["p"]
        p_t_nc[:] = lr_r[kl][re]["t"]["p"]
        p_dfb_nc[:] = mc_st[kl][re]
        
        res_nc.units = "W m^-2"
        fb_e_nc.units = "W m^-2 K^-1"
        fb_l_nc.units = "W m^-2 K^-1"
        fb_t_nc.units = "W m^-2 K^-1"                
        ad_e_nc.units = "W m^-2"
        ad_l_nc.units = "W m^-2"
        ad_t_nc.units = "W m^-2"
        
        res_nc.description = re + f" {toa_sfc} {ascs_desc} radiative response global annual mean"
        fb_e_nc.description = re + f" {toa_sfc} {ascs_desc} feedback years 1-20 regression"
        fb_l_nc.description = re + f" {toa_sfc} {ascs_desc} feedback years 21-150 regression"
        fb_t_nc.description = re + f" {toa_sfc} {ascs_desc} feedback years 1-150 regression"
        ad_e_nc.description = re + f" {toa_sfc} {ascs_desc} forcing adjustment years 1-20 regression"
        ad_l_nc.description = re + f" {toa_sfc} {ascs_desc} forcing adjustment years 21-150 regression"
        ad_t_nc.description = re + f" {toa_sfc} {ascs_desc} forcing adjustment years 1-150 regression"
        p_e_nc.description = re + " p-value years 1-20 regression"
        p_l_nc.description = re + " p-value years 21-150 regression"
        p_t_nc.description = re + " p-value years 1-150 regression"
        p_dfb_nc.description = (f"significance of {re} {toa_sfc} {ascs_desc} feedback change from early to late based " +
                                f"on a Monte Carlo test using {perm} permutations")
    # end for re
    
    # hemispheric mean
    for h in ["NH", "SH"]:
        for re in res[kl].keys():
            fb_h_e_nc = f.createVariable(h + "_" + re + "_fb_e", "f4", "1")
            fb_h_l_nc = f.createVariable(h + "_" + re + "_fb_l", "f4", "1")
            fb_h_t_nc = f.createVariable(h + "_" + re + "_fb_t", "f4", "1")
            ad_h_e_nc = f.createVariable(h + "_" + re + "_adj_e", "f4", "1")
            ad_h_l_nc = f.createVariable(h + "_" + re + "_adj_l", "f4", "1")
            ad_h_t_nc = f.createVariable(h + "_" + re + "_adj_t", "f4", "1")
            p_h_e_nc = f.createVariable(h + "_" + re + "_p_e", "f4", "1")
            p_h_l_nc = f.createVariable(h + "_" + re + "_p_l", "f4", "1")
            p_h_t_nc = f.createVariable(h + "_" + re + "_p_t", "f4", "1")
            
            fb_h_e_nc[:] = lr_h_r[h][kl][re]["e"]["s"]
            fb_h_l_nc[:] = lr_h_r[h][kl][re]["l"]["s"]
            fb_h_t_nc[:] = lr_h_r[h][kl][re]["t"]["s"]
            ad_h_e_nc[:] = lr_h_r[h][kl][re]["e"]["i"]
            ad_h_l_nc[:] = lr_h_r[h][kl][re]["l"]["i"]
            ad_h_t_nc[:] = lr_h_r[h][kl][re]["t"]["i"]
            p_h_e_nc[:] = lr_h_r[h][kl][re]["e"]["p"]
            p_h_l_nc[:] = lr_h_r[h][kl][re]["l"]["p"]
            p_h_t_nc[:] = lr_h_r[h][kl][re]["t"]["p"]
            
            fb_h_e_nc.units = "W m^-2 K^-1"
            fb_h_l_nc.units = "W m^-2 K^-1"
            fb_h_t_nc.units = "W m^-2 K^-1"
            ad_h_e_nc.units = "W m^-2"
            ad_h_l_nc.units = "W m^-2"
            ad_h_t_nc.units = "W m^-2"
            
            fb_h_e_nc.description = h + f" {re} {toa_sfc} {ascs_desc} feedback years 1-20 regression"
            fb_h_l_nc.description = h + f" {re} {toa_sfc} {ascs_desc} feedback years 21-150 regression"
            fb_h_t_nc.description = h + f" {re} {toa_sfc} {ascs_desc} feedback years 1-150 regression"
            ad_h_e_nc.description = h + f" {re} {toa_sfc} {ascs_desc} forcing adjustment years 1-20 regression"
            ad_h_l_nc.description = h + f" {re} {toa_sfc} {ascs_desc} forcing adjustment years 21-150 regression"
            ad_h_t_nc.description = h + f" {re} {toa_sfc} {ascs_desc} forcing adjustment years 1-150 regression"
            p_h_e_nc.description = h + f" {re} p-value years 1-20 regression"
            p_h_l_nc.description = h + f" {re} p-value years 21-150 regression"
            p_h_t_nc.description = h + f" {re} p-value years 1-150 regression"
        # end for re    
    # end for h

    f.description = (f"This file contains the individual {toa_sfc} {ascs_desc} feedbacks calculated via the {kl} " + 
                     "kernels method.\nThe surface temperature variable used to calculate the feedbacks " +
                     f"is {t_var}.\nExperiments: piControl and abrupt-{exp}CO2")
    f.history = "Created " + ti.strftime("%d/%m/%y")

    # close the dataset
    f.close()
        
# end for kl
