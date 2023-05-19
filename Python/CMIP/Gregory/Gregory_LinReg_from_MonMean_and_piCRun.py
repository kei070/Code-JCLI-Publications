"""
Generate estimates for the forcing due to the abrupt-4xCO2 and abrupt-2xCO2 in clear-sky and in all-sky conditions via 
the Gregory method. Results are stored as netCDF file.

Be sure to change data_path and out_path.
"""


#%% imports
import os
import sys
import glob
import copy
import time
import numpy as np
from numpy.random import permutation as perm
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
from scipy.stats import norm as sci_norm
from scipy.stats import percentileofscore as perc_of_score
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Z_Score_P_Value import z_score
from Functions.Func_MonteCarlo_SignificanceTest_V2 import monte_carlo
from Functions.Func_RunMean import run_mean, run_mean_da
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean, an_mean_verb
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Classes.Class_MidpointNormalize import MidpointNormalize
from dask.distributed import Client


#%% establish connection to the client
c = Client("localhost:8786")


#%% set the surface temperature variable that will be used to perform the Gregory regression (tas or ts)
t_var = "tas"


#%% Include clear-sky? Should be True if clear-sky data are available and False if the are not.
with_cs = True


#%% set early and late period threshold year
elp = 20


#%% set the interval over which to average the TOA imbalance at the elp threshold
av_w = 5
dav_w = int(av_w/2)


#%% set the runnning mean
run = 21


#%% length of the experiment in years
n_yr = 150


#%% set the experiment - either "4x" or "2x"
try:
    exp = sys.argv[3]
except:
    exp = "4x"
# end try except


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[4]
except:
    ensemble_b = "r1i1p1f1"
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[5]
except:
    ensemble_f = "r1i1p1f1"
# end try except


#%% CMIP version
try:
    cmip_v = int(sys.argv[2])
except:
    cmip_v = 6
# end try except

if cmip_v == 5:
    import Namelists.Namelist_CMIP5 as nl
    cmip = "CMIP5"
    a4x = "abrupt" + exp + "CO2"
elif cmip_v ==  6:
    import Namelists.Namelist_CMIP6 as nl
    cmip = "CMIP6"
    a4x = "abrupt-" + exp + "CO2"
# end if elif


#%% set the model number in the lists below
try:
    mod = int(sys.argv[1])
except:
    mod = 55
# end try except

print(nl.models_pl[mod])


#%% get the indices for the time interval to be chosen
exp_sta = int(nl.b_times[nl.models[mod]] * 12)
if nl.b_times[nl.models[mod]] < 10:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)    
    b_end_ind = int(n_yr * 12) + b_sta_ind
    print("\npiControl does not start at least 10 year before forcing experiment. " + 
          "Taking start index from namelist: " + str(b_sta_ind) + ".\n")
else:
    b_sta_ind = int(nl.b_times[nl.models[mod]] * 12)
    b_sta_ind = int(b_sta_ind - ((run - 1) / 2) * 12)
    print("\npiControl does start at least 10 year before forcing experiment. " + 
          "Adjusting start index from namelist: " + str(b_sta_ind) + ".\n")
# end if else


b_sta_orig = int(nl.b_times[nl.models[mod]] * 12)    
b_end_orig = int(n_yr * 12) + b_sta_orig


#%% handle the ensemble in the file name    
ensemble_b_d = ""
ensemble_f_d = ""
# handle the ensemble in the file name    
if (ensemble_b != "r1i1p1f1") & (ensemble_b != "r1i1p1"):  # base experiment ensemble (usually piControl)
    ensemble_b_d = "_" + ensemble_b
# end if
if (ensemble_f != "r1i1p1f1") & (ensemble_f != "r1i1p1"):  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f_d = "_" + ensemble_f
# end if


#%%  set paths
# --> must contain the toa, dtoa, and dtas files
data_path = ""

# output path
out_path = "/TOA_Imbalance_piC_Run/{a4x}/{t_var}_Based/"


#%% load nc files
tas_ch_nc = Dataset(data_path + f"Outputs/GlobMean_{t_var}_piC_Run/{a4x}/GlobalMean_a{exp}CO2_{t_var}_piC" + 
                    str(run) + "run_" + nl.models[mod] + ".nc")

toa_ch_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + 
                              "/dtoa_as_cs_Amon_" + nl.models_n[mod] + "_" + a4x + "_piC21Run_" + ensemble_f + "*")[0])
toa_4x_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + 
                              "/toa_Amon_" + nl.models_n[mod] + "_" + a4x + "_" + ensemble_f + "*")[0])
toa_pi_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + 
                              "/toa_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])

if with_cs:
    toacs_4x_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + 
                                    "/toacs_Amon_" + nl.models_n[mod] + "_" + a4x + "_" + ensemble_f + "*")[0])
    toacs_pi_nc = Dataset(glob.glob(data_path + "Data/" + nl.models[mod] + 
                                    "/toacs_Amon_" + nl.models_n[mod] + "_piControl_" + ensemble_b + "*")[0])
# end if
    

#%% get the variant label/ensemble
#if cmip == "CMIP6":
#    var_lab = toa_ch_nc.getncattr("variant_label")
## end if


#%% get the values
tas_ch = tas_ch_nc.variables[t_var + "_ch"][:]
tas_ch_lt = tas_ch_nc.variables[t_var + "_ch_lt"][:]

toa_as_ch = da.ma.masked_array(toa_ch_nc.variables["dtoa_as"])

toa_as_4x = da.ma.masked_array(toa_4x_nc.variables["toa"])[:int(n_yr*12), :, :]

toa_as_pi = da.ma.masked_array(toa_pi_nc.variables["toa"])
toa_as_pi_run = da.ma.masked_array(toa_ch_nc.variables["toa_as_pi_run"])

if with_cs:
    toa_cs_ch = da.ma.masked_array(toa_ch_nc.variables["dtoa_cs"])
    toa_cs_4x = da.ma.masked_array(toacs_4x_nc.variables["toacs"])[:int(n_yr*12), :, :]
    toa_cs_pi = da.ma.masked_array(toacs_pi_nc.variables["toacs"])
# end if    


#%% test if the piControl run is at least 10 longer than the abrupt4xCO2 run
add_yrs_b = False
if ((da.shape(toa_as_pi)[0] - exp_sta) >= (n_yr + 10)*12):    
    b_end_ind = int(n_yr * 12 + exp_sta + ((run - 1) / 2) * 12)
    print("\npiControl run after branch year is at least 10 years longer than abrupt4xCO2. Setting b_end_ind to " + 
          str(b_end_ind) + "\n")
    add_yrs_b = True
else:
    b_end_ind = int(n_yr * 12) + exp_sta  # branch end index
    print("\npiControl run after branch year is fewer than 10 years longer than abrupt4xCO2. Using original" + 
          " b_end_ind: " + str(b_end_ind) + "\n")
    add_yrs_b = True
# end if

toa_as_pi = toa_as_pi[b_sta_ind:b_end_ind, :, :]

if with_cs:
    toa_cs_pi = toa_cs_pi[b_sta_orig:b_end_orig, :, :]
# end if    


#%% load lat and lon
lat = toa_ch_nc.variables["lat"][:]
lon = toa_ch_nc.variables["lon"][:]


#%% if necessary load the "add" files (if the control run does not extend beyond the forced run to calculate the 21-year
#   running mean)
add_yrs = False
try:
    f_toa = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toa_*AddBe.nc"), key=str.casefold)

    toa_as_pi = da.concatenate([da.ma.masked_array(Dataset(f_toa[-1]).variables["toa"], lock=True), toa_as_pi], axis=0)

    print("\nAdded AddBe file. New piControl shape: " + str(da.shape(toa_as_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddBe file added.\n")
try:
    f_toa = sorted(glob.glob(data_path + "/Data/" + nl.models[mod] + "/AddFiles_2d_piC/toa_*AddAf.nc"), key=str.casefold)

    toa_as_pi = da.concatenate([toa_as_pi, da.ma.masked_array(Dataset(f_toa[-1]).variables["toa"], lock=True)], axis=0)

    print("\nAdded AddAf file. New piControl shape: " + str(da.shape(toa_as_pi)) + "\n")
    add_yrs = True
except:
    print("\nNo AddAf file added.\n")    
# end try except


#%% generate the annual and global means of the TOA imbalance changes
print("\nCalculating annual means...\n")
dtoa_as_an = da.mean(toa_as_ch, axis=0).compute()
if with_cs:
    dtoa_cs_an = da.mean(toa_cs_ch, axis=0).compute()
# end if    
toa_as_pi_run_an = da.mean(toa_as_pi_run, axis=0).compute()

print("\n'Change' and running mean piControl quantities done. Continuing with original quantities...\n")
print("a4x AS TOA\n")
toa_as_4x_an = an_mean_verb(toa_as_4x)
print("\n\npiC AS TOA\n")
toa_as_pi_an = an_mean_verb(toa_as_pi)

if with_cs:
    print("\n\na4x CS TOA\n")
    toa_cs_4x_an = an_mean_verb(toa_cs_4x)
    print("\n\npiC CS TOA\n")
    toa_cs_pi_an = an_mean_verb(toa_cs_pi)
# end if    
    

#%% calculate the global means
print("\n\nCalculating global means...\n")
dtoa_as_gl_an = np.zeros((np.shape(dtoa_as_an)[0]))
toa_as_4x_gl_an = np.zeros((np.shape(dtoa_as_an)[0]))
toa_as_pi_run_gl_an = np.zeros((np.shape(toa_as_pi_run_an)[0]))

if with_cs:
    dtoa_cs_gl_an = np.zeros((np.shape(dtoa_as_an)[0]))
    toa_cs_4x_gl_an = np.zeros((np.shape(dtoa_as_an)[0]))
# end if    

for y in np.arange(np.shape(dtoa_as_an)[0]):
    dtoa_as_gl_an[y] = glob_mean(dtoa_as_an[y, :, :], lat, lon)
    toa_as_pi_run_gl_an[y] = glob_mean(toa_as_pi_run_an[y, :, :], lat, lon)
    toa_as_4x_gl_an[y] = glob_mean(toa_as_4x_an[y, :, :], lat, lon)
    
    if with_cs:
        dtoa_cs_gl_an[y] = glob_mean(dtoa_cs_an[y, :, :], lat, lon)
        toa_cs_4x_gl_an[y] = glob_mean(toa_cs_4x_an[y, :, :], lat, lon)
    # end if    
# end for y

toa_as_pi_gl_an = np.zeros((np.shape(toa_as_pi_an)[0]))
for y in np.arange(np.shape(toa_as_pi_an)[0]):
    toa_as_pi_gl_an[y] = glob_mean(toa_as_pi_an[y, :, :], lat, lon)
# end for y

if with_cs:
    toa_cs_pi_gl_an = np.zeros((np.shape(toa_cs_pi_an)[0]))
    for y in np.arange(np.shape(toa_cs_pi_an)[0]):
        toa_cs_pi_gl_an[y] = glob_mean(toa_cs_pi_an[y, :, :], lat, lon)
    # end for y
# end if


#%% calculate the linear trend over the 150 years of piControl corresponding to the abrupt-4xCO2
sl, yi, r, p = lr(np.arange(len(toa_as_pi_gl_an[10:-10])), toa_as_pi_gl_an[10:-10])[:4]
# calculate the change with respect to the linear piControl trend
dtoa_as_lt = toa_as_4x_gl_an - (np.arange(len(toa_as_pi_gl_an[10:-10])) * sl + yi)

if with_cs:
    sl_cs, yi_cs, r, p_cs = lr(np.arange(len(toa_cs_pi_gl_an[10:-10])), toa_cs_pi_gl_an[10:-10])[:4]
    # calculate the change with respect to the linear piControl trend
    dtoa_cs_lt = toa_cs_4x_gl_an - (np.arange(len(toa_cs_pi_gl_an)) * sl_cs + yi_cs)
# end if


#%% perform the linear regression between the TOA imbalances and the tas --> 21-year run of piControl
sl_as, in_as, r, p_as = lr(tas_ch, dtoa_as_gl_an)[:4]
ecs_as = -in_as/sl_as
sl_as_r = np.round(sl_as, decimals=2)
p_as_r = np.round(p_as, decimals=4)

sl_as_e, in_as_e, r, p_as_e = lr(tas_ch[:elp], dtoa_as_gl_an[:elp])[:4]
ecs_as_e = -in_as_e/sl_as_e
sl_as_e_r = np.round(sl_as_e, decimals=2)
p_as_e_r = np.round(p_as_e, decimals=4)

sl_as_l, in_as_l, r, p_as_l = lr(tas_ch[elp:], dtoa_as_gl_an[elp:])[:4]
ecs_as_l = -in_as_l/sl_as_l
sl_as_l_r = np.round(sl_as_l, decimals=2)
p_as_l_r = np.round(p_as_l, decimals=4)

if with_cs:
    sl_cs, in_cs, r, p_cs = lr(tas_ch, dtoa_cs_gl_an)[:4]
    ecs_cs = -in_cs/sl_cs
    sl_cs_r = np.round(sl_cs, decimals=2)
    p_cs_r = np.round(p_cs, decimals=4)

    sl_cs_e, in_cs_e, r, p_cs_e = lr(tas_ch[:elp], dtoa_cs_gl_an[:elp])[:4]
    ecs_cs_e = -in_cs_e/sl_cs_e
    sl_cs_e_r = np.round(sl_cs_e, decimals=2)
    p_cs_e_r = np.round(p_cs_e, decimals=4)

    sl_cs_l, in_cs_l, r, p_cs_l = lr(tas_ch[elp:], dtoa_cs_gl_an[elp:])[:4]
    ecs_cs_l = -in_cs_l/sl_cs_l
    sl_cs_l_r = np.round(sl_cs_l, decimals=2)
    p_cs_l_r = np.round(p_cs_l, decimals=4)
# end if

print("all-sky forcing (total period): " + str(in_as) + " W/m²")
print("all-sky forcing (early period): " + str(in_as_e) + " W/m²")
print("all-sky forcing (late period): " + str(in_as_l) + " W/m²")

if with_cs:
    print("clear-sky forcing (total period): " + str(in_cs) + " W/m²")
    print("clear-sky forcing (early period): " + str(in_cs_e) + " W/m²")
    print("clear-sky forcing (late period): " + str(in_cs_l) + " W/m²")
    
    print("difference: " + str(in_cs-in_as) + " W/m²")
    print("fraction of all-sky: " + str((in_cs-in_as)/in_as * 100) + " %")
# end if


#%% perform the linear regression between the TOA imbalances and the tas --> LINEAR TREND (not 21-year run) of piControl
sl_as_lt, in_as_lt, r, p_as_lt = lr(tas_ch_lt, dtoa_as_lt)[:4]
ecs_as_lt = -in_as_lt/sl_as_lt
sl_as_lt_r = np.round(sl_as_lt, decimals=2)
p_as_lt_r = np.round(p_as_lt, decimals=4)

sl_as_e_lt, in_as_e_lt, r, p_as_e_lt = lr(tas_ch_lt[:elp], dtoa_as_lt[:elp])[:4]
ecs_as_e_lt = -in_as_e_lt/sl_as_e_lt
sl_as_e_lt_r = np.round(sl_as_e_lt, decimals=2)
p_as_e_lt_r = np.round(p_as_e_lt, decimals=4)

sl_as_l_lt, in_as_l_lt, r, p_as_l_lt = lr(tas_ch_lt[elp:], dtoa_as_lt[elp:])[:4]
ecs_as_l_lt = -in_as_l_lt/sl_as_l_lt
sl_as_l_lt_r = np.round(sl_as_l_lt, decimals=2)
p_as_l_lt_r = np.round(p_as_l_lt, decimals=4)

if with_cs:
    sl_cs_lt, in_cs_lt, r, p_cs_lt = lr(tas_ch_lt, dtoa_cs_lt)[:4]
    ecs_cs_lt = -in_cs_lt/sl_cs_lt
    sl_cs_lt_r = np.round(sl_cs_lt, decimals=2)
    p_cs_lt_r = np.round(p_cs_lt, decimals=4)
    
    sl_cs_e_lt, in_cs_e_lt, r, p_cs_e_lt = lr(tas_ch_lt[:elp], dtoa_cs_lt[:elp])[:4]
    ecs_cs_e_lt = -in_cs_e_lt/sl_cs_e_lt
    sl_cs_e_lt_r = np.round(sl_cs_e_lt, decimals=2)
    p_cs_e_lt_r = np.round(p_cs_e_lt, decimals=4)
    
    sl_cs_l_lt, in_cs_l_lt, r, p_cs_l_lt = lr(tas_ch_lt[elp:], dtoa_cs_lt[elp:])[:4]
    ecs_cs_l_lt = -in_cs_l_lt/sl_cs_l_lt
    sl_cs_l_lt_r = np.round(sl_cs_l_lt, decimals=2)
    p_cs_l_lt_r = np.round(p_cs_l_lt, decimals=4)
# end if


#%% perform a significance test on the slope difference of the linear trends (i.e., a significance test on the change
#   over time of the total feedbacks)
z_p_val_as = z_score(tas_ch[:elp], dtoa_as_gl_an[:elp], tas_ch[elp:], dtoa_as_gl_an[elp:])
z_p_val_as_lt = z_score(tas_ch_lt[:elp], dtoa_as_lt[:elp], tas_ch[elp:], dtoa_as_lt[elp:])

print("\n\n")
print("AS piC run: " + str(z_p_val_as))
print("AS piC LR: " + str(z_p_val_as_lt))

if with_cs:
    z_p_val_cs = z_score(tas_ch[:elp], dtoa_cs_gl_an[:elp], tas_ch[elp:], dtoa_cs_gl_an[elp:])
    z_p_val_cs_lt = z_score(tas_ch_lt[:elp], dtoa_cs_lt[:elp], tas_ch[elp:], dtoa_cs_lt[elp:])
    print("CS piC LR: " + str(z_p_val_cs_lt))
    print("CS piC run: " + str(z_p_val_cs))
    print("\n\n")
# end if


#%% consider random permutations of tas_ch and dtoa_as_gl_an
sl_dif = np.zeros(5000)

for i in np.arange(5000):
    
    tas_perm_20 = perm(tas_ch[:20])
    toa_perm_20 = perm(dtoa_as_gl_an[:20])
    tas_perm_130 = perm(tas_ch[20:])
    toa_perm_130 = perm(dtoa_as_gl_an[20:])
    
    sl_p20, in_p20, r, p_p20 = lr(tas_perm_20, toa_perm_20)[:4]
    sl_p130, in_p130, r, p_p130 = lr(tas_perm_130, toa_perm_130)[:4]

    sl_dif[i] = sl_p130 - sl_p20
    
# end for i

# pl.hist(sl_dif)    
# pl.plot()

p_score = perc_of_score(sl_dif, sl_as_l-sl_as_e)/100

if p_score > 0.5:
    p_score = 1 - p_score
# end if    

print("\n" + nl.models[mod] + " quantile of score Monte Carlo test: " + str(p_score) + "\n")


#%% store the clear- and all-sky TOA radiation in a netcdf file

print("\nGenerating nc file...\n")
f = Dataset(out_path + f"TOA_Imbalance_GlobAnMean_and_{t_var}_Based_TF_piC{run}Run_a{exp}_{nl.models[mod]}.nc", 
            "w", format="NETCDF4")

# set the variant label
if cmip == "CMIP6":
    f.setncattr("variant_label_base", ensemble_b)
    f.setncattr("variant_label_forced", ensemble_f)
# end if

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("time", len(dtoa_as_gl_an))
f.createDimension("time170", len(toa_as_pi_gl_an))
f.createDimension("2", 2)
f.createDimension("1", 1)

# create the variables
tim_nc = f.createVariable("time", "f4", "time")
toa_a4x_as_nc = f.createVariable("toa_a4x_as", "f4", "time")
toa_pic_run_as_nc = f.createVariable("toa_pic_run_as", "f4", "time")
toa_pic_as_nc = f.createVariable("toa_pic_as", "f4", "time170")
toa_imb_as_nc = f.createVariable("toa_imb_as", "f4", "time")

toa_imb_lt_as_nc = f.createVariable("toa_imb_lt_as", "f4", "time")
forcing_as_nc = f.createVariable("forcing_as", "f4", "2")
forcing_as_e_nc = f.createVariable("forcing_as_e", "f4", "2")
forcing_as_l_nc = f.createVariable("forcing_as_l", "f4", "2")
elp_imb_as_nc = f.createVariable("elp_imb_as", "f4", "1")
elp_t_nc = f.createVariable("elp_t", "f4", "1")
fb_as_nc = f.createVariable("fb_as", "f4", "2")
fb_as_e_nc = f.createVariable("fb_as_e", "f4", "2")
fb_as_l_nc = f.createVariable("fb_as_l", "f4", "2")
ecs_as_nc = f.createVariable("ecs_as", "f4", "2")
ecs_as_e_nc = f.createVariable("ecs_as_e", "f4", "2")
ecs_as_l_nc = f.createVariable("ecs_as_l", "f4", "2")
p_as_nc = f.createVariable("p_as", "f4", "2")
p_as_e_nc = f.createVariable("p_as_e", "f4", "2")
p_as_l_nc = f.createVariable("p_as_l", "f4", "2")
p_as_dfb_nc = f.createVariable("p_as_dfb", "f4", "2")

# pass the data into the variables
tim_nc[:] = np.arange(len(dtoa_as_gl_an))
toa_a4x_as_nc[:] = toa_as_4x_gl_an
toa_pic_run_as_nc[:] = toa_as_pi_run_gl_an
toa_pic_as_nc[:] = toa_as_pi_gl_an
toa_imb_as_nc[:] = dtoa_as_gl_an
toa_imb_lt_as_nc[:] = dtoa_as_lt
forcing_as_nc[:] = in_as, in_as_lt
forcing_as_e_nc[:] = in_as_e, in_as_e_lt
forcing_as_l_nc[:] = in_as_l, in_as_l_lt
elp_imb_as_nc[:] = np.mean(dtoa_as_gl_an[elp-dav_w:elp+dav_w+1])
elp_t_nc[:] = np.mean(tas_ch[elp-dav_w:elp+dav_w+1])
fb_as_nc[:] = sl_as, sl_as_lt
fb_as_e_nc[:] = sl_as_e, sl_as_e_lt
fb_as_l_nc[:] = sl_as_l, sl_as_l_lt
ecs_as_nc[:] = ecs_as, ecs_as_lt
ecs_as_e_nc[:] = ecs_as_e, ecs_as_e_lt
ecs_as_l_nc[:] = ecs_as_l, ecs_as_l_lt
p_as_nc[:] = p_as, p_as_lt
p_as_e_nc[:] = p_as_e, p_as_e_lt
p_as_l_nc[:] = p_as_l, p_as_l_lt
p_as_dfb_nc[:] = p_score, z_p_val_as

# add units
toa_a4x_as_nc.units = "W m^-2"
toa_pic_run_as_nc.units = "W m^-2"
toa_pic_as_nc.units = "W m^-2"
toa_imb_as_nc.units = "W m^-2"
forcing_as_nc.units = "W m^-2"
forcing_as_e_nc.units = "W m^-2"
forcing_as_l_nc.units = "W m^-2"
elp_imb_as_nc.units = "W m^-2"
elp_t_nc.units = "K"
fb_as_nc.units = "W m^-2 K^-1"
fb_as_e_nc.units = "W m^-2 K^-1"
fb_as_l_nc.units = "W m^-2 K^-1"
ecs_as_nc.units = "K"
ecs_as_e_nc.units = "K"
ecs_as_l_nc.units = "K"

# add a description to each variable
toa_a4x_as_nc.description = ("abrupt-4xCO2 TOA radiative imbalance all-sky global annual mean")
toa_pic_run_as_nc.description = ("piControl TOA radiative imbalance all-sky global " + str(run) + "-year running mean")
toa_pic_as_nc.description = ("piControl TOA radiative imbalance all-sky global annual mean")
toa_imb_as_nc.description = ("change of TOA radiative imbalance all-sky, calculated with respect to 21-year piControl " + 
                             "running mean")
toa_imb_lt_as_nc.description = ("TOA radiative imbalance all-sky, calculated with respect to linear piControl trend")
forcing_as_nc.description = ("all-sky forcing estimate (Gregory method) complete period")
forcing_as_e_nc.description = ("all-sky forcing estimate (Gregory method, early period")
forcing_as_l_nc.description = ("all-sky forcing estimate (Gregory method) late period")
elp_imb_as_nc.desciption = (str(av_w) + 
                            "-year all-sky TOA imbalance with center at early to late period threshold year, i.e. " +
                            "year " + str(elp))
elp_t_nc.desciption = (str(av_w) + 
                       "-year tas change with center at early to late period threshold year, i.e. year " + str(elp))
fb_as_nc.description = ("all-sky feedback estimate (Gregory method) complete period")
fb_as_e_nc.description = ("all-sky feedback estimate (Gregory method) early period")
fb_as_l_nc.description = ("all-sky feedback estimate (Gregory method) late period")
ecs_as_nc.description = ("all-sky effective climate senistivity estimate (Gregory method) complete period")
ecs_as_e_nc.description = ("all-sky effective climate senistivity estimate (Gregory method) early period")
ecs_as_l_nc.description = ("all-sky effective climate senistivity estimate (Gregory method) late period")
p_as_nc.description = ("all-sky feedback estimate p-value (Gregory method) complete period")
p_as_e_nc.description = ("all-sky feedback estimate p-value (Gregory method) early period")
p_as_l_nc.description = ("all-sky feedback estimate p-value (Gregory method) late period")
p_as_dfb_nc.description = ("significance of feedback change from early to late: first value is based on a Monte Carlo " +
                           "test using 5000 random permutations; second value is based on a z-statistic")

if with_cs:
    # create the clear-sky variables
    toa_imb_cs_nc = f.createVariable("toa_imb_cs", "f4", "time")
    toa_imb_lt_cs_nc = f.createVariable("toa_imb_lt_cs", "f4", "time")
    forcing_cs_nc = f.createVariable("forcing_cs", "f4", "2")
    forcing_cs_e_nc = f.createVariable("forcing_cs_e", "f4", "2")
    forcing_cs_l_nc = f.createVariable("forcing_cs_l", "f4", "2")
    elp_imb_cs_nc = f.createVariable("elp_imb_cs", "f4", "1")
    fb_cs_nc = f.createVariable("fb_cs", "f4", "2")
    fb_cs_e_nc = f.createVariable("fb_cs_e", "f4", "2")
    fb_cs_l_nc = f.createVariable("fb_cs_l", "f4", "2")
    ecs_cs_nc = f.createVariable("ecs_cs", "f4", "2")
    ecs_cs_e_nc = f.createVariable("ecs_cs_e", "f4", "2")
    ecs_cs_l_nc = f.createVariable("ecs_cs_l", "f4", "2")
    p_cs_nc = f.createVariable("p_cs", "f4", "2")
    p_cs_e_nc = f.createVariable("p_cs_e", "f4", "2")
    p_cs_l_nc = f.createVariable("p_cs_l", "f4", "2")
    
    # pass the data into the variables
    toa_imb_cs_nc[:] = dtoa_cs_gl_an
    toa_imb_lt_cs_nc[:] = dtoa_cs_lt
    forcing_cs_nc[:] = in_cs, in_cs_lt
    forcing_cs_e_nc[:] = in_cs_e, in_cs_e_lt
    forcing_cs_l_nc[:] = in_cs_l, in_cs_l_lt
    elp_imb_cs_nc[:] = np.mean(dtoa_cs_gl_an[elp-dav_w:elp+dav_w+1])
    fb_cs_nc[:] = sl_cs, sl_cs_lt
    fb_cs_e_nc[:] = sl_cs_e, sl_cs_e_lt
    fb_cs_l_nc[:] = sl_cs_l, sl_cs_l_lt
    ecs_cs_nc[:] = ecs_cs, ecs_cs_lt
    ecs_cs_e_nc[:] = ecs_cs_e, ecs_cs_e_lt
    ecs_cs_l_nc[:] = ecs_cs_l, ecs_cs_l_lt
    p_cs_nc[:] = p_cs, p_cs_lt
    p_cs_e_nc[:] = p_cs_e, p_cs_e_lt
    p_cs_l_nc[:] = p_cs_l, p_cs_l_lt
    
    # pass the data into the variables
    toa_imb_cs_nc.units = "W m^-2"
    forcing_cs_nc.units = "W m^-2"
    forcing_cs_e_nc.units = "W m^-2"
    forcing_cs_l_nc.units = "W m^-2"
    elp_imb_cs_nc.units = "W m^-2"
    fb_cs_nc.units = "W m^-2 K^-1"
    fb_cs_e_nc.units = "W m^-2 K^-1"
    fb_cs_l_nc.units = "W m^-2 K^-1"
    ecs_cs_nc.units = "K"
    ecs_cs_e_nc.units = "K"
    ecs_cs_l_nc.units = "K"
    
    # add a description to each variable
    toa_imb_cs_nc.description = ("change of TOA radiative imbalance clear-sky, calculated with respect to 21-year " +
                                 "piControl running mean")
    toa_imb_lt_cs_nc.description = ("TOA radiative imbalance clear-sky, calculated with respect to linear " +
                                    "piControl trend")
    forcing_cs_e_nc.description = ("clear-sky forcing estimate (Gregory method) complete period")
    forcing_cs_l_nc.description = ("clear-sky forcing estimate (Gregory method) early period")
    forcing_cs_nc.description = ("clear-sky forcing estimate (Gregory method, late period)")
    elp_imb_cs_nc.desciption = (str(av_w) + "-year clear-sky TOA imbalance with center at early to late period " +
                                "threshold year, i.e. year " + str(elp))
    fb_cs_nc.description = ("clear-sky feedback estimate (Gregory method) complete period")
    fb_cs_e_nc.description = ("clear-sky feedback estimate (Gregory method) early period")
    fb_cs_l_nc.description = ("clear-sky feedback estimate (Gregory method) late period")
    ecs_cs_nc.description = ("clear-sky effective climate senistivity estimate (Gregory method) complete period")
    ecs_cs_e_nc.description = ("clear-sky effective climate senistivity estimate (Gregory method) early period")
    ecs_cs_l_nc.description = ("clear-sky effective climate senistivity estimate (Gregory method) late period")
    p_cs_nc.description = ("clear-sky feedback estimate p-value (Gregory method) complete period")
    p_cs_e_nc.description = ("clear-sky feedback estimate p-value (Gregory method) early period")
    p_cs_l_nc.description = ("clear-sky feedback estimate p-value (Gregory method) late period")
# end if

# add attributes

desc_add = ""

if add_yrs:
    desc_add = ("\nNote that the original piControl run did not contain enough years to calculate the running\n"+
                "mean values completely corresponding to the forcing experiment which is why they were\n" + 
                "extrapolated linearly from the adjacent 21 years of simulation.")
# end if

f.description = ("This file contains global mean annual mean radiative imbalance at TOA for all- and clear-sky as " + 
                 "well as the respective forcing, feedback, and sensitiviy estimates from the Gregory method.\n" + 
                 "Forcing, feedback, and ECS estimates each contain two values. For the first, the anomalies " + 
                 f"of {t_var}\n" + 
                 "and TOA imbalance and were calculated with respect to a 21-year running mean of the piControl run\n" + 
                 "whereas for the second the anomalies were caluclated with respect to a linear trend calculated " + 
                 f"from the piControl.\nExperiments: piControl and abrupt-{exp}CO2 {desc_add}")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()