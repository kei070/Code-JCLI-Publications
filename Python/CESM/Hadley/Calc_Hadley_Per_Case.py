"""
Calculate Hadley circulation strength CESM2-SOM.

Generates Fig. S12 in Eiselt and Graversen (2023), JCLI.

Be sure to set the paths in code block "set up paths" correctly.
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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy.stats import pearsonr as pears_corr
from scipy import interpolate
from scipy import signal
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti
import geocat.ncomp as geoc
from dask.distributed import Client

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Region_Mean import region_mean


#%% establish connection to the client
c = Client("localhost:8786")


#%% setup case name
case_con = "Proj2_KUE"
case_com = "Proj2_KUE_4xCO2"
try:
    case = sys.argv[2]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% set up paths
data_com_path = f"/Data/{case_com}/"
data_con_path = f"/Data/{case_con}/"
data_path = f"/Data/{case}/"

pl_path = "/Hadley_and_PAF/"


#%% list and sort the files
f_list_com = sorted(glob.glob(data_com_path + "/mpsi_Files/mpsi_*.nc"), key=str.casefold)
f_list_con = sorted(glob.glob(data_con_path + "/mpsi_Files/mpsi_*.nc"), key=str.casefold)
f_list = sorted(glob.glob(data_path + "/mpsi_Files/mpsi_*.nc"), key=str.casefold)


#%% extract the years from the file names
f_sta = int(f_list[0][-7:-3])
f_sto = int((f_list[-1][-7:-3]))


#%% load the datset
mpsi_com_nc = Dataset(f_list_com[0])
mpsi_con_nc = Dataset(f_list_con[0])
mpsi_nc = Dataset(f_list[0])

# TREFHT data
sat_com_nc = Dataset(glob.glob(data_com_path + "TREFHT_*nc")[0])
sat_con_nc = Dataset(glob.glob(data_con_path + "TREFHT_*nc")[0])
sat_exp_nc = Dataset(glob.glob(data_path + "TREFHT_*nc")[0])


#%% get the streamfunction values - control
mpsi_mon_con = da.ma.masked_array(mpsi_con_nc.variables["mpsi"], lock=True)
if len(f_list_con) > 1:
    for i in np.arange(1, len(f_list_con)):
        dataset_con = Dataset(f_list_con[i])
        mpsi_mon_con = da.concatenate([mpsi_mon_con, da.ma.masked_array(dataset_con.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% get the streamfunction values - forced
mpsi_mon = da.ma.masked_array(mpsi_nc.variables["mpsi"], lock=True)
if len(f_list) > 1:
    for i in np.arange(1, len(f_list)):
        dataset = Dataset(f_list[i])
        mpsi_mon = da.concatenate([mpsi_mon, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% get the streamfunction values - abrupt-4xCO2
mpsi_mon_com = da.ma.masked_array(mpsi_com_nc.variables["mpsi"], lock=True)
if len(f_list) > 1:
    for i in np.arange(1, len(f_list_com)):
        dataset = Dataset(f_list_com[i])
        mpsi_mon_com = da.concatenate([mpsi_mon_com, da.ma.masked_array(dataset.variables["mpsi"], lock=True)])
        # dataset.close()
    # end for i
# end if


#%% aggregate to annual means
mpsi_mon_com = mpsi_mon_com.compute()
mpsi_mon_con = mpsi_mon_con.compute()
mpsi_mon = mpsi_mon.compute()

mpsi_com = an_mean(mpsi_mon_com)
mpsi_con = an_mean(mpsi_mon_con)
mpsi = an_mean(mpsi_mon)


#%% store the number of years the Q-flux experiment is run
nyrs = np.shape(mpsi)[0]


#%% get lat and plev
lat =  mpsi_nc.variables["lat"][:]
plev = mpsi_nc.variables["plev"][:] / 100

# extract the streamfunction for the latitudes between 15 and 25 N and S
hadley_n_com = mpsi_com[:, :, (lat > 15) & (lat < 25)]
hadley_s_com = mpsi_com[:, :,  (lat > -25) & (lat < -15)]
hadley_n_con = mpsi_con[:, :, (lat > 15) & (lat < 25)]
hadley_s_con = mpsi_con[:, :,  (lat > -25) & (lat < -15)]
hadley_n = mpsi[:, :, (lat > 15) & (lat < 25)]
hadley_s = mpsi[:, :,  (lat > -25) & (lat < -15)]

# get the time series via extracting the maximum streamfunction from the Hadley values
had_n_tser_com = np.max(hadley_n_com, axis=(1, 2))
had_s_tser_com = np.min(hadley_s_com, axis=(1, 2))
had_n_tser_con = np.max(hadley_n_con, axis=(1, 2))
had_s_tser_con = np.min(hadley_s_con, axis=(1, 2))
had_n_tser = np.max(hadley_n, axis=(1, 2))
had_s_tser = np.min(hadley_s, axis=(1, 2))


#%% calculate average control mpsi and subtract it from the forced run
mpsi_con_av = np.mean(mpsi_con, axis=0)

dmpsi = mpsi - mpsi_con_av[None, :, :]


#%% plot the stream function
yr = 25
yr_con = yr
if yr >= np.shape(mpsi_con)[0]:
    yr_con = -1
# end if    

levels = np.array([-10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10])

x, y = np.meshgrid(lat, plev)

fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(15, 13), sharey=True, sharex=True)

p1 = axes[0, 0].contourf(x, y, mpsi[yr, :, :] / 1E10, levels=levels, extend="both", cmap=cm.RdBu_r)
axes[0, 0].invert_yaxis()
cb1 = fig.colorbar(p1, ax=axes[0, 0], ticks=levels)
cb1.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")

axes[0, 0].axvline(x=0, linewidth=0.5, c="gray")

axes[0, 0].set_title(case)
axes[0, 0].set_ylabel("Pressure level in hPa")


p2 = axes[0, 1].contourf(x, y, mpsi_com[yr, :, :] / 1E10, levels=levels, extend="both", cmap=cm.RdBu_r)
cb2 = fig.colorbar(p2, ax=axes[0, 1], ticks=levels)
cb2.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")

axes[0, 1].axvline(x=0, linewidth=0.5, c="gray")

axes[0, 1].set_title(case_com)


p3 = axes[1, 0].contourf(x, y, mpsi_con[yr_con, :, :] / 1E10, levels=levels, extend="both", cmap=cm.RdBu_r)
cb3 = fig.colorbar(p3, ax=axes[1, 0], ticks=levels)
cb3.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")

axes[1, 0].axvline(x=0, linewidth=0.5, c="gray")

axes[1, 0].set_title(case_con)
axes[1, 0].set_xlabel("Latitude")
axes[1, 0].set_ylabel("Pressure level in hPa")


p4 = axes[1, 1].contourf(x, y, mpsi_con_av / 1E10, levels=levels, extend="both", cmap=cm.RdBu_r)
cb4 = fig.colorbar(p4, ax=axes[1, 1], ticks=levels)
cb4.set_label("MPSI in 10$^{10}$ kg s$^{-1}$")

axes[1, 1].axvline(x=0, linewidth=0.5, c="gray")

axes[1, 1].set_title(case_con + " average")
axes[1, 1].set_xlabel("Latitude")

fig.subplots_adjust(hspace=0.125,  wspace=0.1, top=0.925)
fig.suptitle(f"Meridional overturning streamfunction year {yr}", fontsize=15)

# pl.savefig(gif_path + f"/dMPSI/dMSPI_4Panel_Yr{yr:02}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% calculate Hadley strength change from control
dhad_n = had_n_tser - np.mean(had_n_tser_con)
dhad_s = had_s_tser - np.mean(had_s_tser_con)
dhad_n_com = had_n_tser_com - np.mean(had_n_tser_con)
dhad_s_com = had_s_tser_com - np.mean(had_s_tser_con)


#%% set up the TREFHT
lat = sat_con_nc.variables["lat"][:]
lon = sat_con_nc.variables["lon"][:]

sat_com = an_mean(sat_com_nc.variables["TREFHT"][:])
sat_con = an_mean(sat_con_nc.variables["TREFHT"][:])
sat_exp = an_mean(sat_exp_nc.variables["TREFHT"][:])

sat_com_gl = glob_mean(sat_com, lat, lon)
sat_con_gl = glob_mean(sat_con, lat, lon)
sat_exp_gl = glob_mean(sat_exp, lat, lon)

x11, x12 = [0], [360]
y11, y12 = [75], [90]
sat_com_ar = region_mean(sat_com, x11, x12, y11, y12, lat, lon)
sat_con_ar = region_mean(sat_con, x11, x12, y11, y12, lat, lon)
sat_exp_ar = region_mean(sat_exp, x11, x12, y11, y12, lat, lon)

x21, x22 = [0], [360]
y21, y22 = [-90], [-75]
sat_com_aar = region_mean(sat_com, x21, x22, y21, y22, lat, lon)
sat_con_aar = region_mean(sat_con, x21, x22, y21, y22, lat, lon)
sat_exp_aar = region_mean(sat_exp, x21, x22, y21, y22, lat, lon)


#%% plot the Hadley-cell strength anomaly with respect to control time series
mks = 3
lwd = 1

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3.5), sharey=True)

axes[0].plot(dhad_n[:nyrs] / 1e10, c="red", label="dQ", linewidth=lwd, markersize=mks)
axes[1].plot(-dhad_s / 1e10, c="red", label="dQ", linewidth=lwd, markersize=mks)
axes[0].plot(dhad_n_com[:nyrs] / 1e10, c="blue", label="no-dQ", linewidth=lwd, markersize=mks)
axes[1].plot(-dhad_s_com[:nyrs] / 1e10, c="blue", label="no-dQ", linewidth=lwd, markersize=mks)

axes[0].axhline(y=0, c="gray", linewidth=0.5)
axes[1].axhline(y=0, c="gray", linewidth=0.5)

# axes[0].legend()
axes[1].legend(fontsize=13, loc="lower left")

# axes[0].set_xlim((-0.5, 10))
# axes[1].set_xlim((-0.5, 10))

axes[0].set_title("NH Hadley cell strength change", fontsize=14)
axes[1].set_title("SH Hadley cell strength change", fontsize=14)
axes[0].set_ylabel("Poleward Hadley cell strength\nchangein 10$^{10}$ kg s$^{-1}$", fontsize=12)
axes[0].set_xlabel("Years since forcing", fontsize=12)
axes[1].set_xlabel("Years since forcing", fontsize=12)

pl.subplots_adjust(wspace=0.05)

pl.savefig(pl_path + "/PDF/Hadley_Cell_Strength_4xCO2_dQ_nodQ.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/Hadley_Cell_Strength_4xCO2_dQ_nodQ.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the Hadley-cell strength time series
mks = 3
lwd = 0.75

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6), sharey=True)

axes[0].plot(had_n_tser_con[:nyrs] / 1e10, c="gray", marker="o", label="NH control", linewidth=lwd, markersize=mks)
axes[1].plot(-had_s_tser_con[:nyrs] / 1e10, c="gray", marker="o", label="SH control", linewidth=lwd, markersize=mks)
axes[0].plot(had_n_tser_com[:nyrs] / 1e10, c="black", marker="s", label="NH " + case_com, linewidth=lwd, markersize=mks)
axes[1].plot(-had_s_tser_com[:nyrs] / 1e10, c="black", marker="s", label="SH " + case_com, linewidth=lwd, markersize=mks)
axes[0].plot(had_n_tser / 1e10, c="red", marker="o", label=case, linewidth=lwd, markersize=mks)
axes[1].plot(-had_s_tser / 1e10, c="red", marker="o", label=case, linewidth=lwd, markersize=mks)

axes[0].axhline(y=np.mean(had_n_tser_con) / 1E10, c="gray", linewidth=0.5)
axes[1].axhline(y=-np.mean(had_s_tser_con) / 1E10, c="gray", linewidth=0.5)

axes[0].legend()
axes[1].legend()

# axes[0].set_xlim((-0.5, 10))
# axes[1].set_xlim((-0.5, 10))

axes[0].set_title("NH Hadley cell strength")
axes[1].set_title("SH Hadley cell strength")
axes[0].set_ylabel("Poleward Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[0].set_xlabel("Years since forcing")
axes[1].set_xlabel("Years since forcing")

# pl.savefig(pl_path + "/PDF/Hadley_Cell_Strength_4xCO2_Ensemble.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Hadley_Cell_Strength_4xCO2_Ensemble.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot the PAF index
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].plot(sat_com_ar / sat_com_gl, c="black", label=case_com + " Yr 51")
axes[0].plot(sat_con_ar / sat_con_gl, c="gray", label="control Yr 51")
axes[0].plot(sat_exp_ar / sat_exp_gl, c="red", label=case)

axes[1].plot(sat_com_aar / sat_com_gl, c="black", linestyle="-", label=case_com + " Yr 51")
axes[1].plot(sat_con_aar / sat_con_gl, c="gray", linestyle="-", label="control Yr 51")
axes[1].plot(sat_exp_aar / sat_exp_gl, c="red", linestyle="-", label=case)

axes[0].legend()

axes[0].set_title("PAF Arctic CESM2-SOM")
axes[0].set_xlabel("Years since forcing")
axes[0].set_ylabel("PAF Arctic")

axes[1].set_title("PAF Antarctic CESM2-SOM")
axes[1].set_xlabel("Years since forcing")
axes[1].set_ylabel("PAF Antarctic")


# pl.savefig(pl_path + "/PDF/PAF_4xCO2_Ensemble.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/PAF_4xCO2_Ensemble.png", bbox_inches="tight", dpi=250)


pl.show()
pl.close()


#%% correlate PAF with Hadley cell strength
lwd = 0.75

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(14, 6))

axes[0].plot(sat_com_ar / sat_com_gl, had_n_tser_com / 1e10, marker="s", c="black", label=case_com + " Yr 51", 
             linewidth=lwd)
axes[0].plot(sat_con_ar / sat_con_gl, had_n_tser_con / 1e10, marker="o", c="gray", label="control Yr 51", linewidth=lwd)
axes[0].plot(sat_exp_ar / sat_exp_gl, had_n_tser / 1e10, marker="o", c="red", label=case, linewidth=lwd)

axes[0].legend()

axes[0].set_title("Arctic PAF v. Hadley cell strength CESM2-SOM")
axes[0].set_ylabel("NH Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[0].set_xlabel("PAF")

axes[1].plot(sat_com_aar / sat_com_gl, -had_s_tser_com / 1e10, marker="s", c="black", label=case_com + " Yr 51", 
             linewidth=lwd)
axes[1].plot(sat_con_aar / sat_con_gl, -had_s_tser_con / 1e10, marker="o", c="gray", label="control Yr 51", 
             linewidth=lwd)
axes[1].plot(sat_exp_aar / sat_exp_gl, -had_s_tser / 1e10, marker="o", c="red", label=case, linewidth=lwd)

axes[1].legend()

axes[1].set_title("Antarctic PAF v. Hadley cell strength CESM2-SOM")
axes[1].set_ylabel("SH Hadley cell strength in 10$^{10}$ kg s$^{-1}$")
axes[1].set_xlabel("PAF")

# pl.savefig(pl_path + "/PDF/Hadley_v_PAF_4xCO2_Ensemble.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + "/PNG/Hadley_v_PAF_4xCO2_Ensemble.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% store the annual values in a netcdf file
print("\nGenerating nc file...\n")

out_path = f"/media/kei070/One Touch/Uni/PhD/Tromsoe_UiT/Work/Running_CESM2/Data/{case}/"
out_name = (f"mpsi_dmpsi_and_Hadley_{case}_{f_sta:04}-{f_sto:04}.nc")

f = Dataset(out_path + out_name, "w", format="NETCDF4")

# specify dimensions, create, and fill the variables; also pass the units
f.createDimension("years", len(had_n_tser))
f.createDimension("lat", len(lat))
f.createDimension("plev", len(plev))

# create the variables
lat_nc = f.createVariable("lat", "f4", "lat") 
plev_nc = f.createVariable("plev", "f4", "plev") 

mpsi_nc = f.createVariable("mpsi", "f4", ("years", "plev", "lat"))
dmpsi_nc = f.createVariable("dmpsi", "f4", ("years", "plev", "lat"))
hadn_nc = f.createVariable("Had_N", "f4", "years")
hads_nc = f.createVariable("Had_S", "f4", "years")
dhadn_nc = f.createVariable("dHad_N", "f4", "years")
dhads_nc = f.createVariable("dHad_S", "f4", "years")

# pass the data into the variables
lat_nc[:] = lat
plev_nc[:] = plev

mpsi_nc[:] = mpsi
dmpsi_nc[:] = dmpsi
hadn_nc[:] = had_n_tser
hads_nc[:] = had_s_tser
dhadn_nc[:] = dhad_n
dhads_nc[:] = dhad_s

# variable units
mpsi_nc.units = "kg / s"
dmpsi_nc.units = "kg / s"
hadn_nc.units = "kg / s"
hads_nc.units = "kg / s"
dhadn_nc.units = "kg / s"
dhads_nc.units = "kg / s"

# variable description
mpsi_nc.description = ("annual mean eridional overturning stream function calculate via NCL")
dmpsi_nc.description = ("change of annual mean eridional overturning stream function")
hadn_nc.description = ("annual mean NH Hadley circulation strength as calculated according to" + 
                       "Feldl and Bordoni (2016)")
hads_nc.description = ("annual mean SH Hadley circulation strength as calculated according to" + 
                       "Feldl and Bordoni (2016)")
dhadn_nc.description = ("change of annual mean NH Hadley circulation strength as calculated according to" + 
                        "Feldl and Bordoni (2016)")
dhads_nc.description = ("change of annual mean SH Hadley circulation strength as calculated according to" + 
                        "Feldl and Bordoni (2016)")

# add some attributes
f.description = ("This file contains the annual mean (d)mspi and (d)Hadley circulation strength for both NH and SH.\n" +
                 "The calculation of the Hadley circulation strength follows Feldl, N. and S. Bordoni (2016) in " +
                 "Journal of Climate 29, pp. 613-622. doi:10.1175/JCLI-D-150424.1")
f.history = "Created " + ti.strftime("%d/%m/%y")

# close the dataset
f.close()



