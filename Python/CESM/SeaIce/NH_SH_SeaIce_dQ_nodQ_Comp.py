"""
Compare NH and SH sea ice area with 4xCO2 and compare the dQ and no-dQ cases.

Generates Fig. S6 in Eiselt and Graversen (2023), JCLI.

Be sure to adjust data_path and pl_path.
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
case_4x51 = "Proj2_KUE_4xCO2"
case_4x41 = "Yr41_4xCO2"
case_4x61 = "Proj2_KUE_4xCO2_61"

case_dq4x51 = "dQ01yr_4xCO2"
case_dq4x41 = "Y41_dQ01"
case_dq4x61 = "Y61_dQ01"


#%% set paths
data_path = "/Data/"
pl_path = "/Plots/SeaIce/"

# os.makedirs(pl_path + "PDF/", exist_ok=True)
# os.makedirs(pl_path + "PNG/", exist_ok=True)


#%% load nc files
nc_4x51 = Dataset(glob.glob(data_path + case_4x51 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_4x41 = Dataset(glob.glob(data_path + case_4x41 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_4x61 = Dataset(glob.glob(data_path + case_4x61 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x51 = Dataset(glob.glob(data_path + case_dq4x51 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x41 = Dataset(glob.glob(data_path + case_dq4x41 + "/SeaIce_Area_NH_SH_*.nc")[0])
nc_dq4x61 = Dataset(glob.glob(data_path + case_dq4x61 + "/SeaIce_Area_NH_SH_*.nc")[0])


#%% load the data
nh_4x51 = da.ma.masked_array(nc_4x51.variables["si_area_n"], lock=True).compute()
nh_4x41 = da.ma.masked_array(nc_4x41.variables["si_area_n"], lock=True).compute()
nh_4x61 = da.ma.masked_array(nc_4x61.variables["si_area_n"], lock=True).compute()

nh_dq4x51 = da.ma.masked_array(nc_dq4x51.variables["si_area_n"], lock=True).compute()
nh_dq4x41 = da.ma.masked_array(nc_dq4x41.variables["si_area_n"], lock=True).compute()
nh_dq4x61 = da.ma.masked_array(nc_dq4x61.variables["si_area_n"], lock=True).compute()

sh_4x51 = da.ma.masked_array(nc_4x51.variables["si_area_s"], lock=True).compute()
sh_4x41 = da.ma.masked_array(nc_4x41.variables["si_area_s"], lock=True).compute()
sh_4x61 = da.ma.masked_array(nc_4x61.variables["si_area_s"], lock=True).compute()

sh_dq4x51 = da.ma.masked_array(nc_dq4x51.variables["si_area_s"], lock=True).compute()
sh_dq4x41 = da.ma.masked_array(nc_dq4x41.variables["si_area_s"], lock=True).compute()
sh_dq4x61 = da.ma.masked_array(nc_dq4x61.variables["si_area_s"], lock=True).compute()


#%% calculate annual means
nh_4x51_an = an_mean(nh_4x51)
nh_4x41_an = an_mean(nh_4x41)
nh_4x61_an = an_mean(nh_4x61)

nh_dq4x51_an = an_mean(nh_dq4x51)
nh_dq4x41_an = an_mean(nh_dq4x41)
nh_dq4x61_an = an_mean(nh_dq4x61)

sh_4x51_an = an_mean(sh_4x51)
sh_4x41_an = an_mean(sh_4x41)
sh_4x61_an = an_mean(sh_4x61)

sh_dq4x51_an = an_mean(sh_dq4x51)
sh_dq4x41_an = an_mean(sh_dq4x41)
sh_dq4x61_an = an_mean(sh_dq4x61)


#%% stack the ensembles
tlen = 20
nh_4x_ens = np.stack([nh_4x51_an[:tlen], nh_4x41_an[:tlen], nh_4x61_an[:tlen]], axis=0)
nh_dq4x_ens = np.stack([nh_dq4x51_an[:tlen], nh_dq4x41_an[:tlen], nh_dq4x61_an[:tlen]], axis=0)
sh_4x_ens = np.stack([sh_4x51_an[:tlen], sh_4x41_an[:tlen], sh_4x61_an[:tlen]], axis=0)
sh_dq4x_ens = np.stack([sh_dq4x51_an[:tlen], sh_dq4x41_an[:tlen], sh_dq4x61_an[:tlen]], axis=0)


#%% calculate ensemble mean
nh_4x_an = np.mean(nh_4x_ens, axis=0)
nh_dq4x_an = np.mean(nh_dq4x_ens, axis=0)
nh_4x_an_std = np.std(nh_4x_ens, axis=0)
nh_dq4x_an_std = np.std(nh_dq4x_ens, axis=0)

sh_4x_an = np.mean(sh_4x_ens, axis=0)
sh_dq4x_an = np.mean(sh_dq4x_ens, axis=0)
sh_4x_an_std = np.std(sh_4x_ens, axis=0)
sh_dq4x_an_std = np.std(sh_dq4x_ens, axis=0)


#%% calculate the p value of the differences of the ensemble means
dq_nh_ttest = ttest_ind(nh_4x_ens, nh_dq4x_ens, axis=0)
dq_sh_ttest = ttest_ind(sh_4x_ens, sh_dq4x_ens, axis=0)


#%% plot - Arctic annual means

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.axhline(y=0, c="gray", linewidth=0.75)
# axes.plot(nh_con[:len(sh_exp)] / 1e6, c="black", label="NH", linewidth=0.5)
# axes.plot(sh_con[:len(sh_exp)] / 1e6, c="gray", label="SH", linewidth=0.5)

axes.plot(nh_4x_an[:tlen] / 1e6, c="blue", linewidth=1, label="no dQ")
axes.plot(nh_4x51_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)
axes.plot(nh_4x41_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)
axes.plot(nh_4x61_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)

axes.plot(nh_dq4x_an[:tlen] / 1e6, c="red", linewidth=1, label="dQ")
axes.plot(nh_dq4x51_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)
axes.plot(nh_dq4x41_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)
axes.plot(nh_dq4x61_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)

axes.axhline(y=0, c="black", linewidth=0.5)
axes.set_ylim((-0.5, 12))

axes.legend()

axes.set_title("Arctic sea-ice extent CESM2-SOM")
axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("Sea-ice extent in 10$^{10}$ km$^2$")

pl.savefig(pl_path + "/PDF/Arctic_SeaIce_4x_dQ014x_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/Arctic_SeaIce_4x_dQ014x_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot - Antarctic annual means

fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(6, 4))

axes.axhline(y=0, c="gray", linewidth=0.75)
# axes.plot(nh_con[:len(sh_exp)] / 1e6, c="black", label="NH", linewidth=0.5)
# axes.plot(sh_con[:len(sh_exp)] / 1e6, c="gray", label="SH", linewidth=0.5)

axes.plot(sh_4x_an[:tlen] / 1e6, c="blue", linewidth=1, label="no dQ")
axes.plot(sh_4x51_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)
axes.plot(sh_4x41_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)
axes.plot(sh_4x61_an[:tlen] / 1e6, c="blue", linestyle=":", linewidth=0.5)

axes.plot(sh_dq4x_an[:tlen] / 1e6, c="red", linewidth=1, label="dQ")
axes.plot(sh_dq4x51_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)
axes.plot(sh_dq4x41_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)
axes.plot(sh_dq4x61_an[:tlen] / 1e6, c="red", linestyle=":", linewidth=0.5)

axes.axhline(y=0, c="black", linewidth=0.5)
axes.set_ylim((-0.5, 12))

axes.legend()

axes.set_title("Antarctic sea-ice extent CESM2-SOM")
axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("Sea-ice extent in 10$^{10}$ km$^2$")

pl.savefig(pl_path + "/PDF/Antarctic_SeaIce_4x_dQ014x_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/Antarctic_SeaIce_4x_dQ014x_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot Artctic and Antarctic sea ice
fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3), sharey=True, sharex=True)

axes[0].plot(nh_4x_an[:tlen] / 1e6, c="blue", label="no dQ")
axes[0].plot(nh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[0].fill_between(np.arange(tlen), (nh_4x_an[:tlen] + nh_4x_an_std[:tlen]) / 1e6, 
                     (nh_4x_an[:tlen] - nh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[0].fill_between(np.arange(tlen), (nh_dq4x_an[:tlen] + nh_dq4x_an_std[:tlen]) / 1e6, 
                     (nh_dq4x_an[:tlen] - nh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[0].legend()

axes[1].plot(sh_4x_an[:tlen] / 1e6, c="blue", label="no dQ")
axes[1].plot(sh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1].fill_between(np.arange(tlen), (sh_4x_an[:tlen] + sh_4x_an_std[:tlen]) / 1e6, 
                     (sh_4x_an[:tlen] - sh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1].fill_between(np.arange(tlen), (sh_dq4x_an[:tlen] + sh_dq4x_an_std[:tlen]) / 1e6, 
                     (sh_dq4x_an[:tlen] - sh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1].axhline(y=0, c="black", linewidth=0.5)

axes[0].set_title("Arctic")
axes[1].set_title("Antarctica")
axes[0].set_xlabel("Years since 4xCO$_2$")
axes[1].set_xlabel("Years since 4xCO$_2$")
axes[0].set_ylabel("Sea-ice area in 10$^{10}$ km$^2$")

axes[0].text(0, 0.5, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
axes[1].text(0, 0.5, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")

fig.suptitle("CESM2-SOM dQ and no-dQ sea-ice area")
fig.subplots_adjust(wspace=0.05, top=0.825)

pl.savefig(pl_path + "/PDF/NH_SH_SeaIce_4x_dQ014x_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/NH_SH_SeaIce_4x_dQ014x_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot Artctic and Antarctic sea ice WITH p value
p_lo_lim = -0.02

fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(10, 3), sharey=False, sharex=True)

ax00 = axes[0].twinx()
ax01 = axes[1].twinx()

axes[0].plot(nh_4x_an[:tlen] / 1e6, c="blue", label="no dQ")
axes[0].plot(nh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[0].fill_between(np.arange(tlen), (nh_4x_an[:tlen] + nh_4x_an_std[:tlen]) / 1e6, 
                     (nh_4x_an[:tlen] - nh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[0].fill_between(np.arange(tlen), (nh_dq4x_an[:tlen] + nh_dq4x_an_std[:tlen]) / 1e6, 
                     (nh_dq4x_an[:tlen] - nh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[0].axhline(y=0, c="black", linewidth=0.5)
axes[0].legend(loc="upper center")

ax00.plot(dq_nh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax00.set_ylim((p_lo_lim, 0.5))
ax00.axhline(y=0.05, c="gray", linewidth=0.5)

axes[1].plot(sh_4x_an[:tlen] / 1e6, c="blue", label="no dQ")
axes[1].plot(sh_dq4x_an[:tlen] / 1e6, c="red", label="dQ")
axes[1].fill_between(np.arange(tlen), (sh_4x_an[:tlen] + sh_4x_an_std[:tlen]) / 1e6, 
                     (sh_4x_an[:tlen] - sh_4x_an_std[:tlen]) / 1e6, facecolor="blue", alpha=0.15)
axes[1].fill_between(np.arange(tlen), (sh_dq4x_an[:tlen] + sh_dq4x_an_std[:tlen]) / 1e6, 
                     (sh_dq4x_an[:tlen] - sh_dq4x_an_std[:tlen]) / 1e6, facecolor="red", alpha=0.15)
axes[1].axhline(y=0, c="black", linewidth=0.5)

ax01.plot(dq_sh_ttest.pvalue, c="black", linewidth=0.5, linestyle=":")
ax01.set_ylim((p_lo_lim, 0.5))
ax01.axhline(y=0.05, c="gray", linewidth=0.5)

axes[0].set_ylim((-0.5, 11.5))
axes[1].set_ylim((-0.5, 11.5))

axes[0].set_title("Northern Hemisphere")
axes[1].set_title("Southern Hemisphere")
axes[0].set_xlabel("Years since 4xCO$_2$")
axes[1].set_xlabel("Years since 4xCO$_2$")
axes[0].set_ylabel("Sea-ice area in 10$^{10}$ km$^2$")
axes[1].set_ylabel("Sea-ice area in 10$^{10}$ km$^2$")

# axes[0].text(0, 0.5, "(a)", fontsize=12, horizontalalignment="center", verticalalignment="center")
# axes[1].text(0, 0.5, "(b)", fontsize=12, horizontalalignment="center", verticalalignment="center")

ax00.set_ylabel("$p$ value")
ax01.set_ylabel("$p$ value")

fig.suptitle("CESM2-SOM dQ and no-dQ sea-ice area")
fig.subplots_adjust(wspace=0.33, top=0.825)

pl.savefig(pl_path + "/PDF/NH_SH_SeaIce_4x_dQ014x_Comp.pdf", bbox_inches="tight", dpi=250)
pl.savefig(pl_path + "/PNG/NH_SH_SeaIce_4x_dQ014x_Comp.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()


#%% plot - annual means
"""
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(10, 6))

axes.axhline(y=0, c="gray", linewidth=0.75)

axes.plot(nh_con_an[:len(sh_exp_an)] / 1e6, c="black", label="NH", linewidth=0.5)
axes.plot(sh_con_an[:len(sh_exp_an)] / 1e6, c="gray", label="SH", linewidth=0.5)

axes.plot(nh_exp_an / 1e6, c="blue", label=case + " NH", linewidth=1)
axes.plot(sh_exp_an / 1e6, c="red", label=case + " SH", linewidth=1)

axes.plot(nh_a4x_an[:len(sh_exp_an)] / 1e6, c="black", label="4xCO$_2$ NH", linewidth=1, linestyle="--")
axes.plot(sh_a4x_an[:len(sh_exp_an)] / 1e6, c="gray", label="4xCO$_2$ SH", linewidth=1, linestyle="--")

axes.legend()

axes.set_title("Sea-ice extent CESM2-SOM " + case)
axes.set_xlabel("Years since forcing")
axes.set_ylabel("Sea-ice extent in 10$^{10}$ km$^2$")

pl.show()
pl.close()
"""