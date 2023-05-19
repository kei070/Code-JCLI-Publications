"""
Load the TREFHT for the different dQ experiments to test linearity.

Generates Fig. S1 in Eiselt and Graversen (2023), JCLI.

Be sure to set data_path and pl_path correctly.
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
from matplotlib import gridspec
from netCDF4 import Dataset
from matplotlib import cm
from scipy.stats import linregress as lr
from scipy import interpolate
from scipy.stats import ttest_ind
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
from Functions.Func_Region_Mean import region_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set the cases
case_con = "Proj2_KUE"

case_0p25dq = "dQ01_1_4_4xCO2"
case_0p5dq1 = "dQ01_Half_4xCO2"
case_0p5dq2 = "dQ01_Half_4xCO2_41"
case_0p5dq3 = "dQ01_Half_4xCO2_61"
case_0p75dq = "dQ01_3_4s_4xCO2"
case_1dq1 = "dQ01yr_4xCO2"
case_1dq2 = "Y41_dQ01"
case_1dq3 = "Y61_dQ01"
case_1p25dq = "dQ01_1p25_4xCO2"
case_nodq1 = "Proj2_KUE_4xCO2"
case_nodq2 = "Yr41_4xCO2"
case_nodq3 = "Proj2_KUE_4xCO2_61"

t_var = "dTREFHT"


#%% set paths 
pl_path = ""

data_path = "/Data/"

dtas_nodq1_nc = Dataset(glob.glob(data_path + f"{case_nodq1}/{t_var}_{case_nodq1}_*.nc")[0])
dtas_nodq2_nc = Dataset(glob.glob(data_path + f"{case_nodq2}/{t_var}_{case_nodq2}_*.nc")[0])
dtas_nodq3_nc = Dataset(glob.glob(data_path + f"{case_nodq3}/{t_var}_{case_nodq3}_*.nc")[0])

dtas_0p25dq_nc = Dataset(glob.glob(data_path + f"{case_0p25dq}/{t_var}_{case_0p25dq}_*.nc")[0])

dtas_0p5dq1_nc = Dataset(glob.glob(data_path + f"{case_0p5dq1}/{t_var}_{case_0p5dq1}_*.nc")[0])
dtas_0p5dq2_nc = Dataset(glob.glob(data_path + f"{case_0p5dq2}/{t_var}_{case_0p5dq2}_*.nc")[0])
dtas_0p5dq3_nc = Dataset(glob.glob(data_path + f"{case_0p5dq3}/{t_var}_{case_0p5dq3}_*.nc")[0])

dtas_0p75dq_nc = Dataset(glob.glob(data_path + f"{case_0p75dq}/{t_var}_{case_0p75dq}_*.nc")[0])

dtas_1dq1_nc = Dataset(glob.glob(data_path + f"{case_1dq1}/{t_var}_{case_1dq1}_*.nc")[0])
dtas_1dq2_nc = Dataset(glob.glob(data_path + f"{case_1dq2}/{t_var}_{case_1dq2}_*.nc")[0])
dtas_1dq3_nc = Dataset(glob.glob(data_path + f"{case_1dq3}/{t_var}_{case_1dq3}_*.nc")[0])

dtas_1p25dq_nc = Dataset(glob.glob(data_path + f"{case_1p25dq}/{t_var}_{case_1p25dq}_*.nc")[0])

si_con_nc = Dataset(glob.glob(data_path + f"{case_con}/SeaIce_Area_NH_SH_{case_con}_*.nc")[0])

si_nodq1_nc = Dataset(glob.glob(data_path + f"{case_nodq1}/SeaIce_Area_NH_SH_{case_nodq1}_*.nc")[0])
si_nodq2_nc = Dataset(glob.glob(data_path + f"{case_nodq2}/SeaIce_Area_NH_SH_{case_nodq2}_*.nc")[0])
si_nodq3_nc = Dataset(glob.glob(data_path + f"{case_nodq3}/SeaIce_Area_NH_SH_{case_nodq3}_*.nc")[0])

si_0p25dq_nc = Dataset(glob.glob(data_path + f"{case_0p25dq}/SeaIce_Area_NH_SH_{case_0p25dq}_*.nc")[0])

si_0p5dq1_nc = Dataset(glob.glob(data_path + f"{case_0p5dq1}/SeaIce_Area_NH_SH_{case_0p5dq1}_*.nc")[0])
si_0p5dq2_nc = Dataset(glob.glob(data_path + f"{case_0p5dq2}/SeaIce_Area_NH_SH_{case_0p5dq2}_*.nc")[0])
si_0p5dq3_nc = Dataset(glob.glob(data_path + f"{case_0p5dq3}/SeaIce_Area_NH_SH_{case_0p5dq3}_*.nc")[0])

si_0p75dq_nc = Dataset(glob.glob(data_path + f"{case_0p75dq}/SeaIce_Area_NH_SH_{case_0p75dq}_*.nc")[0])

si_1dq1_nc = Dataset(glob.glob(data_path + f"{case_1dq1}/SeaIce_Area_NH_SH_{case_1dq1}_*.nc")[0])
si_1dq2_nc = Dataset(glob.glob(data_path + f"{case_1dq2}/SeaIce_Area_NH_SH_{case_1dq2}_*.nc")[0])
si_1dq3_nc = Dataset(glob.glob(data_path + f"{case_1dq3}/SeaIce_Area_NH_SH_{case_1dq3}_*.nc")[0])

si_1p25dq_nc = Dataset(glob.glob(data_path + f"{case_1p25dq}/SeaIce_Area_NH_SH_{case_1p25dq}_*.nc")[0])


#%% load and values
lat = dtas_nodq1_nc.variables["lat"][:]
lon = dtas_nodq1_nc.variables["lon"][:]

dtas_nodq1 = dtas_nodq1_nc.variables["dTREFHT"][:40]
dtas_nodq2 = dtas_nodq2_nc.variables["dTREFHT"][:40]
dtas_nodq3 = dtas_nodq3_nc.variables["dTREFHT"][:40]
dtas_0p25dq = dtas_0p25dq_nc.variables["dTREFHT"][:40]
dtas_0p5dq1 = dtas_0p5dq1_nc.variables["dTREFHT"][:40]
dtas_0p5dq2 = dtas_0p5dq2_nc.variables["dTREFHT"][:40]
dtas_0p5dq3 = dtas_0p5dq3_nc.variables["dTREFHT"][:40]
dtas_0p75dq = dtas_0p75dq_nc.variables["dTREFHT"][:40]
dtas_1dq1 = dtas_1dq1_nc.variables["dTREFHT"][:40]
dtas_1dq2 = dtas_1dq2_nc.variables["dTREFHT"][:40]
dtas_1dq3 = dtas_1dq3_nc.variables["dTREFHT"][:40]
dtas_1p25dq = dtas_1p25dq_nc.variables["dTREFHT"][:40]

sin_con = np.mean(si_con_nc.variables["si_area_n"][:]) / 1e6

sin_nodq1 = an_mean(si_nodq1_nc.variables["si_area_n"][:])[:40] / 1e6
sin_nodq2 = an_mean(si_nodq2_nc.variables["si_area_n"][:])[:40] / 1e6
sin_nodq3 = an_mean(si_nodq3_nc.variables["si_area_n"][:])[:40] / 1e6

sin_0p25dq = an_mean(si_0p25dq_nc.variables["si_area_n"][:])[:40] / 1e6

sin_0p5dq1 = an_mean(si_0p5dq1_nc.variables["si_area_n"][:])[:40] / 1e6
sin_0p5dq2 = an_mean(si_0p5dq2_nc.variables["si_area_n"][:])[:40] / 1e6
sin_0p5dq3 = an_mean(si_0p5dq3_nc.variables["si_area_n"][:])[:40] / 1e6

sin_0p75dq = an_mean(si_0p75dq_nc.variables["si_area_n"][:])[:40] / 1e6

sin_1dq1 = an_mean(si_1dq1_nc.variables["si_area_n"][:])[:40] / 1e6
sin_1dq2 = an_mean(si_1dq2_nc.variables["si_area_n"][:])[:40] / 1e6
sin_1dq3 = an_mean(si_1dq3_nc.variables["si_area_n"][:])[:40] / 1e6

sin_1p25dq = an_mean(si_1p25dq_nc.variables["si_area_n"][:])[:40] / 1e6



#%% stack and average over the ensembles
dtas_1dq = np.mean(np.stack([dtas_1dq1, dtas_1dq2, dtas_1dq3], axis=0), axis=0)
dtas_0p5dq = np.mean(np.stack([dtas_0p5dq1, dtas_0p5dq2, dtas_0p5dq3], axis=0), axis=0)
dtas_nodq = np.mean(np.stack([dtas_nodq1, dtas_nodq2, dtas_nodq3], axis=0), axis=0)

sin_1dq = np.mean(np.stack([sin_1dq1, sin_1dq2, sin_1dq3], axis=0), axis=0)
sin_0p5dq = np.mean(np.stack([sin_0p5dq1, sin_0p5dq2, sin_0p5dq3], axis=0), axis=0)
sin_nodq = np.mean(np.stack([sin_nodq1, sin_nodq2, sin_nodq3], axis=0), axis=0)


#%% calculate the global mean dTas
gl_nodq = glob_mean(dtas_nodq, lat, lon)
gl_0p25dq = glob_mean(dtas_0p25dq, lat, lon)
gl_0p5dq = glob_mean(dtas_0p5dq, lat, lon)
gl_0p75dq = glob_mean(dtas_0p75dq, lat, lon)
gl_1dq = glob_mean(dtas_1dq, lat, lon)
gl_1p25dq = glob_mean(dtas_1p25dq, lat, lon)


#%% extract the Northern Polar region (>60N)
x1, x2 = [0], [360]
y1, y2 = [60], [90]

np_nodq = region_mean(dtas_nodq, x1, x2, y1, y2, lat, lon)
np_0p25dq = region_mean(dtas_0p25dq, x1, x2, y1, y2, lat, lon)
np_0p5dq = region_mean(dtas_0p5dq, x1, x2, y1, y2, lat, lon)
np_0p75dq = region_mean(dtas_0p75dq, x1, x2, y1, y2, lat, lon)
np_1dq = region_mean(dtas_1dq, x1, x2, y1, y2, lat, lon)
np_1p25dq = region_mean(dtas_1p25dq, x1, x2, y1, y2, lat, lon)

x1, x2 = [0], [360]
y1, y2 = [30], [90]

ne_nodq = region_mean(dtas_nodq, x1, x2, y1, y2, lat, lon)
ne_0p25dq = region_mean(dtas_0p25dq, x1, x2, y1, y2, lat, lon)
ne_0p5dq = region_mean(dtas_0p5dq, x1, x2, y1, y2, lat, lon)
ne_0p75dq = region_mean(dtas_0p75dq, x1, x2, y1, y2, lat, lon)
ne_1dq = region_mean(dtas_1dq, x1, x2, y1, y2, lat, lon)
ne_1p25dq = region_mean(dtas_1p25dq, x1, x2, y1, y2, lat, lon)

x1, x2 = [0], [360]
y1, y2 = [-90], [-30]

se_nodq = region_mean(dtas_nodq, x1, x2, y1, y2, lat, lon)
se_0p25dq = region_mean(dtas_0p25dq, x1, x2, y1, y2, lat, lon)
se_0p5dq = region_mean(dtas_0p5dq, x1, x2, y1, y2, lat, lon)
se_0p75dq = region_mean(dtas_0p75dq, x1, x2, y1, y2, lat, lon)
se_1dq = region_mean(dtas_1dq, x1, x2, y1, y2, lat, lon)
se_1p25dq = region_mean(dtas_1p25dq, x1, x2, y1, y2, lat, lon)


#%% calculate the delta
dnp_0p25dq = np_0p25dq - np_nodq[:len(np_0p25dq)]
dnp_0p5dq = np_0p5dq - np_nodq[:len(np_0p5dq)]
dnp_0p75dq = np_0p75dq - np_nodq[:len(np_0p75dq)]
dnp_1dq = np_1dq - np_nodq[:len(np_1dq)]
dnp_1p25dq = np_1p25dq - np_nodq[:len(np_1p25dq)]

dne_0p25dq = ne_0p25dq - ne_nodq[:len(ne_0p25dq)]
dne_0p5dq = ne_0p5dq - ne_nodq[:len(ne_0p5dq)]
dne_0p75dq = ne_0p75dq - ne_nodq[:len(ne_0p75dq)]
dne_1dq = ne_1dq - ne_nodq[:len(ne_1dq)]
dne_1p25dq = ne_1p25dq - ne_nodq[:len(ne_1p25dq)]

dse_0p25dq = se_0p25dq - se_nodq[:len(se_0p25dq)]
dse_0p5dq = se_0p5dq - se_nodq[:len(se_0p5dq)]
dse_0p75dq = se_0p75dq - se_nodq[:len(se_0p75dq)]
dse_1dq = se_1dq - se_nodq[:len(se_1dq)]
dse_1p25dq = se_1p25dq - se_nodq[:len(se_1p25dq)]

dgl_0p25dq = gl_0p25dq - gl_nodq[:len(gl_0p25dq)]
dgl_0p5dq = gl_0p5dq - gl_nodq[:len(gl_0p5dq)]
dgl_0p75dq = gl_0p75dq - gl_nodq[:len(gl_0p75dq)]
dgl_1dq = gl_1dq - gl_nodq[:len(gl_1dq)]
dgl_1p25dq = gl_1p25dq - gl_nodq[:len(gl_1p25dq)]


#%% plot the deltas
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].plot(dnp_0p25dq[:50], c="gray", label="0.25xdQ")
axes[0].plot(dnp_0p5dq[:50], c="blue", label="0.50xdQ")
axes[0].plot(dnp_0p75dq[:50], c="orange", label="0.75xdQ")
axes[0].plot(dnp_1dq[:50], c="red", label="1.00xdQ")
axes[0].plot(dnp_1p25dq[:50], c="violet", label="1.25xdQ")
axes[0].axhline(y=0, c="black", linewidth=0.5)

axes[0].legend()

axes[0].set_xlabel("Years since 4xCO$_2$")
axes[0].set_ylabel("SAT difference to no-dQ")
axes[0].set_title(f"R>{y1[0]}$\degree$N")

axes[1].plot(sin_0p25dq[:50], c="gray", label="0.25xdQ")
axes[1].plot(sin_0p5dq[:50], c="blue", label="0.50xdQ")
axes[1].plot(sin_0p75dq[:50], c="orange", label="0.75xdQ")
axes[1].plot(sin_1dq[:50], c="red", label="1.00xdQ")
axes[1].plot(sin_1p25dq[:50], c="violet", label="1.25xdQ")
axes[1].axhline(y=0, c="black", linewidth=0.5)

pl.show()
pl.close()


#%% plot the SAT change
fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 5))

axes.plot(np_nodq, c="black", label="no-dQ")
axes.plot(np_0p25dq, c="gray", label="0.25xdQ")
axes.plot(np_0p5dq, c="blue", label="0.50xdQ")
axes.plot(np_0p75dq, c="orange", label="0.75xdQ")
axes.plot(np_1dq, c="red", label="1.00xdQ")
axes.plot(np_1p25dq, c="violet", label="1.25xdQ")
axes.axhline(y=0, c="black", linewidth=0.5)

axes.legend()

axes.set_xlabel("Years since 4xCO$_2$")
axes.set_ylabel("SAT difference to no-dQ")
axes.set_title(f"R>{y1[0]}$\degree$N")

pl.show()
pl.close()


#%% calculate the delta T averaged over years 30-40, and sea-ice area over the same period
dnp_0p25dq_30_40 = np.mean(dnp_0p25dq[30:40])
dnp_0p5dq_30_40 = np.mean(dnp_0p5dq[30:40])
dnp_0p75dq_30_40 = np.mean(dnp_0p75dq[30:40])
dnp_1dq_30_40 = np.mean(dnp_1dq[30:40])
dnp_1p25dq_30_40 = np.mean(dnp_1p25dq[30:40])

dne_0p25dq_30_40 = np.mean(dne_0p25dq[30:40])
dne_0p5dq_30_40 = np.mean(dne_0p5dq[30:40])
dne_0p75dq_30_40 = np.mean(dne_0p75dq[30:40])
dne_1dq_30_40 = np.mean(dne_1dq[30:40])
dne_1p25dq_30_40 = np.mean(dne_1p25dq[30:40])

dse_0p25dq_30_40 = np.mean(dse_0p25dq[30:40])
dse_0p5dq_30_40 = np.mean(dse_0p5dq[30:40])
dse_0p75dq_30_40 = np.mean(dse_0p75dq[30:40])
dse_1dq_30_40 = np.mean(dse_1dq[30:40])
dse_1p25dq_30_40 = np.mean(dse_1p25dq[30:40])

sin_nodq_30_40 = np.mean(sin_nodq[30:40])
sin_0p25dq_30_40 = np.mean(sin_0p25dq[30:40]) - sin_nodq_30_40
sin_0p5dq_30_40 = np.mean(sin_0p5dq[30:40]) - sin_nodq_30_40
sin_0p75dq_30_40 = np.mean(sin_0p75dq[30:40]) - sin_nodq_30_40
sin_1dq_30_40 = np.mean(sin_1dq[30:40]) - sin_nodq_30_40
sin_1p25dq_30_40 = np.mean(sin_1p25dq[30:40]) - sin_nodq_30_40

dgl_0p25dq_30_40 = np.mean(dgl_0p25dq[30:40])
dgl_0p5dq_30_40 = np.mean(dgl_0p5dq[30:40])
dgl_0p75dq_30_40 = np.mean(dgl_0p75dq[30:40])
dgl_1dq_30_40 = np.mean(dgl_1dq[30:40])
dgl_1p25dq_30_40 = np.mean(dgl_1p25dq[30:40])


#%% set up the arrays for the scatter plots
dq = 50.87
dq_arr = np.array([0, 0.25 * dq, 0.5 * dq, 0.75 * dq, 1 * dq, 1.25 * dq])
dne_arr = np.array([0, dne_0p25dq_30_40, dne_0p5dq_30_40, dne_0p75dq_30_40, dne_1dq_30_40, dne_1p25dq_30_40])
dse_arr = np.array([0, dse_0p25dq_30_40, dse_0p5dq_30_40, dse_0p75dq_30_40, dse_1dq_30_40, dse_1p25dq_30_40])
dnp_arr = np.array([0, dnp_0p25dq_30_40, dnp_0p5dq_30_40, dnp_0p75dq_30_40, dnp_1dq_30_40, dnp_1p25dq_30_40])
dgl_arr = np.array([0, dgl_0p25dq_30_40, dgl_0p5dq_30_40, dgl_0p75dq_30_40, dgl_1dq_30_40, dgl_1p25dq_30_40])
sin_arr = np.array([0, sin_0p25dq_30_40, sin_0p5dq_30_40, sin_0p75dq_30_40, sin_1dq_30_40, sin_1p25dq_30_40])


#%% perform some linear regressions
slt1, yit1 = lr(dq_arr[[0, 1]], dnp_arr[[0, 1]])[:2]
slt2, yit2 = lr(dq_arr[[0, 2]], dnp_arr[[0, 2]])[:2]
slt3, yit3 = lr(dq_arr[[0, 3]], dnp_arr[[0, 3]])[:2]
slt4, yit4 = lr(dq_arr[[0, 4]], dnp_arr[[0, 4]])[:2]
slt5, yit5 = lr(dq_arr[[0, 5]], dnp_arr[[0, 5]])[:2]


#%% plot the SAT v. the dQ
fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(8, 5))

ax00_1 = axes.twinx()

axes.plot(dq_arr, dnp_arr, c="red", marker="o", label="R>60N SAT")
axes.plot(dq_arr, dne_arr, c="orange", marker="o", label="R>30N SAT")
axes.plot(dq_arr, dse_arr, c="violet", marker="o", label="R>30S SAT")
axes.plot(dq_arr, dgl_arr, c="black", marker="o", label="global SAT")
axes.plot([], [], c="blue", marker="o", label="Arctic sea-ice area")
ax00_1.plot(dq_arr, sin_arr, c="blue", marker="o", label="Arctic sea-ice area")

# axes.plot(dq_arr, dq_arr * slt1 + yit1, c="black", linewidth=0.5)
# axes.plot(dq_arr, dq_arr * slt2 + yit2, c="black", linewidth=0.5)
# axes.plot(dq_arr, dq_arr * slt3 + yit3, c="black", linewidth=0.5)
# axes.plot(dq_arr, dq_arr * slt4 + yit4, c="black", linewidth=0.5)
# axes.plot(dq_arr, dq_arr * slt5 + yit5, c="black", linewidth=0.5)

axes.axhline(y=0, c="black", linewidth=0.5)
ax00_1.axhline(y=0, c="black", linewidth=0.5)
axes.axvline(x=0, c="black", linewidth=0.5)

axes.legend(loc="center left")
# ax00_1.legend(loc="upper center")

axes.set_xlabel("North Atlantic Q-flux change in Wm$^{-2}$")
axes.set_ylabel("SAT difference to no-dQ in K")
ax00_1.set_ylabel("Sea-ice area difference to no-dQ in 10$^6$ km$^2$")

# pl.savefig(pl_path + "SAT_and_Arctic_SeaIce_dQ_Linearity.pdf", bbox_inches="tight", dpi=250)

pl.show()
pl.close()



