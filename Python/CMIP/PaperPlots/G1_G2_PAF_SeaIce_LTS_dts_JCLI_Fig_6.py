"""
Generates Fig. 6 in Eiselt and Graversen (2022), JCLI.

Paper plot EIS, dts, PAF, sea ice for G1 and G2.

Data for this plot is produced by
../Calc_Tas_Change/Calc_and_Store_Tas_and_TOA_G1_G2.py
../LTS_and_EIS/Plot_MultiModelMean_EISonTsGlobMean_TSeries_G1_G2_Comparison.py
../SeaIce/Calc_Store_Plot_NH_SeaIce_G1_G2.py
../Arctic_Amplification/Calc_and_Store_PAF.py

CORRECTION: The quantity called EIS here is actually called "lower tropospheric stability" (LTS). We were following Ceppi
and Gregory (2018), PNAS, who also make this mistake.
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
# from sklearn.linear_model import LinearRegression
from scipy import interpolate
from scipy import signal
from scipy.stats import ttest_ind
import progressbar as pg
import dask.array as da
import time as ti


#%% set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Extract_Geographic_Region import extract_region
from Functions.Func_Region_Mean import region_mean
from Functions.Func_RunMean import run_mean
from Functions.Func_APRP import simple_comp, a_cs, a_oc, a_oc2, plan_al
from Functions.Func_Convert_To_Europe_Centric import eu_centric
from Functions.Func_Lat_Mean import lat_mean
from Functions.Func_Glob_Mean import glob_mean
from Functions.Func_AnMean import an_mean
from Functions.Func_Round import r2, r3, r4
from Functions.Func_Regrid_Data_RectBiVSpline import remap
from Functions.Func_Prep_Fb_Barplot import prep_fb_bars
from Functions.Func_Set_ColorLevels import set_levs_norm_colmap
from Classes.Class_MidpointNormalize import MidpointNormalize


#%% set paths
data_path = "/G1_G2_Comp/"

# plot path
pl_path = ""


#%% consider passing the clear-sky linearity test or not? 
cslt = True
cslt_thr = 15  # --> CSLT relative error threshold in % --> 15% or 20% or... 


#%% set up some initial dictionaries and lists --> NEEDED FOR CSLT !!!
# kernels names for keys
# k_k = ["Sh08", "So08", "BM13", "H17", "P18"]
kl = "Sh08"


#%% add something to the file name concerning the CSLT
cslt_fadd = ""
if cslt:
    cslt_fadd = f"_CSLT{cslt_thr:02}{kl}"
# end if   


#%% load files
dts_nc = Dataset(data_path + f"G1_G2_Arctic_and_Gl_Mean_dts{cslt_fadd}.nc")
eis_nc = Dataset(data_path + f"G1_G2_Arctic_and_Gl_Mean_EIS{cslt_fadd}.nc")
sia_nc = Dataset(data_path + f"G1_G2_NH_SeaIce{cslt_fadd}.nc")
paf_nc = Dataset(data_path + f"G1_G2_PAF{cslt_fadd}.nc")


#%% load values
dts_m_g1 = np.mean(dts_nc.variables["dts_G1"][:], axis=0)
dts_std_g1 = np.std(dts_nc.variables["dts_G1"][:], axis=0)
dts_m_g2 = np.mean(dts_nc.variables["dts_G2"][:], axis=0)
dts_std_g2 = np.std(dts_nc.variables["dts_G2"][:], axis=0)
dts_p = dts_nc.variables["p_G1_G2"][:]

dts_ar_m_g1 = np.mean(dts_nc.variables["dts_Ar_G1"][:], axis=0)
dts_ar_std_g1 = np.std(dts_nc.variables["dts_Ar_G1"][:], axis=0)
dts_ar_m_g2 = np.mean(dts_nc.variables["dts_Ar_G2"][:], axis=0)
dts_ar_std_g2 = np.std(dts_nc.variables["dts_Ar_G2"][:], axis=0)
dts_ar_p = dts_nc.variables["p_Ar_G1_G2"][:]

eis_m_g1 = np.mean(eis_nc.variables["eis_G1"][:], axis=0)
eis_std_g1 = np.std(eis_nc.variables["eis_G1"][:], axis=0)
eis_m_g2 = np.mean(eis_nc.variables["eis_G2"][:], axis=0)
eis_std_g2 = np.std(eis_nc.variables["eis_G2"][:], axis=0)
eis_p = eis_nc.variables["p_G1_G2"][:]

eis_ar_m_g1 = np.mean(eis_nc.variables["eis_Ar_G1"][:], axis=0)
eis_ar_std_g1 = np.std(eis_nc.variables["eis_Ar_G1"][:], axis=0)
eis_ar_m_g2 = np.mean(eis_nc.variables["eis_Ar_G2"][:], axis=0)
eis_ar_std_g2 = np.std(eis_nc.variables["eis_Ar_G2"][:], axis=0)
eis_ar_p = eis_nc.variables["p_Ar_G1_G2"][:]

sia_m_g1 = np.mean(sia_nc.variables["sia_G1"][:] / 1E12, axis=0)
sia_std_g1 = np.std(sia_nc.variables["sia_G1"][:] / 1E12, axis=0)
sia_m_g2 = np.mean(sia_nc.variables["sia_G2"][:] / 1E12, axis=0)
sia_std_g2 = np.std(sia_nc.variables["sia_G2"][:] / 1E12, axis=0)
sia_p = sia_nc.variables["p_G1_G2"][:]

paf_m_g1 = np.mean(paf_nc.variables["PAF_G1"][:], axis=0)
paf_std_g1 = np.std(paf_nc.variables["PAF_G1"][:], axis=0)
paf_m_g2 = np.mean(paf_nc.variables["PAF_G2"][:], axis=0)
paf_std_g2 = np.std(paf_nc.variables["PAF_G2"][:], axis=0)
paf_p = paf_nc.variables["p_G1_G2"][:]


#%% plot all sea-ice areas
fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))
for i in np.arange(7):
    ax.plot(sia_nc.variables["sia_G1"][i, :]/1E12, c="blue")  #, label=sia_nc.variables["mods_G1"][i])
# end for i    
for i in np.arange(8):
    ax.plot(sia_nc.variables["sia_G2"][i, :]/1E12, c="red")
# end for i
ax.set_title("Sea-ice area")
ax.legend()

pl.show()
pl.close()


#%% plot all EISs
fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(10, 6))
for i in np.arange(7):
    ax.plot(eis_nc.variables["eis_G1"][i, :], c="blue")  # , label=sia_nc.variables["mods_G1"][i])
# end for i    
for i in np.arange(8):
    ax.plot(eis_nc.variables["eis_G2"][i, :], c="red")
# end for i
ax.set_title("LTS")
ax.legend()

pl.show()
pl.close()


#%% since in G1 the mean sea-ice area minus the standard deviation sometimes is negative, we calculate the the values
#   here and substitute 0 where values are negative (since negative sea-ice area makes not sense)
sia_g1_lower = sia_m_g1 - sia_std_g1
sia_g1_lower[sia_g1_lower < 0] = 0
sia_g2_lower = sia_m_g2 - sia_std_g2
sia_g2_lower[sia_g2_lower < 0] = 0


#%% 6-panel plot for  JCLI
fsz = 18
pylim = (-0.1, 1.1)

fig, axes = pl.subplots(nrows=3, ncols=2, figsize=(16, 15), sharex=True)

axes[0, 0].plot(paf_m_g1, c="blue")
axes[0, 0].plot(paf_m_g2, c="red")
axes[0, 0].fill_between(np.arange(len(paf_m_g1)), paf_m_g1 - paf_std_g1, paf_m_g1 + paf_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[0, 0].fill_between(np.arange(len(paf_m_g2)), paf_m_g2 - paf_std_g2, paf_m_g2 + paf_std_g2, 
                        facecolor="red", alpha=0.25)
axes[0, 0].set_ylabel("PAF", fontsize=fsz)
axp1 = axes[0, 0].twinx()
axp1.plot(paf_p, c="black", linewidth=0.5)
axp1.axhline(y=0.05, c="gray", linewidth=0.5)
axp1.set_ylabel("p-value", fontsize=fsz)
axp1.set_ylim(pylim)

p1, = axes[0, 1].plot(sia_m_g1, c="blue", label="G1")
p2, = axes[0, 1].plot(sia_m_g2, c="red", label="G2")
axes[0, 1].fill_between(np.arange(len(sia_m_g1)), sia_g1_lower, sia_m_g1 + sia_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[0, 1].fill_between(np.arange(len(sia_m_g2)), sia_g2_lower, sia_m_g2 + sia_std_g2, 
                        facecolor="red", alpha=0.25)
axes[0, 1].set_ylabel("NH sea-ice area in 10$^6$ km$^2$", fontsize=fsz)
axp2 = axes[0, 1].twinx()
p3, = axp2.plot(sia_p, c="black", linewidth=0.5, label="p-value")
axp2.axhline(y=0.05, c="gray", linewidth=0.5)
axp2.set_ylabel("p-value", fontsize=fsz)
axp2.set_ylim(pylim)

axes[0, 1].legend(handles=[p1, p2, p3], loc="upper right", fontsize=fsz, ncol=2)

axes[1, 0].plot(dts_m_g1, c="blue")
axes[1, 0].plot(dts_m_g2, c="red")
axes[1, 0].fill_between(np.arange(len(dts_m_g1)), dts_m_g1 - dts_std_g1, dts_m_g1 + dts_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[1, 0].fill_between(np.arange(len(dts_m_g2)), dts_m_g2 - dts_std_g1, dts_m_g2 + dts_std_g2, 
                        facecolor="red", alpha=0.25)
axes[1, 0].set_ylabel("Global surface warming in K", fontsize=fsz)
axp3 = axes[1, 0].twinx()
axp3.plot(dts_p, c="black", linewidth=0.5)
axp3.axhline(y=0.05, c="gray", linewidth=0.5)
axp3.set_ylabel("p-value", fontsize=fsz)
axp3.set_ylim(pylim)

axes[1, 1].plot(dts_ar_m_g1, c="blue")
axes[1, 1].plot(dts_ar_m_g2, c="red")
axes[1, 1].fill_between(np.arange(len(dts_ar_m_g1)), dts_ar_m_g1 - dts_ar_std_g1, dts_ar_m_g1 + dts_ar_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[1, 1].fill_between(np.arange(len(dts_ar_m_g2)), dts_ar_m_g2 - dts_ar_std_g2, dts_ar_m_g2 + dts_ar_std_g2, 
                        facecolor="red", alpha=0.25)
axes[1, 1].set_ylabel("Arctic surface warming in K", fontsize=fsz)
axp4 = axes[1, 1].twinx()
axp4.plot(dts_ar_p, c="black", linewidth=0.5)
axp4.axhline(y=0.05, c="gray", linewidth=0.5)
axp4.set_ylabel("p-value", fontsize=fsz)
axp4.set_ylim(pylim)

axes[2, 0].plot(eis_m_g1, c="blue")
axes[2, 0].plot(eis_m_g2, c="red")
axes[2, 0].fill_between(np.arange(len(eis_m_g1)), eis_m_g1 - eis_std_g1, eis_m_g1 + eis_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[2, 0].fill_between(np.arange(len(eis_m_g2)), eis_m_g2 - eis_std_g2, eis_m_g2 + eis_std_g2, 
                        facecolor="red", alpha=0.25)
axes[2, 0].set_ylabel("Global LTS in K", fontsize=fsz)
axp5 = axes[2, 0].twinx()
axp5.plot(eis_p, c="black", linewidth=0.5)
axp5.axhline(y=0.05, c="gray", linewidth=0.5)
axp5.set_ylabel("p-value", fontsize=fsz)
axp5.set_ylim(pylim)

axes[2, 1].plot(eis_ar_m_g1, c="blue")
axes[2, 1].plot(eis_ar_m_g2, c="red")
axes[2, 1].fill_between(np.arange(len(eis_ar_m_g1)), eis_ar_m_g1 - eis_ar_std_g1, eis_ar_m_g1 + eis_ar_std_g1, 
                        facecolor="blue", alpha=0.25)
axes[2, 1].fill_between(np.arange(len(eis_ar_m_g2)), eis_ar_m_g2 - eis_ar_std_g2, eis_ar_m_g2 + eis_ar_std_g2, 
                        facecolor="red", alpha=0.25)
axes[2, 1].set_ylabel("Arctic LTS in K", fontsize=fsz)
axp6 = axes[2, 1].twinx()
axp6.plot(eis_ar_p, c="black", linewidth=0.5)
axp6.axhline(y=0.05, c="gray", linewidth=0.5)
axp6.set_ylabel("p-value", fontsize=fsz)
axp6.set_ylim(pylim)

axes[2, 0].set_xlabel("Years since 4$\\times$CO$_2$", fontsize=fsz)
axes[2, 1].set_xlabel("Years since 4$\\times$CO$_2$", fontsize=fsz)

axes[0, 0].tick_params(labelsize=fsz)
axp1.tick_params(labelsize=fsz)
axes[0, 1].tick_params(labelsize=fsz)
axp2.tick_params(labelsize=fsz)
axes[1, 0].tick_params(labelsize=fsz)
axp3.tick_params(labelsize=fsz)
axes[1, 1].tick_params(labelsize=fsz)
axp4.tick_params(labelsize=fsz)
axes[2, 0].tick_params(labelsize=fsz)
axp5.tick_params(labelsize=fsz)
axes[2, 1].tick_params(labelsize=fsz)
axp6.tick_params(labelsize=fsz)

axp1.text(17, 1.03, "(a)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")
axp2.text(17, 1.03, "(b)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")
axp3.text(17, 1.03, "(c)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")
axp4.text(17, 1.03, "(d)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")
axp5.text(17, 1.03, "(e)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")
axp6.text(17, 1.03, "(f)", fontsize=fsz, horizontalalignment="center", verticalalignment="center")

fig.subplots_adjust(wspace=0.5, hspace=0.05)

# pl.savefig(pl_path + f"/PDF/Six_Panel_G1_G2_Comp{cslt_fadd}.pdf", bbox_inches="tight", dpi=250)
# pl.savefig(pl_path + f"/PNG/Six_Panel_G1_G2_Comp{cslt_fadd}.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()



