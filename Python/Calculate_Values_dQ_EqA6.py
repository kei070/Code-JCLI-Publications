"""
This script contains the estimate of the Q-flux change induced by the AMOC change.
For the derivation of the equation see the appendix in our 2nd paper.

For the values of the AMOC see the script
./CMIP/AMOC/Compare_AMOC_2_Groups.py

And for the temperature values see the script
./G1_G2_Regional_Comp/SST_Difference_Regions.py
"""


#%% imports
import numpy as np
import pylab as pl


#%% paths
pl_path = ""


#%% try to estimate the
amoc1, amoc2 = 1.71e10, 2.65e10  # kg/s

# mean over years 18-22 --> where do these values come from?
damoc1, damoc2 = -0.52e10, -1.02e10  # kg/s 
dt1, dt2 = 22.83, 22.46  # K
ddt1, ddt2 = -2.3, -0.71  # K

# values correspond to years 18-22 mean and regions corresponding to the ones where
# the dQ is implemented in CESM2-SOM
damoc1, damoc2 = -0.52e10, -1.02e10  # kg/s 
dt1, dt2 = 24.39, 26.09 # K
ddt1, ddt2 = -2.26, -0.91  # K

# values correspond to years 12-17 mean and regions corresponding to the ones where
# the dQ is implemented in CESM2-SOM
damoc1, damoc2 = -0.38e10, -0.92e10  # kg/s 
dt1, dt2 = 24.36, 25.89 # K
ddt1, ddt2 = -2.28, -1.1  # K

# values correspond to years 12-17 mean and regions corresponding where the dQ is implemented in CESM2-SOM
# in the Tropics and the Bellomo et al. (2021) region in the North Atlantic
damoc1, damoc2 = -0.38e10, -0.92e10  # kg/s 
dt1, dt2 = 22.72, 23.87 # K
ddt1, ddt2 = -1.9, -0.69  # K
# --> use this in the paper

c_p = 4.184e3  # J/kg/K
de1 = (damoc1 * dt1 + amoc1 * ddt1 + damoc1 * ddt1) * c_p # W
de2 = (damoc2 * dt2 + amoc2 * ddt2 + damoc2 * ddt2) * c_p # W
ar_area = 9.828491e12  # m^2
tr_area = 19.280592e12  # m^2

dq_ar1 = de1 / ar_area
dq_tr1 = -de1 / tr_area
dq_ar2 = de2 / ar_area
dq_tr2 = -de2 / tr_area
print(f"energy transport change G1: {de1 * 1e-15} PW")
print(f"energy transport change G2: {de2 * 1e-15} PW")
print(f"Delta energy transport {de2 * 1e-15 - de1 * 1e-15} PW\n")
print(f"Arctic Q-flux change G1: {dq_ar1} W/m^2")
print(f"Tropic Q-flux change G1: {dq_tr1} W/m^2")
print(f"Arctic Q-flux change G2: {dq_ar2} W/m^2")
print(f"Tropic Q-flux change G2: {dq_tr2} W/m^2\n")

print(f"Arctic Q-flux delta: {dq_ar2 - dq_ar1} W/m^2")
print(f"Tropic Q-flux delta: {dq_tr2 - dq_tr1} W/m^2\n")

print(f"G1 term 1: {amoc1 * ddt1 / 1e12 * c_p}   G2 term 1: {amoc2 * ddt2 / 1e12 * c_p}")
print(f"G1 term 2: {damoc1 * dt1 / 1e12 * c_p}   G2 term 2: {damoc2 * dt2 / 1e12 * c_p}")
print(f"G1 term 3: {damoc1 * ddt1 / 1e12 * c_p}   G2 term 3: {damoc2 * ddt2 / 1e12 * c_p}\n")

# print(f"energy transport difference G2 minus G1: {(de2 - de1) / 1E15}")

# raise Exception

#%% generate the time series

# number of months after which dQ has reached its final value
mm_qfinal01 = 12
mm_qfinal05 = 60
mm_qfinal10 = 120

# dQ in Arctic and Tropics, respectively
arctic_dq = 50.87
tropic_dq = -25.93

# dQ/dT
arctic_dqdt01 = arctic_dq / mm_qfinal01
arctic_dqdt05 = arctic_dq / mm_qfinal05
arctic_dqdt10 = arctic_dq / mm_qfinal10
tropic_dqdt01 = tropic_dq / mm_qfinal01
tropic_dqdt05 = tropic_dq / mm_qfinal05
tropic_dqdt10 = tropic_dq / mm_qfinal10

# set up 15-year time series
t_series = np.arange(0, 180, 1)

# calcualte the time series
ar_dq_01_tser = t_series * arctic_dqdt01
ar_dq_01_tser[ar_dq_01_tser > arctic_dq] = arctic_dq
ar_dq_05_tser = t_series * arctic_dqdt05
ar_dq_05_tser[ar_dq_05_tser > arctic_dq] = arctic_dq
ar_dq_10_tser = t_series * arctic_dqdt10
ar_dq_10_tser[ar_dq_10_tser > arctic_dq] = arctic_dq
tr_dq_01_tser = t_series * tropic_dqdt01
tr_dq_01_tser[tr_dq_01_tser < tropic_dq] = tropic_dq
tr_dq_05_tser = t_series * tropic_dqdt05
tr_dq_05_tser[tr_dq_05_tser < tropic_dq] = tropic_dq
tr_dq_10_tser = t_series * tropic_dqdt10
tr_dq_10_tser[tr_dq_10_tser < tropic_dq] = tropic_dq


#%% plot the time series
fig, axes = pl.subplots(ncols=1, nrows=1, figsize=(7, 4.5))

axes.plot(ar_dq_01_tser, c="blue", label="Arctic dQ 1 Year")
axes.plot(ar_dq_05_tser, c="orange", label="Arctic dQ 5 Years")
axes.plot(ar_dq_10_tser, c="red", label="Arctic dQ 10 Years")

axes.plot(tr_dq_01_tser, c="blue", linestyle="--", label="Tropic dQ 1 Year")
axes.plot(tr_dq_05_tser, c="orange", linestyle="--", label="Tropic dQ 5 Years")
axes.plot(tr_dq_10_tser, c="red", linestyle="--", label="Tropic dQ 10 Years")

axes.axhline(y=0, c="grey", linewidth=0.5)
axes.axvline(x=12, c="grey", linewidth=0.5)
axes.axvline(x=60, c="grey", linewidth=0.5)
axes.axvline(x=120, c="grey", linewidth=0.5)

axes.legend(loc=(0.67, 0.45))

axes.set_xlabel("Months after abrupt4xCO$_2$")
axes.set_ylabel("Q-flux change in Wm$^{-2}$")
axes.set_title("Q-flux adjustment CESM2-SOM")

# pl.savefig(pl_path + "dQ_Illustration.png", bbox_inches="tight", dpi=250)

pl.show()
pl.close()



