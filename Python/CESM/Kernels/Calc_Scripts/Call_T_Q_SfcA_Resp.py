"""
Call the necessary scripts to calculate temperature, surface albedo, and water vapour radiative responses via the kernel
method.
"""

#%% imports
import os
import sys
import subprocess


#%% select a case
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% choose the kernels: "Sh08", "So08", "BM13", "H17", "P18"
try:
    kl = sys.argv[2]
except:
    kl = "Sh08"
# end try except


#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[3]
except:
    toa_sfc = "TOA"
# end try except


#%% choose the climate state from which the kernels are calculated; this only pertains to the BM13 kernels for now
#   possible values: "pi", "2x", "4x", "8x"
k_st = "pi"


#%% all-sky (0) or clear-sky (1)
try:
    cs = int(sys.argv[4])
except:
    cs = 1
# end try    


#%% apply stratosphere mask? (1 == yes, 0 == no)
try:
    strat_mask = 1
except:
    strat_mask = int(sys.argv[5])
# end try except

if strat_mask:
    strat_m_str = "with stratosphere mask"
else:
    strat_m_str = "without stratosphere mask"
# end if


#%% adjust the kernel name

# kernels names for keys
k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}


#%% set paths for the subprocess.run command
py_path = "//"
tats_script = "Temperature/Calc_Temp_Resp_KernelMethod.py"
# tats_script2 = "Temperature/Calc_LR_Resp_KernelMethod.py"
q_script = "WaterVapour/Calc_WaterVapour_Resp_KernelMethod.py"
sfcalb_script = "SurfaceAlbedo/Calc_SurfaceAlbedo_Resp_KernelMethod.py"


#%% set some print strings
if cs == 0:
    sky = "all-sky"
if cs == 1:
    sky = "clear-sky"
# end if


#%% execute the subprocess.run command in for-loop
print("\n" + sky + " " + k_n[kl] + " kernels " + strat_m_str + "\n")

print("\nCalculating surface albedo response...\n")
subprocess.run(["python", py_path + sfcalb_script, case, str(cs), kl, k_st, toa_sfc])
print("\nCalculating temperature response...\n")
subprocess.run(["python", py_path + tats_script, case, str(cs), kl, k_st, str(strat_mask), toa_sfc])
# probably do not use the following --> at least that is NOT how it is done in the Pendergrass scripts!
# subprocess.run(["python", py_path + tats_script2, str(mod), str(cmip), str(cs), kl, k_st, str(strat_mask), ensemble_b, 
#                 ensemble_f, exp])
print("\nCalculating water vapour response...\n")
subprocess.run(["python", py_path + q_script, case, str(cs), kl, k_st, str(strat_mask), toa_sfc])

