"""
Call the necessary scripts to calculate temperature, surface albedo, and water vapour radiative "cloud adjustments" via 
the kernel method.
"""

#%% imports
import os
import sys
import subprocess

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)


#%% choose the kernels: "Sh08", "So08", "BM13", "H17", "P18"
try:
    kl = sys.argv[1]
except:
    kl = "So08"
# end try except


#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[9]
except:
    toa_sfc = "SFC"
# end try except


#%% choose the climate state from which the kernels are calculated; this only pertains to the BM13 kernels for now
#   possible values: "pi", "2x", "4x", "8x"
k_st = "pi"


#%% set CMIP version
try:
    cmip = int(sys.argv[2])
except:
    cmip = 6
# end try except
    

#%% set the model
try:
    mod = int(sys.argv[3])
except:
    mod = 49
# end try except


#%% set the experiment
try:
    exp = sys.argv[8]
except:    
    exp = "a4x"
# end try except


#%% set the ensemble
try:  # base experiment ensemble (usually piControl)
    ensemble_b = sys.argv[5]
except:
    ensemble_b = "r1i1p1f1"    
try:  # forced experiments ensemble (usually abrupt-4xCO2)
    ensemble_f = sys.argv[6]
except:
    ensemble_f = "r1i1p1f1"    
# end try except


#%% apply stratosphere mask? (1 == yes, 0 == no)
try:
    strat_mask = 1
except:
    strat_mask = int(sys.argv[4])
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


#%% set up the model list
if cmip == 5:
    import Namelists.Namelist_CMIP5 as nl
    models = nl.models
elif cmip == 6:
    import Namelists.Namelist_CMIP6 as nl
    models = nl.models
# end if elif


#%% set paths for the subprocess.run command
py_path = "/"
tats_script = "Temperature/Calc_Temp_Resp_KernelMethod_CSDiff.py"
q_script = "WaterVapour/Calc_WaterVapour_Resp_KernelMethod_CSDiff.py"
sfcalb_script = "SurfaceAlbedo/Calc_SurfaceAlbedo_Resp_KernelMethod_CSDiff.py"


#%% execute the subprocess.run command in for-loop
print("\n" + models[mod] + " " + k_n[kl] + " AS-CS kernels " + strat_m_str + "\n")

# print("\nCalculating surface albedo response " + models[mod] + "...\n")
subprocess.run(["python", py_path + sfcalb_script, str(mod), str(cmip), kl, k_st, ensemble_b, ensemble_f,
                 exp, toa_sfc])
print("\nCalculating temperature response " + models[mod] + "...\n")
subprocess.run(["python", py_path + tats_script, str(mod), str(cmip), kl, k_st, str(strat_mask), ensemble_b, 
                ensemble_f, exp, toa_sfc])
# print("\nCalculating water vapour response " + models[mod] + "...\n")
subprocess.run(["python", py_path + q_script, str(mod), str(cmip), kl, k_st, str(strat_mask), ensemble_b, 
                ensemble_f, exp, toa_sfc])

