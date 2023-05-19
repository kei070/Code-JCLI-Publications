"""
Call the necessary scripts to calculate the cloud radiative response via the kernel adjustment method.
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
    kl = "So08"
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


#%% SFC suffix for CRE script
sfc_add = ""
if toa_sfc == "SFC":
    sfc_add = "_SFC"
# end if    


#%% adjust the kernel name

# kernels names for keys
k_k = ["Sh08", "So08", "BM13", "H17", "P18", "S18"]

# kernel names for plot titles etc.
k_n = {"Sh08":"Shell", "So08":"Soden", "BM13":"Block&Mauritsen", "H17":"Huang", "P18":"Pendergrass", "S18":"Smith"}
k_p = {"Sh08":"Shell_etal_2008", "So08":"Soden_etal_2008", "BM13":"Block_Mauritsen_2013", "H17":"Huang_etal_2017", 
       "P18":"Pendergrass_etal_2018", "S18":"Smith_etal_2018"}
    

#%% set paths for the subprocess.run command
py_path = "/"
cre_path = "../../PreProcess/"
tats_script = "/Temperature/Calc_Temp_Resp_KernelMethod_CSDiff.py"
q_script = "WaterVapour/Calc_WaterVapour_Resp_KernelMethod_CSDiff.py"
sfcalb_script = "SurfaceAlbedo/Calc_SurfaceAlbedo_Resp_KernelMethod_CSDiff.py"
dcre_script = f"Calc_dCRE{sfc_add}.py"
cld_script = "Cloud/Calc_CloudResp_KernelMethod.py"


#%% execute the subprocess.run command
print("\n" + k_n[kl] + " " + k_st + " kernels " + strat_m_str + "\n")

print("\nCalculating surface albedo adjustment (difference clear-sky minus all-sky kernel)...\n")
subprocess.run(["python", py_path + sfcalb_script, case, kl, k_st, toa_sfc])
print("\nCalculating temperature adjustment (difference clear-sky minus all-sky kernel)...\n")
subprocess.run(["python", py_path + tats_script, case, kl, k_st, str(strat_mask), toa_sfc])
print("\nCalculating water vapour adjustment (difference clear-sky minus all-sky kernel)...\n")
subprocess.run(["python", py_path + q_script, case, kl, k_st, str(strat_mask), toa_sfc])
print("\nCalculating CRE...\n")
subprocess.run(["python", cre_path + dcre_script, case])
print("\nCalculating adjusted cloud radiative response...\n")
subprocess.run(["python", py_path + cld_script, case, kl, k_st, toa_sfc])
    