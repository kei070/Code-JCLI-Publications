"""
Call the calculation of all radiative responses for all available kernels for one model.

NOTE: 1: Possibility to choose clear-sky kernels not yet implemented!
      2: For BM13 only the pi version can be chosen as of yet! 
"""

#%% imports
import os
import sys
import subprocess
import timeit


#%% select a case
try:
    case = sys.argv[1]
except:
    case = "dQ01yr_4xCO2"
# end try except


#%% TOA response or surface response?  --> NOTE: surface response is not available for all kernels
try:
    toa_sfc = sys.argv[9]
except:
    toa_sfc = "TOA"
# end try except


#%% apply stratosphere mask? (1 == yes, 0 == no)
try:
    strat_mask = 1
except:
    strat_mask = int(sys.argv[3])
# end try except


#%% kernels
try:
    kern = sys.argv[7]
    kerns = [kern]
except:
    kerns = ["So08", "Sh08", "BM13", "H17", "P18", "S18"]
    # kerns = ["So08", "Sh08", "BM13"]
    kerns = ["Sh08"]
# end try  


#%% set paths for the subprocess.run command
py_path = "/"  # path to the python scripts
t_q_sa_script = "Call_T_Q_SfcA_Resp.py"
c_script = "Call_All_CloudResp.py"


#%% call the necessary scripts and issue print statements regarding what is being calculated
print(f"\n\n\n\nRunning kernel calculations for model CESM2-SOM case {case} from...\n\n\n")
# for case in ["Y41_dQ01", "Y61_dQ01", "Yr41_4xCO2", "Proj2_KUE_4xCO2_61"]:
for kl in kerns:
    print("\n\n" + kl + " kernels...\n\n")
    subprocess.run(["python", py_path + t_q_sa_script, case, kl, toa_sfc, str(0), str(strat_mask)])  # all-sky
    subprocess.run(["python", py_path + t_q_sa_script, case, kl, toa_sfc, str(1), str(strat_mask)])  # clear-sky
    subprocess.run(["python", py_path + c_script, case, kl, toa_sfc, str(strat_mask)])
# end for kl
# end for case    


