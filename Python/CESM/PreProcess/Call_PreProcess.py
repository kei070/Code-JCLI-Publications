"""
Script for calling all script that pre-process the CESM2 output for usage with plot scripts.
"""


#%% imports
import os
import sys
import subprocess
import numpy as np

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)


#%% set the CESM version and if the sea-ice area is to be calculated
#   as of now this is only used for the Extract_SeaIce script => only important if with_si is True
cesm = 2

# Calculate the sea-ice area?  If too many files have been opened this might not work...
with_si = False

# Is the case a control case? If yes, no delta-quantities will be calculated.
contr = False


#%% set the control case --> enter auto to determine it from the cesm parameter set above
case_con = "auto" 


#%% set the case
case = "dQ01yr_4xCO2"


#%% set the control case if case_con == "auto"
if case_con == "auto":
    if cesm == 1:
        case_con = "SOM_CAM5_cont"
        sta_yr = 51
        end_yr = 80
    elif cesm == 2:
        case_con = "Proj2_KUE"
        sta_yr = 51
        end_yr = 78
    # end if elif
# end if        
    

#%% adjust paths according to CESM version
# if CESM1: add = "_CESM1", if CESM2: add = ""
if cesm == 1:
    add = "_CESM1"
elif cesm == 2:    
    add = ""
# end if elif


#%% make some prints regarding the setup
print(f"\nmodel is CESM{cesm}\n")
print(f"case is {case}\n")
if not contr:
    print(f"control case is {case_con}\n")
    print(f"control case period is {sta_yr} to {end_yr}\n")
else:
    print("this is a control case --> no delta-quantities will be calculated\n")    
if with_si:
    print("sea ice will be handled")
else:
    print("sea ice will NOT be handled --> execute Call_PreProcess_SeaIce.py for that\n\n")
# end if else    
    

#%% call python scripts

# set paths for the subprocess.run command
extr_var = direc + "Extract_Variable.py"
calc_tom = direc + "/Calc_TOM_Imbalance.py"
calc_delta = "Calc_Delta_Quantities.py"
calc_glm = "GlobalMean_Quantities_2.py"
extr_2dvar_from3d = "Extract_Variable_2d_from_3d.py"
calc_lts = direc + "Calc_and_Store_LTS.py"
calc_si = direc + f"Extract_SeaIce{add}.py"
calc_si_area = "Calc_NH_SH_SeaIce_Extent.py"
calc_sfca = "Calc_SfcAlb.py"
calc_dmon = "Calc_Delta_Quantities_Monthly.py"
calc_plna = "Calc_PlanetaryAlbedo.py"
calc_cre = "Calc_CRE.py"
calc_dcre = "Calc_dCRE.py"


#%% call the extract variable procedure
var_l = ["TREFHT", "PS", "FSNS", "FSDS", "FSNT", "SOLIN", "FLNT", "FLNTC", 
         "FSNT", "FSNTC", "FLNS", "LHFLX", "SHFLX", "CLDTOT", "CLDLOW", "CLDMED", "CLDHGH"]
# var_l = ["TREFHT", "PS", "FSNS", "FSDS", "FSNT", "SOLIN", "FLNT", "FLNTC", "FSNT", "FSNTC", "FLNS", "LHFLX", "SHFLX"]
for var in var_l:
    print(f"Extracting {var} in case {case}...")
    subprocess.run(["python", extr_var, var, case])
# end for var    


#%% call the calculate TOM imbalance procedure
print("Calculate TOM imbalance...")
subprocess.run(["python", calc_tom, case, ""])
subprocess.run(["python", calc_tom, case, "C"])


#%% call the calculate delta quantites as well as the calculate global mean procedure
print("Calculate delta and global mean quantities...")
# var_s = ["TREFHT", "tom", "tomC", "sw", "lw", "swC", "lwC"]
var_s = ["TREFHT", "tom", "tomC", "sw", "lw", "swC", "lwC", "CLDTOT", "CLDLOW", "CLDMED", "CLDHGH"]
# var_s = ["swC", "lwC"]
for var_i in var_s:
    print(var_i)
    subprocess.run(["python", calc_glm, var_i, case])
    
    if not contr:
        subprocess.run(["python", calc_delta, var_i, "", case, case_con, str(sta_yr), str(end_yr)])
    # end if        
# end for var_i


#%% calculate the monthly delta for some quantities
if not contr:
    subprocess.run(["python", calc_dmon, "TREFHT", "", case, case_con, str(sta_yr), str(end_yr)])
    subprocess.run(["python", calc_dmon, "tom", "", case, case_con, str(sta_yr), str(end_yr)])
    subprocess.run(["python", calc_dmon, "tomC", "", case, case_con, str(sta_yr), str(end_yr)])
# end if


#%% calculate and store the surface and planetary albedo
print("Calculate surface and planetary albedo...")
subprocess.run(["python", calc_sfca, case])
if not contr:  # surface albedo change
    subprocess.run(["python", calc_dmon, "SfcAlb", "", case, case_con, str(sta_yr), str(end_yr)]) 
# end if
subprocess.run(["python", calc_plna, case])


#%% calculate CRE
subprocess.run(["python", calc_cre, case])

if not contr:
    subprocess.run(["python", calc_dcre, case])
# end if


#%% extract the 700 hPa temperature and calculate LTS
subprocess.run(["python", extr_2dvar_from3d, "T", case, "700"])
subprocess.run(["python", calc_lts, case])


#%% call the extract sea ice and calculate sea-ice area
if with_si:
    print("Extract sea ice and calculate sea-ice area...")
    subprocess.run(["python", calc_si, case, "aice"])
    subprocess.run(["python", calc_si_area, case, "aice"])
# end if 