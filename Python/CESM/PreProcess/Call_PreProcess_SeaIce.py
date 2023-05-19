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
cesm = 2


#%% set the case
case = "dQ01yr_4xCO2"


#%% adjust paths according to CESM version
# if CESM1: add = "_CESM1", if CESM2: add = ""
if cesm == 1:
    add = "_CESM1"
elif cesm == 2:    
    add = ""
# end if elif    


#%% call python scripts

# set paths for the subprocess.run command
calc_si = f"Extract_SeaIce{add}.py"
calc_si_area = "Calc_NH_SH_SeaIce_Extent.py"


#%% call the extract sea ice and calculate sea-ice area
print("Extract sea ice and calculate sea-ice area...")
subprocess.run(["python", calc_si, case, "aice"])
subprocess.run(["python", calc_si_area, case, "aice"])
