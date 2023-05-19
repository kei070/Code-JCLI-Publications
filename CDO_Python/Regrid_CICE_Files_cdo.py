"""
Regrid the CESM CICE output to n80 grid. Only regrids variable aice.
"""

import subprocess
import os
import glob

# set the case
case = "dQ01yr_4xCO2"

# ser variable name
var = "aice"

# set path
data_path = ""
out_path = ""
os.makedirs(out_path, exist_ok=True)

# print(data_path)
# raise Exception

# load the file list
f_list = sorted(glob.glob(data_path + "*.nc"), key=str.casefold)

# print(f_list[0])

# generate the remapweights
subprocess.call(["cdo", "genbil,n80", "-selname," + var, f_list[0], out_path + "remapweights.nc"])

for f in f_list:
    
    # generate output file name
    f_out = case + "_" + var + "_" + f[-10:-3] + ".nc"

    print(f"remap,n80,{data_path}remapweights.nc")

    subprocess.call(["cdo", f"remap,n80,{out_path}remapweights.nc", "-selname," + var, f, out_path + f_out])
# end for f



