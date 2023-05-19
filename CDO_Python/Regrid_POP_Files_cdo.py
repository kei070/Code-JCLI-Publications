"""
Script for regridding the POP input file for CESM2-SOM to the atmosphere grid (96x144).
"""

import subprocess
import os
import glob

# ser variable name
var = "qdp"

# set path
data_path = ""
out_path = ""
os.makedirs(out_path, exist_ok=True)

# print(data_path)
# raise Exception

# load the file list
f_list = sorted(glob.glob(data_path + "pop_frc.gx1v6.210317_coord_dQ.nc"), key=str.casefold)

# print(f_list[0])

# generate the remapweights
subprocess.call(["cdo", "genbil,r96x144", "-selname," + var, f_list[0], out_path + "remapweights.nc"])

# loop over the files and regrid
for f in f_list:
    
    # generate output file name
    f_out = var + "_r96x144_dQ.nc"

    print(f"remap,n80,{data_path}remapweights.nc")

    subprocess.call(["cdo", f"remap,r96x144,{out_path}remapweights.nc", "-selname," + var, f, out_path + f_out])
# end for f



