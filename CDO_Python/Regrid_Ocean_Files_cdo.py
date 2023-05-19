"""
Regrid the CMIP sic or siconc to n80 grid.
"""

import subprocess
import os
import glob

# set model
mod = "SAM0-UNICON"
cmip = "CMIP6"
ens = ""  # "_r1i1p1f1"

if (ens == "_r1i1p1f1") | (ens == "_r1i1p1"):
    ens = ""
# end if

# set the variable
var = "siconc"
if ((var == "sic") & (cmip == "CMIP6")):
    var = "siconc"
# end if

# set path
data_path = ""
data_path = ""

# print(data_path)
# raise Exception

# load the file list
f_list = sorted(glob.glob(data_path + "*.nc"), key=str.casefold)

# print(f_list[0])

# generate the remapweights
subprocess.call(["cdo", "genbil,n80", f_list[0], data_path + "remapweights.nc"])

# generate the directory
str_list = f_list[0].split("/")
path = "/".join(str_list[:-1])
os.makedirs(path + "_rg", exist_ok=True)

# loop over the files and regrid
for f in f_list:
    str_list = f.split("/")
    path = "/".join(str_list[:-1])
    f_out_list = str_list[-1].split("_")
    f_out = "_".join(f_out_list[:-2]) + "_n80_" + f_out_list[-1]

    print(f"remap,n80,{data_path}remapweights.nc")

    subprocess.call(["cdo", f"remap,n80,{data_path}remapweights.nc", f, path + "_rg/" + f_out])
# end for f



