import os
import sys
import numpy as np
from netCDF4 import Dataset

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Extract_Geographic_Region import extract_region


def region_mean(field, x1s, x2s, y1s, y2s, lat, lon, test_plt=False, plot_title="", multi_reg_mean=True):
    
    """
    Calculate the mean value over one or more regions.
    """
    
    regs = []
    for i in np.arange(len(x1s)):
        r_d = extract_region(x1s[i], x2s[i], y1s[i], y2s[i], lat, lon, field, test_plt=test_plt, plot_title=plot_title)
        if np.ndim(r_d[0]) == 2:
            # get the weights for averaging over the regions individually
            r_lons = lon[r_d[1][1][0, :]]
            r_lats = lat[r_d[1][0][:, 0]]
            r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
            r_latw = np.zeros(np.shape(r_d[0]))
            r_latw[:] = np.cos(r_latm / 180 * np.pi)[:, :]
        if np.ndim(r_d[0]) == 3:
            # get the weights for averaging over the regions individually
            r_lons = lon[r_d[1][1][0, :]]
            r_lats = lat[r_d[1][0][:, 0]]
            r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
            r_latw = np.zeros(np.shape(r_d[0]))
            r_latw[:] = np.cos(r_latm / 180 * np.pi)[None, :, :]
        if np.ndim(r_d[0]) == 4:
            # get the weights for averaging over the regions individually
            r_lons = lon[r_d[1][1][0, :]]
            r_lats = lat[r_d[1][0][:, 0]]
            r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
            r_latw = np.zeros(np.shape(r_d[0]))
            r_latw[:] = np.cos(r_latm / 180 * np.pi)[None, None, :, :]
        # end if
        """
        if type(field) is np.ma.core.MaskedArray:
            nyr = np.shape(r_d[0])[0]
            
            temp = np.reshape(r_d[0], (nyr, np.shape(r_d[0])[1] * np.shape(r_d[0])[2]))
            temp_lat = np.reshape(r_latw, (nyr, np.shape(r_d[0])[1] * np.shape(r_d[0])[2]))
            vals = temp[~temp.mask]
            vals = np.reshape(vals, (nyr, int(len(vals)/nyr)))
            lats = temp_lat[~temp.mask]
            lats = np.reshape(lats, (nyr, int(len(lats)/nyr)))
            regs.append(np.average(vals, weights=lats, axis=1))
        else:
            regs.append(np.average(r_d[0], weights=r_latw, axis=(-2, -1)))
        # end if else        
        """
        
        regs.append(np.average(r_d[0], weights=r_latw, axis=(-2, -1)))
    # end for i
    
    regs = np.array(regs)
    
    if multi_reg_mean:
        return np.mean(regs, axis=0)
    else:
        return regs
    # end if else
# end function region_mean()

"""
# set path to some file
data_path = "/media/kei070/Seagate/Uni/PhD/Tromsoe_UiT/Work/CMIP6/Data/CanESM5/"
f_name = "tas_Amon_CanESM5_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc"

# load the file
nc = Dataset(data_path + f_name)

# load the data, lat, and lon
field = nc.variables["tas"][0, :, :]
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]

# set up some coordinates
x1s = [0, 0]
x2s = [360, 360]
y1s = [20, -90]
y2s = [90, -20]


regs = []
for ri in np.arange(len(x1s)):
    r_d = extract_region(x1s[ri], x2s[ri], y1s[ri], y2s[ri], lat, lon, field)
    
    # get the weights for averaging over the regions individually
    r_lons = lon[r_d[1][1][0, :]]
    r_lats = lat[r_d[1][0][:, 0]]
    r_lonm, r_latm = np.meshgrid(r_lons, r_lats)
    r_latw = np.zeros(np.shape(r_d[0]))
    r_latw[:] = np.cos(r_latm / 180 * np.pi)[None, :, :]
    
    # calculate the mean over the regions
    regs.append(np.average(r_d[0], weights=r_latw, axis=(-2, -1)))
    
# end for ri

print(f"\n\n>20N: {regs[0]}   <20S: {regs[1]}   mean: {np.mean([regs[0], regs[1]])}")

# calculate with new function 
reg1 = region_mean(field, [x1s[0]], [x2s[0]], [y1s[0]], [y2s[0]], lat, lon)
reg2 = region_mean(field, [x1s[1]], [x2s[1]], [y1s[1]], [y2s[1]], lat, lon)
regs1_2 = region_mean(field, x1s, x2s, y1s, y2s, lat, lon)
print(f"\n\n>20N: {reg1}   <20S: {reg2}   mean: {regs1_2}")
"""