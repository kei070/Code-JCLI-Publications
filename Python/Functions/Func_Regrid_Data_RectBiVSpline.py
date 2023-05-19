"""
Function for interpolating 2d model fields.

IMPORTANT: lat and lon cannot contain negative values!! So "transform them" to
           ranges 0:360 (lon) and 0:180 (lat).
"""

# import modules
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline


def remap(lon_t, lat_t, lon_o, lat_o, field_o, verbose=False):
    
    """
    Function for remapping a 2d array to a different lat-lon coordinate system.
    
    Parameters:
        :param lon_t: 1d array. Target longitudes in degrees in strictly ascending order. The values exactly 0 and 
                      exactly 360 are not allowed. If these values occur they will by changed by +0.00001 and -0.00001, 
                      respectively.
        :param lat_t: 1d array. Target longitudes in degrees in strictly ascending order. The values exactly 0 and 
                      exactly 180 are not allowed. If these values occur they will by changed by +0.00001 and -0.00001, 
                      respectively.
        :param lon_o: 1d array. Origin longitudes in degrees in strictly ascending order. The values exactly 0 and 
                      exactly 360 are not allowed. If these values occur they will by changed by +0.00001 and -0.00001, 
                      respectively.
        :param lat_o: 1d array. Origin longitudes in degrees in strictly ascending order. The values exactly 0 and 
                      exactly 180 are not allowed. If these values occur they will by changed by +0.00001 and -0.00001, 
                      respectively.     
        :param field_o: 2d array. The original value field to be remapped. The shape has to correspond to (lat, lon).
        :param verbose: Logical. If True some messages are printed. Defaults to False.
    """
    
    # if the lat bounds are exactly 0° and 90° this does not work so adapt the latitudes by subtracting a small value
    add = 0.001
    
    # origin grid
    if lat_o[0] == 0:
        if verbose:
            print("\nAdapting origin lat[0] by + & +" + str(add) + " to " + str(lat_o[0] + add) + "...\n")
        # end if verbose
        lat_o[0] = lat_o[0] + add
    # end if    
    if lat_o[-1] == 180:
        if verbose:
            print("\nAdapting origin lat[-1] by + & -" + str(add) + " to " + str(lat_o[-1] - add) + "...\n")
        # end if verbose
        lat_o[-1] = lat_o[-1] - add
    # end if
    if lon_o[0] == 0:
        if verbose:
            print("\nAdapting origin lon[0] by +" + str(add) + " to " + str(lon_o[0] + add) + "...\n")
        # end if verbose
        lon_o[0] = lon_o[0] + add
    # end if
    if lon_o[-1] == 360:
        if verbose:
            print("\nAdapting origin lon[-1] by +" + str(add) + " to " + str(lon_o[-1] - add) + "...\n")
        # end if verbose
        lon_o[-1] = lon_o[-1] - add
    # end if
    
    # target grid
    if lat_t[0] == 0:
        if verbose:
            print("\nAdapting target lat[0] by + & +" + str(add) + " to " + str(lat_t[0] + add) + "...\n")
        # end if verbose
        lat_t[0] = lat_t[0] + add
    # end if    
    if lat_t[-1] == 180:
        if verbose:
            print("\nAdapting target lat[-1] by + & -" + str(add) + " to " + str(lat_t[-1] - add) + "...\n")
        # end if verbose
        lat_t[-1] = lat_t[-1] - add
    # end if
    if lon_t[0] == 0:
        if verbose:
            print("\nAdapting target lon[0] by +" + str(add) + " to " + str(lon_t[0] + add) + "...\n")
        # end if verbose
        lon_t[0] = lon_t[0] + add
    # end if
    if lon_t[-1] == 360:
        if verbose:
            print("\nAdapting target lon[-1] by +" + str(add) + " to " + str(lon_t[-1] - add) + "...\n")
        # end if verbose
        lon_t[-1] = lon_t[-1] - add
    # end if
    
    # convert the coordinates to radians
    lon_o_r = lon_o / 180 * np.pi
    lon_t_r = lon_t / 180 * np.pi
    lat_t_r = lat_t / 180 * np.pi
    lat_o_r = lat_o / 180 * np.pi
    
    # mesh the lons and lats
    lats_t_r, lons_t_r = np.meshgrid(lat_t_r, lon_t_r)
    
    # get interpolation function (s=0 equals interpolation)
    lut = RectSphereBivariateSpline(lat_o_r, lon_o_r, field_o, s=0)
    
    # perform the interpolation
    field_t = lut.ev(lats_t_r.ravel(), lons_t_r.ravel()).reshape((len(lon_t), len(lat_t))).T
    
    # return the interpolated field
    return field_t
    
# end function remap()
    
