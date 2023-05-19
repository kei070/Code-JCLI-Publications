"""
Function for storing regression difference as nc files
"""

from netCDF4 import Dataset
import time as ti


def store_nc(out_path, var_list, lat, lon, var_names, desc="",
             lon_units = "degrees east", lat_units = "degrees north"):
    
    """
    Function for generating 2-dimensional nc file (lat, lon are the mandatory
    dimensions; see parameters).
    
    Parameters:
        :param out_path: String. Path to the directory where the file is to be
                         stored. Note that it has to contain the file name with
                         the .nc ending.
        :param var_list: List. Contains the 2d-arrays for the variables of the
                         .nc file. The arrays have to correspond to the 
                         [lat, lon] shape.
        :param lat: 1d-array. The latitudes of the variable arrays in var_list.
        :param lon: 1d-array. The longitudes of the variable arrays in 
                    var_list.
        :param var_names: List of strings. A list of with the names of the 
                          variables in var_list. Has to have the same length as
                          var_list.
        :param desc: String. The description if the .nc file. Defaults to empty
                     string.
        :param lon_units: String. The units of the longitudes. Defaults to 
                          "degrees east".
        :param lat_units: String. The units of the latitudes. Defaults to 
                          "degrees north".
        
    """
    
    # open the netcdf file
    f = Dataset(out_path, "w", format="NETCDF4")
    
    # start the list of dimensions; TODO: Add POSSIBLE (!) time axis
    var_dims = ["lon", "lat"]

    # specify dimensions, create, and fill the variables; 
    # also pass the units
    f.createDimension("lon", len(lon))
    f.createDimension("lat", len(lat))
    
    # build the variables
    longitude = f.createVariable("lon", "f8", "lon")
    latitude = f.createVariable("lat", "f8", "lat")

    # pass the remaining data into the variables
    longitude[:] = lon 
    latitude[:] = lat
    
    # flip the list and turn it into a tuple
    var_dims = tuple(var_dims[::-1])
    
    # set up the wind speed createVariable list, Note that it's the wrong way
    # round due to the appending and will later be flipped and then converted
    # to a tuple
    vars_nc = []
    for i in range(len(var_list)):
        
        # create the wind speed variable
        vars_nc.append(f.createVariable(var_names[i], "f4", var_dims))
        
        vars_nc[i][:] = var_list[i]
    
    # end for i
        
    # add attributes
    f.description = desc
    f.history = "Created " + ti.strftime("%d/%m/%y")

    longitude.units = lon_units
    latitude.units = lat_units

    # close the dataset
    f.close()

# end function