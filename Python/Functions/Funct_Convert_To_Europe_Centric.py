"""
Funtion for converting the CMIP5 data and their longitudes to Europe centred
data
"""

# import modules
import numpy as np


def eu_centric(data, lon=None):
    
    """
    Returns in a list if lon is given of the Europe centred data as first 
    element and the Europe centred longitudes as second element.
    If lon is not given only the europe centred data are returned.
    Note that longitude has to be the last dimension!
    Should work for 2 or 3 dimensions.
    
    Parameters:
        :param data: 2 or 3 dimensional array of grid data.
        :param lon: 1 array containing the longitudes for the data array.
    
    """
    print("\nNote that longitude has to be the last dimension!\n")
    
    # rearange the data to Europe-centered        
    x_half = int(1/2 * np.shape(data)[-1])
    
    # make it dependent on the shape (2d/3d)
    if len(np.shape(data)) == 2:
        data_eu = data.copy()
        data_eu[:,:x_half] = data[:,x_half:]
        data_eu[:,x_half:] = data[:,:x_half]
    elif len(np.shape(data)) == 3:
        data_eu = data.copy()
        data_eu[:, :, :x_half] = data[:, :, x_half:]
        data_eu[:, :, x_half:] = data[:, :, :x_half]
    # end if elif

    # do the return
    if not lon is None:
        # change the longitude to Europe-centered
        lon_e = np.add(lon, -180)

        # do the return
        return([data_eu, lon_e])
    else:
        return(data_eu)
    # end if else

# end function euro_centric