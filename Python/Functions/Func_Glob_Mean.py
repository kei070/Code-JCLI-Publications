import numpy as np


def glob_mean(field, lat, lon=None, from_zon=False):
    
    """
    Function for calculating global mean values of a given field.
    
    Parameters:
        :param field: Input numpy array to be globally averaged. Dimensions can be [time, lat, lon], [lat, lon],
                      [time, lat], [lat]. For the latter two from_zon=True is required,
        :param lat: Latitudes as numpy array
        :param lon: Longitudes as numpy array. Not required if from_zon=True.
        :param from_zon: Logical. If True, the input field is expected to be a zonal average with dimension -1 being 
                         latitude. Defaults to zero.
    
    Return: Global mean weighted according to the given latitudes.
    """
    
    print("glob_mean() DOES NOT WORK CORRECTLY WITH MASKED ARRAYS YET!")
    
    weights = np.zeros(np.shape(field))
    if not from_zon:
        lons, lats  = np.meshgrid(lon, lat)
        
        if len(np.shape(field)) == 4:
            weights[:, :, :, :] = np.cos(lats / 180 * np.pi)[None, None, :, :]        
        if len(np.shape(field)) == 3:
            weights[:, :, :] = np.cos(lats / 180 * np.pi)[None, :, :]
        elif len(np.shape(field)) == 2: 
            weights[:, :] = np.cos(lats / 180 * np.pi)
        # end if elif    
        
        result = np.average(field, weights=weights, axis=(-2, -1))
        
    else:
        if len(np.shape(field)) == 2:
            weights[:, :] = np.cos(lat / 180 * np.pi)[None, :]
        else:
            weights[:] = np.cos(lat / 180 * np.pi)
        # end if else
        
        result = np.average(field, weights=weights, axis=-1)
        
    # end if else            
            
    return result
    
# end function glob_mean()
    