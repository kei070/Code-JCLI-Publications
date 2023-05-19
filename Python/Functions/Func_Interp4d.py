"""
1d interpolation for 3d and 4d arrays.
"""

# imports
import numpy as np
from scipy import interpolate


# 1d interpolation for 3d array
def interp3d(coords, vals, interp_coords=None, kind="linear"):
    
    """
    coords can only be either 1d or 3d! interp_coords can (for now) only be 1d!
    """    
    
    if np.ndim(coords) == 1:
        
        # if no interpolation coordinates are given assume that they are the same as the origin coordinates
        if interp_coords is None:
            interp_coords = coords
        # end if

        interp_vals = np.zeros((len(interp_coords), np.shape(vals)[-2], np.shape(vals)[-1]))
        
        for y in range(np.shape(vals)[-2]): 
            for x in range(np.shape(vals)[-1]):
                
                # find the index for the lowest pressure level with data
                sta_ind = np.sum((np.array(vals[:, y, x]) == vals.fill_value) | np.isnan(np.array(vals[:, y, x])))
                # print(sta_ind)
                
                interp_f = interpolate.interp1d(coords[sta_ind:],
                                                np.array(vals[sta_ind:, y, x]),
                                                kind=kind, 
                                                fill_value="extrapolate")
                interp_vals[:, y, x] = interp_f(np.array(interp_coords))
            
            # end for x
        # end for y
    else:
        interp_vals = np.zeros((len(interp_coords), np.shape(vals)[-2], np.shape(vals)[-1]))
        
        for y in range(np.shape(vals)[-2]): 
            for x in range(np.shape(vals)[-1]):
                
                # find the index for the lowest pressure level with data
                sta_ind = np.sum((np.array(vals[:, y, x]) == vals.fill_value) | np.isnan(np.array(vals[:, y, x])))
                # print(sta_ind)
                
                interp_f = interpolate.interp1d(coords[sta_ind:, y, x],
                                                np.array(vals[sta_ind:, y, x]),
                                                kind=kind, 
                                                fill_value="extrapolate")
                interp_vals[:, y, x] = interp_f(np.array(interp_coords))
            
            # end for x
        # end for y
    
    return interp_vals
# end function interp3d()


# 4d interpolation function which just calls interp3d()
def interp4d(t, coords, vals, interp_coords=None, log_co=False, kind="linear"):
    
    """
    coords can only be either 1d or 4d! interp_coords can (for now) only be 1d!
    """
    
    # if no interpolation coordinates are given assume that they are the same as the origin coordinates
    if interp_coords is None:
        interp_coords = coords
    # end if
    
    if log_co:
        coords = np.log(coords)
        interp_coords = np.log(interp_coords)
    # end if    
    
    if np.ndim(coords) == 1:
        return interp3d(coords, vals[t, :, :, :], interp_coords, kind=kind)
    else:
        return interp3d(coords[t, :, :, :], vals[t, :, :, :], interp_coords, kind=kind)
    # end if else
    
# end function interp4d()