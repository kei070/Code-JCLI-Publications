"""
Function for calculating a climatology over a monthly mean data array.
"""

# imports
import numpy as np


def climatology(field, with_std=False):
    """
    Function for calculating a climatology over a monthly mean data array. Works for array of dimensionality 1 to 4. The
    time axis has to be the first dimension.
    """
    
    if np.ndim(field) == 1:
        time_len = int(len(field)/12)
        if with_std:
            return {"mean":np.mean(np.reshape(field, (time_len, 12)), axis=0), 
                    "std":np.std(np.reshape(field, (time_len, 12)), axis=0)}
        else:
            return np.mean(np.reshape(field, (time_len, 12)), axis=0)
        # end if else
    elif np.ndim(field) == 2:
        time_len = int(np.shape(field)[0]/12)
        d1_len = np.shape(field)[1]
        if with_std:
            return {"mean":np.mean(np.reshape(field, (time_len, 12, d1_len)), axis=0),
                    "std":np.std(np.reshape(field, (time_len, 12, d1_len)), axis=0)}
        else:
            return np.mean(np.reshape(field, (time_len, 12, d1_len)), axis=0)
        # end if else 
    elif np.ndim(field) == 3:
        time_len = int(np.shape(field)[0]/12)
        d1_len = np.shape(field)[1]
        d2_len = np.shape(field)[2]
        if with_std:
            return {"mean":np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len)), axis=0),
                    "std":np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len)), axis=0)}
        else:        
            return np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len)), axis=0)
        # end if else 
    elif np.ndim(field) == 4:
        time_len = int(np.shape(field)[0]/12)
        d1_len = np.shape(field)[1]
        d2_len = np.shape(field)[2]
        d3_len = np.shape(field)[3]
        if with_std:
            return {"mean":np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len, d3_len)), axis=0),
                    "std":np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len, d3_len)), axis=0)}
        else:        
            return np.mean(np.reshape(field, (time_len, 12, d1_len, d2_len, d3_len)), axis=0)
        # end if else 
    # end if elif
    
# end function
    