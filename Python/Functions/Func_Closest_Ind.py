"""
Function for searching the index in an array of the value closest to a given value.
"""

import numpy as np

def closest_ind(val, a):
    
    return np.argmin(np.abs(a - val))

# end function closest_ind()