import numpy as np 


def equ_wvp(t):
    
    """
    Function for calculating the equilibirum water vapour pressure via the 
    formula by Buck (1996) (modified from Buck, 1981)
    
    Parameters:
        :param t: 2d or 3d array of air temperature. Unit: degree C.
    """

    # for liquid
    e_s = 0.61121 * np.exp((18.678 - t / 234.5) * (t / (257.14 + t)))
    
    # for ice
    e_si = 0.61115 * np.exp((23.036 - t / 333.7) * (t / (279.82 + t)))
    
    # fill in the values for ice where t < 273.15K
    e_s[np.where(t < 0)] = e_si[np.where(t < 0)]
    
    return e_s
    
# end equ_wvp()
