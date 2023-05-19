
import numpy as np

def flip(a, axis=None):
    """
    Function that flips an array along a given axis. The function uses the numpy.flip() but adds the functionality of NO
    flipping. It furthermore introduces the possibility to use names for the axis parameter. Note that the following 
    assumptions are made: axis="time" is index -4, axis="lev" is index -3, axis="lat" is index -2, axis="lon" is index 
    -1.
    """    
    if axis == "NA":
        return a
    else:
        
        if type(axis) is str:
            if np.ndim(a) == 4:
                # set up a dicitionary for the named axes
                axis_d = {"time":-4, "lev":-3, "lat":-2, "lon":-1}
                return np.flip(a, axis=axis_d[axis])
            elif np.ndim(a) == 3:
                if axis == "lev":
                    return a
                else:
                    # set up a dicitionary for the named axes
                    axis_d = {"time":-3, "lat":-2, "lon":-1}
                    return np.flip(a, axis=axis_d[axis])
            # end if elif
        else:
            return np.flip(a, axis=axis)
    # end if else
# end function flip
