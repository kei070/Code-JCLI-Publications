"""
Function for setting up colormap levels (can also be used as ticks in the colorbar) and norm for a diverging colormap.
"""

# imports
import os
import sys
import copy
import numpy as np
import matplotlib.colors as col
from matplotlib.pyplot import cm

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Round_to_5 import round_to_5
from Classes.Class_MidpointNormalize import MidpointNormalize


def set_levels(data, n_lev=10):
    
    """
    Parameters:
        :param data: Numpy array of arbitrary dimension.
        :param n_lev: Integer. Number of levels. Must be divisible by 2 since (in the current version) the number of 
                      levels > 0 equals the number of levels < 0. Defaults to 10.
    """
    
    # calculate half the number of levels
    n_lev_h = int(n_lev / 2)
    
    # calculate max and min
    c_max = np.max(data)
    c_min = np.min(data)
    
    # calculate the midpoint
    # mid = (c_min + c_max) / 2
    
    if c_min > -2.5:
        c_min_l = -0.1
        
        # set the levels < 0 accordingly
        levels_n = np.array([])
    else:
        c_min_l = round_to_5(c_min)
        
        # set the levels < 0 accordingly
        levels_n = np.linspace(c_min_l, 0, n_lev_h)

        # make sure that the first element is larger 0
        if levels_n[-2] < -1:
            levels_n[-1] = -1
        else:
            levels_n = levels_n[:-1]
        # end if else

    if c_max < 2.5:
        c_max_l = 0.1
        
        # set the levels > 0 accordingly
        levels_p = np.array([])
    else:
        c_max_l = round_to_5(c_max)
    
        # set the levels > 0 accordingly
        levels_p = np.linspace(0, c_max_l, n_lev_h)
        
        # make sure that the first element is larger 0
        if levels_p[1] > 1:
            levels_p[0] = 1
        else:
            levels_p = levels_p[1:]
        # end if else
        
    # end if else
    
    # make sure that the colour interval around 0 is symmetric
    if np.abs(levels_p[0]) > np.abs(levels_n[-1]):
        levels_p[0] = -levels_n[-1]
    else:
        levels_n[-1] = -levels_p[0]
    # end if else
    
    # set up and return the level; also construct and return the norm
    return np.concatenate([levels_n, levels_p]), MidpointNormalize(vmin=c_min_l, midpoint=0, vmax=c_max_l)

# end function set_levels()



def set_levs_norm_colmap(data_in, n_lev=10, c_min=None, c_max=None, round_dir="", factor=-1, quant=1):
    
    """
    Parameters:
        :param data: Numpy array of arbitrary dimension.
        :param n_lev: Integer. Number of levels. Must be divisible by 2 since (in the current version) the number of 
                      levels > 0 equals the number of levels < 0. Defaults to 10.
        :param c_min: Float. Minimum for the colormap. Defaults to None where the function itself gauges the value from 
                      from the data.             
        :param c_max: Float. Maximum for the colormap. Defaults to None where the function itself gauges the value from 
                      from the data.                      
        :param round_dir: String. Used in the round_to_5() function. "Rounding direction" with the three possible values 
                          "off" for rounding off to the next lower value divisible by 5, "up" of rounding up, and "" 
                          (empty string) for rounding normally. Note that 'lower' here means lower with respect to the 
                          absolute value.
        :param factor: Integer. Factor by which the data are to be multiplied before they are rounded to the next 
                       multiple of 5. The point of this functionality is to just have one function that rounds positive
                       numbers to the next multiple of 5 and have it aplicable for e.g. intervals [-1, 0], simply be 
                       multiplying this interval by a factor of 10. Not that if the factor is set to -1 (default) the 
                       function will automatically check the value interval and introduce the factor accordingly.
        :param quant: Float. Between 0 and 1 (inclusive). The quantile of the data that is to be considered the maximum
                      value of the data for the colormap. Defaults to 1, meaning that the maximum value of the array is
                      chosen.
    """
    
    # copy the data to be sure to not mask any values
    data = copy.deepcopy(data_in)
    
    # calculate half the number of levels
    n_lev_h = int(n_lev / 2)
    
    # if c_min or c_max are not given
    readj = False
    if (c_min is None) | (c_max is None):
        
        # set the the readj parameter to True so that the values will later be readjusted by the factor
        readj = True
        
        # make sure the data are a normal numpy array with NaNs as invalid values
        if type(data) == np.ma.core.MaskedArray:
            data_m = data.mask
            data = np.array(data)
            data[data_m] = np.nan
        # end if
        
        # get the quantile according to the parameter quant
        quant_max = np.nanquantile(a=np.abs(data), q=quant)
        
        # set all the values (absolute values!) larger than the values corresponding to quant to NaN
        data[np.abs(data) > quant_max] = np.nan
        
        # if there is no factor different from 1 given try to adapt it automatically
        if factor == -1:
            if np.nanmax(np.abs(data)) >= 5:
                factor = 1        
            if np.nanmax(np.abs(data)) < 5:
                factor = 10
            if np.nanmax(np.abs(data)) < 0.5:
                factor = 100
            if np.nanmax(np.abs(data)) < 0.05:
                factor = 1000
            if np.nanmax(np.abs(data)) < 0.005:
                factor = 10000
            if np.nanmax(np.abs(data)) < 0.0005:
                factor = 100000
            if np.nanmax(np.abs(data)) < 0.00005:
                factor = 1000000
            # end if
        # end if
        
        # multiply the data by the given factor
        data = data * factor
        
        # calculate max and min
        c_max = np.nanmax(data)
        c_min = np.nanmin(data)
    
    else:
        # set the the readj parameter to True so that the values will later be readjusted by the factor
        readj = True

        # if there is no factor different from 1 given try to adapt it automatically
        if factor == -1:
            if np.nanmax(np.abs(np.array([c_min, c_max]))) >= 5:
                factor = 1        
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 5:
                factor = 10
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 0.5:
                factor = 100
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 0.05:
                factor = 1000
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 0.005:
                factor = 10000
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 0.0005:
                factor = 100000
            if np.nanmax(np.abs(np.array([c_min, c_max]))) < 0.00005:
                factor = 1000000
            # end if
        # end if
        
        # multiply the data as well as the given min and max by the given factor
        data = data * factor
        c_min = c_min * factor
        c_max = c_max * factor
    # end if else        
            
    # if c_max rounds to 0 just use the Blues colourmap and if c_min rounds to 0 use Reds
    if c_min > -2.5:
        print("\nRounding the minimum yields zero so the plot uses the Reds colourmap with " + str(n_lev) + " levels\n")
        levels = np.linspace(0, round_to_5(c_max, round_dir), n_lev) / factor
        return levels, None, cm.get_cmap('Reds', n_lev)
    if c_max < 2.5:
        print("\nRounding the minimum yields zero so the plot uses the Blues colourmap with " + str(n_lev) + " levels\n")
        levels = np.linspace(round_to_5(c_min, round_dir), 0, n_lev) / factor
        return levels, None, cm.get_cmap('Blues_r', n_lev)
    # end if
    

    c_min_l = round_to_5(c_min, round_dir)
    
    # set the levels < 0 accordingly
    levels_n = np.linspace(c_min_l, 0, n_lev_h)

    # make sure that the first element is larger 0
    if levels_n[-2] < -1:
        levels_n[-1] = -1
    else:
        levels_n = levels_n[:-1]
    # end if else
    
    c_max_l = round_to_5(c_max, round_dir)

    # set the levels > 0 accordingly
    levels_p = np.linspace(0, c_max_l, n_lev_h)
    
    # make sure that the first element is larger 0
    if levels_p[1] > 1:
        levels_p[0] = 1
    else:
        levels_p = levels_p[1:]
    # end if else
    
    # make sure that the colour interval around 0 is symmetric
    if np.abs(levels_p[0]) > np.abs(levels_n[-1]):
        levels_p[0] = -levels_n[-1]
    else:
        levels_n[-1] = -levels_p[0]
    # end if else
    
    # generate a new colourmap that will respect the asymmetry of the range by adjusting the colour intensity
    top = cm.get_cmap('Reds', 128)
    bottom = cm.get_cmap('Blues_r', 128)
    
    # get the ratios to be able to adjust the colour intensity
    max_min = np.abs(c_max_l / c_min_l)
    min_max = np.abs(c_min_l / c_max_l)
    
    # set the larger than 1 ratio to 1
    if max_min > 1:
        max_min = 1
    if min_max > 1:
        min_max = 1    
    # end if
    
    newcolors = np.vstack((bottom(np.linspace(1 - min_max, 1, 128)), top(np.linspace(0, max_min, 128))))
    newcmp = col.ListedColormap(newcolors, name='RedBlue')
    
    # readjust the max and min values for the midpoint normalisation by the given factor
    # if c_min or c_max are given this is not necessary!!
    if readj:
        c_min_l = c_min_l / factor
        c_max_l = c_max_l / factor
        levels_n = levels_n / factor
        levels_p = levels_p / factor
    # end if
    
    # set up and return the level; also construct and return the norm
    # levels = np.flip(np.concatenate([levels_n, levels_p]))
    levels = np.concatenate([levels_n, levels_p])
    return levels, MidpointNormalize(vmin=c_min_l, midpoint=0, vmax=c_max_l), newcmp, newcolors

# end function set_levs_norm_colmap()