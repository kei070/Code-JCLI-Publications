"""
Function for performing regression with the function from scipy.
"""

# imports
import numpy as np
import progressbar as pg
from scipy import stats

def linear_reg_2d(x, y):
    
    """
    Function for calculating r^2 and slopes of a 2d array y on a 2d array x 
    using the LinearRegression function from the SciKit Learn module.
    
    Parameters:
        :param x: 2d array of floats. The "predictor" variable.
        :param y: 2d array of floats. The "explained" variable. Has to be of
                  exactly the same shape as x.
        
    Return:
        A dictionary with the elements "r_sq" and "slope", the r^2 (explained 
        variance) and slope of the linear model, respectively.
    """
    
    # set up the r^2 and slope arrays
    # r_sq = np.zeros(np.shape(x)[1:])
    sl = np.zeros(np.shape(x)[1:])
    
    # for i in pg.progressbar(range(np.shape(x)[2])):
    for i in range(np.shape(x)[2]):
        for j in range(np.shape(x)[1]):
    
            x_e = x[:, j, i]
            y_e = y[:, j, i]
            
            sl[j, i] = stats.linregress(x_e, y_e)[0]
            
        # end for j
    # end for i
    
    # set up the dictionary for the return
    out = dict()
    
    # out["r_sq"] = r_sq
    out["slope"] = sl
    
    return out
    
# end function linear_reg()    
    
    