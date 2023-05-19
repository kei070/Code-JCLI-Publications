"""
Attempt at "translating" the NCL function pattern_corr to Python.
See http://www.ncl.ucar.edu/Support/talk_archives/2011/att-2603/tst_pattern_cor2.ncl
"""

# imports
import numpy as np


def pattern_corr(x, y, w=None, opt=0):
    
    """
    Function for calculating the Pearson product-moment coefficient of linear correlation between two variables that are 
    respectively the values of the same variables at corresponding locations on two different maps. The function is 
    mostly adapted from the NCL function of the same name. See:
        https://www.ncl.ucar.edu/Document/Functions/Contributed/pattern_cor.shtml 
    
    Parameters:
        :param x: 2d- or 3d-Numpy array with last dimensions [..., lat, lon].
        :param y: 2d- or 3d-Numpy array. Must have same shape as x.
        :param w: Optional. Numpy array of weights. Must have same shape as x and y. Defaults to None.
        :param opt: Integer. opt=0 means a centered pattern correlations will be performed, i.e., the field means of x 
                    and y will be subtracted first. opt=1 means an uncentered pattern correlations will be performed, 
                    i.e., the field means of x and y will NOT be subtracted. Defaults to 0. 
    """
    
    # if no weights are given set the weights array to an array with the same shape as x and y but containing only values 
    # of 1 so that the average is calculated correctly below
    if w is None:
        w = np.ones(np.shape(x))
    # end if        
    
    # x and y must have same dimensions
    if np.ndim(x) != np.ndim(y):
        raise Exception("Error: Dimensions of x and y do not match!")
    # end if
    
    # if the arrays are 3d make sure the means are only taken for the last two dimensions
    ax = 0
    if np.ndim(x) == 3:
        ax = 1
    # end if        
    
    # centered correlation
    if opt == 0:
        x_avg_area = np.sum(x * w, axis=(0+ax, 1+ax)) / np.sum(w, axis=(0+ax, 1+ax))
        y_avg_area = np.sum(y * w, axis=(0+ax, 1+ax)) / np.sum(w, axis=(0+ax, 1+ax))
        
        if np.ndim(x) == 3:
            x_anom = x[:, :, :] - x_avg_area[:, None, None]
            y_anom = y[:, :, :] - y_avg_area[:, None, None]
        else:
            x_anom = x - x_avg_area
            y_anom = y - y_avg_area            
        # end if else                    
        
        xy_cov = np.sum(w * x_anom * y_anom, axis=(0+ax, 1+ax))
        x_anom2 = np.sum(w * x_anom**2, axis=(0+ax, 1+ax))
        y_anom2 = np.sum(w * y_anom**2, axis=(0+ax, 1+ax))
        
    # uncentered correlation
    elif opt == 1:
        xy_cov = np.sum(w * x * y, axis=(0+ax, 1+ax))
        x_anom2 = np.sum(w * x**2, axis=(0+ax, 1+ax))
        y_anom2 = np.sum(w * y**2, axis=(0+ax, 1+ax))
    # end if elif
    
    if np.any(x_anom2 > 0) & np.any(y_anom2 > 0):
        r = xy_cov / (np.sqrt(x_anom2) * np.sqrt(y_anom2))
        return r
    else:
        raise Exception("Error: Sum of squares < 0, division by zero.")
    # end if else        
        
# end function pattern_corr
