"""
Calculate correlation (probably Pearson).
See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
"""
import numpy as np

def corr_coef(x=None, x_mean=None, y=None, y_mean=None):
    """
    Function for calculating the Pearson correlation coefficient.
    """
    
    # check if the means are given and calculate them if not
    if x_mean is None:
        x_mean = np.mean(x)
    if y_mean is None:
        y_mean = np.mean(y)
    # end if
    
    corr = (np.sum((x - x_mean) * (y - y_mean)) / 
            (np.sqrt(np.sum(np.square(x - x_mean))) * 
             np.sqrt(np.sum(np.square(y - y_mean)))))
    
    return corr

# end function corr_coef()   
    