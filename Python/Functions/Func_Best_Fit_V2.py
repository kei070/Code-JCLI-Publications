"""
Function for detrending using numpy's polyfit function.
"""

# imports
import copy
import numpy as np


def best_fit(data, thres=15, max_o = 6):
    
    """
    Function for detrending using numpy's polyfit function.
    
    Parameters:
        :param data: 1d array of values to detrend.
        :param thres: Integer. Threshold value in percent about which the fit 
                      of the next higher order has to be "better" than the 
                      current one. "Better" here means that the standard 
                      deviation if the detrended values reduces by at least 
                      [thres] % when the order of the polynomial fit is 
                      increased by one. Defaults to 15.
        :param max_o: Integer. Maximum order of the polyfit.
    Return:
        The polynomial fit found to be the best according to the criterion
        given in the thres parameter.
    """
    
    # get the "time series" against which to detrend; this is basically just
    # an integer array from zero to len(data)-1 
    ti = np.arange(0, len(data), 1)

    # perform the polynomial fit with increasing order in a while-loop
    # ----------------------------------------------------------------
    
    # set the first polynomial order
    i = 1
    
    # get the polynomial of 1st order
    pol = np.polyfit(ti, data, i)
    
    # generate the actual fit
    p = np.poly1d(pol)
    fit = p(ti)
        
    # detrend the data using the fit of i-th order
    detr = data - fit
        
    # get the std of the detr data
    fit_std = np.std(detr)
    
    # set the first value for the "improvement" to check against the threshold
    improve = 100
    
    while (improve > thres) & (i < max_o):
        
        # increment order of the fit
        i += 1
        
        # get the polynomial of i-th order
        pol = np.polyfit(ti, data, i)
        
        # generate the actual fit
        p = np.poly1d(pol)
        fit = p(ti)
        
        # detrend the data using the fit of i-th order
        detr = data - fit
        
        # get the std of the detr data and add them to the list of stds
        fit_std_new = np.std(detr)
        
        # get the "improvement" in percent
        improve = ((fit_std - fit_std_new) / fit_std * 100)
        # print(improve)
        
        # set the new std to the old one
        fit_std = copy.deepcopy(fit_std_new)
        
    # end while
    
    return fit, i
    
# end function best_fit()
    
    