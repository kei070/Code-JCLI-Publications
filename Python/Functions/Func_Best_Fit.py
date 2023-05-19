"""
Function for detrending using numpy's polyfit function.
"""

# imports
import copy
import numpy as np


def best_fit(data, thres=15):
    
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
    Return:
        The polynomial fit found to be the best according to the criterion
        given in the thres parameter.
    """
    
    # get the "time series" against which to detrend; this is basically just
    # an integer array from zero to len(data)-1 
    ti = np.arange(0, len(data), 1)

    # perform the polynomial fit with increasing order in a for-loop to order 6
    fits = []
    stds = []
    for i in np.arange(1, 7, 1):
        
        # get the polynomial of i-th order
        pol = np.polyfit(ti, data, i)
        
        # generate the actual fit
        p = np.poly1d(pol)
        fit = p(ti)
        
        # add it to the fits list
        fits.append(fit)
        
        # detrend the data using the fit of i-th order
        detr = data - fit
        
        # get the std of the detr data and add them to the list of stds
        stds.append(np.std(detr))
        
    # end for i
    
    # convert the stds list into an array
    stds = np.array(stds)
    
    # find the minimum of the array
    min_ind = np.argmin(stds)
    
    """
    print("\nPolyfit order: " + str(min_ind + 1) + "\n")  # mind the start at 0
    print(stds)
    """
    
    # work the "way back up" the stds and check when the fit is only less than
    # [thres] % "better" than the one of lesser order
    min_ind_rec = copy.deepcopy(min_ind)  # establish a "recent" min_ind var.
    if min_ind > 0:
        for i in np.arange(1, min_ind + 1):
            std_test = ((stds[min_ind - i] - stds[min_ind_rec]) / 
                         stds[min_ind - i] * 100)
            
            """
            print(str(i) + " " + str(std_test))
            """
            
            if std_test < thres:
                min_ind_rec = min_ind - i
            # end if std_test
        # end for i
    # end if min_ind
    
    # get the corresponding fit from the fits array
    b_fit = fits[min_ind_rec]
    
    """
    print("\nPolyfit order after test: " + str(min_ind_rec + 1) + "\n")
    """
    
    # return the fit as well as its order
    return b_fit, min_ind_rec + 1
    
# end function best_fit()
    
    