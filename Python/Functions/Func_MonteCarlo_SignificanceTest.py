"""
Monte Carlo significance test 
"""

import numpy as np
from numpy.random import permutation as np_perm
from scipy.stats import percentileofscore as perc_of_score
from scipy.stats import linregress as lr

def monte_carlo(tas, toa, thres=20, perm=5000):
    
    """
    Parameters:
        tas    first array
        toa    second array
        thres  index at which the time series is to be separated (i.e. the year which separates early and late period);
               defaults to 20
        perm   number of perutations to be performed; defaults to 5000
    
    Return:
        The p-value of the slope change.
    """
    
    sl_dif = np.zeros(5000)

    for i in np.arange(5000):
        
        tas_perm_e = np_perm(tas[:thres])
        toa_perm_e = np_perm(toa[:thres])
        tas_perm_l = np_perm(tas[thres:])
        toa_perm_l = np_perm(toa[thres:])
        
        sl_pe = lr(tas_perm_e, toa_perm_e)[0]
        sl_pl = lr(tas_perm_l, toa_perm_l)[0]
        
        sl_dif[i] = sl_pl - sl_pe
        
    # end for i
    
    # calculate the actual difference, i.e., the value of which the significance is to be calculated
    score = lr(tas[thres:], toa[thres:])[0] - lr(tas[:thres], toa[:thres])[0]
    
    p_score = perc_of_score(sl_dif, score)/100
    if p_score > 0.5:
        p_score = 1 - p_score
    # end if        

    return p_score
    
# end function monte_carlo()
