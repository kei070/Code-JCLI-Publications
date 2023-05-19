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
        A list containing two elements: (1) p-value of the feedback change, (2) p-value of the ECS change.
    """
    
    sl_dif = np.zeros(5000)
    ecs_dif = np.zeros(5000)

    for i in np.arange(5000):
        
        tas_perm_e = np_perm(tas[:thres])
        toa_perm_e = np_perm(toa[:thres])
        tas_perm_l = np_perm(tas[thres:])
        toa_perm_l = np_perm(toa[thres:])
        
        sl_pe, yi_pe = lr(tas_perm_e, toa_perm_e)[:2]
        sl_pl, yi_pl = lr(tas_perm_l, toa_perm_l)[:2]
        
        ecs_dif[i] = yi_pl/sl_pl - yi_pe/sl_pe
        sl_dif[i] = sl_pl - sl_pe
        
    # end for i
    
    # calculate the actual difference, i.e., the value of which the significance is to be calculated
    sl_e, yi_e = lr(tas[:thres], toa[:thres])[:2]
    sl_l, yi_l = lr(tas[thres:], toa[thres:])[:2]
    
    dsl = sl_l - sl_e
    decs = yi_l/sl_l - yi_e/sl_e
    
    
    p_sl = perc_of_score(sl_dif, dsl)/100
    p_ecs = perc_of_score(ecs_dif, decs)/100
    if p_sl > 0.5:
        p_sl = 1 - p_sl
    # end if        
    if p_ecs > 0.5:
        p_ecs = 1 - p_ecs
    # end if        
    
    return p_sl, p_ecs
    
# end function monte_carlo()
