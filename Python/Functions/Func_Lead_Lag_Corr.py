"""
The dumb-man's instantiation of a lead-lag correlation.
"""

import numpy as np
from scipy.stats import linregress as lr

def lead_lag_corr(sig1, sig2, max_lead_lag=10):
    
    """
    Superficial implementation of a lead-lag correlation using scipy's stats.linregress R correlation coefficient.
    Parameters:
        sig1: The signal on which sig2 is regressed; i.e., the x-axis on the scatter-plot.
        sig2: The signal which is regressed on sig1; i.e., the y-axis on the scatter-plot.
        max_lead_lag: Integer. The maximum length of the lead and lag in given units.
    """
    
    r_s1 = []

    for i in np.arange(1, max_lead_lag):
        if i == 0:
            j = None
        else:
            j = -i
        # end if else
        
        r_s1.append(lr(sig2[i:], sig1[:j])[2])
    
    # end for i
    r_s1 = np.array(r_s1)[::-1]
    
    
    r_s2 = []
    for i in np.arange(max_lead_lag):
        if i == 0:
            j = None
        else:
            j = -i
        # end if else
        
        r_s2.append(lr(sig1[i:], sig2[:j])[2])
    
    # end for i
    
    return np.concatenate([r_s1, np.array(r_s2)])
    
# end def    