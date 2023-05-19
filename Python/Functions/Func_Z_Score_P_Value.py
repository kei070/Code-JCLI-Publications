"""
Calculate p-value of the z-score for two linear regression slopes.
"""

import numpy as np
from scipy.stats import linregress as lr
from scipy.stats import norm as sci_norm

def z_score(x1, y1, x2, y2):
    
    sl1, yint1, r1, p1 = lr(x1, y1)[:4]
    sl2, yint2, r2, p2 = lr(x2, y2)[:4]    
    
    n1 = len(x1)
    n2 = len(x2)
    predicted1 = sl1 * x1 + yint1
    predicted2 = sl2 * x2 + yint2
    del_predicted1 = y1 - predicted1
    del_predicted2 = y2 - predicted2
    var_predictor1 = x1 - np.mean(x1)
    var_predictor2 = x2 - np.mean(x2)
    
    ste1 = (1/(n1 - 2) * np.sum(del_predicted1**2) / np.sum(var_predictor1**2))**0.5
    ste2 = (1/(n2 - 2) * np.sum(del_predicted2**2) / np.sum(var_predictor2**2))**0.5
    
    z = (sl1 - sl2) / (ste1**2 + ste2**2)**0.5
    
    if z < 0:
        return sci_norm.cdf(z)
    else:
        return 1 - sci_norm.cdf(z)
    # end if else
    
# end function z_score()    