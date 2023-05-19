import os
import sys
import numpy as np

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Calc_Pres_at_HybridLevels import pres_hybrid_ccm


def dpres_hybrid_ccm(ps, p0, hyai, hybi):
    
    # calculate the pressure with the pres_hybid_ccm() function
    pres = pres_hybrid_ccm(ps, p0, hyai, hybi)
    
    # subtract the level n from level n+1 to get the differences
    if len(np.shape(pres)) == 3:
        return pres[1:, :, :] - pres[:-1, :, :]
    elif len(np.shape(pres)) == 4:
        return pres[:, 1:, :, :] - pres[:, :-1, :, :]
    # end if elif
    
# end function dpres_hybrid_ccm()
    