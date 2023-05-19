import numpy as np


def pres_hybrid_ccm(ps, p0, hyam, hybm):
    
    """
    Function corresponding to the equally named function of ncl. See the URL
    https://www.ncl.ucar.edu/Document/Functions/Built-in/pres_hybrid_ccm.shtml
        
    Parameters:
        :param ps: Surface pressure array. Either 2d or 3d.
        :param p0: Scalar; the surface reference pressure.
        :param hyam: 1d-array of of "hybrid A" coefficients. Must be unitless.
        :param hybm: 1d-array of of "hybrid B" coefficients. Must be unitless.
        
    Return:
        Array with number of dimensions equal to dim(ps)+1 containing the 
        mid-level hybrid pressure.
    """
    
    # get the number of dimensions of ps
    ps_dims = len(np.shape(ps))
    
    # establish the result dimensions
    if ps_dims == 2:
        # dimensions: (lev, lat, lon)
        targ_dims = (len(hyam), np.shape(ps)[0], np.shape(ps)[1])
    elif ps_dims == 3:
        # first dimension is time
        targ_dims = (np.shape(ps)[0], len(hyam), np.shape(ps)[1], 
                     np.shape(ps)[2])
    # end if elif
    
    # expand the arrays to have the same dimensions
    ps_exp = np.zeros(targ_dims)
    hyam_exp = np.zeros(targ_dims)
    hybm_exp = np.zeros(targ_dims)
    
    if ps_dims == 2:
        ps_exp[:, :, :] = ps[None, :, :]
        hyam_exp[:, :, :] = hyam[:, None, None]
        hybm_exp[:, :, :] = hybm[:, None, None]
    elif ps_dims == 3:
        ps_exp[:, :, :, :] = ps[:, None, :, :]
        hyam_exp[:, :, :, :] = hyam[None, :, None, None]
        hybm_exp[:, :, :, :] = hybm[None, :, None, None]
    # end if elif
        
    # execute the calculation and return the result
    return hyam_exp * p0 + hybm_exp * ps_exp
    
# end function pres_hybrid_ccm()
  