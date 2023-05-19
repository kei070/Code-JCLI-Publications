import os
import sys
import numpy as np

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

# from Functions.Func_EquWVPressure import equ_wvp

def mixhum_ptd(p, t):
    
    """
    Function for calculating the mixing ratio from given pressure and dew point
    temperature. This corresponds to the equally named ncl function. See URL:
    https://www.ncl.ucar.edu/Document/Functions/Built-in/mixhum_ptd.shtml
    
    For the formula see URL: 
    https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
    
    Parameters:
        :param p: 2d or 3d array of air pressure. Unit: hPa.
        :param t: 2d or 3d array of air temperature. Must be of same size and 
                    shape as p. Unit: K.
                    
    Return:
        Array of same size as p and t containung the mixing ratio in kg/kg.
    """
    
    # convert the input to numpy arrays if they are scalar
    if np.isscalar(p):
        p = np.array([p])
    if np.isscalar(t):
        t = np.array([t])
    # end if
    
    # calculate the equilibirum water vapour pressure from the August-Roche-
    # Magnus formula
    # e_s = 6.1094 * np.exp(17.625 * t / (t + 243.04)) * 1000  # convert to Pa
    
    es = ((1.0007 + (3.46E-6 * p)) * 6.1121 * np.exp(17.502 * (t - 273.15) / 
           (240.97 + (t - 273.15))))
    
    # saturation mixing ratio wrt liquid water (g/kg)
    wsl = 0.622 * es / (p - es) 
    
    es = ((1.0003 + (4.18E-6 * p)) * 6.1115 * np.exp(22.452 * (t - 273.15) / 
           (272.55 + (t - 273.15))))
    
    # saturation mixing ratio wrt ice (g/kg)
    wsi = 0.622 * es / (p - es)
    
    ws = wsl
    ws[np.where(t < 273.15)] = wsi[np.where(t < 273.15)]
    
    return ws
    
    
    """
    e_s = equ_wvp(t)
    return 621.97 * e_s / (p - e_s)
    """
# end mixhum_ptd()    
    