"""
Function to perform the Feldl-Roe decomposition. See e.g. Goosse et al. (2018) and Singh et al. (2022).
"""

# imports
import os
import sys
import numpy as np
from scipy.stats import linregress as lr
from scipy.interpolate import interp2d as i2d

# set scripts directory and import self-written functions
direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Glob_Mean import glob_mean


# function
def feldl_decomp(dtas, dtoa, dohu, daht, t_flux, lr_flux, pl_flux, q_flux, sa_flux, c_flux, lwc_flux=None, swc_flux=None, 
                 lat=None, elp=20, tlat=None, p11=17, p12=22, p21=130, p22=None):
    """
    Warming contribution decomposition as derived in Feldl and Roe (2013). 
    NOTE: The feedbacks are here calculated NOT via regression but via division. This is inherent in the methodology as 
          we always need the temperature change at some specific point in time. Thus, the averaging periods defined by
          [p11:p12] and [p21:p22] are supposed to represent the centre year of the respective periods and the feedback
          may be interpreted as the feedback up until that year.
    
    Parameters:
        elp: Integer. Defines the period over which the forcing (via the Gregory method) is calculated.
        p11: Integer. Start year of averaging period 1. Defaults to 17.
        p12: Integer. End year of averaging period 1. Defaults to 22.
        p21: Integer. Start year of averaging period 2. Defaults to 130.
        p22: Integer. End year of averaging period 2. Defaults to None, i.e., the end of the time series.
    """
    
    if tlat is not None:
        time_coord = np.arange(np.shape(dtas)[0])
        
        f = i2d(lat, time_coord, dtas)
        dtas = f(tlat, time_coord)

        f = i2d(lat, time_coord, dtoa)
        dtoa = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, dohu)
        dohu = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, daht)
        daht = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, t_flux)
        t_flux = f(tlat, time_coord)

        f = i2d(lat, time_coord, lr_flux)
        lr_flux = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, pl_flux)
        pl_flux = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, q_flux)
        q_flux = f(tlat, time_coord)
        
        f = i2d(lat, time_coord, sa_flux)
        sa_flux = f(tlat, time_coord)

        f = i2d(lat, time_coord, c_flux)
        c_flux = f(tlat, time_coord)
        
        if lwc_flux is not None:
            f = i2d(lat, time_coord, lwc_flux)
            lwc_flux = f(tlat, time_coord)
        if swc_flux is not None:
            f = i2d(lat, time_coord, swc_flux)
            swc_flux = f(tlat, time_coord)
        # end if    
            
        lat = tlat
        
    # end if
    
    # calculate the temperature change at year 20 (mean over years 18-22) and at the "end" (mean over years 131-150)
    dtas_20 = np.mean(dtas[p11:p12, :], axis=0)
    dtas_150 = np.mean(dtas[p21:p22, :], axis=0)

    # same for OHU and AHT
    dohu_20 = np.mean(dohu[p11:p12, :], axis=0)
    dohu_150 = np.mean(dohu[p21:p22, :], axis=0)
    
    daht_20 = np.mean(daht[p11:p12, :], axis=0)
    daht_150 = np.mean(daht[p21:p22, :], axis=0)

    # ... and for the feedback-induced radiative fluxes
    lr_20 = np.mean(lr_flux[p11:p12, :], axis=0)
    lr_150 = np.mean(lr_flux[p21:p22, :], axis=0)

    pl_20 = np.mean(pl_flux[p11:p12, :], axis=0)
    pl_150 = np.mean(pl_flux[p21:p22, :], axis=0)

    q_20 = np.mean(q_flux[p11:p12, :], axis=0)
    q_150 = np.mean(q_flux[p21:p22, :], axis=0)

    sa_20 = np.mean(sa_flux[p11:p12, :], axis=0)
    sa_150 = np.mean(sa_flux[p21:p22, :], axis=0)
    
    c_20 = np.mean(c_flux[p11:p12, :], axis=0)
    c_150 = np.mean(c_flux[p21:p22, :], axis=0)
    
    # calculate the feedbacks with respect to the zonal mean dtas - full, early, late
    t_fb = np.zeros((3, len(lat)))
    lr_fb = np.zeros((3, len(lat)))
    pl_fb = np.zeros((3, len(lat)))
    q_fb = np.zeros((3, len(lat)))
    sa_fb = np.zeros((3, len(lat)))
    c_fb = np.zeros((3, len(lat)))

    # calculate feedbacks via the "division" (i.e., not the regression method)
    lr_fb[1, :], lr_fb[2, :] = lr_20 / dtas_20, lr_150 / dtas_150
    pl_fb[1, :], pl_fb[2, :] = pl_20 / dtas_20, pl_150 / dtas_150
    q_fb[1, :], q_fb[2, :] = q_20 / dtas_20, q_150 / dtas_150
    sa_fb[1, :], sa_fb[2, :] = sa_20 / dtas_20, sa_150 / dtas_150
    c_fb[1, :], c_fb[2, :] = c_20 / dtas_20, c_150 / dtas_150
        
    # calculate the global mean Planck feedback and the deviation
    pl_fbm = glob_mean(pl_fb, lat, from_zon=True)
    pl_fbd = pl_fb - pl_fbm[:, None]

    # calculate the global mean forcing estimate from the Gregory method
    forc = lr(glob_mean(dtas, lat, from_zon=True)[:elp], glob_mean(dtoa, lat, from_zon=True)[:elp])[1]
    
    # calculate the warming contributions
    output = {}

    # Planck
    output["dt_pl_e"] = -pl_fbd[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_pl_l"] = -pl_fbd[2, :] / pl_fbm[2, None] * dtas_150

    # temperature
    output["dt_t_e"] = -t_fb[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_t_l"] = -t_fb[2, :] / pl_fbm[2, None] * dtas_150

    # lapse rate
    output["dt_lr_e"] = -lr_fb[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_lr_l"] = -lr_fb[2, :] / pl_fbm[2, None] * dtas_150

    # water vapour
    output["dt_q_e"] = -q_fb[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_q_l"] = -q_fb[2, :] / pl_fbm[2, None] * dtas_150

    # surface albedo
    output["dt_sa_e"] = -sa_fb[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_sa_l"] = -sa_fb[2, :] / pl_fbm[2, None] * dtas_150

    # cloud
    output["dt_c_e"] = -c_fb[1, :] / pl_fbm[1, None] * dtas_20
    output["dt_c_l"] = -c_fb[2, :] / pl_fbm[2, None] * dtas_150

    # OHU
    output["dt_ohu_e"] = dohu_20 / pl_fbm[1, None]
    output["dt_ohu_l"] = dohu_150 / pl_fbm[2, None]

    # AHT
    output["dt_aht_e"] = -daht_20 / pl_fbm[1, None]
    output["dt_aht_l"] = -daht_150 / pl_fbm[2, None]
    
    # forcing
    output["dt_forc_e"] = -forc /  pl_fbm[1]
    output["dt_forc_l"] = -forc /  pl_fbm[2]

    # calculate the sum
    output["dt_sum_e"] = (output["dt_pl_e"] + output["dt_lr_e"] + output["dt_q_e"] + output["dt_sa_e"] + 
                          output["dt_c_e"] + output["dt_ohu_e"] + output["dt_aht_e"] + output["dt_forc_e"])
    output["dt_sum_l"] = (output["dt_pl_l"] + output["dt_lr_l"] + output["dt_q_l"] + output["dt_sa_l"] + 
                          output["dt_c_l"] + output["dt_ohu_l"] + output["dt_aht_l"] + output["dt_forc_l"])
    
    # output also the (regridded) temperature
    output["dtas"] = dtas
    
    
    if lwc_flux is not None:
        lwc_20 = np.mean(lwc_flux[p11:p12, :], axis=0)
        lwc_150 = np.mean(lwc_flux[p21:p22, :], axis=0)
        lwc_fb = np.zeros((3, len(lat)))
        lwc_fb[1, :], lwc_fb[2, :] = lwc_20 / dtas_20, lwc_150 / dtas_150
        output["dt_lw_c_e"] = -lwc_fb[1, :] / pl_fbm[1, None] * dtas_20
        output["dt_lw_c_l"] = -lwc_fb[2, :] / pl_fbm[2, None] * dtas_150
    if swc_flux is not None:
        swc_20 = np.mean(swc_flux[p11:p12, :], axis=0)
        swc_150 = np.mean(swc_flux[p21:p22, :], axis=0)
        swc_fb = np.zeros((3, len(lat)))
        swc_fb[1, :], swc_fb[2, :] = swc_20 / dtas_20, swc_150 / dtas_150
        output["dt_sw_c_e"] = -swc_fb[1, :] / pl_fbm[1, None] * dtas_20
        output["dt_sw_c_l"] = -swc_fb[2, :] / pl_fbm[2, None] * dtas_150
    # end if
    
    return output
    
# end function    