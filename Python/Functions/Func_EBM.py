"""
Functions for implementing the Geoffroy et al. (2013a,b) EBM-1 and EBM-epsilon.

Note that this will not work in my kern_calc environment but only in the kern_calc2 environment!
"""

# imports
import numpy as np
import pandas as pd
from scipy.stats import linregress as lr
import statsmodels.api as sm


# function EBM-1
def ebm_base(dtas, dtoa, mode="1", forc=None, lamb=None, tau_f_ny=10, tau_s_s=30, tau_s_e=150, eps=1, t_stand=150, 
             t_long=1000):
    
    """
    Function for parameter calibration of the EBM described in Geoffroy et al. (2013a, hereafter G13a).
    Parameters:
        :param dtas: 1-dimensional numpy array containing the global-mean surface-air temperature perturbation calculated
                     from a step-forcing experiment in an AOGCM with respect to the corresponding piControl run. Has to
                     be 150 years long (as of now).
        :param dtoa: Same as dtas but for top-of-atmosphere radiative imbalance.
        :param mode: Character. Either "1" or "eps". The former makes the function resemble the EBM-1 and forcing and
                     feedback are calculated from the given dtas and dtoa via the Gregory method. The latter is the mode
                     to use in the ebm_eps function and forcing, feedback, and heat-uptake efficacy have to be given as
                     paramters forc, lamb, and eps, respectively.
        :param forc: Float. If mode="eps" this is the forcing value used in the equations to calibrate the other 
                     parameters. If mode="1" this parameter is ignored. Defaults to none.
        :param lamb: Float. If mode="eps" this is the feedback value used in the equations to calibrate the other 
                     parameters. If mode="1" this parameter is ignored. Defaults to none.
        :param tau_f_ny: Integer. Number of year to use for the calibration of tau_f (the fast time-scale). Defaults to
                         10 as described in G13a but for some models other values might be necessary to avoid negative 
                         values in the log (see Geoffroy and Saint-Martin, 2020, Appendix a).
        :param tau_s_s: Integer. Start year of the period for the regression for the calibration of tau_s. Defaults to 
                        30 as described in G13a. This should probably not be changed but it may be necessary to avoid
                        inadmissible values in the log-function (see G13a, eq. 17).
        :param tau_s_e: Integer. End year of the period for the regression for the calibration of tau_s. Defaults to 
                        150 as described in G13a. This should probably not be changed but it may be necessary to avoid
                        inadmissible values in the log-function (see G13a, eq. 17).                
        :param t_stand: Integer. Number of years which should be taken from the AOGCM run. Defaults to 150. [This should
                        probably not be changed without further thought and possibly some further changes.]
        :param t_long: Integer. Number of years for which the EBM output is produced. Defaults to 1000.        
        :param eps: Float. Deep-ocean heat uptake efficacy. This parameter is necessary to expand the model to 
                    EBM-epsilon. Defaults to 1 which makes the current EBM equivalent to EBM-1.
    """
    
    # generate the time arrays
    t = np.arange(t_stand)
    t_l = np.arange(t_long)
    
    if mode == "1":
        # calculate lambda and F from the TOA on SAT regression
        lamb, forc = lr(dtas, dtoa)[:2]
        lamb = -lamb
        
        lamb_orig = lamb
        forc_orig = forc
    if mode == "eps":
        # calculate lambda and F from the TOA on SAT regression
        lamb_orig, forc_orig = lr(dtas, dtoa)[:2]
        lamb_orig = -lamb_orig
    # end if
    
    # calculate T_eq from F and lambda
    t_eq = forc / lamb    
    
    # calculate tau_s and a_s, a_f
    log_diff = np.log(t_eq - dtas[tau_s_s:tau_s_e])  # note that this gives undefined values where dtas > t_eq
    
    tau_s_inv, as_temp = lr(np.arange(tau_s_s, tau_s_e), log_diff)[:2]
    
    a_s = np.exp(as_temp - np.log(t_eq))
    tau_s = -1/tau_s_inv
    
    a_f = 1 - a_s
    
    
    # loop over the first ten years and average over them to obtain tau_f
    tau_f_list = []
    
    for yr in np.arange(1, 1+tau_f_ny):
        tau_f_temp = yr / (np.log(a_f) - np.log(1 - dtas[yr-1] / t_eq - a_s * np.exp(-yr / tau_s)))
        tau_f_list.append(tau_f_temp)    
    # end for i, yr
    
    tau_f_arr = np.array(tau_f_list)
    tau_f_arr = np.array(pd.Series(tau_f_arr).interpolate())
    
    tau_f = np.mean(tau_f_arr)
    
    # calculate C, C_0, gamma
    c = lamb / (a_f / tau_f + a_s / tau_s)
    c_0 = (lamb * (tau_f * a_f + tau_s * a_s) - c)
    gam = c_0 / (tau_f / a_s + tau_s * a_f)    
    
    
    # calculate b, b_star, delta (general parameters)
    b = (lamb + gam) / c + gam / c_0
    b_star = (lamb + gam) / c - gam / c_0
    delt = b**2 - 4 * (lamb * gam) / (c * c_0)


    # calculate mode parameters (the ones not calculated yet, i.e., phi_f and phi_s)
    phi_f = c / (2*gam) * (b_star - delt**0.5)
    phi_s = c / (2*gam) * (b_star + delt**0.5)
    
    # apparently necessary in the EBM-eps mode: calculate a'_f, a'_s, tau'_f, tau'_s 
    # if mode == "eps":
    #     tau_f = c * c_0 * (b - delt**0.5) / (2 * lamb * gam)
    #     tau_s = c * c_0 * (b + delt**0.5) / (2 * lamb * gam)
    #     a_f = phi_s * tau_f * lamb / (c * (phi_s - phi_f))
    #     a_s = -phi_f * tau_s * lamb / (c * (phi_s - phi_f))
    # end if

    # calculate the temperatures of the two layers via eqs. (11) and (12) in Part 1
    temp = t_eq - a_f * t_eq * np.exp(-t / tau_f) - a_s * t_eq * np.exp(-t / tau_s)
    temp0 = t_eq - phi_f * a_f * t_eq * np.exp(-t/tau_f) - phi_s * a_s * t_eq * np.exp(-t/tau_s)
    
    # generate a longer timer series for EBM-1
    temp_long = t_eq - a_f * t_eq * np.exp(-t_l / tau_f) - a_s * t_eq * np.exp(-t_l / tau_s)
    temp0_long = t_eq - phi_f * a_f * t_eq * np.exp(-t_l/tau_f) - phi_s * a_s * t_eq * np.exp(-t_l/tau_s)


    # calculate the deep-ocean heat uptake (note that it must be gamma and not gamma' here!)
    h = gam / eps * (temp - temp0)
    h_long = gam / eps * (temp_long - temp0_long)
    
    
    # "extras"
    
    # calculate the upper- and lower-ocean heat uptake temperatures

    # calculate the necessary parameters 
    f_u = c / (lamb * tau_f)
    f_d = (phi_f * c_0) / (lamb * tau_f)
    s_u = c / (lamb * tau_s)
    s_d = (phi_s * c_0) / (lamb * tau_s)

    # calculate the temperatures
    # temp_u = -t_eq * (f_u * a_f * np.exp(-t_l/tau_f) + s_u * a_s * np.exp(-t_l/tau_s))
    # temp_d = -t_eq * (f_d * a_f * np.exp(-t_l/tau_f) + s_d * a_s * np.exp(-t_l/tau_s))
    
    temp_u = -t_eq/lamb * c * (a_f/tau_f * np.exp(-t_l/tau_f) + a_s/tau_s * np.exp(-t_l/tau_s))
    temp_d = -t_eq/lamb * c_0 * (phi_f * a_f/tau_f * np.exp(-t_l/tau_f) + phi_s * a_s/tau_s * np.exp(-t_l/tau_s))
    
    
    # prepare the output dictionary
    output = {}
    
    # scalar quantities
    output["F"] = forc
    output["lambda"] = lamb
    output["T_eq"] = t_eq
    output["gamma"] = gam
    output["C"] = c
    output["C_0"] = c_0
    output["tau_f"] = tau_f
    output["tau_s"] = tau_s
    output["a_f"] = a_f
    output["a_s"] = a_s
    output["phi_f"] = phi_f
    output["phi_s"] = phi_s
    
    # array quantities
    output["T(t)"] = temp
    output["T_0(t)"] = temp0
    output["H(t)"] = h
    output["T(t) long"] = temp_long
    output["N(t)"] = forc - lamb * temp_long
    output["T_0(t) long"] = temp0_long
    output["H(t) long"] = h_long
    output["T_U"] = temp_u
    output["T_D"] = temp_d
    
    # return the output
    return output
    
# end def ebm_base


# function EBM-epsilon
def ebm_eps(dtas, dtoa, tau_f_ny=10, tau_s_s=30, tau_s_e=150, t_stand=150, t_long=1000, eps=1, n_iter=20):

    """
    Function for parameter calibration of the EBM-epsilon described in Geoffroy et al. (2013b). Uses function ebm_base().
    Geoffroy et al. (2013a) is hereafter referred to G13a.
    Parameters:
        :param dtas: 1-dimensional numpy array containing the global-mean surface-air temperature perturbation calculated
                     from a step-forcing experiment in an AOGCM with respect to the corresponding piControl run. Has to
                     be 150 years long (as of now).
        :param dtoa: Same as dtas but for top-of-atmosphere radiative imbalance.
        :param tau_f_ny: Integer. Number of year to use for the calibration of tau_f (the fast time-scale). Defaults to
                         10 but for some models other values might be necessary to avoid negative values in the log (see
                         Geoffroy and Saint-Martin, 2020, Appendix a).
        :param tau_s_s: Integer. Start year of the period for the regression for the calibration of tau_s. Defaults to 
                        30 as described in G13a. This should probably not be changed but it may be necessary to avoid
                        inadmissible values in the log-function (see G13a, eq. 17).
        :param tau_s_e: Integer. End year of the period for the regression for the calibration of tau_s. Defaults to 
                        150 as described in G13a. This should probably not be changed but it may be necessary to avoid
                        inadmissible values in the log-function (see G13a, eq. 17).                         
        :param t_stand: Integer. Number of years which should be taken from the AOGCM run. Defaults to 150. [This should
                        probably not be changed without further thought and possibly some further changes.]
        :param t_long: Integer. Number of years for which the EBM output is produced. Defaults to 1000.        
        :param eps: Float. Deep-ocean heat uptake efficacy. This parameter is necessary to expand the model to 
                    EBM-epsilon. Defaults to 1 which makes the current EBM equivalent to EBM-1.
        :param n_iter: Integer. Number of iterations of epsilon calibration.            
    """
    
    # calculate the values for EBM-1
    ebm = ebm_base(dtas, dtoa, mode="1", tau_f_ny=tau_f_ny, tau_s_s=tau_s_s, tau_s_e=tau_s_e)

    # start a for-loop iteration to calculate the heat uptake efficacy
    eps_list = []
    # t_eq_list = []
    forc_list = []
    lamb_list = []
    
    for i in np.arange(n_iter):
    
        #% set up a dataframe from dtas_agm and h(i-1) to derive F(i), lamb(i), and eps(i)
        mult_lr_x =  pd.DataFrame(data={"SAT":dtas, "H":ebm["H(t)"]})
        
        # also "convert" the dtoa_agm into a dataframe
        mult_lr_y = pd.DataFrame(data={"N":dtoa})
        
        # add a constant (?)
        mult_lr_x = sm.add_constant(mult_lr_x)
               
        # perform the above-mentioned multiple linear regression
        est = sm.OLS(mult_lr_y, mult_lr_x).fit()
        
        # calculate eps from the H-coef from the regression which is -(eps - 1)
        eps_new = -est.params.H + 1
        eps_list.append(eps_new)
        
        # the new feedback parameter
        lamb_new = -est.params.SAT
        lamb_list.append(lamb_new)
        
        # the new forcing
        forc_new = est.params.const
        forc_list.append(forc_new)
        
        # calculate the new EBM values
        ebm = ebm_base(dtas, dtoa, mode="eps", forc=forc_new, lamb=lamb_new, eps=eps_new, t_long=t_long, 
                       tau_f_ny=tau_f_ny, tau_s_s=tau_s_s, tau_s_e=tau_s_e)
        
    # end for i

    
    # calculate linear and deviation term of the radiative response for eps != 1 (see caption of Fig. 1)
    
    # linear term
    lin_term = forc_new - lamb_new * ebm["T(t) long"]
    
    # deviation term
    dev_term = -(eps_new - 1) * ebm["H(t) long"]
    
    
    # calculate the radiative contributions from upper and lower temperature
    r_u = -ebm["lambda"] * ebm["T_U"]
    
    lamb_d = ebm["lambda"] / eps_new
    r_d = -lamb_d * ebm["T_D"]
    

    # prepare the output
    output = {}
    
    # scalar quantities
    output["F"] = ebm["F"]
    output["lambda"] = ebm["lambda"]
    output["gamma"] = ebm["gamma"]
    output["T_eq"] = ebm["T_eq"]
    output["epsilon"] = eps_new
    output["C"] = ebm["C"]
    output["C_0"] = ebm["C_0"]
    output["tau_f"] = ebm["tau_f"]
    output["tau_s"] = ebm["tau_s"]
    
    # array quantities
    output["T(t)"] = ebm["T(t)"]
    output["T_0(t)"] = ebm["T_0(t)"]
    output["H(t)"] = ebm["H(t)"]
    output["T(t) long"] = ebm["T(t) long"]
    output["T_0(t) long"] = ebm["T_0(t) long"]
    output["H(t) long"] = ebm["H(t) long"]
    output["linear term"] = lin_term  # F + lambda * T
    output["deviation term"] = dev_term  # -(eps - 1) * H
    output["N(t)"] = lin_term + dev_term
    output["T_U"] = ebm["T_U"]
    output["T_D"] = ebm["T_D"]
    output["R_U"] = r_u
    output["R_D"] = r_d
    output["eps list"] = np.array(eps_list)
    
    # return the output
    return output
    
# end def ebm_eps