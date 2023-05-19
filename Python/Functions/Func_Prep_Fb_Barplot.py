import numpy as np


def prep_fb_bars(res_s, lr_r, kl):
    
    """
    Function for preparing feedback values for a barplot from feedback: early, late, and difference.
    
    Parameters:
        res_s: List of response abbreviations as strings (e.g. something like "T" for temperature, "Q" for water vapour)
               etc.
        lr_r: Dictionary containing all the feedbacks for all the kernels. The feedbacks are contained as slopes of the 
              linear regression. As an example for how to select the early period (year 1-20) temperature feedback for 
              the BM13 kernels: lr_r["BM13"]["T"]["e"]["s"]
        kl: Kernel abbreviation string, e.g. "BM13" for Block and Mauritsen (2013) kernels.                      
    """
    
    # first prepare the feedbacks and the feedback changes: put them together in arrays and then sort them first with 
    # respect to their sign and then with respect to their magnitude
    fbs_e = np.zeros(len(res_s))
    fbs_l = np.zeros(len(res_s))
    res_a = np.array(res_s)
    for i, re in enumerate(res_s):
        fbs_e[i] = lr_r[kl][re]["e"]["s"]
        fbs_l[i] = lr_r[kl][re]["l"]["s"]
    # end for i, re
    
    # calculate the feedback change
    d_fbs = fbs_l - fbs_e
    
    # sort them with respect to their sign
    fbs_e_p = fbs_e[fbs_e > 0]
    re_e_p = res_a[fbs_e > 0]
    fbs_e_n = fbs_e[fbs_e < 0]
    re_e_n = res_a[fbs_e < 0]
    
    fbs_l_p = fbs_l[fbs_l > 0]
    re_l_p = res_a[fbs_l > 0]
    fbs_l_n = fbs_l[fbs_l < 0]
    re_l_n = res_a[fbs_l < 0]
    
    d_fbs_p = d_fbs[d_fbs > 0]
    d_re_p = res_a[d_fbs > 0]
    d_fbs_n = d_fbs[d_fbs < 0]
    d_re_n = res_a[d_fbs < 0]
    
    # sort the arrays so that the largest (absolute values!) feedback comes first
    re_e_p = re_e_p[np.argsort(np.abs(fbs_e_p))][::-1]
    fbs_e_p = fbs_e_p[np.argsort(np.abs(fbs_e_p))][::-1]
    re_e_n = re_e_n[np.argsort(np.abs(fbs_e_n))][::-1]
    fbs_e_n = fbs_e_n[np.argsort(np.abs(fbs_e_n))][::-1]
    
    re_l_p = re_l_p[np.argsort(np.abs(fbs_l_p))][::-1]
    fbs_l_p = fbs_l_p[np.argsort(np.abs(fbs_l_p))][::-1]
    re_l_n = re_l_n[np.argsort(np.abs(fbs_l_n))][::-1]
    fbs_l_n = fbs_l_n[np.argsort(np.abs(fbs_l_n))][::-1]
    
    d_re_p = d_re_p[np.argsort(np.abs(d_fbs_p))][::-1]
    d_fbs_p = d_fbs_p[np.argsort(np.abs(d_fbs_p))][::-1]
    d_re_n = d_re_n[np.argsort(np.abs(d_fbs_n))][::-1]
    d_fbs_n = d_fbs_n[np.argsort(np.abs(d_fbs_n))][::-1]    
    
    # append a zero to each of the arrays if their longer than 0
    if len(fbs_e_p) > 0:
        fbs_e_p = np.append(np.array([0]), fbs_e_p)
    if len(fbs_l_p) > 0:
        fbs_l_p = np.append(np.array([0]), fbs_l_p)
    if len(fbs_e_n) > 0:
        fbs_e_n = np.append(np.array([0]), fbs_e_n)
    if len(fbs_l_n) > 0:
        fbs_l_n = np.append(np.array([0]), fbs_l_n)
    if len(d_fbs_p) > 0:
        d_fbs_p = np.append(np.array([0]), d_fbs_p)
    if len(fbs_e_n) > 0:
        d_fbs_n = np.append(np.array([0]), d_fbs_n)
    # end for
    
    return fbs_e_p, fbs_e_n, fbs_l_p, fbs_l_n, d_fbs_p, d_fbs_n, re_e_p, re_e_n, re_l_p, re_l_n, d_re_p, d_re_n
    
# end function prep_fb_bars()   
