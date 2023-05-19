"""
Function for calculating seasonal means.
"""

import os
import sys
import numpy as np

def sea_mean(field):

    # get the number of years
    n_mon = np.shape(field)[0]
    n_yr = int(n_mon / 12)
    
    # create the shape of the output array
    sh = list(np.shape(field))
    sh[0] = n_yr
    
    # set up the output dictionary
    sea_means_dict = {"DJF":np.zeros(tuple(sh)), 
                      "MAM":np.zeros(tuple(sh)), 
                      "JJA":np.zeros(tuple(sh)), 
                      "SON":np.zeros(tuple(sh))}
   
    # because the first winter is incomplete (on J and F), offset the winter-index by one
    wint_add = [1, 0, 0, 0]
    
    count = 0
    for sea_i, sea in zip([12, 3, 6, 9], sea_means_dict):
        
        for i, ii in enumerate(np.arange(sea_i, len(field), 12)):
            im = ii - 1
            ip = ii + 1
            sea_means_dict[sea][i+wint_add[count]] = np.mean(field[im:ip], axis=0)
            
        # end for i, ii
        
        count += 1
        
    # end for sea_i, sea    
            
    # special winter treatmeant
    sea_means_dict["DJF"][0] = np.mean(field[:2], axis=0)
    
    # return result
    return sea_means_dict
    
# end function    
    

           
""" test
# field
field = np.arange(120)

sea_means_dict = {"DJF":np.zeros(int(len(field) / 12)), 
                  "MAM":np.zeros(int(len(field) / 12)), 
                  "JJA":np.zeros(int(len(field) / 12)), 
                  "SON":np.zeros(int(len(field) / 12))}

# because the first winter is incomplete (on J and F), offset the winter-index by one
wint_add = [1, 0, 0, 0]

count = 0
for sea_i, sea in zip([12, 3, 6, 9], sea_means_dict):
    
    for i, ii in enumerate(np.arange(sea_i, len(field), 12)):
        im = ii - 1
        ip = ii + 1
        sea_means_dict[sea][i+wint_add[count]] = np.mean(field[im:ip], axis=0)
        
    # end for i, ii
    
    count += 1
    
# end for sea_i, sea    
        
# special winter treatmeant
sea_means_dict["DJF"][0] = np.mean(field[:2], axis=0)
"""
