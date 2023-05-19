"""
Extrapolate temperature to the lower levels.
"""

import os
import sys
import numpy as np
import dask.array as da
import datetime
import timeit


direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Print_In_One_Line import print_one_line
from Functions.Func_Interp4d import interp4d


def extrap_to_lower(ta, levs, lat, lon):
    
    start = timeit.default_timer()
    
    """
    try:
        ta_max = np.max(ta[0, 0, :, :]).compute()
    except:
        ta_max = np.max(ta[0, 0, :, :])
    # end try except
    """

    # print the time of the start of the calculations
    # print("\nStarting ta interpolation ta all p-levels...\n")
    # print(datetime.datetime.now())

    ta_int = []
    for t in np.arange(0, np.shape(ta)[0]):
        print_one_line(str(t) + " ")
        ta_int.append(da.from_array(interp4d(t, levs, ta, levs)))
    # end for t
    
    # set the interpolated values to the original names
    ta = da.stack(ta_int, axis=0)
    
    # print the time of the end of the calculations
    # print("\n")
    # print(datetime.datetime.now())        
    
    return ta.rechunk((75, len(levs), len(lat), len(lon)))
    
    stop = timeit.default_timer()
    print("\nYour program needed: " + str(stop - start) + " seconds\n")
    
# end function extrap_to_lower()
