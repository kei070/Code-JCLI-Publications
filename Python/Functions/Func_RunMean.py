"""
Function for running means.
"""

# imports
import numpy as np
import dask.array as da


def run_mean_da(data, running, axis=0):
    
    # convert the input to a dask array if it is not already
    if type(data) != da.core.Array:
        data = da.from_array(data)
    # end if
    
    c_sum = da.cumsum(data, axis=axis)
    result = c_sum[running:] - da.cumsum(data, axis=axis)[:-running]
    result = da.insert(result, 0, c_sum[running-1], axis=0)
    return result / running

# end function run_mean()


def run_mean(data, running, method="cum", axis=0):
    
    """
    Function for calculating a running mean value.
    
    Parameters:
        :param data:    1d array of values to be averaged.
        :param running: Integer. Number of values to be taken into consideration for the running mean. Note that this 
                        number has to be even for the function to work correctly.
        :param axis:    Integer. Axis over which the running mean is to be calculated. Defaults to 0. Note that axis
                        will probably not yet work, at least for method="cum".
        :param method:  String. Either "cum" for choosing the method using the numpy cumsum function (faster) or "for"
                        for using a for-loop (slower). Defaults to "cum".
                        
    Return:
        1d array of the running means. Note that this array will be of length
        len(data)-running and if the data is a time series the values will 
        start at index running/2 and end at len(data)-running/2 (don't forget
        here that Python starts counting at 0).
    """
    
    if method == "cum":
        if axis != 0:
            print("\nSince your axis is not 0, the result might not be correct.\n")
        # end if
        result = np.cumsum(data, axis=axis)
        result[running:] = result[running:] - np.cumsum(data, axis=axis)[:-running]
        return result[running-1:] / running
    
    elif method == "for":    
    
        run_half = int(running / 2)
        
        sh = np.array(np.shape(data))
        sh[axis] = sh[axis] - int(2 * run_half)
        sh = list(sh)
        
        data_run = np.zeros(sh)
        count = 0
        for i in np.arange(run_half, np.shape(data)[axis] - run_half, 1):
            data_run[count] = np.mean(data[i-run_half:i+run_half], axis=axis)
            
            count += 1
        # end for i
        
        return data_run
    
    # end if elif
    
# end function run_mean()
    