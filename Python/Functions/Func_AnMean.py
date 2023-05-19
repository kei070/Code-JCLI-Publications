import os
import sys
import numpy as np

direc = "../"
os.chdir(direc)
# needed for console
sys.path.append(direc)

from Functions.Func_Print_In_One_Line import print_one_line


def an_mean(field, meth="for", squeeze=True):
    
    """
    Function for calculating the annual means of a given field of monthly mean
    values.
    
    Parameters:
        :param field: The array to be averaged. Note that time has to the 1st
                      (in Python actually the 0th) dimension. Also, the 
                      function only works if the time axis length is a multiple
                      of 12.
        :param meth: Method of averaging. Available are "for" and "re". The
                     former uses a for-loop to average consecutive blocks of 12
                     values. The latter will reshape the input field an 
                     create a new axis of length 12 and then exploit numpy's
                     mean function and average over this new axis. The "re"
                     method might be quicker for 1d-arrays but it is not clear
                     to me yet if it works for arrays with more than 1 
                     dimension. Defaults to "for".
         :param squeeze: Logical. If True (default) the single-dimensional entries will be removed from the shape if the
                         result.
    """
    
    # get the number of years
    n_mon = np.shape(field)[0]
    n_yr = int(n_mon / 12)
    
    # if number of years is 1 just return the mean over axis 1
    if n_yr == 1:
        return np.mean(field, axis=0)
    # end if

    # create the shape of the output array
    sh = list(np.shape(field))
    sh[0] = n_yr
    
    # set up the target array
    result = np.zeros(tuple(sh))
    
    # calculate the annual means according to the chosen method
    if meth == "for":
        yr = 0
        for i in np.arange(0, n_mon, 12):
            result[yr] = np.mean(field[i:i+12], axis=0)
            
            yr += 1
        # end for i
        
    elif meth == "re":
        field_re = field.reshape((n_yr, 12))
        result = np.mean(field_re, axis=1)
    # end if elif
    
    if squeeze:
        return np.squeeze(result)
    else:
        return result
    # end if else
    
# end fucntion an_mean()


def an_mean_verb(field, meth="for", squeeze=True):
    
    """
    Verbose version of the above function: It prints the progress in each for-loop iteration.
    Function for calculating the annual means of a given field of monthly mean
    values.
    
    Parameters:
        :param field: The array to be averaged. Note that time has to the 1st
                      (in Python actually the 0th) dimension. Also, the 
                      function only works if the time axis length is a multiple
                      of 12.
        :param meth: Method of averaging. Available are "for" and "re". The
                     former uses a for-loop to average consecutive blocks of 12
                     values. The latter will reshape the input field an 
                     create a new axis of length 12 and then exploit numpy's
                     mean function and average over this new axis. The "re"
                     method might be quicker for 1d-arrays but it is not clear
                     to me yet if it works for arrays with more than 1 
                     dimension. Defaults to "for".
         :param squeeze: Logical. If True (default) the single-dimensional entries will be removed from the shape if the
                         result.
    """
    
    # get the number of years
    n_mon = np.shape(field)[0]
    n_yr = int(n_mon / 12)

    # create the shape of the output array
    sh = list(np.shape(field))
    sh[0] = n_yr
    
    # set up the target array
    result = np.zeros(tuple(sh))
    
    # calculate the annual means according to the chosen method
    if meth == "for":
        yr = 0
        for i in np.arange(0, n_mon, 12):
            result[yr] = np.mean(field[i:i+12], axis=0)
            print_one_line(str(yr) + "/" + str(n_yr) + "  ")
            
            yr += 1
        # end for i                    
    elif meth == "re":
        field_re = field.reshape((n_yr, 12))
        result = np.mean(field_re, axis=1)
    # end if elif
    print("\n")    
    if squeeze:
        return np.squeeze(result)
    else:
        return result
    # end if else
    
# end fucntion an_mean()
    
