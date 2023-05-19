"""
Attempt to write a function that takes the mean value of a 2d-array north of
a certain latitude.
"""

# imports
import numpy as np


def lat_mean(field, lats=None, lons=None, lat=None, from_zonal=False, zon_inv=False, verbose=False, return_dic="old"):
    
    """
    Function for calculating the area-weighted means with repsect to a given latitude. The function expects the user to
    provide a latitude between 0 and 90N and will return three 
    values: (1) the mean over the area north of the given latitude [n_mean]; (2) the mean over the area south of minus
    the given latitude (i.e., the latitude in the southern hemisphere corresponding to the given latitude) [s_mean]; 
    (3) the mean over the area south of the given latitude and north of minus the given latitude [eq_mean].
    If lat=0, the eq_mean will be returned as 0.
    
    Expecting shape = (lat, time) if from_zonal=True; if zon_inv=True the dimensions are expexted to be (time, lat).
    
    param: return_dic Defaults to "old" which means the old version of the function which returned only the n_mean, the
                      s_mean, and the eq_mean. Set to "new" to return all values.
    
    Return:
        Dictionary with the self-explaining keys "n_mean", "s_mean", "eq_mean", "wo_n_mean", "wo_s_mean", "eq_mean", 
        "n_band_mean", "s_band_mean", "eq_band_mean", "wo_n_band_mean", "wo_s_band_mean".
    """
    
    print("DOES NOT WORK CORRECTLY WITH MASKED ARRAYS YET!")
    

    # get the index north and south of the requested latitude
    n_ind = np.min(np.where(lats > lat)[0])
    # s_ind = np.max(np.where(np.abs(lats[:np.argmin(np.abs(lats))]) > lat))
    s_ind = np.max(np.where(lats[:n_ind] < -lat))
    
    if from_zonal:
        weights = np.cos(lats / 180 * np.pi)
        
        if not zon_inv:
            if np.ndim(field) == 2:
                n_mean = np.mean(field[n_ind:, :] * weights[n_ind:, None], axis=0)
                s_mean = np.mean(field[:s_ind+1, :] * weights[:s_ind+1, None], axis=0)
                wo_n_mean = np.mean(field[:n_ind, :] * weights[:n_ind, None], axis=0)
                wo_s_mean = np.mean(field[s_ind:, :] * weights[s_ind:, None], axis=0)        
                eq_mean = 0
                if lat > 0:
                    eq_mean = np.mean(field[s_ind+1:n_ind, :], weights=weights[s_ind+1:n_ind, None], axis=0)
                # end if            
            elif np.ndim(field) == 1:
                n_mean = np.mean(field[n_ind:] * weights[n_ind:])
                s_mean = np.mean(field[:s_ind+1] * weights[:s_ind+1])
                wo_n_mean = np.mean(field[:n_ind] * weights[:n_ind])
                wo_s_mean = np.mean(field[s_ind:] * weights[s_ind:])
                eq_mean = 0
                if lat > 0:
                    eq_mean = np.mean(field[s_ind+1:n_ind] * weights[s_ind+1:n_ind])
                # end if
            # end if elif
        else:
            if np.ndim(field) == 2:
                n_mean = np.mean(field[:, n_ind:] * weights[None, n_ind:], axis=0)
                s_mean = np.mean(field[:, :s_ind+1] * weights[None, :s_ind+1], axis=0)
                wo_n_mean = np.mean(field[:, :n_ind] * weights[None, :n_ind], axis=0)
                wo_s_mean = np.mean(field[:, s_ind:] * weights[None, s_ind:], axis=0)        
                eq_mean = 0
                if lat > 0:
                    eq_mean = np.mean(field[:, s_ind+1:n_ind] * weights[None, s_ind+1:n_ind], axis=0)
                # end if            
            # end if
        # end if else
    else:        
        # mesh lats and lons for generating the weights
        lonm, latm  = np.meshgrid(lons, lats)
        
        # generate the weights (2 or 3 dimensional)
        if np.ndim(field) == 3:
            weights = np.zeros(np.shape(field))
            weights[:, :, :] = np.cos(latm / 180 * np.pi)[None, :, :]
        elif np.ndim(field) == 2:
            weights = np.cos(latm / 180 * np.pi)
        # end if elif
            
        # get the weighted average of the regions in polar direction of the given
        # latitude on both hemispheres; also calculate the weighted means for the globe WITHOUT these respective regions
        # also calculate the means over the respective latitudinal bands
        if np.ndim(field) == 3:
            n_mean = np.average(field[:, n_ind:, :], 
                                weights=weights[:, n_ind:, :], axis=(-2, -1))
            s_mean = np.average(field[:, :s_ind+1, :], 
                                weights=weights[:, :s_ind+1, :], axis=(-2, -1))
            wo_n_mean = np.average(field[:, :n_ind, :], 
                                   weights=weights[:, :n_ind, :], axis=(-2, -1))
            wo_s_mean = np.average(field[:, s_ind:, :], 
                                   weights=weights[:, s_ind:, :], axis=(-2, -1))
            
            n_band_mean = np.average(field[:, n_ind:, :], 
                                     weights=weights[:, n_ind:, :], axis=-2)
            s_band_mean = np.average(field[:, :s_ind+1, :], 
                                     weights=weights[:, :s_ind+1, :], axis=-2)
            wo_n_band_mean = np.average(field[:, :n_ind, :], 
                                        weights=weights[:, :n_ind, :], axis=-2)
            wo_s_band_mean = np.average(field[:, s_ind:, :], 
                                        weights=weights[:, s_ind:, :], axis=-2)
        elif np.ndim(field) == 2:
            n_mean = np.average(field[n_ind:, :], weights=weights[n_ind:, :])
            s_mean = np.average(field[:s_ind+1, :], weights=weights[:s_ind+1, :])
            wo_n_mean = np.average(field[:n_ind, :], weights=weights[:n_ind, :])
            wo_s_mean = np.average(field[s_ind:, :], weights=weights[s_ind:, :])
            
            n_band_mean = np.average(field[n_ind:, :], weights=weights[n_ind:, :], axis=-2)
            s_band_mean = np.average(field[:s_ind+1, :], weights=weights[:s_ind+1, :], axis=-2)
            wo_n_band_mean = np.average(field[:n_ind, :], weights=weights[:n_ind, :], axis=-2)
            wo_s_band_mean = np.average(field[s_ind:, :], weights=weights[s_ind:, :], axis=-2)
        # end if elif
    
        # get the weighted average of the regions between both indices if it exists
        eq_mean = 0
        eq_band_mean = np.zeros(len(lons))
        if np.ndim(field) == 3:
            eq_mean = np.zeros(np.shape(field)[0])
        # end if            
        if lat > 0:
            if np.ndim(field) == 3:
                eq_mean = np.average(field[:, s_ind+1:n_ind, :], weights=weights[:, s_ind+1:n_ind, :], axis=(-2, -1))
                eq_band_mean = np.average(field[:, s_ind+1:n_ind, :], weights=weights[:, s_ind+1:n_ind, :], axis=-2)
            elif np.ndim(field) == 2:
                eq_mean = np.average(field[s_ind+1:n_ind, :], weights=weights[s_ind+1:n_ind, :])
                eq_band_mean = np.average(field[s_ind+1:n_ind, :], weights=weights[s_ind+1:n_ind, :], axis=-2)
        # end if
    # end if else

    if verbose:
        print(f'north index: {n_ind}\n')
        print(f'north latitude: {lats[n_ind]}\n')
        print(f'south index: {s_ind}\n')
        print(f'south latitude: {lats[s_ind]}\n')
    # end if
        
    # put together the result in a dictionary
    result = dict()
    
    if return_dic == "new":
        result["n_mean"] = n_mean
        result["s_mean"] = s_mean
        result["wo_n_mean"] = wo_n_mean
        result["wo_s_mean"] = wo_s_mean    
        result["eq_mean"] = eq_mean
        
        if not from_zonal:
            result["n_band_mean"] = n_band_mean
            result["s_band_mean"] = s_band_mean
            result["wo_n_band_mean"] = wo_n_band_mean
            result["wo_s_band_mean"] = wo_s_band_mean    
            result["eq_band_mean"] = eq_band_mean
        # end if        
    elif return_dic == "old":
        result = {"n_mean":n_mean, "s_mean":s_mean, "eq_mean":eq_mean}
        # result["n_mean"] = n_mean
        # result["s_mean"] = s_mean
        # sult["eq_mean"] = eq_mean
    # end if elif        
        
    return result
    
# end function lat_mean()
    