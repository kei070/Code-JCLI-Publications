import sys
import numpy as np
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest

def remap(lons_t, lats_t, lons_o, lats_o, field_o, rad_of_i=150000, epsilon=0.5, fill_value=None):
    
    """
    Function for regridding the SST/TOS model data to a requested model grid.
    
    See: https://stackoverflow.com/questions/35734070/interpolating-data- \
    from-one-latitude-longitude-grid-onto-a-different-one/35734381#35734381
    
    And: https://pyresample.readthedocs.io/en/latest/swath.html
    
    Parameters:
        lons_t: 2d array. Longitudes of the target grid. Range -180 to 180.
        lats_t: 2d array. Latitudes of the target grid. Range -90 to 90.
        lons_o: 2d array. Longitudes of the origin grid. Range -180 to 180.
        lats_o: 2d array. Latitudes of the origin grid. Range -90 to 90.
        field_o: 2d array. Origin field to be regridded.
        
        For the following parameters also see the above URLs:
        rad_of_i: Integer. Radius of influence in meters for the interpolation.
        epsilon: Float. Uncertainty in the distance measure.
        fill_value: Value for undefined cells. Use fill_value=None if the imput
                    is a masked array and the mask shall be applied.
                    
    Return:
        2d array. The remapped field on the respective input grid.
    """
    
    # check if the array have the correct shapes
    if (len(np.shape(lons_t)) != 2) | (len(np.shape(lats_t)) != 2):
        print("\nTarget lat or lon have wrong shape! Terminating.\n")
        sys.exit(0)
    # end if
    
    if (len(np.shape(lons_o)) != 2) | (len(np.shape(lats_o)) != 2):
        print("\nOrigin lat or lon have wrong shape! Terminating.\n")
        sys.exit(0)
    # end if
    
    if len(np.shape(field_o)) != 2:
        print("\nOrigin field has wrong shape! Terminating.\n")
        sys.exit(0)
    # end if
        
    
    target_gr = SwathDefinition(lons=lons_t, lats=lats_t)
    origin_gr = SwathDefinition(lons=lons_o, lats=lats_o)

    field_t = resample_nearest(origin_gr, field_o, target_gr, 
                               radius_of_influence=150000, epsilon=0.5, 
                               fill_value=-9999)
    
    return field_t

# end function remap
    