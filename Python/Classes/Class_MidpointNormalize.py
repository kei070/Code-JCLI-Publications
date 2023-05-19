"""
The colorbar normalisation class written by Joe Kington
From:
    http://chris35wills.github.io/matplotlib_diverging_colorbar/

It is not used in any of the plots in the JCLI papers.
"""

import matplotlib.colors as colors
import numpy as np


class MidpointNormalize(colors.Normalize):
    
    """
    Normalise the colorbar so that diverging bars work their waz either side from a prescribed modpoint value
    """
        
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
        
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
        
# end class MidpointNormalize()
