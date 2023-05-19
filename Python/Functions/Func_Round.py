"""
Functions for rounding to 2, 3, and 4 digits.
"""

import numpy as np


def r2(value):
    return np.around(value, 2)
# end def

def r3(value):
    return np.around(value, 3)
# end def

def r4(value):
    return np.around(value, 4)
# end def