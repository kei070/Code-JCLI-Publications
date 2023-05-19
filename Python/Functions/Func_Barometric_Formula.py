"""
Python implementation of the barometric formula to calculate height from pressure and pressure from height.
"""

# imports
import numpy as np


def press(height, p0=1013, temp=288):

    """
    Parameters:
        :param height: Height in m.
        :param p0: Surface pressure in hPa.
        :param temp: Surface temperature in K.
    
    Returns pressure in hPa.
    """
    
    # constant parameters
    g = 9.81  # m/s^2; gravitational acceleration
    r = 8.314  # J/K/mol; gas constant
    m = 0.02896  # mol/kg; mass of air molecule
    
    return p0 * np.exp((-m * g * height) / (r * temp))
    
# end def press
    
    
def height(press, p0=1013, temp=288):
    
    """
    Parameters:
        :param press: Pressure in hPa.
        :param p0: Surface pressure in hPa.
        :param temp: Surface temperature in K.
        
    Returns height in m.
    """
    
    # constant parameters
    g = 9.81  # m/s^2; gravitational acceleration
    r = 8.314  # J/K/mol; gas constant
    m = 0.02896  # mol/kg; mass of air molecule
    
    return -np.log(press / p0) * r * temp / (m * g)
    
# end def height
    
