import numpy as np


def conv_kgkg_gm3(c, t, p, q):

    """
    Function for converting specific CLI or CLW in kg/kg to g/m^3.
    Note the (for me quite unintuitive) definition of the half-levels in ICON (see documentation of the function
    calc_icon_z_intervals().
    Note that the formula used here is basically taken from the RTTOV-12 SVR v1.0 p. 18.
    Note further that RTTOV's kg/kg and ppmv units hold for moist air (see RTTOV-12 User's Guide v1.2 p. 16, 40, 131).
    :param c: CLI or CLW in kg/kg as 3d-array.
    :param t: temperature in K as 3d-array.
    :param p: pressure in Pa as 3d-array.
    :param q: specific humidity in kg/kg as 3d-array.

    :return: CLI or CLW in g/m^3 depending on the input
    """

    R = 8.314  # J/mol/K universal gas constant -> units equivalent to m^3*Pa/K/mol

    Mda = 28.9647  # g/mol molecular mass of dry air

    Rda = R / Mda  # J/g/K gas constant for dry air

    Mwv = 18.01528  # g/mol molecular mass of water (vapour)

    eps = Mwv / Mda

    Rma = Rda * (1 + ((1 - eps) / eps) * q)  # J/g/K

    # do the calculation
    d = p / t / Rma

    # calculate the CLI/CLW per volume
    c_out = c * d

    # return
    return c_out

# end function conv_kgkg_gm3()

