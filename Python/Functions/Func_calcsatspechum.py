import numpy as np
import numexpr as ne


def calcsatspechum(t, p):
    """
    T is temperature, P is pressure in hPa 
    
    From Pendergrass et al. (2018). Does this yield the same result as the mixhum_ptd()?
    """
    ## Formulae from Buck (1981):
    es = (1.0007 + (3.46E-6 * p)) * 6.1121 * np.exp(17.502 * (t - 273.15) / (240.97 + (t - 273.15)))
    wsl = 0.622 * es / (p - es) # saturation mixing ratio wrt liquid water (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    es = (1.0003 + (4.18e-6 * p)) * 6.1115 * np.exp(22.452 * (t - 273.15) / (272.55 + (t - 273.15)))
  
    wsi = 0.622 * es / (p - es) # saturation mixing ratio wrt ice (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    ws = wsl
    
    ws[t < 273.15] = wsi[t < 273.15]
   
    qs = ws / (1 + ws)  # saturation specific humidity, kg/kg
    
    return qs

# end function calcsatsechum()


def da_calcsatspechum(t, p):
    """
    T is temperature, P is pressure in hPa 
    
    From Pendergrass et al. (2018). Does this yield the same result as the mixhum_ptd()?
    """
    ## Formulae from Buck (1981):
    es = (1.0007 + (3.46E-6 * p)) * 6.1121 * np.exp(17.502 * (t - 273.15) / (240.97 + (t - 273.15)))
    wsl = 0.622 * es / (p - es) # saturation mixing ratio wrt liquid water (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    es = (1.0003 + (4.18e-6 * p)) * 6.1115 * np.exp(22.452 * (t - 273.15) / (272.55 + (t - 273.15)))
  
    wsi = 0.622 * es / (p - es) # saturation mixing ratio wrt ice (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    ws = wsl * (t >= 273.15)
    
    ws = ws + wsi * (t < 273.15)
   
    qs = ws / (1 + ws)  # saturation specific humidity, k/kg
    
    return qs

# end function calcsatsechum()


def ne_calcsatspechum(t, p):
    """
    T is temperature, P is pressure in hPa 
    
    From Pendergrass et al. (2018). Does this yield the same result as the mixhum_ptd()?
    """
    ## Formulae from Buck (1981):
    es = ne.evaluate("(1.0007 + (3.46E-6 * p)) * 6.1121 * exp(17.502 * (t - 273.15) / (240.97 + (t - 273.15)))")
    wsl = ne.evaluate("0.622 * es / (p - es)") # saturation mixing ratio wrt liquid water (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    es = ne.evaluate("(1.0003 + (4.18e-6 * p)) * 6.1115 * exp(22.452 * (t - 273.15) / (272.55 + (t - 273.15)))")
  
    wsi = ne.evaluate("0.622 * es / (p - es)") # saturation mixing ratio wrt ice (kg/kg)
    # THIS SHOULD be kg/kg; for g/kg it is 622 and not .622
    
    # ws = ne.evaluate("wsl * (t >= 273.15)")
    
    ws = ne.evaluate("wsl * (t >= 273.15) + wsi * (t < 273.15)")
   
    qs = ne.evaluate("ws / (1 + ws)")  # saturation specific humidity, kg/kg
    
    return qs

# end function calcsatsechum()