"""
Function for rounding to the next number divisible by 5.
"""


def round_to_5(x, round_dir=""):
    
    """
    Parameters:
        :param x: Value to round to some multiple of 5.
        :param round_dir: String. 'Rounding direction' with the three possible values "off" for rounding off to the next
                          lower value divisible by 5, "up" of rounding up, and "" (empty string) for rounding normally.
                          Note that 'lower' here means lower with respect to the absolute value.
    """
    
    # make sure the behaviour is the same for positive and negative numbers
    sign = 1
    if x < 0:
        x = -x
        sign = -1
    # end if
    
    # if x modulo 5 is > 2 round up, else round off
    add = 0
    if (round_dir == "") & (x % 5 > 2):
        add = 5
    # end if
    
    if round_dir == "off":
        add = 0
    elif round_dir == "up":
        if x % 5 != 0:
            add = 5
        else:
            add = 0
        # end if else
    # end if elif
    
    return (int(x / 5) * 5 + add) * sign
    
# end function round_to_5()
