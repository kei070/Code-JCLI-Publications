import sys


def print_one_line(statement):

    """
    Function to print consecutive statements in the same line.
    Note: There will be no space between the single statements.
    
    Parameters:
        :param statement: The statement to be printed. Note that this has to be
                          a string.
    
    Return:
        :return: Nothing is returned.
    """

    sys.stdout.write(statement)
    sys.stdout.flush()

# end function
