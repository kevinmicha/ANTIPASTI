def remove_abc(residue):
    """
    Returns the residue names without the final letter that indicates extension positions.

    """
    if residue[-1] != ' ':
        residue = str(residue[:-1]) + '.' + '{0:0=2d}'.format(ord(residue[-1])-64)
    return float(residue)