def remove_abc(residue):
    """
    Returns the residue names without the final letter that indicates extension positions.

    """
    return residue[:-1]

def extract_script_result(output_string):
    """
    Returns the heavy and light chain lengths from a string provided by an R script.

    """
    output_string = output_string.decode('utf-8')
    h_pos = output_string.find('[1] ') + 4
    l_pos = output_string.rfind('[1] ') + 4

    return output_string[h_pos:h_pos+2], output_string[l_pos:l_pos+2]