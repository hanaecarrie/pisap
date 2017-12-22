""" Pickle.
"""

# Third party import
import pickle


def save_object(obj, filename):
    """Save object into pickle format
    ---------
    Inputs:
    obj -- variable, name of the object in the current workspace
    filename -- string, filename to save the object
    ---------
    Outputs:
    nothing
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load object saved into pickle format
    ---------
    Inputs:
    filename -- string, path where the pickle object is saved
    ---------
    Outputs:
    nothing
    """
    with open(filename, 'rb') as pfile:
        return pickle.load(pfile)
