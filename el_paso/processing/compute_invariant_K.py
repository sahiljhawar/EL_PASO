import numpy as np
from astropy import units as u

def compute_invariant_K(bmirr, XJ):
    # calculate invariant K
    return np.sqrt(bmirr) * XJ
