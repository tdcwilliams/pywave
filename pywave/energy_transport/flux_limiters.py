import numpy as np

from pywave.energy_transport.lib import (
        vector_min, vector_max)

def limiter_van_leer(r):
    """
    van Leer flux limiter
    
    Parameters:
    -----------
    r : numpy.ndarray
        ratio of rhs to lhs gradient
    
    Returns:
    --------
    phi : numpy.ndarray
        weighting for high-order flux
        - 1 means use high-order flux, 0 means use first-order upwind
    """
    return (r + np.abs(r))/(1 + np.abs(r))


def limiter_superbee(r):
    """
    Superbee flux limiter
    
    Parameters:
    -----------
    r : numpy.ndarray
        ratio of rhs to lhs gradient
    
    Returns:
    --------
    phi : numpy.ndarray
        weighting for high-order flux
        - 1 means use high-order flux, 0 means use first-order upwind
    """
    return vector_max(0, vector_max(
        vector_min(1, 2*r), vector_min(r, 2)
        ))


# dictionary for easier access from outside
LIMITERS = {
    None : (lambda x : 1), # no limiting
    'van_leer' : limiter_van_leer,
    'superbee' : limiter_superbee,
    }
