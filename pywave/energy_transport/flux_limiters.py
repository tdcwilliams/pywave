import numpy as np

from pywave.energy_transport.lib import (
        vector_min, vector_max)

def limiter_van_leer(theta):
    """
    van Leer flux limiter
    
    Parameters:
    -----------
    theta : numpy.ndarray
    
    Returns:
    --------
    phi : numpy.ndarray
        weighting for high-order flux
        - 1 means use high-order flux, 0 means use first-order upwind
    """
    return (theta + np.abs(theta))/(1 + np.abs(theta))


def limiter_superbee(theta):
    """
    Superbee flux limiter
    
    Parameters:
    -----------
    theta : numpy.ndarray
    
    Returns:
    --------
    phi : numpy.ndarray
        weighting for high-order flux
        - 1 means use high-order flux, 0 means use first-order upwind
    """
    return vector_max(0, vector_max(
        vector_min(1, 2*theta), vector_min(theta, 2)
        ))


# dictionary for easier access from outside
LIMITERS = {
    None : (lambda x : 1), # no limiting
    'van_leer' : limiter_van_leer,
    'superbee' : limiter_superbee,
    }
