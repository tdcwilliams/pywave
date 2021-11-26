import numpy as np


def diffl(u):
    d = np.zeros_like(u)
    d[1:] = np.diff(u)
    return d

    
def diffr(u):
    d = np.zeros_like(u)
    d[:-1] = np.diff(u)
    return d


def suml(u):
    s = np.zeros_like(u)
    s[0] = 2*u[0]
    s[1:] = u[1:] + u[:-1]
    return s

    
def sumr(u):
    s = np.zeros_like(u)
    s[:-1] = u[:-1] + u[1:]
    s[-1] = 2*u[-1]
    return s


def vector_min(u, v):
    """
    min function that works for vectors as well as floats
    
    Parameters:
    -----------
    u : float or numpy.ndarray
    v : float or numpy.ndarray
    
    Returns:
    --------
    min_uv : float or numpy.ndarray
    """
    return .5*(u+v - np.abs(u-v))


def vector_max(u, v):
    """
    max function that works for vectors as well as floats
    
    Parameters:
    -----------
    u : float or numpy.ndarray
    v : float or numpy.ndarray
    
    Returns:
    --------
    max_uv : float or numpy.ndarray
    """
    return .5*(u+v + np.abs(u-v))
