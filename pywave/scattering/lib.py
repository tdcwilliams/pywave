import numpy as np


def vector_min(x, y):
    """
    vector-friendly min function

    Parameters:
    -----------
    x : float or numpy.ndarray
    y : float or numpy.ndarray

    Returns:
    --------
    min : float or numpy.ndarray
        minimum of x and y
        - float if x and y are both float
        - numpy.ndarray otherwise
    """
    return .5*(x + y) - .5*np.abs(x - y) 


def vector_max(x, y):
    """
    vector-friendly max function

    Parameters:
    -----------
    x : float or numpy.ndarray
    y : float or numpy.ndarray

    Returns:
    --------
    max : float or numpy.ndarray
        maximum of x and y
        - float if x and y are both float
        - numpy.ndarray otherwise
    """
    return .5*(x + y) + .5*np.abs(x - y) 


def conservative_remapping_weights(x_src, x_dst):
    """
    Get remapping weights to go between two 1D grids

    Parameters:
    -----------
    x_src : numpy.ndarray
        1D source grid (len=M+1) - ends of elements
    x_dst : numpy.ndarray
        1D destination grid (len=N+1) - ends of elements

    Returns:
    --------
    w : numpy.ndarray
        2D matrix (M,N)
    """
    m = len(x_src) - 1
    n = len(x_dst) - 1
    w = np.zeros((m,n))
    xl = x_dst[:-1]
    xr = x_dst[1:]
    for i in range(m):
        x0, x1 = x_src[i:i+2]
        overlap = (xr > x0) * (xl < x1)
        w[i,overlap] = (vector_min(xr[overlap], x1)
                - vector_max(xl[overlap], x0))
    return w
