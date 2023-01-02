import numpy as np
#from functools import cached_property
from functools import lru_cache #TODO use cached_property when python >= 3.8 installed

from pywave.energy_transport.stationary_base import StationaryBase


class StationaryIsoBase(StationaryBase):
    def calc_wave_stress(self, x):
        """
        Parameters:
        -----------
        x : numpy.ndarray
            x values where energy should be calculated.
            Shape is (nx,)
        
        Returns:
        --------
        tau_x : numpy.ndarray
            Wave stress on ice.
            Shape is (nx,)
        """
        ep, em = self.calc_expansion(x)
        return (2 * self.beta + self.eps) * (ep - em)


class StationaryIso(StationaryIsoBase):

    #@cached_property
    @property
    @lru_cache()
    def matrix(self):
        """
        Returns:
        --------
        matrix : numpy.ndarray
            Matrix A in equation E_x = A E
        """
        a = - self.eps - self.beta
        b = self.beta
        c = - self.beta
        d = self.eps + self.beta
        return np.array([[a,b], [c,d]])


class StationaryIsoNoDissipation(StationaryIsoBase):
    """ 
    Solve the zero dissipation problem analytically, since our general solution
    doesn't work for repeated eigenvalues.
    """
    def __init__(self, w=.1, beta=0):
        """
        Parameters:
        -----------
        w : float
            MIZ width (>0)
        beta : float
            directional dependence parameter (-1<=beta<=1)
        """
        self.w = w
        self.beta = beta
        self.eps = 0. #no dissipation

    #@cached_property
    @property
    @lru_cache()
    def coeffs(self):
        """
        Returns:
        --------
        coeffs: numpy.ndarray
            coefficients of eigenfunction expansion. Shape is (2,)
        """
        c2 = 1 / (2 + self.beta * self.w)
        c1 = 1 - c2
        return np.array([c1, c2])

    def calc_expansion(self, x):
        """
        Parameters:
        -----------
        x : numpy.ndarray
            x values where energy should be calculated.
            Shape is (nx,)
        
        Returns:
        --------
        E : numpy.ndarray
            energy calculated at x.
            Shape is (2,nx)
            - 1st row is E_+ (energy going to the right)
            - 2nd row is E_- (energy going to the left)
        """
        c1, c2 = self.coeffs
        bx = self.beta * x
        ep = c1 + c2 * (1 - bx)
        em = c1 - c2 * (1 + bx)
        return np.array([ep, em])
