import numpy as np
#from functools import cached_property
from functools import lru_cache #TODO use cached_property when python >= 3.8 installed

from pywave.energy_transport.stationary_base import StationaryBase


class StationaryAnisoBase(StationaryBase):
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
        etot = ep + em
        edel = ep - em
        return 2 * (etot + self.beta * edel) + self.eps * edel


class StationaryAniso(StationaryAnisoBase):
    """ Solve the general problem (non-zero dissipation) """

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
        a = - self.eps - .5 * (1 + self.beta)
        b = - .5 * (1 - self.beta)
        c = - .5 * (1 + self.beta)
        d = self.eps - .5 * (1 - self.beta)
        return np.array([[a,b], [c,d]])


class StationaryAnisoNoDissipation(StationaryAnisoBase):
    """ 
    Solve the zero dissipation problem analytically.
    Mainly to check the general solution method.
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
    def alpha(self):
        """
        Returns:
        --------
        alpha : float
            convenience variable (1 - beta)/(1 + beta)
        """
        b = self.beta
        return (1 - b)/(1 + b)

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
        ex = np.exp(-self.w)
        c2 = ex/(1 + self.alpha * ex)
        c1 = 1 - c2 * self.alpha
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
        cex = c1 * np.exp(-x)
        ep = cex + c2 * self.alpha
        em = cex - c2
        return np.array([ep, em])
