import numpy as np
from medium import Medium

_ZI = np.complex(0, 1)


class WaveMedium(Medium):
    """ Properties of a wave medium (periodic in time dimension) """

    def __init__(self, period=10, xlim=None):
        """
        Class for a general wave medium
        
        Parameters:
        -----------
        xlim : array-like
            (x0, x1), where x0 is the LHS limit and x1 si the RHS limit
            default is (-numpy.Inf, numpy.Inf)
        """
        self.period = period
        self.set_limits(xlim)
        self.solve_disprel()
        self.set_operators()
        self.set_edge_operators()


    @property
    def omega(self):
        """
        Returns:
        --------
        omega : float
            radial wave frequency (1/s)
            = 2\pi/period
        """
        return 2*np.pi/self.period # 1/s
