import numpy as np
from pywave.scattering.helmholtz_1d import Helmholtz1D
from pywave.scattering.open_water_base import OpenWaterBase


class ExtendedShallowWater(Helmholtz1D, OpenWaterBase):
    """
    Class for scattering of linear shallow water waves.
    Extended shallow water of Porter (2019)
    """

    def __init__(self, xlim=None, **kwargs):
        """
        Solves Helmholtz equation
        0 = beta*q_xx + (alpha/h)*q
          = beta*(q_xx + (alpha/(h*beta))*q)
        k = sqrt(alpha/(h*beta))

        Parameters:
        -----------
        rho_water : float
            density of water (kg/m^3)
        depth : float
            depth of water (m)
        period : float
            wave period (s)
        gravity : float
            gravitational acceleration (m/s^2)
        xlim : array-like
            (x0, x1), where x0 is the LHS limit and x1 is the RHS limit
            default is (-numpy.Inf, numpy.Inf)
        """
        OpenWaterBase.__init__(self, **kwargs)
        beta = self.beta
        super().__init__(helmholtz_coef=beta,
                k = np.sqrt(self.wave_number_ow_id/(beta * self.depth)),
                xlim=xlim)
        assert(beta > 0)

    @property
    def wave_number_ow_id(self):
        """
        Returns:
        --------
        alpha : float
            infinite depth wave number for open water (omega^2/g)
        """
        return self.omega**2/self.gravity

    @property
    def beta(self):
        """
        Returns:
        --------
        beta : float
            beta coefficient in Helmholtz equation for q=hU
            (beta*q_x)_x + alpha*q = 0
        """
        return 1 - (self.wave_number_ow_id * self.depth)/3

    @property
    def group_velocity(self):
        """
        Returns:
        --------
        cg : float
            group velocity
        """
        return self.beta * self.phase_velocity

    def set_operators(self):
        """
        set common operators for convenience

        Sets:
        -----
        self.operators
        """
        super().set_operators() # q=hU is helmholtz_u
        self.operators['displacement'] = lambda k : -_ZI*k
        self.operators['horizontal_velocity'] = (
                lambda k : (-_ZI*self.omega)/self.depth)
