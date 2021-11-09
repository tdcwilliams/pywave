from pywave.helmholtz_1d import Helmholtz1D

class ExtendedShallowWater(Helmholtz1D):
    """
    Class for scattering of linear shallow water waves.
    Extended shallow water of Porter (2019)
    """

    def __init__(self, rho_water=1025, depth=100, period=20, gravity=9.81, xlim=None):
        """
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
        self.period = period
        self.rho_water = rho_water
        self.depth = depth
        self.gravity = gravity
        kow = self.omega**2/g
        super().__init__(kappa=self.beta,
                alpha=self.wave_number_ow_id/self.depth, xlim=xlim)


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

