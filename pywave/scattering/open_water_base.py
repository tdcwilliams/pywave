from pywave.scattering.medium import Medium


class OpenWaterBase(Medium):

    def __init__(self, rho_water=1025, depth=100, period=20, gravity=9.81, xlim=None):
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
        self.period = period
        self.rho_water = rho_water
        self.depth = depth
        self.gravity = gravity
        super().__init__(xlim=xlim)


    @property
    def wave_number_ow_id(self):
        """
        Returns:
        --------
        alpha : float
            infinite depth wave number for open water (omega^2/g)
        """
        return self.omega**2/self.gravity
