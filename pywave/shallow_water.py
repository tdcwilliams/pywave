from pywave.helmholtz_1d import Helmholtz1d

class ShallowWater(Helmholtz1d):
    """ Class for scattering of linear shallow water waves """

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
            (x0, x1), where x0 is the LHS limit and x1 si the RHS limit
            default is (-numpy.Inf, numpy.Inf)
        """
        self.period = period
        self.rho_water = rho_water
        self.depth = depth
        self.gravity = gravity
        super().__init__(kappa = depth*gravity, alpha=self.omega**2, xlim=xlim)
