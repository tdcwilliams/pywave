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

    def get_energies(self, a0, a1):
        """
        Determine the energies travelling in each direction

        Parameters:
        -----------
        a0 : numpy.ndarray
            coeffients of the waves travelling to the right
        a1 : numpy.ndarray
            coeffients of the waves travelling to the left

        Returns:
        --------
        e0 : float
            energy travelling to the right
        e1 : float
            energy travelling to the left
        """
        fac = .5 * self.rho_water * self.gravity
        fk = self.operators["displacement"](self.k[0])
        return [fac*np.abs(fk * a)**2 for a in (a0, a1)]
