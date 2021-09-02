class ElasticString:
    """ Properties of an elastic string """

    def __init__(self, m=1, kappa=4, period=3, xlim = None):
        """
        Class to manage string properties
        
        Parameters:
        -----------
        m : float
            mass per unit length (kg/m)
        kappa : float
            string stiffness (Pa/m)
        period : float
            wave period (s)
        xlim : array-like
            (x0, x1), where x0 is the LHS limit and x1 si the RHS limit
            default is (-numpy.Inf, numpy.Inf)
        """
        self.m = m
        self.kappa = kappa
        self.period = period
        self.infinite = False
        self.semi_infinite = False
        if xlim is None or tuple(xlim) == (-np.Inf, np.Inf):
            self.infinite = True
            self.xlim = np.array([-np.Inf, np.Inf])
        else:
            self.xlim = np.array(xlim)
            self.semi_infinite = not np.all(np.isfinite(xlim))
        assert(self.xlim[0] < self.xlim[1])
        assert(len(self.xlim) == 2)


    @property
    def params(self):
        """ return the main parameters (not position)
        
        Returns:
        --------
        params : dict
        """
        return dict(
            m=self.m,
            kappa=self.kappa,
            period=self.period,
        )


    @property
    def c(self):
        """
        Returns:
        --------
        c : float
            wave speed (m/s)
        """
        return np.sqrt(self.kappa/self.m)
    
    
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


    @property
    def k(self):
        """
        Returns:
        --------
        k : numpy.array(float)
            wave frequency (1/m)
            = 2\pi/wavelength
        """
        return np.array([self.omega/self.c])

    
    @property
    def num_modes(self):
        """
        Returns:
        --------
        num_modes : int
            number of wave modes
        """
        return len(self.k)

    
    @property
    def phase_matrix(self):
        """
        Returns:
        --------
        phases : numpy.ndarray
            vector with n-th element e^{i self.k[n] w} where w is the width and self.k
            is the vector of wave numbers
        """
        width = self.xlim[1] - self.xlim[0]
        if np.isfinite(width):
            return np.diag(np.exp(_ZI*width*self.k))
        
    
    def is_in_domain(self, x):
        """
        Parameters:
        -----------
        x : numpy.array(float)
        
        Returns:
        --------
        in_dom : numpy.array(bool)
            bool same size as x marking which members are in the domain
        """
        x0, x1 = self.xlim
        return (x>=x0) * (x<=x1)

    
    def get_expansion(self, x, a0, a1, get_disp=True):
        """
        Calculate a displacement profile

        Parameters:
        -----------
        x : numpy.array(float)
            positions to evaluate the displacement
        a0 : numpy.array(float)
            wave amplitudes for waves travelling from left to right
        a1 : numpy.array(float)
            wave amplitudes for waves travelling from right to left
        get_disp : bool
            get displacement if True, else get stress

        Returns:
        --------
        u : numpy.array(float)
            complex displacement U evaluated at x
        """
        u = np.full_like(x, np.nan, dtype=np.complex)
        b = self.is_in_domain(x)
        xb = x[b]
        x0, x1 = self.xlim
        if self.infinite:
            x0 = x1 = 0
        elif self.semi_infinite:
            x0 = x1 = self.xlim[np.isfinite(self.xlim)]
        c0 = a0 if get_disp else _ZI*self.kappa*self.k*a0
        c1 = a1 if get_disp else -_ZI*self.kappa*self.k*a1
        u[b] = np.exp(_ZI*np.outer(xb - x0, self.k)).dot(c0).flatten()  # (nx,nk) x (nk,1) = (nx,1)
        u[b] += np.exp(_ZI*np.outer(x1 - xb, self.k)).dot(c1).flatten() # (nx,nk) x (nk,1) = (nx,1)
        return u

    
    def get_new(self, xlim=None):
        """ get new class instance with same parameters but different spatial limits
        
        Parameters:
        -----------
        xlim : tuple(float)
            (x0, x1)
            
        Returns:
        --------
        new : new instance of same type as self
        """
        return self.__class__(**self.params, xlim=xlim)
