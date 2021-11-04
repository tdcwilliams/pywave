import numpy as np

_ZI = np.complex(0, 1)


class ElasticString(Medium):
    """ Properties of an elastic string """

    def __init__(self, m=1, kappa=4, period=3, xlim=None):
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
        super().__init__(period=period, xlim=xlim)


    def solve_disprel(self):
        """
        Solve dispersion relation and set the wave number vector

        Sets:
        --------
        k : numpy.ndarray(float)
            vector of wave numbers (1/m)
              = 2\pi/[wavelength]
              = [omega/c]
            where
            c = sqrt(kappa/m) is the phase velocity
        """
        c = np.sqrt(self.kappa/self.m)
        self.k = np.array([self.omega/c])

    
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


    def get_power(self, a0, a1):
        """
        Determine the power input to a segment of string
        Positive power input corresponds to energy travelling into
        the plane with normal vector being the positive x axis (i.e 
        the energy is travelling in the direction of the negative x axis)

        Parameters:
        -----------
        a0 : numpy.ndarray
            coeffients of the waves travelling to the right
        a1 : numpy.ndarray
            coeffients of the waves travelling to the left

        Returns:
        --------
        power : float
            power input from the right.
        """
        fac = .5*self.omega*self.k[0]*self.kappa
        return fac*(np.abs(a1[0])**2 - np.abs(a0[0])**2)
