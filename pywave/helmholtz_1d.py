import numpy as np
from pywave.medium import Medium

_ZI = np.complex(0, 1)


class Helmholtz1D(Medium):
    """ Properties of an elastic string """

    def __init__(self, kappa=1, alpha=1, xlim=None):
        """
        Class to manage medium that satisfies Helmoltz equation in 1D
        kappa*u_{xx} + alpha*u = 0
        
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
        self.alpha = alpha
        self.kappa = kappa
        super().__init__(xlim=xlim)


    def solve_disprel(self):
        """
        Solve dispersion relation and set the wave number vector

        Sets:
        --------
        self.k : numpy.ndarray(float)
            vector of wave numbers (1/m)
            satisfies
                alpha - kappa*k^2 = 0
        """
        self.k = np.array([np.sqrt(self.alpha/self.kappa)])


    def set_operators(self):
        """
        Set some operators for convenience

        Sets:
        -----
        self.operators : dict
        """
        self.operators = dict(
                helmholtz_u=lambda k : 1,
                helmholtz_Kux=lambda k : _ZI*self.kappa*k,
                )


    def set_edge_operators(self):
        """
        Set edge operators to be used in the edge conditions

        Sets:
        -----
        self.edge_operators : dict
            keys are strings (names)
            values are tuples
            (op1,op2)
        """
        self.edge_operators = dict(
                displacement=(
                    self.operators['helmholtz_Kux'],
                    self.operators['helmholtz_u'],
                    )
                )

    
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
