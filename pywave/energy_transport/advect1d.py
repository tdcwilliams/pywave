import numpy as np


class Advect1D:
    
    def __init__(self, dx, dt, scheme='lax_wendroff', limiter=None):
        self.dx = dx
        self.dt = dt
        self.flux = {
            'first_order_upwind' : self.flux_fou,
            'lax_friedrichs' : self.flux_lax_friedrichs,
            'lax_wendroff' : self.flux_lax_wendroff,
        }[scheme]
        self.limiter = {
            None : (lambda x : 1), # no limiting
            'van_leer' : self.limiter_van_leer,
            'superbee' : self.limiter_superbee,
            }[limiter]


    @staticmethod
    def diffl(u):
        d = np.zeros_like(u)
        d[1:] = np.diff(u)
        return d
    
        
    @staticmethod
    def diffr(u):
        d = np.zeros_like(u)
        d[:-1] = np.diff(u)
        return d


    @staticmethod
    def suml(u):
        s = np.zeros_like(u)
        s[0] = 2*u[0]
        s[1:] = u[1:] + u[:-1]
        return s
    
        
    @staticmethod
    def sumr(u):
        s = np.zeros_like(u)
        s[:-1] = u[:-1] + u[1:]
        s[-1] = 2*u[-1]
        return s

    
    @staticmethod
    def min(u, v):
        """
        min function that works for vectors as well as floats
        
        Parameters:
        -----------
        u : float or numpy.ndarray
        v : float or numpy.ndarray
        
        Returns:
        --------
        min_uv : float or numpy.ndarray
        """
        return .5*(u+v - np.abs(u-v))


    @staticmethod
    def max(u, v):
        """
        max function that works for vectors as well as floats
        
        Parameters:
        -----------
        u : float or numpy.ndarray
        v : float or numpy.ndarray
        
        Returns:
        --------
        max_uv : float or numpy.ndarray
        """
        return .5*(u+v + np.abs(u-v))


    def flux_fou(self, u, c):
        """
        flux for first order upwind
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        return c*u


    def flux_lax_friedrichs(self, u, c):
        """
        flux for Lax-Friedrichs - central diference, 1st order
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        r = self.dt/self.dx
        f = c*u
        return .5*(self.sumr(f)-r*self.diffr(f))


    def flux_lax_wendroff(self, u, c):
        """
        flux for Lax-Wendroff - direct space-time, 2nd order
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        r = self.dt/self.dx
        f = c*u
        return .5*( self.sumr(f) - c*r*self.diffr(f) )


    @staticmethod
    def limiter_van_leer(theta):
        """
        van Leer flux limiter
        
        Parameters:
        -----------
        theta : numpy.ndarray
        
        Returns:
        --------
        phi : numpy.ndarray
            weighting for high-order flux
            - 1 means use high-order flux, 0 means use first-order upwind
        """
        return (theta + np.abs(theta))/(1 + np.abs(theta))
    
 
    def limiter_superbee(self, theta):
        """
        Superbee flux limiter
        
        Parameters:
        -----------
        theta : numpy.ndarray
        
        Returns:
        --------
        phi : numpy.ndarray
            weighting for high-order flux
            - 1 means use high-order flux, 0 means use first-order upwind
        """
        return self.max(0, self.max(
            self.min(1, 2*theta), self.min(theta, 2)
            ))


    def limited_flux(self, u, c):
        """
        Return limited flux to prevent oscillations from high-order flux.
        The limiting can be deactivated by instantiating object with limiter=None
        (then self.limiter=1 so only the high-order flux is used).
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        theta = self.diffl(u)/(self.diffr(u)+3.e-14)
        phi = self.limiter(theta)
        return c*u + phi*(self.flux(u, c) - c*u)
 

    def step(self, u, c, alpha):
        """
        Advect u for one time step.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        f  = self.limited_flux(u, c)
        return (u - self.dt*self.diffl(f)/self.dx)
