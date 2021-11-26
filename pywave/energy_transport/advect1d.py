import numpy as np

from pywave.energy_transport.lib import (
        diffl, diffr, sumr)
from pywave.energy_transport.flux_limiters import LIMITERS


class Advect1D:
    
    def __init__(self, dx, dt, scheme='lax_wendroff', limiter=None):
        self.dx = dx
        self.dt = dt
        self.flux = {
            'first_order_upwind' : self.flux_fou,
            'lax_friedrichs' : self.flux_lax_friedrichs,
            'lax_wendroff' : self.flux_lax_wendroff,
        }[scheme]
        self.limiter = LIMITERS[limiter]


    def flux_fou(self, u, c, *args):
        """
        flux for first order upwind
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        dummy args
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        return c*u


    def flux_lax_friedrichs(self, u, c, *args):
        """
        flux for Lax-Friedrichs - central diference, 1st order
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        dummy args
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        r = self.dt/self.dx
        f = c*u
        return .5*(sumr(f)-r*diffr(f))


    def flux_lax_wendroff(self, u, c, *args):
        """
        flux for Lax-Wendroff - direct space-time, 2nd order
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        dummy args
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        r = self.dt/self.dx
        f = c*u
        return .5*( sumr(f) - c*r*diffr(f) )


    def limited_flux(self, u, c, *args):
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
        args for self.flux
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        theta = diffl(u)/(diffr(u)+3.e-14)
        phi = self.limiter(theta)
        return c*u + phi*(self.flux(u, c, *args) - c*u)
 

    def advect(self, u, c, *args):
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
        # do we need to allow for negative velocity
        if isinstance(c, np.ndarray):
            same_dir = np.all(c>0)
            if not same_dir: c_ = -c[::-1]
        else:
            same_dir = (c>0)
            if not same_dir: c_ = -c

        if same_dir:
            s = slice(None, None, None)
            u_ = u
            c_ = c
        else:
            s = slice(None, None, -1)
            u_ = u[s]
        f  = self.limited_flux(u_, c_, *args)
        return (u_ - self.dt*diffl(f)/self.dx)[s]
