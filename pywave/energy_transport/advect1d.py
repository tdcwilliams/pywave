import numpy as np

from pywave.energy_transport.lib import (
        diffl, diffr, sumr)
from pywave.energy_transport.flux_limiters import LIMITERS


class Advect1D:
    
    def __init__(self, dx, dt, nghost=3, scheme='lax_wendroff', limiter=None):
        self.dx = dx
        self.dt = dt
        self.nghost = nghost
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
        r = diffl(u)/(diffr(u)+3.e-14) #ratio of rhs to lhs gradient
        phi = self.limiter(r)
        return c*u + phi*(self.flux(u, c, *args) - c*u)

    def check_u_c(self, u, c):
        """
        Check if we need to allow for negative velocity
        and return modified u,c to work in the different
        flux schemes (defined for advection to right)

        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity

        Returns:
        -----------
        u_tmp : numpy.ndarray
            quantity to be advected
        c_tmp : float or numpy.ndarray
            velocity
        same_dirn : bool
            True if advecting to right
        """
        if isinstance(c, np.ndarray):
            same_dirn = np.all(c>0)
            if not same_dirn: c_ = -c[::-1]
        else:
            same_dirn = (c>0)
            if not same_dirn: c_ = -c

        if same_dirn:
            return u, c, same_dirn
        return u[::-1], c_, same_dirn

    def apply_boundary_conditions(self, u, c, neumann=True, u_in=0):
        """
        Apply boundary conditions

        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        neumann : bool
            True : set u on the ghost cells to u[0]
                to give u_x=0 (Neumann) boundary conditions
            False : set u on the ghost cells to u_in
        u_in : float
            value of u coming from the left boundary

        Returns:
        -----------
        u_tmp : numpy.ndarray
            quantity to be advected
        c_tmp : float or numpy.ndarray
            velocity
        """
        shp = (len(u)+self.nghost,)
        u_ = np.zeros(shp)
        c_ = np.zeros(shp)
        u_[self.nghost:] = u
        c_[self.nghost:] = c
        # fill ghost cells
        u_[:self.nghost] = u[0] if neumann else u_in
        c_[:self.nghost] = c[0]
        return u_, c_

    def advect(self, u, c, *args):
        """
        Advect u for one time step.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        args for self.check_u_c
        kwargs for self.apply_boundary_conditions
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        # do we need to allow for negative velocity?
        u_, c_, same_dirn = self.check_u_c(u, c)
        u_, c_ = self.apply_boundary_conditions(u, c, **kwargs)
        f  = self.limited_flux(u_, c_, *args)
        u_ = u_ - self.dt*diffl(f)/self.dx
        u_ = u_[self.nghost:]
        if same_dirn: return u_
        return u_[::-1]
