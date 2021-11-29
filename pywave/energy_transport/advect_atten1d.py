import numpy as np

from pywave.energy_transport.lib import (
        diffl, diffr, sumr)
from pywave.energy_transport.advect1d import Advect1D


class AdvectAtten1D(Advect1D):


    def __init__(self, dx, dt,
            scheme='lax_wendroff', limiter=None,
            flux_correction_scheme=None,
            u_correction_scheme="split_step"):

        super().__init__(dx, dt, scheme=scheme, limiter=limiter)
        if scheme == "lax_wendroff":
            self.flux = {
                    "explicit" : self.flux_lax_wendroff_explicit,
                    "implicit" : self.flux_lax_wendroff_implicit,
                    None : self.flux_lax_wendroff,
                    }[flux_correction_scheme]
        if u_correction_scheme == "split_step":
            # don't correct flux with alpha if using split step
            assert(flux_correction_scheme is None)
        self.adv_atten = {
                    "explicit" : self.adv_atten_explicit,
                    "implicit" : self.adv_atten_implicit,
                    "split_step" : self.adv_atten_split_step,
                    }[u_correction_scheme]

    
    def flux_lax_wendroff_explicit(self, u, c, alpha):
        """
        Flux for Lax-Wendroff - direct space-time, 2nd order.
        Explicit correction for attenuation.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            attenuation coefficient
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        r = self.dt/self.dx
        f = c*u
        return .5*( sumr(f)*(1-alpha*self.dt) - r*c*diffr(f) )


    def flux_lax_wendroff_implicit(self, u, c, alpha):
        """
        Flux for Lax-Wendroff - direct space-time, 2nd order.
        Implicit correction for attenuation.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            attenuation coefficient
        
        Returns:
        --------
        flux : numpy.ndarray
        """
        return self.flux_lax_wendroff(u, c)/(1+alpha*self.dt)


    def adv_atten_split_step(self, u, c, alpha):
        """
        Advect u for one time step.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            attenuation coefficient
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        return self.advect(u, c)*np.exp(-alpha*self.dt)


    def adv_atten_implicit(self, u, c, alpha):
        """
        Advect and attenuate u for one time step.
        Do attenuation explicitly.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            attenuation coefficient
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        return self.advect(u, c, alpha)/(1+alpha*self.dt)


    def adv_atten_explicit(self, u, c, alpha):
        """
        Advect and attenuate u for one time step.
        Do attenuation explicitly.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            attenuation coefficient
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        # do we need to allow for negative velocity?
        u_, c_, same_dirn = self.check_u_c(u, c)
        if (not same_dirn) and isinstance(alpha, numpy.ndarray):
            alpha_ = alpha[::-1]
        else:
            alpha_ = alpha
        f  = self.limited_flux(u_, c_, alpha_)
        u_ = (1-alpha_*self.dt)*u_ - self.dt*diffl(f)/self.dx
        if same_dirn:
            return u_
        return u_[::-1]
