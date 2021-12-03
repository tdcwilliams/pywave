import numpy as np

from pywave.energy_transport.lib import solve_2d_ode_spectral
from pywave.energy_transport.advect1d import Advect1D


class EnergyTransfer2Dirns(Advect1D):
    """ Class to manage energy transfer between 2 directions """


    def __init__(self, dx, dt,
            unit_scat_source=None, aniso=True,
            scheme='lax_wendroff', limiter=None,
            u_correction_scheme="split_step"):
        """
        Parameters:
        -----------
        unit_scat_source : numpy.ndarray
        aniso : bool
        scheme : str
        limiter : str
        u_correction_scheme : str
        """

        super().__init__(dx, dt, scheme=scheme, limiter=limiter)
        if u_correction_scheme == "split_step":
            # don't correct flux with alpha if using split step
            assert(flux_correction_scheme is None)
        self.step = {
                    "explicit" : self.step_explicit,
                    "implicit" : self.step_implicit,
                    "split_step" : self.step_split,
                    }[u_correction_scheme]

        if unit_scat_source is not None:
            self.unit_scat_source = unit_scat_source
        elif aniso:
            self.unit_scat_source = np.array([[-1,-1],[1,1]])
        else:
            self.unit_scat_source = np.array([[-1,1],[1,-1]])


    def get_source_matrix(self, alpha, gamma, shape):
        """
        Construct elements of the source matrix

        Parameters:
        -----------
        alpha : float or numpy.ndarray
            scattering strength
        gamma : float or numpy.ndarray
            dissipative attenuation coefficient
        shape : tuple
            shape of outputs

        Returns:
        --------
        a: numpy.ndarray
            element (0,0) of source matrix
        b: numpy.ndarray
            element (0,1) of source matrix
        c: numpy.ndarray
            element (1,0) of source matrix
        d: numpy.ndarray
            element (1,1) of source matrix
        """
        # make alpha and gamma be arrays of the right shape
        alp = (alpha if isinstance(alpha, np.ndarray)
                else np.full(shape, alpha))
        gam = (gamma if isinstance(gamma, np.ndarray)
                else np.full(shape, gamma))

        # set the source matrix [[a,b],[c,d]]
        a = alp*self.unit_scat_source[0,0] - gam
        b = alp*self.unit_scat_source[0,1]
        c = alp*self.unit_scat_source[1,0]
        d = alp*self.unit_scat_source[1,1] - gam
        return a, b, c, d


    def step_implicit(self, u, v, c, alpha, gamma):
        """
        Advect and attenuate u for one time step.
        Do attenuation explicitly.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected to right
        v : numpy.ndarray
            quantity to be advected to left
        c : float or numpy.ndarray
            velocity
        alpha : float or numpy.ndarray
            scattering strength
        gamma : float or numpy.ndarray
            dissipative attenuation coefficient
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity advected to right
        v_new : numpy.ndarray
            updated quantity advected to left
        """
        u_ = self.advect(u, c)
        v_ = self.advect(v, -c)
        # set the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, u.shape)
        """
        system to solve:
        u^{n+1}, v^{n+1} = ...
            + (a*u^{n+1} + b*v^{n+1}, c*u^{n+1} + d*v^{n+1})
        ie need to invert
            A = [[1-a,-b],[-c,1-d]]
        This has inverse
            Ainv = [[1-d,b],[c,1-a]]/det
        where
            det = (1-d)*(1-a) - b*c
        """
        det = (1-a)*(1-d) - b*c
        return ((1-d)*u_ + b*v_)/det, (c*u_ + (1-a)*v_)/det


    def step_explicit(self, u, v, c, alpha, gamma):
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
        
        Returns:same_dirn
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        # set the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, u.shape)
        return (
                self.advect(u,  c) + a*u + b*v,
                self.advect(v, -c) + c*u + d*v,
                )


    def step_split(self, u, v, ct, alpha, gamma):
        """
        Advect and attenuate u for one time step.
        Do attenuation explicitly.
        
        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected to right
        v : numpy.ndarray
            quantity to be advected to left
        ct : float or numpy.ndarray
            transport velocity
        alpha : float or numpy.ndarray
            scattering strength
        gamma : float or numpy.ndarray
            dissipative attenuation coefficient

        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity to be advected to right
        v_new : numpy.ndarray
            updated quantity to be advected to left
        """
        u_ = self.advect(u, ct)
        v_ = self.advect(v, -ct)

        # set the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, u.shape)
        u_new, v_new = solve_2d_ode_spectral(
                u_, v_, np.array([self.dt]), a, b, c, d)
        return u_new[0], v_new[0]
