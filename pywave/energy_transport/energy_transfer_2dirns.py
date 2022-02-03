import numpy as np

from pywave.energy_transport.lib import solve_2d_ode_spectral
from pywave.energy_transport.advect1d import Advect1D


class EnergyTransfer2Dirns(Advect1D):
    """ Class to manage energy transfer between 2 directions """

    def __init__(self, dx, dt,
            unit_scat_source=None, aniso=True,
            u_correction_scheme="split_step", **kwargs):
        """
        Parameters:
        -----------
        dx : float
        dt : float
        unit_scat_source : numpy.ndarray
        aniso : bool
        u_correction_scheme : str
        kwargs for Advect1D
        """
        super().__init__(dx, dt, **kwargs)
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

    def get_source_matrix(self, alpha, gamma, shape=None):
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

    def step_implicit(self, u, v, ct, alpha, gamma, neumann=True, u_in=0):
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
            velocity
        alpha : float or numpy.ndarray
            scattering strength
        gamma : float or numpy.ndarray
            dissipative attenuation coefficient
        neumann : bool
            True : set u on the ghost cells to u[0]
                to give u_x=0 (Neumann) boundary conditions
            False : set u on the ghost cells to u_in
        u_in : float
            value of u coming from the left boundary
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity advected to right
        v_new : numpy.ndarray
            updated quantity advected to left
        """
        u_ = self.advect(u,  ct, neumann=neumann, u_in=u_in)
        v_ = self.advect(v, -ct, neumann=neumann, u_in=0)
        # get the source matrix [[a,b],[c,d]] * dt
        a_, b_, c_, d_ = [self.dt * arr
                for arr in self.get_source_matrix(alpha, gamma, shape=u.shape)]
        """
        system to solve:
        u^{n+1}, v^{n+1} = ...
            + (a*u^{n+1} + b*v^{n+1}, c*u^{n+1} + d*v^{n+1})*dt
        ie need to invert
            A = [[1-a_,-b_],[-c_,1-d_]]
            with
            [a_, b_, c_, d_] = [a, b, c, d]*dt
        This has inverse
            Ainv = [[1-d_,b_],[c_,1-a_]]/det
        where
            det = (1-d_)*(1-a_) - b_*c_
        """
        det = (1-a_)*(1-d_) - b_*c_
        return ((1-d_)*u_ + b_*v_)/det, (c_*u_ + (1-a_)*v_)/det

    def step_explicit(self, u, v, c, alpha, gamma, neumann=True, u_in=0):
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
            scattering attenuation coefficient
        gamma : float or numpy.ndarray
            dissipative attenuation coefficient
        neumann : bool
            True : set u on the ghost cells to u[0]
                to give u_x=0 (Neumann) boundary conditions
            False : set u on the ghost cells to u_in
        u_in : float
            value of u coming from the left boundary
        
        Returns:same_dirn
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        # get the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, shape=u.shape)
        return (
                self.advect(u,  c, neumann=neumann, u_in=u_in) + self.dt*(a*u + b*v),
                self.advect(v, -c, neumann=neumann, u_in=0) + self.dt*(c*u + d*v),
                )

    def step_split(self, u, v, ct, alpha, gamma, neumann=True, u_in=0):
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
        neumann : bool
            True : set u on the ghost cells to u[0]
                to give u_x=0 (Neumann) boundary conditions
            False : set u on the ghost cells to u_in
        u_in : float
            value of u coming from the left boundary

        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity to be advected to right
        v_new : numpy.ndarray
            updated quantity to be advected to left
        """
        u_ = self.advect(u,  ct, neumann=neumann, u_in=u_in)
        v_ = self.advect(v, -ct, neumann=neumann, u_in=0)

        # set the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, shape=u.shape)
        u_new, v_new = solve_2d_ode_spectral(
                u_, v_, np.array([self.dt]), a, b, c, d)
        return u_new[0], v_new[0]
