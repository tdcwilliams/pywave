import numpy as np

from pywave.energy_transport.lib import (
        diffl, diffr, sumr)
from pywave.energy_transport.advect1d import Advect1D


class EnergyTransfer2Dirns(Advect1D):
    """ Class to manage energy transfer between 2 directions """


    def __init__(self, dx, dt,
            unit_scat_source=None, aniso=True,
            scheme='lax_wendroff', limiter=None,
            flux_correction_scheme=None,
            u_correction_scheme="split_step"):
        """
        Parameters:
        -----------
        unit_scat_source : numpy.ndarray
        aniso : bool
        scheme : str
        limiter : str
        flux_correction_scheme : str
        u_correction_scheme : str
        """

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
        alp = (alpha if isinstance(alpha, numpy.ndarray)
                else np.full(shape, alpha))
        gam = (gamma if isinstance(gamma, numpy.ndarray)
                else np.full(shape, gamma))

        # set the source matrix [[a,b],[c,d]]
        a = (-gam*self.dt
                + (alp*self.dt)*self.unit_scat_source[0,0])
        b = (alp*self.dt)*self.unit_scat_source[0,1]
        c = (-gam*self.dt
                + (alp*self.dt)*self.unit_scat_source[1,1])
        d = (alp*self.dt)*self.unit_scat_source[1,1]
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
        # convert to the matrix to invert
        # u^{n+1}, v^{n+1} = ...
        #     + (a*u^{n+1} + b*v^{n+1}, c*u^{n+1} + d*v^{n+1})
        a, b, c, d = 1-a, -b, -c, 1-d
        det = a*d - b*c
        return (d*u_ - b*v_)/det, (-c*u_ + a*v_)/det


    def step_explicit(self, u, c, alpha, gamma):
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


    def evolve_unique_evals(self, u, v, lams, ab):
        """
        Evolve ODE
        [du/dt, dv/dt] = [[a,b],[c,d]](u,v) 
        from time 0 to self.dt
        when the eigenvalues of the matrix are not repeated.
        Note that in this case we don't need to know the 2nd
        row of the matrix (ie just need (a,b)).

        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected to right
        v : numpy.ndarray
            quantity to be advected to left
        lams : list(numpy.ndarray)
            list of eigenvalues [lam1,lam2]
        ab : tuple
            (a,b) with
            a: numpy.ndarray
                element (0,0) of source matrix
            b: numpy.ndarray
                element (0,1) of source matrix

        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity to be advected to right
        v_new : numpy.ndarray
            updated quantity to be advected to left
        """
        a, b = ab
        u_new = np.zeros_like(u)
        v_new = np.zeros_like(u)
        for lam in lams:
            h = np.hypot(-b, a-lam)
            ev = np.array([[-b/h], [a-lam]/h]) #2 x nx
            coef = u*ev[0] + v*ev[1]
            ex = np.exp(lam*self.dt)
            u_new += ex * coef * ev[0]
            v_new += ex * coef * ev[1]
        return u_new, v_new


    def evolve_repeated_eval(self, u, v, lam, abcd):
        """
        Evolve ODE
        [du/dt, dv/dt] = [[a,b],[c,d]](u,v) 
        from time 0 to self.dt
        when the eigenvalues of the matrix are repeated

        Parameters:
        -----------
        u : numpy.ndarray
            quantity to be advected to right
        v : numpy.ndarray
            quantity to be advected to left
        lam : numpy.ndarray
            eigenvalue
        abcd : tuple
            (a,b,c,d) with
            a: numpy.ndarray
                element (0,0) of source matrix
            b: numpy.ndarray
                element (0,1) of source matrix
            c: numpy.ndarray
                element (1,0) of source matrix
            d: numpy.ndarray
                element (1,1) of source matrix

        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity to be advected to right
        v_new : numpy.ndarray
            updated quantity to be advected to left
        """

        """
        eigenvector satisfies
        (a-lam)*x+b*y=0
        ie orthogonal to row=[a-lam,b]/h
        (row is normalised to unit vector)
        """
        a, b, c, d = abcd
        rows = {} # key True: use top row of matrix to find aux
                  # key False: use bottom row
        for i, (el0, el1) in enumerate([
                (a-lam, b), #2 x nx
                (c, d-lam), #2 x nx
                ]):
            h = hypot(el0, el1)
            rows[i==0] = [el0/h, el1/h, h]

        ev0 = -el1/h
        ev1 = el0/h
        rows[True] += [ev0]
        rows[False] += [ev1]
        ex = np.exp(lam*self.dt)
        tmp = ex * (u*ev0 + v*ev1)
        u_new = tmp * ev0
        v_new = tmp * ev1

        """
        uv2 = exp(lam*t)*(ev*t + aux)
        WHERE IF ev0 != 0:
        auxiliary vector aux=(x,y) satisfies
        (a-lam)*x+b*y=ev0
        or
        h*row.dot(aux) = ev0
        with row=(a-lam,b)/h a unit vector.

        IF ev0 ==0, THEN ev1 != 0:
        auxiliary vector aux=(x,y) satisfies
        c*x+(d-lam)*y=ev1
        or
        h*row.dot(aux) = ev1
        with row=(c,d-lam)/h a unit vector.

        IN GENERAL:
        h*row.dot(aux) = ev_
        with row=(r0,r1) a unit vector.

        eg aux = (ev_/h)*row +beta*(ev0,ev1)
        Choose beta=0 so aux is orthogonal to (ev0,ev1).
        Then
        h*row.dot(aux) = h*row.dot((ev_/h)*row) = ev_
        => aux = (ev_/h)*row
        => uv2 = (ev_/h)*exp(lam*t)*((h/ev_)*ev*t + row)
        or just
        uv2 = exp(lam*t)*((h/ev_)*ev*t + row) so its coefficient
        is just (u,v).dot(row) since row is a unit vector
        """
        def get_update(b, use):
            r0, r1, h, ev_ = rows[b]
            # h*(r0,r1).dot(aux) = ev_
            tmp = ex[use] * (u[use]*r0[use] + v[use]*r1[use])
            tmp2 = (h*self.dt/ev_)[use]
            du = tmp*(tmp2*ev0[use] + r0[use])
            dv = tmp*(tmp2*ev1[use] + r1[use])
            return du, dv
        for i, use in enumerate((ev0 != 0, ev==0)):
            if np.any(use):
                du, dv = get_update(i==0, use)
                u_new[use] += du
                v_new[use] += dv
        return u_new, v_new


    def step_split(self, u, v, c, alpha, gamma):
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
            updated quantity to be advected to right
        v_new : numpy.ndarray
            updated quantity to be advected to left
        """
        u_ = self.advect(u, c)
        v_ = self.advect(v, -c)

        # set the source matrix [[a,b],[c,d]]
        a, b, c, d = self.get_source_matrix(alpha, gamma, u.shape)
        discr = (a+d)**2 + a*d-b*c
        assert(np.all(discr >= 0))
        lam_av = (a+d)/2 # average of the 2 eigenvalues
        u_new = np.copy(u_)
        v_new = np.copy(v_)

        # 1st do repeated eigenvalues (both = lam_av)
        use = (discr == 0)
        if np.any(use):
            abcd = (a[use], b[use], c[use], d[use])
            u_new[use], v_new[use] = self.evolve_repeated_eval(
                    u_[use], v_[use], lam_av[use], abcd)

        # 2nd do unique eigenvalues
        use = (discr > 0)
        if np.any(use):
            ab = (a[use], b[use])
            dlam = np.sqrt(discr[use])/2
            lams = [lam_av[use] + dlam, lam_av[use]-dlam]
            u_new[use], v_new[use] = self.evolve_unique_evals(
                    u_[use], v_[use], lams, ab)

        return u_new, v_new
