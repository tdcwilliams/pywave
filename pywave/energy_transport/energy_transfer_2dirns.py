import numpy as np

from pywave.energy_transport.lib import (
        diffl, diffr, sumr)
from pywave.energy_transport.advect1d import Advect1D


class EnergyTransfer2Dirns(Advect1D):


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

        self.unit_source_matrix = np.array([[-1,-1],[1,1]])

    
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
            attenuation coefficient
        
        Returns:
        --------
        u_new : numpy.ndarray
            updated quantity
        """
        u_ = self.advect(u, c)
        v_ = self.advect(v, -c)
        (a,b), (c,d) = (
                (1+gamma*self.dt)*np.eye(2)
                + (alpha*self.dt)*self.unit_source_matrix
                )
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
        (a,b), (c,d) = (alpha*self.dt)*self.unit_source_matrix
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

        # make alpha and gamma be arrays of the right shape
        gam = (gamma if isinstance(gamma, numpy.ndarray)
                else np.full_like(u, gamma))
        alp = (alpha if isinstance(alpha, numpy.ndarray)
                else np.full_like(u, alpha))

        # set the source matrix [[a,b],[c,d]]
        a = (-gam*self.dt
                + (alp*self.dt)*self.unit_source_matrix[0,0])
        b = (alp*self.dt)*self.unit_source_matrix[0,1]
        c = (-gam*self.dt
                + (alp*self.dt)*self.unit_source_matrix[1,1])
        d = (alp*self.dt)*self.unit_source_matrix[1,1]
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
            dlam = np.sqrt(discr[use])/2
            lams = [lam_av[use] + dlam, lam_av[use]-dlam]
            ab = (a[use], b[use])
            u_new[use], v_new[use] = self.evolve_unique_evals(
                    u_[use], v_[use], lams, ab)

        return u_new, v_new
