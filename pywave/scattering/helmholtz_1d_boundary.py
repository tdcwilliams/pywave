import numpy as np

from pywave.scattering.medium_boundary import MediumBoundary

_ZI = np.complex(0, 1)


class Helmholtz1DBoundary(MediumBoundary):
    """
    Class for scatterer that is a sharp boundary between 2 media
    that satisfy the 1D Helmholtz equation. Due to their simplicity
    some of the methods of MediumBoundary are duplicated here
    (solve, test_boundary_conditions).
    This is useful for testing purposes.
    """
    
    def solve(self):
        """
        Solve the boundary conditions to set the scattering matrices
        
        Sets:
        -----
        Rp_ : numpy.ndarray(float)
            R^+ matrix (reflection of waves to right)
        Rm_ : numpy.ndarray(float)
            R^- matrix (reflection of waves to left)
        Tp_ : numpy.ndarray(float)
            T^+ matrix (transmission of waves to right)
        Tm_ : numpy.ndarray(float)
            T^- matrix (transmission of waves to left)
        """
        kk0 = self.media[0].k[0]*self.media[0].helmholtz_coef
        kk1 = self.media[1].k[0]*self.media[1].helmholtz_coef
        fac = 1/(kk0 + kk1)
        self.Rp = fac*np.array([[kk0-kk1]])
        self.Tm = fac*np.array([[2*kk1]])
        self.Tp = fac*np.array([[2*kk0]])
        self.Rm = fac*np.array([[kk1-kk0]])

    @property
    def intrinsic_admittance(self):
        """
        Returns:
        --------
        intrinsic_admittance : float
            coefficient \alpha in the conservation of energy equation
        """
        cc = [self.media[i].k[0]*self.media[i].helmholtz_coef for i in [0,-1]]
        return cc[-1]/cc[0]

    def test_boundary_conditions(self, inc_amps=None):
        """
        Test boundary conditions are satisfied
        """
        if inc_amps is None:
             inc_amps = np.array([1]), np.array([0.5])
        ip, im = inc_amps
        sp = self.get_solution_params(0, ip, im)['a1']
        sm = self.get_solution_params(-1, ip, im)['a0']
        u_m = ip.sum() + sp.sum() #U(0^-)
        u_p = im.sum() + sm.sum() #U(0^+)
        sig_m = _ZI*self.media[0].helmholtz_coef*self.media[0].k.dot(ip - sp)
        sig_p = _ZI*self.media[-1].helmholtz_coef*self.media[-1].k.dot(sm - im)
        print(f"u(0) = {u_m} = {u_p}")
        print(f"\sigma(0) = {sig_m} = {sig_p}")
        assert(np.abs(u_m - u_p) < 1e-8)
        assert(np.abs(sig_m - sig_p) < 1e-8)
        print("Boundary conditions are OK")
