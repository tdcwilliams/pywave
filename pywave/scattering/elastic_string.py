import numpy as np
from pywave.scattering.helmholtz_1d import Helmholtz1D

_ZI = np.complex(0, 1)


class ElasticString(Helmholtz1D):
    """ Properties of an elastic string """

    def __init__(self, m=1, kappa=4, period=3, xlim=None):
        """
        Class to manage string properties
        kappa*u_{xx} - m*u_tt = 0
        If periodic this becomes
        0 = kappa*(u_{xx} + (m*omega^2/kappa)*u)
          = kappa*(u_xx + k^2*u)
        ie k = sqrt(m/kappa)*omega
        
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
        self.kappa = kappa
        self.m = m
        self.period = period
        super().__init__(helmholtz_coef=kappa,
                k=np.sqrt(m/kappa)*self.omega, xlim=xlim)

    def set_operators(self):
        """
        Set some operators for convenience

        Sets:
        -----
        self.operators : dict
        """
        super().set_operators()
        self.operators['displacement'] = lambda k : 1
        self.operators['stress'] = self.operators['helmholtz_cux']
