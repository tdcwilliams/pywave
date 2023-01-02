import numpy as np
#from functools import cached_property
from functools import lru_cache #TODO use cached_property when python >= 3.8 installed


class StationaryBase:
    """ Solve the general problem (non-zero dissipation) """
    def __init__(self, eps=0., beta=0, w=.1):
        """
        Parameters:
        -----------
        w : float
            MIZ width (>0)
        beta : float
            Directional dependence parameter (-1<=beta<=1)
        eps : float
            Dissipation (>=0)
            
        Sets:
        --------
        self.w
            MIZ width (>0)
        self.beta
            Directional dependence parameter (-1<=beta<=1)
        self.eps
            Dissipation (>=0)
        """
        self.eps = eps
        self.beta = beta
        self.w = w

    #@cached_property
    @property
    @lru_cache()
    def matrix(self):
        """
        Returns:
        --------
        matrix : numpy.ndarray
            Matrix A in equation E_x = A E
        """
        raise NotImplementedError(
                "matrix property needs to be implemented in child class")

    #@cached_property
    @property
    @lru_cache()
    def eig_tuples(self):
        """
        Returns:
        --------
        eig_tuples : list
            Each iteration is a tuple (lam, vec, xe):
                lam : float
                    Eigenvalue
                vec : numpy.ndarray
                    Eigenvector
                xe : float
                    Source of waves (should decay away from source for 0<=x<=w).
                    If lam<=0, xe=0; if lam>0, xe=w.
        """
        tups = []
        evals, evecs = np.linalg.eig(self.matrix)
        for i in range(2):
            lam = evals[i]
            vec = evecs[:,i]
            xe = 0. if lam <= 0. else self.w
            tups += [(lam, vec, xe)]
        return tups

    #@cached_property
    @property
    @lru_cache()
    def coeffs(self):
        """
        Returns:
        --------
        coeffs: numpy.ndarray
            Coefficients of eigenfunction expansion. Shape is (2,)
        """
        rhs = np.array([1.,0.])
        matrix = np.zeros((2,2))
        x = np.array([0., self.w])
        for i, (lam, vec, xe) in enumerate(self.eig_tuples):
            matrix[:,i] = vec * np.exp(lam * (x - xe))
        return np.linalg.solve(matrix, rhs)

    def calc_expansion(self, x):
        """
        Parameters:
        -----------
        x : numpy.ndarray
            x values where energy should be calculated.
            Shape is (nx,)
        
        Returns:
        --------
        E : numpy.ndarray
            Energy calculated at x.
            Shape is (2,nx)
            - 1st row is E_+ (energy going to the right)
            - 2nd row is E_- (energy going to the left)
        """
        u = 0.
        for (lam, vec, xe), coeff in zip(self.eig_tuples, self.coeffs):
            u += coeff * np.outer(vec, np.exp(lam * (x - xe)))
        return u
    
    def calc_wave_stress(self, x):
        """
        Parameters:
        -----------
        x : numpy.ndarray
            x values where energy should be calculated.
            Shape is (nx,)
        
        Returns:
        --------
        tau_x : numpy.ndarray
            Wave stress on ice.
            Shape is (nx,)
        """
        raise NotImplementedError(
                "calc_wave_stress method needs to be implemented in child class")
