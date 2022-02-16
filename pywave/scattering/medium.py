import numpy as np

_ZI = np.complex(0, 1)


class Medium:
    """
    Properties of a general medium
    - usually a wave problem solved in the frequency domain

    Handles things like x range, eigenfunction expansion,
    edge conditions.
    """

    def __init__(self, xlim=None):
        """
        Class for a general wave medium
        
        Parameters:
        -----------
        xlim : array-like
            (x0, x1), where x0 is the LHS limit and x1 si the RHS limit
            default is (-numpy.Inf, numpy.Inf)
        """
        self.set_limits(xlim)
        self.solve_disprel()
        self.set_operators()
        self.set_edge_operators()

    def set_limits(self, xlim):
        """
        Set limits of domain and characterise as semi-infinite/infinite/finite

        Sets:
        -----
        infinite : bool
        semi_infinite : bool
        xlim : numpy.ndarray(float)
        """
        self.infinite = False
        self.semi_infinite = False
        if xlim is None or tuple(xlim) == (-np.Inf, np.Inf):
            self.infinite = True
            self.xlim = np.array([-np.Inf, np.Inf])
        else:
            self.xlim = np.array(xlim)
            self.semi_infinite = not np.all(np.isfinite(xlim))
        assert(self.xlim[0] < self.xlim[1])
        assert(len(self.xlim) == 2)
    
    def solve_disprel(self):
        """
        solve dispersion relation
        Needs to be defined in child classes

        Sets:
        --------
        self.k : numpy.ndarray(float)
            vector of wave numbers (1/m)
            satisfies
                alpha - kappa*k^2 = 0
        """
        raise NotImplementedError(
                "solve_disprel should be implemented in child class")

    def set_operators(self):
        """
        set common operators for convenience or to be used by Medium.set_edge_operators

        Sets:
        -----
        self.operators
        """
        self.operators = {}

    def set_edge_operators(self):
        """
        set edge operators to be used by Medium.solve

        Sets:
        -----
        self.edge_operators
        """
        self.edge_operators = {}

    @property
    def omega(self):
        """
        Return omega (only works if self.period is defined)

        Returns:
        --------
        omega : float
            radial wave frequency (1/s)
            = 2\pi/period
        """
        return 2*np.pi/self.period # 1/s


    @property
    def num_modes(self):
        """
        Returns:
        --------
        num_modes : int
            number of wave modes
        """
        return len(self.k)

    @property
    def phase_matrix(self):
        """
        Returns:
        --------
        phases : numpy.ndarray
            vector with n-th element e^{i self.k[n] w} where w is the width and self.k
            is the vector of wave numbers
        """
        width = self.xlim[1] - self.xlim[0]
        if np.isfinite(width):
            return np.diag(np.exp(_ZI*width*self.k))

    def get_new(self, xlim=None):
        """ get new class instance with same parameters but different spatial limits
        
        Parameters:
        -----------
        xlim : tuple(float)
            (x0, x1)
            
        Returns:
        --------
        new : new instance of same type as self
            copy of self but with different xlim
        """
        new = self.__new__(self.__class__)
        for k,v in self.__dict__.items():
            if k != 'xlim':
                setattr(new, k, v)
        new.set_limits(xlim)
        return new

    def is_in_domain(self, x):
        """
        Parameters:
        -----------
        x : numpy.array(float)
        
        Returns:
        --------
        in_dom : numpy.array(bool)
            bool same size as x marking which members are in the domain
        """
        x0, x1 = self.xlim
        return (x>=x0) * (x<=x1)

    def get_expansion(self, x, a0, a1, operator=None):
        """
        Calculate a displacement profile

        Parameters:
        -----------
        x : numpy.array(float)
            positions to evaluate the displacement
        a0 : numpy.array(float)
            wave amplitudes for waves travelling from left to right
        a1 : numpy.array(float)
            wave amplitudes for waves travelling from right to left
        operator : str or function
            operator to apply as a function of wave number k
            if a string is used, operator is taken from the dict self.operators
            defined in self.set_operators of child class

        Returns:
        --------
        u : numpy.array(float)
            complex displacement U evaluated at x
        """
        u = np.full_like(x, np.nan, dtype=np.complex)
        b = self.is_in_domain(x)
        xb = x[b]
        x0, x1 = self.xlim
        if self.infinite:
            x0 = x1 = 0
        elif self.semi_infinite:
            x0 = x1 = self.xlim[np.isfinite(self.xlim)]
        if operator is None:
            operator = lambda x : 1
        elif isinstance(operator, str):
            operator = self.operators[operator]
        c0 = a0 * operator(self.k)
        c1 = a1 * operator(-self.k)
        # (nx,nk) x (nk,1) = (nx,1)
        u[b] = np.exp(_ZI*np.outer(xb - x0, self.k)).dot(c0).flatten()
        u[b] += np.exp(_ZI*np.outer(x1 - xb, self.k)).dot(c1).flatten()
        return u

    def get_matrices_forcings_1op(self, op, on_left):

        """
        1st half of matrix columns: "e^{ikx}" eigen functions
        2nd half of matrix columns: "e^{-ikx}" eigen functions
        """
        row_p = op( self.k).reshape(1,-1)
        row_m = op(-self.k).reshape(1,-1)
        med_factor = {True: -1, False: 1}[on_left]
        matrices = med_factor * np.hstack([row_p, row_m])
        if on_left:
            forcings = med_factor * row_p # e^{ikx} forcing
        else:
            forcings = med_factor * row_m # e^{-ikx} forcing
        return matrices, forcings

    def get_matrices_forcings_1pair(self, name, on_left, is_continuous):
        matrices = []
        forcings = []
        op1, op2 = self.edge_operators[name]
        ops = (op1, op2) if is_continuous else (op1)
        for op in ops:
            m, f = med.get_matrices_forcings_1op(op, on_left)
            matrices += [m]
            forcings += [f]
        return np.vstack(mats), np.vstack(forcings)
