import numpy as np
from scipy.special import roots_jacobi
from scipy.special import gamma


class JacobiPolynomials:
    
    """
    Class to handle the Jacobi polynomials, do Gaussian quadrature with them,
    and expand functions in terms of them. It has a bit more functionality than
    scipy.special.jacobi, although it uses that module to find the quadrature points.

    The method get_polys which calculates the recurrence relation to evaluate
    P_n^{alpha,beta} for n=0,1,...,max_degree is an order of magnitude faster
    than looping over max_degree instances of jacobi and then calling them
    """

    def __init__(self, alpha=0, beta=0, max_degree=7, a=-1, b=1):
        """
        Class to handle the Jacobi polynomials.

        Parameters:
        -----------
        alpha : float
            order of singularity at t=1
        beta : float
            order of singularity at t=-1
        max_degree : int
            order of max poly to integrate exactly with Gaussian quadrature
        a : float
            lower limit of interval
        b : float
            upper limit of interval
        """
        assert(alpha>-1)
        assert(beta>-1)
        assert(b>a)    
        self.alpha = alpha
        self.beta = beta
        self.max_degree = max_degree
        self.a = a
        self.b = b

    
    def get_norms(self):
        """
        Returns:
        --------
        h : numpy.ndarray
            norms of the poynomials ie the integrals
            h[n] = \int_{-1}^1{w(x)P_n^2(t)dt}
        """
        hm = np.zeros((self.max_degree + 1,))
        hm[0] = (pow(2, self.alpha + self.beta + 1)
                 * gamma(self.alpha + 1) * gamma(self.beta + 1)
                 / gamma(2 + self.alpha + self.beta ))
        for n in range(1, self.max_degree + 1):
            fac1 = (n + self.alpha)*(n + self.beta)*(2*n + self.alpha + self.beta - 1)
            fac2 = n*(2*n + self.alpha + self.beta + 1)*(n + self.alpha + self.beta)
            hm[n] = fac1*hm[n-1]/fac2
        return hm
    

    def get_polys(self, x):
        '''
        evaluate polynomials at given x
        
        Parameters:
        -----------
        x : numpy.ndarray
            where to evaluate the polynomials

        Returns:
        --------
        polys : numpy.ndarray
            rows correspond to x values, columns to polynomial order
        '''
        m = (self.b - self.a)/2
        t = -1 + (x - self.a)/m
        polys = np.ones((len(t), self.max_degree + 1))
        if self.max_degree == 0:
            return
        polys[:,1] = self.alpha + 1 + (self.alpha + self.beta + 2)*(t-1)/2

        for n in range(2, self.max_degree + 1):
            r1n = (n + self.alpha + self.beta)
            r2n = (r1n + n - 1 ) * (self.alpha ** 2 - self.beta ** 2)
            r3n = (r1n + n - 2) * (r1n + n - 1)*(r1n + n)
            r4n = 2*(n + self.alpha - 1) * (n + self.beta - 1)*(n + r1n)
            r1n = 2 * n * r1n * (r1n + n - 2)
            polys[:,n] = ( (r2n + r3n * t) * polys[:,n-1] - r4n * polys[:,n-2] )/r1n;
        return polys


    def quad_points_weights(self):
        """
        Returns:
        --------
        x : numpy.ndarray
            Gaussian quadrature points
        w : numpy.ndarray
            weights of the Gaussian quadrature scheme
            w.dot(f)[n] = \int_a^b{w(x)P_n^2(t)dx}
        h : numpy.ndarray
            norms of the poynomials ie the integrals
            h[n] = \int_a^b{w(x)P_n^2(t)dx}
        """
        t, w = roots_jacobi(self.max_degree + 1, self.alpha, self.beta)
        m = (self.b - self.a)/2
        fac = pow(m, self.alpha + self.beta + 1)
        hm = self.get_norms()
        return self.a + m * (1+t), fac * w, fac * hm


    def get_inner_product_matrix(self, xwh=None):
        '''
        evaluate polys at quad points
        
        Returns:
        --------
        ipm : numpy.ndarray
            fn = ipm.dot(f), where f is a function evaluated at the
            jacobi quadrature points, and fn is the vector of coefficients in
            the expansion in Jacobi polynomials
        '''
        if xwh is None:
            xwh = self.quad_points_weights()
        x, w, hm = xwh
        pn = self.get_polys(x)
        mat = np.diag(1/hm).dot(pn.T)
        return mat.dot(np.diag(w))
 
        
    def get_coeffs(self, f, xwh=None):
        '''
        Parameters:
        -----------
        f : numpy.ndarray
            function evaluated at the jacobi quadrature points

        Returns:
        --------
        fn : numpy.ndarray
            fn is the vector of coefficients in the expansion in Jacobi polynomials
        '''
        if xwh is None:
            xwh = self.quad_points_weights()
        x, w, hm = xwh
        pn = self.get_polys(x)
        mat = np.diag(1/hm).dot(pn.T)
        return mat.dot(w*f)


    def get_derivatives_factors(self):
        '''
        Returns:
        --------
        jp : JacobiPolynomials
            JacobiPolynomials instance with the alpha,beta and max_degree
            for the derivative of current object

        factors : numpy.ndarray
            factors to multiply coefficients by (see usage in get_derivatives)
        '''
        m = (self.b - self.a)/2
        factors = np.array([(0.5 / m) * (n + self.alpha + self.beta + 1)
               for n in range(1, self.max_degree + 1)])
        jp = JacobiPolynomials(alpha=self.alpha + 1, beta=self.beta + 1,
                               a=self.a, b=self.b, max_degree=self.max_degree - 1)
        return jp, factors


    def get_derivatives(self, x):
        '''
        Evaluate derivatives of polynomials at given x.
        
        Parameters:
        -----------
        x : numpy.ndarray
            where to evaluate the polynomials

        Returns:
        --------
        polys : numpy.ndarray
            rows correspond to x values, columns to polynomial order
        '''
        jp, factors = self.get_derivatives_factors()
        dpn = np.zeros((len(x), self.max_degree +1))
        dpn[:, 1:self.max_degree + 1] = jp.get_polys(x).dot(np.diag(factors))
        return dpn
