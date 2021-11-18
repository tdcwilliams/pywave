import numpy as np
from pywave.open_water_base import OpenWaterBase

_ZI = np.complex(0,1)


class OpenWaterPotentialFlow(OpenWaterBase):

    def __init__(self, num_roots=200, **kwargs):
        super().__init__(**kwargs)
        self.num_roots = num_roots


    @staticmethod
    def increment_nr_small(w, alp):
        c = np.cosh(w)
        s = np.sinh(w)
        f = w*s-alp*c
        df = (1-alp)*s + w*c
        return -f/df


    @staticmethod
    def increment_nr_large(w, alp):
        t = np.tanh(w)
        f = w*t - alp
        df = (1-alp)*t + w
        return -f/df


    @staticmethod
    def increment_nr(w, alp):
        dw = np.zeros_like(w)

        # small real part
        use = np.abs(np.real(w)) < 7.5
        if np.any(use):
            dw[use] = self.increment_nr_small(w[use], alp)

        # large real part
        use = ~use
        if np.any(use):
            dw[use] = self.increment_nr_large(w[use], alp)
        return dw


    @staticmethod
    def increment_nr_imag(w, alp):
        c = np.cos(w)
        s = np.sin(w)
        f  = w*s+alp*c
        df = (1-alp)*s + w*c
        return -f/df


    @staticmethod
    def find_root_nr(inc_nr_fun, guess, params=None, tol=1e-9, maxits=20):

        if params is None: params = {}
        for it in range(maxits):
            dw = inc_nr_fun(w, **params)
            w += dw
            converged = np.abs(dw) < tol*np.abs(w)
            if np.all(converged):
                break
        w[~converged] = np.nan
        return w


    def find_real_root(self, guess=None, **kwargs):
        """
        solve
        k*tanh(kh) = omega^2/g
        w*sinh(w) - (h*omega^2/g)*cosh(w) = 0
        """
        alp = self.wave_number_ow_id*self.depth
        if guess is None:
            w = alp
        else:
            w = self.depth*guess
        w = np.array([w])

        w, = self.find_root_nr(
                self.increment_nr, w, params={'alp' : alp}, **kwargs)
        assert(np.isfinite(w)) #TODO implement catch here
        return np.abs(w)/self.depth


    @staticmethod
    def check_imag_roots(w):
        rh = np.array(
                    [n*np.pi for n in range(len(w))])
        lh = rh - np.pi/2
        w[w > rh] = np.nan
        w[w < lh] = np.nan
        return w


    def find_imag_roots(self, guess=None, **kwargs):
        """
        solve
        k*tanh(kh) = omega^2/g
        w*sin(w) + (h*omega^2/g)*cos(w) = 0
        """

        if guess is None:
            w = np.array(
                    [n*np.pi for n in range(self.num_roots)])
        else:
            w = (-_ZI*self.depth*guess).real

        alp = self.wave_number_ow_id*self.depth
        w = self.find_root_nr(
                self.increment_nr_imag, w, params={'alp' : alp}, **kwargs)
        w = self.check_imag_roots(w)
        assert(np.all(np.isfinite(w))) #TODO implement catch here
        return _ZI*w/self.depth


    def solve_disprel(self, guess=None):

        gr, gi = None, None
        if guess is not None:
            gr = guess[0]
            gi = guess[1:]

        self.k = np.zeros((self.num_roots+1,), dtype=np.complex)
        self.k[0] = self.find_real_root(guess=gr)
        self.k[1:] = self.find_imag_roots(guess=gi)


    def get_norms(self):
        """
        Let
        
        norms_
         = \int_{-h}^0 cosh^2(k*(z+h))dz
         = h/2 + sinh(k*h)cosh(k*h)
         = h/2 + tanh(k*h)*cosh^2(k*h)
         = h/2 + (alp/k)*cosh^2(k*h)

        If t=tanh(k*h) = alp/k, then

        norms
         = norms_/cosh^2(k*h)
         = h/2*(1-t^2) + t
        """
        alp = self.wave_number_ow_id*self.depth
        t = alp/self.k
        return self.depth*(1-t**2)/2 + t
