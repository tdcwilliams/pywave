import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

_ZI = np.complex(0, 1)


class ScattererBase:

    def __init__(self, media):
        """
        Base scatterer class.
        Default calculates the scattering by a change in elastic string properties
        
        Parameters:
        -----------
        lhs : Medium or subclass
            object containing properties of the medium to the left of the boundary
        rhs : Medium or subclass
            object containing properties of the medium to the right of the boundary
        position : float
            x value where the boundary occurs
        """
        self.media = media
        om = [med.omega for med in self.media]
        assert(len(set(om)) == 1)

        
    @property
    def params(self):
        """
        Returns:
        --------
        params : dict
            parameters shared by media are scalars eg period;
            parameters that vary between media are lists
        """
        p = self.media[0].params
        const = ['period']
        variable = [k for k in p if k not in const]
        params = {k:v for k,v in p.items() if k in const}
        for k in variable:
            params[k] = [med.params[k] for med in self.media]
        return params
    

    @property
    def period(self):
        """
        Returns:
        --------
        period : float
            wave period (s)
        """
        return self.media[0].period
    

    @property
    def omega(self):
        """
        Returns:
        --------
        omega : float
            radial wave frequency (1/s)
            = 2\pi/period
        """
        return self.media[0].omega


    @property
    def intrinsic_admittance(self):
        """
        Returns:
        --------
        intrinsic_admittance : float
            coefficient \alpha in the conservation of energy equation
        """
        cc = [self.media[i].k[0]*self.media[i].kappa for i in [0,-1]]
        return cc[-1]/cc[0]


    @property
    def xlim(self):
        """
        Returns:
        --------
        xlim : tuple
            left and right hand limits of scatterer
            (both are self.position for a sharp change in properties)
        """
        return self.media[0].xlim[1], self.media[-1].xlim[0]


    def get_solution_params(self, index, ip, im):
        """
        Get amplitudes of scattered waves

        Parameters:
        -----------
        index : int
            index of the medium you want the wave amplitudes for
        ip : numpy.array(float)
            wave amplitudes of incident waves to right
        im : numpy.array(float)
            wave amplitudes of incident waves to left

        Returns:
        --------
        sp : numpy.array(float)
            wave amplitudes for waves scattered to left
        sm : numpy.array(float)
            wave amplitudes for waves scattered to right
        """
        nmed = len(self.media)
        i = index % nmed
        if i == 0:
            sp = self.Rp.dot(ip) + self.Tm.dot(im)
            return dict(a0=ip, a1=sp.flatten()) # x<0
        if i == nmed - 1:
            sm = self.Tp.dot(ip) + self.Rm.dot(im)
            return dict(a0=sm.flatten(), a1=im) # x>0
        return dict(
            a0 = self.ap[self.slices[i-1],:].dot(ip)
               + self.am[self.slices[i-1],:].dot(im),
            a1 = self.bp[self.slices[i-1],:].dot(ip)
               + self.bm[self.slices[i-1],:].dot(im),
            )


    def test_power_input(self):
        """
        Test the net power input is zero. Checks both
        when the wave is from the right or the left.
        """
        def simple_inputs(from_left):
            nmed = len(self.media)
            ip = np.zeros_like(self.media[0].k)
            im = np.zeros_like(self.media[1].k)
            if from_left:
                ip[0] = 1
            else:
                im[0] = 1
            return ip, im


        def run_power_test(from_left):
            # test power input to left and right hand strings is the same
            ip, im = simple_inputs(from_left)
            pl, pr = [self.media[i].get_power(
                    **self.get_solution_params(i, ip, im))
                    for i in range(2)]
            print(f'Test power input, from_left={from_left}:')
            print(f'\t{pl}, {pr}')
            assert(np.allclose([pl], [pr]))
            print('\tOK' )

        for from_left in [True, False]:
            run_power_test(from_left)


    def test_energy(self):
        """
        Test energy is conserved
        """
        alp = self.intrinsic_admittance
        e = np.abs(self.Rp[0,0])**2 + alp*np.abs(self.Tp[0,0])**2
        assert(np.abs(e-1) < 1e-8)
        e = np.abs(self.Rm[0,0])**2 + np.abs(self.Tm[0,0])**2/alp
        assert(np.abs(e-1) < 1e-8)
        print('Energy is OK')    


    def test_boundary_conditions(self, inc_amps=None):
        """
        Test boundary conditions are satisfied
        """
        if inc_amps is None:
             inc_amps = np.array([1]), np.array([0.5])
        ip, im = inc_amps
        sp = self.get_solution_params(0, ip, im)['a1']
        sm = self.get_solution_params(-1, ip, im)['a0']
        u_m = ip.sum() + sp.sum() #U(0^+)
        u_p = im.sum() + sm.sum() #U(0^-)
        sig_m = _ZI*self.media[0].kappa*self.media[0].k.dot(ip - sp)
        sig_p = _ZI*self.media[-1].kappa*self.media[-1].k.dot(sm - im)
        print(f"u(0) = {u_m} = {u_p}")
        print(f"\sigma(0) = {sig_m} = {sig_p}")
        assert(np.abs(u_m - u_p) < 1e-8)
        assert(np.abs(sig_m - sig_p) < 1e-8)
        print("Boundary conditions are OK")


    def get_expansion(self, x, inc_amps=None, get_disp=True):
        """
        Get displacement profile outside the scatterer

        Parameters:
        -----------
        x : numpy.array(float)
            positions to evaluate the displacement
        inc_amps : list or tuple
            (ip,im) with
                ip : numpy.array(float)
                    wave amplitudes of incident waves to right
                im : numpy.array(float)
                    wave amplitudes of incident waves to left

        Returns:
        --------
        u : numpy.array(float)
            complex displacement U evaluated at x
        label : numpy.array(int)
            label of which media each point of x is in
        """
        if inc_amps is None:
            inc_amps = (np.array([1]), np.array([0]))
        ip, im = inc_amps
        u = np.zeros_like(x, dtype=np.complex)
        x0, x1 = self.xlim
        lbl = np.zeros_like(x, dtype=int)
        for i, med in enumerate(self.media):
            b = med.is_in_domain(x)
            lbl[b] = i
            if not np.any(b):
                continue
            kw = self.get_solution_params(i, ip, im)
            u[b] = med.get_expansion(x[b], **kw, get_disp=True)
        return u, lbl
       

    @property
    def default_plot_range(self):
        fac = 2.5
        lam = 2*np.pi/np.max([np.max(self.media[i].k.real) for i in [0,-1]])
        if self.media[0].infinite:
            return - fac*lam, fac*lam, lam
        xlim = self.media[0].xlim
        x0 = xlim[0]
        if self.media[0].semi_infinite:
            x0 = xlim[1] - fac*lam
        xlim = self.media[-1].xlim
        x1 = xlim[1]
        if self.media[-1].semi_infinite:
            x1 = xlim[0] + fac*lam
        return x0, x1, lam


    def plot_expansion(self, ax=None, t=0, x=None, inc_amps=None,
                      get_disp=True, no_title=False, **kwargs):
        """
        Plot displacement profile

        Parameters:
        -----------
        t : float
            time (s)
        x : numpy.array(float)
            positions (m) to evaluate the displacement
        inc_amps : list or tuple
            (ip,im) with
                ip : numpy.array(float)
                    wave amplitudes of incident waves to right
                im : numpy.array(float)
                    wave amplitudes of incident waves to left
        kwargs for matplotlib.pyplot.plot

        Returns:
        --------
        ax : matplotlib.axes._subplots.AxesSubplot
            plot axis
        """
        if x is None:
            # get the smallest wavelength
            x0, x1, lam = self.default_plot_range
            x = list(np.linspace(x0, x1, 100*int((x1-x0)/lam)))
            # add edges
            x += [med.xlim[1] for med in self.media if np.isfinite(med.xlim[1])]
            x = np.array(sorted(list(set(x))))
            
        # calc displacement
        u, lbl = self.get_expansion(
            x, inc_amps=inc_amps, get_disp=get_disp)
        u *= np.exp(-_ZI*self.omega*t)
        umax = 1.1*np.abs(u).max()
        # plot different media separately to give each a different colour
        if ax is None:
            fig = plt.figure(figsize=(14,7))
            ax = fig.add_subplot(111)
        # reset color cycle
        ax.set_prop_cycle(None)
        # make the plots
        for i in range(len(self.media)):
            b = (lbl == i)
            ax.plot(x[b], u[b].real, **kwargs)
        
        # decorate
        if not no_title:
            ax.set_title(f"u at t={t}s")
        ax.set_xlabel("x, m")
        ax.set_ylabel("u, m")
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-umax, umax])
        return ax
    

    def animate_displacement(self, figname='animation.gif', **kwargs):
        """
        make an animation of the displacement (saved to file)
        
        Parameters:
        -----------
        figname : str
        kwargs for StringBoundary.plot_displacement
        """
        tvec = np.linspace(0, self.period, 24)
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        cam = Camera(fig)
        for t in tvec[:-1]:
            ax = self.plot_expansion(t=t, ax=ax, no_title=True, **kwargs)
            plt.pause(.1)
            cam.snap()
        anim = cam.animate()
        print(f'Saving {figname}')
        plt.close()
        anim.save(figname, fps=12)
