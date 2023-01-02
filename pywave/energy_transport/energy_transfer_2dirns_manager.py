import os
import numpy as np
from matplotlib import pyplot as plt
from configparser import ConfigParser
import json
import glob

from pywave.energy_transport.lib import solve_2d_ode_spectral
from pywave.energy_transport.energy_transfer_2dirns import EnergyTransfer2Dirns


class EnergyTransfer2DirnsManager(EnergyTransfer2Dirns):
    def __init__(self, dx, dt, cg,
                 neumann=True,
                 u_in=1,
                 reflection_coefficient=1,
                 alpha=.1,
                 gamma=0,
                 nx=100,
                 nt=50,
                 outdir='.',
                 figdir='.',
                 output_interval=100,
                 **kwargs):
        super().__init__(dx, dt, **kwargs)
        self.nx = int(nx)
        self.nt = int(nt)
        self.x = np.arange(nx, dtype=float) * dx
        self.neumann = neumann
        self.u_in = u_in
        self.outdir = outdir
        self.figdir = figdir
        self.output_interval = output_interval
        self._set_init_cons()
        self._set_scattering_props(alpha, gamma, cg)
        self._set_steady_cons(alpha, gamma, reflection_coefficient)

    @staticmethod
    def read_config(filename):
        cp = ConfigParser()
        with open(filename, 'r') as f:
            cp.read_file(f)
        sec = 'et2d_dirns'
        mappers = {}
        for opt in [
            'cfl',
            'dx',
            'dt',
            'alpha',
            'gamma',
            'cg',
            'u_in',
            ]:
            mappers[opt] = cp.getfloat
        for opt in [
            'nx',
            'nt',
            'nghost',
            'output_interval',
            ]:
            mappers[opt] = cp.getint
        for opt in [
            'outdir',
            'figdir',
            'scheme',
            'limiter',
            'u_correction_scheme',
            ]:
            mappers[opt] = cp.get
        for opt in [
            'aniso',
            'neumann',
            ]:
            mappers[opt] = cp.getboolean
        return {opt: mappers[opt](sec, opt) for opt in cp.options(sec)}

    @classmethod
    def from_config(cls, filename, verbose=False, **kwargs):
        opts = cls.read_config(filename)
        if verbose:
            print("Options from config file:\n" + json.dumps(opts, indent=4))
        return cls(**opts, **kwargs)

    def _set_init_cons(self):
        self.u0 = np.full_like(self.x, self.u_in)
        self.u0[self.x > self.x[self.nx // 3]] = 0
        self.v0 = np.zeros_like(self.x)
        self.u = self.u0.copy()
        self.v = self.v0.copy()

    def _set_scattering_props(self, alpha, gamma, cg):
        self.scattering_mask = np.zeros_like(self.x)
        self.scattering_mask[self.x > self.x[self.nx // 2]] = 1.
        self.alpha = alpha*self.scattering_mask
        self.gamma = gamma*self.scattering_mask
        self.cg = np.full_like(self.x, cg)

    def _set_steady_cons(self, alpha, gamma, reflection_coefficient):
        a, b, c, d = self.get_source_matrix(alpha, gamma, shape=(1,))
        dcg = np.gradient(self.cg) / self.dx
        abcd = [arr/self.cg[0] for arr in (a - dcg[0], b, -c, dcg[0] - d)]
        u0 = np.array([self.u_in])
        v0 = reflection_coefficient * u0
        self.u_steady = np.full_like(self.x, u0[0])
        self.v_steady = np.full_like(self.x, v0[0])
        wh = np.where(self.scattering_mask == 1.)
        x = self.x[wh]
        self.u_steady[wh], self.v_steady[wh] = (
            arr.flatten() for arr in solve_2d_ode_spectral(u0, v0, x - x[0], *abcd))

    def save_output(self, n):
        t = n * self.dt
        os.makedirs(self.outdir, exist_ok=True)
        npz_file = os.path.join(self.outdir, 'et2d_%5.5i.npz' %n)
        print(f'Saving {npz_file}')
        np.savez(npz_file,
                 x=self.x,
                 u=self.u,
                 v=self.v,
                 t=np.array([t]),
                 n=np.array([n]),
                 u_steady=self.u_steady,
                 v_steady=self.v_steady,
                )

    def run_one_step(self):
        self.u, self.v = self.step(
            self.u, self.v, self.cg, self.alpha, self.gamma,
            neumann=self.neumann, u_in=self.u_in)

    def run(self):
        # save the initial conditions
        self.save_output(0)
        # run the requested steps and save outputs regularly
        for n in range(self.nt):
            self.run_one_step()
            if (n + 1) % self.output_interval == 0:
                self.save_output(n + 1)
        # save the final conditions
        self.save_output(n + 1)
        
    def plot_one_step(self, x, t, u, v, u_steady, v_steady):
        fig, axes = plt.subplots(
            nrows=3, ncols=1, sharex=True, squeeze=True)
        for (ax, y, ys, ylabel) in zip(
            axes,
            [u, v, u + v],
            [u_steady, v_steady, u_steady + v_steady],
            ['$E_+$', '$E_-$', '$E_+E_-$'],
        ):
            ax.plot(x, ys, 'r')
            ax.plot(x, y, '--')
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_ylim([1e-4, 3])                         
        axes[0].set_title("t=%5.3f" %float(t))
        axes[-1].set_xlabel("x, m")
        return fig
        
    def plot_steps(self):
        os.makedirs(self.figdir, exist_ok=True)
        for npz in sorted(glob.glob(f'{self.outdir}/*.npz')):
            with np.load(npz) as f:
                fields = dict(f)
            n = int(fields.pop('n'))

            plt.close()
            fig = self.plot_one_step(**fields)
            figname = os.path.join(
                self.figdir,
                'et2d_%5.5i.png' %n,
            )
            print(f'Saving {figname}')
            fig.savefig(figname)
