import numpy as np

from pywave.scattering.scatterer_base import ScattererBase


class MultipleScatterer(ScattererBase):
    
    def __init__(self, scatterers):
        self.scatterers = scatterers
        self._set_media()
        self._set_sizes()

    def _set_media(self):
        s0 = self.scatterers[0]
        self.media = s0.media[:1]
        for s1 in self.scatterers[1:]:
            med = s1.media[0]
            xlim = [s0.media[0].xlim[1], med.xlim[1]]
            self.media += [med.get_new(xlim=xlim)]
            s0 = s1
        self.media += s1.media[1:]
        self.num_media = len(self.media)
        self.num_interior_media = self.num_media - 2

    def _set_sizes(self):
        self.slices = []
        self.num_unknowns = 0
        for med in self.media[1:-1]:
            n_unk = self.num_unknowns
            self.num_unknowns += med.num_modes
            self.slices += [slice(n_unk, self.num_unknowns)]

    def _assemble_matrices(self):
        z = lambda : np.zeros((self.num_unknowns, self.num_unknowns), dtype=np.complex)
        a00, a01, a10, a11 = [z(), z(), z(), z()]
        for i, med in enumerate(self.media[1:-1]):
            m_diag = med.phase_matrix
            a01[self.slices[i], self.slices[i]] = (
                self.scatterers[i].Rm.dot(m_diag))
            a10[self.slices[i], self.slices[i]] = (
                self.scatterers[i+1].Rp.dot(m_diag))
            if i < self.num_interior_media - 1:
                a00[self.slices[i+1], self.slices[i]] = (
                    self.scatterers[i+1].Tp.dot(m_diag))
            if i > 0:
                a11[self.slices[i-1], self.slices[i]] = (
                    self.scatterers[i].Tm.dot(m_diag))
        return a00, a01, a10, a11

    def _assemble_vectors(self):
        bp = np.zeros((self.num_unknowns, self.media[0].num_modes), dtype=np.complex)
        bp[self.slices[0],:] = self.scatterers[0].Tp
        bm = np.zeros((self.num_unknowns, self.media[-1].num_modes), dtype=np.complex)
        bm[self.slices[-1],:] = self.scatterers[-1].Tm
        return bp, bm

    def _eliminate(self, a10, a11, bm):
        _, nm = bm.shape
        c = np.linalg.solve(np.eye(self.num_unknowns) - a11,
                            np.hstack([bm, a10]))
        return c[:,nm:], c[:,:nm]

    def _set_scattering_matrices(self):
        mat = self.scatterers[0].Tm.dot(self.media[1].phase_matrix)
        self.Rp = self.scatterers[0].Rp + mat.dot(self.bp[self.slices[0],:])
        self.Tm = mat.dot(self.bm[self.slices[0],:])
        mat = self.scatterers[-1].Tp.dot(self.media[-2].phase_matrix)
        self.Rm = self.scatterers[-1].Rm + mat.dot(self.am[self.slices[-1],:])
        self.Tp = mat.dot(self.ap[self.slices[-1],:])

    def solve(self):
        """ Solve the multiple scattering problem TODO implement! """
        a00, a01, a10, a11 = self._assemble_matrices()
        bp, bm = self._assemble_vectors()
        c10, cm = self._eliminate(a10, a11, bm)
        a = np.linalg.solve(np.eye(self.num_unknowns) - a00 - a01.dot(c10),
                            np.hstack([bp, a01.dot(cm)]))
        _, npos = bp.shape
        b = c10.dot(a)
        b[:,npos:] += cm
        self.ap, self.am = a[:,:npos], a[:,npos:]
        self.bp, self.bm = b[:,:npos], b[:,npos:]
        self._set_scattering_matrices()

    def test_boundary_conditions(self, inc_amps=None):
        if inc_amps is None:
            inc_amps = np.array([[1]]), np.array([[.5]])
        ip, im = inc_amps
        bvals_l = []
        for i, med in enumerate(self.media):
            bvals_l += [dict()]
            coeffs = self.get_solution_params(i, ip, im)
            x = med.xlim[np.isfinite(med.xlim)].flatten()
            for get_disp, name in zip(
                    [True, False], ['displacement', 'stress']):
                bvals_l[-1][name] = med.get_expansion(x, **coeffs, get_disp=get_disp)
        bvals_r = np.roll(bvals_l, -1)[:-1]
        bvals_l = np.array(bvals_l)[:-1]
        for i, (bv_l, bv_r) in enumerate(zip(bvals_l, bvals_r)):
            print(f'\nTest boundary conditions at boundary {i}:')
            for name, val in bv_l.items():
                print(f'{name} = {val[-1]} = {bv_r[name][0]}?')
                assert(np.allclose(val[-1:], bv_r[name][:1]))
        print("Boundary conditions are OK")
