import unittest
from mock import patch, call, MagicMock

import numpy as np

from pywave.scattering import lib


class TestLib(unittest.TestCase):

    def test_vector_min(self):

        self.assertEqual(1., lib.vector_min(1., 2.))
        self.assertTrue(np.allclose(
                np.array([1.,-1.]),
                lib.vector_min(1., np.array([2.,-1.]))
                ))
        self.assertTrue(np.allclose(
                np.array([1.,-1.]),
                lib.vector_min(
                    np.array([1.,1.]), np.array([2.,-1.])
                    )))

    def test_vector_max(self):

        self.assertEqual(2., lib.vector_max(1., 2.))
        self.assertTrue(np.allclose(
                np.array([2.,1.]),
                lib.vector_max(1., np.array([2.,-1.]))
                ))
        self.assertTrue(np.allclose(
                np.array([2.,1.]),
                lib.vector_max(
                    np.array([1.,1.]), np.array([2.,-1.])
                    )))

    def test_conservative_remapping_weights(self):

        # low res to high res
        x_src = np.linspace(0., 1., 11)
        x_dst = np.linspace(0., 1., 41)
        w = lib.conservative_remapping_weights(x_src, x_dst)
        self.assertTrue(np.allclose(
            w.sum(axis=0), np.ones((10))
            ))

        w2 = np.zeros((40,10))
        for i in range(10):
            w2[4*i:4*(i+1), i] = .25
        self.assertTrue(np.allclose(w, w2))

        # high res to low res
        x_src = np.linspace(0., 1., 41)
        x_dst = np.linspace(0., 1., 11)
        w = lib.conservative_remapping_weights(x_src, x_dst)
        self.assertTrue(np.allclose(
            w.sum(axis=0), np.ones((40))
            ))

        w2 = np.zeros((10,40))
        for i in range(10):
            w2[i, 4*i:4*(i+1)] = 1
        self.assertTrue(np.allclose(w, w2))


if __name__ == "__main__":
    unittest.main()
