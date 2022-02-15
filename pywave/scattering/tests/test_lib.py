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


if __name__ == "__main__":
    unittest.main()
