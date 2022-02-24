import unittest
from mock import patch, call, MagicMock, DEFAULT

import numpy as np

from pywave.tests.pywave_test_base import PywaveTestBase
from pywave.scattering.medium import Medium, _ZI


class MediumTest(PywaveTestBase):

    @patch.multiple(Medium,
            set_limits=DEFAULT,
            solve_disprel=DEFAULT,
            set_operators=DEFAULT,
            set_edge_operators=DEFAULT,
            )
    def test_init(self, **kwargs):
        """ test phase_velocity """
        med = Medium(xlim='xlim')
        kwargs['set_limits'].assert_called_once_with('xlim')
        kwargs['solve_disprel'].assert_called_once_with()
        kwargs['set_operators'].assert_called_once_with()
        kwargs['set_edge_operators'].assert_called_once_with()

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            )
    def test_set_limits(self):
        """ test phase_velocity """
        med = Medium()

        # default (infinite)
        xlim = [-np.inf, np.inf]
        med.set_limits(None)
        self.assertTrue(med.infinite)
        self.assertFalse(med.semi_infinite)
        self.assertIsInstance(med.xlim, np.ndarray)
        self.assertEqual(list(med.xlim), xlim)

        # infinite
        med.set_limits(xlim)
        self.assertTrue(med.infinite)
        self.assertFalse(med.semi_infinite)
        self.assertIsInstance(med.xlim, np.ndarray)
        self.assertEqual(list(med.xlim), xlim)

        # semi-infinite (1)
        xlim = [-np.inf, 0]
        med.set_limits(xlim)
        self.assertFalse(med.infinite)
        self.assertTrue(med.semi_infinite)
        self.assertIsInstance(med.xlim, np.ndarray)
        self.assertEqual(list(med.xlim), xlim)

        # semi-infinite (2)
        xlim = [0, np.inf]
        med.set_limits(xlim)
        self.assertFalse(med.infinite)
        self.assertTrue(med.semi_infinite)
        self.assertIsInstance(med.xlim, np.ndarray)
        self.assertEqual(list(med.xlim), xlim)

        # finite (1)
        xlim = [0, 1.]
        med.set_limits(xlim)
        self.assertFalse(med.infinite)
        self.assertFalse(med.semi_infinite)
        self.assertIsInstance(med.xlim, np.ndarray)
        self.assertEqual(list(med.xlim), xlim)

        # finite (2): error
        xlim = [2, 1.]
        with self.assertRaises(AssertionError):
            med.set_limits(xlim)

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_solve_disprel(self):
        """ test error raised for solve_disprel """
        med = Medium()
        with self.assertRaises(NotImplementedError):
            med.solve_disprel()

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_set_operators(self):
        """ test set_operators """
        med = Medium()
        med.set_operators()
        self.assertEqual(med.operators, {})

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_set_edge_operators(self):
        """ test set_edge_operators """
        med = Medium()
        med.set_edge_operators()
        self.assertEqual(med.edge_operators, {})

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            )
    def test_omega(self):
        """ test omega """
        med = Medium()
        med.period = 10.
        self.assertEqual(med.omega, .2*np.pi)

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            omega=DEFAULT,
            )
    def test_phase_velocity(self, **kwargs):
        """ test phase_velocity """
        med = Medium()
        med.omega = 3
        med.k = np.array([2.,3.])
        self.assertEqual(med.phase_velocity, 1.5)

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_group_velocity(self):
        """ test error raised for group_velocity """
        med = Medium()
        with self.assertRaises(NotImplementedError):
            med.group_velocity

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            )
    def test_num_modes(self, **kwargs):
        """ test num_modes """
        med = Medium()
        n = 10
        med.k = np.random.normal(size=(n,))
        self.assertEqual(med.num_modes, n)

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            )
    def test_phase_matrix(self, **kwargs):
        """ test phase_matrix """
        med = Medium()
        w = 10.
        med.xlim = [1., 1+w]
        med.k = np.random.uniform(size=(5,))
        self.assertTrue(np.allclose(
            med.phase_matrix,
            np.diag(np.exp(_ZI*w*med.k)),
            ))

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_get_energies(self, **kwargs):
        """ test error raised for get_energies """
        a0, a1 = np.array([[2.], [3.]])
        med = Medium()
        with self.assertRaises(NotImplementedError):
            med.get_energies(a0, a1)

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            group_velocity=DEFAULT,
            get_energies=DEFAULT,
            )
    def test_get_energy_flux(self, **kwargs):

        cg = 1.5
        a0, a1 = np.array([[2.], [3.]])
        e0, e1 = np.array([[4.], [5.]])

        med = Medium()
        med.group_velocity = cg
        kwargs["get_energies"].return_value = e0, e1
        f0, f1 = med.get_energy_flux(a0, a1)
        self.assert_mock_has_calls(
            kwargs["get_energies"], [call(a0, a1)])
        self.assertTrue(np.allclose(f0, -cg*e0))
        self.assertTrue(np.allclose(f1, cg*e1))


if __name__ == "__main__":
    unittest.main()
