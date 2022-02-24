import unittest
from mock import patch, call, MagicMock, DEFAULT

import numpy as np

from pywave.tests.pywave_test_base import PywaveTestBase
from pywave.scattering.medium import Medium


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
            omega=DEFAULT,
            )
    def test_phase_velocity(self, **kwargs):
        """ test phase_velocity """
        med = Medium()
        med.omega = 3
        med.k = np.array([2.,3.])
        self.assertEqual(med.phase_velocity, 1.5)

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_group_velocity(self, **kwargs):
        """ test error raised for group_velocity """
        med = Medium()
        with self.assertRaises(NotImplementedError):
            med.group_velocity

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
