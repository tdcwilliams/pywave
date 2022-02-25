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

    @patch.multiple(Medium,
            solve_disprel=DEFAULT,
            set_operators=DEFAULT,
            set_edge_operators=DEFAULT,
            )
    def test_get_new(self, **kwargs):
        """ test phase_matrix """
        xlim = np.array([0.,1.])
        med = Medium(xlim=xlim)
        atts = vars(med)
        del atts['xlim']

        xlim2 = xlim + 1.
        med2 = med.get_new(xlim2)
        self.assertTrue(np.allclose(med2.xlim, xlim2))
        atts2 = vars(med2)
        del atts2['xlim']
        self.assertEqual(atts, atts2)

        kwargs['solve_disprel'].assert_called_once_with()
        kwargs['set_operators'].assert_called_once_with()
        kwargs['set_edge_operators'].assert_called_once_with()

    @patch.multiple(Medium, __init__=MagicMock(return_value=None))
    def test_is_in_domain(self):
        med = Medium()
        med.xlim = np.array([2.,3])
        x = [-np.inf, -5., 2., 2.5, 3., 5., np.inf]
        y = np.array([False, False, True, True, True, False, False])
        self.assertTrue(np.allclose(
            med.is_in_domain(x), y))

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            )
    def test_x_scattering_src(self):
        """ test phase_velocity """
        med = Medium()

        # infinite
        med.set_limits([-np.inf, np.inf])
        self.assertEqual(list(med.x_scattering_src), [0,0])

        # semi-infinite
        med.set_limits([-np.inf, 1.])
        self.assertEqual(list(med.x_scattering_src), [1.,1.])
        med.set_limits([1., np.inf])
        self.assertEqual(list(med.x_scattering_src), [1.,1.])

        # finite
        med.set_limits([.5, 1.])
        self.assertEqual(list(med.x_scattering_src), [.5,1.])

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            x_scattering_src=DEFAULT,
            is_in_domain=DEFAULT,
            )
    def test_get_expansion_1(self, **kwargs):
        """ test with default operator"""
        x0 = -.5
        x1 = .5
        a0 = np.array([1., 1+_ZI])
        a1 = np.array([2., 2+_ZI])
        k = np.array([1., 1+_ZI])
        x = np.linspace(-.5, .5, 10)
        y = 0
        for i in range(2):
            y += a0[i]*np.exp(_ZI*k[i]*(x-x0))
            y += a1[i]*np.exp(_ZI*k[i]*(x1-x))

        x = np.array([-1., *x, 1.])
        b = np.ones_like(x, dtype=bool)
        b[0] = b[-1] = False
        kwargs['is_in_domain'].return_value = b
        med = Medium()
        med.x_scattering_src = np.array([x0, x1])
        med.k = k
        y2 =med.get_expansion(x, a0, a1)
        self.assertTrue(np.allclose(y, y2[1:-1]))
        for i in [0,-1]:
            self.assertTrue(np.isnan(y2[i]))

    @patch.multiple(Medium,
            __init__=MagicMock(return_value=None),
            x_scattering_src=DEFAULT,
            is_in_domain=DEFAULT,
            )
    def test_get_expansion_2(self, **kwargs):
        """ test with custom operator"""
        x0 = -.5
        x1 = .5
        a0 = np.array([1., 1+_ZI])
        a1 = np.array([2., 2+_ZI])
        k = np.array([1., 1+_ZI])
        op = lambda x : x**2 - x**3
        x = np.linspace(-.5, .5, 10)
        y = 0
        for i in range(2):
            y += a0[i]*op( k[i])*np.exp(_ZI*k[i]*(x-x0))
            y += a1[i]*op(-k[i])*np.exp(_ZI*k[i]*(x1-x))

        x = np.array([-1., *x, 1.])
        b = np.ones_like(x, dtype=bool)
        b[0] = b[-1] = False
        kwargs['is_in_domain'].return_value = b
        med = Medium()
        med.x_scattering_src = np.array([x0, x1])
        med.k = k
        y2 =med.get_expansion(x, a0, a1, operator=op)
        self.assertTrue(np.allclose(y, y2[1:-1]))
        for i in [0,-1]:
            self.assertTrue(np.isnan(y2[i]))

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
