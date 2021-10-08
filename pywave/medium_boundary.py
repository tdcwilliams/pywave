import numpy as np

from pywave.scatterer_base import ScattererBase

_ZI = np.complex(0, 1)


class MediumBoundary(ScattererBase):
    
    def __init__(self, lhs, rhs, position=0):
        """
        Base scatterer class.
        Default calculates the scattering by a change in properties
        
        Parameters:
        -----------
        lhs : ElasticString or subclass
            object containing properties of the string segment to the left of the boundary
        rhs : ElasticString or subclass
            object containing properties of the string segment to the right of the boundary
        position : float
            x value where the boundary occurs
        """
        self.position = position
        super().__init__(media = [
            lhs.get_new(xlim=(-np.Inf, position)),
            rhs.get_new(xlim=(position, np.Inf)),
            ])


    def solve(self):
        """
        Solve the boundary conditions to set the scattering matrices
        
        Sets:
        -----
        Rp_ : numpy.ndarray(float)
            R^+ matrix (reflection of waves to right)
        Rm_ : numpy.ndarray(float)
            R^- matrix (reflection of waves to left)
        Tp_ : numpy.ndarray(float)
            T^+ matrix (transmission of waves to right)
        Tm_ : numpy.ndarray(float)
            T^- matrix (transmission of waves to left)
        """
        raise NotImplementedError(
                "solve method needs to be implemented in child classes")


    def test_boundary_conditions(self, inc_amps=None):
        """
        Test boundary conditions are satisfied
        """
        raise NotImplementedError(
                "test_boundary_conditions method needs to be implemented in child classes")
