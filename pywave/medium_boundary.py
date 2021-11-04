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
        lhs : Medium or subclass
            object containing properties of the medium segment to the left of the boundary
        rhs : Medium or subclass
            object containing properties of the medium segment to the right of the boundary
        position : float
            x value where the boundary occurs
        """
        self.position = position
        super().__init__(media = [
            lhs.get_new(xlim=(-np.Inf, position)),
            rhs.get_new(xlim=(position, np.Inf)),
            ])


    def solve(self, condition_types=None):
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
        forcings = []
        matrices = []
        for n, med in enumerate(self.media):
            # sort the operators
            med_factor = -1 ** n
            edge_ops = []
            for (op1, op2), is_continuous in zip(
                    med.edge_operators, condition_types):
                edge_ops += [med_factor * op1]
                if is_continuous:
                    edge_ops += [med_factor * op2]

            """
            1st half of matrix columns: "e^{ikx}" eigen functions
            2nd half of matrix columns: "e^{-ikx}" eigen functions
            """
            nk = len(med.k)
            matrix = np.zeros((len(edge_ops), 2*nk)
            for i, op in enumerate(edge_ops):
                for j, k in enumerate(med.k):
                matrix[i, j]    = op(k)
                matrix[i, j+nk] = op(-k)
            """
            for LHS (n=0):
                want the incident waves to be from left to right so choose
                "+k" columns from matrix
            for RHS (n=1):
                want the incident waves to be from right to left so choose
                "-k" columns from matrix
            """
            forcings += [matrix[:,n*nk:(n+1)*nk]]

        unknowns = np.linalg.solve(
            np.hstack(matrices), np.hstack(forcings))
        nk0 = len(self.media[0].k)
        nk1 = len(self.media[1].k)
        self.Rp_ = unknowns[:nk0, :nk0]            #reflected to left
        self.Tm_ = unknowns[nk0:2*nk0, nk0:]       #transmitted from right
        self.Tm_ = unknowns[2*nk0:2*nk0+nk1, :nk0] #transmitted from left
        self.Rm_ = unknowns[2*nk0+nk1:, nk0:]      #reflected from right


    def test_boundary_conditions(self, inc_amps=None):
        """
        Test boundary conditions are satisfied
        """
        raise NotImplementedError(
                "test_boundary_conditions method needs to be implemented in child classes")
