import numpy as np

from pywave.scattering.scatterer_base import ScattererBase

_ZI = np.complex(0, 1)


class MediumBoundary(ScattererBase):
    
    def __init__(self, lhs, rhs, position=0, condition_types=None):
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
        self.set_condition_types(condition_types=condition_types)

    def sort_edge_operators(self):
        """
        Returns:
        --------
        names_common : list(str)
            list of edge operators common to both media
        names_diff : tuple
            names_diff[i] is type list(str) with
            list of edge operators in self.media[i] that is not
            in the other one
        """
        names0 = set(self.media[0].edge_operators)
        names1 = set(self.media[1].edge_operators)
        names_common = names0.intersection(names1)
        names0 = names0.difference(names_common)
        names1 = names1.difference(names_common)
        return names_common, (names0, names1)

    def set_condition_types(self, condition_types=None):
        """
        Set the type of edge conditions to apply

        Parameters:
        -----------
        condition_types : list(dict)
            should be one element for each medium
            each element should be of type dict(bool)
            with names corresponding to medium.edge_operators

        Sets:
        -----
        self.condition_types : list(dict)
            same as condition_types
            default is operators common to the 2 media should be continuous
            and others should be zero
        """
        names_common, _ = self.sort_edge_operators()
        if condition_types is not None:
            assert(len(condition_types)==2)
            for med, ct in zip(self.media, condition_types):
                assert(list(ct) == names_common)
            self.condition_types = condition_types
            return
        self.condition_types = []
        for med in self.media:
            self.condition_types += [{name : True
                for name in med.edge_operators}]

    def assemble(self):
        """
        Create the system of linear equations to solve

        Returns:
        -----
        matrix : numpy.ndarray
            matrix to invert to solve the system
        forcing : numpy.ndarray
            vectors on right hand side corresponding to incident wave forcing
        """
        names_common, names_diff = self.sort_edge_operators()
        # num conditions
        nrows = 2*len(names_common) + np.sum([len(lst) for lst in names_diff])
        # num unknowns
        nk0, nk1 = [len(med.k) for med in self.media]
        ncols = nk0 + nk1
        matrix = np.zeros((nrows,ncols), dtype=np.complex)
        forcing = np.zeros((nrows,ncols), dtype=np.complex)
        slices = [slice(None, nk0), slice(nk0, None)]
        n_eqns = 0

        # common operators first
        # - may be continuous or zero
        for name in names_common:
            for n, med in enumerate(self.media):
                # sort the operators
                med_factor = pow(-1, n)
                ops = med.edge_operators[name]
                is_continuous = self.condition_types[n][name]
                if is_continuous:
                    for j, op in enumerate(ops):
                        matrix[n_eqns+j,slices[n]] = (
                                med_factor*op(-med_factor*med.k))
                        forcing[n_eqns+j,slices[n]] = (
                                -med_factor*op(med_factor*med.k))
                else:
                    op1 = ops[0]
                    matrix[n_eqns+n,slices[n]] = op1(-med_factor*med.k)
                    forcing[n_eqns+n,slices[n]] = -op1(med_factor*med.k)
            n_eqns += 2

        # operators that are not common
        # - must be zero
        for n, med in enumerate(self.media):
            med_factor = -1 ** n
            for name in names_diff[n]:
                op1 = med.edge_operators[name][0]
                matrix[num_eqns+n,slices[n]] = op1(-med_factor*med.k)
                forcing[n_eqns+n,slices[n]] = -op1(med_factor*med.k)
                n_eqns += 1

        assert(n_eqns==nrows)
        return matrix, forcing

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

        # set up and solve the system
        matrix, forcing = self.assemble()
        unknowns = np.linalg.solve(matrix, forcing)

        # sort result
        nk0 = len(self.media[0].k)
        self.Rp = unknowns[:nk0, :nk0] #reflected to left
        self.Tm = unknowns[:nk0, nk0:] #transmitted from right
        self.Tp = unknowns[nk0:, :nk0] #transmitted from left
        self.Rm = unknowns[nk0:, nk0:] #reflected from right

    def test_boundary_conditions(self, inc_amps=None):
        """
        Test boundary conditions are satisfied
        """
        raise NotImplementedError(
                "test_boundary_conditions method needs to be implemented in child classes")
