from pywave.scattering.extended_shallow_water import ExtendedShallowWater


class ShallowWater(ExtendedShallowWater):
    """
    Class for scattering of linear shallow water waves.
    Special case of extended shallow water of Porter (2019)
    - beta=1, when vertical KE is neglected in the Lagrangian
    """

    @property
    def beta(self):
        """
        Override definition of beta in ExtendedShallowWater to
        give the ordinary shallow water equation

        Returns:
        --------
        beta : float
            beta coefficient in Helmholtz equation for q=hU
            (beta*q_x)_x + alpha*q = 0
        """
        return 1
