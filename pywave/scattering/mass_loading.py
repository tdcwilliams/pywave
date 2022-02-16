from pywave.scattering.extended_shallow_water import ExtendedShallowWater


class MassLoadingBase(ExtendedShallowWater):
    """
    Base class for mass loaded water
    """

    def __init__(self, rho_ice=922.5, thickness=1, **kwargs):

        self.rho_ice = rho_ice
        self.thickness = thickness
        super().__init__(**kwargs)

    @property
    def draft(self):
        """
        Returns:
        --------
        draft : float
            part of the thickness that is submerged
        """
        return self.rho_ice * self.thickness / self.rho_water


class MassLoadingESW(MassLoadingBase):
    """
    Class for scattering of linear extended shallow water waves
    covered by a floating material modelled by the mass loading model
    (vertical KE included but no horizontal KE, and no horizontal interactions
    (no PE)).
    Generalizes extended shallow water of Porter (2019) by modifying beta
    """

    @property
    def beta(self):
        """
        Override definition of beta in ExtendedShallowWater to
        give the mass-loaded extended shallow water equation

        Returns:
        --------
        beta : float
            beta coefficient in Helmholtz equation for q=hU
            (beta*q_x)_x + alpha*q = 0
        """
        el = self.draft + self.depth/3
        return 1 - self.wave_number_ow_id * el


class MassLoadingSW(MassLoadingBase):
    """
    Class for scattering of linear shallow water waves
    covered by a floating material modelled by the mass loading model
    (vertical KE included but no horizontal KE, and no horizontal interactions
    (no PE)).
    Generalizes extended shallow water of Porter (2019) by modifying beta
    """

    @property
    def beta(self):
        """
        Override definition of beta in ExtendedShallowWater to
        give the mass-loaded shallow water equation

        Returns:
        --------
        beta : float
            beta coefficient in Helmholtz equation for q=hU
            (beta*q_x)_x + alpha*q = 0
        """
        return 1 - self.wave_number_ow_id * self.draft
