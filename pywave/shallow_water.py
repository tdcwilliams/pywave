from pywave.elastic_string import ElasticString


class ShallowWater(ElasticString):
    def __init__(self, rho_water=1025, depth=100, period=20, gravity=9.81, xlim=None):
        super().__init__(m=1, kappa = depth*gravity, period=period, xlim=xlim)
        self.rho_water = rho_water
        self.depth = depth
        self.gravity = gravity
