import math

import numpy as np
from scipy.optimize import brentq

from .bachelier import NoTimeValueError
from .base import CallPut, Greeks, OptionModel, parse_cp

FD_REL_VOL = 1e-4
FD_REL_SPOT = 1e-4
FD_REL_RATE = 1e-4

INTRINSIC_ULP_TOL = 8.0

class BinomialModel(OptionModel):
    """
    Cox-Ross-Rubinstein binomial model for American options on Spot price.

    u = exp(vol * sqrt(dt)),  d = 1/u,  p = (exp(r*dt) - d) / (u - d)
    """

    def __init__(self, steps: int = 200):
        if steps < 3:
            raise ValueError(f"Steps must be >= 3 for Greeks extraction, got {steps}")
        self.steps = steps

    def _tree(self, spot, strike, tau, vol, rate, sign):
        """
        Vectorized backward induction. 
        """
        root = 0.0
        v1 = 0.0
        v2 = 0.0
        return root, v1, v2
    
    def _central_vol(self, spot, strike, tau, vol, hv, rate, sign):
        p_up = self._tree(spot, strike, tau, vol + hv, rate, sign)[0]
        p_dn = self._tree(spot, strike, tau, vol - hv, rate, sign)[0]
        return (p_up - p_dn) / (2.0 * hv)
    
    def price( self,
            spot: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut
            ) -> float:
        ...

    def greeks(self,
            spot: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut,
        ) -> Greeks:
        ...

    def implied_vol(self,
            price: float,
            spot: float,
            strike: float,
            tau: float,
            rate: float,
            cp: CallPut,
        ) -> float:
        ...