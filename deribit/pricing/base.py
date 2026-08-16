from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

CallPut = Literal["call", "put"]

@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    vomma: float

class OptionModel(ABC):
    @abstractmethod
    def price(
        self, 
        forward: float, 
        strike: float, 
        tau: float, 
        vol: float, 
        rate: float, 
        cp: CallPut
    ) -> float:
        """
        Calculates fair price by X pricing model.

        :param forward: Forward price of the underlying asset (e.g. $65,000 for BTC-28AUG26)
        :param strike: Strike price of the option
        :param tau: Time to expiration, annualized (days/365)
        :param vol: Annualized volatility (0.15 indicates 15%)
        :param rate: Annualized risk-free interest rate (0.05 indicates 0.5%)
        :param cp: Option type string ("call", "put")
        """
        ...

    @abstractmethod
    def greeks(
        self,
        forward: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        cp: CallPut
    ) -> Greeks:
        """
        Calculates Greeks (Delta, Gamma, Vega, Theta)

        :param forward: Forward price of the underlying asset (e.g. $65,000 for BTC-28AUG26)
        :param strike: Strike price of the option
        :param tau: Time to expiration, annualized (days/365
        :param vol: Annualized volatility (0.15 indicates 15%)
        :param rate: Annualized risk-free interest rate (0.05 indicates 0.5%)
        :param cp: Option type string ("call", "put")
        """
        ...

    
    @abstractmethod
    def implied_vol(
        self,
        price: float,
        forward: float,
        strike: float,
        tau: float,
        rate: float,
        cp: CallPut
    ) -> float:
        """
        Inverts model price to calculate implied volatility via root-finding.

        :param price: Market dollar price of the option
        :param forward: Forward price of the underlying asset (e.g. $65,000 for BTC-28AUG26)
        :param strike: Strike price of the option
        :param tau: Time to expiration, annualized (days/365)
        :param rate: Annualized risk-free interest rate (0.05 indicates 0.5%)
        :param cp: Option type string ("call", "put")
        """
        ...