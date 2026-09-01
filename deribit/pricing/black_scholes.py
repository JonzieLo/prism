import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from .base import CallPut, Greeks, OptionModel, parse_cp

class BlackScholesModel(OptionModel):
    def price( self, 
            spot: float, 
            strike: float, 
            tau: float, 
            vol: float, 
            rate: float, 
            cp: CallPut):
        if tau <= 0.0 or vol <= 0.0:
            is_call = parse_cp(cp) == 1
            intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
            return intrinsic * math.exp(-rate * max(0.0, tau))

        df = math.exp(-rate * tau)
        std_dev = vol * math.sqrt(tau)
        d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / std_dev
        d2 = d1 - std_dev

        is_call = parse_cp(cp) == 1
        if is_call:
            return spot * norm.cdf(d1) - strike * df * norm.cdf(d2)
        return strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    def greeks(self,
            spot: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut):
        if tau <= 0.0 or vol <= 0.0:
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, vanna=0.0, vomma=0.0)

        df = math.exp(-rate * tau)
        sqrt_tau = math.sqrt(tau)
        std_dev = vol * sqrt_tau

        d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / std_dev
        d2 = d1 - std_dev
        is_call = parse_cp(cp) == 1

        delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
        gamma = norm.pdf(d1) / (spot * std_dev)
        vega = spot * norm.pdf(d1) * sqrt_tau

        theta_decay = -(spot * norm.pdf(d1) * vol) / (2.0 * sqrt_tau)
        if is_call:
            theta = theta_decay - rate * strike * df * norm.cdf(d2)
            rho = strike * tau * df * norm.cdf(d2)
        else:
            theta = theta_decay + rate * strike * df * norm.cdf(-d2)
            rho = -strike * tau * df * norm.cdf(-d2)

        vanna = -norm.pdf(d1) * d2 / vol
        vomma = vega * d1 * d2 / vol

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, vanna=vanna, vomma=vomma)

    def implied_vol(self,
            price: float,
            spot: float,
            strike: float,
            tau: float,
            rate: float,
            cp: CallPut):
        df = math.exp(-rate * tau)
        is_call = parse_cp(cp) == 1
        intrinsic = max(0.0, spot - strike * df) if is_call else max(0.0, strike * df - spot)

        if price < intrinsic:
            raise ValueError(f"Price {price} is below intrinsic value {intrinsic}")
        if price == intrinsic:
            return 0.0
        if price >= (spot if is_call else strike * df):
            raise ValueError(f"Price {price} exceeds maximum ceiling bound")

        def obj(v: float) -> float:
            return self.price(spot, strike, tau, v, rate, cp) - price

        vol_output = None
        v = 0.30
        # Newton-Raphson
        for _ in range(20):
            p = self.price(spot, strike, tau, v, rate, cp)
            diff = p - price
            v_greeks = self.greeks(spot, strike, tau, v, rate, cp)
            if v_greeks.vega < 1e-12:
                break

            step = diff / v_greeks.vega
            if abs(step) < 1e-10:
                if v - step > 0:
                    vol_output = v - step
                break
        # Brent's Method
        if vol_output == None:
            try:
                vol_output = brentq(obj, 1e-6, 5.0, xtol = 1e-10)
            except Exception:
                raise ValueError(f"Failed to converge implied vol for price {price}")
            
        if not math.isfinite(vol_output) or vol_output <= 0.0:
            raise ValueError(f"Implied volatility solver produced invalid root: {vol_output}")

        eps = float(np.finfo(float).eps)
        tolerance = max(1e-9, 100.0 * eps * price)

        repriced = self.price(spot, strike, tau, vol_output, rate, cp)
        price_diff = abs(repriced - price)

        if price_diff > tolerance:
            raise ValueError(
                f"Implied vol repricing check failed: market={price:.6f}, "
                f"repriced={repriced:.6f}, diff={price_diff:.2e} > tol={tolerance:.2e}"
            )

        return vol_output


class Black76Model(OptionModel):
    def price(self, 
            forward: float, 
            strike: float, 
            tau: float, 
            vol: float, 
            rate: float, 
            cp: CallPut):
        if tau <= 0.0 or vol <= 0.0: # At or past expiration
            is_call = parse_cp(cp) == 1
            intrinsic = max(0.0, forward - strike) if is_call else max(0.0, strike - forward)
            return intrinsic * math.exp(-rate * max(0.0, tau))
    
        df = math.exp(-rate * tau)
        std_dev = vol * math.sqrt(tau)
        d1 = (math.log(forward/strike) + 0.5 * vol * vol * tau) / std_dev
        d2 = d1 - std_dev

        is_call = parse_cp(cp) == 1
        if is_call:
            return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        else:
            return df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

    def greeks(self,
            forward: float,
            strike: float,
            tau: float,
            vol: float,
            rate: float,
            cp: CallPut):
        if tau <= 0.0 or vol <= 0.0: # At or past expiration
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, vanna=0.0, vomma=0.0)

        df = math.exp(-rate * tau)
        std_dev = vol * math.sqrt(tau)
        d1 = (math.log(forward/strike) + 0.5 * vol * vol * tau) / std_dev
        d2 = d1 - std_dev

        is_call = parse_cp(cp) == 1

        delta = df * norm.cdf(d1) if is_call else -df * norm.cdf(-d1)
        gamma = df * norm.pdf(d1) / (forward * std_dev)
        vega = forward * df * norm.pdf(d1) * math.sqrt(tau)
        theta = -forward * df * norm.pdf(d1) * vol / (2 * math.sqrt(tau)) - rate * strike * df * norm.cdf(d2) + rate * forward * df * norm.cdf(d1) if is_call else -forward * df * norm.pdf(d1) * vol / (2 * math.sqrt(tau)) + rate * strike * df * norm.cdf(-d2) - rate * forward * df * norm.cdf(-d1) 
        rho = -tau * df * (forward * norm.cdf(d1) - strike * norm.cdf(d2)) if is_call else -tau * df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
        vanna = vega / forward * (1 - d1/std_dev)
        vomma = vega * d1 * d2 / vol

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, vanna=vanna, vomma=vomma)

    def implied_vol(self,
            price: float,
            forward: float,
            strike: float,
            tau: float,
            rate: float,
            cp: CallPut):
        df = math.exp(-rate * tau)
        is_call = parse_cp(cp) == 1
        intrinsic = df * max(0.0, forward - strike) if is_call else df * max(0.0, strike - forward)

        if price <= intrinsic:
            raise ValueError(f"Price {price} is below intrinsic value {intrinsic}")
        if price >= df * (forward if is_call else strike):
            raise ValueError(f"Price {price} exceeds ceiling bound {df * (forward if is_call else strike)}")

        def obj(v: float) -> float:
            return self.price(forward, strike, tau, v, rate, cp) - price

        vol_output = None
        v = 0.30
        # Newton-Raphson
        for _ in range(20):
            p = self.price(forward, strike, tau, v, rate, cp)
            diff = p - price
            v_greeks = self.greeks(forward, strike, tau, v, rate, cp)
            if v_greeks.vega < 1e-10:
                break
            step = diff / v_greeks.vega
            if abs(step) < 1e-10:
                if v - step > 0:
                    vol_output = v - step
                break

        # Brent's Method
        if vol_output == None:
            try:
                return brentq(obj, 1e-6, 5.0, xtol = 1e-10)
            except Exception:
                raise ValueError(f"Failed to converge implied vol for price {price}")
            
        if not math.isfinite(vol_output) or vol_output <= 0.0:
            raise ValueError(f"Implied volatility solver produced invalid root: {vol_output}")

        eps = float(np.finfo(float).eps)
        tolerance = max(1e-9, 100.0 * eps * price)

        repriced = self.price(forward, strike, tau, vol_output, rate, cp)
        price_diff = abs(repriced - price)

        if price_diff > tolerance:
            raise ValueError(
                f"Implied vol repricing check failed: market={price:.6f}, "
                f"repriced={repriced:.6f}, diff={price_diff:.2e} > tol={tolerance:.2e}"
            )

        return vol_output