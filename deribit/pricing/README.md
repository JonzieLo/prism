# Pricing Models, Inverse Settlement, and Greeks

This folder contains model mathematics. Market-data selection and surface
calibration belong elsewhere.

## Common interface

Each `OptionModel` exposes:

```python
price(...)
greeks(...)
implied_vol(...)
```

Implemented models:

- Black-Scholes in spot space;
- Black-76 in forward space;
- Bachelier with normal volatility;
- European CRR binomial tree.

The inverse layer is a denomination and Greek adapter, not a separate
stochastic volatility model.

## Spot and forward representations

When:

\[
F=Se^{rT},
\]

Black-Scholes and Black-76 produce the same option price but different native
partial derivatives. For example:

\[
\Delta_S=e^{rT}\Delta_F,
\qquad
\Gamma_S=e^{2rT}\Gamma_F.
\]

A Black-76 rho holds the forward fixed. A Black-Scholes rho moves the forward
through the spot carry relationship. Every reported Greek must therefore state
its value currency, bumped variable, and held-fixed convention.

## Inverse payoff and replication

An inverse BTC call pays:

\[
H_T^{BTC}=\frac{(S_T-K)^+}{S_T}.
\]

Multiplying by the delivery price produces the ordinary USD call payoff:

\[
S_TH_T^{BTC}=(S_T-K)^+.
\]

This supports a replication argument: replicate the USD payoff and convert its
terminal value into BTC at delivery. The current BTC quote is:

\[
c=\frac{V}{S}.
\]

Its raw coin delta is:

\[
\frac{\partial c}{\partial S}
=\frac{S\Delta_S-V}{S^2}.
\]

Net Transaction Delta is the scaled coin exposure:

\[
\mathrm{NTD}
=S\frac{\partial c}{\partial S}
=\Delta_S-c.
\]

Theta and rho require chain-rule terms when converting fixed-forward Greeks to
the paper's fixed-spot convention:

\[
\Theta_S=\Theta_F-rF\Delta_F,
\]

\[
\rho_S=\rho_F+TF\Delta_F.
\]

## Implied-volatility solvers

The lognormal models validate intrinsic and ceiling bounds, solve the price
residual, and verify the returned root by repricing. Bachelier returns normal
volatility in USD per square-root year. CRR inversion is numerical and should
converge toward Black-76 as tree resolution increases.

Deribit's published mark IV is not a model input. It may be retained only as a
post-hoc comparison.

## Numerical verification

Analytic Greeks are checked against central finite differences. Direct second
derivatives use a larger relative bump than first derivatives to avoid
float64 cancellation.
