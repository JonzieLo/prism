# Volatility Observations, SVI, and Static Arbitrage

This folder converts a cleaned snapshot into a model-ready volatility smile.

## Observation contract

`pipeline.py` selects one canonical OTM option per expiry and strike:

- put when \(K<F\);
- call when \(K>F\);
- tighter valid side at the forward.

Coin premiums are converted into USD with the index. Black-76 inversion uses
the independently derived expiry forward. Each accepted observation stores:

- snapshot and source-row identity;
- expiry forward, rate, and time;
- bid, midpoint, and ask in coin and USD;
- own bid, midpoint, and ask IV where recoverable;
- log-moneyness;
- total variance;
- spread and liquidity fields.

Excluded observations retain a typed reason.

## Surface coordinates

The model coordinate is:

$$
k=\ln(K/F_T).
$$

The fitted quantity is:

$$
w(k,T)=\sigma_{\mathrm{imp}}(k,T)^2T.
$$

Log-moneyness aligns the forward point across expiries. Total variance is the
natural coordinate for calendar comparisons and SVI wing behavior.

## Raw SVI

`svi.py` implements:

$$
w(k)
=a+b\left[
\rho(k-m)+\sqrt{(k-m)^2+\eta^2}
\right].
$$

The model is a structural prior, not a claim that market observations are
noise. It compresses a strike-by-strike smile into level, skew, displacement,
curvature, and wing slopes.

The implementation exposes analytic first and second derivatives, implied
volatility, minimum variance, and asymptotic wing slopes.

### Comparison with the reference script

The supplied reference script and PRISM use the same raw-SVI family. The
reference script calibrates a quasi-SVI representation:

$$
a+dy+c\sqrt{y^2+1},
\qquad
y=(k-m)/\eta,
$$

then maps it to raw SVI using:

$$
b=c/\eta,
\qquad
\rho=d/c.
$$

Its two-step algorithm solves the linear \(a,d,c\) subproblem for each
candidate \(m,\eta\), then applies Nelder-Mead to \(m,\eta\). PRISM currently
uses deterministic multistart SLSQP directly on the five raw parameters. These
are alternative calibration algorithms for the same smile family.

PRISM does not copy the reference script directly because it does not provide:

- snapshot and source-row provenance;
- an explicit guarantee that the fitted target is total variance rather than
  raw IV despite the variable name `iv`;
- market-price repricing residuals;
- bid-ask-normalized residuals;
- deterministic calibration metadata;
- SVI density, call-convexity, bounds, or wing checks.

The reference two-step optimizer remains a useful later benchmark. If both
algorithms fit the same observations, their fitted curves and residuals should
be compared even when their parameter vectors differ.

## Calibration

`calibration.py` fits one expiry at a time using deterministic multistart
optimization in total-variance space. The first implementation deliberately
uses unweighted least squares:

$$
\min_\theta\sum_i
\left[w(k_i;\theta)-w_i^{market}\right]^2.
$$

After fitting, every observation is repriced with Black-76. Residuals are
reported in:

- total variance;
- implied volatility;
- USD price;
- units of the quoted half-spread.

The sign convention is always market minus model. Positive residuals indicate
market richness relative to SVI.

## Butterfly checks

`arbitrage.py` evaluates:

$$
g(k)
=
\left(1-\frac{kw'(k)}{2w(k)}\right)^2
-\frac{w'(k)^2}{4}
\left(\frac1{w(k)}+\frac14\right)
+\frac{w''(k)}2.
$$

A valid fitted slice requires nonnegative \(g(k)\) within tolerance on the
tested grid. PRISM independently reconstructs Black-76 calls and checks:

- positive total variance;
- price bounds;
- call monotonicity in strike;
- call convexity in strike;
- left and right wing slopes in \([0,2)\).

The finite grid and wing range are explicit diagnostics. Passing them is not a
proof over every real strike.

## Current scope and next steps

The current checkpoint fits and validates one expiry. It does not yet provide:

- calendar no-arbitrage across expiries;
- maturity interpolation;
- a complete 3D surface;
- model-free 30-day variance;
- DVOL comparison;
- historical SVI signals.

The first research extension will test whether executable leave-one-out SVI
residuals mean-revert through subsequent snapshots.
