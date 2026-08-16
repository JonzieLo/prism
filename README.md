# PRISM: Live Volatility Surface, Pricing Engine and Quoting Simulator

## Market Data Transport

PRISM streams live order book, instrument, and index data from Deribit (BTC, ETH, SOL).

### REST vs. Persistent WebSockets

To price option chains accurately, market snapshots must be captured with zero internal latency skew between the index price and option quotes. Standard REST batching requires multiple independent HTTP connections, creating concurrency bottlenecks and severe exchange server-side time skew.

PRISM uses **JSON-RPC 2.0 over a single persistent WebSocket stream (`wss://`)**:
* **Multiplexed Single Pipe:** All 4 snapshot frames (`index`, `instruments`, `options`, `futures`) are pipelined down a single WebSocket connection.
* **Request Correlation:** Unique incrementing `id` parameters correlate asynchronous frame returns back to their caller keys.
* **Connection Persistence:** Eliminates repeated TCP/TLS handshakes, reducing post-warmup request dispatch latency to sub-millisecond windows.

### Transport Performance (Testnet Benchmark)

| Metric | Async HTTP REST | Persistent WebSocket (`wss://`) | Notes |
| :--- | :--- | :--- | :--- |
| **Warm-Up Cost** | ~250 ms / request | **~1,100–1,300 ms** | One-time TCP/TLS handshake establishing `wss://`. |
| **RTT Window** | ~250–300 ms | **~30–300 ms** | Physical fiber RTT to Equinix LD4 (London). |
| **Server Skew (`usOut`)** | **2,000–8,000+ ms** | **~15–30 ms** | Timestamp delta between index and options processing. |

### Data Coherence & Server Skew

Fetching 4 separate REST endpoints introduces **2 to 8 seconds of exchange server skew**. In fast markets, a $200 index move mid-fetch skews option mark prices and distorts the implied volatility surface. Pipelining JSON-RPC frames over a single WebSocket forces Deribit to process the snapshot sequentially in **< 30 ms**, delivering a synchronized freeze-frame of the market.

---

## Pricing Engine & Model Layer

PRISM uses a pluggable model interface (`OptionModel`) exposing standard `price()`, `greeks()`, and `implied_vol()` signatures.

### Implemented Models

* **Black-Scholes (Spot-Based):** Standard spot pricing model ($S$). Assumes continuous lognormal asset dynamics and evaluates spot-constrained derivatives ($\frac{\partial V}{\partial S}\big|_r$).
* **Black-76 (Forward-Based):** Futures/forward pricing model ($F$). Evaluates forward-constrained derivatives ($\frac{\partial V}{\partial F}\big|_r$).

### Spot vs. Forward Greeks
While Black-Scholes and Black-76 yield identical option prices when $F = S e^{r\tau}$, their **Greeks represent different partial derivatives**:

* Bumping Spot $S$ implicitly moves Forward $F$ via cost-of-carry ($F = S e^{r\tau}$).
* Chain Rule ($\frac{\partial F}{\partial S} = e^{r\tau}$) links Spot and Forward sensitivities:

$$\Delta_{\text{BS}} = \frac{\partial V}{\partial S} = \frac{\partial V}{\partial F} \cdot \frac{\partial F}{\partial S} = e^{r\tau} \Delta_{76} = \Phi(d_1) \quad (\text{Call})$$

$$\Gamma_{\text{BS}} = \frac{\partial^2 V}{\partial S^2} = e^{2r\tau} \Gamma_{76} = \frac{\phi(d_1)}{S \sigma \sqrt{\tau}}$$

$$\text{Rho}_{\text{BS}} = +K \tau e^{-r\tau} \Phi(d_2) \quad \text{vs.} \quad \text{Rho}_{76} = -\tau \cdot \text{CallPrice}_{76}$$

*Note:* A Black-Scholes call is **long rates** (higher rates increase forward drift, raising call value), whereas a Black-76 call is **short rates** (higher rates increase the discount factor $e^{-r\tau}$ on a fixed forward).

### Model Formulations ($q = 0$)
| Metric / Greek | Black-Scholes (Spot $S$) | Black-76 (Forward $F$) |
| :--- | :--- | :--- |
| **$d_1$** | $\frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}$ | $\frac{\ln(F/K) + \frac{1}{2}\sigma^2\tau}{\sigma\sqrt{\tau}}$ |
| **$d_2$** | $d_1 - \sigma\sqrt{\tau}$ | $d_1 - \sigma\sqrt{\tau}$ |
| **Call Price** | $S \Phi(d_1) - K e^{-r\tau} \Phi(d_2)$ | $e^{-r\tau} [F \Phi(d_1) - K \Phi(d_2)]$ |
| **Call Delta ($\Delta$)** | $\Phi(d_1)$ | $e^{-r\tau} \Phi(d_1)$ |
| **Gamma ($\Gamma$)** | $\frac{\phi(d_1)}{S \sigma \sqrt{\tau}}$ | $\frac{e^{-r\tau} \phi(d_1)}{F \sigma \sqrt{\tau}}$ |
| **Vega ($\mathcal{V}$)** | $S \phi(d_1) \sqrt{\tau}$ | $F e^{-r\tau} \phi(d_1) \sqrt{\tau}$ |
| **Call Rho ($\rho$)** | $+K \tau e^{-r\tau} \Phi(d_2)$ | $-\tau \cdot \text{CallPrice}_{76}$ |

## Implied Volatility Engine

Implied volatility ($\sigma$) is solved using a two-stage root finder:

1. **Newton-Raphson (Primary):**
   $$\sigma_{n+1} = \sigma_n - \frac{V(\sigma_n) - P_{\text{market}}}{\mathcal{V}(\sigma_n)}$$
   Converges in 3–4 iterations for near-the-money quotes.
2. **Brent's Method Fallback:**
   For deep OTM options, analytical Vega ($\mathcal{V}$) approaches zero, causing Newton-Raphson to diverge or divide by zero. If Vega falls below $10^{-12}$, the solver falls back to `scipy.optimize.brentq` over $[10^{-6}, 5.0]$ to guarantee $10^{-10}$ convergence.

### Volatility Space Stopping Rule

Stopping rules evaluated in dollar space ($|P_{\text{calc}} - P_{\text{market}}| < 10^{-10}$) break down on deep OTM options where Vega is tiny ($\sim 10^{-4}$), leaving errors as large as $10^{-6}$ in volatility space. PRISM enforces convergence directly in **volatility space**:

$$\left| \frac{P_{\text{calc}} - P_{\text{market}}}{\mathcal{V}} \right| < 10^{-10}$$

---

## Unit Testing & Verification

The pricing test suite (`tests/test_pricing.py`) runs 57 automated assertions covering IV inversion, Put-Call Parity, boundary conditions, and analytical Greeks verified against central finite differences.

### Relative Finite-Difference Step Scaling ($h$)

Using a fixed absolute bump size (e.g., $h = 10^{-4}$) at $S = \$65,000$ falls below the float64 ULP cancellation floor for 2nd derivatives, producing negative Gamma values from machine roundoff noise. 

Central difference step sizes scale relative to the variable magnitude ($x$) based on float64 machine epsilon ($\epsilon \approx 2.22 \times 10^{-16}$):

* **1st Derivatives ($\Delta, \mathcal{V}, \Theta, \rho, \text{Vanna}, \text{Vomma}$):** $h = x \cdot \epsilon^{1/3} \approx x \cdot (6 \times 10^{-6})$
* **2nd Derivatives ($\Gamma$):** $h = x \cdot \epsilon^{1/4} \approx x \cdot (1.2 \times 10^{-4})$

At $S = \$65,000$, Gamma is tested using $h \approx \$7.80$, guaranteeing numerical stability.

### Test Coverage Summary

* **$10^{-10}$ IV Roundtrips:** Verifies $P \to \sigma \to P$ recovery across strikes ($\$50\text{k}, \$65\text{k}, \$80\text{k}$) and volatilities ($20\%, 55\%, 120\%$) for both Spot and Forward models.
* **Put-Call Parity:** Asserts $C - P = S - K e^{-r\tau}$ (BS) and $C - P = e^{-r\tau}(F - K)$ (B76) to $10^{-8}$ precision.
* **Finite-Difference Convergence:** Analytical Greeks match central differences within relative tolerances ($10^{-8}$ for Delta/Vega, $10^{-6}$ for Gamma/Vanna/Vomma).
* **Arbitrage Bounds:** Sub-intrinsic and above-ceiling prices explicitly raise `ValueError` to prevent surface fitting corruption.