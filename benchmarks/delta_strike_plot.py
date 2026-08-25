import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


INDEX = 65_000.0
TAU = 0.25
RATE = 0.03
LOGNORMAL_VOL = 0.55
NORMAL_VOL = INDEX * LOGNORMAL_VOL
STEPS = 400


def crr_delta(forward, strike, tau, vol, rate, steps):
    dt = tau / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    p = 1.0 / (1.0 + u)
    step_df = math.exp(-rate * dt)

    j = np.arange(steps + 1)
    terminal = forward * np.exp((steps - 2.0 * j) * math.log(u))
    values = np.maximum(terminal - strike, 0.0)

    while len(values) > 2:
        values = step_df * (p * values[:-1] + (1.0 - p) * values[1:])

    return (values[0] - values[1]) / (forward * (u - d))


forward = INDEX * math.exp(RATE * TAU)
df = math.exp(-RATE * TAU)
strikes = np.linspace(5_000.0, 140_000.0, 300)

std = LOGNORMAL_VOL * math.sqrt(TAU)
d1_spot = (
    np.log(INDEX / strikes)
    + (RATE + 0.5 * LOGNORMAL_VOL**2) * TAU
) / std
d1_forward = (
    np.log(forward / strikes) + 0.5 * LOGNORMAL_VOL**2 * TAU
) / std
d2_forward = d1_forward - std

bs_delta = norm.cdf(d1_spot)
b76_delta = df * norm.cdf(d1_forward)
normal_d = (forward - strikes) / (NORMAL_VOL * math.sqrt(TAU))
bachelier_delta = df * norm.cdf(normal_d)
crr = np.array(
    [
        crr_delta(forward, strike, TAU, LOGNORMAL_VOL, RATE, STEPS)
        for strike in strikes
    ]
)

# Total derivative of c_BTC = C_USD / X when F = X exp(rT).
inverse_delta = strikes * df * norm.cdf(d2_forward) / INDEX**2

with plt.rc_context(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "#F7F6F2",
        "axes.facecolor": "#F7F6F2",
        "axes.edgecolor": "#D4D1CA",
        "text.color": "#28251D",
        "axes.labelcolor": "#28251D",
        "xtick.color": "#28251D",
        "ytick.color": "#28251D",
    }
):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
        layout="constrained",
    )

    ax1.plot(strikes, bs_delta, color="#20808D", lw=2.4, label="Black–Scholes spot Δ")
    ax1.plot(
        strikes,
        b76_delta,
        color="#A84B2F",
        lw=2.2,
        ls="--",
        label="Black–76 forward Δ",
    )
    ax1.plot(
        strikes,
        bachelier_delta,
        color="#7A39BB",
        lw=2.0,
        ls="-.",
        label="Bachelier forward Δ",
    )
    ax1.plot(
        strikes,
        crr,
        color="#1B474D",
        lw=1.4,
        alpha=0.85,
        label=f"CRR forward Δ ({STEPS} steps)",
    )
    ax1.axvline(forward, color="#7A7974", lw=1.0, ls=":")
    ax1.text(
        forward + 1_300,
        0.91,
        f"Forward ≈ ${forward:,.0f}",
        color="#7A7974",
        fontsize=9,
    )
    ax1.set_ylim(-0.02, 1.03)
    ax1.set_ylabel("USD / forward delta")
    ax1.set_title(
        "Traditional call deltas fall as strike rises\n"
        "CRR approaches Black–76; Bachelier differs most in the wings",
        loc="left",
    )
    ax1.legend(frameon=False, ncol=2, loc="lower left")

    ax2.plot(
        strikes,
        inverse_delta * 100_000.0,
        color="#DA7101",
        lw=2.6,
        label="BTC-premium spot sensitivity",
    )
    peak = int(np.argmax(inverse_delta))
    ax2.scatter(
        [strikes[peak]],
        [inverse_delta[peak] * 100_000.0],
        color="#DA7101",
        edgecolor="#28251D",
        s=45,
        zorder=3,
    )
    ax2.annotate(
        "Low in both wings:\n"
        "deep ITM premium is near its 1 BTC ceiling;\n"
        "deep OTM exercise probability is near zero",
        xy=(strikes[peak], inverse_delta[peak] * 100_000.0),
        xytext=(88_000, inverse_delta[peak] * 100_000.0 * 0.68),
        arrowprops={"arrowstyle": "-", "color": "#7A7974"},
        fontsize=9,
        color="#28251D",
    )
    ax2.axvline(forward, color="#7A7974", lw=1.0, ls=":")
    ax2.set_xlabel("Strike K (USD)")
    ax2.set_ylabel("BTC delta × 100,000")
    ax2.set_title(
        "Inverse call delta is hump-shaped after the BTC conversion\n"
        r"$c_{BTC}=C_{USD}/X$ and "
        r"$\partial c_{BTC}/\partial X=K e^{-rT}N(d_2)/X^2$",
        loc="left",
    )
    ax2.legend(frameon=False, loc="upper right")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#D4D1CA", lw=0.8, alpha=0.7)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Delta vs. strike: model choice and inverse denomination",
        fontsize=18,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.savefig(
        "./delta_vs_strike.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )