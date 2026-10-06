# Benchmarks and Reproducible Figures

Benchmark modules are thin orchestration layers. Pricing, forward, and surface
mathematics remain under `deribit/`.

## Commands

### Delta and inverse exposure

```bash
python -m benchmarks.delta_strike_plot \
  --db snapshots.db \
  --snapshot-id 34 \
  --output figs/delta_vs_strike.png
```

### Forward curve and chain diagnostics

```bash
python -m benchmarks.forward_curve_plot \
  --db snapshots.db \
  --snapshot-id 34 \
  --output figs/forward_curve.png
```

### Raw IV or total variance

```bash
python -m benchmarks.raw_vol_surface_plot \
  --db snapshots.db \
  --snapshot-id 34 \
  --model black76 \
  --x-axis log_moneyness \
  --y-axis total_variance \
  --output figs/raw_iv_total_variance.png
```

### One fitted SVI smile

```bash
python -m benchmarks.svi_smile_plot \
  --db snapshots.db \
  --snapshot-id 34 \
  --expiry 1790323200000 \
  --max-abs-k 0.75 \
  --output figs/svi_smile.png
```

If `--expiry` is omitted, the benchmark prints and selects the expiry with the
largest number of eligible observations.

## Coach Neel checkpoint

The SVI figure is intended to answer:

1. What is the structural prior?
2. Which points determine the fit?
3. Where does the market disagree with SVI?
4. Does that disagreement exceed the executable spread?
5. Does the fitted slice pass butterfly checks?

The next feedback question is:

> Is a leave-one-out SVI residual, measured against executable bid and ask,
> a sensible relative-value signal to test before building more complex
> surface models?
