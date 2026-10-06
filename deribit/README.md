# Market Data, Snapshots, Forwards, and Chain Hygiene

This folder owns the transformation from raw Deribit responses into
reproducible market observations.

## Snapshot transport

`config.py` defines the JSON-RPC requests for:

- option instruments;
- option book summaries;
- future book summaries;
- the underlying index.

`ws_client.py` sends those requests through one persistent WebSocket and records
the local receipt timestamp for each response. `store.py` persists both raw
responses and snapshot metadata in SQLite.

Every downstream result should retain `snapshot_id`. Historical comparison is
otherwise impossible to reproduce.

## Normalized option chain

`chain.py` joins market rows to instrument specifications and constructs
`OptionQuote` records. A coin midpoint exists only when the quote has:

- a bid;
- an ask;
- a positive bid;
- an ask not below the bid.

The USD-equivalent midpoint is:

\[
V_{\mathrm{USD}}=Xc,
\]

where \(X\) is the contemporaneous index and \(c\) is the coin premium. The
index performs denomination conversion. It is not substituted for the expiry
forward in a forward model.

## Options-implied forward

For inverse call and put premiums at the same strike:

\[
c-p=1-\frac{K}{F},
\]

so:

\[
F=\frac{K}{1-c+p}.
\]

Each complete strike pair provides:

- midpoint forward;
- synthetic-buy forward from executable sides;
- synthetic-sell forward from executable sides;
- matched-size diagnostics.

`forward_curve.py` aggregates diagnostic midpoint estimates by expiry using the
median and records dispersion. `forwards.py` compares the result with the
corresponding traded future.

Midpoint basis is diagnostic:

\[
\mathrm{Basis}_{bps}
=10{,}000\left(\frac{F_{\mathrm{options}}}
{F_{\mathrm{future}}}-1\right).
\]

It is not automatically tradeable because execution requires compatible call,
put, future, size, fee, and margin assumptions.

## Hygiene and accounting

`hygiene.py` records use-specific issues including:

- missing or nonfinite bid and ask;
- zero or nonpositive prices;
- crossed books;
- spread wider than the midpoint;
- invalid parity denominators;
- missing post-hoc comparison fields.

`segmentation.py` summarizes behavior by expiry, log-moneyness, and liquidity.
Every raw option row must reconcile to either a normalized quote or a chain
issue. Every normalized row must reconcile to a complete pair or pairing issue.

## Snapshot history

`SnapshotStore` supports:

- loading one raw snapshot;
- loading its metadata;
- listing snapshots newest first;
- selecting the latest snapshot by currency.

These APIs are the foundation for later A-versus-B surface comparison and P&L
attribution.
