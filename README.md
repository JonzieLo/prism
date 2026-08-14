# PRISM: Live Volatility Surface, Pricing Engine and Quoting Simulator

## Transport Layer & Optimizations ##
This project connects to Deribit, a crypto options and futures exchange. Deribit's API is free and provides orderbook data for BTC, ETH, SOL and other cryptocurrencies.

### 1. REST vs WebSocket ###
Deribit offers two endpoints: standard HTTP REST & WebSocket (JSON-RPC 2.0). 

To achieve microsecond-level market synchronicity, the transport layer is connected using **JSON-RPC 2.0 over persistent WebSockets (`wss://`)**. Standard HTTP REST endpoints requires each REST request operate over an independent TCP/TLS handshake, firing multiple REST calls creates severe concurrency issues at the exchange matching engine level.

Advantages of JSON-RPC over WebSocket:
* **Multiplexed Single-Pipe Design:** Instead of opening 4 separate HTTP connections, all 4 request frames (`index`, `instruments`, `options`, `futures`) are pipelined down a single, long-lived WebSocket stream.
* **Response Correlation:** Requests are tagged with incrementing `id` parameters to map incoming asynchronous frames back to their respective request keys.
* **Persistent Connection Reuse:** Opening a WebSocket connection incurs an initial TCP + TLS + HTTP 101 Upgrade overhead (~1,100–1,300 ms cold start). By retaining a persistent connection across snapshot captures, sub-millisecond network request dispatch is achieved.

---
### 2. Empirical Benchmark Results

Below is a live benchmark comparison capturing BTC/ETH options, futures, and index data from Deribit (Testnet):

| Metric | HTTP REST (Async Gather) | Persistent WebSocket (`wss://`) | Operational Significance |
| :--- | :--- | :--- | :--- |
| **Initial Connection Warm-Up** | N/A (~250 ms per request) | **~1,100 – 1,300 ms** | One-time startup handshake cost for establishing `wss://`. |
| **Warmed-Up Wall-Clock Window** | ~250 – 300 ms | **~30 – 300 ms** | Local desktop RTT is bounded by physical fiber distance to London (Equinix LD4). |
| **Exchange Server Skew (`usOut`)** | **2,000 – 8,000+ ms** | **~15 – 30 ms** | **Critical metric:** Server-side timestamp delta between index and options processing. |
---

### 3. Key Takeaway: Market Data Synchronicity

* **The Problem with REST (Server Skew):** HTTP REST requests result in processing gaps of **3 to 8 seconds**. During volatile markets, an index price moving +$200 between calls corrupts implied volatility calculations, making snapshots unusable.
* **WebSocket Solution (Exchange Coherence):** Over a single WebSocket stream, Deribit processes all pipelined JSON-RPC frames sequentially on the same worker pipeline in **< 30 ms**. Even when queried from a local desktop, the returned snapshot represents a perfectly coherent freeze-frame of the market.