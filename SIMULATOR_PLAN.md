# Simulator Plan

A plain-language plan for the simulator described in [paper/main.tex](paper/main.tex).

---

## 1. What question is the simulator answering?

When a request arrives at an LLM serving system that splits **prefill** and **decode** onto different GPUs, the router has to pick a decode GPU at some point. It can pick:

- **Early (on arrival):** the prefill GPU knows where the KV cache is going, so it can start sending each layer's KV cache as soon as that layer is computed. The transfer overlaps with prefill. The catch: the router chose using information that may be stale by the time the data actually moves.
- **Late (after prefill finishes):** the router picks the decode GPU using fresh information about load and congestion. The catch: the whole KV cache has to be sent *after* prefill, so the user waits for the full transfer.

**Our hypothesis:** early binding wins only while the network path can carry KV bytes at least as fast as prefill produces them. If the network is slower than that, the KV data piles up anyway and the overlap doesn't help much, and late binding's better choice wins.

We'll call that ratio **ρ (rho)**:

```
ρ = (bandwidth the KV transfer actually gets on its path) / (rate prefill produces KV bytes)
```

- ρ ≥ 1 → network keeps up → early binding should win.
- ρ < 1 → network is the bottleneck → late binding may win.

The simulator lets us sweep the things that change ρ (oversubscription, background traffic, prompt length, load) and see which policy gives lower latency.

---

## 2. Terms used in this plan

| Term | Plain meaning |
|---|---|
| **Prefill** | Processing the whole prompt in one go. Compute-heavy. Produces the first output token and the KV cache. |
| **Decode** | Generating output tokens one at a time. Memory-heavy. Needs the KV cache from prefill. |
| **KV cache** | The saved "keys and values" for every token in every layer. Decode needs it so it doesn't redo prefill. For Llama-3.1-70B it is about **320 KB per token** (80 layers × 8 KV heads × 128 dims × 2 for K and V × 2 bytes). A 4,000-token prompt is ~1.3 GB. |
| **Instance** | One copy of the model running on a group of GPUs. Here: 4 GPUs per instance (tensor parallelism 4), so 2 instances per 8-GPU server. |
| **NVLink domain** | GPUs inside one server, connected by very fast links. Transfers inside it are nearly free. |
| **Leaf–spine** | A common datacenter network. Each rack has a **leaf** switch connecting its servers. Leaves connect to each other through **spine** switches. Same-rack traffic goes server → leaf → server. Cross-rack traffic goes server → leaf → spine → leaf → server. |
| **Oversubscription** | When a leaf's links *up* to the spines have less total capacity than its links *down* to servers. 2:1 means only half the servers can send cross-rack at full speed at once. Higher oversubscription = more congestion for cross-rack KV transfers. |
| **Background traffic** | Other data on the same network (not our KV transfers) that competes for bandwidth. |
| **Flow-level simulation** | We don't simulate individual packets. We treat each transfer as a "flow" that gets some share of each link's bandwidth, and recompute the shares whenever a flow starts or ends. Much faster than packet simulation, good enough for our question. |
| **Max-min fairness** | The rule for splitting bandwidth: flows sharing a bottleneck link get equal shares; any bandwidth a flow can't use (because it's limited elsewhere) goes to the others. |
| **Discrete-event simulation** | The simulator keeps a list of future events ("prefill finishes at t=1.23s", "flow ends at t=1.40s"), always jumps to the next one, updates state, and schedules new events. No real clock is involved. |
| **TTFT** | Time to first token: request arrival → first output token. |
| **Second-token latency** | Arrival → second output token. In a split system, the first token comes from prefill, but the second needs decode, so it needs the KV cache to have arrived. This is where the binding choice shows up most clearly. |
| **p99** | The 99th percentile: the latency that only 1% of requests exceed. Shows the bad tail, not just the average. |
| **Trace** | A recorded log of real requests (arrival time, prompt length, output length) we replay. |

---

## 3. What we are building, piece by piece

The diagram in the paper has five boxes. Blue boxes (Router, Network) are what we model in detail. White boxes (Prefill, Decode) use timing numbers from published measurements.

```
   Trace file ──► Workload ──► Router ──► Prefill instances ──► Network ──► Decode instances ──► Metrics
                                 ▲                                │
                                 └──── (delayed) congestion info ─┘
```

### 3.1 Event engine
The core loop. A priority queue of events sorted by time. Pop the earliest, handle it, possibly add new events. Everything else plugs into this.

### 3.2 Workload (requests)
- Load the **Mooncake** traces (FAST25 release) and the **Azure LLM 2023** traces (code + conversation). Each request needs: arrival time, prompt length, output length.
- Ability to speed up or slow down arrivals (to vary load), and to filter by prompt length.
- Also a simple synthetic generator (e.g. random arrivals, fixed prompt length) for testing.

### 3.3 Cluster and topology
- 128 GPUs = 16 servers × 8 GPUs = 32 instances (2 per server).
- Servers grouped into racks (e.g. 4 servers per rack → 4 racks, but make it configurable).
- Each instance is labelled **prefill** or **decode**; the P:D split is a setting we sweep.
- Links: server ↔ leaf, leaf ↔ spine. Each has a bandwidth. Oversubscription is set by scaling the leaf ↔ spine bandwidth.
- Routing a transfer = finding the list of links between two servers. In leaf–spine this is easy: same server (NVLink, effectively free), same rack (2 links), different racks (4 links, pick a spine by hashing — like ECMP in real networks).

### 3.4 Network (flow-level, max-min fair)
- Keep a set of active flows. Each flow has: remaining bytes, list of links it crosses, current rate.
- Whenever a flow is added or finishes, recompute every flow's rate using **water-filling**:
  1. Find the link where the fair share (remaining capacity ÷ number of unfrozen flows on it) is smallest.
  2. Give all flows on that link that share; freeze them; subtract their rate from every link they use.
  3. Repeat until all flows are frozen.
- Then compute when the next flow will finish and schedule that event.
- Also report per-link utilization so the router can read it.

### 3.5 Background traffic
- Extra flows between random server pairs that compete with KV transfers.
- Settings: average load (e.g. 30% of link capacity) and **burstiness** (steady trickle vs. occasional big bursts). An on/off pattern is enough.

### 3.6 Prefill instances (from profiling data)
- Each has a queue. When free, it takes the next request (or a small batch) and runs prefill.
- Prefill time comes from a **lookup table** of measured numbers (prompt length → time) from Frontier / Vidur / SplitwiseSim's `perf_model.csv`. We don't simulate GPU math.
- For layer-wise streaming, assume each of the 80 layers takes an equal slice of the prefill time, so layer *i*'s KV is ready at `start + (i/80) × prefill_time`.

### 3.7 KV transfer
- **Late binding:** one flow of `prompt_tokens × 320 KB` starts when prefill finishes.
- **Early binding:** 80 smaller pieces, each released when its layer finishes. Simplest version: treat it as one flow whose "available bytes" grow over time, so it can never send data that hasn't been computed yet.
- Done when the last byte arrives at the decode instance.

### 3.8 Decode instances (from profiling data)
- Each has limited memory for KV caches (how many requests it can hold).
- Runs in iterations: every iteration produces one token for every request in its batch. Iteration time comes from a lookup table (batch size, context length → time).
- A request joins the batch once its KV cache has fully arrived and there is memory for it.

### 3.9 Router and the "stale view"
- The router doesn't see the true network state. It sees a **snapshot** that is refreshed every *Δ* milliseconds (the configurable delay). This is how we model "early binding uses stale info."
- The snapshot includes: link utilization along candidate paths, decode queue lengths / free memory.

### 3.10 Binding policies
1. **Early binding (Mooncake-style):** on arrival, pick prefill and decode instance together (e.g. least-loaded decode, maybe prefer same rack). Stream KV layer by layer.
2. **Late binding (NetKV-style):** on arrival, pick prefill only. When prefill finishes, pick the decode instance with the best expected "transfer time + queue wait" using the current snapshot. Send KV in one go.
3. **Threshold / hybrid:** estimate ρ for this request (prompt length + current path bandwidth). If ρ ≥ threshold, bind early; otherwise bind late.
4. *(If time permits)* **Early + migrate (Llumnix-style):** bind early, but if the chosen decode instance becomes a bad choice, move the request later.

All policies should share one interface (e.g. `on_arrival(request)`, `on_prefill_done(request)`) so adding new ones is easy.

### 3.11 Metrics and output
For each request record: arrival, prefill start/end, transfer start/end, first token time, second token time, which instances were used.
Summaries per run: mean and p99 of TTFT and second-token latency, plus "exposed transfer time" (how much of the transfer the user actually waited for). Write one CSV per run so plots can be made separately.

---

## 4. The simple math model (do this first, on paper)

Before trusting the simulator, write down what we *expect*, ignoring queues:

- **Late binding:** `second token ≈ prefill_time + KV_size / bandwidth + one decode step`
- **Early binding:** transfer runs alongside prefill, so only the leftover sticks out:
  `second token ≈ prefill_time + max(small, KV_size / bandwidth − prefill_time) + one decode step`

If bandwidth is high (ρ ≥ 1), early binding hides almost all the transfer. If bandwidth is low, both end up waiting for the network, and early binding loses its advantage — and is worse if its stale choice picked a congested path.

**Rough sanity check** (to be replaced with real numbers): a 4-GPU H100 instance might prefill on the order of ~10k tokens/s for a 70B model, so it produces roughly 10k × 320 KB ≈ 3–5 GB/s of KV (≈ 25–40 Gbps). That is below one 400 Gbps NIC, but once you add oversubscription, background traffic, and several transfers sharing one uplink, the share a single transfer gets can drop below it. So the crossover point is plausibly inside realistic settings — that's why the question is interesting.

The simulator's results with no queuing should match these formulas. That is our first correctness check.

---

## 5. How we check the simulator is right

1. **Network unit tests:**
   - One flow on an empty link gets full bandwidth.
   - Two flows on one link get half each.
   - A flow limited elsewhere leaves its unused share to others (max-min).
   - Cross-rack flows slow down when oversubscription goes up; same-rack flows don't.
2. **Match the math model** (section 4) with one request at a time and no background traffic.
3. **Reproduce published trends** from DistServe, NetKV, and ObjectCache — the shape of the curves, not exact numbers.
4. **Cross-check a few setups in LLMServingSim 2.0** and compare latencies.

---

## 6. Experiments (what we sweep)

| Knob | Why it matters | Example values |
|---|---|---|
| Oversubscription | Lower cross-rack bandwidth → lower ρ | 1:1, 2:1, 4:1, 8:1 |
| Background traffic load & burstiness | Steals bandwidth, makes snapshots go stale | 0%, 30%, 60%; steady vs bursty |
| Prompt length / workload | Longer prompts = bigger KV cache, longer prefill | Mooncake (long), Azure code, Azure conversation |
| Request rate (load) | More queuing, more competing transfers | low → near saturation |
| P:D ratio | How many decode targets, how many cross-rack | e.g. 1:1, 1:2, 1:3 |
| Router staleness Δ | How old the router's view is | 0, 1 ms, 10 ms, 100 ms |

Output: a **map** of "which policy wins" over these settings, and a fitted **threshold rule** (e.g. "bind early if estimated ρ > X") that becomes the hybrid policy.

---

## 7. Suggested code layout (Python)

```
sim/
  engine.py        # event queue and main loop
  workload.py      # trace loaders + synthetic generator
  topology.py      # servers, racks, links, path lookup
  network.py       # flows + max-min fair rate calculation
  background.py    # background traffic generator
  instances.py     # prefill and decode instances, queues, memory
  profiles.py      # lookup tables for prefill/decode timing
  router.py        # stale snapshots + policy interface
  policies/        # early.py, late.py, threshold.py, migrate.py
  metrics.py       # per-request records, summaries, CSV output
configs/           # one YAML per experiment setup
experiments/       # scripts that run sweeps
analysis/          # plotting notebooks/scripts
tests/             # unit tests (network first!)
data/              # traces and profiling tables (downloaded, not committed if large)
```

---

## 8. Build order and timeline

The paper's schedule, mapped to the pieces above. (As of Nov 1 the repo has no code yet, so weeks 1–2 are effectively still open — worth checking where you actually are.)

| Week | Goal | Pieces |
|---|---|---|
| 1 (Oct 19) | Math model + data | Section 4 on paper; download traces; collect prefill/decode timing tables (3.2, 3.6 tables) |
| 2 (Oct 26) | Basic simulator runs | Event engine (3.1), topology (3.3), network + tests (3.4, 5.1), requests/queues/KV sizes (3.2, 3.6, 3.8) — late binding only, no background traffic |
| 3 (Nov 2) | All policies | Layer-wise streaming (3.7), background traffic (3.5), stale router view (3.9), early/late/threshold policies (3.10) |
| 4 (Nov 9) | Verified | Checks in section 5 |
| 5 (Nov 16) | Main results | Sweeps in section 6; migration policy if time |
| 6 (Nov 23) | Results done | Fit threshold rule, sensitivity checks, final plots |
| 7 (Nov 30) | Write-up | Paper and slides |

**Tip:** get the smallest end-to-end version working first (synthetic requests → late binding → one link → metrics), then add pieces one at a time. Each addition should come with a quick test.

---

## 9. Open decisions to settle early

1. **Build our own vs. extend LLMServingSim / Frontier.** The paper says "based on" them. Recommendation: write our own small flow-level simulator in Python (the network and router are the parts we care about and neither tool models them the way we need), and borrow their **timing data** and use them for **validation**.
2. **Exact definition of ρ** — per-request (using that request's path bandwidth) or per-configuration (average)? Section 1 assumes per-request path bandwidth ÷ prefill KV production rate.
3. **NIC bandwidth per instance** — H100 servers usually have one 400 Gbps NIC per GPU, so a 4-GPU instance could use up to 4 NICs. Decide whether a KV transfer uses all of them or one.
4. **Rack size and spine count** — pick a default (e.g. 4 servers per rack) and how many spines.
5. **Prefill batching** — one request at a time (simpler) or batched (more realistic)?
6. **Does the first token wait for anything on the decode side?** Defined here as coming straight from prefill; confirm that matches how the papers we compare against measure TTFT.
