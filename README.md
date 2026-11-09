# Early vs. late decode binding in disaggregated LLM serving

A flow-level, discrete-event simulator for the question in
[paper/main.tex](paper/main.tex): when should the router bind a request to a
decode instance? Early, on arrival, with layer-wise KV streaming during
prefill; or late, after prefill, using fresher congestion information? The
design follows [SIMULATOR_PLAN.md](SIMULATOR_PLAN.md).

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
data/download.sh                       # traces + SplitwiseSim profiling table
.venv/bin/python experiments/build_profiles.py   # (tables are already committed)
```

## Running

```bash
.venv/bin/python -m pytest                                   # network + math-model checks
.venv/bin/python -m experiments.run --set router.policy=early network.oversubscription=4
.venv/bin/python -m experiments.sweep configs/sweep_oversub_bg.yaml
.venv/bin/python -m analysis.plot_sweep results/oversub_bg/summary.csv \
    --x network.oversubscription --y background.load
.venv/bin/python -m analysis.math_model                      # plan §4 curves
```

`experiments.run` writes `results/<name>/requests.csv` (one row per request:
arrival, prefill start/end, transfer start/end, first/second token, instances,
rho estimate) and `summary.json` (mean/p50/p99 TTFT, second-token latency,
exposed transfer time, utilisations, ...). Sweeps write one
`results/<sweep>/summary.csv` row per grid point.

Every knob is documented in [configs/default.yaml](configs/default.yaml);
override any of them with `--set dotted.key=value` or in a sweep's `grid`.

## Layout (plan §7)

| File | Plan section | What it does |
|---|---|---|
| [sim/engine.py](sim/engine.py) | 3.1 | Event queue and main loop |
| [sim/workload.py](sim/workload.py) | 3.2 | Mooncake / Azure 2023 loaders, synthetic generator, rate scaling |
| [sim/topology.py](sim/topology.py) | 3.3 | Servers, racks, leaf–spine links, ECMP path lookup |
| [sim/network.py](sim/network.py) | 3.4, 3.7 | Flows, max-min water-filling, layer-wise streaming flows |
| [sim/background.py](sim/background.py) | 3.5 | On/off background traffic on the core |
| [sim/profiles.py](sim/profiles.py) | 3.6, 3.8 | Prefill / decode timing tables |
| [sim/instances.py](sim/instances.py) | 3.6, 3.8 | Prefill queues + batching, decode memory + continuous batching |
| [sim/router.py](sim/router.py) | 3.9 | Stale snapshots, bandwidth and decode-cost estimates |
| [sim/policies/](sim/policies/) | 3.10 | `early`, `late`, `threshold` behind one interface |
| [sim/metrics.py](sim/metrics.py) | 3.11 | Per-request CSV, run summaries |
| [sim/system.py](sim/system.py) | 3 | Wires it all together; request lifecycle |
| [analysis/math_model.py](analysis/math_model.py) | 4 | Closed-form model |
| [tests/](tests/) | 5.1, 5.2 | Network unit tests, simulator vs. closed form |

## Decisions taken on the plan's open questions (§9)

1. **Own simulator** in Python; timing data borrowed, not the code.
2. **rho per request**: router's estimated path bandwidth at bind time ÷
   (KV bytes / prefill time). Reported as `rho_est_*`.
3. **NICs**: a KV transfer uses `kv_nics_per_instance` (default 1 × 400G).
   The fabric is sized from `nics_per_server` (default 8, one per GPU);
   oversubscription is relative to that.
4. **Topology**: 4 servers per rack (4 racks), 4 spines.
5. **Prefill batching**: queued prompts batched up to 8192 tokens.
6. **First token** comes straight from prefill; the second needs the KV at
   the decode instance and one decode iteration.

Other modelling choices worth knowing:

* **Timing data**: SplitwiseSim's measured vLLM numbers for Llama-2-70B on
  H100, TP=4 (same KV layout as Llama-3.1-70B). Prefill beyond 8k tokens is
  extrapolated with a quadratic fit; decode adds a memory-bandwidth term for
  context beyond the measured ~640 tokens.
* **Streaming** is a fluid: KV appears at a constant rate during prefill.
  This ignores the last layer's tail (≤1/80 of prefill time).
* **Decode memory** must be reserved before KV moves. Early binding reserves
  it on arrival, so memory is held through prefill queueing and prefill.
* **Background traffic** stays on the leaf–spine core, is rate-limited
  (not elastic), and is spread over all spines. The carried load is reported
  as `bg_carried_load`.
* **Router**: it sees link and decode state through a snapshot refreshed every
  `staleness_s`, and remembers its own bindings until the next refresh.
  Prefill dispatch uses exact queue state, because the router owns it. Early
  and late binding use the same decode-scoring function; only the timing and
  the overlap credit differ.
* Requests whose prompt + output exceed one decode instance's KV memory are
  dropped at load time; output length is clamped to at least 2.

## Status against the plan

Done: §3.1–3.11 (except the optional migration policy), §4, §5.1, §5.2, and
the sweep and plotting tooling for §6.

Not done:

* **Early + migrate** (Llumnix-style) policy, which the plan marks optional.
* **§5.3/5.4**: reproducing DistServe / NetKV / ObjectCache trends and
  cross-checking against LLMServingSim 2.0.
* **Fitting the threshold rule** (§6, week 6). The `threshold` policy exists
  with a fixed `router.threshold` (default 1.0).
