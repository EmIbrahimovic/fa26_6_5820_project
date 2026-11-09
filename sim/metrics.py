"""Per-request records and run summaries (plan §3.11)."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .system import Simulation
    from .workload import Request

FIELDS = [
    "id", "arrival", "prompt_tokens", "output_tokens", "prefill_inst", "decode_inst",
    "bind_mode", "bind_time", "path_kind", "rho_est", "est_bw", "kv_bytes",
    "prefill_start", "prefill_end", "prefill_time", "mem_reserved",
    "transfer_start", "transfer_end", "second_token", "completion",
]


def write_requests_csv(reqs: Iterable["Request"], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS + ["ttft", "second_token_latency", "exposed_transfer"])
        for r in reqs:
            w.writerow([getattr(r, f) for f in FIELDS] + [ttft(r), second_latency(r), exposed(r)])


def ttft(r: "Request") -> float:
    return r.prefill_end - r.arrival


def second_latency(r: "Request") -> float:
    return r.second_token - r.arrival


def exposed(r: "Request") -> float:
    """Transfer time the user waited for: KV arrival after the first token."""
    return max(r.transfer_end - r.prefill_end, 0.0)


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stats(prefix: str, xs: list[float]) -> dict:
    xs = [x for x in xs if not math.isnan(x)]
    return {
        f"{prefix}_mean": sum(xs) / len(xs) if xs else math.nan,
        f"{prefix}_p50": percentile(xs, 50),
        f"{prefix}_p99": percentile(xs, 99),
    }


def summarize(sim: "Simulation") -> dict:
    reqs = sim.requests
    done = [r for r in reqs if not math.isnan(r.second_token)]
    out: dict = {
        "policy": sim.policy.name,
        "n_requests": len(reqs),
        "n_second_token": len(done),
        "n_completed": sum(1 for r in reqs if not math.isnan(r.completion)),
        "sim_time": sim.engine.now,
        "arrival_rps": (len(reqs) - 1) / reqs[-1].arrival if len(reqs) > 1 and reqs[-1].arrival > 0 else math.nan,
        "mean_prompt": sum(r.prompt_tokens for r in reqs) / len(reqs),
    }
    out.update(_stats("ttft", [ttft(r) for r in done]))
    out.update(_stats("second", [second_latency(r) for r in done]))
    out.update(_stats("exposed", [exposed(r) for r in done]))
    out.update(_stats("prefill_queue", [r.prefill_start - r.arrival for r in done]))
    out.update(_stats("decode_wait", [r.second_token - max(r.transfer_end, r.prefill_end) for r in done]))
    out.update(_stats("mem_wait", [max(r.mem_reserved - r.bind_time, 0.0) for r in done]))
    rhos = [r.rho_est for r in done if not math.isnan(r.rho_est)]
    out["rho_est_median"] = percentile(rhos, 50)
    out["rho_est_p10"] = percentile(rhos, 10)
    out["frac_early"] = sum(r.bind_mode == "early" for r in done) / max(len(done), 1)
    for k in ("server", "rack", "cross"):
        out[f"frac_{k}"] = sum(r.path_kind == k for r in done) / max(len(done), 1)
    # Achieved (actual) rho: KV bytes / transfer duration vs. production rate.
    ach = []
    for r in done:
        dur = r.transfer_end - max(r.transfer_start, r.prefill_start)
        if dur > 0 and r.prefill_time > 0:
            # bytes per second actually delivered relative to production rate;
            # for streaming this is capped near 1 by construction.
            ach.append((r.kv_bytes / dur) / (r.kv_bytes / r.prefill_time))
    out["rho_achieved_median"] = percentile(ach, 50)
    topo = sim.topo
    if sim.engine.now > 0:
        cap = topo.rack_uplink_total * topo.n_racks * sim.engine.now
        out["bg_carried_load"] = sim.net.bytes_by_kind.get("bg", 0.0) / cap
        out["kv_core_load"] = sim.kv_core_bytes / cap
    out["prefill_util"] = sum(p.busy_time for p in sim.prefill) / (len(sim.prefill) * max(sim.engine.now, 1e-9))
    out["decode_util"] = sum(d.busy_time for d in sim.decode) / (len(sim.decode) * max(sim.engine.now, 1e-9))
    out["events"] = sim.engine.events_processed
    return out
