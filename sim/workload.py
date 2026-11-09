"""Requests and trace loading (plan §3.2).

Supported sources (``workload.kind``; for traces the loader is picked from
the file extension, so ``mooncake``/``azure``/``trace`` are interchangeable):

* ``mooncake``: FAST'25 Mooncake traces (JSONL, ``timestamp`` in ms,
  ``input_length``, ``output_length``).
* ``azure``: Azure LLM inference trace 2023 (CSV ``TIMESTAMP,ContextTokens,
  GeneratedTokens``), code or conversation.
* ``synthetic``: Poisson arrivals with fixed or log-normal prompt lengths.

Transforms: skip the first ``skip`` requests, keep ``max_requests``, filter by
prompt length, then rescale time either by ``rate_scale`` (2.0 = twice the
arrival rate) or to an absolute ``target_rps``.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(eq=False)
class Request:
    id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int  # total output tokens, including the one from prefill

    # Filled in by the simulation.
    prefill_inst: int = -1
    decode_inst: int = -1
    bind_mode: str = ""          # "early" or "late"
    bind_time: float = math.nan
    path_kind: str = ""          # "server", "rack", "cross"
    rho_est: float = math.nan    # router's estimate of rho at bind time
    est_bw: float = math.nan     # router's estimated path bandwidth, bytes/s
    prefill_start: float = math.nan
    prefill_end: float = math.nan  # = first token time
    prefill_time: float = math.nan
    mem_reserved: float = math.nan
    transfer_start: float = math.nan
    transfer_end: float = math.nan
    kv_bytes: float = 0.0
    second_token: float = math.nan
    completion: float = math.nan
    decoded: int = 0

    # Internal state.
    reserved: bool = field(default=False, repr=False)
    flow: object = field(default=None, repr=False)

    @property
    def mem_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def _load_mooncake(path: Path) -> list[tuple[float, int, int]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append((d["timestamp"] / 1000.0, int(d["input_length"]), int(d["output_length"])))
    return rows


def _load_azure(path: Path) -> list[tuple[float, int, int]]:
    rows = []
    t0 = None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ts = r["TIMESTAMP"]
            # Fractional seconds have 7 digits; datetime accepts 6.
            if "." in ts:
                head, frac = ts.split(".")
                ts = f"{head}.{frac[:6]}"
            t = datetime.fromisoformat(ts).timestamp()
            if t0 is None:
                t0 = t
            rows.append((t - t0, int(r["ContextTokens"]), int(r["GeneratedTokens"])))
    return rows


def _synthetic(w: dict, rng: random.Random) -> list[tuple[float, int, int]]:
    n = int(w.get("max_requests") or 1000)
    rate = float(w["rate"])
    t = 0.0
    rows = []
    for _ in range(n):
        t += rng.expovariate(rate)
        if w.get("prompt_sigma", 0):
            mu = math.log(w["prompt_tokens"]) - w["prompt_sigma"] ** 2 / 2
            p = int(rng.lognormvariate(mu, w["prompt_sigma"]))
        else:
            p = int(w["prompt_tokens"])
        rows.append((t, max(p, 1), int(w["output_tokens"])))
    return rows


def load_requests(cfg: dict, root: Path, max_tokens_per_request: Optional[int] = None) -> list[Request]:
    w = cfg["workload"]
    rng = random.Random(cfg.get("seed", 0))
    kind = w["kind"]
    if kind == "synthetic":
        rows = _synthetic(w, rng)
    elif kind in ("mooncake", "azure", "trace"):
        # The file format decides the loader, so sweeps can vary workload.path.
        path = root / w["path"]
        rows = _load_mooncake(path) if path.suffix == ".jsonl" else _load_azure(path)
    else:
        raise ValueError(f"unknown workload kind {kind!r}")

    rows.sort(key=lambda r: r[0])
    rows = rows[int(w.get("skip", 0)):]
    lo, hi = w.get("min_prompt"), w.get("max_prompt")
    if lo is not None:
        rows = [r for r in rows if r[1] >= lo]
    if hi is not None:
        rows = [r for r in rows if r[1] <= hi]
    if max_tokens_per_request is not None:
        rows = [r for r in rows if r[1] + max(r[2], 2) <= max_tokens_per_request]
    if kind != "synthetic" and w.get("max_requests"):
        rows = rows[: int(w["max_requests"])]
    if not rows:
        raise ValueError("workload is empty after filtering")

    t0 = rows[0][0]
    span = rows[-1][0] - t0
    scale = 1.0
    if w.get("target_rps"):
        native = (len(rows) - 1) / span if span > 0 else float(w["target_rps"])
        scale = native / float(w["target_rps"])
    elif w.get("rate_scale"):
        scale = 1.0 / float(w["rate_scale"])
    # Every request needs >=2 output tokens so that a second token exists.
    return [Request(i, (t - t0) * scale, p, max(o, 2)) for i, (t, p, o) in enumerate(rows)]
