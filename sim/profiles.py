"""Prefill/decode timing from profiling tables (plan §3.6, §3.8).

We do not simulate GPU math. Timings come from lookup tables built by
``experiments/build_profiles.py`` from published measurements.

Prefill
  ``prefill_time(n)``: piecewise-linear interpolation of the table; beyond the
  largest measured prompt we extrapolate with a least-squares fit
  ``a + b*n + c*n^2`` (attention makes prefill super-linear).
  A batch of prompts runs as one forward pass over ``sum(n)`` tokens, minus
  the cross-request attention term that a single long prompt would have paid:
  ``T(batch) = prefill_time(sum n) - c * ((sum n)^2 - sum n^2)``.

Decode
  One iteration produces one token for every request in the batch.
  ``decode_iter_time(batch, ctx_tokens) = table(batch) + extra KV reads``,
  where the table was measured at ~``decode_ref_ctx`` tokens of context per
  request and the extra term charges the remaining KV bytes against effective
  HBM bandwidth (decode is memory-bound).
"""
from __future__ import annotations

import bisect
import csv
from pathlib import Path


def _read_table(path: Path) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Linear interpolation; linear extrapolation from the end segments."""
    if x <= xs[0]:
        if len(xs) == 1:
            return ys[0]
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return max(ys[0] + slope * (x - xs[0]), 0.0)
    if x >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _fit_quadratic(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares y = a + b x + c x^2 (normal equations, 3x3)."""
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    m = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return tuple(m[i][3] / m[i][i] for i in range(3))  # type: ignore[return-value]


class PerfModel:
    def __init__(self, cfg: dict, root: Path):
        p = cfg["profile"]
        self.px, self.py = _read_table(root / p["prefill_csv"])
        self.dx, self.dy = _read_table(root / p["decode_csv"])
        self.kv_bytes_per_token = float(cfg["model"]["kv_bytes_per_token"])
        self.decode_ref_ctx = float(p.get("decode_ref_ctx", 640))
        self.hbm_bw = float(p["decode_eff_hbm_GBps"]) * 1e9
        self.a, self.b, self.c = _fit_quadratic(self.px, self.py)
        # Only use a non-negative quadratic term for batch correction.
        self.c = max(self.c, 0.0)

    def prefill_time(self, n: float) -> float:
        if n <= self.px[-1]:
            return _interp(self.px, self.py, n)
        # Continuous extrapolation: shift the quadratic fit to meet the table.
        fit = lambda x: self.a + self.b * x + self.c * x * x  # noqa: E731
        return self.py[-1] + fit(n) - fit(self.px[-1])

    def prefill_time_batch(self, lens: list[int]) -> float:
        if len(lens) == 1:
            return self.prefill_time(lens[0])
        tot = sum(lens)
        cross = tot * tot - sum(n * n for n in lens)
        return max(self.prefill_time(tot) - self.c * cross, max(self.prefill_time(n) for n in lens))

    def decode_iter_time(self, batch: int, ctx_tokens: float) -> float:
        if batch <= 0:
            return 0.0
        base = _interp(self.dx, self.dy, batch)
        extra_tokens = max(ctx_tokens - batch * self.decode_ref_ctx, 0.0)
        return base + extra_tokens * self.kv_bytes_per_token / self.hbm_bw

    def kv_bytes(self, prompt_tokens: int) -> float:
        return prompt_tokens * self.kv_bytes_per_token
