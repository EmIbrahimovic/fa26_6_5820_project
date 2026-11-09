"""Threshold / hybrid binding: on arrival, estimate rho for this request on
its best early candidate,

    rho_hat = estimated path bandwidth / (KV bytes / estimated prefill time)

and bind early iff rho_hat >= ``router.threshold``; otherwise wait and bind
late. The threshold is meant to be fitted from sweep results (plan §6)."""
from __future__ import annotations

from .base import Policy


class ThresholdBinding(Policy):
    name = "threshold"

    def __init__(self, sim, cfg):
        super().__init__(sim, cfg)
        self.theta = float(cfg["router"]["threshold"])

    def on_arrival(self, req, view):
        perf = self.sim.perf
        t_pf = perf.prefill_time(req.prompt_tokens)
        dec, bw = self.router.best_decode(view, req, overlap=t_pf)
        rho = bw / (perf.kv_bytes(req.prompt_tokens) / t_pf)
        if rho >= self.theta:
            self._record(req, bw, t_pf)
            return dec
        return None
