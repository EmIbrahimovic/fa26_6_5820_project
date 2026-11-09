from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..instances import DecodeInstance, PrefillInstance
    from ..router import View
    from ..system import Simulation
    from ..workload import Request


class Policy:
    name = "base"

    def __init__(self, sim: "Simulation", cfg: dict):
        self.sim = sim
        self.router = sim.router
        self.cfg = cfg

    def choose_prefill(self, req: "Request") -> "PrefillInstance":
        # The router dispatches every prefill itself, so it knows prefill
        # queues exactly; staleness only applies to network and decode state.
        now = self.sim.engine.now
        n = len(self.sim.prefill)
        return min(self.sim.prefill, key=lambda p: (p.est_wait(now), (p.idx - req.id) % n))

    def on_arrival(self, req: "Request", view: "View") -> Optional["DecodeInstance"]:
        return None

    def on_prefill_done(self, req: "Request", view: "View") -> "DecodeInstance":
        dec, bw = self.router.best_decode(view, req, overlap=0.0)
        self._record(req, bw, req.prefill_time)
        return dec

    def _record(self, req: "Request", bw: float, prefill_time: float) -> None:
        kv = self.sim.perf.kv_bytes(req.prompt_tokens)
        req.est_bw = bw
        req.rho_est = bw / (kv / prefill_time) if prefill_time > 0 else float("inf")
