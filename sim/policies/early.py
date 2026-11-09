"""Early binding (Mooncake-style): pick the decode instance on arrival and
stream KV layer by layer while prefill runs. The choice uses whatever the
router knows at arrival time, which may be stale by the time bytes move
(prefill queueing + prefill time + snapshot age)."""
from __future__ import annotations

from .base import Policy


class EarlyBinding(Policy):
    name = "early"

    def on_arrival(self, req, view):
        t_pf = self.sim.perf.prefill_time(req.prompt_tokens)
        dec, bw = self.router.best_decode(view, req, overlap=t_pf)
        self._record(req, bw, t_pf)
        return dec
