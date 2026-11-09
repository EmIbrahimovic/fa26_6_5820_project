"""Router state and the stale view (plan §3.9).

The router never reads the true network state directly. It reads a
:class:`View`: a snapshot of link utilisation and decode-instance load taken
every ``staleness_s`` seconds (0 = a fresh snapshot for every decision).

Between snapshots the router does remember its *own* decisions (it made
them): bindings since the last snapshot are added on top of the snapshot so
it does not send every request in a window to the same decode instance.

Bandwidth estimate for a candidate path, per link: a new flow gets at least
its max-min fair share ``cap / (n + 1)`` and at most the idle capacity
``cap - used``; we take the larger. The router does not know which spine
ECMP will pick, so for the core it averages the estimate over spines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .network import INF

if TYPE_CHECKING:
    from .instances import DecodeInstance
    from .system import Simulation
    from .workload import Request


class View:
    __slots__ = ("time", "used", "nflows", "dec_reserved", "dec_waiting", "dec_batch", "dec_ctx",
                 "local_flows", "local_used", "local_mem", "local_batch")

    def __init__(self, sim: "Simulation"):
        links = sim.topo.links
        self.time = sim.engine.now
        self.used = [l.used for l in links]
        self.nflows = [len(l.flows) for l in links]
        decs = sim.decode
        self.dec_reserved = [d.reserved for d in decs]
        self.dec_waiting = [d.waiting_tokens for d in decs]
        self.dec_batch = [len(d.batch) + len(d.ready) + d.inbound + len(d.waiting) for d in decs]
        self.dec_ctx = [d.batch_ctx for d in decs]
        self.local_flows = [0] * len(links)
        self.local_used = [0.0] * len(links)
        self.local_mem = [0] * len(decs)
        self.local_batch = [0] * len(decs)


class Router:
    def __init__(self, sim: "Simulation", cfg: dict):
        r = cfg["router"]
        self.sim = sim
        self.staleness = float(r["staleness_s"])
        self.mem_penalty = float(r["mem_wait_penalty_s"])
        self.select = r["decode_select"]
        cap = cfg["network"].get("kv_flow_max_gbps")
        self.flow_cap = float(cap) * 1e9 / 8 if cap else INF
        self.snap: View | None = None

    def start(self) -> None:
        if self.staleness > 0:
            self._refresh()

    def _refresh(self) -> None:
        self.snap = View(self.sim)
        self.sim.engine.after(self.staleness, self._refresh, daemon=True)

    def view(self) -> View:
        if self.staleness <= 0 or self.snap is None:
            return View(self.sim)
        return self.snap

    def note_bind(self, view: View, req: "Request", dec: "DecodeInstance", rate: float) -> None:
        """Remember our own decision until the next snapshot. ``rate`` is the
        rate we expect the new transfer to use (production rate for a
        streaming transfer, the estimated path bandwidth otherwise)."""
        if view is not self.snap:
            return
        view.local_mem[dec.idx] += req.mem_tokens
        view.local_batch[dec.idx] += 1
        links = self._est_links(req.prefill_inst, dec.inst)
        core = self.sim.topo.n_spines if len(links) > 2 else 1
        for i, link in enumerate(links):
            view.local_flows[link.idx] += 1
            # Core links: ECMP spreads the flow over spines on average.
            view.local_used[link.idx] += rate / core if i >= 2 else rate

    # --- estimates ---------------------------------------------------------
    def _est_links(self, a: int, b: int):
        topo = self.sim.topo
        kind = topo.path_kind(a, b)
        if kind == "server":
            return [topo.nvlink[topo.server_of(a)]]
        if kind == "rack":
            return [topo.up[a], topo.down[b]]
        ra, rb = topo.rack_of(a), topo.rack_of(b)
        return [topo.up[a], topo.down[b]] + topo.l2s[ra] + topo.s2l[rb]

    @staticmethod
    def _link_est(view: View, link) -> float:
        i = link.idx
        cap = link.capacity
        n = view.nflows[i] + view.local_flows[i]
        fair = cap / (n + 1)
        idle = cap - view.used[i] - view.local_used[i]
        return max(idle, fair)

    def path_bw(self, view: View, a: int, b: int) -> float:
        topo = self.sim.topo
        kind = topo.path_kind(a, b)
        est = self._link_est
        if kind == "server":
            bw = est(view, topo.nvlink[topo.server_of(a)])
        elif kind == "rack":
            bw = min(est(view, topo.up[a]), est(view, topo.down[b]))
        else:
            ra, rb = topo.rack_of(a), topo.rack_of(b)
            core = sum(min(est(view, u), est(view, d)) for u, d in zip(topo.l2s[ra], topo.s2l[rb]))
            core /= topo.n_spines
            bw = min(est(view, topo.up[a]), est(view, topo.down[b]), core)
        return min(bw, self.flow_cap)

    def decode_cost(self, view: View, dec: "DecodeInstance", req: "Request") -> float:
        i = dec.idx
        perf = self.sim.perf
        batch = view.dec_batch[i] + view.local_batch[i] + 1
        ctx = view.dec_ctx[i] + req.prompt_tokens
        cost = perf.decode_iter_time(batch, ctx)
        free = dec.capacity - view.dec_reserved[i] - view.dec_waiting[i] - view.local_mem[i]
        if free < req.mem_tokens:
            cost += self.mem_penalty
        return cost

    def best_decode(self, view: View, req: "Request", overlap: float) -> tuple["DecodeInstance", float]:
        """Pick the decode instance minimising (exposed transfer + decode cost).

        ``overlap`` is how much of the transfer can hide behind prefill
        (prefill duration for early binding, 0 for late binding).
        Returns (instance, estimated path bandwidth).
        """
        kv = self.sim.perf.kv_bytes(req.prompt_tokens)
        best, best_cost, best_bw = None, INF, 0.0
        for dec in self.sim.decode:
            bw = self.path_bw(view, req.prefill_inst, dec.inst)
            cost = self.decode_cost(view, dec, req)
            if self.select == "network":
                cost += max(kv / bw - overlap, 0.0)
            # Deterministic tie-break that spreads load: prefer lower index
            # offset by request id.
            cost += 1e-9 * ((dec.idx - req.id) % len(self.sim.decode))
            if cost < best_cost:
                best, best_cost, best_bw = dec, cost, bw
        return best, best_bw
