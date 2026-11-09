"""Flow-level network with max-min fair sharing (plan §3.4, §3.7).

Every active transfer is a :class:`Flow` crossing a list of links. Whenever
the set of flows (or a flow's demand cap) changes, rates are recomputed by
water-filling:

1. Among unfrozen flows, find the smallest of (a) the fair share
   ``remaining_capacity / unfrozen_flows`` over all links and (b) the
   smallest per-flow demand cap.
2. Freeze the flows that hit that limit at that rate and subtract their rate
   from every link they cross.
3. Repeat until every flow is frozen.

Layer-wise streaming (early binding) is modelled as a fluid: a streaming flow
has a production window ``[prod_start, prod_end]`` during which its bytes
become available at a constant rate ``size / (prod_end - prod_start)`` (each
of the model's layers takes an equal slice of prefill). A streaming flow can
never send bytes that do not exist yet:

* *backlogged* (sent < produced): behaves as a normal elastic flow;
* *caught up* (sent == produced, still producing): demand capped at the
  production rate;
* after ``prod_end``: a normal flow with whatever is left.

The fluid model ignores the ~1/n_layers tail (the last layer's KV can only be
sent after prefill ends). For 80 layers that is <=1.25% of prefill time when
rho >= 1; see ``tests/test_math_model.py``.

The network keeps a single pending timer for the earliest flow event
(completion, a streaming flow catching up with production, or production
ending) and re-arms it after every change.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

from .engine import Engine, Event
from .topology import Link

INF = math.inf
TIME_TOL = 1e-9  # seconds; events this close together are processed together


class Flow:
    __slots__ = (
        "id", "links", "size", "sent", "rate", "cap", "on_done", "kind", "meta",
        "prod_start", "prod_end", "prod_rate", "caught_up",
        "start_time", "end_time", "next_t", "next_kind",
    )

    def __init__(
        self,
        links: list[Link],
        size: float,
        *,
        kind: str = "kv",
        cap: float = INF,
        on_done: Optional[Callable[["Flow"], None]] = None,
        prod_start: Optional[float] = None,
        prod_end: Optional[float] = None,
        meta=None,
    ):
        if not links:
            raise ValueError("flow needs at least one link")
        self.id = -1
        self.links = links
        self.size = float(size)
        self.sent = 0.0
        self.rate = 0.0
        self.cap = float(cap)
        self.on_done = on_done
        self.kind = kind
        self.meta = meta
        self.prod_start = prod_start
        self.prod_end = prod_end
        if prod_start is not None and prod_end is not None and prod_end > prod_start:
            self.prod_rate = self.size / (prod_end - prod_start)
        else:
            self.prod_start = self.prod_end = None
            self.prod_rate = INF
        self.caught_up = False
        self.start_time = None
        self.end_time = None
        self.next_t = INF
        self.next_kind = None

    def producing(self, t: float) -> bool:
        return self.prod_end is not None and t < self.prod_end - TIME_TOL

    def produced(self, t: float) -> float:
        if self.prod_end is None or t >= self.prod_end:
            return self.size
        if t <= self.prod_start:
            return 0.0
        return self.size * (t - self.prod_start) / (self.prod_end - self.prod_start)

    def effective_cap(self, t: float) -> float:
        if self.caught_up and self.producing(t):
            return min(self.cap, self.prod_rate)
        return self.cap


class Network:
    def __init__(self, engine: Engine, links: Iterable[Link]):
        self.engine = engine
        self.links = list(links)
        self.flows: set[Flow] = set()
        self._next_id = 0
        self._last_t = 0.0
        self._timer: Optional[Event] = None
        self._touched: set[Link] = set()
        self.bytes_by_kind: dict[str, float] = {}
        self.recomputes = 0

    # --- public API --------------------------------------------------------
    def add_flow(self, flow: Flow) -> Flow:
        self.add_flows([flow])
        return flow

    def add_flows(self, flows: list[Flow]) -> None:
        """Add several flows with a single rate recomputation."""
        self._advance()
        now = self.engine.now
        for flow in flows:
            flow.id = self._next_id
            self._next_id += 1
            flow.start_time = now
            # A streaming flow that starts with nothing produced yet is caught up.
            if flow.producing(now) and flow.produced(now) <= flow.sent:
                flow.caught_up = True
            self.flows.add(flow)
            for link in flow.links:
                link.flows.add(flow)
        self._reallocate()

    def remove_flow(self, flow: Flow) -> None:
        self.remove_flows([flow])

    def remove_flows(self, flows: list[Flow]) -> None:
        self._advance()
        for flow in flows:
            if flow in self.flows:
                self._detach(flow)
        self._reallocate()

    def set_cap(self, flow: Flow, cap: float) -> None:
        self._advance()
        flow.cap = cap
        self._reallocate()

    # --- internals ---------------------------------------------------------
    def _detach(self, flow: Flow) -> None:
        self.flows.discard(flow)
        for link in flow.links:
            link.flows.discard(flow)
        flow.end_time = self.engine.now

    def _advance(self) -> None:
        now = self.engine.now
        dt = now - self._last_t
        if dt > 0:
            acc = self.bytes_by_kind
            for f in self.flows:
                if f.rate > 0:
                    sent = f.sent + f.rate * dt
                    lim = f.produced(now) if f.prod_end is not None else f.size
                    if sent > lim:
                        sent = lim
                    acc[f.kind] = acc.get(f.kind, 0.0) + (sent - f.sent)
                    f.sent = sent
        self._last_t = now

    def _reallocate(self) -> None:
        self._waterfill()
        self._mark_squeezed()
        self._arm_timer()

    def _mark_squeezed(self) -> None:
        """A caught-up streaming flow allocated less than its production rate
        starts accumulating backlog again. Its cap was not binding, so
        clearing the flag does not change the allocation."""
        now = self.engine.now
        for f in self.flows:
            if f.caught_up and f.producing(now) and f.rate < f.prod_rate * (1 - 1e-9):
                f.caught_up = False

    def _waterfill(self) -> None:
        self.recomputes += 1
        now = self.engine.now
        for link in self._touched:
            link.used = 0.0
        self._touched = set()
        flows = self.flows
        if not flows:
            return
        rem: dict[Link, float] = {}
        cnt: dict[Link, int] = {}
        capped = []
        for f in flows:
            f.rate = -1.0  # sentinel: unfrozen
            c = f.effective_cap(now)
            if c < INF:
                capped.append((c, f.id, f))
            for link in f.links:
                if link in cnt:
                    cnt[link] += 1
                else:
                    cnt[link] = 1
                    rem[link] = link.capacity
        capped.sort()
        ci = 0
        unfrozen = len(flows)

        def freeze(f: Flow, r: float) -> None:
            f.rate = r
            for link in f.links:
                rem[link] -= r
                cnt[link] -= 1

        while unfrozen:
            share = INF
            bottleneck = None
            for link, c in cnt.items():
                if c > 0:
                    s = rem[link] / c
                    if s < share:
                        share = s
                        bottleneck = link
            if share < 0:
                share = 0.0
            while ci < len(capped) and capped[ci][2].rate >= 0:
                ci += 1
            if ci < len(capped) and capped[ci][0] <= share:
                # Freezing a flow at a cap <= the minimum fair share can only
                # raise every link's share, so all such flows go in one pass.
                while ci < len(capped) and capped[ci][0] <= share:
                    f = capped[ci][2]
                    if f.rate < 0:
                        freeze(f, capped[ci][0])
                        unfrozen -= 1
                    ci += 1
                continue
            for f in bottleneck.flows:
                if f.rate < 0:
                    freeze(f, share)
                    unfrozen -= 1

        for link in rem:
            link.used = link.capacity - rem[link]
            self._touched.add(link)

    def _arm_timer(self) -> None:
        now = self.engine.now
        best = INF
        for f in self.flows:
            t, kind = INF, None
            if f.producing(now):
                remaining_prod = f.prod_end
                if f.caught_up:
                    t, kind = remaining_prod, "prod_end"
                else:
                    backlog = f.produced(now) - f.sent
                    if f.rate > f.prod_rate:
                        t_catch = now + backlog / (f.rate - f.prod_rate)
                        if t_catch < remaining_prod:
                            t, kind = t_catch, "catch"
                        else:
                            t, kind = remaining_prod, "prod_end"
                    else:
                        t, kind = remaining_prod, "prod_end"
            elif f.size < INF and f.rate > 0:
                t, kind = now + (f.size - f.sent) / f.rate, "done"
            f.next_t, f.next_kind = t, kind
            if t < best:
                best = t
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if best < INF:
            self._timer = self.engine.at(best, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        self._advance()
        now = self.engine.now
        done: list[Flow] = []
        for f in self.flows:
            if f.next_t > now + TIME_TOL:
                continue
            if f.next_kind == "done":
                f.sent = f.size
                done.append(f)
            elif f.next_kind == "catch":
                f.sent = f.produced(now)
                f.caught_up = True
            elif f.next_kind == "prod_end":
                f.caught_up = False
                if f.prod_end is not None:
                    # From now on it is an ordinary flow.
                    if f.size - f.sent <= max(1.0, 1e-9 * f.size):
                        f.sent = f.size
                        done.append(f)
                    f.prod_start = f.prod_end = None
                    f.prod_rate = INF
        for f in done:
            self._detach(f)
        self._reallocate()
        for f in done:
            if f.on_done is not None:
                f.on_done(f)
