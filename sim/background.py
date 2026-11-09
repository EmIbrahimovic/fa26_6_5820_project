"""Background traffic on the fabric core (plan §3.5).

Each rack has ``sources_per_rack`` on/off sources. While ON, a source sends
rate-limited traffic from its rack's leaf to a random other rack's leaf,
spread evenly over all spines (one sub-flow per spine, as ECMP would spread
many small flows). It competes with KV transfers on leaf uplinks/downlinks
and re-picks its destination every period.

* mean offered load per rack = ``load`` x rack uplink capacity
* ``duty_cycle`` = fraction of time a source is ON. While ON it sends at
  ``load x uplink / sources / duty_cycle``, so the mean is the same but traffic
  arrives in bursts. 1.0 = steady.
* ON/OFF durations are exponential with means ``period_s x duty`` and
  ``period_s x (1 - duty)``.

Background flows are rate-limited (not elastic), so they consume roughly the
configured load and leave the rest to KV transfers; when a burst exceeds a
link's capacity, max-min sharing caps it (the carried load is reported in the
run summary as ``bg_carried_load``).

The traffic stays off the instance NICs: it stands for other jobs and storage
traffic sharing the leaf-spine core, not for the serving instances.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .network import INF, Flow

if TYPE_CHECKING:
    from .system import Simulation


class BackgroundTraffic:
    def __init__(self, sim: "Simulation", cfg: dict, rng: random.Random):
        b = cfg["background"]
        self.sim = sim
        self.rng = rng
        self.load = float(b["load"])
        self.duty = min(max(float(b["duty_cycle"]), 1e-3), 1.0)
        self.period = float(b["period_s"])
        self.spr = int(b["sources_per_rack"])
        topo = sim.topo
        self.demand_on = self.load * topo.rack_uplink_total / self.spr / self.duty
        self.flows: dict[tuple[int, int], list[Flow]] = {}

    def start(self) -> None:
        if self.load <= 0 or self.sim.topo.n_racks < 2:
            return
        eng = self.sim.engine
        for r in range(self.sim.topo.n_racks):
            for k in range(self.spr):
                src = (r, k)
                if self.rng.random() < self.duty:
                    eng.at(0.0, self._on, src, daemon=True)
                else:
                    eng.at(self.rng.expovariate(1 / (self.period * (1 - self.duty))), self._on, src, daemon=True)

    def _on(self, src: tuple[int, int]) -> None:
        topo = self.sim.topo
        r = src[0]
        dst = self.rng.randrange(topo.n_racks - 1)
        if dst >= r:
            dst += 1
        per_spine = self.demand_on / topo.n_spines
        fl = [Flow(topo.core_path(r, dst, k), INF, kind="bg", cap=per_spine) for k in range(topo.n_spines)]
        self.sim.net.add_flows(fl)
        self.flows[src] = fl
        self.sim.engine.after(self.rng.expovariate(1 / (self.period * self.duty)), self._off, src, daemon=True)

    def _off(self, src: tuple[int, int]) -> None:
        self.sim.net.remove_flows(self.flows.pop(src))
        if self.duty >= 1.0:
            self._on(src)
        else:
            self.sim.engine.after(self.rng.expovariate(1 / (self.period * (1 - self.duty))), self._on, src,
                                  daemon=True)
