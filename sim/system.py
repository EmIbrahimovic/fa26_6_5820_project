"""Wires everything together: workload -> router -> prefill -> network ->
decode -> metrics (plan §3, diagram in the paper).

Request lifecycle
  1. arrival: policy picks a prefill instance; it may also bind a decode
     instance now (early). The request joins the prefill queue.
  2. bind (early on arrival, late after prefill): the decode instance must
     grant KV memory before any bytes move.
  3. transfer starts once memory is granted *and* prefill has started.
     Early + prefill still running -> streaming flow (bytes appear as layers
     finish). Otherwise -> ordinary flow of the remaining KV.
  4. prefill end = first token. KV fully arrived -> request joins the decode
     batch; the first decode iteration it is part of yields the second token.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from .background import BackgroundTraffic
from .config import ROOT
from .engine import Engine
from .instances import DecodeInstance, PrefillInstance
from .network import INF, Flow, Network
from .policies import make_policy
from .profiles import PerfModel
from .router import Router
from .topology import Topology
from .workload import Request, load_requests


def assign_roles(topo: Topology, pd_ratio, placement: str) -> list[str]:
    p, d = pd_ratio
    n = topo.n_instances
    n_prefill = max(1, min(n - 1, round(n * p / (p + d))))
    if placement == "by_server":
        order = list(range(n))
    elif placement == "instance_mixed":
        order = sorted(range(n), key=lambda i: (i % topo.ips, i // topo.ips))
    elif placement == "rack_mixed":
        def key(i):
            s = topo.server_of(i)
            return (s % topo.spr, s // topo.spr, i % topo.ips)
        order = sorted(range(n), key=key)
    else:
        raise ValueError(f"unknown placement {placement!r}")
    roles = ["decode"] * n
    for i in order[:n_prefill]:
        roles[i] = "prefill"
    return roles


class Simulation:
    def __init__(self, cfg: dict, root: Path = ROOT, requests: list[Request] | None = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 0))
        self.engine = Engine()
        self.topo = Topology(cfg)
        self.net = Network(self.engine, self.topo.links)
        self.perf = PerfModel(cfg, root)

        inst_cfg = cfg["instances"]
        cap_tokens = int(float(inst_cfg["decode_kv_capacity_gb"]) * 1e9 / self.perf.kv_bytes_per_token)
        self.roles = assign_roles(self.topo, cfg["cluster"]["pd_ratio"], cfg["cluster"]["placement"])
        self.prefill: list[PrefillInstance] = []
        self.decode: list[DecodeInstance] = []
        for i, role in enumerate(self.roles):
            if role == "prefill":
                self.prefill.append(PrefillInstance(len(self.prefill), i, self,
                                                    int(inst_cfg["prefill_max_batch_tokens"])))
            else:
                self.decode.append(DecodeInstance(len(self.decode), i, self, cap_tokens,
                                                  int(inst_cfg["decode_max_batch"])))
        self.decode_by_inst = {d.inst: d for d in self.decode}

        cap = cfg["network"].get("kv_flow_max_gbps")
        self.kv_flow_cap = float(cap) * 1e9 / 8 if cap else INF
        self.kv_core_bytes = 0.0

        self.router = Router(self, cfg)
        self.policy = make_policy(cfg["router"]["policy"], self, cfg)
        self.background = BackgroundTraffic(self, cfg, random.Random(self.rng.random()))
        self.requests = requests if requests is not None else load_requests(cfg, root, cap_tokens)
        self._next_arrival = 0

    # --- driver ------------------------------------------------------------
    def run(self, until: float | None = None) -> "Simulation":
        self.router.start()
        self.background.start()
        if self.requests:
            self.engine.at(self.requests[0].arrival, self._arrive)
        self.engine.run(until)
        return self

    def _arrive(self) -> None:
        req = self.requests[self._next_arrival]
        self._next_arrival += 1
        if self._next_arrival < len(self.requests):
            self.engine.at(self.requests[self._next_arrival].arrival, self._arrive)
        req.kv_bytes = self.perf.kv_bytes(req.prompt_tokens)
        p = self.policy.choose_prefill(req)
        req.prefill_inst = p.inst
        view = self.router.view()
        dec = self.policy.on_arrival(req, view)
        if dec is not None:
            self._bind(req, dec, "early", view)
        p.enqueue(req)

    # --- callbacks from instances -----------------------------------------
    def on_prefill_start(self, req: Request) -> None:
        if req.reserved and req.flow is None:
            self._start_transfer(req)

    def on_prefill_done(self, req: Request) -> None:
        if req.decode_inst < 0:
            view = self.router.view()
            dec = self.policy.on_prefill_done(req, view)
            self._bind(req, dec, "late", view)

    def on_memory_reserved(self, req: Request) -> None:
        req.reserved = True
        req.mem_reserved = self.engine.now
        if not math.isnan(req.prefill_start) and req.flow is None:
            self._start_transfer(req)

    # --- binding and transfer ---------------------------------------------
    def _bind(self, req: Request, dec: DecodeInstance, mode: str, view) -> None:
        req.decode_inst = dec.inst
        req.bind_mode = mode
        req.bind_time = self.engine.now
        req.path_kind = self.topo.path_kind(req.prefill_inst, dec.inst)
        if mode == "early":
            rate = req.kv_bytes / self.perf.prefill_time(req.prompt_tokens)
        else:
            rate = req.est_bw
        self.router.note_bind(view, req, dec, min(rate, req.est_bw))
        if dec.request_memory(req):
            self.on_memory_reserved(req)

    def _start_transfer(self, req: Request) -> None:
        now = self.engine.now
        links = self.topo.path(req.prefill_inst, req.decode_inst, key=req.id)
        streaming = req.bind_mode == "early" and now < req.prefill_end
        flow = Flow(
            links, req.kv_bytes, kind="kv", cap=self.kv_flow_cap, on_done=self._transfer_done, meta=req,
            prod_start=req.prefill_start if streaming else None,
            prod_end=req.prefill_end if streaming else None,
        )
        req.flow = flow
        req.transfer_start = now
        if req.path_kind == "cross":
            self.kv_core_bytes += req.kv_bytes
        self.net.add_flow(flow)

    def _transfer_done(self, flow: Flow) -> None:
        req: Request = flow.meta
        req.transfer_end = self.engine.now
        req.flow = None
        self.decode_by_inst[req.decode_inst].kv_arrived(req)
