"""Cluster and leaf-spine topology (plan §3.3).

Layout: ``n_servers`` servers, ``instances_per_server`` model instances per
server (TP=4 on 8 GPUs -> 2), ``servers_per_rack`` servers per leaf switch,
``n_spines`` spine switches.

Links (all unidirectional, capacities in bytes/s):

* ``inst{i}.up`` / ``inst{i}.down``: the NIC(s) a KV transfer may use for
  instance ``i`` (``kv_nics_per_instance`` x ``nic_gbps``).
* ``srv{s}.nvlink``: intra-server transfers (effectively free).
* ``leaf{r}->spine{k}`` / ``spine{k}->leaf{r}``: leaf uplinks/downlinks.

Oversubscription is defined against the *full* server NIC capacity
(``nics_per_server`` x ``nic_gbps`` per server), the usual definition:
rack uplink total = servers_per_rack * nics_per_server * nic_gbps / oversub,
split evenly across the spines.

Routing: same server -> NVLink; same rack -> [src NIC up, dst NIC down];
cross rack -> [src up, leaf->spine, spine->leaf, dst down], spine chosen by
an ECMP-style hash of (src, dst, flow key).
"""
from __future__ import annotations

from dataclasses import dataclass, field

GBPS = 1e9 / 8  # bytes/s per Gbit/s


@dataclass(eq=False)
class Link:
    idx: int
    name: str
    capacity: float  # bytes/s
    kind: str  # "nic_up", "nic_down", "nvlink", "uplink", "downlink"
    flows: set = field(default_factory=set, repr=False)
    used: float = 0.0  # current allocated rate, bytes/s

    def __hash__(self) -> int:  # identity hash, cheap
        return self.idx


class Topology:
    def __init__(self, cfg: dict):
        c = cfg["cluster"]
        n = cfg["network"]
        self.n_servers = int(c["n_servers"])
        self.ips = int(c["instances_per_server"])
        self.spr = int(c["servers_per_rack"])
        self.n_spines = int(n["n_spines"])
        if self.n_servers % self.spr:
            raise ValueError("n_servers must be a multiple of servers_per_rack")
        self.n_racks = self.n_servers // self.spr
        self.n_instances = self.n_servers * self.ips

        nic_bw = float(n["nic_gbps"]) * GBPS
        inst_bw = nic_bw * float(n["kv_nics_per_instance"])
        server_bw = nic_bw * float(n["nics_per_server"])
        self.oversub = float(n["oversubscription"])
        self.rack_uplink_total = self.spr * server_bw / self.oversub
        per_spine = self.rack_uplink_total / self.n_spines
        nvlink_bw = float(n["nvlink_GBps"]) * 1e9

        self.links: list[Link] = []

        def mk(name: str, cap: float, kind: str) -> Link:
            link = Link(len(self.links), name, cap, kind)
            self.links.append(link)
            return link

        self.up = [mk(f"inst{i}.up", inst_bw, "nic_up") for i in range(self.n_instances)]
        self.down = [mk(f"inst{i}.down", inst_bw, "nic_down") for i in range(self.n_instances)]
        self.nvlink = [mk(f"srv{s}.nvlink", nvlink_bw, "nvlink") for s in range(self.n_servers)]
        self.l2s = [[mk(f"leaf{r}->spine{k}", per_spine, "uplink") for k in range(self.n_spines)]
                    for r in range(self.n_racks)]
        self.s2l = [[mk(f"spine{k}->leaf{r}", per_spine, "downlink") for k in range(self.n_spines)]
                    for r in range(self.n_racks)]

    # --- location helpers -------------------------------------------------
    def server_of(self, inst: int) -> int:
        return inst // self.ips

    def rack_of(self, inst: int) -> int:
        return self.server_of(inst) // self.spr

    def path_kind(self, a: int, b: int) -> str:
        if self.server_of(a) == self.server_of(b):
            return "server"
        if self.rack_of(a) == self.rack_of(b):
            return "rack"
        return "cross"

    # --- routing -----------------------------------------------------------
    def ecmp_spine(self, a: int, b: int, key: int) -> int:
        h = (a * 1_000_003 + b * 7_919 + key * 104_729) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0x5BD1E995) & 0xFFFFFFFF
        h ^= h >> 15
        return h % self.n_spines

    def path(self, a: int, b: int, key: int = 0) -> list[Link]:
        kind = self.path_kind(a, b)
        if kind == "server":
            return [self.nvlink[self.server_of(a)]]
        if kind == "rack":
            return [self.up[a], self.down[b]]
        ra, rb = self.rack_of(a), self.rack_of(b)
        s = self.ecmp_spine(a, b, key)
        return [self.up[a], self.l2s[ra][s], self.s2l[rb][s], self.down[b]]

    def core_path(self, ra: int, rb: int, spine: int) -> list[Link]:
        return [self.l2s[ra][spine], self.s2l[rb][spine]]
