"""Network unit tests (plan §5.1) plus layer-wise streaming behaviour."""
import math

import pytest

from sim.config import load_config
from sim.engine import Engine
from sim.network import INF, Flow, Network
from sim.topology import Link, Topology

GB = 1e9


def mk_links(*caps):
    return [Link(i, f"l{i}", c, "test") for i, c in enumerate(caps)]


def run_flows(links, specs):
    """specs: list of (link indices, size, kwargs). Returns (engine, flows)."""
    eng = Engine()
    net = Network(eng, links)
    flows = []
    for idx, size, kw in specs:
        f = Flow([links[i] for i in idx], size, **kw)
        f.on_done = lambda f: None
        flows.append(f)
        net.add_flow(f)
    return eng, net, flows


def test_single_flow_full_bandwidth():
    links = mk_links(10 * GB)
    eng, net, (f,) = run_flows(links, [([0], 20 * GB, {})])
    assert f.rate == pytest.approx(10 * GB)
    eng.run()
    assert f.end_time == pytest.approx(2.0)


def test_two_flows_share_equally():
    links = mk_links(10 * GB)
    eng, net, (a, b) = run_flows(links, [([0], 10 * GB, {}), ([0], 10 * GB, {})])
    assert a.rate == pytest.approx(5 * GB) and b.rate == pytest.approx(5 * GB)
    eng.run()
    assert a.end_time == pytest.approx(2.0) and b.end_time == pytest.approx(2.0)


def test_max_min_leftover_goes_to_others():
    # Flow A crosses a 2 GB/s link and the shared 10 GB/s link; B and C only
    # the shared link. A is limited to 2, B and C split the remaining 8.
    links = mk_links(10 * GB, 2 * GB)
    eng, net, (a, b, c) = run_flows(links, [([0, 1], INF, {}), ([0], INF, {}), ([0], INF, {})])
    assert a.rate == pytest.approx(2 * GB)
    assert b.rate == pytest.approx(4 * GB)
    assert c.rate == pytest.approx(4 * GB)


def test_rate_cap_is_respected_and_leftover_shared():
    links = mk_links(10 * GB)
    eng, net, (a, b) = run_flows(links, [([0], INF, {"cap": 1 * GB}), ([0], INF, {})])
    assert a.rate == pytest.approx(1 * GB)
    assert b.rate == pytest.approx(9 * GB)


def test_rates_update_when_flow_finishes():
    links = mk_links(10 * GB)
    # small finishes at t=1 (5 GB at 5 GB/s); big then gets 10 GB/s.
    eng, net, (small, big) = run_flows(links, [([0], 5 * GB, {}), ([0], 15 * GB, {})])
    eng.run()
    assert small.end_time == pytest.approx(1.0)
    assert big.end_time == pytest.approx(2.0)  # 5 GB by t=1, 10 GB more at 10 GB/s


def _cross_rack_rate(oversub, n_flows=8):
    cfg = load_config(overrides={"network": {"oversubscription": oversub}})
    topo = Topology(cfg)
    eng = Engine()
    net = Network(eng, topo.links)
    flows = []
    # n_flows senders in rack 0 to receivers in rack 1, plus one same-rack flow.
    for k in range(n_flows):
        a = k                    # rack 0 holds instances 0..7
        b = topo.ips * topo.spr + k  # rack 1
        f = Flow(topo.path(a, b, key=k), INF)
        flows.append(f)
        net.add_flow(f)
    same = Flow(topo.path(topo.n_instances - 2, topo.n_instances - 1 - topo.ips, key=0), INF)
    net.add_flow(same)
    return sum(f.rate for f in flows), same.rate, topo


def test_oversubscription_slows_cross_rack_only():
    cross1, same1, topo = _cross_rack_rate(1.0)
    cross8, same8, _ = _cross_rack_rate(8.0)
    nic = topo.up[0].capacity
    # 1:1 -> every cross-rack flow gets its full NIC.
    assert cross1 == pytest.approx(8 * nic)
    # 8:1 -> rack uplink is 4 x 400G x 8 / 8 = 1.6 Tb/s total; with ECMP
    # collisions the 8 flows get at most that.
    assert cross8 <= topo.rack_uplink_total * (1 + 1e-9)
    assert cross8 < cross1
    assert same1 == pytest.approx(nic) and same8 == pytest.approx(nic)


# --- streaming --------------------------------------------------------------

def test_streaming_flow_cannot_outrun_production():
    # 10 GB produced over [0, 2] (5 GB/s) on a 100 GB/s link: rate capped at
    # production, done exactly when production ends.
    links = mk_links(100 * GB)
    eng, net, (f,) = run_flows(links, [([0], 10 * GB, {"prod_start": 0.0, "prod_end": 2.0})])
    assert f.caught_up and f.rate == pytest.approx(5 * GB)
    eng.run()
    assert f.end_time == pytest.approx(2.0)


def test_streaming_flow_on_slow_link():
    # 10 GB over [0, 2] but the link carries 2 GB/s: done at 5 s, as a plain
    # flow would be.
    links = mk_links(2 * GB)
    eng, net, (f,) = run_flows(links, [([0], 10 * GB, {"prod_start": 0.0, "prod_end": 2.0})])
    eng.run()
    assert f.end_time == pytest.approx(5.0)


def test_streaming_flow_starting_late_catches_up():
    # Production [0, 4] of 8 GB (2 GB/s); flow joins at t=2 with 4 GB backlog,
    # link 6 GB/s: catches up after 4/(6-2)=1 s (t=3), then 2 GB/s until t=4.
    links = mk_links(6 * GB)
    eng = Engine()
    net = Network(eng, links)
    f = Flow([links[0]], 8 * GB, prod_start=0.0, prod_end=4.0)
    eng.at(2.0, net.add_flow, f)
    eng.run()
    assert f.end_time == pytest.approx(4.0)
    assert f.sent == pytest.approx(8 * GB)


def test_streaming_flow_squeezed_then_released():
    # Streaming flow (5 GB/s production over [0,2], 10 GB) shares a 6 GB/s
    # link with a 3 GB bulk flow from t=0: it gets 3 GB/s until the bulk flow
    # ends at t=1, then it can catch up.
    links = mk_links(6 * GB)
    eng, net, (s, bulk) = run_flows(
        links, [([0], 10 * GB, {"prod_start": 0.0, "prod_end": 2.0}), ([0], 3 * GB, {})])
    eng.run()
    assert bulk.end_time == pytest.approx(1.0)
    # By t=1 streaming sent 3 GB, produced 5 GB -> backlog 2 GB; rate 6 vs
    # production 5 -> catches up at t=3 > prod_end, so at t=2 it has sent 9 GB
    # and the remaining 1 GB takes 1/6 s.
    assert s.end_time == pytest.approx(2.0 + 1 / 6)


def test_bytes_conserved():
    links = mk_links(7 * GB, 3 * GB)
    eng, net, flows = run_flows(links, [([0], 4 * GB, {}), ([0, 1], 2 * GB, {}), ([1], 5 * GB, {}),
                                        ([0], 6 * GB, {"prod_start": 0.0, "prod_end": 1.5})])
    eng.run()
    for f in flows:
        assert f.sent == pytest.approx(f.size)
        assert math.isfinite(f.end_time)
    assert net.bytes_by_kind["kv"] == pytest.approx(17 * GB)
