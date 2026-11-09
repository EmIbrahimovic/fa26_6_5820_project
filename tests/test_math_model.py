"""Check the simulator against the closed-form model of plan §4, one request
at a time, no background traffic, no queueing:

  late : second token = T_pf + KV / B             + decode step
  early: second token = T_pf + max(0, KV / B - T_pf) + decode step

B is set by putting a single prefill and decode instance in different racks
and varying the NIC speed.
"""
import pytest

from analysis.math_model import predict_second_token
from sim import Simulation, load_config, summarize
from sim.workload import Request


def one_request_cfg(policy, nic_gbps, prompt):
    return load_config(overrides={
        "cluster": {"n_servers": 2, "servers_per_rack": 1, "instances_per_server": 1,
                    "pd_ratio": [1, 1], "placement": "by_server"},
        "network": {"nic_gbps": nic_gbps, "oversubscription": 1.0, "n_spines": 1},
        "router": {"policy": policy, "staleness_s": 0.0},
        "workload": {"kind": "synthetic", "max_requests": 1, "prompt_tokens": prompt, "output_tokens": 4},
    })


@pytest.mark.parametrize("policy", ["early", "late"])
@pytest.mark.parametrize("nic_gbps", [5, 20, 40, 100, 400])
@pytest.mark.parametrize("prompt", [1024, 4096, 16000])
def test_single_request_matches_formula(policy, nic_gbps, prompt):
    cfg = one_request_cfg(policy, nic_gbps, prompt)
    req = Request(0, 0.0, prompt, 4)
    sim = Simulation(cfg, requests=[req]).run()
    r = sim.requests[0]
    assert r.path_kind == "cross"
    expected = predict_second_token(sim.perf, prompt, nic_gbps * 1e9 / 8, policy)
    assert r.second_token - r.arrival == pytest.approx(expected, rel=1e-6)


def test_early_hides_transfer_iff_rho_ge_1():
    prompt = 4096
    for nic, hidden in [(400, True), (5, False)]:
        cfg = one_request_cfg("early", nic, prompt)
        sim = Simulation(cfg, requests=[Request(0, 0.0, prompt, 4)]).run()
        r = sim.requests[0]
        exposed = r.transfer_end - r.prefill_end
        rho = (nic * 1e9 / 8) / (r.kv_bytes / r.prefill_time)
        assert (rho >= 1) == hidden
        assert (exposed < 1e-9) == hidden


def test_full_run_completes_all_requests():
    cfg = load_config(overrides={"workload": {"max_requests": 150}, "background": {"load": 0.3, "duty_cycle": 0.3}})
    for policy in ["early", "late", "threshold"]:
        cfg["router"]["policy"] = policy
        m = summarize(Simulation(cfg).run())
        assert m["n_completed"] == m["n_requests"] == 150
        assert m["bg_carried_load"] > 0
