"""Closed-form latency model (plan §4) and a plot of where early binding
stops paying off.

  late : second token ~= T_pf + KV/B + decode_step
  early: second token ~= T_pf + max(0, KV/B - T_pf) + decode_step
  rho   = B / (KV / T_pf)

Usage: python -m analysis.math_model   (writes results/math_model.png)
"""
from __future__ import annotations

from pathlib import Path

from sim.profiles import PerfModel


def predict_second_token(perf: PerfModel, prompt: int, bw: float, policy: str) -> float:
    t_pf = perf.prefill_time(prompt)
    xfer = perf.kv_bytes(prompt) / bw
    step = perf.decode_iter_time(1, prompt + 1)
    if policy == "late":
        return t_pf + xfer + step
    return t_pf + max(0.0, xfer - t_pf) + step


def rho(perf: PerfModel, prompt: int, bw: float) -> float:
    return bw / (perf.kv_bytes(prompt) / perf.prefill_time(prompt))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sim.config import ROOT, load_config

    cfg = load_config()
    perf = PerfModel(cfg, ROOT)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    prompts = [1024, 4096, 16384, 65536]
    gbps = [2 ** k for k in range(0, 10)]  # 1 .. 512 Gb/s
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in prompts:
        bw = [g * 1e9 / 8 for g in gbps]
        gain = [predict_second_token(perf, p, b, "late") - predict_second_token(perf, p, b, "early") for b in bw]
        axes[0].plot(gbps, [g * 1000 for g in gain], marker="o", label=f"{p} tok")
        axes[1].plot(gbps, [rho(perf, p, b) for b in bw], marker="o", label=f"{p} tok")
    axes[0].set(xscale="log", yscale="log", xlabel="KV path bandwidth (Gb/s)",
                ylabel="late - early, 2nd token (ms)", title="Overlap benefit of early binding")
    axes[1].axhline(1, color="grey", ls="--")
    axes[1].set(xscale="log", yscale="log", xlabel="KV path bandwidth (Gb/s)", ylabel="rho",
                title="rho = B / KV production rate")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "math_model.png", dpi=130)
    print("KV production rate (Gb/s) by prompt length:")
    for p in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        r = perf.kv_bytes(p) / perf.prefill_time(p)
        print(f"  {p:>6} tok: T_pf={perf.prefill_time(p):7.3f}s  KV={perf.kv_bytes(p) / 1e9:6.2f} GB  "
              f"rate={r * 8 / 1e9:6.1f} Gb/s")
    print("wrote", out / "math_model.png")


if __name__ == "__main__":
    main()
