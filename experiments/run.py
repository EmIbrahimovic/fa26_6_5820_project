"""Run one simulation.

  python -m experiments.run [--config configs/x.yaml] [--set key=value ...] [--name NAME]

Writes results/<name>/requests.csv (one row per request) and summary.json,
and prints the summary.
"""
from __future__ import annotations

import argparse
import json
import time

from sim import Simulation, load_config, summarize, write_requests_csv
from sim.config import ROOT, parse_overrides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    ap.add_argument("--name", default=None, help="output subdirectory (default: policy name)")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    t0 = time.time()
    sim = Simulation(cfg).run()
    summary = summarize(sim)
    summary["wall_s"] = time.time() - t0

    out = ROOT / cfg["output"]["dir"] / (args.name or cfg["router"]["policy"])
    write_requests_csv(sim.requests, out / "requests.csv")
    with open(out / "summary.json", "w") as fh:
        json.dump({"config": cfg, "summary": summary}, fh, indent=2)
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k:<{width}}  {v:.4g}" if isinstance(v, float) else f"{k:<{width}}  {v}")
    print("wrote", out)


if __name__ == "__main__":
    main()
