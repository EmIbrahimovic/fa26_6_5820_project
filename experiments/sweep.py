"""Run a grid of simulations in parallel (plan §6).

A sweep file looks like::

    name: oversub_bg
    base:                 # overrides applied to configs/default.yaml
      workload: {max_requests: 1000}
    grid:                 # dotted key -> list of values (cartesian product)
      network.oversubscription: [1, 2, 4, 8]
      background.load: [0.0, 0.3, 0.6]
      router.policy: [early, late, threshold]

Usage:
  python -m experiments.sweep configs/sweep_oversub.yaml [-j 8] [--save-requests]

Writes results/<name>/summary.csv with one row per run: the grid values
followed by every summary metric.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from multiprocessing import Pool

import yaml

from sim import Simulation, load_config, summarize, write_requests_csv
from sim.config import ROOT, deep_merge, set_dotted


def expand(sweep: dict) -> list[dict]:
    grid = sweep.get("grid", {})
    keys = list(grid)
    runs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        runs.append(dict(zip(keys, values)))
    return runs


def run_one(job):
    base, point, save_dir = job
    over: dict = {}
    for k, v in point.items():
        set_dotted(over, k, v)
    cfg = load_config(overrides=deep_merge(base, over))
    t0 = time.time()
    sim = Simulation(cfg).run()
    s = summarize(sim)
    s["wall_s"] = time.time() - t0
    if save_dir:
        tag = "_".join(f"{k.split('.')[-1]}={v}" for k, v in point.items())
        write_requests_csv(sim.requests, ROOT / save_dir / "requests" / f"{tag}.csv")
    return point, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save-requests", action="store_true")
    args = ap.parse_args()

    with open(args.sweep) as fh:
        sweep = yaml.safe_load(fh)
    name = sweep["name"]
    base = sweep.get("base", {})
    out_dir = f"results/{name}"
    points = expand(sweep)
    jobs = [(base, p, out_dir if args.save_requests else None) for p in points]
    print(f"{name}: {len(jobs)} runs on {args.jobs} workers")

    rows = []
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for i, (point, s) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append({**point, **s})
            print(f"[{i}/{len(jobs)}] {point} second_mean={s['second_mean']:.3f} "
                  f"p99={s['second_p99']:.3f} ({s['wall_s']:.1f}s)", flush=True)

    keys = list(points[0])
    rows.sort(key=lambda r: tuple(str(r[k]) for k in keys))
    path = ROOT / out_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
