"""Plot a sweep's summary.csv.

  python -m analysis.plot_sweep results/<name>/summary.csv [--metric second_mean] [--x KEY] [--y KEY]

Produces, next to the CSV:
  * <metric>_by_<x>.png - metric vs. x for each policy, one panel per value of
    the remaining grid keys;
  * winner_<metric>.png - which policy wins at each (x, y) grid point (when
    --y is given), annotated with the early-vs-late gap;
  * rho_vs_gain.png     - (late - early) / late against the median estimated
    rho of late-bound requests: the hypothesis says the gain shrinks below
    rho ~ 1.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICY_KEY = "router.policy"


def load(path: Path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    grid_keys = [k for k in rows[0] if "." in k]
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows, grid_keys


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--metric", default="second_mean")
    ap.add_argument("--x", default=None)
    ap.add_argument("--y", default=None)
    args = ap.parse_args()
    path = Path(args.summary)
    rows, keys = load(path)
    others = [k for k in keys if k != POLICY_KEY]
    x = args.x or others[0]
    rest = [k for k in others if k != x]
    m = args.metric

    # 1) metric vs x per policy, one panel per combination of the rest
    panels = defaultdict(list)
    for r in rows:
        panels[tuple(r[k] for k in rest)].append(r)
    n = len(panels)
    cols = min(n, 4)
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.2 * nrows), squeeze=False, sharey=True)
    for ax, (pk, pr) in zip(axes.flat, sorted(panels.items(), key=lambda kv: tuple(map(str, kv[0])))):
        by_pol = defaultdict(list)
        for r in pr:
            by_pol[r[POLICY_KEY]].append((r[x], r[m]))
        for pol, pts in sorted(by_pol.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=pol)
        ax.set_title(", ".join(f"{k.split('.')[-1]}={fmt(v)}" for k, v in zip(rest, pk)) or "all", fontsize=9)
        ax.set_xlabel(x.split(".")[-1])
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel(m)
    axes[0][0].legend()
    fig.tight_layout()
    out = path.parent / f"{m}_by_{x.split('.')[-1]}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # 2) winner map over (x, y), other keys fixed to their first value
    if args.y:
        y = args.y
        fixed = [k for k in rest if k != y]
        first = {k: sorted({r[k] for r in rows}, key=str)[0] for k in fixed}
        sel = [r for r in rows if all(r[k] == first[k] for k in fixed)]
        xs = sorted({r[x] for r in sel}, key=lambda v: (isinstance(v, str), v))
        ys = sorted({r[y] for r in sel}, key=lambda v: (isinstance(v, str), v))
        cell = defaultdict(dict)
        for r in sel:
            cell[(r[x], r[y])][r[POLICY_KEY]] = r[m]
        pols = sorted({r[POLICY_KEY] for r in sel})
        colors = {p: c for p, c in zip(pols, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"])}
        fig, ax = plt.subplots(figsize=(1.3 * len(xs) + 2, 1.0 * len(ys) + 1.5))
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                c = cell.get((xv, yv))
                if not c:
                    continue
                win = min(c, key=c.get)
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=colors[win], alpha=0.75))
                txt = win
                if "early" in c and "late" in c:
                    gap = (c["late"] - c["early"]) / c["late"] * 100
                    txt += f"\nE vs L {gap:+.1f}%"
                ax.text(i, j, txt, ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(xs)), [fmt(v) for v in xs])
        ax.set_yticks(range(len(ys)), [fmt(v) for v in ys])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(-0.5, len(ys) - 0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        sub = ", ".join(f"{k.split('.')[-1]}={fmt(v)}" for k, v in first.items())
        ax.set_title(f"lowest {m}" + (f" ({sub})" if sub else "") + "\n+% = early better than late", fontsize=9)
        fig.tight_layout()
        out = path.parent / f"winner_{m}.png"
        fig.savefig(out, dpi=130)
        print("wrote", out)

    # 3) relative gain of early over late vs. rho (from the late run)
    pairs = defaultdict(dict)
    for r in rows:
        pairs[tuple(r[k] for k in others)][r[POLICY_KEY]] = r
    pts = []
    for c in pairs.values():
        if "early" in c and "late" in c:
            rho = c["late"]["rho_est_median"]
            gain = (c["late"][m] - c["early"][m]) / c["late"][m]
            pts.append((rho, gain))
    if pts:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter([p[0] for p in pts], [100 * p[1] for p in pts])
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(1, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("median estimated rho (late-bound run)")
        ax.set_ylabel(f"early gain in {m} (%)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = path.parent / "rho_vs_gain.png"
        fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    main()
