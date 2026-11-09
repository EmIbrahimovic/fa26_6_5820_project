"""Build the prefill/decode lookup tables used by sim/profiles.py.

Source: SplitwiseSim's ``data/perf_model.csv`` (measured vLLM numbers),
rows ``model=llama2-70b, hardware=h100-80gb, tensor_parallel=4``. Llama-2-70B
has the same KV layout as Llama-3.1-70B (80 layers, 8 KV heads, head dim 128),
so the per-token KV size and attention cost match; the weights are the same
size class. Repeated measurements are averaged.

Outputs (small, committed):
  data/profiles/prefill_llama70b_h100_tp4.csv   prompt_tokens,time_s
  data/profiles/decode_llama70b_h100_tp4.csv    batch_size,time_s

Usage: python experiments/build_profiles.py [path/to/perf_model.csv]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/profiles/splitwise_perf_model.csv"


def main() -> None:
    prefill = defaultdict(list)  # prompt_size (batch 1) -> prompt_time ms
    decode = defaultdict(list)   # batch size (prompt 512) -> token_time ms
    with open(SRC) as fh:
        for row in csv.DictReader(fh):
            if (row["model"], row["hardware"], row["tensor_parallel"]) != ("llama2-70b", "h100-80gb", "4"):
                continue
            p, b = int(row["prompt_size"]), int(row["batch_size"])
            if b == 1:
                prefill[p].append(float(row["prompt_time"]))
            if p == 512:
                decode[b].append(float(row["token_time"]))
    out = ROOT / "data/profiles"
    with open(out / "prefill_llama70b_h100_tp4.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_tokens", "time_s"])
        for p in sorted(prefill):
            w.writerow([p, f"{sum(prefill[p]) / len(prefill[p]) / 1000:.6f}"])
    with open(out / "decode_llama70b_h100_tp4.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_size", "time_s"])
        for b in sorted(decode):
            w.writerow([b, f"{sum(decode[b]) / len(decode[b]) / 1000:.6f}"])
    print("wrote", out)


if __name__ == "__main__":
    main()
