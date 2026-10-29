"""Produce standalone figures and provenance for the saved-data reanalysis."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/review_audit"
runs = list(csv.DictReader((OUT / "run_statistics.csv").open()))
spec = list(csv.DictReader((OUT / "specificity_per_run.csv").open()))
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
for panel, ax in enumerate(axes):
    values = []
    for cond in ["real100_m10", "real100"]:
        if panel == 0:
            data = {int(r["seed"]): float(r["generalization_rate"]) * 100 for r in runs if r["condition"] == cond}
        else:
            key = "added_mean" if panel == 1 else "delta"
            data = {
                int(r["seed"]): float(r[key])
                for r in spec
                if r["condition"] == cond and r["threshold"] == "0.8" and r["metric"] == "disjoint_equal_k3" and r[key]
            }
        values.append(data)
    shared = sorted(set(values[0]) & set(values[1]))
    if panel == 2:
        values = [{s: d[s] for s in shared} for d in values]
    for seed in shared:
        ax.plot([0, 1], [d[seed] for d in values], color="0.78", linewidth=0.8, zorder=1)
    for i, data in enumerate(values):
        ax.scatter(np.full(len(data), i), list(data.values()), s=28, alpha=0.7, zorder=2)
        ax.scatter(i, np.mean(list(data.values())), marker="D", color="black", s=40, zorder=3)
    ax.set_xticks([0, 1], [f"m=10 (n={len(values[0])})", f"m=50 (n={len(values[1])})"])
    ax.grid(axis="y", alpha=0.2)
    if panel > 0:
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Accepted-update transfer (%)")
axes[0].set_title("Transfer: all 10 seed pairs")
axes[1].set_ylabel("Mean added-rule specificity")
axes[1].set_title("Equal k=3, disjoint references")
axes[2].set_ylabel("Added-minus-matched gap")
axes[2].set_title("Same five eligible seed pairs")
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(OUT / f"matched_reanalysis.{suffix}", dpi=220)
plt.close(fig)
inputs = []
for cond in ["real100_m10", "real100"]:
    for seed in range(10):
        folder = ROOT / f"runs/augmentation_sweep/{cond}_s{seed}"
        for name in [
            "research_log.jsonl",
            "run_log.json",
            "sweep_meta.json",
            "cross_spec.txt",
            "rule_utility.json",
            "oracle_prune.json",
        ]:
            path = folder / name
            if path.exists():
                inputs.append(
                    {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
(OUT / "provenance.json").write_text(
    json.dumps(
        {
            "inputs": inputs,
            "embedding_model": "Qwen/Qwen3-Embedding-4B",
            "embedding_max_tokens": 512,
            "subset_size": 3,
            "matching_thresholds": [0.7, 0.75, 0.8],
            "source_permutations": 2000,
            "rng_seed": 20261028,
            "note": "No new task/reflection calls; original artifacts unchanged.",
        },
        indent=2,
    )
)
print("Wrote PDF, PNG, and input hashes")
