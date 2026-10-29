"""Reproduce rule-utility associations from saved scores and cached embeddings."""

import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

from minibatch_apo.paper import ablate_correlate, analyze_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/review_audit"
cache = np.load(OUT / "embeddings.npz")
lookup = dict(zip(cache["texts"].tolist(), cache["vectors"].tolist(), strict=True))


def cached_only(texts, model_name):
    return [lookup[text] for text in texts]


ablate_correlate._embed_rules = cached_only
analyze_run._embed_rules = cached_only
rows = []
summary = {}
for cond in ["real100_m10", "real100"]:
    pooled = []
    seeds = []
    for seed in range(5):
        path = ROOT / f"runs/augmentation_sweep/{cond}_s{seed}"
        data = json.loads((path / "rule_utility.json").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            specificities = ablate_correlate.faithful_specificities(str(path), [r["text"] for r in data["rules"]])
        pairs = [(s, r["utility"]) for s, r in zip(specificities, data["rules"], strict=True) if s is not None]
        pooled.extend(pairs)
        for s, r in zip(specificities, data["rules"], strict=True):
            rows.append(
                {"condition": cond, "seed": seed, "rule_idx": r["idx"], "specificity": s, "utility": r["utility"]}
            )
        corr = stats.spearmanr(*zip(*pairs, strict=True)) if len(pairs) > 2 else None
        oracle = json.loads((path / "oracle_prune.json").read_text())
        assert oracle["n_rules"] == len(data["rules"])
        assert oracle["n_dropped"] == sum(r["utility"] < 0 for r in data["rules"])
        assert np.isclose(oracle["base_acc"], data["base_acc"])
        assert np.isclose(oracle["sum_neg_estimate"], -sum(min(r["utility"], 0) for r in data["rules"]))
        seeds.append(
            {
                "seed": seed,
                "n_rules": len(data["rules"]),
                "n_matched_specificity": len(pairs),
                "rho": float(corr.statistic) if corr else None,
                "harmful_fraction": np.mean([r["utility"] < 0 for r in data["rules"]]),
                "oracle_gain": oracle["oracle_gain"],
                "summed_negative_estimate": oracle["sum_neg_estimate"],
            }
        )
    corr = stats.spearmanr(*zip(*pooled, strict=True))
    summary[cond] = {
        "n_paired_rules": len(pooled),
        "pooled_rho": float(corr.statistic),
        "pooled_p_naive": float(corr.pvalue),
        "seeds": seeds,
    }
with (OUT / "utility_pairs.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "utility_audit.json").write_text(json.dumps(summary, indent=2))
for cond, data in summary.items():
    print(cond, data["n_paired_rules"], data["pooled_rho"], data["pooled_p_naive"])
print("PASS: oracle count, baseline, and summed-negative-estimate consistency for all 10 runs")
