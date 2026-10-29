"""Within-run acceptance-margin association from offline audit outputs."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "outputs/review_audit"
rng = np.random.default_rng(20261028)
rows = list(csv.DictReader((OUT / "accepted_margins.csv").open()))
results = {}
for cond in ["real100_m10", "real100"]:
    group = [r for r in rows if r["condition"] == cond]
    blocks = []
    per_seed = []
    for seed in range(10):
        records = [r for r in group if int(r["seed"]) == seed]
        x = np.array([float(r["margin"]) for r in records])
        y = np.array([float(r["validation_delta"]) for r in records])
        usable = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
        per_seed.append({"seed": seed, "n": len(x), "rho": float(stats.spearmanr(x, y).statistic) if usable else None})
        if len(x) >= 2:
            xr, yr = stats.rankdata(x), stats.rankdata(y)
            blocks.append((xr - xr.mean(), yr - yr.mean()))
    x = np.concatenate([a for a, _ in blocks])
    y = np.concatenate([b for _, b in blocks])
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    observed = np.dot(x, y) / denominator
    null = []
    for _ in range(10000):
        null.append(sum(np.dot(a, rng.permutation(b)) for a, b in blocks) / denominator)
    results[cond] = {
        "per_seed": per_seed,
        "within_run_centered_rank_association": float(observed),
        "within_run_permutation_p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed))) / 10001),
        "note": "Exploratory association; permutations assume within-run exchangeability and do not remove adaptive search dependence.",
    }
(OUT / "margin_within_run.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
