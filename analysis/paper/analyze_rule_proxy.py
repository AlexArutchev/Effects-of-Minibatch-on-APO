"""Compute training-minus-validation specificity for selected prompts."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from minibatch_apo.paper import analyze_run
from minibatch_apo.paper.sweep_diversity import build_datasets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/rule_ablations"
cache = np.load(ROOT / "outputs/specificity_checks/embeddings.npz")
lookup = dict(zip(cache["texts"].tolist(), cache["vectors"].tolist(), strict=True))
extra = OUT / "proxy_embeddings.npz"
if extra.exists():
    cached = np.load(extra)
    lookup.update(zip(cached["texts"].tolist(), cached["vectors"].tolist(), strict=True))
trainsets = {}
for seed in range(5):
    train, validation = build_datasets(7, seed)
    trainsets[seed] = [e["input"] for e in train]
validation = [e["input"] for e in validation]
texts = sorted(set(validation + [t for pool in trainsets.values() for t in pool]))
missing = [t for t in texts if t not in lookup]
if missing:
    print(f"Embedding {len(missing)} previously uncached problem texts locally", flush=True)
    lookup.update(zip(missing, analyze_run._embed_rules(missing, "Qwen/Qwen3-Embedding-4B"), strict=True))
np.savez_compressed(extra, texts=np.asarray(texts), vectors=np.asarray([lookup[t] for t in texts], dtype=np.float32))
val = np.asarray([lookup[t] for t in validation])
summary, rows = {}, []
for condition in ["real100_m10", "real100"]:
    pooled = []
    per_seed = []
    for seed in range(5):
        data = json.loads((OUT / f"{condition}_s{seed}" / "rule_utility.json").read_text())
        rules = np.asarray([lookup[r["text"]] for r in data["rules"]])
        train = np.asarray([lookup[t] for t in trainsets[seed]])

        def cosine(a, b):
            return (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T

        specificity = cosine(rules, train).max(axis=1) - cosine(rules, val).max(axis=1)
        utilities = np.asarray([r["utility"] for r in data["rules"]])
        corr = stats.spearmanr(specificity, utilities)
        per_seed.append({"seed": seed, "n": len(rules), "rho": float(corr.statistic)})
        pooled.extend(zip(specificity, utilities, strict=True))
        rows.extend(
            {"condition": condition, "seed": seed, "rule_idx": i, "specificity": float(s), "utility": float(u)}
            for i, (s, u) in enumerate(zip(specificity, utilities, strict=True))
        )
    corr = stats.spearmanr(*zip(*pooled, strict=True))
    summary[condition] = {
        "n": len(pooled),
        "rho": float(corr.statistic),
        "p_naive": float(corr.pvalue),
        "per_seed": per_seed,
    }
(OUT / "proxy_summary.json").write_text(json.dumps(summary, indent=2))
with (OUT / "proxy_pairs.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
