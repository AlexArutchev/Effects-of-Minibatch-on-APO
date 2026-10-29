"""Summarize corrected ablations, using unchanged data for the three reused runs."""

import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

from minibatch_apo.paper import ablate_correlate, analyze_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/corrected_ablations"


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    missing = [item["run"] for item in plan if not (OUT / item["run"] / "rule_utility.json").exists()]
    if missing:
        raise RuntimeError("Incomplete runs: " + ", ".join(missing))
    cache = np.load(ROOT / "outputs/review_audit/embeddings.npz")
    lookup = dict(zip(cache["texts"].tolist(), cache["vectors"].tolist(), strict=True))

    def cached_only(texts, model_name):
        return [lookup[text] for text in texts]

    ablate_correlate._embed_rules = cached_only
    analyze_run._embed_rules = cached_only
    summary, rows = {}, []
    for condition in ["real100_m10", "real100"]:
        pooled, seeds = [], []
        for item in [item for item in plan if item["condition"] == condition]:
            folder = OUT / item["run"]
            data = json.loads((folder / "rule_utility.json").read_text())
            if item["redo"]:
                assert data["prompt_sha256"] == item["prompt_sha256"]
                assert len(data["baseline_scores"]) == data["n_val"] == 100
                assert all(len(rule["scores_without"]) == 100 for rule in data["rules"])
            texts = [rule["text"] for rule in data["rules"]]
            with contextlib.redirect_stdout(io.StringIO()):
                scores = ablate_correlate.faithful_specificities(item["source"], texts)
            pairs = [(s, rule["utility"]) for s, rule in zip(scores, data["rules"], strict=True) if s is not None]
            pooled.extend(pairs)
            for rule, specificity in zip(data["rules"], scores, strict=True):
                rows.append(
                    {
                        "run": item["run"],
                        "seed": item["seed"],
                        "condition": condition,
                        "specificity": specificity,
                        **{k: rule[k] for k in ["idx", "utility", "acc_without"]},
                    }
                )
            correlation = stats.spearmanr(*zip(*pairs, strict=True)) if len(pairs) > 2 else None
            row = {
                "seed": item["seed"],
                "reused": not item["redo"],
                "iteration": item["iteration"],
                "n_rules": len(texts),
                "n_matched": len(pairs),
                "base_acc": data["base_acc"],
                "rho": float(correlation.statistic) if correlation else None,
                "harmful_fraction": float(np.mean([r["utility"] < 0 for r in data["rules"]])),
            }
            oracle_path = folder / "oracle_prune.json"
            if oracle_path.exists():
                oracle = json.loads(oracle_path.read_text())
                row["oracle_gain"] = oracle["oracle_gain"]
                row["summed_negative_estimate"] = oracle["sum_neg_estimate"]
            seeds.append(row)
        correlation = stats.spearmanr(*zip(*pooled, strict=True)) if len(pooled) > 2 else None
        summary[condition] = {
            "seeds": seeds,
            "n_paired_rules": len(pooled),
            "pooled_rho": float(correlation.statistic) if correlation else None,
            "pooled_p_naive": float(correlation.pvalue) if correlation else None,
            "harmful_fraction_mean": float(np.mean([s["harmful_fraction"] for s in seeds])),
            "note": "Pooled rule p-values do not account for dependence within prompts.",
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "rule_pairs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
