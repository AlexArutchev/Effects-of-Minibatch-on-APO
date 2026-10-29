"""Recompute supporting rule statistics using only saved corrected outcomes."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from minibatch_apo.paper.analyze_specificity_entropy import normalized_entropy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/corrected_ablations"
rows = list(csv.DictReader((OUT / "rule_pairs.csv").open()))
summary = {}
for condition in ["real100_m10", "real100"]:
    group = [r for r in rows if r["condition"] == condition and r["specificity"]]
    prompts = [[r for r in group if int(r["seed"]) == seed] for seed in range(5)]
    pooled = np.array([float(r["specificity"]) for r in group])
    lo, hi = float(pooled.min()), float(pooled.max())
    within = np.mean([np.std([float(r["specificity"]) for r in prompt]) for prompt in prompts])
    between = np.std([np.mean([float(r["specificity"]) for r in prompt]) for prompt in prompts])
    entropy = np.mean([normalized_entropy([float(r["specificity"]) for r in prompt], lo, hi) for prompt in prompts])
    utilities = [float(r["utility"]) for r in group]
    general = [float(r["utility"]) for r in group if float(r["specificity"]) < 0]
    specific = [float(r["utility"]) for r in group if float(r["specificity"]) >= 0]
    summary[condition] = {
        "within_prompt_spec_sd": float(within),
        "between_prompt_spec_sd": float(between),
        "normalized_entropy": float(entropy),
        "matched_utility_mean": float(np.mean(utilities)),
        "matched_utility_sd": float(np.std(utilities)),
        "general_n": len(general),
        "general_utility_mean": float(np.mean(general)) if general else None,
        "specific_utility_mean": float(np.mean(specific)) if specific else None,
    }
    oracle = [json.loads((OUT / f"{condition}_s{seed}" / "oracle_prune.json").read_text()) for seed in range(5)]
    summary[condition]["oracle_mean"] = float(np.mean([o["oracle_gain"] for o in oracle]))
    summary[condition]["oracle_sd_sample"] = float(np.std([o["oracle_gain"] for o in oracle], ddof=1))
    summary[condition]["summed_negative_mean"] = float(np.mean([o["sum_neg_estimate"] for o in oracle]))
# Pooled descriptive mirror check, retained separately from run-level inference.
general = [float(r["utility"]) for r in rows if r["specificity"] and float(r["specificity"]) < 0]
specific = [float(r["utility"]) for r in rows if r["specificity"] and float(r["specificity"]) >= 0]
summary["pooled_mirror"] = {
    "general_n": len(general),
    "general_mean": float(np.mean(general)),
    "general_t_p_naive": float(stats.ttest_1samp(general, 0).pvalue),
    "general_minus_specific": float(np.mean(general) - np.mean(specific)),
    "welch_p_naive": float(stats.ttest_ind(general, specific, equal_var=False).pvalue),
}
(OUT / "extended_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
