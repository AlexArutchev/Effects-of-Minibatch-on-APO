"""Offline reanalysis of saved GEPA runs; never calls task/reflection APIs."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from minibatch_apo.paper.analyze_run import extract_rules_from_text
from minibatch_apo.paper.analyze_sweep import compute_run_stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/specificity_checks"
CONDITIONS = ["real100_m10", "real100"]


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    n = len(d)
    se = stats.sem(d) if n > 1 else float("nan")
    ci = stats.t.interval(0.95, n - 1, loc=np.mean(d), scale=se) if se else (np.mean(d), np.mean(d))
    null = [abs(np.mean(d * signs)) for signs in itertools.product([-1, 1], repeat=n)]
    return {
        "n": n,
        "mean_m10": float(a.mean()),
        "mean_m50": float(b.mean()),
        "difference": float(d.mean()),
        "ci95": list(map(float, ci)),
        "paired_t_p": float(stats.ttest_rel(b, a).pvalue),
        "exact_sign_flip_p": float(np.mean(np.asarray(null) >= abs(d.mean()) - 1e-12)),
    }


def load_runs():
    runs = []
    for cond in CONDITIONS:
        for seed in range(10):
            path = ROOT / f"runs/augmentation_sweep/{cond}_s{seed}"
            recs = records(path / "research_log.jsonl")
            meta = read_json(path / "sweep_meta.json")
            acc = [r for r in recs if r.get("status") == "accepted"]
            runs.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "path": path,
                    "records": recs,
                    "accepted": acc,
                    "stats": compute_run_stats(meta, recs),
                }
            )
    return runs


def summarize_runs(runs):
    rows, margins = [], []
    for run in runs:
        path, acc = run["path"], run["accepted"]
        row = dict(condition=run["condition"], **run["stats"])
        row.pop("trajectory")
        text = (path / "cross_spec.txt").read_text()
        import re

        match = re.search(r"Δ\s+:\s+([+-]?\d+\.\d+)", text)
        row["original_specificity"] = float(match[1]) if match else None
        row["original_specificity_defined"] = bool(match)
        row["n_transferred"] = sum(r["valset_score"] > r["parent_score"] for r in acc)
        rows.append(row)
        evaluation_log = read_json(path / "run_log.json")
        for r in acc:
            score_matches = [
                entry for entry in evaluation_log if entry.get("new_program_idx") == r["new_candidate_idx"]
            ]
            if len(score_matches) != 1:
                raise ValueError(f"Ambiguous score match: {path}, {r['iteration']}")
            entry = score_matches[0]
            before = sum(entry["subsample_scores"])
            after = sum(entry["new_subsample_scores"])
            if after <= before:
                raise ValueError("Accepted candidate without strict minibatch improvement")
            margins.append(
                {
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "iteration": r["iteration"],
                    "before": before,
                    "after": after,
                    "margin": None if before is None or after is None else after - before,
                    "normalized_margin": (after - before) / len(entry["subsample_scores"]),
                    "validation_delta": r["valset_score"] - r["parent_score"],
                }
            )
    write_csv("run_statistics.csv", rows)
    write_csv("accepted_margins.csv", margins)
    summary = {}
    for metric in ["generalization_rate", "improvement", "acceptance_rate", "n_accepted", "n_proposals"]:
        values = [[r[metric] for r in rows if r["condition"] == c] for c in CONDITIONS]
        summary[metric] = paired_summary(*values)
    summary["exclusion_groups"] = []
    for cond in CONDITIONS:
        for defined in [False, True]:
            group = [r for r in rows if r["condition"] == cond and r["original_specificity_defined"] == defined]
            if group:
                summary["exclusion_groups"].append(
                    dict(
                        condition=cond,
                        defined=defined,
                        n=len(group),
                        **{
                            k: float(np.mean([r[k] for r in group]))
                            for k in ["generalization_rate", "improvement", "n_accepted"]
                        },
                    )
                )
    summary["margin_available"] = sum(r["margin"] is not None for r in margins)
    summary["accepted_total"] = len(margins)
    for cond in CONDITIONS:
        group = [r for r in margins if r["condition"] == cond and r["margin"] is not None]
        if len(group) > 2:
            test = stats.spearmanr([r["margin"] for r in group], [r["validation_delta"] for r in group])
            summary[f"margin_correlation_{cond}"] = {
                "n": len(group),
                "rho": float(test.statistic),
                "p": float(test.pvalue),
            }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def get_embeddings(texts):
    cache = OUT / "embeddings.npz"
    if cache.exists():
        data = np.load(cache)
        if data["texts"].tolist() == texts:
            return data["vectors"]
    from minibatch_apo.paper.analyze_run import _embed_rules

    print(f"Embedding {len(texts)} unique saved texts", flush=True)
    vectors = np.asarray(_embed_rules(texts, "Qwen/Qwen3-Embedding-4B"), dtype=np.float32)
    np.savez_compressed(cache, texts=np.asarray(texts), vectors=vectors)
    return vectors


def expected_max(sim, ids, k=3):
    """Exact mean max over all uniform k-subsets; no subsampling error."""
    n = len(ids)
    if n < k:
        return np.full(len(sim), np.nan)
    ordered = np.sort(sim[:, ids], axis=1)
    weights = np.array([math.comb(j, k - 1) / math.comb(n, k) if j >= k - 1 else 0 for j in range(n)])
    return ordered @ weights


def specificity(runs):
    texts = set()
    for run in runs:
        for r in run["accepted"]:
            for key in ["old_texts", "new_texts"]:
                texts.update(extract_rules_from_text(r[key]["system_prompt"]))
            texts.update(
                ex["Inputs"] for ex in r.get("reflective_dataset", {}).get("system_prompt", []) if ex.get("Inputs")
            )
    texts = sorted(texts)
    vectors = get_embeddings(texts)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = {text: i for i, text in enumerate(texts)}
    rows = []
    batch_rows = []
    rng = np.random.default_rng(20261028)
    for run in runs:
        acc = run["accepted"]
        batches = [
            list(
                dict.fromkeys(
                    ex["Inputs"] for ex in r.get("reflective_dataset", {}).get("system_prompt", []) if ex.get("Inputs")
                )
            )
            for r in acc
        ]
        for i, batch in enumerate(batches):
            overlaps = [len(set(batch) & set(other)) for j, other in enumerate(batches) if j != i]
            batch_rows.append(
                {
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "iteration": acc[i]["iteration"],
                    "n_distinct": len(batch),
                    "mean_reference_overlap": float(np.mean(overlaps)) if overlaps else None,
                }
            )
        # Build one array of rules, preserving their occurrence/update identity.
        rules = []
        own = []
        match_scores = []
        for i, r in enumerate(acc):
            child = extract_rules_from_text(r["new_texts"]["system_prompt"])
            parent = extract_rules_from_text(r["old_texts"]["system_prompt"])
            if not child:
                continue
            ce = vectors[[index[t] for t in child]]
            pe = vectors[[index[t] for t in parent]]
            scores = (ce @ pe.T).max(axis=1) if parent else np.zeros(len(child))
            rules.extend(child)
            own.extend([i] * len(child))
            match_scores.extend(scores)
        if not rules or len(acc) < 2:
            continue
        own = np.asarray(own)
        match_scores = np.asarray(match_scores)
        problems = sorted({t for b in batches for t in b})
        pidx = {t: i for i, t in enumerate(problems)}
        bids = [[pidx[t] for t in b] for b in batches]
        sim = vectors[[index[t] for t in rules]] @ vectors[[index[t] for t in problems]].T
        full = np.column_stack([sim[:, ids].max(axis=1) for ids in bids])
        equal = np.column_stack([expected_max(sim, ids) for ids in bids])
        ar = np.arange(len(rules))
        raw = full[ar, own] - (full.sum(axis=1) - full[ar, own]) / (len(acc) - 1)
        eq = equal[ar, own] - (equal.sum(axis=1) - equal[ar, own]) / (len(acc) - 1)
        disjoint = np.full(len(rules), np.nan)
        for i in range(len(acc)):
            mask = own == i
            refs = [
                expected_max(sim, sorted(set(other) - set(bids[i])))
                for j, other in enumerate(bids)
                if i != j and len(set(other) - set(bids[i])) >= 3
            ]
            if refs:
                disjoint[mask] = equal[mask, i] - np.mean(refs, axis=0)[mask]
        for threshold in [0.70, 0.75, 0.80]:
            added = match_scores < threshold
            survived = ~added
            for name, values in [("original", raw), ("equal_k3", eq), ("disjoint_equal_k3", disjoint)]:
                a = values[added & np.isfinite(values)]
                s = values[survived & np.isfinite(values)]
                # All-added mean remains defined even where the gap is not.
                delta = float(a.mean() - s.mean()) if len(a) and len(s) else None
                rows.append(
                    {
                        "condition": run["condition"],
                        "seed": run["seed"],
                        "threshold": threshold,
                        "metric": name,
                        "added_n": len(a),
                        "matched_n": len(s),
                        "added_mean": float(a.mean()) if len(a) else None,
                        "matched_mean": float(s.mean()) if len(s) else None,
                        "delta": delta,
                    }
                )
            # Permute source batch assignments jointly at update level, preserving rule groups.
            # Uses equal-size statistic; retains actual batch overlap under the null.
            observed = eq[added].mean() if added.any() else np.nan
            null = []
            for _ in range(2000):
                perm = rng.permutation(len(acc))
                assigned = perm[own]
                val = equal[ar, assigned] - (equal.sum(axis=1) - equal[ar, assigned]) / (len(acc) - 1)
                null.append(val[added].mean() if added.any() else np.nan)
            rows.append(
                {
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "threshold": threshold,
                    "metric": "source_permutation_equal_k3",
                    "added_n": int(added.sum()),
                    "matched_n": int(survived.sum()),
                    "added_mean": float(observed),
                    "permutation_p": float((1 + sum(x >= observed for x in null)) / (len(null) + 1)),
                    "null_mean": float(np.mean(null)),
                    "null_sd": float(np.std(null)),
                }
            )
        print(f"Analyzed {run['condition']} seed {run['seed']}: {len(rules)} rule occurrences", flush=True)
    write_csv("specificity_per_run.csv", rows)
    write_csv("batch_overlap.csv", batch_rows)
    summary = []
    for threshold in [0.70, 0.75, 0.80]:
        for metric in ["original", "equal_k3", "disjoint_equal_k3"]:
            groups = [
                [r for r in rows if r["condition"] == cond and r["threshold"] == threshold and r["metric"] == metric]
                for cond in CONDITIONS
            ]
            byseed = [{r["seed"]: r for r in g} for g in groups]
            shared = [s for s in range(10) if all(s in g and g[s].get("delta") is not None for g in byseed)]
            item = {"threshold": threshold, "metric": metric}
            for cond, g in zip(CONDITIONS, groups, strict=False):
                valid = [r["delta"] for r in g if r["delta"] is not None]
                added = [r["added_mean"] for r in g if r["added_mean"] is not None]
                item[cond] = {
                    "n": len(valid),
                    "gap_mean": float(np.mean(valid)) if valid else None,
                    "gap_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                    "all_added_mean": float(np.mean(added)),
                }
            if len(shared) > 1:
                item["paired_gap"] = paired_summary(*[[g[s]["delta"] for s in shared] for g in byseed])
            item["paired_added"] = paired_summary(*[[g[s]["added_mean"] for s in range(10)] for g in byseed])
            summary.append(item)
    (OUT / "specificity_summary.json").write_text(json.dumps(summary, indent=2))
    print("Specificity analysis complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    summarize_runs(runs)
    if args.embeddings:
        specificity(runs)


if __name__ == "__main__":
    main()
