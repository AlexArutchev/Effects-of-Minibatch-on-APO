"""Paper-specific analyze specificity entropy protocol and helpers."""

import argparse
import json
import math
import os
import statistics

from minibatch_apo.paper.ablate_correlate import faithful_specificities, parse_run_name

ENTROPY_BINS = 8


def load_scored(run_dir: str) -> list[tuple[float, float]]:
    """Return (specificity, utility) pairs for rules with a confident birth match."""
    with open(os.path.join(run_dir, "rule_utility.json")) as f:
        rules = json.load(f)["rules"]
    specs = faithful_specificities(run_dir, [r["text"] for r in rules])
    return [(s, r["utility"]) for s, r in zip(specs, rules, strict=False) if s is not None]


def normalized_entropy(values: list[float], lo: float, hi: float, bins: int = ENTROPY_BINS) -> float:
    """Shannon entropy of a binned distribution, normalized to [0, 1] (1 = uniform)."""
    if len(values) < 2 or hi <= lo:
        return float("nan")
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    total = sum(counts)
    ent = -sum((c / total) * math.log(c / total) for c in counts if c)
    return ent / math.log(bins)


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def summarize_condition(cond: str, per_prompt: list[list[tuple[float, float]]]) -> None:
    per_prompt = [p for p in per_prompt if len(p) >= 2]  # need spread within a prompt
    all_specs = [s for p in per_prompt for s, _ in p]
    if not all_specs:
        print(f"  {cond:<14}  (no scorable prompts)")
        return
    lo, hi = min(all_specs), max(all_specs)

    within_spec = statistics.mean([_std([s for s, _ in p]) for p in per_prompt])
    within_util = statistics.mean([_std([u for _, u in p]) for p in per_prompt])
    prompt_means = [statistics.mean([s for s, _ in p]) for p in per_prompt]
    between_spec = _std(prompt_means)
    var_ratio = within_spec**2 / (within_spec**2 + between_spec**2) if (within_spec or between_spec) else float("nan")
    entropies = [e for e in (normalized_entropy([s for s, _ in p], lo, hi) for p in per_prompt) if not math.isnan(e)]
    ent = statistics.mean(entropies) if entropies else float("nan")

    dominance = "per-rule" if var_ratio > 0.5 else "population"
    sharpness = "sharp" if ent < 0.6 else "spread"
    print(f"  {cond:<14}  prompts={len(per_prompt):<2} rules={len(all_specs):<3}")
    print(f"      within-prompt  spec std = {within_spec:.4f}   (util std = {within_util:.4f} for scale)")
    print(f"      between-prompt spec std = {between_spec:.4f}   (population signal)")
    print(f"      variance ratio within/(within+between) = {var_ratio:.2f}  -> {dominance}-dominated")
    print(f"      mean normalized entropy = {ent:.2f}  ({sharpness})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Within-prompt specificity spread / entropy (ByCS Fig-4 analog).")
    p.add_argument("run_dirs", nargs="+", help="Run dirs with rule_utility.json")
    p.add_argument("--out", help="Optional JSON of per-rule {run, specificity, utility} records")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    by_cond: dict[str, list[list[tuple[float, float]]]] = {}
    records: list[dict] = []
    for run_dir in args.run_dirs:
        cond, _ = parse_run_name(run_dir)
        pairs = load_scored(run_dir)
        by_cond.setdefault(cond, []).append(pairs)
        records += [{"run": os.path.basename(run_dir.rstrip("/")), "specificity": s, "utility": u} for s, u in pairs]

    print(f"\n{'=' * 78}")
    print("  Within-prompt specificity spread (ByCS Fig-4 analog: sharp = can't rank rules)")
    print(f"{'=' * 78}")
    for cond in sorted(by_cond):
        summarize_condition(cond, by_cond[cond])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n  wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
