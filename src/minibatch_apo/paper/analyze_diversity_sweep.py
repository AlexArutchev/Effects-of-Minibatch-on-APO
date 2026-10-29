"""Cross-condition analysis for the training-set diversity sweep.

Loads every (diversity_level, seed) condition from the sweep root, groups by
diversity level, prints a summary table, runs a one-way ANOVA across levels,
and optionally saves a figure and refreshes diversity_summary.json.

Handles the uneven seed design (10 seeds at levels 1/2/4/7, 5 at levels 3/5/6)
by iterating the expected conditions from sweep_diversity.SEEDS_BY_LEVEL, so
stray directories like d1_s4_broken are never picked up.

Usage::

    uv run python src/minibatch_apo/paper/analyze_diversity_sweep.py runs/diversity_sweep
    uv run python src/minibatch_apo/paper/analyze_diversity_sweep.py runs/diversity_sweep --plot
    uv run python src/minibatch_apo/paper/analyze_diversity_sweep.py runs/diversity_sweep --specificity
"""

import argparse
import json
import math
import re
from pathlib import Path

from minibatch_apo.paper.analyze_sweep import compute_run_stats, load_run
from minibatch_apo.paper.sweep_diversity import DIVERSITY_LEVELS, SEEDS_BY_LEVEL, SUBJECTS_BY_LEVEL

# ── loading ───────────────────────────────────────────────────────────────────


SPEC_DELTA_RE = re.compile(r"Δ\s+:\s+([+-]?\d+\.\d+)")


def load_specificity_delta(run_dir: Path) -> float | None:
    """Parse the overall Δ (added - survived specificity) from cross_spec.txt."""
    spec_path = run_dir / "cross_spec.txt"
    if not spec_path.exists():
        return None
    match = SPEC_DELTA_RE.search(spec_path.read_text())
    return float(match.group(1)) if match else None


def load_grouped(sweep_root: Path, with_specificity: bool = False) -> dict[int, list[dict]]:
    """Load every expected condition and group run stats by diversity level."""
    grouped: dict[int, list[dict]] = {}
    for d in DIVERSITY_LEVELS:
        for s in SEEDS_BY_LEVEL[d]:
            run_dir = sweep_root / f"d{d}_s{s}"
            loaded = load_run(run_dir)
            if loaded is None:
                continue
            meta, records = loaded
            if not records:
                continue
            stats = compute_run_stats(meta, records)
            stats["num_types"] = meta["num_types"]
            stats["subjects"] = meta.get("subjects", SUBJECTS_BY_LEVEL[d])
            if with_specificity:
                stats["specificity_delta"] = load_specificity_delta(run_dir)
            grouped.setdefault(d, []).append(stats)
    return grouped


# ── stats helpers ─────────────────────────────────────────────────────────────


def _clean(xs: list) -> list[float]:
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs: list) -> float:
    xs = _clean(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list) -> float:
    xs = _clean(xs)
    if len(xs) < 2:
        return float("nan")
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(mu: float, sd: float, pct: bool = False) -> str:
    scale = 100.0 if pct else 1.0
    if math.isnan(mu):
        return "        —"
    if math.isnan(sd):
        return f"{mu * scale:7.3f}  "
    return f"{mu * scale:6.3f}±{sd * scale:.3f}"


# ── one-way ANOVA (unbalanced, fixed-effects) ─────────────────────────────────


def one_way_anova(grouped: dict[int, list[dict]], metric: str) -> dict | None:
    """One-way ANOVA of *metric* across diversity levels. Handles unequal cell sizes."""
    cells = {d: _clean([r.get(metric) for r in runs]) for d, runs in grouped.items()}
    cells = {d: xs for d, xs in cells.items() if len(xs) >= 2}
    if len(cells) < 2:
        return None

    all_values = [x for xs in cells.values() for x in xs]
    grand_mean = _mean(all_values)

    ss_between = sum(len(xs) * (_mean(xs) - grand_mean) ** 2 for xs in cells.values())
    ss_within = sum((x - _mean(xs)) ** 2 for xs in cells.values() for x in xs)
    ss_total = ss_between + ss_within

    df_between = len(cells) - 1
    df_within = len(all_values) - len(cells)
    if df_within <= 0 or ss_within == 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    try:
        from scipy import stats

        p_value = float(stats.f.sf(f_stat, df_between, df_within))
    except ImportError:
        p_value = float("nan")

    return {
        "df_between": df_between,
        "df_within": df_within,
        "F": f_stat,
        "p": p_value,
        "eta_sq": ss_between / ss_total if ss_total else float("nan"),
    }


def print_anova(anova: dict | None, metric_label: str, across: str = "diversity levels") -> None:
    print(f"\n  One-way ANOVA across {across} — {metric_label}")
    if anova is None:
        print("    Skipped: fewer than 2 levels with 2+ runs, or no within-cell variance.")
        return
    p_str = f"{anova['p']:.4f}" if not math.isnan(anova["p"]) else "—"
    print(
        f"    F({anova['df_between']}, {anova['df_within']}) = {anova['F']:.3f}   "
        f"p = {p_str}   eta² = {anova['eta_sq']:.3f}"
    )


# ── summary table ─────────────────────────────────────────────────────────────


def print_summary_table(grouped: dict[int, list[dict]], with_specificity: bool = False) -> None:
    print(f"\n{'=' * 95}")
    print("  Diversity sweep — summary (mean ± std across seeds)")
    print(f"{'=' * 95}")
    spec_header = f"  {'spec Δ':>12}" if with_specificity else ""
    print(
        f"  {'d':>3}  {'n':>3}  {'seed_acc':>9}  {'best_acc':>14}  "
        f"{'Δacc':>12}  {'accept%':>12}  {'gen%':>12}  {'proposals':>9}{spec_header}"
    )
    print(
        f"  {'─' * 3}  {'─' * 3}  {'─' * 9}  {'─' * 14}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 9}"
        + (f"  {'─' * 12}" if with_specificity else "")
    )

    for d in sorted(grouped):
        runs = grouped[d]
        spec_col = ""
        if with_specificity:
            spec_values = [r.get("specificity_delta") for r in runs]
            spec_col = f"  {_fmt(_mean(spec_values), _std(spec_values)):>12}"
        print(
            f"  {d:>3}  {len(runs):>3}  {_mean([r['seed_score'] for r in runs]):>9.3f}  "
            f"{_fmt(_mean([r['best_score'] for r in runs]), _std([r['best_score'] for r in runs])):>14}  "
            f"{_fmt(_mean([r['improvement'] for r in runs]), _std([r['improvement'] for r in runs])):>12}  "
            f"{_fmt(_mean([r['acceptance_rate'] for r in runs]), _std([r['acceptance_rate'] for r in runs]), pct=True):>12}  "
            f"{_fmt(_mean([r['generalization_rate'] for r in runs]), _std([r['generalization_rate'] for r in runs]), pct=True):>12}  "
            f"{_mean([r['n_proposals'] for r in runs]):>9.0f}{spec_col}"
        )

    print()
    print("  Columns:")
    print("    d           — number of MATH subject categories in the training set")
    print("    seed_acc    — val accuracy of the seed prompt (no optimization)")
    print("    best_acc    — best val accuracy achieved within the rollout budget")
    print("    Δacc        — best_acc - seed_acc (optimization gain)")
    print("    accept%     — fraction of proposals accepted on the minibatch test")
    print("    gen%        — fraction of accepted proposals that also improved val score")
    if with_specificity:
        print("    spec Δ      — cross-iteration specificity (added - survived); gaps where cross_spec.txt is missing")


# ── JSON summary ──────────────────────────────────────────────────────────────


def write_summary_json(grouped: dict[int, list[dict]], out_path: Path) -> None:
    """Refresh diversity_summary.json (same shape as the original ad-hoc export)."""
    keep = (
        "num_types",
        "subjects",
        "seed",
        "seed_score",
        "best_score",
        "improvement",
        "acceptance_rate",
        "generalization_rate",
        "n_proposals",
        "n_accepted",
        "specificity_delta",
    )
    summary = {str(d): [{k: r[k] for k in keep if k in r} for r in runs] for d, runs in sorted(grouped.items())}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary → {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_results(grouped: dict[int, list[dict]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available — skipping plots")
        return

    levels = sorted(grouped)
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("improvement", axes[0], "Val accuracy gain (Δ = best - seed)", 1.0),
        ("generalization_rate", axes[1], "Generalization rate", 100.0),
        ("acceptance_rate", axes[2], "Acceptance rate", 100.0),
    ]
    for metric, ax, title, scale in panels:
        means = [_mean([r[metric] for r in grouped[d]]) * scale for d in levels]
        stds = [_std([r[metric] for r in grouped[d]]) * scale for d in levels]
        stds = [0.0 if math.isnan(s) else s for s in stds]
        for d in levels:
            ys = [x * scale for x in _clean([r[metric] for r in grouped[d]])]
            jitter = rng.uniform(-0.12, 0.12, size=len(ys))
            ax.scatter(np.full(len(ys), d) + jitter, ys, s=14, alpha=0.45, color="steelblue", zorder=2)
        ax.errorbar(levels, means, yerr=stds, marker="o", color="darkorange", capsize=4, zorder=3, label="mean ± std")
        ax.set_xticks(levels)
        ax.set_xlabel("Training-set diversity (# MATH subjects)")
        ax.set_ylabel(title + ("" if scale == 1.0 else " (%)"))
        ax.set_title(title)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Diversity sweep (minibatch=10, fixed 7-subject valset)")
    plt.tight_layout()
    out_path = out_dir / "diversity_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the training-set diversity sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sweep_root", nargs="?", default="runs/diversity_sweep", help="Root directory of the sweep")
    parser.add_argument("--plot", action="store_true", help="Save the summary figure")
    parser.add_argument(
        "--specificity", action="store_true", help="Also load cross_spec.txt Δ per run (gaps where missing)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    grouped = load_grouped(sweep_root, with_specificity=args.specificity)

    n_expected = sum(len(SEEDS_BY_LEVEL[d]) for d in DIVERSITY_LEVELS)
    n_loaded = sum(len(v) for v in grouped.values())
    print(f"Loaded {n_loaded}/{n_expected} conditions.")
    for d in DIVERSITY_LEVELS:
        if len(grouped.get(d, [])) != len(SEEDS_BY_LEVEL[d]):
            print(f"  Warning: d={d} has {len(grouped.get(d, []))}/{len(SEEDS_BY_LEVEL[d])} seeds")

    print_summary_table(grouped, with_specificity=args.specificity)

    print_anova(one_way_anova(grouped, "improvement"), "val accuracy gain")
    print_anova(one_way_anova(grouped, "generalization_rate"), "generalization rate")
    if args.specificity:
        print_anova(one_way_anova(grouped, "specificity_delta"), "specificity Δ")

    write_summary_json(grouped, sweep_root / "diversity_summary.json")

    if args.plot:
        plot_results(grouped, sweep_root)
    else:
        print("Run with --plot to generate the summary figure.")
