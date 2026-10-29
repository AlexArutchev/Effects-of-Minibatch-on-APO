"""Cross-condition analysis for the augmentation sweep.

Groups runs by augmentation condition, prints a summary table, runs a one-way
ANOVA plus the pairwise contrasts the design was built around (Welch t-tests):

  duplicate  vs baseline   — does a bigger pool help at all, even with nothing new?
  paraphrase vs duplicate  — the mechanism contrast: reflection material vs pure
                             acceptance-test reliability
  synthetic  vs paraphrase — do genuinely new problems beat reworded ones?
  real100    vs synthetic  — how much of the real-data upper bound does synthesis recover?

Usage::

    uv run python src/minibatch_apo/paper/analyze_augmentation.py
    uv run python src/minibatch_apo/paper/analyze_augmentation.py --plot
    uv run python src/minibatch_apo/paper/analyze_augmentation.py --specificity
"""

import argparse
import math
from pathlib import Path

from minibatch_apo.paper.analyze_diversity_sweep import (
    _clean,
    _fmt,
    _mean,
    _std,
    load_specificity_delta,
    one_way_anova,
    print_anova,
)
from minibatch_apo.paper.analyze_sweep import compute_run_stats, load_run
from minibatch_apo.paper.sweep_augmentation import CONDITIONS, SEEDS, SWEEP_ROOT

PAIRWISE_CONTRASTS = [
    ("duplicate", "baseline"),
    ("paraphrase", "duplicate"),
    ("synthetic", "paraphrase"),
    ("real100", "synthetic"),
    # Deconfound the duplicate-baseline pool-size/minibatch confound:
    ("real100", "real100_m10"),  # minibatch effect at a fixed 100-real pool
    ("real100_m10", "baseline"),  # pool-size effect at a fixed minibatch=10
]


def load_grouped(sweep_root: Path, with_specificity: bool = False) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for c in CONDITIONS:
        for s in SEEDS:
            run_dir = sweep_root / f"{c}_s{s}"
            loaded = load_run(run_dir)
            if loaded is None:
                continue
            meta, records = loaded
            if not records:
                continue
            stats = compute_run_stats(meta, records)
            stats["condition"] = c
            if with_specificity:
                stats["specificity_delta"] = load_specificity_delta(run_dir)
            grouped.setdefault(c, []).append(stats)
    return grouped


def print_summary_table(grouped: dict[str, list[dict]], with_specificity: bool = False) -> None:
    print(f"\n{'=' * 100}")
    print("  Augmentation sweep — summary (mean ± std across seeds)")
    print(f"{'=' * 100}")
    spec_header = f"  {'spec Δ':>12}" if with_specificity else ""
    print(
        f"  {'condition':<11}  {'n':>3}  {'seed_acc':>9}  {'best_acc':>14}  "
        f"{'Δacc':>12}  {'accept%':>12}  {'gen%':>12}{spec_header}"
    )
    print(
        f"  {'─' * 11}  {'─' * 3}  {'─' * 9}  {'─' * 14}  {'─' * 12}  {'─' * 12}  {'─' * 12}"
        + (f"  {'─' * 12}" if with_specificity else "")
    )
    for c in CONDITIONS:
        runs = grouped.get(c, [])
        if not runs:
            print(f"  {c:<11}    0  (no completed runs)")
            continue
        spec_col = ""
        if with_specificity:
            spec_values = [r.get("specificity_delta") for r in runs]
            spec_col = f"  {_fmt(_mean(spec_values), _std(spec_values)):>12}"
        print(
            f"  {c:<11}  {len(runs):>3}  {_mean([r['seed_score'] for r in runs]):>9.3f}  "
            f"{_fmt(_mean([r['best_score'] for r in runs]), _std([r['best_score'] for r in runs])):>14}  "
            f"{_fmt(_mean([r['improvement'] for r in runs]), _std([r['improvement'] for r in runs])):>12}  "
            f"{_fmt(_mean([r['acceptance_rate'] for r in runs]), _std([r['acceptance_rate'] for r in runs]), pct=True):>12}  "
            f"{_fmt(_mean([r['generalization_rate'] for r in runs]), _std([r['generalization_rate'] for r in runs]), pct=True):>12}"
            f"{spec_col}"
        )


def print_contrasts(grouped: dict[str, list[dict]], metric: str, label: str) -> None:
    """Welch t-tests for the planned pairwise contrasts."""
    try:
        from scipy import stats
    except ImportError:
        print("  scipy not available — skipping pairwise contrasts")
        return
    print(f"\n  Planned contrasts (Welch t-test) — {label}")
    print(f"  {'contrast':<26}  {'Δmean':>8}  {'t':>7}  {'p':>8}")
    print(f"  {'─' * 26}  {'─' * 8}  {'─' * 7}  {'─' * 8}")
    for hi, lo in PAIRWISE_CONTRASTS:
        a = _clean([r.get(metric) for r in grouped.get(hi, [])])
        b = _clean([r.get(metric) for r in grouped.get(lo, [])])
        if len(a) < 2 or len(b) < 2:
            print(f"  {hi + ' - ' + lo:<26}  insufficient runs")
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {hi + ' - ' + lo:<26}  {_mean(a) - _mean(b):>+8.3f}  {t.statistic:>7.2f}  {t.pvalue:>8.4f}")


def plot_results(grouped: dict[str, list[dict]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        ("improvement", axes[0], "Val accuracy gain (Δ = best - seed)", 1.0),
        ("generalization_rate", axes[1], "Generalization rate (%)", 100.0),
    ]
    xs = np.arange(len(CONDITIONS))
    condition_labels = [condition.replace("_m10", "-m10") for condition in CONDITIONS]
    for metric, ax, title, scale in panels:
        means, stds = [], []
        for i, c in enumerate(CONDITIONS):
            ys = [y * scale for y in _clean([r.get(metric) for r in grouped.get(c, [])])]
            means.append(_mean(ys) if ys else float("nan"))
            sd = _std(ys)
            stds.append(0.0 if math.isnan(sd) else sd)
            jitter = rng.uniform(-0.12, 0.12, size=len(ys))
            ax.scatter(np.full(len(ys), i) + jitter, ys, s=16, alpha=0.5, color="steelblue", zorder=2)
        ax.errorbar(xs, means, yerr=stds, marker="o", color="darkorange", capsize=4, zorder=3, label="mean ± std")
        ax.set_xticks(xs)
        ax.set_xticklabels(condition_labels, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Augmentation sweep: 10 real base problems, expanded pools up to 100, fixed 7-subject valset")
    plt.tight_layout()
    out_path = out_dir / "augmentation_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the augmentation sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sweep_root", nargs="?", default=SWEEP_ROOT, help="Root directory of the sweep")
    parser.add_argument("--plot", action="store_true", help="Save the summary figure")
    parser.add_argument("--specificity", action="store_true", help="Also load cross_spec.txt Δ per run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    grouped = load_grouped(sweep_root, with_specificity=args.specificity)

    n_loaded = sum(len(v) for v in grouped.values())
    print(f"Loaded {n_loaded}/{len(CONDITIONS) * len(SEEDS)} runs.")
    for c in CONDITIONS:
        if len(grouped.get(c, [])) != len(SEEDS):
            print(f"  Warning: {c} has {len(grouped.get(c, []))}/{len(SEEDS)} seeds")

    print_summary_table(grouped, with_specificity=args.specificity)

    print_anova(one_way_anova(grouped, "improvement"), "val accuracy gain", across="conditions")
    print_contrasts(grouped, "improvement", "val accuracy gain")

    print_anova(one_way_anova(grouped, "generalization_rate"), "generalization rate", across="conditions")
    print_contrasts(grouped, "generalization_rate", "generalization rate")

    if args.specificity:
        print_anova(one_way_anova(grouped, "specificity_delta"), "specificity Δ", across="conditions")
        print_contrasts(grouped, "specificity_delta", "specificity Δ")

    if args.plot:
        plot_results(grouped, sweep_root)
    else:
        print("\nRun with --plot to generate the summary figure.")
