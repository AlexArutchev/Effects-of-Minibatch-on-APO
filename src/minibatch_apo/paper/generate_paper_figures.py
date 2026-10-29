"""Generate reproducible figures for the minibatch prompt-optimization paper.

The script writes paper-ready PDF and PNG versions of a methodology diagram and
the fixed-pool deconfounding result. It also copies the two existing results plots
used by the manuscript into ``figures`` so the paper does not depend on the
ignored ``runs`` directory at build time.

Usage:
    uv run python src/minibatch_apo/paper/generate_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from minibatch_apo.paper.analyze_augmentation import load_grouped

PAPER_FIGURE_DIR = Path("figures")
DECONFOUND_CONDITIONS = ("real100_m10", "real100")


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    return outputs


def build_methodology_pipeline(output_dir: Path) -> list[Path]:
    """Draw the GEPA minibatch-to-update pipeline used in every experiment."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.2, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.8)
    ax.axis("off")

    boxes = [
        (0.25, "Sample $m$ MATH\nproblems", "#DCEAF7"),
        (2.65, "Task LM produces\nanswers and traces", "#E8F3E8"),
        (5.05, "Reflection LM writes\na revised prompt", "#FCE8CC"),
        (7.45, "Evaluate candidate;\nupdate Pareto set", "#E9E3F5"),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.85),
            1.95,
            0.95,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#4A5568",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + 0.975, 1.325, label, ha="center", va="center", fontsize=13)

    arrow_kwargs = {"arrowstyle": "-|>", "mutation_scale": 15, "linewidth": 1.2, "color": "#4A5568"}
    for x in (2.22, 4.62, 7.02):
        ax.add_patch(FancyArrowPatch((x, 1.325), (x + 0.38, 1.325), **arrow_kwargs))

    ax.text(1.225, 2.24, "fresh reflection minibatch", ha="center", va="center", fontsize=11, color="#2C5282")
    ax.text(6.025, 2.24, "textual update", ha="center", va="center", fontsize=11, color="#9C4221")
    ax.text(8.425, 2.24, "held-out transfer", ha="center", va="center", fontsize=11, color="#553C9A")
    ax.text(
        5,
        0.23,
        "We vary the number of distinct problems in the first box and measure the specificity and transfer of the resulting updates.",
        ha="center",
        va="center",
        fontsize=11,
    )
    return _save_figure(fig, output_dir, "methodology_pipeline")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def build_deconfound_figure(grouped: Mapping[str, Sequence[Mapping[str, float]]], output_dir: Path) -> list[Path]:
    """Plot the matched real100 contrast that isolates reflection minibatch size."""
    import matplotlib.pyplot as plt
    import numpy as np

    missing = [condition for condition in DECONFOUND_CONDITIONS if not grouped.get(condition)]
    if missing:
        raise ValueError(f"Missing deconfound conditions: {', '.join(missing)}")

    labels = ["$m=10$\n(real100-m10)", "$m=50$\n(real100)"]
    panels = [
        ("generalization_rate", "Generalization rate (%)", 100.0, "#2B6CB0"),
        ("specificity_delta", "Specificity gap $\\Delta_{\\mathrm{spec}}$", 1.0, "#C05621"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    rng = np.random.default_rng(0)
    xs = np.arange(len(DECONFOUND_CONDITIONS))

    for metric, ylabel, scale, color in panels:
        ax = axes[panels.index((metric, ylabel, scale, color))]
        means: list[float] = []
        stds: list[float] = []
        for x, condition in enumerate(DECONFOUND_CONDITIONS):
            values = [float(row[metric]) * scale for row in grouped[condition] if row.get(metric) is not None]
            means.append(_mean(values))
            stds.append(_sample_std(values))
            jitter = rng.uniform(-0.075, 0.075, size=len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, color=color, alpha=0.45, s=24, zorder=2)
        ax.errorbar(xs, means, yerr=stds, color=color, marker="o", capsize=4, linewidth=1.7, zorder=3)
        ax.set_xticks(xs, labels, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel(ylabel, fontsize=19)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_title("Generalization rate", fontsize=20)
    axes[1].set_title("Specificity gap", fontsize=20)
    fig.suptitle("Fixed 100-problem pool", y=1.02, fontsize=22)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fixed_pool_deconfound")


def copy_existing_results_figures(project_root: Path, output_dir: Path) -> list[Path]:
    """Copy existing reproducible study plots into the paper asset directory."""
    sources = {
        project_root / "runs/minibatch_sweep/minibatch_sweep.png": output_dir / "minibatch_sweep.png",
        project_root / "rule_survival.png": output_dir / "rule_survival.png",
        project_root / "runs/diversity_sweep/diversity_sweep.png": output_dir / "diversity_sweep.png",
        project_root / "runs/factorial_sweep/factorial_interaction.png": output_dir / "factorial_interaction.png",
        project_root / "runs/augmentation_sweep/augmentation_sweep.png": output_dir / "augmentation_sweep.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source, destination in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def build_oracle_pruning_figure(project_root: Path, output_dir: Path, ablation_root: Path | None = None) -> list[Path]:
    """Plot ablation results, requiring all five seeds per condition."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ablation_root is None:
        ablation_root = project_root / "outputs/rule_ablations"
    rows: dict[str, list[dict[str, float]]] = {"real100": [], "real100_m10": []}
    for condition in rows:
        for seed in range(5):
            path = ablation_root / f"{condition}_s{seed}" / "oracle_prune.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows[condition].append(json.loads(path.read_text()))

    labels = ["$m=50$\n(real100)", "$m=10$\n(real100-m10)"]
    predicted = [
        np.mean([row["sum_neg_estimate"] for row in rows[condition]]) for condition in ("real100", "real100_m10")
    ]
    observed = [np.mean([row["oracle_gain"] for row in rows[condition]]) for condition in ("real100", "real100_m10")]
    observed_std = [
        np.std([row["oracle_gain"] for row in rows[condition]], ddof=1) for condition in ("real100", "real100_m10")
    ]

    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, predicted, width, label="Sum of selected negative utilities", color="#8FB9E3")
    ax.bar(
        x + width / 2,
        observed,
        width,
        yerr=observed_std,
        capsize=4,
        label="Observed joint pruning gain",
        color="#D9864A",
    )
    for i, condition in enumerate(("real100", "real100_m10")):
        ax.scatter(
            i + width / 2 + np.linspace(-0.06, 0.06, 5),
            [row["oracle_gain"] for row in rows[condition]],
            s=16,
            color="#333333",
            zorder=4,
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation/Pareto accuracy gain")
    ax.set_title("Oracle-pruning diagnostic")
    ax.set_ylim(min(-0.085, min(observed) - max(observed_std) - 0.02), max(predicted) + 0.085)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "oracle_pruning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER_FIGURE_DIR)
    parser.add_argument("--sweep-root", type=Path, default=Path("runs/augmentation_sweep"))
    parser.add_argument("--ablation-root", type=Path, default=Path("outputs/rule_ablations"))
    args = parser.parse_args()

    outputs = build_methodology_pipeline(args.output_dir)
    grouped = load_grouped(args.sweep_root, with_specificity=True)
    outputs.extend(build_deconfound_figure(grouped, args.output_dir))
    outputs.extend(copy_existing_results_figures(Path.cwd(), args.output_dir))
    outputs.extend(build_oracle_pruning_figure(Path.cwd(), args.output_dir, args.ablation_root))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
