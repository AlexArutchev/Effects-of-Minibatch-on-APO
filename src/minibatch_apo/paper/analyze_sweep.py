"""Cross-condition analysis for the minibatch size sweep.

Loads all completed runs from a sweep root, computes per-condition statistics,
prints a summary table, and optionally saves a three-panel figure.

Usage::

    uv run python src/minibatch_apo/paper/analyze_sweep.py runs/minibatch_sweep
    uv run python src/minibatch_apo/paper/analyze_sweep.py runs/minibatch_sweep --plot
"""

import argparse
import json
import math
from pathlib import Path


# ── loading ───────────────────────────────────────────────────────────────────


def load_run(run_dir: Path) -> tuple[dict, list[dict]] | None:
    """Load sweep_meta.json and research_log.jsonl from a run directory.

    Returns None if either file is missing (run not started or incomplete).
    """
    meta_path = run_dir / "sweep_meta.json"
    log_path = run_dir / "research_log.jsonl"
    if not meta_path.exists() or not log_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return meta, records


# ── per-run statistics ────────────────────────────────────────────────────────


def compute_run_stats(meta: dict, records: list[dict]) -> dict:
    m = meta["minibatch_size"]
    val_size = meta["val_size"]

    accepted = [r for r in records if r.get("status") == "accepted"]
    total = len(records)

    acceptance_rate = len(accepted) / total if total > 0 else float("nan")

    # Generalization: accepted candidate improved val score over its parent.
    # parent_score is the parent's aggregate val score (logged by ResearchCallback
    # via on_candidate_selected). valset_score is the new candidate's val score.
    generalizing = [
        r for r in accepted
        if r.get("valset_score") is not None
        and r.get("parent_score") is not None
        and r["valset_score"] > r["parent_score"]
    ]
    generalization_rate = len(generalizing) / len(accepted) if accepted else float("nan")

    # Seed score: parent_score of the first record where parent is candidate 0.
    seed_records = [r for r in records if r.get("parent_candidate_idx") == 0]
    seed_score = seed_records[0]["parent_score"] if seed_records else None

    val_scores = [r["valset_score"] for r in accepted if r.get("valset_score") is not None]
    best_score = max(val_scores) if val_scores else seed_score

    # Build val accuracy trajectory as (cumulative_metric_calls, best_val_so_far).
    # Prefer the logged total_metric_calls (from SweepCallback); fall back to
    # reconstructing from cost model: seed=val_size, each iter=2m, accepted+=val_size.
    has_logged_budget = all("total_metric_calls" in r for r in accepted)

    trajectory: list[tuple[int, float]] = []
    if seed_score is not None:
        trajectory.append((val_size, seed_score))  # baseline after seed eval

    if has_logged_budget and accepted:
        running_best = seed_score if seed_score is not None else 0.0
        for r in sorted(accepted, key=lambda x: x["total_metric_calls"]):
            if r.get("valset_score") is not None:
                running_best = max(running_best, r["valset_score"])
                trajectory.append((r["total_metric_calls"], running_best))
    else:
        # Reconstruction: seed eval consumed val_size calls; each iter costs 2m
        # for minibatch evals plus val_size more if accepted.
        cumulative = val_size
        running_best = seed_score if seed_score is not None else 0.0
        for r in sorted(records, key=lambda x: x.get("iteration", 0)):
            cumulative += 2 * m
            if r.get("status") == "accepted":
                cumulative += val_size
                if r.get("valset_score") is not None:
                    running_best = max(running_best, r["valset_score"])
                    trajectory.append((cumulative, running_best))

    return {
        "minibatch_size": m,
        "seed": meta["seed"],
        "seed_score": seed_score,
        "best_score": best_score,
        "improvement": (best_score - seed_score) if (best_score is not None and seed_score is not None) else None,
        "acceptance_rate": acceptance_rate,
        "generalization_rate": generalization_rate,
        "n_proposals": total,
        "n_accepted": len(accepted),
        "trajectory": trajectory,
    }


# ── summary table ─────────────────────────────────────────────────────────────


def _mean(xs: list) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
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


def print_summary_table(grouped: dict[int, list[dict]]) -> None:
    print(f"\n{'='*85}")
    print("  Minibatch sweep — summary (mean ± std across seeds)")
    print(f"{'='*85}")
    print(
        f"  {'m':>5}  {'n':>3}  {'seed_acc':>10}  {'best_acc':>14}  "
        f"{'Δacc':>12}  {'accept%':>12}  {'gen%':>12}  {'proposals':>9}"
    )
    print(f"  {'─'*5}  {'─'*3}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*9}")

    for m in sorted(grouped):
        runs = grouped[m]
        n = len(runs)

        seed_acc = _mean([r["seed_score"] for r in runs])
        best_acc_mu = _mean([r["best_score"] for r in runs])
        best_acc_sd = _std([r["best_score"] for r in runs])
        imp_mu = _mean([r["improvement"] for r in runs])
        imp_sd = _std([r["improvement"] for r in runs])
        ar_mu = _mean([r["acceptance_rate"] for r in runs])
        ar_sd = _std([r["acceptance_rate"] for r in runs])
        gr_mu = _mean([r["generalization_rate"] for r in runs])
        gr_sd = _std([r["generalization_rate"] for r in runs])
        props = _mean([r["n_proposals"] for r in runs])

        print(
            f"  {m:>5}  {n:>3}  {seed_acc:>10.3f}  "
            f"{_fmt(best_acc_mu, best_acc_sd):>14}  "
            f"{_fmt(imp_mu, imp_sd):>12}  "
            f"{_fmt(ar_mu, ar_sd, pct=True):>12}  "
            f"{_fmt(gr_mu, gr_sd, pct=True):>12}  "
            f"{props:>9.0f}"
        )

    print()
    print("  Columns:")
    print("    seed_acc    — val accuracy of the seed prompt (no optimization)")
    print("    best_acc    — best val accuracy achieved within the 2000-rollout budget")
    print("    Δacc        — best_acc − seed_acc (optimization gain)")
    print("    accept%     — fraction of proposals accepted on the minibatch test")
    print("    gen%        — fraction of accepted proposals that also improved val score")
    print("                  (gen% / accept% ≈ 1 means the acceptance test is reliable)")
    print("    proposals   — total reflective mutation attempts within budget")


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_results(grouped: dict[int, list[dict]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available — skipping plots")
        return

    ms = sorted(grouped)
    x = np.arange(len(ms))
    labels = [str(m) for m in ms]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── Panel 1: Δ accuracy vs m ──────────────────────────────────────────────
    ax = axes[0]
    imp_mus = [_mean([r["improvement"] for r in grouped[m]]) for m in ms]
    imp_sds = [_std([r["improvement"] for r in grouped[m]]) for m in ms]
    imp_sds_clean = [0.0 if math.isnan(s) else s for s in imp_sds]
    bars = ax.bar(x, imp_mus, yerr=imp_sds_clean, capsize=5,
                  color="#4caf50", alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Minibatch size", fontsize=16)
    ax.set_ylabel("Val accuracy gain (Δ = best − seed)", fontsize=16)
    ax.set_title("Optimization gain vs minibatch size", fontsize=17)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Annotate bar tops
    for bar, mu in zip(bars, imp_mus):
        if not math.isnan(mu):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{mu:.3f}", ha="center", va="bottom", fontsize=10)

    # ── Panel 2: Acceptance & generalization rate vs m ────────────────────────
    ax = axes[1]
    width = 0.35
    ar_mus = [_mean([r["acceptance_rate"] for r in grouped[m]]) * 100 for m in ms]
    ar_sds = [_std([r["acceptance_rate"] for r in grouped[m]]) * 100 for m in ms]
    gr_mus = [_mean([r["generalization_rate"] for r in grouped[m]]) * 100 for m in ms]
    gr_sds = [_std([r["generalization_rate"] for r in grouped[m]]) * 100 for m in ms]
    ar_sds_clean = [0.0 if math.isnan(s) else s for s in ar_sds]
    gr_sds_clean = [0.0 if math.isnan(s) else s for s in gr_sds]

    ax.bar(x - width / 2, ar_mus, width, yerr=ar_sds_clean, capsize=4,
           color="#2196f3", alpha=0.85, label="Acceptance rate", edgecolor="white")
    ax.bar(x + width / 2, gr_mus, width, yerr=gr_sds_clean, capsize=4,
           color="#ff9800", alpha=0.85, label="Generalization rate", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Minibatch size", fontsize=16)
    ax.set_ylabel("Rate (%)", fontsize=16)
    ax.set_title("Acceptance vs generalization rate", fontsize=17)
    ax.legend(fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # ── Panel 3: Val accuracy trajectories by m ───────────────────────────────
    ax = axes[2]
    budget_max = 2000
    grid = np.linspace(0, budget_max, 400)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ms)))

    for m, color in zip(ms, colors):
        runs_with_traj = [r for r in grouped[m] if len(r["trajectory"]) >= 1]
        if not runs_with_traj:
            continue

        interp_trajs = []
        for r in runs_with_traj:
            traj = r["trajectory"]
            # Step-function: at budget B, accuracy = last accepted val score before B
            # Build duplicate points to force step-function shape via np.interp
            xs_raw = [t[0] for t in traj]
            ys_raw = [t[1] for t in traj]
            # Extend to budget_max with the final value
            xs_step = [0.0]
            ys_step = [ys_raw[0]]
            for bx, by in zip(xs_raw, ys_raw):
                xs_step.append(bx - 1e-6)
                xs_step.append(bx)
                ys_step.append(ys_step[-1])
                ys_step.append(by)
            xs_step.append(budget_max)
            ys_step.append(ys_step[-1])
            interp_trajs.append(np.interp(grid, xs_step, ys_step))

        mean_traj = np.mean(interp_trajs, axis=0)
        ax.plot(grid, mean_traj, label=f"m={m}", color=color, linewidth=1.8)
        if len(interp_trajs) > 1:
            std_traj = np.std(interp_trajs, axis=0)
            ax.fill_between(grid, mean_traj - std_traj, mean_traj + std_traj,
                            alpha=0.12, color=color)

    ax.set_xlabel("Cumulative metric calls (rollouts)", fontsize=16)
    ax.set_ylabel("Best val accuracy so far", fontsize=16)
    ax.set_title("Val accuracy trajectory by minibatch size", fontsize=17)
    ax.legend(fontsize=12, loc="lower right")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0, budget_max)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = out_dir / "minibatch_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")

    # Raw summary JSON for downstream use
    summary = {
        str(m): [
            {k: v for k, v in r.items() if k != "trajectory"}
            for r in grouped[m]
        ]
        for m in ms
    }
    json_path = out_dir / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze results from the minibatch size sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sweep_root",
        help="Root directory of the sweep (e.g. runs/minibatch_sweep)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a three-panel figure to sweep_root/minibatch_sweep.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep_root = Path(args.sweep_root)

    if not sweep_root.exists():
        raise SystemExit(f"Sweep root not found: {sweep_root}")

    grouped: dict[int, list[dict]] = {}
    skipped = []
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        result = load_run(run_dir)
        if result is None:
            skipped.append(run_dir.name)
            continue
        meta, records = result
        if not records:
            print(f"  Warning: {run_dir.name} has no records yet — skipping")
            continue
        stats = compute_run_stats(meta, records)
        grouped.setdefault(meta["minibatch_size"], []).append(stats)

    if skipped:
        print(f"Skipped (no sweep_meta.json or research_log.jsonl): {skipped}")

    if not grouped:
        raise SystemExit(f"No completed runs found in {sweep_root}")

    print(f"\nLoaded runs for minibatch sizes: {sorted(grouped)}")
    for m, runs in sorted(grouped.items()):
        seeds_done = [r["seed"] for r in runs]
        print(f"  m={m:>3}: {len(runs)} seed(s) — seeds {seeds_done}")

    print_summary_table(grouped)

    if args.plot:
        plot_results(grouped, sweep_root)
    else:
        print("Run with --plot to generate the figure.")
