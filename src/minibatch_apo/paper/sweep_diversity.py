"""Paper-specific sweep diversity protocol and helpers."""

import argparse
import json
import os
import random

import gepa
from gepa.core.callbacks import BudgetUpdatedEvent, CandidateAcceptedEvent, CandidateRejectedEvent
from minibatch_apo.paper.math_competition import extract_boxed_answer
from minibatch_apo.paper.research_callback import ResearchCallback

# ── experiment constants ──────────────────────────────────────────────────────

TASK_LM = "openai/gpt-4.1-mini"
REFLECTION_LM = "openai/gpt-4.1"
MAX_ROLLOUTS = 1000
TRAIN_SIZE = 100
VAL_SIZE = 100
LEVELS = [4, 5]
MINIBATCH_SIZE = 10

DIVERSITY_LEVELS = [1, 2, 3, 4, 5, 6, 7]

# Uneven seed budget: existing levels (already had 5 seeds from the factorial
# sweep) get 5 more to firm up the diversity=4 anomaly; new intermediate
# levels get 5 seeds to establish the curve shape. See module docstring.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS_BY_LEVEL: dict[int, list[int]] = {
    1: SEEDS,
    2: SEEDS,
    3: SEEDS[:5],
    4: SEEDS,
    5: SEEDS[:5],
    6: SEEDS[:5],
    7: SEEDS,
}

# Fixed seed for valset construction. Never changes — same valset for all conditions.
DATASET_SEED = 42

# All 7 MATH subjects at levels 4–5.
ALL_SUBJECTS = [
    "Algebra",
    "Number Theory",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Prealgebra",
    "Precalculus",
]

# Which subjects each diversity level uses for training.
# D1 uses the single most common subject (Algebra) so we have plenty of examples.
# D4 picks 4 conceptually distinct subjects. D7 uses all.
SUBJECTS_BY_LEVEL: dict[int, list[str]] = {
    1: ["Algebra"],
    2: ["Algebra", "Number Theory"],
    3: ["Algebra", "Number Theory", "Counting & Probability"],
    4: ["Algebra", "Number Theory", "Counting & Probability", "Geometry"],
    5: ["Algebra", "Number Theory", "Counting & Probability", "Geometry", "Intermediate Algebra"],
    6: [
        "Algebra",
        "Number Theory",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Prealgebra",
    ],
    7: ALL_SUBJECTS,
}

SEED_PROMPT = {
    "system_prompt": (
        "You are a helpful assistant. Answer the question. "
        "Put your final answer in the format '### <answer>'"
    )
}


# ── dataset builder ───────────────────────────────────────────────────────────


def _load_math_by_subject(levels: list[int]) -> dict[str, list[dict]]:
    """Load MATH (train + test split) and group examples by subject.

    Filters to the given difficulty levels and drops any example where the
    boxed answer cannot be extracted. Returns a dict mapping each subject
    in ALL_SUBJECTS to a flat list of GEPA-format examples.
    """
    from datasets import load_dataset

    ds = load_dataset("nlile/hendrycks-MATH-benchmark")

    def keep(ex: dict) -> bool:
        return ex["level"] in levels and extract_boxed_answer(ex["solution"]) is not None

    def to_example(x: dict) -> dict:
        return {
            "input": x["problem"],
            "additional_context": {
                "solution": x["solution"],
                "subject": x["subject"],
                "level": x["level"],
            },
            "answer": "### " + extract_boxed_answer(x["solution"]),
        }

    by_subject: dict[str, list[dict]] = {s: [] for s in ALL_SUBJECTS}
    for split in ("train", "test"):
        for x in ds[split]:
            if keep(x) and x["subject"] in by_subject:
                by_subject[x["subject"]].append(to_example(x))
    return by_subject


def _per_subject_counts(total: int, n_subjects: int) -> list[int]:
    """Divide *total* as evenly as possible across *n_subjects*.

    Returns a list of length *n_subjects* that sums to *total*.
    The remainder is distributed one-per-subject from the front of the list.

    Example: total=100, n_subjects=7 → [15, 15, 14, 14, 14, 14, 14]  (100 total)
    """
    base, rem = divmod(total, n_subjects)
    return [base + (1 if i < rem else 0) for i in range(n_subjects)]


def build_datasets(
    num_types: int,
    run_seed: int,
    levels: list[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build (trainset, valset) for the given diversity level.

    Valset is always the same 100 problems regardless of *num_types*.
    It is stratified across all 7 MATH subjects and built using DATASET_SEED,
    so every call with any *num_types* gets the identical held-out set.

    Trainset uses only the subjects for *num_types* (see SUBJECTS_BY_LEVEL).
    Examples are drawn from the per-subject pool that remains after the valset
    reservation, so there is no train/val overlap.

    Args:
        num_types: Number of training subject categories (1, 2, 4, or 7).
        run_seed: Seed for shuffling the training pool (varies per GEPA run).
        levels: MATH difficulty levels to include. Defaults to [4, 5].

    Returns:
        (trainset, valset) — valset is identical across all conditions.

    Raises:
        ValueError: if a subject does not have enough examples after reserving
            the valset. This should not happen for levels [4, 5] with the
            chosen subjects, but the check guards against future config changes.
    """
    if levels is None:
        levels = LEVELS

    by_subject = _load_math_by_subject(levels)

    # ── Step 1: build the fixed valset with DATASET_SEED ─────────────────────
    # Shuffle each subject pool with a locked seed, then take the first
    # val_counts[i] examples as the valset for subject i. These are excluded
    # from the training pool for ALL conditions.
    val_counts = _per_subject_counts(VAL_SIZE, len(ALL_SUBJECTS))
    valset: list[dict] = []
    val_indices_by_subject: dict[str, set[int]] = {}

    val_rng = random.Random(DATASET_SEED)
    for subject, n_val in zip(ALL_SUBJECTS, val_counts):
        pool = by_subject[subject]
        indices = list(range(len(pool)))
        val_rng.shuffle(indices)
        chosen_indices = set(indices[:n_val])
        val_indices_by_subject[subject] = chosen_indices
        valset.extend(pool[i] for i in sorted(chosen_indices))

    val_rng.shuffle(valset)  # mix subjects in the valset (cosmetic)

    # ── Step 2: build the training set from the remaining pool ────────────────
    subjects_to_use = SUBJECTS_BY_LEVEL[num_types]
    train_counts = _per_subject_counts(TRAIN_SIZE, len(subjects_to_use))

    train_rng = random.Random(run_seed)
    trainset: list[dict] = []
    for subject, n_train in zip(subjects_to_use, train_counts):
        pool = by_subject[subject]
        remaining = [i for i in range(len(pool)) if i not in val_indices_by_subject.get(subject, set())]
        if len(remaining) < n_train:
            raise ValueError(
                f"Subject '{subject}' has only {len(remaining)} examples left after "
                f"valset reservation; need {n_train} for training. "
                f"Try reducing TRAIN_SIZE or VAL_SIZE."
            )
        train_rng.shuffle(remaining)
        trainset.extend(pool[i] for i in remaining[:n_train])

    train_rng.shuffle(trainset)  # mix subjects in the trainset
    return trainset, valset


def check_availability(levels: list[int] | None = None) -> None:
    """Print per-subject example counts to verify the dataset has enough data.

    Run this before the sweep if you are unsure whether all subjects have
    enough examples for your TRAIN_SIZE / VAL_SIZE settings.
    """
    if levels is None:
        levels = LEVELS
    print(f"Checking MATH example counts at levels {levels}...")
    by_subject = _load_math_by_subject(levels)
    val_counts = dict(zip(ALL_SUBJECTS, _per_subject_counts(VAL_SIZE, len(ALL_SUBJECTS))))

    print(f"\n  {'Subject':<30}  {'Total':>7}  {'Reserved for val':>17}  {'Available for train':>20}")
    print(f"  {'─'*30}  {'─'*7}  {'─'*17}  {'─'*20}")
    for subject in ALL_SUBJECTS:
        total = len(by_subject[subject])
        n_val = val_counts[subject]
        n_train = total - n_val
        flag = "  ← WARNING: < 25 for training" if n_train < 25 else ""
        print(f"  {subject:<30}  {total:>7}  {n_val:>17}  {n_train:>20}{flag}")
    print()


# ── callback ──────────────────────────────────────────────────────────────────


class SweepCallback(ResearchCallback):
    """Extends ResearchCallback to log cumulative metric calls per record.

    Adds ``total_metric_calls`` to every flushed record so the analysis script
    can plot val accuracy trajectories against budget rather than iteration count.
    """

    def __init__(self, run_dir: str) -> None:
        super().__init__(run_dir)
        self._current_metric_calls: int = 0

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        self._current_metric_calls = event["metric_calls_used"]

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._get(event["iteration"])["total_metric_calls"] = self._current_metric_calls
        super().on_candidate_accepted(event)

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self._get(event["iteration"])["total_metric_calls"] = self._current_metric_calls
        super().on_candidate_rejected(event)


# ── runner ────────────────────────────────────────────────────────────────────


def is_complete(run_dir: str) -> bool:
    """A run is only complete if it has both a state checkpoint and a flushed log.

    ``gepa_state.bin`` alone is not sufficient: GEPA checkpoints it early, so a
    run that crashed or was killed mid-iteration (before ``research_log.jsonl``
    was ever written) would otherwise be silently treated as done.
    """
    return os.path.exists(os.path.join(run_dir, "gepa_state.bin")) and os.path.exists(
        os.path.join(run_dir, "research_log.jsonl")
    )


def run_condition(num_types: int, seed: int, sweep_root: str) -> None:
    """Run one (diversity_level, seed) condition and write results to disk."""
    label = f"d{num_types}_s{seed}"
    run_dir = os.path.join(sweep_root, label)

    if is_complete(run_dir):
        print(f"[{label}] Already complete — skipping (delete {run_dir} to rerun)")
        return

    subjects = SUBJECTS_BY_LEVEL[num_types]
    print(f"\n{'='*60}")
    print(f"  Condition : diversity={num_types} subject(s)  seed={seed}")
    print(f"  Subjects  : {', '.join(subjects)}")
    print(f"  Run dir   : {run_dir}")
    print(f"{'='*60}\n")

    print("Building datasets...")
    trainset, valset = build_datasets(num_types=num_types, run_seed=seed)

    # Compute subject breakdowns for metadata
    train_subject_counts: dict[str, int] = {}
    for ex in trainset:
        s = ex.get("additional_context", {}).get("subject", "unknown")
        train_subject_counts[s] = train_subject_counts.get(s, 0) + 1

    val_subject_counts: dict[str, int] = {}
    for ex in valset:
        s = ex.get("additional_context", {}).get("subject", "unknown")
        val_subject_counts[s] = val_subject_counts.get(s, 0) + 1

    print(f"  Train : {len(trainset)} examples — {train_subject_counts}")
    print(f"  Val   : {len(valset)} examples — {val_subject_counts}")

    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "num_types": num_types,
        "subjects": subjects,
        "seed": seed,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "max_rollouts": MAX_ROLLOUTS,
        "minibatch_size": MINIBATCH_SIZE,
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "levels": LEVELS,
        "dataset_seed": DATASET_SEED,
        "train_subject_counts": train_subject_counts,
        "val_subject_counts": val_subject_counts,
    }
    with open(os.path.join(run_dir, "sweep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=trainset,
        valset=valset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_ROLLOUTS,
        run_dir=run_dir,
        reflection_minibatch_size=MINIBATCH_SIZE,
        callbacks=[SweepCallback(run_dir)],
        seed=seed,
        track_best_outputs=False,
    )

    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\n[{label}] Done.")
    print(f"  Seed score : {seed_score:.4f}")
    print(f"  Best score : {best_score:.4f}")
    print(f"  Δ          : {best_score - seed_score:+.4f}")
    print(f"  Rollouts   : {result.total_metric_calls}")
    print(f"  Candidates : {result.num_candidates}")


# ── smoke test ────────────────────────────────────────────────────────────────


def run_test(sweep_root: str) -> None:
    """Smoke test: one tiny D1 run to confirm API key and pipeline (~$0.50, ~2 min)."""
    import shutil

    run_dir = os.path.join(sweep_root, "test_run")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    print("Running smoke test (diversity=1, 20 train, 20 val, 50 rollouts)...")
    trainset, valset = build_datasets(num_types=1, run_seed=0)
    trainset = trainset[:20]
    valset = valset[:20]
    print(f"  Train: {len(trainset)} | Val: {len(valset)}")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sweep_meta.json"), "w") as f:
        json.dump({"num_types": 1, "subjects": ["Algebra"], "seed": 0, "test": True}, f, indent=2)

    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=trainset,
        valset=valset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=50,
        run_dir=run_dir,
        reflection_minibatch_size=MINIBATCH_SIZE,
        callbacks=[SweepCallback(run_dir)],
        seed=0,
        track_best_outputs=False,
    )

    print("\nSmoke test passed.")
    print(f"  Seed score : {result.val_aggregate_scores[0]:.4f}")
    print(f"  Best score : {result.val_aggregate_scores[result.best_idx]:.4f}")
    print(f"  Rollouts   : {result.total_metric_calls}")
    print(f"\nRun the full sweep:")
    print(f"  uv run python src/minibatch_apo/paper/sweep_diversity.py --all")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled type-diversity sweep on Hendrycks MATH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-root",
        default="runs/diversity_sweep",
        help="Root directory for all sweep runs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all (diversity_level, seed) conditions sequentially",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test: verify API key and pipeline (~$0.50)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print per-subject example counts and exit (no runs)",
    )
    parser.add_argument(
        "--diversity",
        type=int,
        choices=DIVERSITY_LEVELS,
        help="Number of MATH subject types for a single run (1-7)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for a single run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check:
        check_availability()
    elif args.test:
        run_test(args.sweep_root)
    elif args.all:
        conditions = [(d, s) for d in DIVERSITY_LEVELS for s in SEEDS_BY_LEVEL[d]]
        pending = [(d, s) for d, s in conditions if not is_complete(os.path.join(args.sweep_root, f"d{d}_s{s}"))]
        print(f"{len(conditions) - len(pending)}/{len(conditions)} conditions already complete.")
        print(f"Running {len(pending)} pending conditions (~${len(pending) * 1:.0f}-${len(pending) * 1.5:.0f})")
        for d, s in pending:
            run_condition(d, s, args.sweep_root)
        print(f"\nAll {len(conditions)} conditions complete. Analyze with:")
        print(f"  uv run python src/minibatch_apo/paper/analyze_diversity_sweep.py {args.sweep_root} --plot")
    elif args.diversity is not None:
        run_condition(args.diversity, args.seed, args.sweep_root)
    else:
        parse_args()  # triggers --help
