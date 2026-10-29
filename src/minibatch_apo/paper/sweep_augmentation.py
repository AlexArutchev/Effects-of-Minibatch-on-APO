"""Paper-specific sweep augmentation protocol and helpers."""

import argparse
import json
import os
import random

import gepa
from minibatch_apo.paper.math_competition import extract_boxed_answer
from minibatch_apo.paper.sweep_diversity import (
    MAX_ROLLOUTS,
    REFLECTION_LM,
    SEED_PROMPT,
    TASK_LM,
    VAL_SIZE,
    SweepCallback,
    build_datasets,
    is_complete,
)

# real100_m10 = the 100-real-problem pool run at minibatch=10 (instead of real100's
# 50). It deconfounds pool size from minibatch size in the duplicate-baseline
# contrast: real100 vs real100_m10 isolates the minibatch effect at a fixed real
# pool; baseline vs real100_m10 isolates the pool effect at a fixed minibatch=10.
CONDITIONS = ["baseline", "duplicate", "paraphrase", "synthetic", "real100", "real100_m10"]
SEEDS = list(range(10))  # was [0..4]; confirmatory round adds seeds 5-9

N_REAL = 10
POOL_SIZE = 100
MINIBATCH_BASELINE = 10  # = N_REAL: the minibatch is the whole training set
MINIBATCH_AUGMENTED = 50

SWEEP_ROOT = "runs/augmentation_sweep"
DATA_DIR = os.path.join(SWEEP_ROOT, "data")
CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")

# Over-generation for deferred (Claude-side) verification: some candidates will
# fail the blind solve, so generate spares to avoid a second OpenAI round.
PARA_OVERGEN = 12  # candidates per problem (need 9 verified)
SYN_OVERGEN = 120  # synthetic candidates (need 90 verified)

# Verification retries per needed item before giving up on a generation pass.
MAX_ATTEMPT_FACTOR = 3


# ── base pools ────────────────────────────────────────────────────────────────


def build_pools(run_seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (seed10, real100, valset) for this run seed.

    real100 is the standard diversity-sweep trainset at num_types=7 (100 real
    problems, stratified over all subjects, shuffled by run_seed). seed10 is its
    first 10 problems, so baseline/real100 are a matched pair within each seed.
    The valset is the fixed DATASET_SEED valset shared with all prior sweeps.
    """
    real100, valset = build_datasets(num_types=7, run_seed=run_seed)
    return real100[:N_REAL], real100, valset


# ── answer verification ───────────────────────────────────────────────────────


def _norm(ans: str | None) -> str:
    return "".join(ans.split()) if ans else ""


def _intended_answer(example: dict) -> str:
    """The ground-truth answer string of a GEPA example (strips the '### ' prefix)."""
    return example["answer"].removeprefix("### ").strip()


def _verify_problem(problem_text: str, intended_answer: str, solver) -> bool:
    """Independently solve *problem_text* and check the answer matches."""
    response = solver(
        "Solve the following math problem. Show your reasoning, then give your "
        "final answer inside \\boxed{}.\n\nProblem:\n" + problem_text
    )
    solved = extract_boxed_answer(response)
    return _norm(solved) == _norm(intended_answer) and _norm(solved) != ""


# ── paraphrase generation ─────────────────────────────────────────────────────

PARAPHRASE_PROMPT = """Rewrite the following math problem with different wording, sentence structure, \
and (where natural) surface context, while keeping the mathematical content EXACTLY the same: \
identical quantities, identical constraints, identical answer. Do not simplify, do not add hints, \
do not change any numbers.

Output ONLY the rewritten problem text, nothing else.

Problem:
{problem}"""


def _generate_paraphrases(seed10: list[dict], run_seed: int, lm) -> list[dict]:
    """Return a pool of POOL_SIZE examples: the 10 originals + 9 verified paraphrases each."""
    per_problem = POOL_SIZE // N_REAL - 1  # 9 paraphrases per original
    pool: list[dict] = list(seed10)
    for i, ex in enumerate(seed10):
        intended = _intended_answer(ex)
        kept = 0
        attempts = 0
        while kept < per_problem and attempts < per_problem * MAX_ATTEMPT_FACTOR:
            attempts += 1
            text = lm(PARAPHRASE_PROMPT.format(problem=ex["input"])).strip()
            if not text or not _verify_problem(text, intended, lm):
                continue
            pool.append(
                {
                    "input": text,
                    "answer": ex["answer"],
                    "additional_context": {
                        **ex.get("additional_context", {}),
                        "augmentation": "paraphrase",
                        "source_input": ex["input"],
                    },
                }
            )
            kept += 1
        print(f"  problem {i + 1}/{N_REAL}: kept {kept}/{per_problem} paraphrases ({attempts} attempts)")
        if kept < per_problem:
            # Fall back to duplicates of the original so the pool size stays fixed.
            pool.extend([dict(ex) for _ in range(per_problem - kept)])
    random.Random(run_seed).shuffle(pool)
    return pool


# ── synthetic generation ──────────────────────────────────────────────────────

SYNTHETIC_PROMPT = """Here are some competition math problems (difficulty: MATH levels 4-5):

{examples}

Write ONE new, original problem of similar difficulty and style (any topic among the examples' \
topics is fine), followed by a complete worked solution that ends with the final answer inside \\boxed{{}}.

Use exactly this format:

PROBLEM:
<problem text>

SOLUTION:
<worked solution ending in \\boxed{{answer}}>"""


def _generate_synthetic(seed10: list[dict], run_seed: int, lm) -> list[dict]:
    """Return a pool of POOL_SIZE examples: the 10 originals + verified synthesized problems."""
    needed = POOL_SIZE - N_REAL
    rng = random.Random(run_seed)
    pool: list[dict] = list(seed10)
    kept = 0
    attempts = 0
    while kept < needed and attempts < needed * MAX_ATTEMPT_FACTOR:
        attempts += 1
        shots = rng.sample(seed10, 3)
        examples = "\n\n".join(f"Problem {j + 1}:\n{s['input']}" for j, s in enumerate(shots))
        response = lm(SYNTHETIC_PROMPT.format(examples=examples))
        if "PROBLEM:" not in response or "SOLUTION:" not in response:
            continue
        problem = response.split("PROBLEM:", 1)[1].split("SOLUTION:", 1)[0].strip()
        solution = response.split("SOLUTION:", 1)[1].strip()
        answer = extract_boxed_answer(solution)
        if not problem or not answer:
            continue
        # Independent re-solve must reproduce the generator's own answer.
        if not _verify_problem(problem, answer, lm):
            continue
        pool.append(
            {
                "input": problem,
                "answer": "### " + answer,
                "additional_context": {
                    "solution": solution,
                    "subject": "synthetic",
                    "level": "synthetic",
                    "augmentation": "synthetic",
                },
            }
        )
        kept += 1
        if kept % 10 == 0:
            print(f"  {kept}/{needed} synthetic problems verified ({attempts} attempts)")
    if kept < needed:
        print(f"  WARNING: only {kept}/{needed} verified; padding with duplicates of the originals")
        pool.extend(dict(rng.choice(seed10)) for _ in range(needed - kept))
    rng.shuffle(pool)
    return pool


# ── deferred (Claude-side) verification ──────────────────────────────────────
#
# --gen-data --skip-verify generates candidates WITHOUT the OpenAI verification
# solves and writes, per (seed, arm):
#   candidates/seed{s}_{arm}_candidates.json — full records incl. intended answers
#   candidates/seed{s}_{arm}_problems.json   — [{id, problem}] only, for a blind solver
# An external solver (e.g. a Claude agent) then writes:
#   candidates/seed{s}_{arm}_solutions.json  — {"<id>": "<final answer>"}
# --finalize compares solved vs intended answers and assembles the standard
# verified pools. The solver file carries no answers, so the check stays blind.


def _candidate_paths(condition: str, run_seed: int) -> tuple[str, str, str]:
    base = os.path.join(CANDIDATES_DIR, f"seed{run_seed}_{condition}")
    return base + "_candidates.json", base + "_problems.json", base + "_solutions.json"


def _generate_paraphrase_candidates(seed10: list[dict], lm) -> list[dict]:
    candidates: list[dict] = []
    for i, ex in enumerate(seed10):
        made = 0
        for _ in range(PARA_OVERGEN):
            text = lm(PARAPHRASE_PROMPT.format(problem=ex["input"])).strip()
            if text:
                candidates.append(
                    {"id": len(candidates), "source_idx": i, "input": text, "intended_answer": _intended_answer(ex)}
                )
                made += 1
        print(f"  problem {i + 1}/{N_REAL}: {made} paraphrase candidates")
    return candidates


def _generate_synthetic_candidates(
    seed10: list[dict], run_seed: int, lm, n_target: int = SYN_OVERGEN, id_start: int = 0
) -> list[dict]:
    rng = random.Random(run_seed + id_start)  # offset so top-up rounds draw fresh few-shot combos
    candidates: list[dict] = []
    attempts = 0
    while len(candidates) < n_target and attempts < n_target * 2:
        attempts += 1
        shots = rng.sample(seed10, 3)
        examples = "\n\n".join(f"Problem {j + 1}:\n{s['input']}" for j, s in enumerate(shots))
        response = lm(SYNTHETIC_PROMPT.format(examples=examples))
        if "PROBLEM:" not in response or "SOLUTION:" not in response:
            continue
        problem = response.split("PROBLEM:", 1)[1].split("SOLUTION:", 1)[0].strip()
        solution = response.split("SOLUTION:", 1)[1].strip()
        answer = extract_boxed_answer(solution)
        if not problem or not answer:
            continue
        candidates.append(
            {"id": id_start + len(candidates), "input": problem, "intended_answer": answer, "solution": solution}
        )
        if len(candidates) % 20 == 0:
            print(f"  {len(candidates)}/{n_target} synthetic candidates ({attempts} attempts)")
    return candidates


def top_up(condition: str, run_seed: int, n: int) -> None:
    """Append *n* fresh candidates to an existing candidates file (synthetic only).

    Used when the verified yield fell short of the pool quota. New candidates get
    ids continuing after the existing ones; the blind solver only needs to solve
    the ids missing from the solutions file. Delete the finalized pool for this
    (seed, condition) before re-running --finalize.
    """
    if condition != "synthetic":
        raise SystemExit("--top-up currently supports only --condition synthetic")
    cand_path, _, _ = _candidate_paths(condition, run_seed)
    if not os.path.exists(cand_path):
        raise SystemExit(f"No existing candidates file at {cand_path}; run --gen-data --skip-verify first")

    from gepa.lm import LM

    with open(cand_path) as f:
        existing = json.load(f)
    print(f"[seed {run_seed}] {condition}: {len(existing)} existing candidates; generating {n} more...")
    seed10, _, _ = build_pools(run_seed)
    new = _generate_synthetic_candidates(seed10, run_seed, LM(REFLECTION_LM), n_target=n, id_start=len(existing))
    _write_candidates(condition, run_seed, existing + new)
    print(f"  {len(new)} new candidates (ids {len(existing)}-{len(existing) + len(new) - 1}).")
    print("  Next: blind-solve the new ids, then delete the finalized pool and re-run --finalize.")


def _write_candidates(condition: str, run_seed: int, candidates: list[dict]) -> None:
    cand_path, prob_path, _ = _candidate_paths(condition, run_seed)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    with open(cand_path, "w") as f:
        json.dump(candidates, f, indent=2)
    with open(prob_path, "w") as f:
        json.dump([{"id": c["id"], "problem": c["input"]} for c in candidates], f, indent=2)
    print(f"  wrote {len(candidates)} candidates -> {cand_path}")
    print(f"  blind problems file           -> {prob_path}")


def finalize(seeds: list[int]) -> None:
    """Assemble verified pools from candidates + externally produced solutions."""
    for s in seeds:
        seed10, _, _ = build_pools(s)
        rng = random.Random(s)
        for condition in ("paraphrase", "synthetic"):
            final_path = _cache_path(condition, s)
            if os.path.exists(final_path):
                print(f"[seed {s}] {condition}: final pool already exists ({final_path})")
                continue
            cand_path, _, sol_path = _candidate_paths(condition, s)
            if not (os.path.exists(cand_path) and os.path.exists(sol_path)):
                print(f"[seed {s}] {condition}: missing candidates or solutions file — skipping")
                continue
            with open(cand_path) as f:
                candidates = json.load(f)
            with open(sol_path) as f:
                solutions = json.load(f)

            verified = [
                c
                for c in candidates
                if _norm(solutions.get(str(c["id"]))) == _norm(c["intended_answer"]) and _norm(c["intended_answer"])
            ]
            print(f"[seed {s}] {condition}: {len(verified)}/{len(candidates)} candidates verified")

            pool: list[dict] = list(seed10)
            if condition == "paraphrase":
                per_problem = POOL_SIZE // N_REAL - 1
                for i, ex in enumerate(seed10):
                    mine = [c for c in verified if c["source_idx"] == i][:per_problem]
                    for c in mine:
                        pool.append(
                            {
                                "input": c["input"],
                                "answer": ex["answer"],
                                "additional_context": {
                                    **ex.get("additional_context", {}),
                                    "augmentation": "paraphrase",
                                    "source_input": ex["input"],
                                },
                            }
                        )
                    if len(mine) < per_problem:
                        pool.extend([dict(ex) for _ in range(per_problem - len(mine))])
            else:
                needed = POOL_SIZE - N_REAL
                for c in verified[:needed]:
                    pool.append(
                        {
                            "input": c["input"],
                            "answer": "### " + c["intended_answer"],
                            "additional_context": {
                                "solution": c.get("solution", ""),
                                "subject": "synthetic",
                                "level": "synthetic",
                                "augmentation": "synthetic",
                            },
                        }
                    )
                if len(verified) < needed:
                    print(f"  WARNING: only {len(verified)}/{needed} verified; padding with duplicates")
                    pool.extend(dict(rng.choice(seed10)) for _ in range(needed - len(verified)))

            rng.shuffle(pool)
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(final_path, "w") as f:
                json.dump(pool, f, indent=2)
            print(f"  final pool ({len(pool)} examples) -> {final_path}")


# ── cached pool access ────────────────────────────────────────────────────────


def _cache_path(condition: str, run_seed: int) -> str:
    return os.path.join(DATA_DIR, f"seed{run_seed}_{condition}.json")


def get_augmented_pool(condition: str, run_seed: int, allow_generate: bool = False) -> list[dict]:
    """Load the cached paraphrase/synthetic pool.

    A cache miss only triggers live (paid) OpenAI generation when allow_generate is
    True — i.e. from --gen-data. Training runs pass allow_generate=False so a missing
    pool fails loudly instead of silently burning budget mid-sweep.
    """
    path = _cache_path(condition, run_seed)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    if not allow_generate:
        raise FileNotFoundError(
            f"No cached {condition} pool for seed {run_seed} at {path}. Pre-generate it first "
            f"(--gen-data [--skip-verify] --seed {run_seed}); training runs will not live-generate pools."
        )

    from gepa.lm import LM

    print(f"Generating {condition} pool for seed {run_seed} (cached to {path})...")
    seed10, _, _ = build_pools(run_seed)
    lm = LM(REFLECTION_LM)
    generate = _generate_paraphrases if condition == "paraphrase" else _generate_synthetic
    pool = generate(seed10, run_seed, lm)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)
    return pool


def build_trainset(condition: str, run_seed: int) -> list[dict]:
    seed10, real100, _ = build_pools(run_seed)
    if condition == "baseline":
        return seed10
    if condition == "duplicate":
        pool = [dict(ex) for ex in seed10 for _ in range(POOL_SIZE // N_REAL)]
        random.Random(run_seed).shuffle(pool)
        return pool
    if condition in ("paraphrase", "synthetic"):
        return get_augmented_pool(condition, run_seed)
    if condition in ("real100", "real100_m10"):
        return real100
    raise ValueError(f"Unknown condition: {condition}")


# ── runner ────────────────────────────────────────────────────────────────────


def run_condition(condition: str, seed: int) -> None:
    label = f"{condition}_s{seed}"
    run_dir = os.path.join(SWEEP_ROOT, label)

    if is_complete(run_dir):
        print(f"[{label}] Already complete — skipping (delete {run_dir} to rerun)")
        return

    minibatch_size = MINIBATCH_BASELINE if condition in ("baseline", "real100_m10") else MINIBATCH_AUGMENTED
    print(f"\n{'=' * 60}")
    print(f"  Condition : {condition}  seed={seed}  minibatch={minibatch_size}")
    print(f"  Run dir   : {run_dir}")
    print(f"{'=' * 60}\n")

    trainset = build_trainset(condition, seed)
    _, _, valset = build_pools(seed)
    print(f"  Train: {len(trainset)} | Val: {len(valset)}")

    os.makedirs(run_dir, exist_ok=True)
    n_augmented = sum(1 for ex in trainset if ex.get("additional_context", {}).get("augmentation"))
    meta = {
        "condition": condition,
        "seed": seed,
        "n_real_seed_problems": N_REAL,
        "pool_size": len(trainset),
        "n_augmented": n_augmented,
        "minibatch_size": minibatch_size,
        "val_size": VAL_SIZE,
        "max_rollouts": MAX_ROLLOUTS,
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
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
        reflection_minibatch_size=minibatch_size,
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


def gen_data(seeds: list[int], skip_verify: bool = False) -> None:
    """Pre-generate paraphrase + synthetic pools for the given seeds.

    With skip_verify=True, only candidates are generated (no OpenAI verification
    solves); an external blind solver + --finalize completes the pools.
    """
    if not skip_verify:
        for s in seeds:
            for condition in ("paraphrase", "synthetic"):
                path = _cache_path(condition, s)
                if os.path.exists(path):
                    print(f"[seed {s}] {condition}: cached ({path})")
                else:
                    get_augmented_pool(condition, s, allow_generate=True)
                    print(f"[seed {s}] {condition}: generated -> {path}")
        return

    from gepa.lm import LM

    lm = LM(REFLECTION_LM)
    for s in seeds:
        seed10, _, _ = build_pools(s)
        for condition in ("paraphrase", "synthetic"):
            if os.path.exists(_cache_path(condition, s)):
                print(f"[seed {s}] {condition}: final pool already cached — skipping")
                continue
            cand_path, _, _ = _candidate_paths(condition, s)
            if os.path.exists(cand_path):
                print(f"[seed {s}] {condition}: candidates already generated ({cand_path})")
                continue
            print(f"[seed {s}] {condition}: generating candidates (no verification)...")
            if condition == "paraphrase":
                candidates = _generate_paraphrase_candidates(seed10, lm)
            else:
                candidates = _generate_synthetic_candidates(seed10, s, lm)
            _write_candidates(condition, s, candidates)
    print("\nCandidates ready. Have a blind solver produce *_solutions.json for each")
    print(f"*_problems.json in {CANDIDATES_DIR}, then run --finalize.")


def run_test() -> None:
    """Smoke test: tiny duplicate-condition run to confirm the pipeline (~$0.50)."""
    import shutil

    run_dir = os.path.join(SWEEP_ROOT, "test_run")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    print("Running smoke test (duplicate condition, 20 val, 60 rollouts)...")
    trainset = build_trainset("duplicate", 0)
    _, _, valset = build_pools(0)
    valset = valset[:20]

    os.makedirs(run_dir, exist_ok=True)
    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=trainset,
        valset=valset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=60,
        run_dir=run_dir,
        reflection_minibatch_size=MINIBATCH_AUGMENTED,
        callbacks=[SweepCallback(run_dir)],
        seed=0,
        track_best_outputs=False,
    )
    print("\nSmoke test passed.")
    print(f"  Seed score : {result.val_aggregate_scores[0]:.4f}")
    print(f"  Rollouts   : {result.total_metric_calls}")


def print_status() -> None:
    total = done = 0
    print(f"\n{'Condition':<22}  {'Status':<10}")
    print("-" * 40)
    for c in CONDITIONS:
        for s in SEEDS:
            run_dir = os.path.join(SWEEP_ROOT, f"{c}_s{s}")
            complete = is_complete(run_dir)
            print(f"{c + '_s' + str(s):<22}  {'complete' if complete else 'pending'}")
            total += 1
            done += complete
    print(f"\n{done}/{total} runs complete.")
    for c in ("paraphrase", "synthetic"):
        cached = sum(os.path.exists(_cache_path(c, s)) for s in SEEDS)
        print(f"{c} pools cached: {cached}/{len(SEEDS)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentation sweep: manufactured vs real training data on Hendrycks MATH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gen-data", action="store_true", help="Pre-generate augmented pools for all seeds")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="With --gen-data: generate candidates only; verification is done externally + --finalize",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Assemble verified pools from candidates + externally produced *_solutions.json",
    )
    parser.add_argument(
        "--top-up",
        type=int,
        metavar="N",
        help="Append N fresh candidates to an existing candidates file (requires --condition synthetic --seed)",
    )
    parser.add_argument("--all", action="store_true", help="Run all pending conditions sequentially")
    parser.add_argument("--test", action="store_true", help="Smoke test (~$0.50)")
    parser.add_argument("--status", action="store_true", help="Print completion status and exit")
    parser.add_argument("--condition", choices=CONDITIONS, help="Condition for a single run")
    parser.add_argument("--seed", type=int, default=None, help="Seed for a single run / --gen-data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.top_up is not None:
        if args.condition is None or args.seed is None:
            raise SystemExit("--top-up requires --condition and --seed")
        top_up(args.condition, args.seed, args.top_up)
    elif args.gen_data:
        gen_data([args.seed] if args.seed is not None else SEEDS, skip_verify=args.skip_verify)
    elif args.finalize:
        finalize([args.seed] if args.seed is not None else SEEDS)
    elif args.test:
        run_test()
    elif args.status:
        print_status()
    elif args.all:
        pending = [
            (c, s) for c in CONDITIONS for s in SEEDS if not is_complete(os.path.join(SWEEP_ROOT, f"{c}_s{s}"))
        ]
        print(f"{len(CONDITIONS) * len(SEEDS) - len(pending)}/{len(CONDITIONS) * len(SEEDS)} already complete.")
        print(f"Running {len(pending)} pending runs...")
        for c, s in pending:
            run_condition(c, s)
        print("\nAll runs complete. Analyze with:")
        print("  uv run python src/minibatch_apo/paper/analyze_augmentation.py")
    elif args.condition is not None:
        run_condition(args.condition, args.seed if args.seed is not None else 0)
    else:
        parse_args()
