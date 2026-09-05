"""Shared condition expansion and execution for the public study protocols."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gepa
from minibatch_apo.augmentation import duplicate_pool, load_external_pool
from minibatch_apo.config import StudyConfig
from minibatch_apo.data import build_math_datasets
from minibatch_apo.study_callback import StudyCallback


@dataclass(frozen=True)
class Condition:
    label: str
    seed: int
    diversity: int
    minibatch_size: int
    augmentation_arm: str | None = None


def expand_conditions(config: StudyConfig) -> list[Condition]:
    """Return every requested experimental condition in deterministic order."""
    experiment = config.experiment
    if experiment.kind == "diversity":
        return [
            Condition(f"d{diversity}_s{seed}", seed, diversity, experiment.minibatch_sizes[0])
            for diversity in experiment.diversity_levels
            for seed in experiment.seeds
        ]
    if experiment.kind == "factorial":
        return [
            Condition(f"d{diversity}_m{minibatch}_s{seed}", seed, diversity, minibatch)
            for diversity in experiment.diversity_levels
            for minibatch in experiment.minibatch_sizes
            for seed in experiment.seeds
        ]
    return [
        Condition(f"{arm}_s{seed}", seed, len(config.dataset.subjects), _augmentation_minibatch(config, arm), arm)
        for arm in experiment.augmentation_arms
        for seed in experiment.seeds
    ]


def _augmentation_minibatch(config: StudyConfig, arm: str) -> int:
    if arm in {"baseline", "real_m10"}:
        return config.experiment.base_minibatch_size
    return config.experiment.augmented_minibatch_size


def run_condition(config: StudyConfig, condition: Condition, *, rerun: bool = False, smoke_test: bool = False) -> Path:
    """Run one condition, writing only aggregate logs unless consent enables text logging."""
    run_dir = Path(config.output_dir) / config.experiment.kind / condition.label
    completion_marker = run_dir / "study_log.jsonl"
    if completion_marker.exists() and not rerun:
        print(f"[{condition.label}] already has a study log; skipping")
        return run_dir

    trainset, valset = _datasets_for_condition(config, condition)
    if smoke_test:
        trainset = trainset[: min(20, len(trainset))]
        valset = valset[: min(20, len(valset))]
        max_metric_calls = min(50, config.max_metric_calls)
    else:
        max_metric_calls = config.max_metric_calls
    if condition.minibatch_size > len(trainset):
        raise ValueError(f"minibatch size {condition.minibatch_size} exceeds train set size {len(trainset)}")

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "condition": condition.__dict__,
        "study_config": config.to_dict(),
        "train_size": len(trainset),
        "val_size": len(valset),
        "max_metric_calls": max_metric_calls,
        "smoke_test": smoke_test,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    callback = StudyCallback(
        run_dir,
        include_prompt_text=config.logging.include_prompt_text,
        include_reflective_dataset=config.logging.include_reflective_dataset,
        include_raw_model_outputs=config.logging.include_raw_model_outputs,
    )
    result = gepa.optimize(
        seed_candidate={"system_prompt": config.seed_prompt},
        trainset=trainset,
        valset=valset,
        task_lm=config.task_lm,
        reflection_lm=config.reflection_lm,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        reflection_minibatch_size=condition.minibatch_size,
        callbacks=[callback],
        seed=condition.seed,
        track_best_outputs=False,
    )
    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"[{condition.label}] seed={seed_score:.4f}, best={best_score:.4f}, delta={best_score - seed_score:+.4f}")
    return run_dir


def _datasets_for_condition(
    config: StudyConfig, condition: Condition
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if condition.augmentation_arm is None:
        return build_math_datasets(config.dataset, diversity=condition.diversity, run_seed=condition.seed)

    experiment = config.experiment
    full_pool, valset = build_math_datasets(
        config.dataset,
        diversity=len(config.dataset.subjects),
        run_seed=condition.seed,
        train_size=experiment.augmented_train_size,
    )
    seed_pool = full_pool[: experiment.base_train_size]
    arm = condition.augmentation_arm
    if arm == "baseline":
        return seed_pool, valset
    if arm == "duplicate":
        return duplicate_pool(seed_pool, experiment.augmented_train_size), valset
    if arm in {"real", "real_m10"}:
        return full_pool, valset
    if experiment.pools_dir is None:
        raise ValueError(f"augmentation arm {arm!r} requires experiment.pools_dir")
    pool_path = Path(experiment.pools_dir) / f"{arm}_s{condition.seed}.json"
    return load_external_pool(pool_path, experiment.augmented_train_size), valset
