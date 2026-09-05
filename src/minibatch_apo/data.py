"""Deterministic construction of MATH study splits."""

from __future__ import annotations

import random
from typing import Any

from minibatch_apo.config import DatasetConfig
from minibatch_apo.math import extract_boxed_answer


def _split_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]


def _load_by_subject(config: DatasetConfig) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(config.name)
    grouped: dict[str, list[dict[str, Any]]] = {subject: [] for subject in config.subjects}
    for split_name in ("train", "test"):
        for row in dataset[split_name]:
            answer = extract_boxed_answer(row["solution"])
            if row["level"] not in config.levels or row["subject"] not in grouped or answer is None:
                continue
            grouped[row["subject"]].append(
                {
                    "input": row["problem"],
                    "answer": "### " + answer,
                    "additional_context": {"subject": row["subject"], "level": row["level"]},
                }
            )
    return grouped


def build_math_datasets(
    config: DatasetConfig, *, diversity: int, run_seed: int, train_size: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a fixed stratified validation set and a disjoint train set.

    The validation split depends only on ``dataset_seed``. Therefore all conditions
    in a sweep compare against identical held-out examples.
    """
    if diversity < 1 or diversity > len(config.subjects):
        raise ValueError(f"diversity must be between 1 and {len(config.subjects)}")
    grouped = _load_by_subject(config)
    valset: list[dict[str, Any]] = []
    reserved: dict[str, set[int]] = {}
    validation_rng = random.Random(config.dataset_seed)
    for subject, count in zip(config.subjects, _split_counts(config.val_size, len(config.subjects)), strict=True):
        indices = list(range(len(grouped[subject])))
        validation_rng.shuffle(indices)
        chosen = set(indices[:count])
        if len(chosen) != count:
            raise ValueError(f"not enough {subject} examples to construct the validation set")
        reserved[subject] = chosen
        valset.extend(grouped[subject][index] for index in sorted(chosen))
    validation_rng.shuffle(valset)

    requested_size = train_size if train_size is not None else config.train_size
    trainset: list[dict[str, Any]] = []
    train_rng = random.Random(run_seed)
    selected_subjects = config.subjects[:diversity]
    for subject, count in zip(selected_subjects, _split_counts(requested_size, len(selected_subjects)), strict=True):
        available = [index for index in range(len(grouped[subject])) if index not in reserved[subject]]
        train_rng.shuffle(available)
        if len(available) < count:
            raise ValueError(f"not enough {subject} examples after validation reservation; need {count}")
        trainset.extend(grouped[subject][index] for index in available[:count])
    train_rng.shuffle(trainset)
    return trainset, valset
