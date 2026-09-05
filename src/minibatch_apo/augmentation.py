"""Safe loading helpers for externally prepared augmentation pools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_external_pool(path: Path, expected_size: int) -> list[dict[str, Any]]:
    """Load a user-prepared JSON pool and validate its minimal GEPA schema."""
    with path.open(encoding="utf-8") as handle:
        pool = json.load(handle)
    if not isinstance(pool, list):
        raise ValueError(f"{path} must contain a JSON list")
    if len(pool) != expected_size:
        raise ValueError(f"{path} has {len(pool)} examples; expected {expected_size} examples")
    for index, example in enumerate(pool):
        if (
            not isinstance(example, dict)
            or not isinstance(example.get("input"), str)
            or not isinstance(example.get("answer"), str)
        ):
            raise ValueError(f"{path} example {index} must contain string input and answer fields")
    return pool


def duplicate_pool(seed_examples: list[dict[str, Any]], target_size: int) -> list[dict[str, Any]]:
    """Expand a seed pool by exact repetition, for acceptance-reliability controls."""
    if not seed_examples:
        raise ValueError("cannot duplicate an empty pool")
    return [dict(seed_examples[index % len(seed_examples)]) for index in range(target_size)]
