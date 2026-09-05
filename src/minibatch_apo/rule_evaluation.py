"""Configurable leave-one-out rule evaluation and oracle-pruning analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from minibatch_apo.config import DatasetConfig
from minibatch_apo.data import build_math_datasets
from minibatch_apo.rules import _RULE_START


def split_rule_blocks(prompt: str) -> tuple[list[str], list[list[str]]]:
    """Split a prompt into an unablated preamble and independently removable rules."""
    preamble: list[str] = []
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in prompt.splitlines():
        if _RULE_START.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    if current is not None:
        blocks.append(current)
    return preamble, blocks


def assemble_without(preamble: list[str], blocks: list[list[str]], drop_indices: set[int]) -> str:
    """Reassemble a prompt after removing zero or more rule blocks."""
    kept_lines = [line for index, block in enumerate(blocks) if index not in drop_indices for line in block]
    return "\n".join(preamble + kept_lines)


def utility_summary(utilities: list[float]) -> dict[str, float | int]:
    """Summarize negative leave-one-out utilities without making another model call."""
    negative = [utility for utility in utilities if utility < 0]
    return {
        "rule_count": len(utilities),
        "negative_rule_count": len(negative),
        "sum_negative_utility": -sum(negative),
    }


def load_best_prompt(run_dir: Path) -> str:
    """Load the prompt from a text-logging study run's best accepted proposal."""
    log_path = run_dir / "study_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    accepted = [record for record in records if record.get("status") == "accepted" and "new_prompt" in record]
    if not accepted:
        raise ValueError("No accepted prompt text found. Enable opted-in prompt logging before running this analysis.")
    best = next((record for record in accepted if record.get("is_best")), accepted[-1])
    prompt = best["new_prompt"].get("system_prompt")
    if not isinstance(prompt, str):
        raise ValueError("The best proposal does not contain a system_prompt component")
    return prompt


def evaluate_prompt(adapter: DefaultAdapter, valset: list[dict[str, Any]], prompt: str) -> float:
    scores = adapter.evaluate(valset, {"system_prompt": prompt}).scores
    return sum(scores) / len(scores)


def leave_one_out(adapter: DefaultAdapter, valset: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    """Measure every rule's utility, defined as full-prompt accuracy minus ablated accuracy."""
    preamble, blocks = split_rule_blocks(prompt)
    if not blocks:
        raise ValueError("No bullet-like rules were found in the prompt")
    base_accuracy = evaluate_prompt(adapter, valset, assemble_without(preamble, blocks, set()))
    rows = []
    for index, block in enumerate(blocks):
        ablated_accuracy = evaluate_prompt(adapter, valset, assemble_without(preamble, blocks, {index}))
        rows.append(
            {
                "index": index,
                "text": " ".join(line.strip() for line in block if line.strip()),
                "accuracy_without_rule": ablated_accuracy,
                "utility": base_accuracy - ablated_accuracy,
            }
        )
    return {"base_accuracy": base_accuracy, "rules": rows, **utility_summary([row["utility"] for row in rows])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory with opted-in prompt text")
    parser.add_argument("--task-lm", required=True, help="LiteLLM model used for the held-out evaluations")
    parser.add_argument("--utility-file", type=Path, help="Where to write/read leave-one-out utilities")
    parser.add_argument(
        "--oracle", action="store_true", help="Drop all negative-utility rules and run one joint re-score"
    )
    parser.add_argument("--dry", action="store_true", help="Only print a summary of an existing utility file")
    args = parser.parse_args()

    utility_path = args.utility_file or args.run_dir / "rule_utility.json"
    if args.dry:
        result = json.loads(utility_path.read_text(encoding="utf-8"))
    else:
        metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
        dataset = DatasetConfig(**metadata["study_config"]["dataset"])
        _, valset = build_math_datasets(dataset, diversity=len(dataset.subjects), run_seed=0)
        result = leave_one_out(DefaultAdapter(model=args.task_lm), valset, load_best_prompt(args.run_dir))
        utility_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {utility_path}")
    print(json.dumps(utility_summary([row["utility"] for row in result["rules"]]), indent=2))

    if args.oracle:
        if args.dry:
            raise SystemExit("--oracle requires evaluation; omit --dry")
        prompt = load_best_prompt(args.run_dir)
        preamble, blocks = split_rule_blocks(prompt)
        drop_indices = {row["index"] for row in result["rules"] if row["utility"] < 0}
        oracle_accuracy = evaluate_prompt(
            DefaultAdapter(model=args.task_lm), valset, assemble_without(preamble, blocks, drop_indices)
        )
        print(
            json.dumps(
                {"dropped_rules": len(drop_indices), "oracle_gain": oracle_accuracy - result["base_accuracy"]}, indent=2
            )
        )


if __name__ == "__main__":
    main()
