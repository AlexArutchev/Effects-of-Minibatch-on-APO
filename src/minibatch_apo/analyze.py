"""Analyze aggregate study logs and optional rule-specificity records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from minibatch_apo.rules import cross_iteration_specificity


def load_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load every study_log.jsonl below a protocol output directory."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for log_path in sorted(root.glob("*/study_log.jsonl")):
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        grouped[log_path.parent.name] = rows
    return grouped


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compute acceptance and validation summaries for one condition."""
    accepted = [record for record in records if record.get("status") == "accepted"]
    validation_scores = [float(record["valset_score"]) for record in accepted if "valset_score" in record]
    return {
        "proposals": len(records),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "mean_accepted_valset_score": mean(validation_scores) if validation_scores else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Protocol directory, for example outputs/.../factorial")
    parser.add_argument(
        "--rules", action="store_true", help="Compute lexical cross-iteration rule specificity from opted-in text logs"
    )
    args = parser.parse_args()
    grouped = load_records(args.run_root)
    if not grouped:
        raise SystemExit(f"No study_log.jsonl files found under {args.run_root}")
    for label, records in grouped.items():
        summary = summarize(records)
        print(f"{label}: {summary['accepted']}/{summary['proposals']} accepted ({summary['acceptance_rate']:.1%})")
        if args.rules:
            specificity = cross_iteration_specificity(records)
            if specificity:
                rates = [float(row["later_match_rate"]) for row in specificity]
                spread = stdev(rates) if len(rates) > 1 else 0.0
                print(f"  lexical later-example match rate: {mean(rates):.3f} ± {spread:.3f} ({len(rates)} updates)")
            else:
                print("  no rule data; rerun with opted-in prompt and reflective-dataset logging")


if __name__ == "__main__":
    main()
