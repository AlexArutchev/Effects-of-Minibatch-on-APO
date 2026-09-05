"""Command-line driver shared by the diversity, factorial, and augmentation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from minibatch_apo.config import load_config
from minibatch_apo.experiments import expand_conditions, run_condition


def main(default_kind: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run the configurable {default_kind} study protocol.")
    parser.add_argument("config", type=Path, help="Path to a JSON configuration file")
    parser.add_argument("--all", action="store_true", help="Run every condition in the configuration")
    parser.add_argument("--condition", action="append", default=[], help="Run one named condition; repeat as needed")
    parser.add_argument("--smoke-test", action="store_true", help="Use at most 20 examples and 50 metric calls")
    parser.add_argument("--rerun", action="store_true", help="Run even when a completed summary log already exists")
    parser.add_argument("--dry-run", action="store_true", help="List selected conditions without calling an LM")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.experiment.kind != default_kind:
        raise SystemExit(
            f"{args.config} configures {config.experiment.kind!r}; this entry point runs {default_kind!r}."
        )
    conditions = expand_conditions(config)
    if args.condition:
        selected = [condition for condition in conditions if condition.label in args.condition]
        missing = sorted(set(args.condition) - {condition.label for condition in selected})
        if missing:
            raise SystemExit(f"Unknown condition(s): {', '.join(missing)}")
    elif args.all:
        selected = conditions
    else:
        raise SystemExit("Choose --all or at least one --condition. Use --dry-run to inspect the expansion first.")
    for condition in selected:
        print(condition.label)
    if args.dry_run:
        return
    for condition in selected:
        run_condition(config, condition, rerun=args.rerun, smoke_test=args.smoke_test)
