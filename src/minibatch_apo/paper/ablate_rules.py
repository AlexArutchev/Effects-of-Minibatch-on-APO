"""Paper-specific ablate rules protocol and helpers."""

import argparse
import json
import os

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from minibatch_apo.paper.analyze_run import RULE_START
from minibatch_apo.paper.sweep_diversity import TASK_LM, build_datasets

SMOKE_VAL = 20
SMOKE_RULES = 3


def load_best_prompt(run_dir: str) -> str:
    """Return the system_prompt of the best accepted candidate for this run."""
    records = [json.loads(line) for line in open(os.path.join(run_dir, "research_log.jsonl"))]
    accepted = [r for r in records if r.get("status") == "accepted" and "new_texts" in r]
    if not accepted:
        raise ValueError(f"{run_dir}: no accepted candidates in research_log.jsonl")
    best = max(accepted, key=lambda r: r.get("valset_score", float("-inf")))
    return best["new_texts"]["system_prompt"]


def split_rule_blocks(text: str) -> tuple[list[str], list[list[str]]]:
    """Split a prompt into (preamble_lines, rule_blocks) preserving original lines.

    A rule block starts at a RULE_START line (bullet / numbered / bold header) and
    runs until the next one; continuation and blank lines stay with their block.
    Lines before the first rule are the preamble — never ablated (removing the
    intro would break the prompt rather than test a rule).
    """
    preamble: list[str] = []
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if RULE_START.match(line):
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


def _assemble(preamble: list[str], blocks: list[list[str]], drop: int | None) -> str:
    kept = [b for j, b in enumerate(blocks) if j != drop]
    lines = preamble + [ln for b in kept for ln in b]
    return "\n".join(lines)


def _accuracy(adapter: DefaultAdapter, valset: list[dict], prompt_text: str) -> float:
    scores = adapter.evaluate(valset, {"system_prompt": prompt_text}).scores
    return sum(scores) / len(scores)


def ablate_run(run_dir: str, adapter: DefaultAdapter, valset: list[dict], smoke: bool = False) -> dict:
    prompt = load_best_prompt(run_dir)
    preamble, blocks = split_rule_blocks(prompt)
    if smoke:
        blocks = blocks[:SMOKE_RULES]
    if not blocks:
        raise ValueError(f"{run_dir}: no ablatable rule blocks found (prompt may be prose-only)")

    base_acc = _accuracy(adapter, valset, _assemble(preamble, blocks, drop=None))
    print(f"[{os.path.basename(run_dir)}] base_acc={base_acc:.3f} over {len(valset)} val | {len(blocks)} rules")

    rules_out = []
    for j, block in enumerate(blocks):
        acc_wo = _accuracy(adapter, valset, _assemble(preamble, blocks, drop=j))
        util = base_acc - acc_wo
        text = " ".join(ln.strip() for ln in block if ln.strip())
        rules_out.append({"idx": j, "text": text, "acc_without": acc_wo, "utility": util})
        print(f"    rule {j:>2}: util={util:+.3f}  (acc_without={acc_wo:.3f})  {text[:70]}")

    return {"run": run_dir, "n_val": len(valset), "base_acc": base_acc, "rules": rules_out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-level leave-one-out utility (Stage 1).")
    p.add_argument("run_dirs", nargs="+", help="Run directories to ablate (uses each run's best candidate)")
    p.add_argument("--smoke", action="store_true", help=f"Cheap check: {SMOKE_VAL} val problems, first {SMOKE_RULES} rules")
    p.add_argument("--force", action="store_true", help="Recompute even if rule_utility.json exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _, valset = build_datasets(num_types=7, run_seed=0)
    if args.smoke:
        valset = valset[:SMOKE_VAL]
    adapter = DefaultAdapter(model=TASK_LM)

    for run_dir in args.run_dirs:
        out_path = os.path.join(run_dir, "rule_utility_smoke.json" if args.smoke else "rule_utility.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[{os.path.basename(run_dir)}] {os.path.basename(out_path)} exists — skipping (--force to redo)")
            continue
        result = ablate_run(run_dir, adapter, valset, smoke=args.smoke)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    -> {out_path}\n")


if __name__ == "__main__":
    main()
