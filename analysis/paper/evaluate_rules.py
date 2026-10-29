"""Evaluate highest-scoring prompts with resumable leave-one-out ablations.

Dry run is the default. --execute enables paid GPT-4.1-mini calls.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

from gepa.adapters.default_adapter.default_adapter import ContainsAnswerEvaluator
from minibatch_apo.paper.ablate_rules import _assemble, load_best_prompt, split_rule_blocks
from minibatch_apo.paper.sweep_diversity import TASK_LM, build_datasets

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/rule_ablations"
CONDITIONS = ["real100_m10", "real100"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load_key():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment or project .env")


def checkpoint_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    keys = [(r["evaluation"], r["problem_idx"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate checkpoint outcomes")
    return rows


def build_plan(valset):
    plan = []
    for condition in CONDITIONS:
        for seed in range(5):
            source = ROOT / f"runs/augmentation_sweep/{condition}_s{seed}"
            records = [json.loads(line) for line in (source / "research_log.jsonl").read_text().splitlines()]
            accepted = [r for r in records if r.get("status") == "accepted"]
            best = max(accepted, key=lambda r: r["valset_score"])
            prompt = load_best_prompt(str(source))
            preamble, blocks = split_rule_blocks(prompt)
            saved_path = source / "rule_utility.json"
            saved = json.loads(saved_path.read_text()) if saved_path.exists() else {"rules": []}
            rule_texts = [" ".join(line.strip() for line in block if line.strip()) for block in blocks]
            same = (
                rule_texts == [r["text"] for r in saved["rules"]]
                and saved.get("prompt_sha256") == digest(prompt)
                and saved.get("validation_sha256") == digest(valset)
                and saved.get("n_val") == len(valset)
            )
            needs_evaluation = not same
            plan.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "run": f"{condition}_s{seed}",
                    "source": str(source),
                    "needs_evaluation": needs_evaluation,
                    "iteration": best["iteration"],
                    "candidate_idx": best["new_candidate_idx"],
                    "logged_score": best["valset_score"],
                    "prompt": prompt,
                    "prompt_sha256": digest(prompt),
                    "n_rules": len(blocks),
                    "model": TASK_LM,
                    "validation_sha256": digest(valset),
                    "loo_calls": (len(blocks) + 1) * len(valset) if needs_evaluation else 0,
                    "protocol": "DefaultAdapter messages, substring evaluator, unset temperature/max_tokens; outputs stripped.",
                }
            )
    return plan


def api_outcome(prompt, example, problem_idx, evaluation):
    import litellm

    response = litellm.completion(
        model=TASK_LM,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": example["input"]}],
        num_retries=3,
        drop_params=True,
        timeout=180,
    )
    output = response.choices[0].message.content
    if output is None or response.choices[0].finish_reason != "stop":
        # Never silently score truncation/refusal as an ordinary complete answer.
        raise RuntimeError(f"Incomplete response for {evaluation}/{problem_idx}: {response.choices[0].finish_reason}")
    output = output.strip()
    usage = response.usage
    input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    cost = ((input_tokens - cached_tokens) * 0.40 + cached_tokens * 0.10 + output_tokens * 1.60) / 1_000_000
    return {
        "evaluation": evaluation,
        "problem_idx": problem_idx,
        "score": ContainsAnswerEvaluator()(example, output).score,
        "response": output,
        "model_returned": response.model,
        "response_id": response.id,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "finish_reason": response.choices[0].finish_reason,
    }


def evaluate(prompt, valset, label, checkpoint, budget, workers, api=api_outcome):
    rows = checkpoint_rows(checkpoint)
    saved = {r["problem_idx"]: r for r in rows if r["evaluation"] == label}
    pending = [i for i in range(len(valset)) if i not in saved]
    # Conservative reservation: maximum model output 32,768 tokens, plus UTF-8 input byte count.
    reservation = (
        32768 * 1.60 + (len(prompt.encode()) + max(len(x["input"].encode()) for x in valset) + 128) * 0.40
    ) / 1e6
    # Keep workers occupied while slower requests finish. Checkpoint each answer.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        failures = []
        while pending or active:
            while pending and len(active) < workers and not failures:
                if budget["spent"] + reservation * (len(active) + 1) > budget["limit"]:
                    failures.append("Cost cap reached")
                    break
                i = pending.pop(0)
                active[executor.submit(api, prompt, valset[i], i, label)] = i
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    # Save successes from the entire wave before stopping.
                    failures.append(type(error).__name__)
                    continue
                with checkpoint.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                saved[row["problem_idx"]] = row
                budget["spent"] += row["estimated_cost_usd"]
            if failures and not active:
                break
        if failures:
            raise RuntimeError(f"Stopped ({', '.join(failures)}); resume with same command")
    return [saved[i]["score"] for i in range(len(valset))]


def run_one(item, valset, budget, workers, oracle):
    folder = OUTPUT / item["run"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != item:
            raise ValueError("Manifest changed; refusing to mix prompt/protocol outcomes")
    else:
        atomic_json(manifest, item)
    if not item["needs_evaluation"]:
        for name in ["rule_utility.json", "oracle_prune.json"]:
            source = Path(item["source"]) / name
            destination = folder / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("Cached result changed")
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
        print(f"CACHED {item['run']}", flush=True)
        return
    checkpoint = folder / "outcomes.jsonl"
    preamble, blocks = split_rule_blocks(item["prompt"])
    baseline = evaluate(_assemble(preamble, blocks, None), valset, "baseline", checkpoint, budget, workers)
    base_acc = sum(baseline) / len(baseline)
    rules = []
    print(f"{item['run']} baseline={base_acc:.3f}; {len(blocks)} rules; cumulative ${budget['spent']:.2f}", flush=True)
    for idx, block in enumerate(blocks):
        scores = evaluate(_assemble(preamble, blocks, idx), valset, f"drop_{idx}", checkpoint, budget, workers)
        accuracy = sum(scores) / len(scores)
        rules.append(
            {
                "idx": idx,
                "text": " ".join(line.strip() for line in block if line.strip()),
                "acc_without": accuracy,
                "utility": base_acc - accuracy,
                "scores_without": scores,
            }
        )
        atomic_json(folder / "rule_utility.partial.json", {"base_acc": base_acc, "rules": rules})
        print(
            f"{item['run']} rule {idx + 1}/{len(blocks)} utility={base_acc - accuracy:+.3f}; cumulative ${budget['spent']:.2f}",
            flush=True,
        )
    atomic_json(
        folder / "rule_utility.json",
        {
            "run": item["source"],
            "n_val": len(valset),
            "base_acc": base_acc,
            "baseline_scores": baseline,
            "rules": rules,
            "selection": "highest_logged_validation_score",
            "candidate_idx": item["candidate_idx"],
            "iteration": item["iteration"],
            "prompt_sha256": item["prompt_sha256"],
            "validation_sha256": item["validation_sha256"],
        },
    )
    if oracle:
        dropped = {r["idx"] for r in rules if r["utility"] < 0}
        prompt = "\n".join(preamble + [line for j, block in enumerate(blocks) if j not in dropped for line in block])
        scores = evaluate(prompt, valset, "oracle", checkpoint, budget, workers) if dropped else baseline
        accuracy = sum(scores) / len(scores)
        atomic_json(
            folder / "oracle_prune.json",
            {
                "run": item["source"],
                "condition": item["condition"],
                "base_acc": base_acc,
                "n_rules": len(blocks),
                "n_dropped": len(dropped),
                "oracle_acc": accuracy,
                "oracle_gain": accuracy - base_acc,
                "scores": scores,
                "sum_neg_estimate": -sum(min(r["utility"], 0) for r in rules),
            },
        )
    print(f"COMPLETED {item['run']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cost-cap", type=float, default=80)
    args = parser.parse_args()
    if args.workers < 1 or args.cost_cap <= 0:
        parser.error("workers and cost-cap must be positive")
    _, valset = build_datasets(7, 0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = build_plan(valset)
    atomic_json(OUTPUT / "plan.json", plan)
    atomic_json(OUTPUT / "validation.json", valset)
    for item in plan:
        print(
            f"{'EVALUATE' if item['needs_evaluation'] else 'CACHED'} {item['run']}: iteration={item['iteration']}, rules={item['n_rules']}, calls={item['loo_calls']}",
            flush=True,
        )
    oracle_calls = sum(item["needs_evaluation"] for item in plan) * len(valset) if args.oracle else 0
    print(
        f"Total LOO calls: {sum(i['loo_calls'] for i in plan)}; oracle adds at most {oracle_calls}; cost cap ${args.cost_cap:.2f}",
        flush=True,
    )
    if not args.execute:
        return
    load_key()
    spent = sum(
        row["estimated_cost_usd"]
        for checkpoint in OUTPUT.glob("*/outcomes.jsonl")
        for row in checkpoint_rows(checkpoint)
    )
    budget = {"spent": spent, "limit": args.cost_cap}
    started = time.time()
    for item in plan:
        run_one(item, valset, budget, args.workers, args.oracle)
    atomic_json(
        OUTPUT / "completion.json",
        {
            "estimated_cost_usd": budget["spent"],
            "elapsed_seconds_this_invocation": time.time() - started,
            "cached_runs": sum(not item["needs_evaluation"] for item in plan),
            "evaluated_runs": sum(item["needs_evaluation"] for item in plan),
            "oracle": args.oracle,
        },
    )
    print(f"ALL DONE; recorded API cost ${budget['spent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
