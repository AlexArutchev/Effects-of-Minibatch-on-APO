"""Paper-specific ablate correlate protocol and helpers."""

import argparse
import json
import os
import re

from minibatch_apo.paper.analyze_run import (
    _cosine,
    _embed_rules,
    load_run,
    rule_accumulation,
)
from minibatch_apo.paper.sweep_augmentation import build_trainset
from minibatch_apo.paper.sweep_diversity import build_datasets

EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
MATCH_THRESHOLD = 0.80


def parse_run_name(run_dir: str) -> tuple[str, int]:
    """runs/.../real100_m10_s3 -> ('real100_m10', 3)."""
    m = re.match(r"(.+)_s(\d+)$", os.path.basename(run_dir.rstrip("/")))
    if not m:
        raise ValueError(f"Cannot parse condition/seed from {run_dir}")
    return m.group(1), int(m.group(2))


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def _max_sim(rule_emb: list[float], texts: list[str]) -> float:
    return max(_cosine(rule_emb, e) for e in _embed_rules(texts, EMBED_MODEL))


def proxy_specificities(cond: str, seed: int, rule_texts: list[str], val_texts: list[str]) -> list[float | None]:
    """max_sim(rule, train) - max_sim(rule, val), no lineage needed."""
    train_texts = [ex["input"] for ex in build_trainset(cond, seed)]
    out: list[float | None] = []
    for t in rule_texts:
        emb = _embed_rules([t], EMBED_MODEL)[0]
        out.append(_max_sim(emb, train_texts) - _max_sim(emb, val_texts))
    return out


def _minibatch_texts_by_iter(records: list[dict]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for rec in records:
        if rec.get("status") != "accepted":
            continue
        texts = [
            ex.get("Inputs", "")
            for exs in rec.get("reflective_dataset", {}).values()
            for ex in exs
            if ex.get("Inputs")
        ]
        if texts:
            out[rec["iteration"]] = texts
    return out


def faithful_specificities(run_dir: str, rule_texts: list[str]) -> list[float | None]:
    """Cross-iteration specificity per best-prompt rule; None if no birth match."""
    records = load_run(run_dir)
    mb_by_iter = _minibatch_texts_by_iter(records)
    if len(mb_by_iter) < 2:
        return [None] * len(rule_texts)

    # (iteration, added_rule_text) with embeddings, to find each rule's birth iteration.
    acc = rule_accumulation(records, embed_model=EMBED_MODEL, threshold=MATCH_THRESHOLD)
    added = [(s["iteration"], ar) for s in acc for ar in s["added_rules"] if s["iteration"] in mb_by_iter]
    added_embs = _embed_rules([t for _, t in added], EMBED_MODEL) if added else []

    out: list[float | None] = []
    for t in rule_texts:
        emb = _embed_rules([t], EMBED_MODEL)[0]
        # birth iteration = best-matching added rule, if confident
        birth, best = None, -1.0
        for (it, _), ae in zip(added, added_embs, strict=False):
            c = _cosine(emb, ae)
            if c > best:
                best, birth = c, it
        if birth is None or best < MATCH_THRESHOLD:
            out.append(None)
            continue
        own = _max_sim(emb, mb_by_iter[birth])
        others = [_max_sim(emb, txts) for it, txts in mb_by_iter.items() if it != birth]
        out.append(own - (sum(others) / len(others) if others else 0.0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: correlate rule specificity with rule utility.")
    p.add_argument("run_dirs", nargs="+", help="Run dirs that already have rule_utility.json")
    p.add_argument("--metric", choices=["faithful", "proxy"], default="faithful", help="Specificity definition")
    p.add_argument("--out", help="Optional path to write the paired (specificity, utility) records as JSON")
    args = p.parse_args()

    val_texts = [ex["input"] for ex in build_datasets(num_types=7, run_seed=0)[1]]
    if args.metric == "proxy":  # pre-embed val once for reuse across runs
        _embed_rules(val_texts, EMBED_MODEL)

    all_pairs: list[dict] = []
    print(f"\n  metric = {args.metric}")
    print(f"  {'run':<18}  {'matched/n':>10}  {'spearman(spec,util)':>20}  {'mean util':>10}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*20}  {'-'*10}")
    for run_dir in args.run_dirs:
        cond, seed = parse_run_name(run_dir)
        util_path = os.path.join(run_dir, "rule_utility.json")
        if not os.path.exists(util_path):
            print(f"  {os.path.basename(run_dir):<18}  (no rule_utility.json — run ablate_rules.py first)")
            continue
        rules = json.load(open(util_path))["rules"]
        texts = [r["text"] for r in rules]

        if args.metric == "faithful":
            specs = faithful_specificities(run_dir, texts)
        else:
            specs = proxy_specificities(cond, seed, texts, val_texts)

        paired = [(s, r) for s, r in zip(specs, rules, strict=False) if s is not None]
        for s, r in paired:
            all_pairs.append({"run": os.path.basename(run_dir), "specificity": s, **r})

        n_used = len(paired)
        if n_used >= 3:
            rho = _spearman([s for s, _ in paired], [r["utility"] for _, r in paired])
            mu = sum(r["utility"] for _, r in paired) / n_used
            print(f"  {os.path.basename(run_dir):<18}  {n_used:>4}/{len(rules):<5}  {rho:>+20.3f}  {mu:>+10.4f}")
        else:
            print(f"  {os.path.basename(run_dir):<18}  {n_used:>4}/{len(rules):<5}  (too few matched rules)")

    if len(all_pairs) >= 3:
        rho = _spearman([x["specificity"] for x in all_pairs], [x["utility"] for x in all_pairs])
        print(f"\n  Pooled (confounded by condition): n={len(all_pairs)}  spearman={rho:+.3f}")
        print("  -> Read the per-run rows above for the confound-free signal.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_pairs, f, indent=2)
        print(f"\n  wrote {len(all_pairs)} paired records -> {args.out}")


if __name__ == "__main__":
    main()
