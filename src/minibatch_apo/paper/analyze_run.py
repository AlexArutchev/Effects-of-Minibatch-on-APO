"""Analysis script for GEPA prompt evolution research.

Loads research_log.jsonl produced by ResearchCallback and computes:
  - Edit distance between old and new prompt texts
  - Feedback type distribution
  - Acceptance rates by feedback type and edit distance
  - Learning curve (valset score vs. rollout count)

Usage::

    uv run python src/minibatch_apo/paper/analyze_run.py runs/my_run
    uv run python src/minibatch_apo/paper/analyze_run.py runs/run1 runs/run2  # cross-run comparison
"""

import argparse
import difflib
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> list[dict]:
    path = Path(run_dir) / "research_log.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No research_log.jsonl in {run_dir}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Prompt diff utilities
# ---------------------------------------------------------------------------

def token_edit_distance(a: str, b: str) -> float:
    """Fraction of tokens in the longer text that differ (0=identical, 1=completely different)."""
    a_tokens = a.split()
    b_tokens = b.split()
    matcher = difflib.SequenceMatcher(None, a_tokens, b_tokens)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    total = max(len(a_tokens), len(b_tokens), 1)
    return 1.0 - matching / total


def unified_diff_summary(a: str, b: str) -> dict:
    """Return counts of added/removed lines and the diff text."""
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return {"added_lines": added, "removed_lines": removed, "diff": "".join(diff)}


def extract_added_rules(old: str, new: str) -> list[str]:
    """Extract discrete rules added to the prompt.

    Parses the added lines from the diff into individual rules by treating
    bullet points (-, *, digits.) and bold headers (**text:**) as rule
    boundaries. Continuation lines (indented or plain prose) are folded into
    the preceding rule.
    """
    a_lines = old.splitlines(keepends=True)
    b_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))

    added_lines = [
        l[1:]  # strip leading +
        for l in diff
        if l.startswith("+") and not l.startswith("+++")
    ]

    rule_start = re.compile(
        r"^\s*("
        r"\-\s+"          # bullet -
        r"|\*\s+"         # bullet *
        r"|\d+\.\s+"      # numbered 1.
        r"|\*\*[^*]+\*\*" # bold header **text**
        r")"
    )

    rules: list[str] = []
    current: list[str] = []

    for line in added_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if rule_start.match(line):
            if current:
                rules.append(" ".join(current))
            current = [stripped]
        else:
            # Continuation of previous rule, or standalone prose sentence
            if current:
                current.append(stripped)
            elif stripped.endswith(".") or stripped.endswith(":"):
                rules.append(stripped)

    if current:
        rules.append(" ".join(current))

    return rules


def print_rule_analysis(records: list[dict], component: str = "system_prompt") -> None:
    """Print per-iteration rule counts and the text of each added rule."""
    accepted = [r for r in records if r.get("status") == "accepted"]
    print(f"\n{'='*60}")
    print(f"  Added rules per accepted candidate")
    print(f"{'='*60}")

    total_rules_by_iter: list[tuple[int, int]] = []
    for r in sorted(accepted, key=lambda x: x.get("iteration", 0)):
        iteration = r["iteration"]
        old = r.get("old_texts", {}).get(component, "")
        new = r.get("new_texts", {}).get(component, "")
        rules = extract_added_rules(old, new)
        valset = r.get("valset_score")
        vs_str = f"{valset:.3f}" if valset is not None else "   —"
        print(f"\nIteration {iteration} | valset={vs_str} | {len(rules)} rule(s) added:")
        for i, rule in enumerate(rules, 1):
            # Truncate long rules for readability
            display = rule if len(rule) <= 120 else rule[:117] + "..."
            print(f"  {i}. {display}")
        total_rules_by_iter.append((iteration, len(rules)))

    print(f"\n{'─'*40}")
    print(f"  Summary: rules added per accepted candidate")
    counts = [c for _, c in total_rules_by_iter]
    if counts:
        print(f"  Mean  : {sum(counts)/len(counts):.1f}")
        print(f"  Max   : {max(counts)}")
        print(f"  Min   : {min(counts)}")
        print(f"  Total : {sum(counts)}")


# ---------------------------------------------------------------------------
# Rule survival tracking
# ---------------------------------------------------------------------------

RULE_START = re.compile(
    r"^\s*("
    r"\-\s+"           # bullet -
    r"|\*\s+"          # bullet *
    r"|\d+\.\s+"       # numbered 1.
    r"|\*\*[^*]+\*\*"  # bold header **text**
    r")"
)


def extract_rules_from_text(text: str) -> list[str]:
    """Parse a full prompt into a list of discrete rules/bullets."""
    rules: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if RULE_START.match(line):
            if current:
                rules.append(" ".join(current))
            current = [stripped]
        else:
            if current:
                current.append(stripped)
            elif stripped.endswith(".") or stripped.endswith(":"):
                rules.append(stripped)
    if current:
        rules.append(" ".join(current))
    return rules


def _rule_similarity(a: str, b: str) -> float:
    """Token-overlap similarity between two rule strings (0–1)."""
    return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


# Cache embeddings by rule text to avoid re-running inference across iterations.
_EMBEDDING_CACHE: dict[str, list[float]] = {}
_HF_MODEL = None
_HF_TOKENIZER = None


def _load_hf_model(model_name: str):
    global _HF_MODEL, _HF_TOKENIZER
    if _HF_MODEL is None:
        from transformers import AutoModel, AutoTokenizer
        import torch
        print(f"Loading {model_name} from HuggingFace (first time may download weights)...")
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _HF_MODEL = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _HF_MODEL = _HF_MODEL.to(device)
        _HF_MODEL.eval()
        print(f"Model loaded on {device}.")
    return _HF_MODEL, _HF_TOKENIZER


def _embed_rules(rules: list[str], model_name: str) -> list[list[float]]:
    """Return embeddings for rules, using the cache for any already seen.

    Runs local inference via HuggingFace transformers (e.g. Qwen/Qwen3-Embedding-4B).
    Uses last-token pooling with L2 normalisation, which is standard for Qwen embedding models.
    Processes in batches of 32 to avoid OOM on large rule sets.
    """
    import torch
    import torch.nn.functional as F

    uncached = [r for r in rules if r not in _EMBEDDING_CACHE]
    if uncached:
        model, tokenizer = _load_hf_model(model_name)
        batch_size = 32
        all_embeddings: list[list[float]] = []
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            # Last non-padding token (standard for decoder-based embedding models).
            # The tokenizer pads on the right, so a plain [:, -1, :] would read a PAD
            # position for every sequence shorter than the batch maximum, making a
            # text's embedding depend on whichever neighbours it was batched with.
            # Gather the true final token per sequence via the attention mask instead.
            hidden = outputs.last_hidden_state
            last_idx = encoded["attention_mask"].sum(dim=1) - 1
            embeddings = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().float().tolist())
        for text, emb in zip(uncached, all_embeddings):
            _EMBEDDING_CACHE[text] = emb
    return [_EMBEDDING_CACHE[r] for r in rules]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0


def match_rules_semantic(
    old_rules: list[str],
    new_rules: list[str],
    model_name: str,
    threshold: float = 0.80,
) -> tuple[list[str], list[str], list[str]]:
    """Same greedy best-match logic as match_rules but uses cosine similarity on embeddings."""
    old_embeddings = _embed_rules(old_rules, model_name)
    new_embeddings = _embed_rules(new_rules, model_name)

    matched_old: set[int] = set()
    survived: list[str] = []
    added: list[str] = []

    for new_rule, new_emb in zip(new_rules, new_embeddings):
        best_score = 0.0
        best_idx = -1
        for i, old_emb in enumerate(old_embeddings):
            s = _cosine(old_emb, new_emb)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_score >= threshold:
            survived.append(new_rule)
            matched_old.add(best_idx)
        else:
            added.append(new_rule)

    dropped = [r for i, r in enumerate(old_rules) if i not in matched_old]
    return survived, added, dropped


def match_rules(
    old_rules: list[str],
    new_rules: list[str],
    threshold: float = 0.6,
) -> tuple[list[str], list[str], list[str]]:
    """Classify new_rules relative to old_rules.

    Returns:
        survived: rules in new_rules with a close match in old_rules
        added:    rules in new_rules with no close match in old_rules
        dropped:  rules in old_rules with no close match in new_rules
    """
    matched_old: set[int] = set()
    survived: list[str] = []
    added: list[str] = []

    for new_rule in new_rules:
        best_score = 0.0
        best_idx = -1
        for i, old_rule in enumerate(old_rules):
            s = _rule_similarity(old_rule, new_rule)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_score >= threshold:
            survived.append(new_rule)
            matched_old.add(best_idx)
        else:
            added.append(new_rule)

    dropped = [r for i, r in enumerate(old_rules) if i not in matched_old]
    return survived, added, dropped


def rule_accumulation(
    records: list[dict],
    component: str = "system_prompt",
    threshold: float = 0.6,
    embed_model: str | None = None,
) -> list[dict]:
    """Build per-iteration rule survival stats across accepted candidates.

    When embed_model is set (e.g. "Qwen/Qwen3-Embedding-4B"), uses cosine
    similarity on local HuggingFace embeddings instead of token-overlap fuzzy
    matching (recommended threshold: 0.80).

    Returns a list of dicts (one per accepted candidate) with keys:
        iteration, valset_score, n_rules, n_survived, n_added, n_dropped,
        survived_rules, added_rules, dropped_rules
    """
    accepted = sorted(
        [r for r in records if r.get("status") == "accepted"],
        key=lambda x: x.get("iteration", 0),
    )
    if not accepted:
        return []

    # Pre-embed all unique rules upfront so the model is only loaded once.
    if embed_model:
        # sorted(), not list(): set iteration order for strings varies between
        # processes (hash randomization), which would change how texts are grouped
        # into batches and therefore the embeddings themselves.
        all_texts = sorted({
            rule
            for rec in accepted
            for rule in (
                extract_rules_from_text(rec.get("old_texts", {}).get(component, ""))
                + extract_rules_from_text(rec.get("new_texts", {}).get(component, ""))
            )
        })
        print(f"Embedding {len(all_texts)} unique rules...")
        _embed_rules(all_texts, embed_model)
        print("Done.")

    results = []
    for rec in accepted:
        iteration = rec["iteration"]
        # Use the actual parent that was mutated, not the previous accepted candidate.
        parent_rules = extract_rules_from_text(rec.get("old_texts", {}).get(component, ""))
        curr_rules = extract_rules_from_text(rec.get("new_texts", {}).get(component, ""))

        if embed_model:
            survived, added, dropped = match_rules_semantic(
                parent_rules, curr_rules, embed_model, threshold
            )
        else:
            survived, added, dropped = match_rules(parent_rules, curr_rules, threshold)

        results.append({
            "iteration": iteration,
            "valset_score": rec.get("valset_score"),
            "parent_candidate_idx": rec.get("parent_candidate_idx"),
            "n_rules": len(curr_rules),
            "n_survived": len(survived),
            "n_added": len(added),
            "n_dropped": len(dropped),
            "survived_rules": survived,
            "added_rules": added,
            "dropped_rules": dropped,
        })
    return results


def print_rule_accumulation(
    records: list[dict],
    component: str = "system_prompt",
    threshold: float = 0.6,
    verbose: bool = False,
    embed_model: str | None = None,
) -> None:
    """Print rule survival analysis and save a stacked bar chart."""
    stats = rule_accumulation(records, component, threshold, embed_model)
    if not stats:
        print("No accepted candidates found.")
        return

    mode = f"semantic ({embed_model})" if embed_model else "fuzzy (token overlap)"
    print(f"\n{'='*70}")
    print(f"  Rule survival across accepted candidates  [{mode}, threshold={threshold}]")
    print(f"{'='*70}")
    print(f"  {'Iter':>5}  {'Parent':>7}  {'Valset':>7}  {'Total':>6}  {'Survived':>9}  {'Added':>6}  {'Dropped':>8}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*6}  {'─'*8}")
    for s in stats:
        vs = f"{s['valset_score']:.3f}" if s["valset_score"] is not None else "    —"
        parent = f"cand {s['parent_candidate_idx']}" if s.get("parent_candidate_idx") is not None else "  seed"
        print(
            f"  {s['iteration']:>5}  {parent:>7}  {vs:>7}  {s['n_rules']:>6}"
            f"  {s['n_survived']:>9}  {s['n_added']:>6}  {s['n_dropped']:>8}"
        )
        if verbose:
            for r in s["added_rules"]:
                display = r[:110] + "..." if len(r) > 110 else r
                print(f"    + {display}")
            for r in s["dropped_rules"]:
                display = r[:110] + "..." if len(r) > 110 else r
                print(f"    - {display}")

    # Stacked bar chart
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        iters = [s["iteration"] for s in stats]
        survived = np.array([s["n_survived"] for s in stats])
        added = np.array([s["n_added"] for s in stats])
        dropped = np.array([s["n_dropped"] for s in stats])
        valset = [s["valset_score"] or 0 for s in stats]

        x = np.arange(len(iters))
        width = 0.55

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(10, len(iters) * 0.65 + 2), 7),
            gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
        )

        ax1.bar(x, survived, width, label="Survived", color="#4caf50")
        ax1.bar(x, added, width, bottom=survived, label="Added", color="#2196f3")
        ax1.bar(x, dropped, width, bottom=survived + added, label="Dropped", color="#f44336", alpha=0.7)
        ax1.set_ylabel("Rule count", fontsize=16)
        ax1.set_title("Rule survival across accepted candidates", fontsize=17)
        ax1.legend(loc="upper left", fontsize=13)
        ax1.tick_params(axis="y", labelsize=11)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2.plot(x, valset, marker="o", color="black", linewidth=1.5, markersize=4)
        ax2.set_ylabel("Valset acc.", fontsize=16)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(i) for i in iters], fontsize=11)
        ax2.set_xlabel("Iteration", fontsize=16)
        ax2.tick_params(axis="y", labelsize=11)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        out = Path("rule_survival.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved chart to {out}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Rule specificity analysis
# ---------------------------------------------------------------------------

def check_rule_specificity(
    records: list[dict],
    embed_model: str,
    component: str = "system_prompt",
    threshold: float = 0.80,
) -> None:
    """Test whether added rules are more minibatch-specific than survived rules.

    For each accepted candidate:
      - Gets added_rules and survived_rules via semantic matching
      - Embeds the 3 minibatch problem texts
      - Computes max cosine similarity of each rule to any minibatch problem
      - Reports mean similarity for added vs survived rules

    If added rules score higher, they are more minibatch-specific (overfitted).
    """
    stats = rule_accumulation(records, component, threshold, embed_model)

    print(f"\n{'='*70}")
    print(f"  Rule specificity: similarity to minibatch problems")
    print(f"  (higher = more specific to the 3 minibatch examples)")
    print(f"{'='*70}")
    print(f"  {'Iter':>5}  {'Valset':>7}  {'Survived sim':>13}  {'Added sim':>10}  {'Δ (added−surv)':>15}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*13}  {'─'*10}  {'─'*15}")

    all_survived_sims: list[float] = []
    all_added_sims: list[float] = []

    accepted = {r["iteration"]: r for r in records if r.get("status") == "accepted"}

    for s in stats:
        rec = accepted.get(s["iteration"])
        if rec is None:
            continue

        # Extract minibatch problem texts
        minibatch_texts = [
            ex.get("Inputs", "")
            for examples in rec.get("reflective_dataset", {}).values()
            for ex in examples
            if ex.get("Inputs")
        ]
        if not minibatch_texts:
            continue

        # Embed minibatch problems and rules (cache handles repeats)
        mb_embs = _embed_rules(minibatch_texts, embed_model)

        def max_sim_to_minibatch(rule: str) -> float:
            rule_emb = _embed_rules([rule], embed_model)[0]
            return max(_cosine(rule_emb, mb_emb) for mb_emb in mb_embs)

        survived_sims = [max_sim_to_minibatch(r) for r in s["survived_rules"]] if s["survived_rules"] else []
        added_sims = [max_sim_to_minibatch(r) for r in s["added_rules"]] if s["added_rules"] else []

        surv_mean = sum(survived_sims) / len(survived_sims) if survived_sims else float("nan")
        added_mean = sum(added_sims) / len(added_sims) if added_sims else float("nan")
        delta = added_mean - surv_mean

        vs = f"{s['valset_score']:.3f}" if s["valset_score"] is not None else "    —"
        surv_str = f"{surv_mean:.3f}" if survived_sims else "     —"
        added_str = f"{added_mean:.3f}" if added_sims else "   —"
        delta_str = f"{delta:+.3f}" if (survived_sims and added_sims) else "      —"
        print(f"  {s['iteration']:>5}  {vs:>7}  {surv_str:>13}  {added_str:>10}  {delta_str:>15}")

        all_survived_sims.extend(survived_sims)
        all_added_sims.extend(added_sims)

    if all_survived_sims and all_added_sims:
        surv_overall = sum(all_survived_sims) / len(all_survived_sims)
        added_overall = sum(all_added_sims) / len(all_added_sims)
        print(f"\n  Overall mean similarity to minibatch:")
        print(f"    Survived rules : {surv_overall:.4f}")
        print(f"    Added rules    : {added_overall:.4f}")
        print(f"    Δ              : {added_overall - surv_overall:+.4f}")
        if added_overall > surv_overall:
            print(f"\n  ✓ Hypothesis supported: added rules are more minibatch-specific.")
        else:
            print(f"\n  ✗ Hypothesis not supported: survived rules are equally or more specific.")


def check_cross_iteration_specificity(
    records: list[dict],
    embed_model: str,
    component: str = "system_prompt",
    threshold: float = 0.80,
) -> None:
    """Cross-iteration specificity test for rule overfitting.

    For each rule, computes:
      own_sim   = max cosine similarity to its own minibatch problems
      other_sim = mean of (max cosine similarity to each OTHER iteration's minibatch)
      relative_specificity = own_sim - other_sim

    A positive relative_specificity means the rule is disproportionately similar
    to its own minibatch vs. minibatches in general. If added rules have higher
    relative_specificity than survived rules, that is strong evidence of overfitting.
    """
    stats = rule_accumulation(records, component, threshold, embed_model)
    accepted = {r["iteration"]: r for r in records if r.get("status") == "accepted"}

    # Collect minibatch texts per iteration
    mb_texts_by_iter: dict[int, list[str]] = {}
    for s in stats:
        rec = accepted.get(s["iteration"])
        if rec is None:
            continue
        texts = [
            ex.get("Inputs", "")
            for examples in rec.get("reflective_dataset", {}).values()
            for ex in examples
            if ex.get("Inputs")
        ]
        if texts:
            mb_texts_by_iter[s["iteration"]] = texts

    # Pre-embed all minibatch texts
    all_mb_texts = sorted({t for texts in mb_texts_by_iter.values() for t in texts})
    print(f"Embedding {len(all_mb_texts)} minibatch problem texts...")
    _embed_rules(all_mb_texts, embed_model)
    print("Done.")

    def max_sim_to(rule_emb: list[float], texts: list[str]) -> float:
        return max(_cosine(rule_emb, _embed_rules([t], embed_model)[0]) for t in texts)

    print(f"\n{'='*70}")
    print(f"  Cross-iteration specificity  (own minibatch vs. all other minibatches)")
    print(f"  relative_specificity = own_sim − other_sim  (higher = more overfitted)")
    print(f"{'='*70}")
    print(f"  {'Iter':>5}  {'Valset':>7}  {'Surv spec':>10}  {'Added spec':>11}  {'Δ (added−surv)':>15}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*10}  {'─'*11}  {'─'*15}")

    all_surv_spec: list[float] = []
    all_added_spec: list[float] = []
    other_iters_cache: dict[int, list[str]] = {}  # cached "all other" texts per iter

    for s in stats:
        iteration = s["iteration"]
        if iteration not in mb_texts_by_iter:
            continue

        own_texts = mb_texts_by_iter[iteration]
        other_texts = [t for it, texts in mb_texts_by_iter.items() if it != iteration for t in texts]
        if not other_texts:
            continue

        def relative_spec(rule: str) -> float:
            emb = _embed_rules([rule], embed_model)[0]
            own = max_sim_to(emb, own_texts)
            other = sum(max_sim_to(emb, mb_texts_by_iter[it])
                        for it in mb_texts_by_iter if it != iteration) / (len(mb_texts_by_iter) - 1)
            return own - other

        surv_specs = [relative_spec(r) for r in s["survived_rules"]] if s["survived_rules"] else []
        added_specs = [relative_spec(r) for r in s["added_rules"]] if s["added_rules"] else []

        surv_mean = sum(surv_specs) / len(surv_specs) if surv_specs else float("nan")
        added_mean = sum(added_specs) / len(added_specs) if added_specs else float("nan")
        delta = added_mean - surv_mean

        vs = f"{s['valset_score']:.3f}" if s["valset_score"] is not None else "    —"
        surv_str = f"{surv_mean:+.4f}" if surv_specs else "         —"
        added_str = f"{added_mean:+.4f}" if added_specs else "          —"
        delta_str = f"{delta:+.4f}" if (surv_specs and added_specs) else "              —"
        print(f"  {iteration:>5}  {vs:>7}  {surv_str:>10}  {added_str:>11}  {delta_str:>15}")

        all_surv_spec.extend(surv_specs)
        all_added_spec.extend(added_specs)

    if all_surv_spec and all_added_spec:
        surv_overall = sum(all_surv_spec) / len(all_surv_spec)
        added_overall = sum(all_added_spec) / len(all_added_spec)
        delta_overall = added_overall - surv_overall
        print(f"\n  Overall mean relative specificity:")
        print(f"    Survived rules : {surv_overall:+.4f}")
        print(f"    Added rules    : {added_overall:+.4f}")
        print(f"    Δ              : {delta_overall:+.4f}")
        if delta_overall > 0:
            print(f"\n  ✓ Added rules are more overfitted to their minibatch than survived rules.")
        else:
            print(f"\n  ✗ No evidence that added rules are more minibatch-specific than survived rules.")


# ---------------------------------------------------------------------------
# Feedback classification
# ---------------------------------------------------------------------------

FEEDBACK_PATTERNS: list[tuple[str, list[str]]] = [
    ("format_error", [
        r"output format", r"invalid output", r"structured output",
        r"missing.*###", r"expected.*###", r"###.*missing",
        r"json.*format", r"parse error", r"failed to parse",
        r"expected.*but got", r"wrong format",
    ]),
    ("wrong_answer", [
        r"incorrect", r"wrong answer", r"incorrect answer", r"expected.*got",
        r"does not match", r"off by", r"calculation",
    ]),
    ("reasoning_error", [
        r"reasoning", r"logic", r"step", r"approach", r"misunderstood",
        r"flawed", r"incorrect reasoning",
    ]),
    ("incomplete", [
        r"incomplete", r"stopped early", r"truncated", r"did not finish",
        r"partial", r"no answer",
    ]),
    ("hallucination", [
        r"hallucin", r"fabricat", r"not in the", r"not mentioned", r"made up",
    ]),
]


def classify_feedback(feedback_text: str) -> str:
    """Assign one of the FEEDBACK_PATTERNS categories (or 'other')."""
    text = feedback_text.lower()
    for category, patterns in FEEDBACK_PATTERNS:
        if any(re.search(p, text) for p in patterns):
            return category
    return "other"


def dominant_feedback_type(reflective_dataset: dict) -> str:
    """Return the most common feedback type across all examples and components."""
    counts: dict[str, int] = {}
    for examples in reflective_dataset.values():
        for ex in examples:
            fb = ex.get("Feedback", "") or ""
            cat = classify_feedback(fb)
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich(records: list[dict]) -> list[dict]:
    """Add computed fields to each record."""
    enriched = []
    for rec in records:
        r = dict(rec)
        old_texts = r.get("old_texts", {})
        new_texts = r.get("new_texts", {})

        # Compute per-component edit distances and take the max/mean
        edit_distances = {}
        for comp in set(list(old_texts) + list(new_texts)):
            old = old_texts.get(comp, "")
            new = new_texts.get(comp, "")
            edit_distances[comp] = token_edit_distance(old, new)

        r["edit_distance"] = (
            sum(edit_distances.values()) / len(edit_distances) if edit_distances else 0.0
        )
        r["edit_distances_per_component"] = edit_distances

        # Dominant feedback type
        r["dominant_feedback_type"] = dominant_feedback_type(r.get("reflective_dataset", {}))

        # Score delta on the minibatch
        before = r.get("subsample_score_before")
        after = r.get("subsample_score_after")
        if before is not None and after is not None:
            r["subsample_score_delta"] = after - before
        else:
            r["subsample_score_delta"] = None

        enriched.append(r)
    return enriched


# ---------------------------------------------------------------------------
# Summary statistics (no pandas dependency required)
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        v = str(r.get(key, "unknown"))
        groups.setdefault(v, []).append(r)
    return groups


def print_summary(records: list[dict], label: str = "") -> None:
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    accepted = [r for r in records if r.get("status") == "accepted"]
    rejected = [r for r in records if r.get("status") == "rejected"]

    print(f"\nTotal proposals : {len(records)}")
    print(f"  Accepted      : {len(accepted)}")
    print(f"  Rejected      : {len(rejected)}")
    if records:
        print(f"  Accept rate   : {len(accepted)/len(records)*100:.1f}%")

    # Edit distance
    ed_acc = [r["edit_distance"] for r in accepted if r.get("edit_distance") is not None]
    ed_rej = [r["edit_distance"] for r in rejected if r.get("edit_distance") is not None]
    print(f"\nEdit distance (fraction of tokens changed):")
    print(f"  Accepted mean : {_mean(ed_acc):.3f}")
    print(f"  Rejected mean : {_mean(ed_rej):.3f}")

    # Score delta
    delta_acc = [r["subsample_score_delta"] for r in accepted if r.get("subsample_score_delta") is not None]
    delta_rej = [r["subsample_score_delta"] for r in rejected if r.get("subsample_score_delta") is not None]
    print(f"\nMinibatch score delta (after - before):")
    print(f"  Accepted mean : {_mean(delta_acc):+.3f}")
    print(f"  Rejected mean : {_mean(delta_rej):+.3f}")

    # Acceptance rate by feedback type
    print(f"\nAcceptance rate by dominant feedback type:")
    by_fb = _group_by(records, "dominant_feedback_type")
    for fb_type in sorted(by_fb):
        group = by_fb[fb_type]
        acc_count = sum(1 for r in group if r.get("status") == "accepted")
        print(f"  {fb_type:<20}: {acc_count}/{len(group)} = {acc_count/len(group)*100:.1f}%")

    # Valset learning curve (accepted proposals only)
    if accepted:
        print(f"\nValset score trajectory (accepted proposals):")
        print(f"  {'Iter':>6}  {'Valset score':>12}  {'Best?':>6}")
        for r in sorted(accepted, key=lambda x: x.get("iteration", 0)):
            vs = r.get("valset_score")
            best = r.get("is_best", False)
            vs_str = f"{vs:.4f}" if vs is not None else "    —"
            best_str = "✓" if best else ""
            print(f"  {r['iteration']:>6}  {vs_str:>12}  {best_str:>6}")

    # Feedback type distribution across all examples
    print(f"\nFeedback type distribution (all proposals):")
    fb_counts: dict[str, int] = {}
    for r in records:
        fb_counts[r.get("dominant_feedback_type", "unknown")] = (
            fb_counts.get(r.get("dominant_feedback_type", "unknown"), 0) + 1
        )
    total = sum(fb_counts.values())
    for fb_type in sorted(fb_counts, key=lambda k: -fb_counts[k]):
        pct = fb_counts[fb_type] / total * 100 if total > 0 else 0
        print(f"  {fb_type:<20}: {fb_counts[fb_type]:>4} ({pct:.1f}%)")


def print_prompt_diffs(records: list[dict], n: int = 3) -> None:
    """Print the diffs for the first N accepted proposals."""
    accepted = [r for r in records if r.get("status") == "accepted"]
    print(f"\n{'='*60}")
    print(f"  Prompt diffs for first {min(n, len(accepted))} accepted proposals")
    print(f"{'='*60}")
    for r in accepted[:n]:
        print(f"\n--- Iteration {r['iteration']} | valset_score={r.get('valset_score')} ---")
        for comp in r.get("components", list(r.get("old_texts", {}).keys())):
            old = r.get("old_texts", {}).get(comp, "")
            new = r.get("new_texts", {}).get(comp, "")
            diff = unified_diff_summary(old, new)
            print(f"  Component: {comp}")
            print(f"  Edit distance: {r['edit_distances_per_component'].get(comp, '?'):.3f}")
            print(f"  Lines added: {diff['added_lines']} | Lines removed: {diff['removed_lines']}")
            if diff["diff"]:
                # Print up to 20 lines of the diff
                diff_lines = diff["diff"].splitlines()[:20]
                for line in diff_lines:
                    print(f"    {line}")
                if len(diff["diff"].splitlines()) > 20:
                    print(f"    ... ({len(diff['diff'].splitlines()) - 20} more lines)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GEPA research logs")
    parser.add_argument("run_dirs", nargs="+", help="One or more run directories containing research_log.jsonl")
    parser.add_argument("--diffs", action="store_true", help="Print prompt diffs for first 3 accepted proposals")
    parser.add_argument("--n-diffs", type=int, default=3, help="Number of accepted proposal diffs to show (default: 3)")
    parser.add_argument("--rules", action="store_true", help="Print added rules for each accepted proposal")
    parser.add_argument("--survival", action="store_true", help="Track rule survival across accepted candidates")
    parser.add_argument("--survival-verbose", action="store_true", help="Also print added/dropped rule text")
    parser.add_argument("--survival-threshold", type=float, default=None, help="Similarity threshold (default: 0.6 fuzzy, 0.80 semantic)")
    parser.add_argument("--embed-model", default=None, help="HuggingFace model for semantic similarity, e.g. Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--specificity", action="store_true", help="Test whether added rules are more minibatch-specific than survived rules (requires --embed-model)")
    parser.add_argument("--cross-specificity", action="store_true", help="Cross-iteration specificity test: own minibatch vs all other minibatches (requires --embed-model)")
    args = parser.parse_args()
    # Default threshold depends on mode
    if args.survival_threshold is None:
        args.survival_threshold = 0.80 if args.embed_model else 0.6

    all_runs: list[tuple[str, list[dict]]] = []
    for run_dir in args.run_dirs:
        records = load_run(run_dir)
        enriched = enrich(records)
        all_runs.append((run_dir, enriched))

    for run_dir, records in all_runs:
        print_summary(records, label=run_dir)
        if args.diffs:
            print_prompt_diffs(records, n=args.n_diffs)
        if args.rules:
            print_rule_analysis(records)
        if args.specificity:
            if not args.embed_model:
                print("--specificity requires --embed-model")
            else:
                check_rule_specificity(records, args.embed_model, threshold=args.survival_threshold)
        if args.cross_specificity:
            if not args.embed_model:
                print("--cross-specificity requires --embed-model")
            else:
                check_cross_iteration_specificity(records, args.embed_model, threshold=args.survival_threshold)
        if args.survival or args.survival_verbose:
            print_rule_accumulation(
                records,
                threshold=args.survival_threshold,
                verbose=args.survival_verbose,
                embed_model=args.embed_model,
            )

    if len(all_runs) > 1:
        all_records = []
        for _, records in all_runs:
            all_records.extend(records)
        print_summary(all_records, label="COMBINED (all runs)")


if __name__ == "__main__":
    main()
