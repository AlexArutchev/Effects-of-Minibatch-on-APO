"""Rule extraction and cross-iteration specificity measurements."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

_RULE_START = re.compile(r"^\s*(-\s+|\*\s+|\d+\.\s+|\*\*[^*]+\*\*)")


def extract_rules(prompt: str) -> list[str]:
    """Split a prompt into bullet-like rules, retaining prose lines with a rule."""
    rules: list[str] = []
    current: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _RULE_START.match(line):
            if current:
                rules.append(" ".join(current))
            current = [stripped]
        elif current:
            current.append(stripped)
        elif stripped.endswith((".", ":")):
            rules.append(stripped)
    if current:
        rules.append(" ".join(current))
    return rules


def added_rules(old_prompt: str, new_prompt: str, threshold: float = 0.8) -> list[str]:
    """Return rules in the update that do not closely match a prior rule."""
    old_rules = extract_rules(old_prompt)
    return [
        rule
        for rule in extract_rules(new_prompt)
        if not any(_similarity(rule, old_rule) >= threshold for old_rule in old_rules)
    ]


def cross_iteration_specificity(
    records: Iterable[dict], component: str = "system_prompt"
) -> list[dict[str, float | int]]:
    """Measure how much added rules target later reflection examples.

    Specificity is the fraction of later reflection examples whose text contains
    at least one content token from an added rule. This lightweight lexical
    proxy makes the analysis runnable without downloading an embedding model.
    Users reproducing the paper's embedding analysis can replace ``_similarity``
    with an embedding-based matcher while retaining the same input schema.
    """
    ordered = sorted(records, key=lambda record: int(record.get("iteration", 0)))
    output: list[dict[str, float | int]] = []
    for index, record in enumerate(ordered[:-1]):
        old_prompt = record.get("old_prompt", {}).get(component, "")
        new_prompt = record.get("new_prompt", {}).get(component, "")
        rules = added_rules(old_prompt, new_prompt)
        later_dataset = ordered[index + 1].get("reflective_dataset", {}).get(component, [])
        examples = [str(item.get("Inputs", "")) for item in later_dataset if isinstance(item, dict)]
        if not rules or not examples:
            continue
        match_rate = sum(any(_targets(rule, example) for rule in rules) for example in examples) / len(examples)
        output.append(
            {"iteration": int(record.get("iteration", 0)), "added_rules": len(rules), "later_match_rate": match_rate}
        )
    return output


def _similarity(first: str, second: str) -> float:
    return difflib.SequenceMatcher(None, first.lower().split(), second.lower().split()).ratio()


def _targets(rule: str, example: str) -> bool:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]{4,}", rule)}
    example_tokens = {token.lower() for token in re.findall(r"[A-Za-z]{4,}", example)}
    return bool(tokens & example_tokens)
