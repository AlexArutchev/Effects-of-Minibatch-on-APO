"""Checks for highest-score selection and resumable rule evaluations."""

import json

import pytest

from minibatch_apo.paper.ablate_rules import load_best_prompt


def test_selects_highest_scoring_candidate(tmp_path):
    records = [
        {"status": "accepted", "is_best": True, "valset_score": 0.6, "new_texts": {"system_prompt": "early"}},
        {"status": "accepted", "is_best": True, "valset_score": 0.8, "new_texts": {"system_prompt": "best"}},
        {"status": "accepted", "is_best": False, "valset_score": 0.5, "new_texts": {"system_prompt": "later"}},
    ]
    (tmp_path / "research_log.jsonl").write_text("\n".join(map(json.dumps, records)))
    assert load_best_prompt(str(tmp_path)) == "best"


def test_rejected_candidate_cannot_be_selected(tmp_path):
    records = [
        {"status": "accepted", "valset_score": 0.7, "new_texts": {"system_prompt": "accepted"}},
        {"status": "rejected", "valset_score": 0.9, "new_texts": {"system_prompt": "rejected"}},
    ]
    (tmp_path / "research_log.jsonl").write_text("\n".join(map(json.dumps, records)))
    assert load_best_prompt(str(tmp_path)) == "accepted"


def test_no_accepted_candidate_fails(tmp_path):
    (tmp_path / "research_log.jsonl").write_text("")
    with pytest.raises(ValueError, match="no accepted"):
        load_best_prompt(str(tmp_path))


def test_checkpoint_resume_avoids_repeating_paid_calls(tmp_path):
    from analysis.paper.evaluate_rules import evaluate

    calls = []

    def fake_api(prompt, example, idx, label):
        calls.append(idx)
        return {"evaluation": label, "problem_idx": idx, "score": idx % 2, "estimated_cost_usd": 0.001}

    examples = [{"input": "question", "answer": "### 1", "additional_context": {}} for _ in range(3)]
    checkpoint = tmp_path / "outcomes.jsonl"
    budget = {"spent": 0.0, "limit": 1.0}
    assert evaluate("prompt", examples, "baseline", checkpoint, budget, 2, fake_api) == [0, 1, 0]
    assert len(calls) == 3
    assert evaluate("prompt", examples, "baseline", checkpoint, budget, 2, fake_api) == [0, 1, 0]
    assert len(calls) == 3


def test_budget_guard_prevents_calls(tmp_path):
    from analysis.paper.evaluate_rules import evaluate

    def forbidden_api(*args):
        pytest.fail("API called despite exhausted budget")

    with pytest.raises(RuntimeError, match="Cost cap"):
        evaluate(
            "prompt",
            [{"input": "question"}],
            "baseline",
            tmp_path / "outcomes.jsonl",
            {"spent": 0.01, "limit": 0.01},
            1,
            forbidden_api,
        )


def test_duplicate_checkpoint_is_rejected(tmp_path):
    from analysis.paper.evaluate_rules import checkpoint_rows

    row = {"evaluation": "baseline", "problem_idx": 0, "score": 1, "estimated_cost_usd": 0.001}
    checkpoint = tmp_path / "outcomes.jsonl"
    checkpoint.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        checkpoint_rows(checkpoint)


def test_plan_checks_prompt_and_validation_before_using_cache(tmp_path, monkeypatch):
    from analysis.paper import evaluate_rules

    monkeypatch.setattr(evaluate_rules, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_rules, "CONDITIONS", ["real100"])
    validation = [{"input": "question", "answer": "### 1", "additional_context": {}}]
    prompt = "Solve carefully.\n- Check the answer."
    for seed in range(5):
        folder = tmp_path / f"runs/augmentation_sweep/real100_s{seed}"
        folder.mkdir(parents=True)
        record = {
            "status": "accepted",
            "valset_score": 0.8,
            "iteration": 2,
            "new_candidate_idx": 1,
            "new_texts": {"system_prompt": prompt},
        }
        (folder / "research_log.jsonl").write_text(json.dumps(record) + "\n")
        if seed in [1, 2]:
            cached = {
                "rules": [{"text": "- Check the answer."}],
                "n_val": 1,
                "validation_sha256": evaluate_rules.digest(validation),
                "prompt_sha256": evaluate_rules.digest(prompt if seed == 1 else "Different preamble."),
            }
            (folder / "rule_utility.json").write_text(json.dumps(cached))
    plan = evaluate_rules.build_plan(validation)
    assert len(plan) == 5
    assert [item["needs_evaluation"] for item in plan] == [True, False, True, True, True]
