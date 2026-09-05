from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_expand_conditions_covers_each_public_protocol() -> None:
    from minibatch_apo.config import StudyConfig
    from minibatch_apo.experiments import expand_conditions

    diversity = StudyConfig.from_dict(
        {"experiment": {"kind": "diversity", "diversity_levels": [1, 3], "seeds": [0, 2]}}
    )
    factorial = StudyConfig.from_dict(
        {"experiment": {"kind": "factorial", "diversity_levels": [1, 2], "minibatch_sizes": [3, 10], "seeds": [0]}}
    )
    augmentation = StudyConfig.from_dict(
        {"experiment": {"kind": "augmentation", "augmentation_arms": ["baseline", "duplicate"], "seeds": [1]}}
    )

    assert [condition.label for condition in expand_conditions(diversity)] == ["d1_s0", "d1_s2", "d3_s0", "d3_s2"]
    assert [condition.label for condition in expand_conditions(factorial)] == [
        "d1_m3_s0",
        "d1_m10_s0",
        "d2_m3_s0",
        "d2_m10_s0",
    ]
    assert [condition.label for condition in expand_conditions(augmentation)] == ["baseline_s1", "duplicate_s1"]


def test_config_rejects_raw_logs_without_explicit_consent() -> None:
    from minibatch_apo.config import StudyConfig

    with pytest.raises(ValueError, match="consent"):
        StudyConfig.from_dict({"logging": {"include_raw_model_outputs": True}})


def test_summary_callback_omits_text_and_raw_outputs_by_default(tmp_path: Path) -> None:
    from minibatch_apo.study_callback import StudyCallback

    callback = StudyCallback(tmp_path)
    callback.on_candidate_selected(
        {"iteration": 1, "candidate_idx": 0, "candidate": {"system_prompt": "private"}, "score": 0.2}
    )
    callback.on_evaluation_end(
        {
            "iteration": 1,
            "candidate_idx": 0,
            "scores": [0.1, 0.2],
            "has_trajectories": False,
            "parent_ids": [],
            "outputs": [],
            "trajectories": None,
            "objective_scores": None,
            "is_seed_candidate": False,
        }
    )
    callback.on_proposal_end(
        {
            "iteration": 1,
            "new_instructions": {"system_prompt": "new private text"},
            "prompts": {},
            "raw_lm_outputs": {"system_prompt": "raw private response"},
        }
    )
    callback.on_candidate_rejected({"iteration": 1, "old_score": 0.3, "new_score": 0.1, "reason": "no improvement"})

    record = json.loads((tmp_path / "study_log.jsonl").read_text())
    assert record == {
        "iteration": 1,
        "parent_candidate_idx": 0,
        "parent_score": 0.2,
        "subsample_score_after": 0.1,
        "subsample_score_before": 0.3,
        "reject_reason": "no improvement",
        "status": "rejected",
        "metric_calls": 0,
    }


def test_external_augmentation_pool_must_have_the_expected_size(tmp_path: Path) -> None:
    from minibatch_apo.augmentation import load_external_pool

    path = tmp_path / "pool.json"
    path.write_text(json.dumps([{"input": "x", "answer": "### 1"}]))

    with pytest.raises(ValueError, match="expected 2 examples"):
        load_external_pool(path, expected_size=2)


def test_rule_utility_helpers_preserve_the_preamble_and_summarize_negative_rules() -> None:
    from minibatch_apo.rule_evaluation import (
        assemble_without,
        split_rule_blocks,
        utility_summary,
    )

    preamble, rules = split_rule_blocks("Solve carefully.\n- Check units.\n- State the final answer.")

    assert assemble_without(preamble, rules, {0}) == "Solve carefully.\n- State the final answer."
    assert utility_summary([0.2, -0.1, -0.3]) == {
        "rule_count": 3,
        "negative_rule_count": 2,
        "sum_negative_utility": 0.4,
    }
