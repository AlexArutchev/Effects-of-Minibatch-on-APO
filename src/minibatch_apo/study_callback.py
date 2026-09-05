"""Callback that records aggregate study outcomes without storing model text by default."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    EvaluationEndEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    ValsetEvaluatedEvent,
)


class StudyCallback:
    """Write one compact, privacy-preserving JSONL record for each proposal."""

    def __init__(
        self,
        run_dir: Path,
        *,
        include_prompt_text: bool = False,
        include_reflective_dataset: bool = False,
        include_raw_model_outputs: bool = False,
    ) -> None:
        self._path = run_dir / "study_log.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._include_prompt_text = include_prompt_text
        self._include_reflective_dataset = include_reflective_dataset
        self._include_raw_model_outputs = include_raw_model_outputs
        self._records: dict[int, dict[str, Any]] = {}
        self._evaluation_counts: dict[int, int] = {}
        self._metric_calls = 0

    def _record(self, iteration: int) -> dict[str, Any]:
        if iteration not in self._records:
            self._records[iteration] = {"iteration": iteration}
            self._evaluation_counts[iteration] = 0
        return self._records[iteration]

    def _flush(self, iteration: int) -> None:
        record = self._records.pop(iteration, None)
        self._evaluation_counts.pop(iteration, None)
        if record is None:
            return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        self._metric_calls = event["metric_calls_used"]

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        record = self._record(event["iteration"])
        record["parent_candidate_idx"] = event["candidate_idx"]
        record["parent_score"] = event["score"]

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        if self._include_prompt_text:
            self._record(event["iteration"])["old_prompt"] = dict(event["parent_candidate"])

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        if self._include_reflective_dataset:
            self._record(event["iteration"])["reflective_dataset"] = event["dataset"]

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        if event["is_seed_candidate"]:
            return
        iteration = event["iteration"]
        record = self._record(iteration)
        count = self._evaluation_counts[iteration]
        if count == 0:
            record["subsample_score_before"] = sum(event["scores"])
        elif count == 1:
            record["subsample_score_after"] = sum(event["scores"])
        self._evaluation_counts[iteration] = count + 1

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        record = self._record(event["iteration"])
        if self._include_prompt_text:
            record["new_prompt"] = dict(event["new_instructions"])
        if self._include_raw_model_outputs:
            record["raw_model_outputs"] = dict(event["raw_lm_outputs"])

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        if event["iteration"]:
            record = self._record(event["iteration"])
            record["valset_score"] = event["average_score"]
            record["is_best"] = event["is_best_program"]

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        record = self._record(event["iteration"])
        record["status"] = "accepted"
        record["new_candidate_idx"] = event["new_candidate_idx"]
        record["metric_calls"] = self._metric_calls
        self._flush(event["iteration"])

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        record = self._record(event["iteration"])
        record["status"] = "rejected"
        record["subsample_score_before"] = event["old_score"]
        record["subsample_score_after"] = event["new_score"]
        record["reject_reason"] = event["reason"]
        record["metric_calls"] = self._metric_calls
        self._flush(event["iteration"])
