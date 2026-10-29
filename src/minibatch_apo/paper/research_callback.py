"""ResearchCallback: per-iteration logging for prompt evolution analysis.

Writes one JSONL record per iteration to {run_dir}/research_log.jsonl.
Each record contains the reflective dataset (actual feedback shown to the
reflection LM), old and new prompt texts, subsample scores before and after,
the acceptance decision, and the valset score for accepted proposals.

Usage::

    from minibatch_apo.paper.research_callback import ResearchCallback
    import gepa

    result = gepa.optimize(
        seed_candidate={"system_prompt": "..."},
        trainset=trainset,
        valset=valset,
        run_dir="runs/my_run",
        callbacks=[ResearchCallback("runs/my_run")],
        ...
    )
"""

import json
import os
from typing import Any

from gepa.core.callbacks import (
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    EvaluationEndEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    ValsetEvaluatedEvent,
)


class ResearchCallback:
    """Logs per-iteration data to a JSONL file for prompt evolution research.

    Each JSONL record corresponds to one reflective mutation attempt and
    contains everything needed to analyze what drove the proposal and whether
    it was accepted:

    - ``iteration``: GEPA iteration number
    - ``status``: ``"accepted"`` or ``"rejected"``
    - ``parent_candidate_idx``: index of the parent candidate that was mutated
    - ``parent_score``: parent's aggregate valset score
    - ``subsample_score_before``: sum of minibatch scores for the parent
    - ``subsample_score_after``: sum of minibatch scores for the proposed candidate
    - ``valset_score``: aggregate valset score of the new candidate (accepted only)
    - ``is_best``: whether this candidate became the new best (accepted only)
    - ``components``: list of component names that were updated
    - ``old_texts``: dict of component name → old text (parent's)
    - ``new_texts``: dict of component name → proposed text
    - ``reflective_dataset``: dict of component name → list of feedback records
      (each record has ``Inputs``, ``Generated Outputs``, ``Feedback``)
    - ``raw_lm_outputs``: dict of component name → raw reflection LM response
    """

    def __init__(self, run_dir: str) -> None:
        self._path = os.path.join(run_dir, "research_log.jsonl")
        os.makedirs(run_dir, exist_ok=True)
        # Buffer keyed by iteration number, supports parallel proposals
        self._buffer: dict[int, dict[str, Any]] = {}
        # Track evaluation order per iteration: first eval=parent, second=proposed
        self._eval_count: dict[int, int] = {}

    def _get(self, iteration: int) -> dict[str, Any]:
        if iteration not in self._buffer:
            self._buffer[iteration] = {"iteration": iteration}
            self._eval_count[iteration] = 0
        return self._buffer[iteration]

    def _flush(self, iteration: int) -> None:
        record = self._buffer.pop(iteration, None)
        self._eval_count.pop(iteration, None)
        if record is None:
            return
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        rec = self._get(event["iteration"])
        rec["parent_candidate_idx"] = event["candidate_idx"]
        rec["parent_score"] = event["score"]

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        rec = self._get(event["iteration"])
        rec["components"] = event["components"]
        rec["reflective_dataset"] = event["dataset"]

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        rec = self._get(event["iteration"])
        rec["old_texts"] = dict(event["parent_candidate"])

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        iteration = event["iteration"]
        # Skip seed candidate evaluation (iteration 0 is not a reflective mutation)
        if event["is_seed_candidate"]:
            return
        rec = self._get(iteration)
        count = self._eval_count.get(iteration, 0)
        if count == 0:
            # First eval in this iteration = parent candidate
            rec["subsample_score_before"] = sum(event["scores"])
        elif count == 1:
            # Second eval = proposed candidate
            rec["subsample_score_after"] = sum(event["scores"])
        self._eval_count[iteration] = count + 1

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        rec = self._get(event["iteration"])
        rec["new_texts"] = dict(event["new_instructions"])
        rec["raw_lm_outputs"] = dict(event["raw_lm_outputs"])

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        # Iteration 0 is the seed candidate baseline — skip
        if event["iteration"] == 0:
            return
        rec = self._get(event["iteration"])
        rec["valset_score"] = event["average_score"]
        rec["is_best"] = event["is_best_program"]

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        rec = self._get(event["iteration"])
        rec["status"] = "accepted"
        rec["new_candidate_idx"] = event["new_candidate_idx"]
        self._flush(event["iteration"])

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        rec = self._get(event["iteration"])
        rec["status"] = "rejected"
        # on_candidate_rejected always carries both scores; use them as ground truth
        rec["subsample_score_before"] = event["old_score"]
        rec["subsample_score_after"] = event["new_score"]
        rec["reject_reason"] = event["reason"]
        self._flush(event["iteration"])
