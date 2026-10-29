# Paper analyses

These scripts analyze saved experiment logs and evaluate rules in each run's
highest-scoring prompt. Configurable experiment runners live in
`src/minibatch_apo/`.

- `analyze_specificity.py`: paired run statistics, equal-size and disjoint-reference
  specificity, matching thresholds, and source-batch permutations.
- `margin_analysis.py`: acceptance-margin associations within runs.
- `evaluate_rules.py`: resumable leave-one-out evaluations and joint rule deletion.
- `summarize_rules.py`: rule-utility correlations and per-seed summaries.
- `analyze_rule_statistics.py`: specificity spread, entropy, and utility comparisons.
- `analyze_rule_proxy.py`: training-minus-validation specificity proxy.
- `rebuild_oracle.py`: oracle-pruning figure from released numerical results.

Released summaries are in `results/rule_ablations/` and
`results/specificity_checks/`. Scripts processing local experiment inputs write
under `outputs/` and read the paper protocols' `research_log.jsonl` schema. These
raw inputs include prompt text where needed and are not distributed.

Run `evaluate_rules.py` without flags to inspect its plan. Paid evaluations
require `--execute` and an API key supplied through the environment or a local
`.env` file. The runner selects candidates by their highest logged validation
score, saves per-problem outcomes, and skips matching cached evaluations.
