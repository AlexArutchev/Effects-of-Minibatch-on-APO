# The Effects of Minibatches on Automatic Prompt Optimization

Code and numerical results for the study of minibatch effects in GEPA on MATH
levels 4–5, using GPT-4.1-mini as the task model and GPT-4.1 for reflection.
GEPA is installed as a dependency; this repository contains the study protocols,
analysis code, configurations, and results.

## Key results

- On the same 100-problem training pool, increasing minibatch size from 10 to 50
  raises the mean fraction of accepted updates that improve validation accuracy
  from **63.5% to 90.0%**. This does not consistently improve final accuracy under
  a fixed evaluation budget.
- Added rules have lower source-batch specificity at the larger minibatch size,
  including after controlling comparison size and reference-batch overlap.
- Subject diversity, a larger training pool at fixed minibatch size, and repeated
  copies of the same problems do not show the same transfer improvement.
- Across **295 rules** in ten highest-scoring prompts, specificity has little
  association with leave-one-out utility. Joint deletion of rules with negative
  measured utility provides no average validation improvement.

The same minibatch supplies reflection feedback and candidate acceptance scores.
The experiments do not separate those two mechanisms. The validation/Pareto set
also influences search and is not an independent test set.

## Repository contents

- `src/minibatch_apo/`: configurable experiment runners and analysis tools.
- `src/minibatch_apo/paper/`: helpers for the paper's saved-log analyses.
- `analysis/paper/`: specificity controls, acceptance-margin analysis, rule
  evaluation, plotting, and tests.
- `results/rule_ablations/`: numerical rule-utility and oracle-pruning results.
- `results/specificity_checks/`: specificity and accepted-update summaries.
- `figures/`: paper figures generated from the numerical results.

## Setup

```bash
uv sync --extra dev --extra analysis
export OPENAI_API_KEY=...
```

Model names, budgets, dataset sizes, seeds, and experiment grids are configurable
in `src/minibatch_apo/configs/`. Embedding analyses additionally require
`uv sync --extra dev --extra analysis --extra embeddings`.

## Run experiments

Inspect the diversity grid without making model calls:

```bash
uv run python -m minibatch_apo.run_diversity \
  src/minibatch_apo/configs/diversity.json --all --dry-run
```

Remove `--dry-run` to execute it. Use `--smoke-test` for a small run or
`--condition d4_s0` for one condition. Other protocols run with:

```bash
uv run python -m minibatch_apo.run_factorial \
  src/minibatch_apo/configs/factorial.json --all

uv run python -m minibatch_apo.run_augmentation \
  src/minibatch_apo/configs/augmentation.json --all
```

The augmentation runner constructs baseline, duplicate, and real-data conditions.
Paraphrase and synthetic conditions use local pools specified in the configuration.

## Analyze results

```bash
uv run python -m minibatch_apo.analyze outputs/prompt_optimization_study/factorial
uv run python analysis/paper/rebuild_oracle.py
```

The figure command uses released numerical results and makes no model calls.
The paper's embedding analyses use Qwen3-Embedding-4B and locally saved prompt
and reflection texts. Logs omit those texts by default; enable the corresponding
logging options when collecting data for rule-level analysis. Raw transcripts,
benchmark pools, embedding caches, and credentials are not included.

See `analysis/paper/README.md` for the analysis scripts. The resumable rule
evaluator selects each run's highest-scoring candidate and requires `--execute`
for paid evaluations; otherwise it prints the evaluation plan.

## Tests

```bash
uv run python -m pytest tests analysis/paper
```
