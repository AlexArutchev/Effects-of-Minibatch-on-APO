# Effects of Minibatch Size on Automatic Prompt Optimization

Reproduction code for experiments on how a reflection minibatch affects the
updates produced by automatic prompt optimization. The repository contains only
the study code; it uses [GEPA](https://github.com/gepa-ai/gepa) as an installed
dependency and does not vendor GEPA's framework source, documentation, or
benchmarks. Released numerical summaries are under `results/`; raw experiment
outputs remain local.

## What is included

- Configurable diversity, factorial, and augmentation protocols
- Deterministic MATH split construction
- Compact, privacy-preserving study logs
- Aggregate, rule-specificity, leave-one-out utility, and oracle-pruning analyses
- JSON templates for the configurations used in the study

No API keys, model transcripts, generated pools, raw runs, or paper drafts are
committed. Those remain local by design.

## Revised paper results

The revised analyses include final-best candidate selection, corrected rule
ablations, specificity controls for comparison size and batch overlap, threshold
checks, and acceptance-margin analyses. The selected ablation candidate now has
the highest logged validation score, rather than being the first new record.

- `results/corrected_ablations/` contains the corrected ten-run summaries and
  per-run oracle results. Seven ablations were replaced and three were reused.
- `results/review_audit/` contains the saved-data audit and specificity checks.
- `analysis/review_audit/` contains the analysis scripts and regression tests.
- `src/minibatch_apo/paper/` contains the original study helpers needed by those
  scripts. The configurable runners below remain the entry points for new runs.
- `figures/oracle_pruning.pdf` is the updated pruning figure.

Rebuild that figure using only the released numerical results:

```bash
uv sync --extra dev --extra analysis
uv run python analysis/review_audit/rebuild_oracle.py
uv run python -m pytest tests analysis/review_audit
```

Most other post-hoc scripts require locally saved `research_log.jsonl` files from
the original protocols, including prompt text where needed. Those inputs and
embedding caches are not distributed. Install the `embeddings` extra for the
Qwen embedding analyses. Paid ablation reruns require a locally configured API
key and an explicit `--execute` argument; the default is a dry run.

## Setup

```bash
uv sync --extra dev
export OPENAI_API_KEY=...
```

The defaults use LiteLLM model strings. Change the model names, rollout budget,
dataset sizes, seeds, output path, and experiment grid in the JSON config files.

## Run an experiment

Inspect the diversity grid before incurring model costs:

```bash
uv run python -m minibatch_apo.run_diversity \
  src/minibatch_apo/configs/diversity.json --all --dry-run
```

Remove `--dry-run` to execute the conditions. Use `--smoke-test` for a small
run, or `--condition d4_s0` to run a named condition. The factorial and
augmentation protocols work the same way:

```bash
uv run python -m minibatch_apo.run_factorial \
  src/minibatch_apo/configs/factorial.json --all

uv run python -m minibatch_apo.run_augmentation \
  src/minibatch_apo/configs/augmentation.json --all
```

The augmentation runner creates baseline, duplicate, and real-data controls
itself. For paraphrase and synthetic arms, provide local JSON pools as described
in the augmentation config; this avoids hidden model calls while training.

## Analyze results

```bash
uv run python -m minibatch_apo.analyze outputs/prompt_optimization_study/factorial
```

Logs contain aggregate metrics only by default. Prompt text, reflection examples,
and raw model outputs are stored only if the configuration explicitly enables
them and acknowledges the privacy risk. That opt-in is needed for the
rule-specificity and leave-one-out analyses.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright src
```
