# Paper revision analyses

This directory contains the saved-data checks and corrected ablation runner used
for the revised minibatch study. The configurable public experiment runners live
in `src/minibatch_apo/`.

- `run_audit.py` checks candidate selection and computes equal-size,
  disjoint-reference, threshold, and source-permutation specificity results.
- `margin_analysis.py` measures within-run acceptance-margin associations.
- `rerun_ablations.py` selects final-best candidates, checkpoints individual
  outcomes, and reuses the three ablations that already evaluated those prompts.
  It makes paid API calls only with `--execute`.
- `summarize_corrected.py`, `extended_corrected.py`, and `proxy_corrected.py`
  summarize the corrected rule evaluations.

Released numerical results are under `results/corrected_ablations/` and
`results/review_audit/`. Post-hoc scripts use `outputs/` for local analyses of raw
experiment inputs. The audit results describe the original-result checks;
the corrected ablation summaries supersede the original ablations. The updated
oracle plot is `figures/oracle_pruning.pdf`.

Raw transcripts, prompt-bearing checkpoints, embedding caches, validation data,
and credentials are not included in the release. Most post-hoc scripts require
those locally saved experiment inputs. Analysis dependencies include NumPy,
SciPy, Matplotlib, and, for local embeddings, PyTorch and Transformers. The paid
runner also uses python-dotenv to read a local `.env` file.
