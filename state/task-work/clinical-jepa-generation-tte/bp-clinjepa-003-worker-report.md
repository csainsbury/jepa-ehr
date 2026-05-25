# BP-CLINJEPA-003 worker report — horizon-conditioned autoregression repair

Date: 2026-05-25

## Design

Implemented a minimal safe-public horizon-conditioned mean-token path for v0B latent autoregression.

- Backward-compatible default: old checkpoints with only `embedding` + `predictor` metadata load as `recursive` and keep the previous recursive rollout behavior.
- New explicit mode: checkpoints may declare `autoregression_mode: horizon_conditioned` with `max_horizons` and `horizon_count_trained` metadata.
- Horizon-conditioned baseline: shared mean-token context encoder plus horizon-specific linear heads, one per future horizon. This is deliberately narrow: it tests whether explicit horizon/time heads can beat the BP002 time-shift failure before any renderer bridge.
- Export integration: rollout export now inspects checkpoint metadata and uses either recursive rollout or horizon-conditioned heads, writing only aggregate manifest metadata about the mode.

## Files changed

- `clinical_jepa/arms/v0b/mean_token_model.py` — new shared mean-token model/checkpoint loader with recursive and horizon-conditioned modes.
- `clinical_jepa/arms/v0b/train_minimal_jepa.py` — added CLI flags and checkpoint metadata for horizon-conditioned training: `--autoregression-mode`, `--horizon-count`, `--horizon-stride-tokens`, `--max-horizons`.
- `clinical_jepa/eval/export_mean_token_rollouts.py` — loads model via checkpoint metadata and exports horizon-conditioned rollouts when available.
- `tests/test_horizon_conditioned_rollouts.py` — synthetic HDF5/checkpoint regression proving horizon-conditioned heads separate aligned horizons from time-shift control by cosine.
- `README.md` — notes BP003 horizon-conditioned repair boundary.

## Validation run

All validation used synthetic/public fixtures only:

- `.venv/bin/python -m unittest discover -s tests -p 'test_horizon_conditioned_rollouts.py'` — passed.
- `.venv/bin/python -m unittest discover -s tests -p 'test_export_mean_token_rollouts.py'` — passed.
- `.venv/bin/python -m unittest discover -s tests -p 'test_autoregression_readiness.py'` — passed.
- `.venv/bin/python -m compileall -q clinical_jepa` — passed.
- `PATH="$PWD/.venv/bin:$PATH" bash scripts/run_schema_validation_checks.sh /tmp/clinical_jepa_schema_check_bp003` — passed; discovered 29 tests and validated synthetic schema artifacts.
- `git diff --check` — passed before commit.

## Commit

Code commit: `d9338de Add horizon-conditioned mean-token rollouts`.

## Safety notes

- Did not read raw clinical HDF5, governed sidecar arrays, embeddings, checkpoints, source-ID maps, block IDs, patient hashes, token rows, patient-level data, or private data.
- Did not restart Vast or use remote compute.
- The only HDF5/checkpoint artifacts created were synthetic temporary fixtures inside unit tests.
- No renderer, generated-sequence, clinical-utility, treatment-effect, or INSPECT-optimisation claim is made.

## BP003 decision

Status: **promote to governed capped run, with review gate**.

The public/synthetic repair scaffold is ready. The next step should be a reviewed local governed capped run that trains/exports with `--autoregression-mode horizon_conditioned --horizon-count 2` using the same BP002 capped MIMIC same-source evaluation and `--control-mode all`. Promotion to renderer bridge should still require the governed capped run to beat matched-random plus query-shift, target-shift, and especially time-shift controls.
