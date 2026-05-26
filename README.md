# jepa-ehr

Clinical-JEPA scaffolding for leakage-aware latent prediction experiments over tokenised EHR sequences.

**Prior-art boundary:** Yang et al. 2026 `Clin-JEPA` is direct prior art for generic JEPA-style latent rollout over EHR patient trajectories. This repo should not be read as claiming first-EHR-JEPA novelty; the local focus is governed tokenised-EHR readout, FlatASCEND/ORCA reader-speaker bridging, accurate autoregression/readout, leakage controls, and TTE-style generation readiness. INSPECT remains an external stress test, not the active optimization target.

**Current v0 claim, stated cautiously:** Clinical-JEPA v0 shows robust latent future-block retrieval on governed, re-keyed tokenised EHR sequences. The current development focus is no longer optimizing external transfer; it is testing whether latent predictions can support accurate autoregressive futures through same-source matched rollout, pseudo-rendering, and eventually explicit renderer/speaker gates. BP-CLINJEPA-005 adds safe-public horizon-specification diagnostics so target windows/strides can be tested for separability before any heavier autoregressive model or renderer bridge is promoted. BP-CLINJEPA-006 adds a safe-public stronger autoregressive latent-predictor scaffold for that revised gate; it is still latent readout only, not event generation.

This repository contains code, schemas, example configs, and aggregate/sanitized pilot notes. It does **not** contain clinical data, patient-level records, source identifiers, source-ID maps, token-level examples, embeddings, checkpoints, credentials, or download tokens.

## What is here

- `clinical_jepa/` — split manifests, target-block extraction, leakage audits, v0A/v0B/v0D/v0E scaffold code, aggregate TTE scenario feasibility, pseudo-rendering readiness, horizon-spec diagnostics, and latent autoregression readiness helpers.
- `schemas/` — JSON schemas for manifests and aggregate result artifacts, including scenario-feasibility and pseudo-rendering-readiness outputs.
- `configs/v0/*.example.yaml` — placeholder configs only, including a BP005 horizon-spec candidate grid.
- `scripts/` — preflight, synthetic checks, safe data-bundle preflight, and aggregate probe helpers.
- `docs/` — design notes and sanitized aggregate pilot results, including:
  - `docs/clinical-jepa-v0-synthesis.html` — clean synthesis report with cautious claim, figure/table plan, and decisions.
  - `docs/clinical-jepa-v0-research-narrative.md` — longer aggregate research narrative.
  - `docs/b1a-real-pilot-progress-report.md` — detailed aggregate progress report.
  - `docs/clinical-jepa-next-experiment-brief.md` — single targeted follow-up experiment brief, not approval to run it.

## Safety boundary

Do not commit or upload:

- raw EHR data, tokenised patient-level bundles, HDF5 data files, source-id maps, or per-patient examples;
- model checkpoints or embedding arrays;
- `.env` files, passphrases, credentials, private keys, or time-limited download URLs.

The default `.gitignore` blocks common data/checkpoint/secret artifacts.

## Quick synthetic check

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash scripts/run_synthetic_scaffold_checks.sh
bash scripts/run_schema_validation_checks.sh
python -m unittest tests/test_synthetic_pipeline.py tests/test_validation.py tests/test_scenario_feasibility.py tests/test_pseudo_rendering.py
```

Real-data runs require an approved de-identified/tokenised local bundle and must keep outputs aggregate-only unless separately governed.

## Horizon-spec scaffold

BP-CLINJEPA-005 uses `clinical_jepa.eval.horizon_specs` to compare candidate target-window / stride / horizon-count specifications before heavier autoregressive modelling. The CLI emits aggregate target-horizon separability summaries and placeholder command plans for reviewed local runs only:

```bash
python -m clinical_jepa.eval.horizon_specs \
  --spec-config configs/v0/horizon_specs.example.yaml \
  --output-dir /tmp/clinical_jepa_horizon_specs
```

Do not replace placeholders with governed paths in committed configs, docs, or reports.

## Transformer autoregressive scaffold

BP-CLINJEPA-006 adds `clinical_jepa.arms.v0e` and `clinical_jepa.eval.export_transformer_autoreg_rollouts` for a small direct multi-horizon transformer latent predictor. It is designed for synthetic/public tests first and reviewed local governed capped runs later, using the BP005 `event32_stride128` primary gate and `event64_stride128` sensitivity gate.

Example dry run only:

```bash
python -m clinical_jepa.arms.v0e.train_transformer_autoreg \
  --dataset-config configs/v0/dataset.example.yaml \
  --target-blocks /tmp/synthetic-target-blocks.json \
  --leakage-report /tmp/synthetic-leakage-pass.json \
  --output-dir /tmp/clinical_jepa_transformer_autoreg \
  --dry-run
```

Do not commit checkpoints, rollout arrays, JSONL sidecars, or governed paths.

## Interpretation boundary

The v0 results support a representation-learning/readout-engineering claim, not a clinical utility, novelty-over-Clin-JEPA, or causal treatment-effect claim. The current development gate is accurate autoregression/readout on governed same-source target windows with leakage, time-shift, utilisation/contact-density, and negative controls. INSPECT transfer is useful later as an external stress test, but should not displace the autoregression objective.
