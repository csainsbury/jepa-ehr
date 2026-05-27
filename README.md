# jepa-ehr

Clinical-JEPA scaffolding for leakage-aware latent prediction experiments over tokenised EHR sequences.

**Prior-art boundary:** Yang et al. 2026 `Clin-JEPA` is direct prior art for generic JEPA-style latent rollout over EHR patient trajectories. This repo should not be read as claiming first-EHR-JEPA novelty; the local focus is governed tokenised-EHR readout, FlatASCEND/ORCA reader-speaker bridging, accurate autoregression/readout, leakage controls, and TTE-style generation readiness. INSPECT remains an external stress test, not the active optimization target.

**Current v0 claim, stated cautiously:** Clinical-JEPA v0 shows robust latent future-block retrieval on governed, re-keyed tokenised EHR sequences. The current development focus is no longer optimizing external transfer; it is testing whether latent predictions can support accurate autoregressive futures through same-source matched rollout, pseudo-rendering, and eventually explicit renderer/speaker gates. BP-CLINJEPA-005 adds safe-public horizon-specification diagnostics so target windows/strides can be tested for separability before any heavier autoregressive model or renderer bridge is promoted. BP-CLINJEPA-006 adds a safe-public stronger autoregressive latent-predictor scaffold for that revised gate; it is still latent readout only, not event generation. BP-CLINJEPA-007 explores a differentiated coded-event future-summary/readout scaffold for a FlatASCEND-compatible speaker bridge, again without generated sequences or clinical/treatment claims. BP-CLINJEPA-008 refines that route with scenario-specific coded-summary ontology diagnostics so base-rate-dominated targets can be rejected before speaker/readout promotion. BP-CLINJEPA-009 tightens the target-ontology scaffold further with medication-change/drop/restart-style modes and utilisation-stratified/residualized diagnostics before any speaker bridge is considered. BP-CLINJEPA-010 separates future-only target definitions from fixed context/utilisation strata so definition-adjacent change/drop labels can be rejected before speaker/readout promotion.

This repository contains code, schemas, example configs, and aggregate/sanitized pilot notes. It does **not** contain clinical data, patient-level records, source identifiers, source-ID maps, token-level examples, embeddings, checkpoints, credentials, or download tokens.

## What is here

- `clinical_jepa/` — split manifests, target-block extraction, leakage audits, v0A/v0B/v0D/v0E scaffold code, aggregate TTE scenario feasibility, pseudo-rendering readiness, horizon-spec diagnostics, coded-event future-summary/scenario/conditional-outcome readout, and latent autoregression readiness helpers.
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

BP-CLINJEPA-006 adds `clinical_jepa.arms.v0e` and `clinical_jepa.eval.export_transformer_autoreg_rollouts` for a small direct multi-horizon transformer latent predictor. It is designed for synthetic/public tests first and reviewed local governed capped runs later, using the BP005 `event32_stride128` primary gate and `event64_stride128` sensitivity gate. The default repaired v0E path predicts into a fixed mean-token target space and emits prediction/target collapse diagnostics so high raw cosine cannot pass silently when target geometry is non-discriminative.

Example dry run only:

```bash
python -m clinical_jepa.arms.v0e.train_transformer_autoreg \
  --dataset-config configs/v0/dataset.example.yaml \
  --target-blocks /tmp/synthetic-target-blocks.json \
  --leakage-report /tmp/synthetic-leakage-pass.json \
  --output-dir /tmp/clinical_jepa_transformer_autoreg \
  --target-encoder-mode fixed_mean_token \
  --dry-run
```

Do not commit checkpoints, rollout arrays, JSONL sidecars, or governed paths.

## Coded-event future-summary scaffold

BP-CLINJEPA-007 adds `clinical_jepa.speaker.future_summary` as a safe-public scaffold for FlatASCEND-compatible coded-event future-summary readout. It scores aggregate family distributions and multi-label family presence from synthetic or reviewed local pre-extracted coded-event summaries. It does **not** render or generate event sequences.

Command-plan scaffold only:

```bash
python -m clinical_jepa.speaker.future_summary \
  --spec-config configs/v0/future_summary.example.yaml \
  --output-dir /tmp/clinical_jepa_future_summary \
  --emit-command-plan
```

Real-data use requires reviewed local pre-extraction into governed sidecars and aggregate-only sync-back; do not commit local row summaries, raw tokens, sidecars, checkpoints, or governed paths.

## Scenario-specific coded-summary ontology scaffold

BP-CLINJEPA-008 adds `clinical_jepa.speaker.scenario_ontology` for synthetic/public testing of narrower coded-summary target ontologies after broad MED/LAB/STATE summaries proved base-rate dominated. BP-CLINJEPA-009 extends the same scaffold with refined medication-change/drop/restart-style modes, optional prior-context keys, utilisation-stratified and residualized controls, and definition-adjacent diagnostics. It supports configurable target-family predicates, presence/start/continuation/change/drop/restart summary modes, empirical-prior/context/utilisation-control baselines, base-rate-domination diagnostics, and negative-control event hooks. It does **not** render or generate event sequences.

Command-plan scaffold only:

```bash
python -m clinical_jepa.speaker.scenario_ontology \
  --spec-config configs/v0/scenario_ontology.example.yaml \
  --output-dir /tmp/clinical_jepa_scenario_ontology \
  --emit-command-plan
```

Real-data use requires reviewed local pre-extraction into governed sidecars and aggregate-only sync-back; do not commit local row summaries, raw tokens, sidecars, checkpoints, or governed paths. Aggregate reports suppress target-family names and predicates by default. BP009-style local use should treat continuation/absence-style modes as definition-adjacent controls and should not promote speaker/readout work unless candidate targets survive empirical-prior, utilisation-stratified, residualized, and negative-control checks.

## Conditional future-event outcome scaffold

BP-CLINJEPA-010 adds `clinical_jepa.speaker.conditional_outcome` to test future-only target outcomes within fixed context/utilisation strata. Context predicates define strata and optional readout scores; they do not define the target label. This is intended to catch definition-adjacent leakage from change/drop labels before any speaker bridge is considered. It does **not** render or generate event sequences.

Command-plan scaffold only:

```bash
python -m clinical_jepa.speaker.conditional_outcome \
  --spec-config configs/v0/conditional_outcome.example.yaml \
  --output-dir /tmp/clinical_jepa_conditional_outcome \
  --emit-command-plan
```

Real-data use requires reviewed local pre-extraction into governed sidecars and aggregate-only sync-back; do not commit local row summaries, raw tokens, sidecars, checkpoints, or governed paths. Aggregate reports suppress target-family names and predicates by default. BP010-style local use should not promote speaker/readout work unless future-only targets beat stratum-prior, utilisation, time-shift, matched-random, and negative-control baselines.

## Interpretation boundary

The v0 results support a representation-learning/readout-engineering claim, not a clinical utility, novelty-over-Clin-JEPA, or causal treatment-effect claim. The current development gate is accurate autoregression/readout on governed same-source target windows with leakage, time-shift, utilisation/contact-density, and negative controls. INSPECT transfer is useful later as an external stress test, but should not displace the autoregression objective.
