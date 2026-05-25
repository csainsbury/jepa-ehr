# jepa-ehr

Clinical-JEPA scaffolding for leakage-aware latent prediction experiments over tokenised EHR sequences.

**Prior-art boundary:** Yang et al. 2026 `Clin-JEPA` is direct prior art for generic JEPA-style latent rollout over EHR patient trajectories. This repo should not be read as claiming first-EHR-JEPA novelty; the local focus is governed tokenised-EHR readout, FlatASCEND/ORCA reader-speaker bridging, external INSPECT transfer, leakage controls, and TTE-style generation/readout readiness.

**Current v0 claim, stated cautiously:** Clinical-JEPA v0 shows robust latent future-block retrieval on governed, re-keyed tokenised EHR sequences, including external INSPECT transfer. The current main evidence line is the scaled mean-token v0B scaffold; first transformer+EMA variants are strong on MIMIC and show clear INSPECT signal, but do not yet beat the simpler scaffold under the demanding MIMIC→INSPECT zero-shot transfer gate.

This repository contains code, schemas, example configs, and aggregate/sanitized pilot notes. It does **not** contain clinical data, patient-level records, source identifiers, source-ID maps, token-level examples, embeddings, checkpoints, credentials, or download tokens.

## What is here

- `clinical_jepa/` — split manifests, target-block extraction, leakage audits, v0A/v0B/v0D scaffold code, aggregate TTE scenario feasibility, and pseudo-rendering readiness helpers.
- `schemas/` — JSON schemas for manifests and aggregate result artifacts, including scenario-feasibility and pseudo-rendering-readiness outputs.
- `configs/v0/*.example.yaml` — placeholder configs only.
- `scripts/` — preflight, synthetic checks, safe data-bundle preflight, and aggregate probe helpers.
- `docs/` — design notes and sanitized aggregate pilot results, including:
  - `docs/clinical-jepa-v0-synthesis.html` — clean synthesis report with cautious claim, transfer gate, figure/table plan, and decisions.
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

## Interpretation boundary

The v0 results support a representation-learning claim, not a clinical utility, novelty-over-Clin-JEPA, or causal treatment-effect claim. Candidate-normalized INSPECT transfer is the current promotion gate for architecture decisions; MIMIC-only improvements are insufficient.
