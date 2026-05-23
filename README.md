# jepa-ehr

Clinical-JEPA scaffolding for leakage-aware latent prediction experiments over tokenised EHR sequences.

This repository contains code, schemas, example configs, and aggregate/sanitized pilot notes. It does **not** contain clinical data, patient-level records, source identifiers, embeddings, checkpoints, credentials, or download tokens.

## What is here

- `clinical_jepa/` — split manifests, target-block extraction, leakage audits, v0A/v0B/v0D scaffold code.
- `schemas/` — JSON schemas for manifests and aggregate result artifacts.
- `configs/v0/*.example.yaml` — placeholder configs only.
- `scripts/` — preflight, synthetic checks, safe data-bundle preflight, and aggregate probe helpers.
- `docs/` — design notes and sanitized aggregate pilot results.

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
python -m unittest tests/test_synthetic_pipeline.py tests/test_validation.py
```

Real-data runs require an approved de-identified/tokenised local bundle and must keep outputs aggregate-only unless separately governed.
