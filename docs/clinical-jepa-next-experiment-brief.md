# Clinical-JEPA next experiment brief — transfer-guarded transformer+EMA

Date: 2026-05-24

Status: **do not run until after write-up review**.

## Decision context

The B1a v0 synthesis supports scaled mean-token v0B as the current main evidence line. First transformer+EMA variants were very strong on MIMIC and retained clear INSPECT signal, but did not yet beat scaled mean-token models under the demanding MIMIC→INSPECT zero-shot transfer gate. Therefore the next GPU experiment, if any, should be a single targeted transfer-improvement test rather than a broad architecture sweep.

## Question

Can a regularised transformer+EMA model close the INSPECT transfer gap while retaining strong MIMIC retrieval?

## Default training boundary

- Train on MIMIC only.
- Keep INSPECT locked for one-shot external evaluation.
- If INSPECT is used in training, that must be an explicit separate decision because it changes the external-validation story.

## Single experiment sketch

Model: `transformer-ema-transfer-guarded-v0`

Configuration envelope:

- 2-layer transformer+EMA context/target encoders.
- 256–384d representation width.
- stronger dropout and weight decay than the first transformer+EMA variants.
- higher EMA decay.
- predictor bottleneck to reduce source-specific memorisation.
- shorter training / early-stop checkpoint selection using MIMIC dev gap64 retrieval and non-collapse diagnostics.

## Required evaluations

Use the same aggregate-only evaluation stack as v0:

1. MIMIC gap0/gap16/gap64 retrieval.
2. INSPECT gap0/gap16/gap64 retrieval.
3. Candidate-normalized retrieval with min 128 / max 512 candidates per group.
4. Target-row shuffle, query-row shuffle, and within-group time-shift controls.
5. Collapse diagnostics: dev cosine, variance/effective rank.
6. Leakage audit unchanged/pass.

## Primary success gate

Candidate-normalized INSPECT gap64 must beat the scaled mean-token 256d/25k reference:

- reference INSPECT gap64 R@10: `0.4845`
- reference INSPECT gap64 MRR: `0.2982`

A MIMIC-only gain is insufficient.

## Secondary gates

- INSPECT gap16 should also improve over or match the mean-token reference.
- Controls must remain far below observed retrieval.
- Horizon degradation should remain sensible.
- No collapse or effective-rank warning.
- No leakage-audit regression.

## Stop criteria

Stop after one run if:

- INSPECT gap64 remains below the mean-token reference by more than a small tolerance;
- controls rise materially toward observed retrieval;
- representation collapse diagnostics worsen;
- implementation requires new data access, raw/MEDS-lite inputs, or T2 outcome labels.

Do not launch a sweep without a new written rationale.

## Governance boundary

Do not include raw/patient-level data, HDF5s, source-ID maps, embeddings, checkpoints, secrets, `.env` files, passfiles, or time-limited URLs in public artifacts. v0C raw/MEDS-lite and outcome-proximal T2 labels remain gated until explicitly approved.
