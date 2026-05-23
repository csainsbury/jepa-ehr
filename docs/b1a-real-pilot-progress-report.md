# Clinical-JEPA B1a real pilot progress report

Date: 2026-05-23

## Status

Completed first real-data pilot atoms on Vast using the approved re-keyed FlatASCEND B1a tokenised bundle.

Remote pilot root:

```text
/workspace/clinical-jepa-autonomous-run/run-workspace/state/task-work/clinical-jepa-pilot/v0/real-b1a-pilot-70k
```

Local sanitized snapshot:

```text
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-real-pilot/
```

## Data / safety

- Dataset: re-keyed FlatASCEND B1a MIMIC + INSPECT tokenised bundle.
- Bundle on Vast: `/workspace/data/clinical-jepa-v0/clinical-jepa-v0-b1a-mimic-inspect-rekeyed-20260523`
- Original source IDs exported: no.
- Source-ID mapping exported: no.
- Raw dates/free text/direct identifiers: no.
- Leakage audit for pilot target blocks: pass.

## Split / target extraction

Metadata-only split manifest generated from inherited source splits:

- train sequences: `341,303`
- dev sequences: `41,971`
- test sequences: `43,306`

Pilot target block extraction caps: 50k train / 10k dev / 10k test sequences.

Extracted blocks:

- train: T0 `41,562`, T1 `47,309`
- dev: T0 `8,332`, T1 `9,447`
- test: T0 `8,302`, T1 `9,457`
- total: `124,409`

## v0D aggregate baselines

Context-mode/prior aggregate baselines on T0 token-family labels:

- next medication family: dev top1 `0.132`, test top1 `0.141`
- next lab family: dev top1 `0.159`, test top1 `0.159`
- next state family: dev top1 `0.790`, test top1 `0.793`

## v0A frozen FlatASCEND scaffold

FlatASCEND B1a 85M checkpoint staged/audited on Vast:

- checkpoint step: `100000`
- parameter count: `86,022,538`

Extracted prefix-only FlatASCEND embeddings:

- 12k context smoke: passed
- 60k context scale probe: passed
- embedding shape: `60,000 × 768`

v0A 60k ridge probe top1 results:

- next medication family:
  - final-mean dev `0.235`, test `0.229`
  - final-token dev `0.216`, test `0.231`
- next lab family:
  - final-mean dev `0.524`, test `0.517`
  - final-token dev `0.849`, test `0.845`
- next state family:
  - final-mean dev `0.899`, test `0.901`
  - final-token dev `0.982`, test `0.983`

Interpretation note: final-token FlatASCEND embeddings are a strong autoregressive baseline for immediate next-token-family prediction, so high lab/state scores should be treated as a baseline strength and care-process/autocorrelation diagnostic, not as JEPA evidence by themselves.

## v0B minimal JEPA

Real minimal JEPA runs completed on re-keyed B1a target blocks.

Dev cosine progression:

- 400 steps: `0.4417` cosine, loss `0.5583`, effective rank `60.58`
- 2,500 steps: `0.4814` cosine, loss `0.5186`, effective rank `65.77`
- 10,000 steps: `0.5125` cosine, loss `0.4875`, effective rank `69.88`

This suggests the minimal v0B latent prediction scaffold is learning non-collapsed target representations on real tokenised data.


## Context-family controls

Added explicit context-token-family controls over all T0 pilot blocks (`58,196` scanned). These help interpret the strong v0A final-token probe.

Mode-context baseline roughly matches v0D context-mode/prior:

- next medication family: dev `0.132`, test `0.141`
- next lab family: dev `0.159`, test `0.159`
- next state family: dev `0.790`, test `0.793`

Final raw context-token family alone does **not** explain the strong v0A final-token lab/state probes:

- next lab family final-context token baseline: dev `0.156`, test `0.155` vs v0A final-token ridge dev `0.849`, test `0.845`
- next state family final-context token baseline: dev `0.857`, test `0.856` vs v0A final-token ridge dev `0.982`, test `0.983`

This supports treating FlatASCEND final-token hidden states as a strong autoregressive representation baseline, not merely a raw last-token copy.


## v0B downstream probes

Added probes over the 10k-step v0B learned context-prediction embeddings (`58,196` T0 blocks). Ridge probe top1:

- next medication family: dev `0.211`, test `0.218`
- next lab family: dev `0.228`, test `0.233`
- next state family: dev `0.884`, test `0.884`

Interpretation: v0B learned embeddings beat simple context-mode/prior baselines for medication and lab family prediction, but remain below frozen FlatASCEND final-token/mean probes for lab/state. State-family performance is near the empirical-prior ceiling because state labels are highly imbalanced.

## Retrieval and horizon-gap sensitivity

Implemented aggregate target-retrieval metrics for v0B predicted context embeddings against v0B target mean embeddings. On immediate T0 blocks, retrieval against same-split/same-target-type sampled distractors (`4096` candidates per group) was:

- Recall@1 `0.2727`
- Recall@5 `0.4663`
- Recall@10 `0.5526`
- MRR `0.3670`
- median rank `7`

Horizon-gap sensitivity using the same 10k-step v0B checkpoint:

| Gap between context and T0 target | Blocks | Recall@10 | MRR | med test top1 | lab test top1 | state test top1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 events | `58,196` | `0.5526` | `0.3670` | `0.218` | `0.233` | `0.884` |
| 16 events | `51,478` | `0.4727` | `0.3057` | `0.210` | `0.201` | `0.872` |
| 64 events | `34,150` | `0.3877` | `0.2339` | `0.204` | `0.191` | `0.864` |

Interpretation: retrieval degrades monotonically with event gap, which is the expected direction for a horizon-sensitive predictor. Medication/lab family probes degrade more modestly; state-family scores remain high because the state label distribution is highly imbalanced.

## Current outputs of interest

```text
v0/real-b1a-pilot-70k/splits/split-manifest.json
v0/real-b1a-pilot-70k/target-blocks/target-block-manifest.json
v0/real-b1a-pilot-70k/leakage-audit.json
v0/real-b1a-pilot-70k/v0D-real/summary.md
v0/real-b1a-pilot-70k/v0A-real-60k/embedding-cache-manifest.json
v0/real-b1a-pilot-70k/v0A-real-60k-probes/summary.md
v0/real-b1a-pilot-70k/v0B-real-10k/summary.md
v0/real-b1a-pilot-70k/v0B-real-10k-probes-retrieval-fast/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap16/v0B-real-10k-probes-retrieval/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap64/v0B-real-10k-probes-retrieval/summary.md
v0/real-b1a-pilot-70k/controls/context-family/summary.md
```

## Recommended next atoms

1. Add matched-distractor retrieval policies using utilisation/sequence-length strata, not only same split/target type.
2. Add explicit v0A retrieval, or a v0A predictor head, so v0A and v0B can be compared on retrieval rather than only probes.
3. Add INSPECT-as-external-validation rather than only using the staged MIMIC primary split.
4. Only then consider a larger v0B encoder/EMA target architecture beyond the current mean-token scaffold.
