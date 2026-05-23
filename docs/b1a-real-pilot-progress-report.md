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

## v0A target-embedding retrieval comparison

Added v0A target-block FlatASCEND embeddings and trained ridge predictors from v0A context embeddings to v0A target embeddings, using the same retrieval harness and sampled same-split/same-target-type distractor policy. Best v0A retrieval by gap:

| Gap between context and T0 target | Blocks | Best v0A predictor | v0A Recall@10 | v0A MRR | v0B Recall@10 | v0B MRR |
|---:|---:|---|---:|---:|---:|---:|
| 0 events | `60,000` | final-token context → target-mean | `0.3019` | `0.1684` | `0.5526` | `0.3670` |
| 16 events | `51,478` | context-mean → target-mean | `0.2310` | `0.1273` | `0.4727` | `0.3057` |
| 64 events | `34,150` | context-mean → target-mean | `0.1255` | `0.0660` | `0.3877` | `0.2339` |

Interpretation: in this first retrieval framing, the 10k-step v0B minimal JEPA scaffold substantially outperforms the v0A frozen-FlatASCEND target-embedding predictor on retrieval, despite v0A remaining much stronger on immediate lab/state classification probes. This is promising for v0B as a non-teacher-circular latent-prediction line, but the comparison still needs matched utilisation/sequence-length distractors before promotion.

## Matched retrieval and shuffle controls

Added matched-distractor retrieval using same split, target type, context-length bin, and target-length bin.

| Gap between context and T0 target | v0B Recall@10 | v0B MRR | best v0A predictor | v0A Recall@10 | v0A MRR |
|---:|---:|---:|---|---:|---:|
| 0 events | `0.5557` | `0.3695` | final-token context → target-mean | `0.3290` | `0.1867` |
| 16 events | `0.4850` | `0.3156` | context-mean → target-mean | `0.2256` | `0.1239` |
| 64 events | `0.4180` | `0.2549` | context-mean → target-mean | `0.1352` | `0.0709` |

Target-embedding shuffle controls stayed near chance:

| Gap | v0B observed R@10 | v0B shuffle R@10 | v0A observed R@10 | v0A shuffle R@10 |
|---:|---:|---:|---:|---:|
| 0 events | `0.5557` | `0.0031` | `0.3290` | `0.0091` |
| 16 events | `0.4850` | `0.0035` | `0.2256` | `0.0036` |
| 64 events | `0.4180` | `0.0045` | `0.1352` | `0.0054` |

A stricter utilisation/sequence-length/context-count matched policy increased the number of candidate groups and still favored v0B:

| Gap | v0B Recall@10 | v0B MRR | best v0A predictor | v0A Recall@10 | v0A MRR | candidate groups, v0B/v0A |
|---:|---:|---:|---|---:|---:|---:|
| 0 events | `0.7622` | `0.5592` | final-token context → target-mean | `0.5787` | `0.3732` | `721 / 2005` |
| 16 events | `0.7167` | `0.5084` | context-mean → target-mean | `0.4313` | `0.2522` | `676 / 674` |
| 64 events | `0.6927` | `0.4744` | context-mean → target-mean | `0.3722` | `0.2036` | `614 / 612` |

Query-shuffle and within-group time-shift controls for the length-matched setting stayed near chance:

| Gap | arm | observed R@10 | target-shuffle R@10 | query-shuffle R@10 | time-shift R@10 |
|---:|---|---:|---:|---:|---:|
| 0 events | v0B | `0.5557` | `0.0031` | `0.0030` | `0.0034` |
| 0 events | v0A | `0.3290` | `0.0091` | `0.0093` | `0.0099` |
| 16 events | v0B | `0.4850` | `0.0035` | `0.0031` | `0.0034` |
| 16 events | v0A | `0.2256` | `0.0036` | `0.0037` | `0.0045` |
| 64 events | v0B | `0.4180` | `0.0045` | `0.0041` | `0.0046` |
| 64 events | v0A | `0.1352` | `0.0054` | `0.0050` | `0.0047` |

Interpretation: the v0B retrieval advantage survives length and utilisation/context-count matched distractors. Target-row shuffle, query-row shuffle, and within-group time-shift controls argue against trivial row-order, target-distribution, or nearby-time indexing artifacts.

## INSPECT external validation

Added INSPECT as a separate external-validation source from the staged B1a bundle. Target extraction and leakage audits passed for INSPECT T0 blocks:

| Gap | INSPECT T0 blocks | v0B Recall@10 | v0B MRR | exploratory v0A Recall@10 | exploratory v0A MRR |
|---:|---:|---:|---:|---:|---:|
| 0 events | `18,183` | `0.3469` | `0.2079` | `0.6580` | `0.4381` |
| 16 events | `17,782` | `0.3088` | `0.1825` | `0.2923` | `0.1735` |
| 64 events | `16,669` | `0.2319` | `0.1338` | `0.2086` | `0.1177` |

Important caveat: the v0A comparison above is **exploratory within-INSPECT ridge**, not a MIMIC-trained v0A transfer predictor. The v0B checkpoint is the existing MIMIC-trained 10k-step checkpoint evaluated on INSPECT blocks.

A fairer MIMIC-trained v0A transfer ridge was then fitted on MIMIC train embeddings and applied to INSPECT embeddings:

| Gap | MIMIC-trained v0A transfer best predictor | v0A transfer Recall@10 | v0A transfer MRR | v0B INSPECT Recall@10 | v0B INSPECT MRR |
|---:|---|---:|---:|---:|---:|
| 0 events | final-token context → target-mean | `0.1385` | `0.0730` | `0.3469` | `0.2079` |
| 16 events | context-mean → target-mean | `0.0373` | `0.0208` | `0.3088` | `0.1825` |
| 64 events | context-mean → target-mean | `0.0313` | `0.0164` | `0.2319` | `0.1338` |

Interpretation: within-INSPECT v0A can fit strong gap-0 source-specific mappings, but MIMIC-trained v0A transfer to INSPECT is much weaker than the MIMIC-trained v0B checkpoint under this retrieval framing.

## v0B scaled 256d / 25k follow-up

Queued and ran a scaled mean-token v0B follow-up with 256-dimensional embeddings, 25k steps, larger context/target caps, and utilisation/sequence/context-count matched retrieval.

Training diagnostics:

- dev cosine: `0.4538`
- dev effective rank: `131.39`

Retrieval results:

| Evaluation source/gap | Recall@10 | MRR | median rank | evaluated blocks |
|---|---:|---:|---:|---:|
| MIMIC gap 0 | `0.8597` | `0.6748` | `1` | `58,058` |
| MIMIC gap 16 | `0.7666` | `0.5621` | `2` | `51,344` |
| MIMIC gap 64 | `0.7251` | `0.5064` | `3` | `34,025` |
| INSPECT gap 0 | `0.7181` | `0.5015` | `3` | `18,004` |
| INSPECT gap 16 | `0.6688` | `0.4479` | `3` | `17,598` |
| INSPECT gap 64 | `0.5779` | `0.3687` | `6` | `16,496` |

Interpretation caution: this is still the mean-token scaffold, not a true EMA/transformer JEPA architecture. The retrieval jump is large and useful, but should be checked against the scaled-model query/time-shift controls now queued before over-interpreting.

Sanitized aggregate snapshots:

```text
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/matched-retrieval/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/shuffle-controls/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/util-matched/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/query-time-controls/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/inspect-external/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/v0a-inspect-transfer/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/v0b-scaled-256d-25k/
```

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
v0/real-b1a-pilot-70k/v0A-real-60k-target-predictor/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap16/v0B-real-10k-probes-retrieval/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap16/v0A-target-predictor/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap64/v0B-real-10k-probes-retrieval/summary.md
v0/real-b1a-pilot-70k/horizon-gap/gap64/v0A-target-predictor/summary.md
v0/real-b1a-pilot-70k/controls/context-family/summary.md
```

## Recommended next atoms

1. Finish the scaled v0B 256d/25k query/time-shift controls now running/queued on Vast.
2. Finish the queued v0B 512d/20k scale sweep and compare against 256d/25k.
3. Run controls against the 512d/20k model if it improves or matches 256d/25k.
4. Then consider a true larger v0B encoder/EMA target architecture beyond the mean-token scaffold.
