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

## v0B scaled mean-token follow-ups

Queued and ran scaled mean-token v0B follow-ups with larger context/target caps and utilisation/sequence/context-count matched retrieval.

### 256d / 25k

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

Scaled-model query/time-shift controls are no longer near zero under the very fine utilisation/context-count policy, but remain far below the observed scores:

| Evaluation source/gap | observed R@10 | target-shuffle R@10 | query-shuffle R@10 | time-shift R@10 |
|---|---:|---:|---:|---:|
| MIMIC gap 0 | `0.8597` | `0.0767` | `0.0775` | `0.0754` |
| MIMIC gap 16 | `0.7666` | `0.0814` | `0.0815` | `0.0794` |
| MIMIC gap 64 | `0.7251` | `0.1074` | `0.1075` | `0.1035` |
| INSPECT gap 0 | `0.7181` | `0.1462` | `0.1461` | `0.1438` |
| INSPECT gap 16 | `0.6688` | `0.1452` | `0.1461` | `0.1421` |
| INSPECT gap 64 | `0.5779` | `0.1400` | `0.1418` | `0.1379` |

### 512d / 20k

A 512-dimensional, 20k-step sweep improved MIMIC retrieval further but was lower than the 256d/25k run on INSPECT transfer.

Training diagnostics:

- dev cosine: `0.3889`
- dev effective rank: `196.23`

Retrieval results:

| Evaluation source/gap | Recall@10 | MRR | median rank | evaluated blocks |
|---|---:|---:|---:|---:|
| MIMIC gap 0 | `0.9319` | `0.7996` | `1` | `58,058` |
| MIMIC gap 16 | `0.8199` | `0.6188` | `1` | `51,344` |
| MIMIC gap 64 | `0.7265` | `0.5097` | `2` | `34,025` |
| INSPECT gap 0 | `0.7004` | `0.4842` | `3` | `18,004` |
| INSPECT gap 16 | `0.6430` | `0.4238` | `4` | `17,598` |
| INSPECT gap 64 | `0.5540` | `0.3458` | `7` | `16,496` |

Candidate-normalized retrieval with only groups containing at least 128 candidates and at most 512 sampled candidates preserved the ordering: 512d best on MIMIC, 256d best on INSPECT transfer.

| Model / source-gap | Recall@10 | MRR | evaluated blocks | skipped small-group queries |
|---|---:|---:|---:|---:|
| 10k / MIMIC gap 0 | `0.7774` | `0.5700` | `47,253` | `10,943` |
| 10k / MIMIC gap 16 | `0.7284` | `0.5133` | `41,246` | `10,232` |
| 10k / MIMIC gap 64 | `0.6711` | `0.4473` | `24,950` | `9,200` |
| 256d/25k / MIMIC gap 0 | `0.8723` | `0.6891` | `47,253` | `10,943` |
| 256d/25k / MIMIC gap 16 | `0.7791` | `0.5692` | `41,246` | `10,232` |
| 256d/25k / MIMIC gap 64 | `0.7064` | `0.4837` | `24,950` | `9,200` |
| 256d/25k / INSPECT gap 0 | `0.6427` | `0.4340` | `11,421` | `6,762` |
| 256d/25k / INSPECT gap 16 | `0.5891` | `0.3820` | `11,285` | `6,497` |
| 256d/25k / INSPECT gap 64 | `0.4845` | `0.2982` | `10,691` | `5,978` |
| 512d/20k / MIMIC gap 0 | `0.9412` | `0.8171` | `47,253` | `10,943` |
| 512d/20k / MIMIC gap 16 | `0.8349` | `0.6345` | `41,246` | `10,232` |
| 512d/20k / MIMIC gap 64 | `0.7102` | `0.4881` | `24,950` | `9,200` |
| 512d/20k / INSPECT gap 0 | `0.6232` | `0.4175` | `11,421` | `6,762` |
| 512d/20k / INSPECT gap 16 | `0.5605` | `0.3621` | `11,285` | `6,497` |
| 512d/20k / INSPECT gap 64 | `0.4596` | `0.2775` | `10,691` | `5,978` |

Interpretation caution: these are still mean-token scaffolds, not true EMA/transformer JEPA architectures. Scaling improves retrieval substantially, but candidate-set/group-size effects and source-transfer differences must be reported explicitly. Later 256d/50k and 384d/40k controls confirmed the same pattern: 256d/25k, 256d/50k, and 384d/40k are tightly clustered on INSPECT transfer, with 256d/25k marginally best on candidate-normalized INSPECT gap 64.

Sanitized aggregate snapshots:

```text
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/matched-retrieval/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/shuffle-controls/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/util-matched/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/query-time-controls/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/inspect-external/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/v0a-inspect-transfer/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/v0b-scaled-256d-25k/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/scaled-query-time-controls/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/v0b-scaled-512d-20k/
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/b1a-robustness-controls/candidate-normalized/
```

## Transformer+EMA architecture diagnostics

Completed the first transformer+EMA JEPA evaluations for `256d/20k`, `256d/40k`, `384d/15k`, and `128d/25k`, using utilisation/context-count matched retrieval, candidate-normalized retrieval, and target/query/time-shift controls.

Candidate-normalized results:

| Model / source-gap | Recall@10 | MRR | evaluated blocks | target-shuffle R@10 |
|---|---:|---:|---:|---:|
| 256d/20k / MIMIC gap 0 | `0.9569` | `0.8919` | `47,737` | `0.0223` |
| 256d/20k / MIMIC gap 16 | `0.7870` | `0.5702` | `41,730` | `0.0227` |
| 256d/20k / MIMIC gap 64 | `0.6823` | `0.4404` | `25,434` | `0.0254` |
| 256d/20k / INSPECT gap 0 | `0.5385` | `0.3250` | `12,288` | `0.0265` |
| 256d/20k / INSPECT gap 16 | `0.4932` | `0.2876` | `12,152` | `0.0255` |
| 256d/20k / INSPECT gap 64 | `0.4077` | `0.2260` | `11,560` | `0.0226` |
| 256d/40k / MIMIC gap 0 | `0.9539` | `0.8963` | `47,737` | `0.0228` |
| 256d/40k / MIMIC gap 16 | `0.7919` | `0.5765` | `41,730` | `0.0230` |
| 256d/40k / MIMIC gap 64 | `0.6664` | `0.4235` | `25,434` | `0.0257` |
| 256d/40k / INSPECT gap 0 | `0.5151` | `0.3034` | `12,288` | `0.0243` |
| 256d/40k / INSPECT gap 16 | `0.4587` | `0.2584` | `12,152` | `0.0249` |
| 256d/40k / INSPECT gap 64 | `0.3851` | `0.2120` | `11,560` | `0.0240` |
| 384d/15k / MIMIC gap 0 | `0.9608` | `0.9064` | `47,737` | `0.0219` |
| 384d/15k / MIMIC gap 16 | `0.8244` | `0.6213` | `41,730` | `0.0222` |
| 384d/15k / MIMIC gap 64 | `0.7039` | `0.4636` | `25,434` | `0.0248` |
| 384d/15k / INSPECT gap 0 | `0.5612` | `0.3390` | `12,288` | `0.0257` |
| 384d/15k / INSPECT gap 16 | `0.5197` | `0.3077` | `12,152` | `0.0255` |
| 384d/15k / INSPECT gap 64 | `0.4278` | `0.2397` | `11,560` | `0.0249` |
| 128d/25k / MIMIC gap 0 | `0.9003` | `0.7190` | `47,737` | `0.0224` |
| 128d/25k / MIMIC gap 16 | `0.7043` | `0.4669` | `41,730` | `0.0235` |
| 128d/25k / MIMIC gap 64 | `0.6466` | `0.3916` | `25,434` | `0.0246` |
| 128d/25k / INSPECT gap 0 | `0.4903` | `0.2848` | `12,288` | `0.0252` |
| 128d/25k / INSPECT gap 16 | `0.4646` | `0.2615` | `12,152` | `0.0246` |
| 128d/25k / INSPECT gap 64 | `0.3806` | `0.2047` | `11,560` | `0.0255` |

Interpretation: transformer+EMA is very strong on MIMIC and controls remain near chance. It transfers clearly above controls, but does not yet beat the best scaled mean-token scaffold under the demanding MIMIC→INSPECT zero-shot transfer gate. The 384d/15k transformer is the best transformer+EMA transfer variant; 256d/20k improves transfer over 256d/40k but remains below 384d/15k; 128d/25k is lower than the other transformer variants. The current main evidence line therefore remains scaled mean-token v0B, while transformer+EMA remains a promising transfer-aware architecture target rather than a failed direction.

Final aggregate comparison snapshot:

```text
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/transformer-ema-eval/transformer-jepa-comparison-20260524T174601Z.md
```

Sanitized aggregate snapshot:

```text
state/workflows/2026-05-23-clinical-jepa-blueprint/vast-snapshots/transformer-ema-eval/
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

1. Stop/park Vast compute now that the queued aggregate diagnostics are complete and copied locally.
2. Promote aggregate-only scripts/docs/results into the public repo as appropriate, with no data, embeddings, checkpoints, source-ID maps, secrets, or patient-level outputs.
3. Write the v0 result narrative around scaled mean-token v0B as the current main evidence line and transformer+EMA as a strong, promising architecture diagnostic that has not yet beaten mean-token under the hard MIMIC→INSPECT zero-shot gate.
4. Defer any new GPU experiment until after synthesis; if needed, make it a targeted regularised/source-balanced transformer+EMA run rather than a broad sweep.
