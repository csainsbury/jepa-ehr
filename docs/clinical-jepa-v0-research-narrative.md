# Clinical-JEPA v0 research narrative: what we did, why, what happened, and what comes next

Date: 2026-05-24

This note is an aggregate-only account of the Clinical-JEPA v0 pilot so far. It intentionally omits raw/token-level examples, source identifiers, source-ID mappings, patient-level records, free text, dates, credentials, checkpoint files, and governed data paths.

**Prior-art boundary:** Yang et al. 2026 `Clin-JEPA` is direct prior art for generic JEPA-style latent rollout over EHR patient trajectories. This narrative therefore frames local novelty around governed tokenised-EHR readout, FlatASCEND/ORCA reader-speaker bridging, external INSPECT transfer, leakage controls, and TTE-style generation/readout readiness — not around being the first EHR-JEPA.

## Executive summary

The v0 programme started as a question: can a JEPA-style latent prediction objective learn useful clinical state-transition representations from tokenised EHR sequences, in a way that is not merely a FlatASCEND/autoregressive teacher copy?

So far, the answer is: **yes, there is a real signal**, and it survives several increasingly strict checks.

The strongest result so far is not a finished model; it is a staged chain of evidence:

1. A minimal v0B JEPA scaffold learned non-collapsed target representations on real re-keyed FlatASCEND B1a tokenised data.
2. v0B beat direct context-family baselines and v0A frozen-FlatASCEND target-retrieval predictors under multiple retrieval framings.
3. v0B retrieval degraded monotonically with larger event gaps, which is the expected direction for a horizon-sensitive predictor.
4. Length-matched, utilisation/context-count matched, shuffle, query-shuffle, and time-shift controls did not explain the main retrieval signal.
5. External INSPECT validation showed source-transfer signal for v0B; MIMIC-trained v0A transfer was much weaker.
6. Scaling the mean-token v0B scaffold improved retrieval strongly, with 256d variants giving the best INSPECT transfer under candidate-normalized evaluation.
7. First transformer+EMA variants were very strong on MIMIC and retained clear INSPECT signal, but did not yet beat the scaled mean-token scaffold under the demanding MIMIC→INSPECT zero-shot transfer gate.

The current interpretation is: **Clinical-JEPA v0 supports a cautious representation-learning and source-transfer claim**, not a first-EHR-JEPA or clinical-utility claim. Scaled mean-token v0B is the current main evidence line because it is strongest under the MIMIC→INSPECT zero-shot transfer gate; transformer+EMA is a promising architecture diagnostic that already transfers clearly above controls, but needs transfer-aware regularisation or source-robust training before it becomes the mainline.

---

## 1. Literature framing: JEPA and autoregression are not enemies

### Intuition

The initial literature review was to avoid building the wrong conceptual model. A naive reading might frame JEPA as the opposite of autoregression: latent prediction vs token prediction. The newer papers do not support that binary. Sequential JEPAs, action-conditioned JEPAs, energy-based JEPAs, variational JEPAs, and domain-specific JEPAs all blur the line.

For clinical data, that matters because EHRs are temporal event streams. A useful Clinical-JEPA should not pretend that sequence order and autoregressive structure are irrelevant. Instead, it should ask whether predicting **future latent clinical state** adds something beyond predicting the next explicit code/token.

### What we did

- Ingested the JEPA/autoregression paper batch into Cog.
- Updated the JEPA concept page and related synthesis pages.
- Settled the working metaphor:
  - **FlatASCEND** = explicit coded-event speaker / rollout generator.
  - **Clinical-JEPA** = latent patient-state reader / transition model.

### Outcome

This gave the v0 design principle: do not make v0 a FlatASCEND-only derivative, and do not overclaim JEPA as non-autoregressive. The later Clin-JEPA prior-art update adds a second rule: do not claim generic EHR-JEPA novelty. Instead, run a substrate/readout bake-off where the differentiator is governed external transfer, leakage controls, and an explicit route from latent states to clinical futures.

### Rationale for next step

The next step had to be a controlled pilot, not a paper-only synthesis: compare frozen autoregressive representations, direct baselines, and JEPA-style latent prediction on the same governed tokenised EHR substrate.

---

## 2. Data governance and substrate selection

### Intuition

Before modelling, the key risk was governance and leakage. Tokenised EHR is still sensitive. Also, original HDF5 group names and source IDs can leak provenance. A useful pilot needed real enough data to test signal, but not raw/patient-identifying data in the modelling or transfer path.

### What we did

- Chose the canonical dataset version: FlatASCEND B1a governed outcomes v1.
- Built a re-keyed tokenised bundle:
  - no original source IDs exported;
  - no source-ID mapping exported;
  - no raw dates/free text/direct identifiers;
  - no patient examples copied into reports.
- Encrypted the bundle before Backblaze transfer.
- Staged the approved bundle on Vast.

### Outcome

The real tokenised B1a substrate was staged successfully:

- total sequences: `445,580`
- total tokens/events: `105,403,476`
- MIMIC train/dev/test sequences: `341,303 / 41,971 / 43,306`
- INSPECT train/dev/test sequences: `15,187 / 1,907 / 1,906`

### Rationale for next step

With a governed data substrate available, the next requirement was to define prefix-safe target blocks and leakage audits before training anything.

---

## 3. Split manifest, target blocks, and leakage audit

### Intuition

A JEPA-like objective can accidentally cheat if context windows overlap target windows, if target labels leak into context, or if splits are made after windowing. The first real modelling atom therefore needed target blocks and leakage checks, not a model.

### What we did

- Used inherited source splits rather than re-splitting patient/admission units after windowing.
- Extracted prefix-safe T0 and T1 blocks from MIMIC.
- Ran a leakage audit checking window boundaries, split labels, duplicate block IDs, and cached-embedding rules.

### Outcome

Pilot target extraction succeeded:

| Split | T0 blocks | T1 blocks |
|---|---:|---:|
| train | `41,562` | `47,309` |
| dev | `8,332` | `9,447` |
| test | `8,302` | `9,457` |

Total blocks: `124,409`.

Leakage audit: **pass**.

### Rationale for next step

Once target blocks were valid, the next step was not immediately JEPA training. We first needed simple baselines to know what trivial care-process structure could already explain.

---

## 4. v0D direct aggregate baselines

### Intuition

EHR event streams contain strong local autocorrelation and care-process regularities. A model can look impressive if it simply learns that common state/lab/medication families repeat. Direct baselines give a floor and reveal label imbalance.

### What we did

Ran context-family and empirical-prior baselines for next token-family labels on T0 windows.

### Outcome

v0D aggregate baselines:

| Task | test top1 |
|---|---:|
| next medication family | `0.141` |
| next lab family | `0.159` |
| next state family | `0.793` |

The high state-family baseline showed major imbalance/autocorrelation.

### Rationale for next step

A proper representation arm needed to beat these simple baselines, especially for medication/lab families where priors were weak. The next step was to test the frozen FlatASCEND representation arm.

---

## 5. v0A frozen FlatASCEND scaffold

### Intuition

FlatASCEND is a strong autoregressive model trained on the same token substrate. Its hidden states are a natural baseline: if frozen FlatASCEND already gives excellent representations, Clinical-JEPA must show a different advantage rather than merely rediscovering FlatASCEND.

### What we did

- Staged the FlatASCEND B1a 85M checkpoint.
- Extracted prefix-only hidden states for up to 60k T0 blocks.
- Ran ridge probes from frozen hidden states to next token-family labels.

### Outcome

FlatASCEND was very strong for immediate lab/state probes:

| Task / representation | test top1 |
|---|---:|
| next medication family, final mean | `0.229` |
| next medication family, final token | `0.231` |
| next lab family, final mean | `0.517` |
| next lab family, final token | `0.845` |
| next state family, final mean | `0.901` |
| next state family, final token | `0.983` |

### Interpretation

Frozen FlatASCEND final-token hidden states are a strong autoregressive/care-process representation baseline. High lab/state scores should not be read as JEPA evidence; they are the standard that v0B must beat or complement.

### Rationale for next step

A direct label probe was not enough. We needed a JEPA-like latent prediction task: predict the representation of a future block, not just a family label.

---

## 6. v0B minimal JEPA scaffold

### Intuition

The first v0B model was deliberately simple: mean token embeddings for context and target, with a small predictor trained by cosine loss. This was not meant to be the final architecture. It was a scaffold to test whether a latent prediction objective learns anything non-collapsed on real clinical token streams.

### What we did

Trained mean-token v0B runs on real T0 blocks.

### Outcome

Dev cosine improved with training:

| Run | dev cosine | dev loss | effective rank |
|---|---:|---:|---:|
| 400 steps | `0.4417` | `0.5583` | `60.58` |
| 2,500 steps | `0.4814` | `0.5186` | `65.77` |
| 10,000 steps | `0.5125` | `0.4875` | `69.88` |

Downstream probes from the 10k v0B context-prediction embeddings:

| Task | test top1 |
|---|---:|
| next medication family | `0.218` |
| next lab family | `0.233` |
| next state family | `0.884` |

### Interpretation

The scaffold was learning non-collapsed representations and beat simple context/prior baselines for medication/lab. It did not beat frozen FlatASCEND on immediate lab/state probes, but that was expected: FlatASCEND is an autoregressive next-token specialist.

### Rationale for next step

The main question became: can v0B predict **future target-block representations**, and does it degrade with horizon in the expected way?

---

## 7. Retrieval framing and horizon-gap sensitivity

### Intuition

Retrieval is closer to the JEPA objective than label probing. If the context prediction is meaningful, its predicted latent vector should retrieve the true future target block from distractors. Increasing the event gap should make the task harder.

### What we did

- Built an aggregate retrieval harness.
- Evaluated v0B predicted context embeddings against target embeddings.
- Added horizon gaps of 0, 16, and 64 events.
- Added v0A target-embedding retrieval for comparison.

### Outcome: first retrieval framing

| Gap | v0B Recall@10 / MRR | best v0A Recall@10 / MRR |
|---|---:|---:|
| 0 | `0.5526 / 0.3670` | `0.3019 / 0.1684` |
| 16 | `0.4727 / 0.3057` | `0.2310 / 0.1273` |
| 64 | `0.3877 / 0.2339` | `0.1255 / 0.0660` |

### Interpretation

v0B outperformed the v0A frozen-FlatASCEND target-prediction ridge in this retrieval framing, and retrieval degraded monotonically with event gap.

### Rationale for next step

This was promising but too easy to overinterpret. Same-split/target-type distractors might still reward utilisation, sequence length, or care-intensity structure. The next step was stricter distractor matching and controls.

---

## 8. Matched distractors and shuffle/time controls

### Intuition

Clinical retrieval can be gamed by non-clinical signals: sequence length, care intensity, lab/med/state density, and candidate-set size. Matching distractors makes retrieval less likely to reward broad utilisation strata instead of future-state prediction.

### What we did

Added progressively stricter retrieval policies:

1. same split + target type;
2. plus context-length and target-length bins;
3. plus sequence-length and context med/lab/state count bins.

Added controls:

- target-row shuffle;
- query-row shuffle;
- within-group time-shift;
- candidate-normalized retrieval with minimum 128 and maximum 512 candidates per group.

### Outcome: length-matched retrieval

| Gap | v0B Recall@10 / MRR | best v0A Recall@10 / MRR |
|---|---:|---:|
| 0 | `0.5557 / 0.3695` | `0.3290 / 0.1867` |
| 16 | `0.4850 / 0.3156` | `0.2256 / 0.1239` |
| 64 | `0.4180 / 0.2549` | `0.1352 / 0.0709` |

Length-matched query/time controls stayed near chance. For example:

| Gap / arm | observed R@10 | target-shuffle | query-shuffle | time-shift |
|---|---:|---:|---:|---:|
| gap 0 / v0B | `0.5557` | `0.0031` | `0.0030` | `0.0034` |
| gap 64 / v0B | `0.4180` | `0.0045` | `0.0041` | `0.0046` |
| gap 0 / v0A | `0.3290` | `0.0091` | `0.0093` | `0.0099` |

### Outcome: utilisation/context-count matched retrieval

| Gap | v0B Recall@10 / MRR | best v0A Recall@10 / MRR |
|---|---:|---:|
| 0 | `0.7622 / 0.5592` | `0.5787 / 0.3732` |
| 16 | `0.7167 / 0.5084` | `0.4313 / 0.2522` |
| 64 | `0.6927 / 0.4744` | `0.3722 / 0.2036` |

### Interpretation

The stricter utilisation matching did not remove the signal; v0B still beat v0A. However, very fine matching creates smaller candidate groups, so raw Recall@10 can rise. That is why candidate-normalized evaluation became necessary.

### Rationale for next step

The next step was to test source transfer: if the signal is useful, it should not be entirely MIMIC-specific.

---

## 9. INSPECT external validation

### Intuition

A source-specific shortcut can look impressive on one dataset. INSPECT provides an external validation arm using the same re-keyed governed bundle but a different source distribution.

### What we did

- Extracted INSPECT T0 target blocks for gaps 0, 16, and 64.
- Ran leakage audits: pass.
- Evaluated the MIMIC-trained v0B checkpoint on INSPECT.
- Compared with two v0A variants:
  - exploratory within-INSPECT ridge;
  - fairer MIMIC-trained v0A ridge transferred to INSPECT.

### Outcome: MIMIC-trained v0B on INSPECT

| Gap | v0B Recall@10 / MRR |
|---|---:|
| 0 | `0.3469 / 0.2079` |
| 16 | `0.3088 / 0.1825` |
| 64 | `0.2319 / 0.1338` |

### Outcome: fairer MIMIC-trained v0A transfer to INSPECT

| Gap | v0A transfer Recall@10 / MRR | v0B INSPECT Recall@10 / MRR |
|---|---:|---:|
| 0 | `0.1385 / 0.0730` | `0.3469 / 0.2079` |
| 16 | `0.0373 / 0.0208` | `0.3088 / 0.1825` |
| 64 | `0.0313 / 0.0164` | `0.2319 / 0.1338` |

### Interpretation

v0B showed real external-source retrieval signal. The MIMIC-trained v0A transfer predictor was much weaker, even though v0A could fit a strong within-INSPECT gap-0 mapping. This suggests v0B is not merely copying a FlatASCEND source-specific mapping.

### Rationale for next step

With the basic signal established, the next question was whether scaling the scaffold improves retrieval and source transfer.

---

## 10. Scaling the mean-token v0B scaffold

### Intuition

The original v0B was deliberately minimal. Scaling dimensionality, training steps, batch size, and context/target caps tests whether the objective has headroom before investing in a more complex architecture.

### What we did

Ran several mean-token scale sweeps:

- 10k baseline;
- 256d / 25k;
- 512d / 20k;
- 256d / 50k;
- 384d / 40k.

### Outcome: utilisation-matched retrieval

| Model | MIMIC gap 0 R@10 | MIMIC gap 64 R@10 | INSPECT gap 0 R@10 | INSPECT gap 64 R@10 |
|---|---:|---:|---:|---:|
| 10k | `0.7622` | `0.6927` | not run | not run |
| 256d / 25k | `0.8597` | `0.7251` | `0.7181` | `0.5779` |
| 512d / 20k | `0.9319` | `0.7265` | `0.7004` | `0.5540` |
| 256d / 50k | `0.8967` | `0.7271` | `0.7156` | `0.5740` |
| 384d / 40k | `0.9209` | `0.7202` | `0.7125` | `0.5650` |

### Outcome: candidate-normalized retrieval

Candidate-normalized retrieval used only groups with at least 128 candidates and sampled at most 512 candidates per group.

| Model | MIMIC gap 0 R@10 | MIMIC gap 64 R@10 | INSPECT gap 0 R@10 | INSPECT gap 64 R@10 |
|---|---:|---:|---:|---:|
| 10k | `0.7774` | `0.6711` | not run | not run |
| 256d / 25k | `0.8723` | `0.7064` | `0.6427` | `0.4845` |
| 512d / 20k | `0.9412` | `0.7102` | `0.6232` | `0.4596` |
| 256d / 50k | `0.9084` | `0.7096` | `0.6414` | `0.4812` |
| 384d / 40k | `0.9301` | `0.6988` | `0.6402` | `0.4716` |

### Outcome: scaled-model controls

For the scaled mean-token models, target/query/time-shift controls were no longer near zero under the fine utilisation policy, because candidate groups are smaller and more homogeneous. But controls remained far below observed retrieval.

Example, 256d/50k:

| Source/gap | observed R@10 | target-shuffle | query-shuffle | time-shift |
|---|---:|---:|---:|---:|
| MIMIC gap 0 | `0.8967` | `0.0763` | `0.0771` | `0.0759` |
| MIMIC gap 64 | `0.7271` | `0.1073` | `0.1069` | `0.1035` |
| INSPECT gap 0 | `0.7156` | `0.1457` | `0.1464` | `0.1430` |
| INSPECT gap 64 | `0.5740` | `0.1396` | `0.1408` | `0.1395` |

### Interpretation

Scaling helped substantially, but the source-transfer pattern matters:

- 512d/20k is strongest on MIMIC.
- 256d/25k and 256d/50k are slightly stronger on INSPECT transfer.
- Longer 256d training improved MIMIC but did not materially improve INSPECT over 256d/25k.
- 384d/40k sits between 256d and 512d on MIMIC, but does not beat 256d on INSPECT.

This suggests a capacity/source-transfer trade-off or over-specialisation risk in the mean-token scaffold.

### Rationale for next step

The mean-token scaffold has been useful, but it is not a true JEPA architecture. The next step is to test whether a sequence encoder with an EMA target branch changes the learning dynamics and improves transfer.

---

## 11. Transformer + EMA target JEPA architecture test

### Intuition

A real JEPA should have separate context and target encoders, usually with a slow-moving/EMA target branch to avoid representational collapse and teacher leakage. The mean-token scaffold tests the objective; a transformer+EMA model tests whether richer sequence structure helps.

### What we ran

Completed first transformer+EMA architecture evaluations on Vast:

- `transformer-ema-256d-20k`: 2-layer transformer, 256d, 8 heads, 20k steps; dev cosine `0.5836`.
- `transformer-ema-256d-40k`: 2-layer transformer, 256d, 8 heads, 40k steps; dev cosine `0.5600`.
- `transformer-ema-384d-15k`: 2-layer transformer, 384d, 8 heads, 15k steps; dev cosine `0.5447`.
- `transformer-ema-128d-25k`: 2-layer transformer, 128d, 4 heads, 25k steps.

All transformer+EMA variants used the same aggregate-only retrieval, candidate-normalized retrieval, and target/query/time-shift controls as the scaled mean-token scaffold.

### Outcome: candidate-normalized retrieval

| Model | MIMIC gap 0 R@10 | MIMIC gap 64 R@10 | INSPECT gap 0 R@10 | INSPECT gap 64 R@10 |
|---|---:|---:|---:|---:|
| transformer+EMA 384d/15k | `0.9608` | `0.7039` | `0.5612` | `0.4278` |
| transformer+EMA 256d/20k | `0.9569` | `0.6823` | `0.5385` | `0.4077` |
| transformer+EMA 256d/40k | `0.9539` | `0.6664` | `0.5151` | `0.3851` |
| transformer+EMA 128d/25k | `0.9003` | `0.6466` | `0.4903` | `0.3806` |

Target/query/time-shift controls for these transformer+EMA runs stayed near chance under the candidate-normalized policy, around R@10 `0.022–0.027`, while observed retrieval was much higher.

### Interpretation

Transformer+EMA strongly improves or matches the best MIMIC gap-0 retrieval and transfers clearly above controls, but does **not yet** beat the best scaled mean-token scaffold under the demanding MIMIC→INSPECT zero-shot transfer gate. The 384d/15k model transfers best among transformer+EMA variants, while shorter 256d training improves transfer over 256d/40k and the smaller 128d/25k run is lower than the other transformer variants. This suggests source-specialisation and training/capacity effects rather than a simple “larger/longer is better” rule.

This means transformer+EMA should not yet replace the mean-token v0B scaffold as the mainline. It is a valuable and promising architecture diagnostic showing strong within-source sequence modelling and non-trivial zero-shot source transfer; the remaining question is how to improve transfer robustness enough to beat the simpler scaffold.

### Final comparison snapshot

The final candidate-normalized comparison ranks the scaled mean-token models above all transformer+EMA variants on INSPECT gap 0/16/64. On INSPECT gap 64, the top models are:

| Rank | Model | Family | R@10 | MRR |
|---:|---|---|---:|---:|
| 1 | scaled-256d/25k | mean-token | `0.4845` | `0.2982` |
| 2 | 256d/50k | mean-token | `0.4812` | `0.2923` |
| 3 | 384d/40k | mean-token | `0.4716` | `0.2904` |
| 4 | scaled-512d/20k | mean-token | `0.4596` | `0.2775` |
| 5 | transformer+EMA 384d/15k | transformer+EMA | `0.4278` | `0.2397` |


---

## Current conclusions

### What is already established

- The governed re-keyed tokenised B1a pipeline works end-to-end.
- Prefix-safe target extraction and leakage auditing are operational.
- v0B latent target prediction learns non-collapsed real-data representations.
- Retrieval signal degrades with horizon in the expected direction.
- v0B beats direct context-family baselines and v0A target-retrieval predictors under several retrieval policies.
- External INSPECT validation shows real transfer signal.
- Controls do not reduce the observed signal to row-order, query-order, or simple time-neighbour artifacts.

### What remains uncertain

- How much of the retrieval signal is clinically meaningful patient-state transition vs care-process/utilisation structure.
- Whether mean-token scaling is discovering robust clinical state or increasingly fitting source-specific token co-occurrence geometry.
- Whether the mean-token scaffold's superior INSPECT transfer reflects true robustness or a simpler representation that avoids source-specialised sequence shortcuts.
- Whether INSPECT should be used only as locked external validation or also in source-balanced training for later v1.

### Best next steps

1. Treat scaled mean-token v0B, especially 256d/25k and 256d/50k, as the current main evidence line.
2. Treat transformer+EMA as a promising architecture diagnostic rather than the mainline until a transfer-aware regularised or source-balanced variant beats mean-token transfer.
3. Pause new Vast compute and write up the aggregate findings before designing another architecture sweep.
4. If another experiment is needed later, make it targeted: regularised/source-balanced transformer+EMA or held-out-source-aware early stopping, not a broad scaling sweep.
5. Keep v0C raw/MEDS-lite and outcome-proximal T2 labels gated until explicit approval.

---

## Short version for a slide

Clinical-JEPA v0 has moved from idea to controlled real-data pilot, while direct Clin-JEPA prior art now constrains the generic novelty claim. The JEPA objective learns non-collapsed future-block representations, beats frozen-FlatASCEND target-retrieval baselines under matched distractors, degrades sensibly with horizon, and transfers to INSPECT. Scaling the mean-token scaffold gives the strongest external-transfer results so far. First transformer+EMA variants are very strong on MIMIC and transfer clearly above controls, but do not yet beat the scaled mean-token scaffold under the hard MIMIC→INSPECT zero-shot gate, so they are promising architecture diagnostics rather than the current v0B mainline.
