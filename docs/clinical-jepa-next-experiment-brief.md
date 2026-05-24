# Clinical-JEPA next experiment brief — generation and TTE readiness first

Date: 2026-05-24

Status: **do not run compute until after write-up review**.

## Why this replaces the transfer-optimisation brief

The previous follow-up brief jumped too quickly to transfer-aware transformer+EMA optimisation. That is not wrong eventually, but it is not the next conceptual bottleneck.

The original Clinical-JEPA question is broader:

> Can a JEPA-style latent patient-state model help produce autoregressive clinical futures that are clinically reasonable?

The v0 work so far answered an upstream representation question: JEPA-style latent prediction learns a real future-block signal and transfers under a hard MIMIC→INSPECT zero-shot test. It did **not** yet answer whether the latent state can be rendered into clinically reasonable explicit event sequences.

Therefore the next experiment should test **generation/readout readiness and TTE-style data quality**, not transfer optimisation.

## How far we got technically

Completed:

- Governed re-keyed B1a MIMIC + INSPECT tokenised substrate.
- Prefix-safe target-block extraction and leakage audits.
- v0A frozen FlatASCEND baselines.
- v0B mean-token JEPA latent prediction.
- Transformer+EMA JEPA latent prediction diagnostics.
- Matched/candidate-normalized retrieval and target/query/time-shift controls.
- INSPECT external zero-shot evaluation.

Not yet completed:

- No JEPA decoder/renderer that emits explicit future token/time sequences.
- No JEPA-conditioned autoregressive rollout.
- No evaluation of JEPA-generated sequences for clinical plausibility.
- No TTE-style cohort/specification audit of generated futures.
- T1 medication-change blocks were extracted in the pilot, but the main retrieval/architecture wave focused on T0 future-block prediction. T2 outcome-proximal labels remain gated/unconfirmed.

Current technical position:

- Clinical-JEPA can predict latent future-block representations.
- The nearest-neighbour target-retrieval setup can be used as a **retrieval-based pseudo-renderer** for a first data-quality/generation-readiness test.
- FlatASCEND remains the explicit autoregressive speaker/generator substrate. A hybrid route would let JEPA predict latent future state first, then either retrieve or condition/render explicit sequences.

## Next question

Before another architecture or transfer-learning run, answer:

> Are the available tokenised sequences and target blocks good enough to support clinically meaningful generation/readout tests, especially TTE-style incident-user or active-comparator scenarios?

## Proposed next atom: metadata availability audit

Before a real scenario scan, run an aggregate-only metadata availability audit. It should answer whether the required per-block aggregate fields are present or safely derivable without opening raw data or exporting row-level examples.

Required fields include structural target-block fields, context/target length, med/lab/state count summaries, equivalent-contact markers, and negative-control markers. Missing fields must be reported as missing; they must not be silently interpreted as zero clinical events.

This gate can be run locally on safe JSON/JSONL manifests or in a governed environment that emits only the aggregate audit report. It is still **not** a reason to restart Vast unless the approved bundle or metadata sidecar exists only there.

## Proposed next atom: TTE/specification data-quality audit

Purpose: test whether the governed B1a token substrate supports clean clinical decision-point cohorts and sequence-quality checks before asking JEPA to generate futures.

This is **not** a causal estimate and **not** a treatment-effect claim. It is a data/specification readiness test.

### Candidate TTE-style specification card

Pick one medication-initiation scenario only, preferably one already compatible with FlatASCEND/ASCEND token families and available in MIMIC + INSPECT.

Minimum card fields:

- clinical question / decision point;
- eligibility window;
- incident-user definition;
- active comparator or clinically plausible non-exposed index event;
- time zero;
- baseline lookback;
- follow-up horizon;
- outcome/proxy candidates;
- censoring rules;
- required negative controls;
- surveillance/contact-intensity controls;
- leakage exclusions;
- minimum cohort-size and positivity thresholds.

### Aggregate outputs

For each source and split, write aggregate-only tables:

- eligible subjects/sequences;
- incident initiators;
- comparator candidates;
- baseline lookback completeness;
- follow-up availability;
- outcome/proxy event rates;
- medication/lab/state observation density;
- equivalent-contact availability for comparators;
- source differences MIMIC vs INSPECT;
- reasons for exclusion;
- warning flags for immortal-time, prevalent-user, time-lag, detection, or endpoint-adjacent leakage risk.

No patient-level rows, examples, source identifiers, HDF5s, embeddings, checkpoints, or raw tokens should be copied into reports.

## Proposed next atom: retrieval-based pseudo-rendering smoke test

If the TTE/specification audit passes, use existing JEPA embeddings without training a new model:

1. For each eligible context, predict a latent future state.
2. Retrieve top-k observed target blocks under the existing matched/candidate-normalized policy.
3. Treat retrieved target blocks as a **pseudo-rendered future set**, not a generated sequence.
4. Evaluate whether these retrieved futures are clinically/specification-consistent in aggregate.

Aggregate checks:

- event-family distribution vs observed futures;
- medication/lab/state density;
- horizon/time-gap plausibility;
- TTE eligibility/follow-up consistency;
- treatment-strategy consistency;
- impossible or contradictory transition rates;
- negative-control event rates;
- surveillance/contact-intensity artifacts.

This tests whether the latent representation can select clinically plausible futures before building a decoder.

## Proposed later atom: explicit renderer/rollout bridge

Only after the above passes, choose one renderer route:

1. **Retrieval renderer:** return nearest observed future blocks as a non-generative clinical analogue / case-based future set.
2. **FlatASCEND speaker bridge:** use Clinical-JEPA to choose or condition a latent future state, then use FlatASCEND-style autoregressive continuation as the explicit event speaker.
3. **JEPA decoder head:** train a small decoder from predicted latent state to future token/time distributions, with strong syntax/time plausibility checks.

Do not start with a large new transformer+EMA optimisation run until the renderer question is framed.

## Success gates

The TTE/data-quality atom succeeds if:

- at least one scenario has enough incident-user and comparator candidates in MIMIC;
- the same scenario is at least measurable in INSPECT;
- time zero, baseline lookback, and follow-up can be defined without future leakage;
- outcome/proxy candidates are prefix-safe or explicitly marked as not ready;
- surveillance/contact-intensity controls are available;
- exclusion reasons and source differences are reportable in aggregate.

The pseudo-rendering atom succeeds if:

- retrieved futures are far more specification-consistent than shuffle controls;
- horizon/time and event-family distributions are plausible;
- negative controls do not move in clinically nonsensical ways;
- artifacts point to fixable data/specification issues rather than model collapse.

## Stop criteria

Stop before new model training if:

- no clean incident-user/comparator scenario can be specified;
- required outcome/proxy labels are not prefix-safe;
- source differences make INSPECT uninterpretable for the scenario;
- controls suggest retrieval/pseudo-rendering is mostly utilisation or contact-density matching;
- the task requires v0C raw/MEDS-lite or T2 outcome-proximal labels without explicit approval.

## Relation to transfer optimisation

Transfer-aware transformer+EMA regularisation remains a plausible later experiment, but it should be downstream of the generation/readout question. If the substrate cannot yet support clinically meaningful TTE-style generation tests, improving MIMIC→INSPECT retrieval alone will not answer the core Clinical-JEPA/ORCA question.

## Governance boundary

Do not include raw/patient-level data, HDF5s, source-ID maps, embeddings, checkpoints, secrets, `.env` files, passfiles, or time-limited URLs in public artifacts. v0C raw/MEDS-lite and outcome-proximal T2 labels remain gated until explicitly approved.
