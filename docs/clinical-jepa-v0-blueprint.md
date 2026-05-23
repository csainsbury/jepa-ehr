---
title: Clinical-JEPA pilot atomic blueprint
created: 2026-05-23
updated: 2026-05-23
status: draft-for-automated-development
project: ascend-orca-clinical-jepa
intended_use: outcome-loops-and-physarum
safety_tier: safe_distilled
canonical_sources:
  - wiki/concepts/joint-embedding-predictive-architecture.md
  - wiki/synthesis/orca-and-jepa-representation-space-translation.md
  - wiki/sources/2026-05-23-jepa-autoregression-batch.md
  - wiki/entities/flatascend.md
  - wiki/sources/2026-04-11-sainsbury26-flatascend.md
  - wiki/sources/2026-04-11-sainsbury26-orca-iatric.md
  - wiki/synthesis/ascend-research-arc.md
supporting_scouts:
  - clinical-jepa-blueprint/wiki-synthesis.md
  - clinical-jepa-blueprint/skeptic.md
  - clinical-jepa-blueprint/analogy.md
---

# Clinical-JEPA pilot atomic blueprint

## 0. One-line objective

Build a staged Clinical-JEPA experimental programme that tests whether **latent clinical-state prediction** can improve on, complement, or reveal limitations of the existing [[entities/flatascend|FlatASCEND]] / [[entities/orca|ORCA]] substrate, without prematurely locking into FlatASCEND-derived representations.

The core project claim to test is:

> Clinical-JEPA should be a **reader / latent transition model** for patient state; FlatASCEND remains the **speaker / explicit coded-event rollout model**. The first milestone is not a full model, but a fair substrate bake-off that decides whether to scaffold from FlatASCEND, train on flat tokens from scratch, or move toward richer raw/MEDS/OMOP-lite event representations.

## 1. What this blueprint is for

This file is intended as a durable, machine-usable plan for:

- `/outcome define ...` — freeze artifact-level outcome contracts and rubrics;
- `/outcome evaluate ...` — grade implementation artifacts and route bundles;
- `/taskroutes ...` — generate or refine candidate route cards;
- `/tasktick ...` / `self-closing-task` — execute bounded planning or implementation ticks;
- Physarum `/scout` waves — explore literature, analogy, and skeptic directions before route selection;
- Nightshift / local workers — run bounded local development jobs with explicit stop gates.

This is **not** approval to access raw clinical data, modify project repos, or run external systems. It is a planning artifact.

## 2. Source-backed starting assumptions

1. JEPA is latent prediction: predict target representations from context representations rather than reconstructing raw observations (source: [[concepts/joint-embedding-predictive-architecture]]).
2. Newer JEPA work blurs the boundary with autoregression: sequential, probabilistic, bidirectional, action-conditioned and generative-adjacent JEPA variants exist (source: [[sources/2026-05-23-jepa-autoregression-batch]]).
3. ORCA is **not currently a JEPA**; it is a broader language-of-EHR programme whose empirical substrate is FlatASCEND plus alignment/translation analyses (source: [[synthesis/orca-and-jepa-representation-space-translation]], [[sources/2026-04-11-sainsbury26-orca-iatric]]).
4. FlatASCEND is a strong existing speaker/generator baseline, using flat composite tokens, a GPT-style decoder, continuous time head, scheduled sampling, and incident-user pharmacological evaluation (source: [[entities/flatascend]], [[sources/2026-04-11-sainsbury26-flatascend]]).
5. FlatASCEND has known weaknesses: cross-site zero-shot transfer, outpatient long-gap temporal fidelity, and observational confounding in pharmacological directions (source: [[sources/2026-04-11-sainsbury26-flatascend]]).
6. The ASCEND arc repeatedly warns that evaluation must avoid trivial distributional metrics, leakage, teacher circularity, pseudo-replication, and overclaiming causal meaning from observational EHR (source: [[synthesis/ascend-research-arc]]).

## 3. Non-negotiable guardrails

### 3.1 Data and sensitivity

- Do not copy raw clinical data, restricted full text, governance documents, or patient-level records into cloud-agent context.
- Use only safe distilled wiki/source pages in agent prompts unless Chris explicitly approves a local/sanitized workflow.
- Raw/MEDS/OMOP-lite work is a **route option**, not a default. It requires a separate data-access/governance checklist before implementation.

### 3.2 Experimental integrity

- Patient-level split always; no trajectory-level pseudo-replication.
- Prefix-only / leakage-safe windows for outcome tasks.
- Predeclare target blocks, endpoints, time horizons, and abandon gates before tuning.
- Include trivial baselines: empirical prior, bag/count/time features, utilisation intensity, and direct query-conditioned baseline.
- Treat action-conditioned transitions as predictive/associational unless a separate causal design is approved.

### 3.3 Conceptual boundaries

- Do not claim FlatASCEND is a JEPA.
- Do not claim Clinical-JEPA replaces FlatASCEND unless explicit event rendering/rollout tasks prove it.
- Do not call action-conditioned latent shifts treatment effects without target-trial/causal design.
- Do not use similarity to FlatASCEND states as the main success criterion.

## 4. Programme structure

```text
v0 — substrate bake-off
  Compare scaffolded FlatASCEND states vs flat-token JEPA from scratch vs raw/MEDS-lite JEPA from scratch vs direct query baseline.
  Output: substrate decision + evidence-backed next route.

v1 — proper Clinical-JEPA
  Train real context encoder + EMA target encoder + predictor on the winning substrate.
  Output: representation model that beats or complements FlatASCEND on ORCA-relevant tests.

v2 — action-conditioned latent world model
  Add medication/intervention/action descriptors and optional renderer/multimodal extensions.
  Output: latent transition tests under incident-user / ORCA / cross-site stress evaluations.
```

## 5. Shared artifact layout for development

Recommended workspace root for implementation planning:

```text
state/task-work/clinical-jepa-pilot/
├── README.md
├── outcome-spec.md
├── route-cards.md
├── data-access-checklist.md
├── splits/
│   ├── split-spec.md
│   └── split-manifest.json
├── leakage-rules/
│   ├── forbidden-token-rules.md
│   └── leakage-audit-plan.md
├── v0/
│   ├── protocol.md
│   ├── arms.md
│   ├── metrics.md
│   ├── run-manifest.json
│   └── results-template.md
├── v1/
│   ├── protocol.md
│   ├── architecture.md
│   ├── training-plan.md
│   └── results-template.md
├── v2/
│   ├── protocol.md
│   ├── action-descriptors.md
│   ├── incident-user-design.md
│   └── results-template.md
└── reviews/
    ├── skeptic-review.md
    ├── leakage-review.md
    └── outcome-loop-feedback.md
```

Implementation repos/code should live separately; this workspace should contain plans, safe summaries, manifests, metrics, and review artifacts.

---

# Part A — v0 substrate bake-off

## A0. v0 outcome contract

**Outcome title:** Clinical-JEPA v0 substrate bake-off protocol and first implementation scaffold.

**Done means:** A reproducible v0 protocol exists with frozen splits, leakage rules, target blocks, model arms, baselines, metrics, and decision gates sufficient for a local worker to implement the first arm without further conceptual decisions.

**Hard gate:** v0 must compare at least two routes, and preferably all four:

- v0A — frozen FlatASCEND hidden-state target;
- v0B — flat-token JEPA from scratch;
- v0C — raw/MEDS/OMOP-lite JEPA from scratch;
- v0D — direct query-conditioned baseline.

## A1. v0 core question

> Which substrate deserves to become the main Clinical-JEPA line: FlatASCEND-derived states, FlatASCEND-style flat tokens with a new JEPA encoder, or richer raw/MEDS/OMOP-lite event data?

## A2. v0 frozen design choices

### A2.1 Primary dataset order

1. **MIMIC-IV tokenised/open pipeline first** — best existing FlatASCEND performance and rich ORCA history.
2. **INSPECT second** — outpatient/generalisation contrast.
3. **eICU-CRD as locked stress test** — do not tune on it initially because FlatASCEND zero-shot transfer already fails.
4. **SCI-Diabetes only after explicit approval** — use for improver paradox if governance and extraction path are safe.

### A2.2 Target block schemas

Keep v0 deliberately small. Use two required target blocks and one optional stress block.

| ID | Target block | Definition sketch | Why |
|---|---|---|---|
| T0 | next fixed event window | next N events or next K hours/days after prefix | easiest smoke test |
| T1 | medication-change interval | window around medication start/switch or class initiation | links to FlatASCEND pharmacology |
| T2 | outcome-proximal block | pre-shock/pre-death/pre-lab-deterioration window | harder stress test; optional in first pass |

### A2.3 Required splits

- Patient-level split.
- Temporal holdout if data volume permits.
- No patient overlap across train/dev/test.
- For target blocks, context must not include target or endpoint-adjacent confirmation tokens.
- Store split manifests with patient IDs hashed or otherwise safely summarized; do not expose raw identifiers.

### A2.4 Required baselines

1. Empirical prior by target-block type and horizon.
2. Bag/count/time model using prefix tokens and simple timing features.
3. Utilisation-intensity baseline: visits, labs, meds, sequence length, measurement density.
4. Frozen FlatASCEND context embedding probe.
5. Direct query-conditioned head: `history + horizon + query -> event/outcome probability`.

## A3. v0 arms

### v0A — frozen FlatASCEND hidden-state target

**Purpose:** fastest smoke test; not a standalone Clinical-JEPA claim.

```text
FlatASCEND tokenised context
  -> frozen FlatASCEND context embedding
  -> predictor(context_embedding, horizon_descriptor)
  -> predicted future target embedding

FlatASCEND tokenised target block
  -> frozen FlatASCEND hidden states
  -> pooled target embedding
```

Variants:

- v0A.1 linear predictor;
- v0A.2 MLP predictor;
- v0A.3 small transformer predictor over context-token embeddings.

Required ablations:

- target layer: shallow/token layer vs mid clinical-concept layer vs final layer;
- pooling: mean vs attention pooling;
- horizon descriptor present vs absent.

Success is **not** target loss alone. It must show independent downstream utility beyond frozen FlatASCEND context embeddings.

### v0B — flat-token Clinical-JEPA from scratch

**Purpose:** test whether FlatASCEND tokenisation is sufficient without inheriting FlatASCEND representations.

```text
flat composite context tokens + time deltas
  -> context encoder
  -> predictor(context_state, horizon/mask descriptors)
  -> predicted target embedding

flat composite target tokens + time deltas
  -> EMA target encoder
  -> target embedding
```

Use collapse regularisation from the start:

- variance/covariance regularisation;
- effective-rank monitoring;
- target-encoder EMA;
- stop if average-target prediction dominates.

### v0C — raw/MEDS/OMOP-lite Clinical-JEPA from scratch

**Purpose:** test whether richer raw event representation provides headroom over flat tokens.

This is not an open-ended raw-data rebuild. It is a minimal richer event schema:

```text
event = {
  timestamp_or_relative_time,
  code,
  code_system,
  event_type,
  numeric_value_if_present,
  unit_if_present,
  medication_route_if_present,
  dose_bin_if_present,
  visit_or_admission_id,
  static_demographics_available_for_modelling
}
```

Minimum implementation requirement:

- use approved/safe local data pipeline only;
- keep a reproducible extraction manifest;
- include value/hierarchy/time channels only if they can be extracted safely;
- avoid attempting every raw field in v0.

This route should win if tasks require information missing from flat tokens, e.g. dose, route, continuous lab thresholds, explicit visit structure, code hierarchy, or cross-site ontology mapping.

### v0D — direct query-conditioned baseline

**Purpose:** reviewer-proofing and product-interface pressure.

```text
history + horizon + query_descriptor -> probability / score for target event/outcome
```

Queries may include:

- next medication class;
- next lab quintile / direction;
- event onset within horizon;
- mortality/shock within horizon.

If this beats Clinical-JEPA on fixed endpoints, Clinical-JEPA may still be useful as a representation learner, but not as the preferred endpoint interface.

## A4. v0 atomic tasks

### Protocol atoms

- **CJ-V0-P01** — Create v0 protocol skeleton with frozen scope, data sources, target blocks, arms, baselines, metrics, and gates.
- **CJ-V0-P02** — Create data-access/checklist distinguishing safe distilled planning, tokenised-data implementation, and raw/MEDS-lite implementation.
- **CJ-V0-P03** — Define patient-level split policy and split manifest schema.
- **CJ-V0-P04** — Define leakage audit rules: forbidden future tokens, endpoint confirmation, horizon boundaries, sequence truncation.
- **CJ-V0-P05** — Define target block extraction rules for T0/T1/T2.

### Implementation atoms

- **CJ-V0-I01** — Implement target-block extractor on safe tokenised sequences.
- **CJ-V0-I02** — Implement shared metrics harness: retrieval, probes, baselines, collapse diagnostics.
- **CJ-V0-I03** — Implement v0A cached FlatASCEND embedding extraction.
- **CJ-V0-I04** — Train v0A.1 linear and v0A.2 MLP predictor.
- **CJ-V0-I05** — Implement v0B minimal EMA target encoder over flat tokens.
- **CJ-V0-I06** — Implement raw/MEDS-lite schema adapter stub and extraction feasibility report.
- **CJ-V0-I07** — Implement v0D direct query baseline.

### Review atoms

- **CJ-V0-R01** — Run leakage review before first training run.
- **CJ-V0-R02** — Run skeptic review for teacher circularity and care-intensity confounding.
- **CJ-V0-R03** — Run outcome-loop evaluation of v0 protocol artifacts.
- **CJ-V0-R04** — Physarum scout wave if raw/MEDS-lite feasibility or novelty remains uncertain.

## A5. v0 metrics

### Representation metrics

- target embedding retrieval Recall@1/5/10;
- MRR against held-out target blocks;
- true target rank among distractors matched by horizon/site/utilisation;
- CKA/Procrustes only as diagnostic, not as success.

### Clinical metrics

- prefix-only mortality/shock probe;
- next lab category/direction;
- next medication family;
- onset vs recurrence where feasible;
- medication/lab preservation in retrieved target windows.

### Collapse/care-process diagnostics

- effective rank;
- embedding variance by dimension;
- covariance spectrum;
- nearest-neighbour patient diversity;
- correlation with sequence length, prior visit count, lab count, medication count, admission duration;
- horizon sensitivity: predictions must change when horizon changes;
- patient/time shuffle controls.

## A6. v0 decision gates

### Promote v0A as main scaffold only if

- beats empirical/bag/utilisation baselines on target retrieval;
- improves or complements frozen FlatASCEND context embeddings on at least one independent clinical probe;
- layer/pooling ablation shows clinical-state targets outperform shallow token targets;
- gains survive sequence-length/utilisation matching;
- results can be described as more than “predicting FlatASCEND from FlatASCEND.”

### Promote v0B as main line if

- no collapse;
- matches or beats v0A without teacher circularity;
- performs well on at least two target block types;
- preserves medication/lab trajectory information;
- cross-site or temporal holdout degradation is less severe than v0A.

### Promote v0C/raw-lite if

- richer event fields materially improve tasks that require information absent from flat tokens;
- extraction is reproducible and governance-safe;
- gains persist after leakage and utilisation controls;
- raw-lite does not become an unbounded ETL project.

### Park the whole JEPA route if

- all arms are matched by simple bag/utilisation/query baselines;
- embeddings collapse or encode mainly healthcare-contact intensity;
- improvements are only same-site/easy-target retrieval;
- no version produces an independent advantage over FlatASCEND-derived representations.

---

# Part B — v1 proper Clinical-JEPA

## B0. v1 outcome contract

**Outcome title:** Clinical-JEPA v1 architecture and training run on chosen substrate.

**Done means:** A real JEPA model with context encoder, EMA target encoder, predictor, anti-collapse regularisation, frozen evaluation suite, and locked baselines is trained and evaluated on the chosen v0 substrate.

## B1. v1 core question

> Can a real Clinical-JEPA learn a better latent patient-state representation than FlatASCEND-derived embeddings or simple temporal encoders?

## B2. v1 architecture template

```text
context sequence/window
  -> context encoder
  -> context state
  -> predictor(context state, horizon, site, mask/action descriptor)
  -> predicted target embedding

target sequence/window
  -> EMA target encoder
  -> target embedding

loss = latent prediction loss + collapse regularisation + optional energy/probabilistic term
```

## B3. v1 variants

| ID | Variant | When to use |
|---|---|---|
| v1A | flat-token Clinical-JEPA | if v0B wins or is close to v0A |
| v1B | raw/MEDS-lite Clinical-JEPA | if v0C shows real headroom |
| v1C | teacher-regularised hybrid | if v0A is useful but circularity needs reducing |
| v1D | sequential/probabilistic JEPA | if multiple ordered target blocks or uncertainty matter |

## B4. v1 atomic tasks

- **CJ-V1-P01** — Freeze chosen substrate and rationale from v0.
- **CJ-V1-P02** — Write architecture spec: encoders, predictor, target EMA, collapse regulariser, horizons.
- **CJ-V1-P03** — Write training config schema and model-card template.
- **CJ-V1-I01** — Implement context encoder and EMA target encoder.
- **CJ-V1-I02** — Implement predictor with horizon/site/mask descriptors.
- **CJ-V1-I03** — Implement anti-collapse loss and diagnostics dashboard.
- **CJ-V1-I04** — Train v1A or v1B on MIMIC train split.
- **CJ-V1-I05** — Evaluate on MIMIC test and INSPECT/eICU stress split as permitted.
- **CJ-V1-I06** — Run frozen attentive probes, not only mean-pooled linear probes.
- **CJ-V1-R01** — Run outcome-loop evaluation against v1 success rubric.
- **CJ-V1-R02** — Run fresh skeptic review for leakage, care-process collapse, and known-method reduction.

## B5. v1 evaluation

### Required comparisons

- v0 best arm;
- frozen FlatASCEND context embeddings;
- bag/count/time baseline;
- utilisation-intensity baseline;
- direct query-conditioned baseline;
- BEHRT/encoder-style baseline if available.

### Required tasks

1. Latent target retrieval.
2. Prefix-only clinical probes.
3. Cloze-style next lab/medication prediction.
4. Cross-site alignment or stress test.
5. Onset vs recurrence / chronic-code recurrence separation where feasible.
6. Collapse/care-process diagnostics.

## B6. v1 decision gates

Promote to v2 only if Clinical-JEPA beats or complements FlatASCEND-derived representations on at least two independent axes:

- better latent retrieval under matched distractors;
- better frozen clinical probe;
- better cross-site alignment/stress transfer;
- better cloze/outcome prediction;
- healthier representation geometry;
- better onset/trajectory sensitivity.

Park if:

- gains are only target-loss improvements;
- gains disappear under utilisation matching;
- query-conditioned baseline dominates fixed endpoint tasks;
- medication/lab detail is washed out;
- the model is indistinguishable from standard masked EHR modelling without a stronger target-block/evaluation story.

---

# Part C — v2 action-conditioned Clinical-JEPA

## C0. v2 outcome contract

**Outcome title:** Action-conditioned Clinical-JEPA latent transition pilot.

**Done means:** A Clinical-JEPA model predicts future latent clinical state conditioned on a clinical action/intervention descriptor and passes incident-user-style, negative-control, and ORCA-style semantic preservation tests.

## C1. v2 core question

> Does adding clinical actions/interventions turn Clinical-JEPA into a useful latent transition model, beyond generic representation learning?

## C2. v2 model template

```text
current patient context/state
+ action descriptor
+ horizon descriptor
+ optional site/modality descriptor
  -> predictor/world-model
  -> predicted future latent clinical state
```

Actions may include:

- medication start;
- medication switch;
- drug class A vs B;
- dose/route bin if raw-lite supports it;
- protocol intervention;
- target-trial treatment arm label;
- observation/comparator condition.

## C3. v2 variants

| ID | Variant | Purpose |
|---|---|---|
| v2A | action-conditioned pharmacological JEPA | latent version of FlatASCEND pharmacology tests |
| v2B | latent-first speaker bridge | JEPA predicts state; FlatASCEND/decoder renders events |
| v2C | cross-site/cycle consistency | MIMIC→INSPECT→MIMIC or EHR→English→EHR latent preservation |
| v2D | query-conditioned state head | EveryQuery/RAVEN-style interface on top of JEPA state |
| v2E | PRIMA-HD multimodal extension | retinal image as dated latent-state observation; optional only |

## C4. v2 atomic tasks

- **CJ-V2-P01** — Define allowed action descriptors and causal language boundaries.
- **CJ-V2-P02** — Select 2–3 pharmacological tests with clean mechanistic expectations and known confounding controls.
- **CJ-V2-P03** — Write incident-user paired-prefix design with patient-level statistics.
- **CJ-V2-I01** — Implement action-conditioned predictor.
- **CJ-V2-I02** — Implement action-shuffle, horizon-shuffle, and negative-control tests.
- **CJ-V2-I03** — Compare latent predicted directions against FlatASCEND explicit rollouts.
- **CJ-V2-I04** — If renderer is used, evaluate preservation of latent clinical decision after rendering.
- **CJ-V2-I05** — Optional PRIMA-HD: implement retinal embedding as dated observation only after linked data governance is ready.
- **CJ-V2-R01** — Run skeptic review for causal overclaim and DPO-like circularity.
- **CJ-V2-R02** — Run outcome-loop evaluation against action-conditioned rubric.

## C5. v2 evaluation

### Pharmacology / action tests

Use tests such as:

- warfarin → INR;
- steroid → glucose;
- diuretic → potassium;
- statin/null controls;
- insulin/glucose confounding check.

Required:

- patient-level paired statistics;
- action-shuffle control;
- negative-control outcome;
- baseline severity/indication stratification;
- no claim of causal treatment effect unless separately designed.

### ORCA-style tests

- medication preservation;
- lab trajectory direction;
- improver-paradox recovery if SCI-Diabetes is available;
- early septic-shock prefix signal;
- cross-dialect alignment;
- cycle consistency if site/modality mapping is included.

### Renderer tests

If a FlatASCEND/decoder renderer is added, evaluate:

- whether rendered events preserve predicted medication/lab/outcome state;
- whether rendering introduces hallucinated or clinically contradictory events;
- whether latent-first rendering beats direct FlatASCEND rollout on semantic preservation.

## C6. v2 decision gates

Promote to full paper/project if:

- action-conditioned latent transitions recover clean mechanistic directions at least as well as FlatASCEND, with clearer uncertainty or fewer artefacts;
- action/horizon shuffles collapse effect sizes appropriately;
- negative controls remain negative;
- ORCA phenomena reproduce or sharpen in latent space;
- cross-site/cycle tests preserve medication/outcome semantics better than baseline alignment.

Do not promote if:

- action conditioning tracks prescribing indication or care intensity only;
- effects disappear after baseline severity/utilisation matching;
- latent shifts are uninterpretable and cannot be probed or rendered safely;
- evaluation/reward circularity appears similar to the FlatASCEND DPO failure.

---

# Part D — raw-vs-flat decision tree

Use this decision tree after v0.

```text
Did v0A/v0B beat simple baselines and FlatASCEND context embeddings?
  ├─ no -> do not continue scaffolded route; inspect v0C/v0D or park Clinical-JEPA.
  └─ yes
      ↓
Did gains survive utilisation/sequence-length/lab-density controls?
  ├─ no -> likely care-process model; redesign target blocks or raw-lite controls.
  └─ yes
      ↓
Did v0C raw-lite show gains on tasks requiring absent flat-token information?
  ├─ yes -> promote raw/MEDS-lite as v1 main route.
  └─ no
      ↓
Did v0B match v0A without teacher circularity?
  ├─ yes -> promote flat-token JEPA from scratch as v1.
  └─ no
      ↓
Use v0A only as bootstrap/scaffold, with explicit plan to replace teacher targets in v1.
```

High-severity triggers to abandon FlatASCEND scaffold as main line:

1. no gain over frozen FlatASCEND embeddings;
2. bag/utilisation baseline parity;
3. performance dominated by care intensity;
4. target tasks need fields absent from flat tokens;
5. cross-site brittleness follows FlatASCEND failure mode;
6. only successes are prior ORCA/FlatASCEND discoveries;
7. collapse diagnostics fail;
8. honest methods sentence reads only: “we predict FlatASCEND states from FlatASCEND tokens.”

---

# Part E — risk register

| Risk | Trigger | Mitigation | Stop condition |
|---|---|---|---|
| Teacher circularity | target and evaluation both FlatASCEND-derived | independent probes, compare to teacher baseline, v0B/v0C arms | no independent gain |
| Token bottleneck | flat tokens omit dose/value/hierarchy needed for task | raw/MEDS-lite arm | raw-lite wins on absent-information tasks |
| Care-process collapse | embeddings track visits/labs/contact intensity | matched controls, utilisation baseline | gains vanish after matching |
| Leakage | future/endpoint tokens enter context | prefix-only rules, forbidden-token audit | leakage audit fails |
| Pseudo-replication | many windows/trajectories per patient treated independent | patient-level split/statistics | cannot aggregate at patient level |
| Known-method reduction | reviewer says masked EHR BERT / EveryQuery | target-block novelty, baselines, JEPA-specific collapse/eval | no distinct contribution |
| Causal overclaim | action-conditioned results described as effects | incident-user framing, negative controls, cautious language | cannot separate indication/confounding |
| Raw-route sprawl | raw-lite expands into ETL rebuild | minimal schema, fixed spike budget | no reproducible extractor |
| PHI/governance exposure | raw/patient-level data in unsafe context | local-only/sanitized workflows | data cannot be handled safely |

---

# Part F — reviewer objections to pre-answer

1. **“Is this just BEHRT/CEHR-BERT masked modelling?”**
   - Answer with target-block design, EMA JEPA objective, latent retrieval, action-conditioned transitions, and ORCA-style representation tests.
2. **“If targets are FlatASCEND states, why is this not distillation?”**
   - v0A is explicitly a bootstrap arm only; v0B/v0C are non-teacher arms; success requires independent gains.
3. **“Does the model learn healthcare utilisation?”**
   - Include utilisation baselines, matching, correlation diagnostics, and negative controls.
4. **“Why not direct endpoint prediction/querying?”**
   - Include v0D direct query baseline and concede it may be the better endpoint interface if it wins.
5. **“Are action-conditioned shifts causal?”**
   - No; they are predictive/associational unless separate target-trial design is added.
6. **“Why not start from raw data?”**
   - v0C gives raw/MEDS-lite a fair chance; promote it if it wins on tasks where flat tokens lose information.

---

# Part G — Outcome-loop scaffolding

## G1. Suggested first `/outcome define` request

```text
/outcome define Clinical-JEPA v0 substrate bake-off protocol.
Artifact root: state/task-work/clinical-jepa-pilot/v0/
Reference blueprint: state/workflows/2026-05-23-clinical-jepa-blueprint/atomic-blueprint.md
Good enough: a local worker can implement v0A and v0B without making conceptual decisions; target blocks, splits, leakage rules, baselines, metrics, and decision gates are frozen; raw/MEDS-lite is scoped as a bounded feasibility arm, not an open-ended rebuild.
```

## G2. Suggested outcome spec draft

```markdown
# Outcome spec — Clinical-JEPA v0 substrate bake-off protocol

## Title
Clinical-JEPA v0 substrate bake-off protocol

## Description
A reproducible, leakage-aware protocol for comparing FlatASCEND-hidden-state scaffold, flat-token JEPA from scratch, raw/MEDS-lite JEPA feasibility, and direct query baseline under a shared evaluation harness.

## Artifact globs
- `*.md`
- `*.json`
- `v0/**/*.md`
- `splits/**/*.json`
- `leakage-rules/**/*.md`

## Required artifacts
- `v0/protocol.md`
- `v0/arms.md`
- `v0/metrics.md`
- `splits/split-spec.md`
- `leakage-rules/leakage-audit-plan.md`
- `data-access-checklist.md`

## Reference files
- `state/workflows/2026-05-23-clinical-jepa-blueprint/atomic-blueprint.md`
- `wiki/concepts/joint-embedding-predictive-architecture.md`
- `wiki/synthesis/orca-and-jepa-representation-space-translation.md`
- `wiki/sources/2026-05-23-jepa-autoregression-batch.md`

## Rubric

### Scope freeze
The protocol defines v0A/v0B/v0C/v0D or explicitly justifies any omitted arm, with bounded first-run scope.

### Leakage safety
Patient-level split, prefix-only context, target-block horizon rules, forbidden future tokens, and endpoint-adjacent leakage audit are specified.

### Baseline adequacy
Empirical prior, bag/count/time, utilisation-intensity, frozen FlatASCEND, and direct query-conditioned baselines are included or explicitly deferred with rationale.

### Evaluation adequacy
Metrics include target retrieval, independent clinical probes, ORCA-style checks where feasible, and collapse/care-process diagnostics.

### Decision gates
Promote/park/abandon gates are predeclared, including conditions for switching from FlatASCEND scaffold to raw/MEDS-lite.

### Automation readiness
A local worker can implement the next atom without unresolved conceptual choices.

## Safety tier
safe_distilled

## Budget
- Max iterations: 3
- Max wall minutes: 60
- Max cost USD: 0.00

## Guardrails
- No raw writes.
- No patient-level data in cloud context.
- No canonical wiki/task edits unless explicitly approved.
- Deterministic gates override LLM scoring.
```

## G3. Deterministic checks to add later

- required files present;
- no artifact path under `raw/`;
- no obvious patient identifiers in markdown artifacts;
- every v0 arm has baseline and metric entries;
- every metric has split and leakage condition;
- every promote gate has a matching park/abandon gate.

---

# Part H — Physarum route cards

These route cards can seed `/taskroutes` or `/scout` follow-up.

## Route: Frozen FlatASCEND scaffold

### Core idea
Use FlatASCEND tokenised sequences and frozen FlatASCEND hidden states as v0 target embeddings.

### Why it may be rewarding
- novelty: medium-low as standalone, medium as bootstrap;
- conceptual elegance: medium;
- project fit: very high;
- tractability: high;
- evidence strength: existing FlatASCEND assets.

### Evidence
- FlatASCEND model/datasets/results: [[entities/flatascend]], [[sources/2026-04-11-sainsbury26-flatascend]].
- ORCA/JEPA synthesis says FlatASCEND speaker and Clinical-JEPA reader are complementary: [[synthesis/orca-and-jepa-representation-space-translation]].

### Project relevance
Fastest smoke test and engineering scaffold.

### Next move
Create v0A protocol and implement cached embedding extraction.

### Risks / objections
Teacher-student circularity; inherited FlatASCEND weaknesses; not a standalone Clinical-JEPA claim.

### Verification needed
Independent downstream gain beyond frozen FlatASCEND context embeddings; utilisation controls; layer ablation.

### Status recommendation
verify / bootstrap only

### Confidence
high

## Route: Flat-token JEPA from scratch

### Core idea
Train a real JEPA encoder/EMA-target/predictor over the existing flat composite token streams.

### Why it may be rewarding
- novelty: medium;
- conceptual elegance: high;
- project fit: high;
- tractability: medium-high;
- evidence strength: strong local tokenisation, JEPA literature support.

### Project relevance
Best compromise between avoiding teacher circularity and avoiding raw-data sprawl.

### Next move
Implement v0B minimal encoder/EMA target on the same target blocks as v0A.

### Risks / objections
May reduce to masked EHR modelling; flat tokens may cap capability.

### Verification needed
Beats v0A or matches it without circularity; collapse diagnostics healthy; target-block ablations matter.

### Status recommendation
promote for v0 bake-off

### Confidence
medium-high

## Route: Raw/MEDS-lite JEPA from scratch

### Core idea
Train Clinical-JEPA on a richer, reproducible event schema with numeric values, hierarchy, visits, and dose/route where safely available.

### Why it may be rewarding
- novelty: medium-high;
- conceptual elegance: medium;
- project fit: medium;
- tractability: medium-low;
- evidence strength: plausible headroom, higher implementation risk.

### Project relevance
Only route that can clearly escape FlatASCEND token/representation ceilings.

### Next move
Write a raw/MEDS-lite feasibility checklist and one tiny extractor spec; do not build full raw pipeline until v0C gate is met.

### Risks / objections
ETL sprawl, leakage, governance, sparsity, reproducibility.

### Verification needed
Wins specifically on tasks needing information absent from flat tokens; extraction is governance-safe and reproducible.

### Status recommendation
verify as bounded parallel spike

### Confidence
medium

## Route: Direct query-conditioned baseline

### Core idea
Add `history + query + horizon -> event/outcome score` baseline, EveryQuery-style, to pressure-test whether JEPA is needed for endpoint tasks.

### Why it may be rewarding
- novelty: low-medium;
- conceptual elegance: medium;
- project fit: high as baseline/interface;
- tractability: medium;
- evidence strength: strong reviewer-proofing.

### Project relevance
Essential control. May become the safer product endpoint interface even if JEPA is the representation learner.

### Next move
Define 3–5 query descriptors and implement v0D baseline in shared metrics harness.

### Risks / objections
Could outperform JEPA on fixed endpoints and shift the project direction.

### Verification needed
Compare on fixed endpoint tasks; if it wins, preserve Clinical-JEPA for representation/alignment/world-model goals.

### Status recommendation
promote as mandatory baseline

### Confidence
high

## Route: Action-conditioned latent pharmacology

### Core idea
Condition latent transitions on medications/interventions and evaluate paired future-state movement.

### Why it may be rewarding
- novelty: high;
- conceptual elegance: high;
- project fit: high;
- tractability: medium;
- evidence strength: high-upside but risky.

### Project relevance
The v2 route that turns Clinical-JEPA into a world-model candidate.

### Next move
Park until v1 has healthy representations; then define one incident-user action test.

### Risks / objections
Confounding by indication, causal overclaim, DPO-like circularity.

### Verification needed
Action shuffle, negative controls, patient-level paired tests, severity/utilisation stratification.

### Status recommendation
park until v1; then verify

### Confidence
medium

## Route: PRIMA-HD multimodal latent-state observation

### Core idea
Treat retinal images as dated observations of latent vascular/metabolic state that improve future EHR latent prediction.

### Why it may be rewarding
- novelty: high;
- conceptual elegance: high;
- project fit: medium-high;
- tractability: low-medium;
- evidence strength: future-dependent.

### Project relevance
Strong v2/v3 extension, not part of first EHR-only proof.

### Next move
Park until linked retinal/EHR data access and EHR-only v1 are ready.

### Risks / objections
Governance, image/site artefacts, modality missingness, clinical overclaim.

### Verification needed
Date-aligned real retinal/EHR links; image-patient and date-shuffle controls; comparison to supervised late fusion.

### Status recommendation
park

### Confidence
medium

---

# Part I — Suggested Physarum scout prompt

Use this if the next step is exploratory rather than implementation:

```text
/scout ascend-orca "Clinical-JEPA v0 substrate frontier. Why now: JEPA/autoregression paper batch suggests latent clinical-state prediction could complement FlatASCEND, but we need to choose between FlatASCEND-hidden-state scaffold, flat-token JEPA from scratch, raw/MEDS-lite JEPA from scratch, and direct query-conditioned baselines. Target: implementation novelty, reviewer risk, and substrate decision. Safe context: state/workflows/2026-05-23-clinical-jepa-blueprint/atomic-blueprint.md plus maintained wiki pages only; do not inspect raw clinical data. Constraints: avoid teacher circularity, care-process confounding, PHI exposure, causal overclaim, and unbounded raw ETL. Known dead ends: calling FlatASCEND a JEPA; evaluating only by similarity to FlatASCEND; treating action-conditioned shifts as causal effects. Return promoted/verify/park/reject route cards and the smallest next executable development atom."
```

Expected scout outputs:

- literature/precedent route risks;
- analogy/frame route cards;
- skeptic/dead-end route cards;
- checkpoint with recommended first development route;
- candidate route-ledger entries under `ascend` or `orca` depending on project naming.

---

# Part J — First bounded development tick

If Chris wants to execute rather than scout, the smallest safe next tick is:

```text
/tasktick Create the Clinical-JEPA v0 protocol workspace from the atomic blueprint. Build only planning artifacts under state/task-work/clinical-jepa-pilot/v0/: protocol.md, arms.md, metrics.md, split-spec.md, leakage-audit-plan.md, and data-access-checklist.md. Do not touch raw data or project repos. Use the blueprint at state/workflows/2026-05-23-clinical-jepa-blueprint/atomic-blueprint.md as the reference and prepare it for /outcome evaluation.
```

Expected artifacts:

- `state/task-work/clinical-jepa-pilot/v0/protocol.md`
- `state/task-work/clinical-jepa-pilot/v0/arms.md`
- `state/task-work/clinical-jepa-pilot/v0/metrics.md`
- `state/task-work/clinical-jepa-pilot/splits/split-spec.md`
- `state/task-work/clinical-jepa-pilot/leakage-rules/leakage-audit-plan.md`
- `state/task-work/clinical-jepa-pilot/data-access-checklist.md`

Outcome-loop evaluation should run before implementation.
