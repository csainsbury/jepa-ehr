---
title: Encode-empty (Option A) — design blueprint (design-panel synthesis)
created: 2026-07-06
status: BLUEPRINT — awaiting Pi blueprint gate before latent-side implementation
produced_by: 3-proposal design panel + adversarial critique + synthesis (workflow encode-empty-design)
scope: make a zero-event wall-clock target a first-class, collapse-safe, decodable latent in the v0B MeanToken JEPA
---

# Encode-empty blueprint

## Chosen mechanism — HYBRID (occupancy hurdle head + frozen z_empty prototype)
Reject Proposal 1 (v0B-only `[NO_EVENT]` vocab bump 1050→1051) as dominated
(shared-embedding-table drift forces a tuned repulsion aux; worst generalization).
Adopt a hybrid of the two strong proposals:

- **Occupancy/count "hurdle" head (P3) — the authoritative silence signal.** A
  supervised binary occupancy logit (BCE) + a `log1p` count head (Huber). Absence
  lives in a **scalar channel** the scale-invariant cosine loss cannot represent or
  game → eliminates the `mean_embed(zero)+cosine+var-floor` degeneracy by
  construction. It is also the **only fair frozen-350M comparator path** (an external
  occupancy probe on the frozen 350M *context* state respects the locked 1050 vocab).
- **Frozen `register_buffer` z_empty prototype (P2) — silence as a latent.** A
  seeded, unit-norm, **frozen (not a Parameter)** vector returned target-side for
  empty blocks: `target_latent(ids, is_empty) = where(is_empty, z_empty, mean_embed(ids))`.
  Zero new params, immovable (cannot chase the predictor even in principle), model-
  agnostic, and it bypasses the confirmed all-`[PAD]` zero-vector / NaN-cosine bug.

**Division of labour:** the occupancy scalar is the authoritative empty/non-empty
label (resolves the empty-vs-1-event boundary that direction-only cosine cannot);
z_empty gives silence a decodable, retrievable latent region. They cross-check — if
z_empty drifts into the populated manifold, occupancy still separates; if occupancy
degenerates to base-rate, the beats-marginal gate fires; disagreement is a collapse
alarm, and rung-1 empty-recall is a live falsifier.

## Collapse-avoidance (4 structural locks)
1. **Absence in a scalar channel** — occupancy BCE is direction-free; cosine cannot
   game it. Base-rate collapse caught by a pre-registered per source×horizon
   "beats-marginal" AUC/Brier falsifier.
2. **z_empty frozen buffer under `no_grad`** — zero gradient reaches it; it physically
   cannot co-collapse with the predictor onto a shared axis. Populated rows keep their
   own `mean_embed` cosine targets pushing predictions *away* from z_empty.
3. **Empties excluded from the variance regularizer** — the single shared attractor is
   never a batch-variance sink; populated manifold spread preserved (optional VICReg
   off-diagonal cov on non-empty preds).
4. **Empty-fraction cap / class-balanced sampler** — "always predict silence" cannot
   dominate the gradient. Plus stratified diagnostics (cos(pred,z_empty) high for
   empty / low for populated; empty-vs-populated margin ≥ pre-registered 0.15).

**Test-enforced hard invariants:** z_empty is a buffer not a Parameter (unit test);
all target construction stays inside `no_grad` (AST/grep test); the single-span-reader
invariant — no consumer computes a span from a raw `target_start_ref` outside the
helper (grep test), because one un-migrated site silently reintroduces `arr[0:]`
upstream of the encoder where the model cannot detect it.

## Component change list (see the 8 sites in downstream-minus1-integration-scoping.md)
- **Shared helper** `clinical_jepa/targets/block_spans.py` (new, imported everywhere):
  `is_empty_target`, `read_target_span`(→ empty array, never `arr[0:]`),
  `empty_target_len`(0 for empty), `target_occupancy`(occ∈{0,1}, count). Single source
  of truth; the ONLY span reader.
- **SITE 1 — leakage boundary BUG** `audit/run_leakage_audit.py:264` (do FIRST):
  `context_end_ref >= target_start_ref` = `10 >= -1` → false `horizon_boundary`
  violation → audit fails on any empty run. Skip empty blocks in the boundary/dup
  loop; count `empty_target_audited`.
- **v0B model** `arms/v0b/mean_token_model.py`: frozen `empty_prototype` buffer
  (seeded, unit-norm, init separation check/reseed); `target_latent(ids,is_empty)`;
  `occupancy_head`+`count_head` (per-horizon ModuleLists in horizon mode);
  `predict_occupancy_from_latent`; architecture_metadata + strict checkpoint round-trip.
- **v0B train (SITE 2)** `arms/v0b/train_minimal_jepa.py`: helper-routed `_read_examples`
  (drop the `len==0→skip` + the `t0=max(0,-1)` misread); carry per-horizon
  `(ids, is_empty, count)`; loss = cosine(pred, target_latent) over all rows (empties→
  z_empty) + λ_occ·BCE + λ_count·Huber[masked] + var-reg on **non-empty** preds only;
  balanced sampler; collapse diagnostics.
- **SITES 3–8** (v0e / extract_flatascend+`target_len=0` / rollout exports / v0a
  predictor / scan_feasibility `target_len=0` / v0d count): route through the helper;
  encode-empty where the arm has a latent path, else skip-with-count + reported rate;
  never `arr[0:]`.
- **Rung-1 decoder D**: input `[z+, occ_logit, count_pred]`; binary empty + count-0
  head; decode rule = if empty predicted, emit "empty/count 0", skip token/Δt recon;
  empty precision/recall (floor ≥0.95) as a live collapse falsifier; exact-order/KS-D
  on non-empty cells only.
- **retrieval.py**: length-0 / occupancy-empty bin; candidate normalization gains an
  occupancy-class dim (source+horizon+length+rate+occupancy); separate empty-class R@k.
- **Wall-clock readiness + composite driver**: per source×horizon feasible/non-empty/
  saturated/empty/valid-matched + `empty_target_rate`; feed the composite rung −1
  driver so an empty-saturated horizon is flagged conditional/incomplete (Pi item 3).
- **Comparator (frozen 350M)**: occupancy probe on the frozen context state (fair);
  token-generation metrics EXCLUDE empty with a reported per source×horizon coverage
  denominator (Pi Q6). The z_empty latent claim is v0B-only, reported as such.

## Key risks
- Latent-side safety rests on two test-guarded invariants (buffer-not-Parameter,
  targets-in-`no_grad`); regress either → pred+z_empty co-collapse the var-floor won't
  catch. Mitigated by asserts + the empty-vs-populated margin alarm.
- Any un-migrated site reintroduces `arr[0:]` undetectably → single-span-reader test.
- High empty-rate gaming (SCID ~38% at W=1d) → empty-fraction cap + beats-marginal gate.
- Empty-vs-low-activity proximity: 1-event futures sit near z_empty's direction; the
  occupancy head is the authoritative boundary (latent region advisory).
- **Censored ≠ silence**: a single z_empty conflates remission vs discharged/censored
  silence; MIMIC admission-boundary censoring must be excluded UPSTREAM before empty
  encoding (precondition, not z_empty's job).
- Added surface: 2 heads + 3 loss terms + a load-bearing beats-marginal gate + sampler.

## Recommended sequencing
1. Shared helper + single-span-reader test scaffold. **(mechanism-independent)**
2. SITE-1 leakage boundary bug fix + regression test. **(mechanism-independent)**
3. v0B model: frozen prototype + occupancy/count heads + target_latent + tests.
4. v0B train loop (SITE 2): helper-routed examples + hybrid loss + sampler + diagnostics.
5. Migrate SITES 3–8 through the helper (+ per-site mixed-block fixtures).
6. SITE 4 embedding cache (target_len=0, always-emit context state) + 350M probe.
7. retrieval (occupancy-empty bin + normalization).
8. Rung-1 decoder empty/count-0 head + live falsifier metric.
9. Wall-clock readiness variant + composite-driver hook.
10. End-to-end wall-clock leakage audit (empty-heavy) + high-empty stress test.
11. **Pi blueprint gate (this doc) BEFORE the within-source wall-clock rung 0.**

Steps 1–2 are mechanism-independent (pure `-1`-misread correctness) and can proceed
now; steps 3+ (the collapse-sensitive latent machinery) wait on Pi's gate.

## Open questions for Pi (blueprint gate)
1. Approve the HYBRID vs pure-scalar P3 — is silence-as-a-genuine-latent (frozen
   z_empty) required for rung 0/1, or is the supervised occupancy scalar sufficient?
   (Hybrid re-admits empties into the cosine branch as z_empty targets, stop-grad + out
   of var-reg — acceptable?)
2. z_empty as a **frozen** buffer (immovable, no separation reg) vs learnable-with-
   separation-reg; approve the frozen default + an init separation check that reseeds
   if |cos(z_empty, populated centroid)| > ~0.15 at dim=128.
3. Pre-register the empty-class gates: empty recall ≥0.95, empty-vs-populated margin
   ≥0.15, occupancy beats-marginal AUC/Brier per cell, and an **empty-vs-1-event**
   separation metric (not just aggregate recall). Confirm thresholds.
4. Comparator fairness: is an external occupancy probe on the frozen 350M context state
   an acceptable empty-class comparison, or exclude empty from ALL 350M metrics?
   (Token-generation excluded + reported coverage denominator — satisfies Q6?)
5. Balanced sampling / empty-fraction cap per batch — approve; does capping bias the
   learned base-rate (requiring an explicit calibration/recalibration report)?
6. Horizon semantics: pin `horizon_count=1` for the first wall-clock rung; use
   per-horizon occupancy heads for multi-horizon (partial-empty native); defer
   recursive-empty ("once empty always empty" is wrong for wall-clock gaps). Approve?
7. **Censored ≠ silence**: confirm MIMIC admission-boundary censoring must exclude
   censored windows UPSTREAM before empty encoding (precondition, not z_empty's job).
