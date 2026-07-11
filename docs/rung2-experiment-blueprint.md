---
title: Rung 2 — SEPARATED blueprint (rollout diagnosis · count interface · order target · continuous-time head)
created: 2026-07-11
status: DRAFT — design panel (2 proposers) + Cog preflight + fable5 falsifier pass folded in (7 blocking changes); T4 learned-target BARRED from dev; pre-registration pending Pi Rung-2 blueprint gate
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); per-source encode-empty v0B latent (frozen, reused from Rung 0/1)
reporting: aggregate-only; DEV-ONLY (inside the encoder's circle, Pi Q8); TEST sealed (no --confirm-test path)
scope: 4 INDEPENDENT sub-gates, independent manifests, no cross-rescue (Pi ruling). Nomination ≠ adoption.
---

# Rung 2 — separated blueprint

Pi ruled that Rung 2 must **not** bundle a rollout diagnostic, a count-interface choice, an
order-target, and a continuous-time head into one ungated run. This blueprint separates them
into **four independent sub-gates**, each with its own pre-registration, gate, and manifest;
**no sub-gate rescues another**. The governing principle throughout: judge **prediction-achieved**
fidelity — a predictor *trained on context to reach a target*, tested — not the target's decode
ceiling (the Rung-1 nominations were ceilings only). Everything is DEV-only inside the encoder's
circle; **on real dev the ceiling of any decision is `NOMINATE`, never `ADOPT`** — a generation
claim is certified only against the semi-synthetic oracle (Rung-5 spec).

**Shared notation (sub-gates 3/4):** `D(z⁺)` = TARGET-CEILING (decode a property from the *true*
target latent — what Rung 1 measured); `D(ẑ)` = PREDICTION-ACHIEVED (decode from the *predictor's*
output `ẑ = g(context)`, **sampled, never the conditional mean**); `Δ_split = D(z⁺) − D(ẑ)` = the
attribution — high `D(z⁺)` + large `Δ_split` ⇒ `PREDICTOR_BOTTLENECK` (go distributional); low
`D(z⁺)` ⇒ `TARGET_BOTTLENECK` (change the target).

## fable5 falsifier pass — verdict + folded-in changes
An adversarial fable5 pass (its own "circle" ladder framing) ruled: **Rung 2 cannot run sub-gate 3
as drafted; it CAN run sub-gates 1, 2, 4 and the *frozen-target* part of 3 at `NOMINATE`/diagnostic
ceiling, conditional on the blocking fixes below.** The systemic error it caught: escaping the
*decoder* prior is not escaping the *content/rate* prior that lives in the encoder — and for order
and timing the content/rate prior is exactly the dominant confound. Folded-in **BLOCKING** changes:

1. **T4 (learned VQ target) is `NOT_EVALUABLE` for dev nomination — one rung below NOMINATE —
   until the semi-synthetic oracle exists.** A co-trained target-encoder + codebook + decoder
   *self-manufactures* its own `D(z⁺)` ceiling (encoder writes order into codes → decoder reads it,
   jointly optimised); no dev-side control separates "encoder wrote order" from "order is
   conditionally present." T4 may be built + unit-tested, but its dev order numbers do not nominate.
2. **Content-matched / rate-matched wrong-instance swaps everywhere.** A random wrong instance
   certifies only "reads content/rate," not "reads order/timing." Swaps must hold the multiset
   (order) / occupancy+rate (timing) fixed. Highest-leverage single fix (sub-gates 3 & 4).
3. **Precedence-frequency is NOT prior-resistant** — its marginal form is *maximised* by a
   generative prior (Rung 1 already showed the residual order signal is a content→order prior).
   Reformulate as **per-instance skill over a content-matched precedence prior** (baseline sees only
   the multiset); the raw agreement is demoted to a necessary sanity check, not a trusted floor.
4. **Stratified marginal for timing CRPS-skill** — condition the marginal on (occupancy/length bin,
   base-rate quantile, horizon) + add an explicit **rate-only baseline head** to beat; 4A keyed on
   **multiplicity swap-excess** (p₀ ≈ marginal zero-fraction is a freebie). Blocks context-length→rate
   / occupancy→zero-fraction leakage.
5. **Property-specific corruption for the ENTRY sensitivity curve** — corrupting `z⁺` corrupts
   *content* too, so a content-prior-only order score still degrades. Order needs **permute-at-fixed-
   multiset** corruption; timing is exempt (Δt never encoded).
6. **Bandwidth-fair Δ_split** — add a frozen-E seq-of-latents **quantised to T4's bit budget** as the
   matched-bandwidth ceiling; decompose `1 = representation_loss + bandwidth_loss + predictor_loss`;
   Δ_split is intra-candidate/intra-bandwidth only (cross-target T2-vs-T4 ranking is void).
7. **Pre-register decoding temperature + report a sweep** (marginal metrics peak at high T,
   per-instance at low T; selecting T on dev leaks).

Nice-to-have (correctness): demote `ρ_t` to descriptive and promote `d_t^self`-vs-`d_t^NN` + the
exposure gap to load-bearing (normalise `d_t^NN` by the ambient true-true NN distance); calibrate
`PERT_EPS` with an ε-sweep + report the Jacobian top-singular-value; relabel sub-gate 2 as an
**interface-calibration comparison** (A wins the point-accuracy race near by construction — the real
deliverables are the sufficiency-vs-predictability gap and the calibration asymmetry).

**Revised honest stop line:** *dev can NOMINATE directions from FROZEN targets and never-encoded
quantities, measured against content/rate-matched priors; it can neither ADOPT anything nor evaluate
a co-trained (learned) target — the learned VQ codebook and every counterfactual/order-certification
claim wait for the semi-synthetic oracle.* On dev, the ceiling of a learned-target decision is
`NOT_EVALUABLE`; of any other decision, `NOMINATE`.

## Sequencing & entry precondition
Sub-gates run in order 1 → 2 → 3 → 4 but gate independently. Sub-gates 3/4 (which **train** a
target/codebook/head) are additionally guarded by an **ENTRY GATE pulled forward from the Rung-3
falsifier**: the candidate decoder's **latent-corruption sensitivity curve must be non-flat**
(graded z⁺ corruption must monotonically degrade `D(z⁺)`); a flat curve ⇒ the decoder reconstructs
from its generative prior, not the latent ⇒ that candidate is `NOT_EVALUABLE` (no rescue). This is
the load-bearing defence for the deepened circle (learned target-encoder + codebook + decoder are
three co-adapted learned components that can *jointly* mask latent failure).

Inherited from the Rung-1 harness (`rung1_contract` spine): cluster (patient-disjoint) bootstrap
`N_BOOT=2000/SEED=20260523`; deterministic patient-disjoint derangement swap; `matched_head_hidden`
budget; worst-primary-cell conjunction; MIMIC 2 d sensitivity-only; cluster floors; `config_hash` +
`evaluator_provenance`; aggregate-only manifests. Cells: SCID {30,90,365,730} d; MIMIC {0.25,0.5,1} d
(+2 d sensitivity).

---

## Sub-gate 1 — incumbent NO-TRAINING d_t / v_t rollout diagnosis
**Object:** the existing recursive latent rollout of the frozen v0B predictor
(`predict_rollout_from_latent`, `recursive` vs `horizon_conditioned`). **Nothing new is trained.**
This is a *diagnostic*, not a programme go/no-go — failure selects a stabiliser, never "dead".

**Data path:** extend `export_mean_token_rollouts.py` — `--unit wall_clock` (rollout step *h* ↔ the
frozen wall-clock horizon block `[t_query, t_query+W_h)`, so the 4 SCID / 3 MIMIC horizons **are**
the multi-step ladder), `--autoregression-mode {recursive,horizon_conditioned}`, a **teacher-forced**
export (roll from the true previous-horizon latent), and a seeded **perturbation-ensemble** export.

**Metrics (cosine, per source×horizon cell, cluster-bootstrap CIs):**
- **Drift** `d_t^NN = 1−cos(ẑ_t, nearest patient-disjoint TRUE latent)` (off-manifold drift);
  `d_t^self = 1−cos(ẑ_t, z_t^true)` (drift from own truth); **exposure gap** `g_t = d_t^{self,free} −
  d_t^{self,tf}` (free-running − teacher-forced → isolates error compounding).
- **Velocity/collapse (v0B predictor is DETERMINISTIC → no sampling ensemble):**
  `v_t^pop` = dispersion (trace-cov / effective-rank) of the predicted cloud; **collapse ratio
  `ρ_t = v_t^pop / V_t^true`** — the **load-bearing control**: `ρ_t→0` = genuine attractor collapse;
  `ρ_t≈const` with both clouds shrinking = **genuine long-horizon unimodality, NOT a defect**;
  `v_t^pert` = perturbation-ensemble spread (finite-difference local-contraction / Jacobian-norm).
  **Attractor identity:** `cos(ẑ_t, z̄_global)`, `cos(ẑ_t, z_empty)` name which fixed point a
  collapse converges to.
- Slopes vs step / vs log W_h with paired CIs.

**Load-bearing discriminator (fable5 correctness fix):** the frame-robust collapse signal is
**`d_t^self → d_t^NN`** (own truth no better than the nearest wrong instance ⇒ predictions stopped
being conditional) + the exposure gap `g_t`; **`ρ_t` is DESCRIPTIVE only** — a deterministic
mean-regressor has `ρ_t<1` at step 1 by Jensen (not a rollout pathology), and `ρ_t` cannot separate
"truth widens, prediction can't" from real collapse. Normalise `d_t^NN` by the ambient true-true NN
distance (collapse compresses all distances). `PERT_EPS` calibrated by an ε-sweep; report the
Jacobian top-singular-value + along-trajectory product, not a single-step norm.

**Signatures (diagnostic, per cell):** HEALTHY · DRIFT_DOMINANT (exposure gap widens, `d_t^NN`
climbs off-manifold) · COLLAPSE_DOMINANT (`d_t^self`→`d_t^NN`, `ẑ_t→z̄/z_empty`) · GENUINE_UNIMODALITY
(control — do **not** misread honest mean-regression as collapse).
`EMA_NONSTATIONARITY` leg is **DEFERRED/NOT_EVALUABLE** (single frozen checkpoint; no live EMA) —
stated explicitly, not inferred. Frozen constants: `DRIFT_SLOPE_TAU=0.02`, `COLLAPSE_RATIO_RHO_STAR=0.5`,
`PERT_EPS`, `PERT_ENSEMBLE_N`.

**Baselines:** teacher-forced vs free-running; recursive vs horizon_conditioned (isolates recursive
compounding); frozen `z_empty` / global mean; K=1 harness null. **DEV-only/circle:** predicted and
true latents share one frame → self-referential; mandatory caveat "diagnoses rollout dynamics on
dev only, not external validation". **Compute:** no training; forward passes + numpy + bootstraps ≪1
GPU-hr.

---

## Sub-gate 2 — count interface choice (predictor vs target channel)
**Two different objects (Pi):** **A** = factorized CONTEXT head (`predict_occupancy_from_latent` /
`count_head`: predict future count *from context*); **B** = concatenated TARGET scalar
(`[mean⊕log1pN]`, the Rung-1 nomination — count as a target dimension). **The honesty pivot:**
Rung-1's exact-count-from-`z⁺` = 1.000 is **target-side representational sufficiency** (the identity —
count concatenated then read back), **not prediction**. Adoption needs **prediction-achieved**: can a
predictor trained on **context only** reach the count?

**Protocol:** freeze the v0B context representation; train **only** the count read-out, matched
budget, TRAIN→DEV-once, C1 leak-free (invariance-tested). Score **both A and B strictly as
context→future-count predictors**; the 1.000 sufficiency number is a labelled
`information_scope=target_representation_readout` line that **never drives the decision**.
**Metrics:** exact-count-from-context (gate 0.80) + floor-adjusted excess (>0.10) vs modal / context-
length rate-prior / patient-disjoint NN-copy; **calibration** (occupancy AUC/Brier + count PIT) — A
is a proper hurdle distribution, B a point estimate in cosine space with no native uncertainty
(pre-registered to favour A on ties). **Decision (`ADOPT_MARGIN_DELTA=0.02`):** `ADOPT_CONCAT(B)` iff
B beats A by the paired margin **and** clears gate+excess; `ADOPT_FACTORIZED(A)` on tie/loss
(default — supplies calibrated uncertainty, keeps count out of the cosine target geometry) ⇒ the
Rung-1 count_concat nomination is **DECLINED-for-adoption** (sufficiency ≠ predictive superiority);
`NEITHER_ADEQUATE` iff both fail from context (context→count predictability is the binding
constraint). **fable5 reframe:** A wins the point-accuracy race near by construction (A has a proper
count loss; B's count is an unoptimised cosine side-channel) — so give B a **fair cosine+count
multitask predictor**, and read the gate primarily as an **interface-calibration comparison** (the
real deliverables are the sufficiency-vs-predictability gap and A's calibrated occupancy/count
distribution vs B's uncertainty-free point estimate), not a horse-race. **Compute:** two tiny matched
heads/cell ≪1 GPU-hr.

---

## Sub-gate 3 — order-preserving target + prediction-achieved order gate
Rung-1 forces a **non-pooling** candidate (pooled targets have no order ceiling) and
**density-stratified** metrics (report by occupancy bin — the decay is the signal). Candidate ladder
(bandwidth priced in **bits** = `L·log₂K` or `dim·32`):

| ID | target z⁺ | learned surface | role |
|----|-----------|-----------------|------|
| **T0** mean_embed | pooled | none | permutation-invariant floor (order = content-prior) |
| **T1** pooled-ordinal (param-free ρ rank code + outer-product moment) | predictor+decoder | cheap-order **null control** to beat |
| **T2** seq-of-latents (frozen-E ordered stack) | predictor+decoder | **CEILING anchor** — `D(z⁺)≈1` ⇒ any `D(ẑ)` shortfall is 100% predictor-side |
| **T3** ordinal-tagged seq-of-latents | predictor+decoder | order-robust seq target |
| **T4** VQ ordered codes (TRAINED codebook, EMA target encoder) | target-encoder+codebook+decoder+predictor | **BUILT + unit-tested only; `NOT_EVALUABLE` for dev nomination until the oracle** (self-manufactured ceiling — fable5 #1). The Rung-2 dev order nomination comes from T1–T3 (frozen targets). |

**Training (only T4 trains a target):** two-phase — Phase A EMA/stop-grad learn `f_ξ` on the future
block + freeze codebook; Phase B train predictor + decoders against the frozen target. **Bandwidth
pricing:** every candidate reports `latent_bits`; a T4 win must beat (i) mean_embed quantised to the
same bits and (ii) a **frozen-random-codebook** control at matched bits — else the "win" is
bandwidth, not learning. **Codebook adequacy:** perplexity / dead-code fraction / usage entropy; a
collapsed codebook ⇒ `NOT_EVALUABLE`. **Predictor:** deterministic-mean (blurs under multimodal
order) **vs** distributional/sampled (categorical AR/parallel over codes; **all order metrics on the
sampled ẑ**).

**Metrics (tiered by circle-exposure; the decoder-free ones are the trusted channels):**
1. **Per-instance precedence SKILL over a content-matched prior** (fable5 #3) — the achieved P(a≺b)
   must beat a baseline that sees only the instance's *multiset* and emits its typical ordering,
   cluster-bootstrapped CI-positive. (Raw precedence *agreement* is maximised by a generative prior,
   so it is a necessary sanity check only, NOT a trusted floor.)
2. **Code / latent-sequence edit distance** (readout-level, not fully external): normalised
   Levenshtein — corroborative.
3. **Token-decoded exact-order + tie-aware Kendall-τ** (interpretable but **most maskable** →
   subordinate; reuse `exact_order_hits`, `tie_aware_exact_order_hits`, `kendall_tau_tie_aware`).

**Gate (conjunctive, worst-cell):** G3.1 achieved order-skill lower-CI ≥0.10 over the content-prior
AND a **content-matched (fixed-multiset) predictor-side swap** (fable5 #2); G3.2 beats T0 by a
multiplicity-corrected margin; G3.3 the **content-matched precedence skill must also be CI-positive**
(else `PRIOR_MASKED`, void); G3.4 report the **3-way bandwidth-fair decomposition**
`representation/bandwidth/predictor` at matched bits (fable5 #6) — Δ_split intra-candidate only;
G3.5 the ENTRY curve uses **permute-at-fixed-multiset** order corruption (fable5 #5). Decoding
temperature pre-registered + swept (fable5 #7). Outcome on dev (T1–T3 only): `NOMINATE_DIRECTION` or
(expected, given Rung 1) `ESCALATE_REDESIGN`; **never ADOPT**; **T4 `NOT_EVALUABLE` until the oracle**. **Compute:** ~few GPU-hours (small `f_ξ`
K≈256–1024/L≈8–16, small predictor, matched decoders).

---

## Sub-gate 4 — continuous-time head + zero/simultaneity-aware calibration gate
Rung-1 forces: Δt is a **point mass** (~70% zeros SCID / ~98% MIMIC = simultaneous events); TAP
clears both timing gates at **SCID-30 d only**. **Structural advantage:** raw Δt is a
**never-encoded raw quantity** → CRPS-skill/KS on Δt is **decoder-free and external by
construction** — the *least* circular gate in the programme.

**Head (marked-TPP on the latent):** occupancy/timeline sub-model (how many distinct timestamps) +
**simultaneity-multiplicity** sub-model (co-occurrence cluster size) + **hurdle** `p₀=P(Δt=0|context)`
(reuse `train_hurdle_timing_head`, now context-conditioned) + positive-tail conditional density
(monotone-quantile pinball or log-normal/Weibull mixture; time-rescaling randomized-PIT KS).

**Two SEPARATE gates (non-compensatory — the point mass must not mask absent tail signal):**
- **4A zero/simultaneity:** conditional `p₀` reliability + multiplicity calibration (its swap-excess
  isolates *which instances cluster*, beyond the easy marginal zero-fraction).
- **4B positive-tail:** on Δt>0, zero-aware randomized-PIT **KS-D upper-CI ≤0.05 AND CRPS-skill
  lower-CI ≥0.05 over a STRATIFIED marginal** (conditioned on occupancy/length bin × base-rate
  quantile × horizon — NOT the global marginal) **AND** beats an explicit **rate-only baseline head**
  (given only scalar rate/occupancy/horizon), CI-positive, under a **rate/occupancy-matched swap**
  (fable5 #4). Blocks context-length→rate and occupancy→zero-fraction leakage; 4A keyed on the
  multiplicity swap-excess (not p₀ reliability, which the marginal zero-fraction gives free).

Overall = **4A ∧ 4B**, worst-cell. **D(z⁺) vs D(ẑ) + the SCID-30 question:** (Q1) does the trained
head clear 4A∧4B **prediction-achieved** at SCID-30 (ceiling reachable by a predictor)? (Q2) does it
**extend** the reachable set beyond TAP's one cell? Only-SCID-30 ⇒ scoped
`NOMINATE_DIRECTION(short_horizon_ct_head)`; all-cell ⇒ broad `NOMINATE`; flat ⇒
`RETAIN_INCUMBENT`/`ESCALATE`. **Adequacy:** `TIMING_CLUSTER_FLOOR=500` + `TIMING_INTERVAL_FLOOR=1000`
+ the precision sim; report zero-fraction and positive-interval counts separately. **Compute:** ≪1
GPU-hr incremental (reuses Rung-1 hurdle machinery; shares the context predictor with sub-gate 3).

---

## Anti-circularity (the falsifier, sub-gates 3/4) — where the fable5 pass concentrates
1. **Latent-corruption sensitivity curve** (ENTRY GATE) — flat ⇒ NOT_EVALUABLE.
2. **`D(z⁺)` vs `D(ẑ)` split** — the T2 anchor (`D(z⁺)_order≈1`) makes predictor-vs-target loss
   identifiable.
3. **Content/rate-prior-resistant lower bounds** — order: per-instance precedence *skill over a
   content-matched prior* (raw agreement is prior-maximised, NOT trusted); timing: CRPS-skill over a
   *stratified* marginal + a rate-only baseline. These escape the decoder prior but only partly the
   encoder's content/rate prior — labelled honestly.
4. **Content-matched / rate-matched wrong-instance swap on the PREDICTOR** (fable5 #2) — hold the
   multiset (order) / occupancy+rate (timing) fixed, else the swap certifies only "reads content/rate."
5. **Non-circular anchor for ADOPTION = the semi-synthetic oracle** (Rung-5 spec) with a known
   generative order/intensity rule — order nominates on dev, certifies only against the oracle; raw
   Δt is external enough to go further on dev but a counterfactual timing claim still needs the
   oracle.

## Module / CLI layout (all DEV-only, no --confirm-test)
`clinical_jepa/eval/rung2_contract.py` (re-exports Rung-1 constants + Rung-2 constants/labels) ·
`export_mean_token_rollouts.py` (EXTEND: wall-clock step, teacher-forced, perturbation ensemble) ·
`rung2_rollout_diag.py` (sub-gate 1) · `rung2_count_interface.py` (sub-gate 2) ·
`clinical_jepa/targets/order_targets.py` + `clinical_jepa/arms/rung2/train_target_encoder.py` +
`train_predictor.py` + `rung2_order_probes.py` + `rung2_order_gate.py` (sub-gate 3) ·
`rung2_ct_head.py` + `rung2_timing_probes.py` + `rung2_timing_gate.py` (sub-gate 4) · per-sub-gate
`rung2_verdict.py` blocks. Configs under `configs/rung2/`; governed sidecars under
`run-workspace/local-governed/rung2/` (gitignored).

## Cog imports
- `[[synthesis/orca-and-jepa-representation-space-translation]]` (I-JEPA: predict the *latent* of the
  future block, not raw tokens) → T4 (contextualised VQ codes) is the adoption target; T2/T3
  frozen-E per-event targets are demoted to *ceiling anchors* (raw-token-in-latent-clothing).
- Default *separate generator/executor from verifier* → decoder-free precedence-frequency (order)
  and raw-Δt CRPS-skill (timing) are the verifier channels the learned decoder cannot mask.
- `[[synthesis/orca-external-prior-specification-tests]]` → raw Δt flagged as the one genuinely
  external quantity (never embedded) → sub-gate 4 least circular; order has no external real-data
  anchor short of the oracle.
- **Not followed:** `[[concepts/target-trial-emulation]]` (no action/counterfactual contrast here).

## Open questions routed to the fable5 falsifier pass + Pi gate
1. Is the T2 ceiling anchor (`D(z⁺)_order≈1`) an honest predictor-bottleneck isolator, or does its
   `L·D` bandwidth make the split incomparable to T4's bit-budget?
2. Is precedence-frequency agreement a strong enough decoder-free order anchor to **nominate on
   dev**, or is *any* real-data order metric too inside the circle (⇒ order can't nominate without
   the oracle)?
3. Should 4A (zero/simultaneity) be gated separately from 4B (tail), or is a single joint zero-aware
   proper score less gameable?
4. Must the semi-synthetic oracle exist **before** any learned-target (T4) run (no learned target on
   dev at all until the oracle is built)?

**Scout trigger:** SUGGEST a sanitized Cog scout on *"bandwidth-and-circle-depth-priced order-
preserving JEPA targets (VQ ordered codes vs param-free seq-of-latents) and marked-TPP heads with a
dominant Δt=0 point mass; decoder-free vs decoder-exposed prediction-achieved attribution"* — do not
launch without authorisation.
