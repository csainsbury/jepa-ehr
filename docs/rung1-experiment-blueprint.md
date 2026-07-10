---
title: Rung 1 — frozen-decode ceiling BLUEPRINT (target-representation decision gate)
created: 2026-07-10
status: designed (3-proposal panel + synthesis + self-critique); routed to Pi R7 blueprint gate
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); latent = fresh v0B mean-token JEPA
reporting: aggregate-only (per source×horizon-cell metrics, CIs, gate verdicts; no seq ids / tokens / embeddings / real paths)
scope: DEV-ONLY architecture-selection ceiling (inside the encoder's circle; Pi Q8) — never an external/generation claim
---

# Rung 1 — frozen-decode ceiling

**Question (Pi Q3 hand-off):** does the pooled target latent `z⁺` retain fine event /
count / order / timing information *at all* — i.e. what is the **decode ceiling** that
upper-bounds any generator built on this latent? This rung answers it and, by a
pre-registered parameter-free **target contrast**, converts the (structurally expected)
order/timing negative into an **actionable Rung-2 target choice**.

## 1. Structural priors — what `z⁺` can and cannot contain (the spine)

`z⁺ = target_latent(ids) = mean_embed(ids) = (1/N)·Σᵢ E[tᵢ] = Eᵀp`, where `E ∈ ℝ^{V×D}`
is the frozen embedding table (V=1050, padding_idx=0 excluded), `c ∈ ℤ^V` the multiset
count vector, `N = Σc`, `p = c/N` on the simplex. This closed form fixes each property's
a-priori ceiling **before any number is measured**:

| Property | Structural status in `z⁺` | A-priori decoder-free ceiling |
|---|---|---|
| (1) token-set / multiset marginals | `z⁺` **is** `Eᵀp` → `p` is the *only* thing it encodes | **HIGH** — recoverable up to `col-space(E)` + row near-degeneracy; analytically invertible |
| (2) count / cardinality `N` | divided out by `1/N`; survives only via `‖z⁺‖` / concentration ↔ `N` | **LOW–MODERATE, genuinely uncertain** → a real test of the 0.80 gate |
| (3) order | `z⁺` is **provably permutation-invariant** (no positional term) | **= chance, provable** — any decoder-free order probe on `z⁺` is constant under permutation |
| (4) Δt inter-event timing | `cumulative_days` / `time_deltas` are **never embedded** into `z⁺` | **= token-identity rate-proxy channel only** (DURATION/vigintile anchors); else 0 |

**Order (3) and timing (4) are structurally near-foregone fails.** Measuring the exact
ceiling is still high-value: it (a) empirically confirms the structural claim on real data
and would expose any *leak* (order/timing baked into token identity) as a first-order
finding; (b) quantifies the generative-prior confound `decoder − decoder-free`; (c) sets
the hard upper bound on any `z⁺`-only generator and **drives the Rung-2 target decision**.

## 2. The circle — why decoder-free probes are authoritative

`D(z⁺)` is inside the encoder's own latent → a **DEV-ONLY ceiling, never external
validation** (identical status to Rung-0, Pi Q8). The load-bearing hazard (design-doc
R2.2): a co-trained decoder can **mask latent failure via its generative prior** — emit
marginal order / modal count / a marginal Δt distribution and clear the gate while carrying
zero per-instance latent signal. Therefore the whole rung is built on a **floor + excess**
principle:

- **M0 — prior floor:** decode with *no* usable latent (cell marginal / wrong-instance
  swap / interpolate-`z⁺`-to-cell-mean). This is the "reads-its-prior" score.
- **M1 — decoder-free lower bound:** the *simplest* readout from `z⁺` (analytic `(Eᵀ)⁺` /
  NNLS; closed-form ridge; multinomial-logistic). Weakest possible prior.
- **M2 — trained-decoder upper bound:** an expressive read-out head `D(z⁺)` — the headline,
  but **prior-confounded**.

**Gate rule (central):** a property is *genuinely decodable from `z⁺`* only if the
**latent-attributable excess `M1 − M0`** clears the frozen gate with a patient-bootstrap CI.
If `M2` clears but `M1 − M0` does not, the surplus is the decoder's prior → the cell is
recorded **`PRIOR-MASKED`**, not decodable. All three (M0, M1, M2) reported per cell.

## 3. The target-representation contrast (the actionable reframe) — *addition to the frozen spec, for Pi to rule*

A bare "`mean_embed` fails order/timing" is structurally foregone and tells Rung 2 nothing:
it cannot separate "unrecoverable from *any* cheap pooled target ⇒ abandon pooling" from
"trivially recoverable in a slightly richer pooled target ⇒ cheap fix". So we run the
ceiling as a **controlled contrast**: hold the decode probe, decoder capacity/budget, splits,
candidate normalisation, and source×horizon cells **identical**, and vary **only** the
target-construction function `g:(ids, cumulative_days, t_query, W, is_empty) → z⁺`.

Define per-event normalized within-window time `τᵢ=(tᵢ−t_query)/W ∈ [0,1)`, inter-event
`Δtᵢ`, ordinal rank `rᵢ`. `φ,ψ` are **fixed** (untrained) sinusoidal feature maps of small
width `d_time ∈ {4,8}`. All arms are **parameter-free deterministic functions of the same
target span** (via `block_spans.read_target_span`) using the **same frozen `E`** — so there
is no retraining, no new encoder, and **zero retrofit surface**. Silence routes to the frozen
`z_empty` (whole-window for A/B, per-slot for D).

| Arm | Construction | Order | Timing | Dim | Role |
|---|---|---|---|---|---|
| **A. `mean_embed`** (incumbent) | `(1/N)Σ E(idᵢ)` | none | none | D | the reference ceiling (Pi's frozen object) |
| **B1. TAP-concat** | `[mean E ⊕ mean φ(τᵢ,logΔtᵢ)]` | none | **event-time distribution / rate** | D+d_time | primary **timing** rescue |
| **D-sub. SetK** | M equal-**wall-clock** slots (`subwindow_blocks.carve_subwindows`), `[m₁⊕…⊕m_M]`, empty slot→`z_empty` | **coarse** | **coarse rate profile** | M·D | primary **structural** rescue = the Rung-2 seq-of-latents candidate |

**Core Rung-1 arms = {A, B1, D-sub}** — the minimal set giving ≥1 timing-preserving and ≥1
order-preserving alternative under identical probes. **Deferred to Rung-1b** (needs training
/ retrofit management): PosTag (ordinal counterpart), TAP-moment (type×time cross term),
`[mean ⊕ log1p N]` count control, fixed-k-means VQ, learned slot-attention / learned VQ. A
deferred learned arm, if ever included, is trained **train-split only on a probe-blind
objective** (commitment/EMA-kmeans — never order/count/KS) and frozen before any decode.

**Alt-arms are ceiling-only.** The JEPA predictor was trained to predict `mean_embed`, so the
predictor-achieved `D(ẑ)` and the `D(z⁺) vs D(ẑ)` split exist **only for arm A**. B1/D-sub
report the **ceiling `D(z⁺)` only** (decode from the *true* alternative target); the predictor
gap for a winning alternative becomes a Rung-2 concern. State this explicitly in every
alt-arm result.

## 4. Per-property probes, frozen gates, baselines

Probes computed **per source×horizon cell**, on **non-empty** windows (empties owned by the
existing falsifier, §5); order/Δt require **≥2 events**. Decoder-free (M1) is primary; the
trained head (M2) is the prior-confounded upper bound.

| Property | Decoder-free probe (M1) | Frozen gate (metric) | Hard baselines (M0 / floor) | Expectation (arm A) |
|---|---|---|---|---|
| (1) marginals | analytic `(Eᵀ)⁺` + simplex-NNLS; ridge/logistic `z⁺→p` | **descriptive** (named summary-stat cross-check, not a programme gate): macro-F1, set-Jaccard, cosine/TV vs `max(chance, NN)` | cell-marginal freq; matched-NN | **PASS** (high) |
| (2) count | ridge/GLM `z⁺→N` and `→log1p N` (incl. `‖z⁺‖`) | **exact-count ≥ 0.80 per cell** on N≥1 **+ ≥0.10 margin over max(chance,NN)** | modal/mean count; context-length rate-prior; NN-copy count | uncertain — real test |
| (3) order | pairwise-order concordance (= norm. Kendall-τ), **conditioned on true multiset** | **exact ordered-seq recon ≥ 0.70 per cell** (τ reported as the graded ceiling number) | prevalence-matched perm; **content-prior order** (bigram/precedence from the multiset); canonical-sort | **FAIL** (provable ≈ chance) |
| (4) timing | quantile regression `z⁺→` per-window inter-event `Δt` deciles | **KS-D ≤ 0.05 per cell (PIT time-rescaling)** + calibration/coverage, **gate on the upper-95%-CI of KS-D**, **and conditional must beat marginal** | cell-marginal Δt; NN Δt-quantiles | **FAIL** (provable) |

- **Timing mechanics (two independent neutralizers, both required):** (i) **per-window PIT
  time-rescaling** — rescale each observed `Δt` by the probe's per-window `F̂_w`, pool PIT
  values *within the cell*, KS-D vs Uniform/Exp(1); a disguised marginal fails PIT whenever
  windows differ in rate → this *is* the mechanical "no pooled-only rescue". (ii)
  **conditional-vs-marginal skill Δ** — CRPS/NLL reduction of the true `Δt` under the
  `z⁺`-conditional probe vs a marginal sampler; **timing is claimed only if Δ lower-CI > 0.**
  KS-D≈0 with Δ≈0 ⇒ "marginal reproduction only", **not** timing fidelity.
- **KS-D honesty:** gate on the **upper 95% CI** of the per-cell KS-D (patient bootstrap), not
  the point estimate — the correct skeptic-favouring inversion of "not α-rejection" (small n ⇒
  wide CI ⇒ cannot certify ⇒ does not pass). Never pooled.

## 5. Guard battery (floor+excess), ranked; non-compensatory flagged

**Tier-1 (must-pass; a failure voids the ceiling claim):**
1. **Latent-dependence / corruption floor (G1 ≡ M0).** Interpolate `z(λ)=(1−λ)z⁺+λ·z̄_cell`,
   `λ∈{0,.25,.5,.75,1}`, **+ wrong-instance swap** (decode `z⁺` paired with a shuffled
   within-cell target). Recon must **degrade monotonically** toward the λ=1/swap floor
   (isotonic violation-fraction ≤ 0.10; Spearman ρ ≤ −0.9 for order/count). A **flat** curve
   ⇒ the decoder reads its prior, not `z⁺`. The λ=1/swap point **is M0**.
2. **Floor pair (G6 baselines × G2 attribution).** Every gate is met by `M1 − M0` (or, for
   the headline, `M2` beating **max over the hard-baseline set** with non-overlapping CI),
   never by the raw score. Order must beat **content-prior order** (else "0.70 reflects a
   content→order prior, not latent order — expected"); count must beat **modal-count** and
   the **context-length rate-prior**.
3. **Memorisation / NN copy-floor (Cog integrity gate).** NN-in-latent (cosine to nearest
   **matched** train `z⁺`); shingle-3gram/LCS overlap of `D(z⁺)` vs true / NN / random-matched
   target; **subtract the NN-copy baseline** (`recon_adjusted = recon − recon(NN-copy)`,
   require CI-positive); rare-token decile audit (rare hits traceable to the NN target ⇒
   copying); output-entropy / near-duplicate rate (diversity collapse). Aggregate rates only —
   **no sequences or NN pairs ever leave the governed boundary**.
4. **Conditional-vs-marginal timing Δ (§4.ii)** — non-compensatory for *any* timing claim.

**Tier-2 (evaluability / promotion gates):**
5. **KS-D sample-size honesty** — per-cell adequacy floor (≥500 matched windows **and**
   non-empty median occupancy ≥2; ≥200 inter-event intervals for a timing cell); gate on
   upper-CI; **all evaluable cells conjunctive** (no pooled rescue); under-floor cells =
   `NOT_EVALUABLE` (never silently passed).
6. **Scissors + circle caveat (mandatory report string).** "`D(z⁺)` is a dev-only upper
   bound inside the encoder's circle; it bounds, does not demonstrate, generation; order/timing
   figures are prior-inflated by construction and must be read against the Guard-1/3 floors."
   Standing **scissors alarm**: a headline gate PASS while G1 latent-info ≈ floor ⇒
   between-cluster discriminability masquerading as within-cluster fidelity.
7. **G0 hygiene (preconditions; violation voids the rung).** Readout on **train**, selected on
   **dev**, **test sealed**; report train-vs-dev decode gap (overfit alarm); pre-declare the
   full cell set; **decoder-capacity saturation sweep** (a latent-bound ceiling saturates in
   capacity, a prior-bound one keeps climbing); patient/sequence-level bootstrap only.
8. **Empty-class falsifier (existing `rung1_decode.py`, keep).** Empty recall ≥ 0.95 **with
   precision + FPR** (recall alone gameable); count-0 a real class; order/Δt on non-empty only.

**Diagnostic only (route next rung, never pass/fail):** mean-vs-sample crispness gap (v0B
predictor is deterministic ⇒ N/A this rung; flagged for a future stochastic predictor);
`M2 − M1` magnitude (size of the prior reservoir); full latent-corruption battery
(`D(z⁺)` vs `D(ẑ)` component split) is the **Rung-3** falsifier — only the cheap M0 floor is
pulled forward here.

## 6. Cells, splits, coverage, adequacy

- **Cells:** SCID {30,90,365,730} d; MIMIC {0.25,0.5,1} d primary **+ 2 d sensitivity**
  (reported apart, non-gating — inherits the Rung-0 primary-band rule). Worst evaluable cell
  must clear; conjunctive; no pooled-only rescue; no source-specific relaxation.
- **Splits:** probes/decoder fit on **TRAIN `z⁺` only**; hyper-params on a TRAIN-internal
  fold; headline on **DEV**; **TEST sealed** (single-use `--confirm-test` after the decision
  rule is locked on dev).
- **Coverage/denominators (report all per cell):** n_nonempty, n(≥2-event) and its fraction,
  and the per-source `is_outcome`/censoring **refusal denominators** (Pi R3 Q6: SCID ~15% vs
  MIMIC ~0%). MIMIC short horizons (0.25/0.5 d) are the binding coverage risk for order/Δt →
  pre-registered they may report `NOT_EVALUABLE` rather than a weakened gate.
- **Bootstrap:** patient/sequence-level percentile CIs, n_boot=2000, seed 20260523; rep-vs-A
  contrasts use **synchronous paired** resampling (Rung-0 `paired_gap_streams` idiom).

## 7. Decision rule → the Rung-2 target record

Per property `p ∈ {count, order, timing}`, decided on **dev**, confirmed once on **test**;
mirrors the Rung-0 three-way plus a SWITCH branch. A cell is evaluable only if it clears the
§6 adequacy floor, else `INCONCLUSIVE` (fix power first). Gate on `M1−M0` excess.

- **COUNT:** A clears ≥0.80 (worst cell, w/ margin) ⇒ **KEEP `mean_embed` for rate/count**
  (expected; already owned by the context occupancy/count head). Else a cheaper/other arm
  clears while A fails ⇒ **SWITCH count target**. Else all fail ⇒ flag count for redesign.
- **ORDER:** A expected FAIL. A parameter-free order-preserving arm (D-sub) clears ≥0.70
  **and** survives the position-shuffle falsifier **and** the capacity control ⇒
  **SWITCH_TARGET → cheapest clearing arm** (record it as the Rung-2 order-capable target). A
  graded-but-none-clears ladder (A→D-sub monotone rising) ⇒ **ESCALATE to Rung-2 with a
  pre-scouted direction** (seq-of-latents / VQ — commit to the learned version). Flat A→D-sub
  ⇒ **ESCALATE_REDESIGN** (the pooling-target *family* is inadequate; needs a per-event /
  autoregressive target). — strongest "change targets" signal.
- **TIMING (KS-D within cell + conditional>marginal):** B1 clears **every** evaluable cell
  and survives the time-shuffle falsifier ⇒ **SWITCH timing target → B1**. B1 improves but
  doesn't clear ⇒ **timing belongs in a dedicated continuous-time head** (marked-TPP on the
  latent, P3), *not* the pooled target. No arm moves timing ⇒ confirms timing cannot ride a
  pooled target → route to the CT head.

**Combined verdict** = `{KEEP_MEAN_EMBED | SWITCH_TARGET:<arm> | ESCALATE_REDESIGN |
INCONCLUSIVE}` per property + an overall record naming the exact Rung-2 target per property
(e.g. "count→mean_embed; order→SetK M=4; timing→CT head"). **This is what makes the
foregone negative actionable.**

## 8. Module / CLI layout (consistent with `clinical_jepa/eval` + `arms/v0b`)

- `clinical_jepa/targets/target_reps.py` — **NEW.** `build_target_rep(name, ids,
  cumulative_days, t_query, window_days, is_empty, *, model, cfg)` for
  `{mean_embed, tap_concat, setk_sub}` (+ deferred). Parameter-free; frozen `E`;
  `block_spans.read_target_span` + `subwindow_blocks.carve_subwindows`; per-arm silence→`z_empty`.
- `clinical_jepa/eval/export_target_latents.py` — **NEW bridge.** Per source×horizon×split×arm:
  export `z_plus.npy` (fp16) + `target_props.jsonl` (multiset count vector, `N`, ordered token
  ids, inter-event `Δt`, occupancy, denominators). Reuses the governed-sidecar +
  censored-exclude + `is_outcome`/endpoint target-span scan pattern of
  `export_coarse_fine_latents.py`. Sidecar root gitignored.
- `clinical_jepa/eval/rung1_probes.py` — **NEW decoder-free library** (numpy/torch, aggregate):
  `(Eᵀ)⁺`/NNLS marginal inverse; ridge count; pairwise-order + τ; Δt-quantile PIT/KS-D;
  chance + matched-NN baselines; M0 interpolation/instance-swap floor; patient bootstrap
  (Rung-0 idiom).
- `clinical_jepa/eval/rung1_decode.py` — **KEEP + EXTEND.** Retain the empty/count hurdle +
  recall≥0.95 falsifier; add the trained upper-bound heads (token-set / order / Δt), the
  capacity sweep, and the shuffle falsifiers.
- `clinical_jepa/eval/rung1_ceiling.py` — **NEW driver** (argparse mirroring
  `export_mean_token_rollouts` / `rung0_horizon_decay`): per cell × arm build `z⁺`, fit on
  train, eval on dev, hold test; emit paired per-query records; aggregate.
- `clinical_jepa/eval/rung1_verdict.py` — **NEW.** §5 floor+excess gates + PRIOR-MASKED joint
  rule + §7 target-decision rule → `rung1-ceiling-manifest.json`
  (`per_source[source][W][arm][property]` + `decisions` map). Mirrors `rung0_verdict`.
- Config: `configs/rung1/sources-config.json` (per source: horizons, primary band, adequacy
  floors, arms, `d_time`, `M`, `d*` capacity width). Governed sidecar root:
  `run-workspace/local-governed/rung1/…` (gitignored placeholder only).

## 9. Compute — fits "training-free / tiny heads on one 3090 Ti"

Export = embedding lookup + masked mean/slot-mean over target spans (no transformer) → minutes
for 10⁵–10⁶ windows. Decoder-free probes: ridge closed-form `D×D` (D≈256–384) → ms;
NNLS/logistic/KS-D + bootstraps → seconds–minutes (vectorised, Rung-0 idiom). Trained heads:
few-hundred AdamW steps → minutes each; bounded by `~3 arms × ~4 probes × ~7 cells` tiny heads.
**Total < 1 GPU-hour; no representation training.**

## 10. Governance / aggregate-only

`z⁺` sidecars and per-window property arrays are **local governed gitignored artifacts**
(identical status to Rung-0). Published outputs = per source×horizon-cell aggregate metrics
(recoveries, exact-count, KS-D, τ, CIs), gate/joint/target verdicts, calibration/coverage as
**binned aggregate arrays**, refusal/evaluable denominators. Guard-3 overlap/copy metrics are
population rates only; the underlying pairs never leave the boundary. **No raw tokens,
per-patient rows, embeddings, codebooks, checkpoints, or real paths.** Inherited cross-cutting
guards: DATASET source-mask (target span structurally excludes it), `is_outcome`/endpoint
target-span scan (Q7), source×horizon-matched normalisation on every NN baseline. **Whole rung
labelled DEV-ONLY architecture-selection signal (inside the circle; Pi Q8).** TEST sealed.

## Cog imports

- `[[synthesis/orca-and-jepa-representation-space-translation]]` — the separation of *latent
  prediction* from *later rendering*, and "targets are the choosable object". **Changed the
  plan:** legitimises §3 (treat the *target-construction function* as the tunable variable this
  rung adjudicates, before any renderer) and keeps `D(z⁺)` framed as representation-ceiling, not
  generation evidence.
- Default planning principle *separate generator/executor from verifier* → **changed the plan:**
  the decoder-free probe (M1) is the verifier and is authoritative; the trained decoder (M2,
  the generator side) may never self-certify — the §2 floor+excess rule.
- Default principles *watch reward-hacking / over-claiming / sensitive-data* → **changed the
  plan:** the whole Guard battery (§5), the PRIOR-MASKED verdict, and the aggregate-only /
  circle caveat.
- Cog dream (soft) *floor-calibrated memorisation/diversity/privacy integrity gate* →
  **changed the plan:** Guard 3 as a **non-compensatory** copy-floor subtraction rather than an
  afterthought.
- **Not followed:** the pack's ORCA cross-site EHR↔English framing (out of scope for a
  target-decodability gate) and the `target-trial-emulation` page (no causal-contrast in this
  rung) — noted, not adopted.

**Scout trigger:** this blueprint materially moves the frontier (it pre-selects the Rung-2
target and can confirm the "change targets" branch). **SUGGEST** a Cog/Hydra scout wave on
*"cheapest order/timing-preserving JEPA target representations — seq-of-latents vs VQ vs
time-augmented pooling — and their frozen-decode ceilings; generative-prior masking of latent
failure"*. Per CLAUDE.md, **do not launch** without Chris's explicit authorisation.

## Open questions for Pi (R7 blueprint gate)

1. **Is the §3 parameter-free target contrast in-scope for Rung 1**, or should Rung 1 report
   only the incumbent `mean_embed` ceiling and defer the contrast to a Rung-1b? (Recommendation:
   include the 3 core parameter-free arms now — zero retrofit surface, and it is what makes the
   negative decision-useful.)
2. **Freeze the order metric:** exact ordered-sequence recon (≥0.70 gate) conditioned on the
   true multiset, with pairwise-τ as the graded companion — confirm, and freeze **M** for SetK
   (proposed M=4) before any run (the 0.70 gate is metric- and M-dependent).
3. **Freeze the latent-attributable-excess margin** (proposed ≥0.10 absolute `M1−M0` over the
   gate) and the **KS-D upper-CI** gating rule (vs point estimate).
4. **Freeze the per-cell adequacy floors** for order/Δt (≥500 windows, median occupancy ≥2,
   ≥200 inter-event intervals) and the `NOT_EVALUABLE` handling for MIMIC short horizons.
5. Confirm the **conditional-vs-marginal timing Δ** (CRPS/NLL, lower-CI>0) as a
   **non-compensatory** requirement for any timing-decodability claim.
