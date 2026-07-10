---
title: Rung 1 — frozen-decode ceiling BLUEPRINT (Rung 1a incumbent ceiling + 1b target panel) — Pi R7-revised
created: 2026-07-10
status: designed (3-proposal panel + synthesis); Pi R7 = REVISE folded in; pre-registration frozen pending Pi R8 confirm
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); latent = fresh v0B mean-token JEPA
reporting: aggregate-only (per source×horizon-cell metrics, CIs, gate verdicts; no seq ids / tokens / embeddings / real paths)
scope: DEV-ONLY architecture-selection ceiling (inside the encoder's circle; Pi Q8) — TEST inaccessible this run
---

# Rung 1 — frozen-decode ceiling

**Question (Pi Q3 hand-off):** what is the **decode ceiling** of the pooled target latent
`z⁺` — the upper bound on what any generator built on it could reconstruct? Split (Pi R7 #4):

- **Rung 1a** — the frozen **incumbent `mean_embed` ceiling**, *independently verdictable*.
- **Rung 1b** — a pre-registered **parameter-free target-contrast panel**, *secondary*
  architecture selection. An alternative arm may only **`NOMINATE_TARGET_FOR_RUNG2`**, never
  `SWITCH_TARGET` — no predictor has yet been trained to reach any alternative target; Rung 2
  must train it and test prediction-achieved fidelity before adoption.

> **R7 revision log (folded in):** (1) per-readout on-manifold swap-floor excess — a weak M1
> can no longer label a nonlinear M2 "prior-masked"; (2) unconditional exact-order stays the
> headline, oracle-multiset order is a *mechanistic companion*, arm A uses a permutation-pair
> invariance test not a content-changing swap; (3) SetK renamed *temporal-slot* and gated on
> slot-wise structure, not exact order; (4) 1a/1b split + NOMINATE-not-SWITCH; (5) added the
> parameter-free `[mean⊕log1pN]` count arm + dimensional empty reps; (6) bandwidth/parameter/
> byte pricing + frozen multiplicity; (7) hurdle/mixed timing + randomized PIT for zero/tied
> Δt; (8) no `--confirm-test` in the ordinary driver. Q3 excess reworked, Q4 cluster floors +
> ≥1000 intervals + precision simulation, Q5 CRPS-skill lower-CI ≥0.05.

## 1. Structural priors — what `z⁺` can and cannot contain (the spine)

`z⁺ = mean_embed(ids) = (1/N)·Σᵢ E[tᵢ] = Eᵀp` — masked mean of target-token embeddings; `E`
the frozen table (V=1050), `c` the multiset counts, `N=Σc`, `p=c/N`. This closed form fixes
each property's a-priori status. **Stated honestly (Pi R7):** *direct* order is **absent
beyond the multiset**; *direct* timing is **absent** — but **content-derived proxy
predictability may remain** (a strong content→order prior; token-identity rate proxies).
Proxy predictability is measurable but is **not** the same as retaining observed per-instance
order/timing.

| Property | Structural status in `z⁺` | A-priori ceiling |
|---|---|---|
| (1) multiset marginals | `z⁺` **is** `Eᵀp` | **HIGH** — analytically invertible up to `col-space(E)` / row degeneracy |
| (2) count `N` | divided out by `1/N`; survives only via `‖z⁺‖` / concentration | **LOW–MODERATE, genuinely uncertain** — a real test of 0.80 |
| (3) order | `z⁺` **provably permutation-invariant** → no instance order **beyond the multiset** | direct = **none beyond multiset**; a **content→order prior** may still score |
| (4) Δt timing | `cumulative_days`/`time_deltas` **never embedded** | direct = **none**; a **token-identity rate proxy** may still score |

Measuring the exact ceiling is high-value: it empirically bounds any `z⁺`-only generator,
quantifies the content-proxy magnitude, and **drives the Rung-2 target decision**.

## 2. Attribution — per-readout floor-adjusted excess (Pi R7 #1, the load-bearing fix)

`D(z⁺)` is inside the encoder's own latent → a **DEV-ONLY ceiling, never external
validation** (Pi Q8). The hazard (design-doc R2.2): a co-trained decoder can **mask latent
failure via its generative prior**. But — **a weak readout's failure cannot label a stronger
readout "prior-masked."** So each readout is judged by **its own** floor-adjusted excess.

**Readout ladder** (all fit on TRAIN / internal-train only):
- **M0** — prior floor: the **on-manifold wrong-instance swap** is the PRIMARY floor —
  `z_swap` = a different real target latent drawn within the *same* predeclared
  source×horizon×occupancy/matching cell. (Interpolation-to-mean is a *graded diagnostic*
  only, not the floor.)
- **M1a** — **truly decoder-free**: analytic `(Eᵀ)⁺` / simplex-NNLS multiset inverse (no
  training at all).
- **M1b** — **trained-simple** readouts (ridge, multinomial-logistic, count GLM, Δt quantile
  regression). These *are trained* and can learn priors — so they get the same swap test.
- **M2** — expressive trained decoder `D(z⁺)` (the head in `rung1_decode.py`, extended).

**Per-readout excess:** `excess_j = score(Mⱼ(z⁺), y) − score(Mⱼ(z_swap), y)` for
higher-is-better metrics (skill-score / sign-reversed for losses), with a **paired
patient/cluster bootstrap** CI, swap fixed within the predeclared cell.

**Classification (per property × cell):**
- M1 absolute gate **and** M1 excess pass ⇒ **SIMPLE/ANALYTIC decodable**;
- M2 absolute gate **and M2's own** excess/corruption/copy-floor pass ⇒ **NONLINEAR
  decodable** — *whether or not M1 passes* (a weak linear verifier cannot veto genuine
  nonlinear latent information);
- M2 absolute gate passes **but M2's own** excess/corruption fails ⇒ **PRIOR-MASKED**;
- neither absolute gate clears ⇒ **NOT-DECODABLE / INCONCLUSIVE** (by precision, §6).

`M1−M0` remains an interpretable **lower-bound** result but is **not** the sole authority over
M2. All of {M0, M1a, M1b, M2} scores + per-readout excesses reported per cell.

## 3. The Rung-1b target-contrast panel (parameter-free; NOMINATE-only)

Hold the decode probe, per-head parameter/FLOP budget, splits, matching, and cells
**identical**; vary **only** the parameter-free target-construction `g:(ids, cumulative_days,
t_query, W, is_empty) → z⁺` (frozen `E`, no retraining, zero retrofit surface). `φ,ψ` are
**fixed** untrained sinusoidal maps, width `d_time` (frozen before dev). Empty is defined
**dimensionally** per arm (Pi R7 #5).

| Arm | Rung | Construction | Direct order | Direct timing | Dim | Nominates |
|---|---|---|---|---|---|---|
| **A. `mean_embed`** (incumbent) | **1a** | `(1/N)Σ E(idᵢ)` | none beyond multiset | none | D | — (the reference ceiling) |
| **B1. TAP-concat** | 1b | `[mean E ⊕ mean φ(τᵢ,logΔtᵢ)]`; empty `[z_empty ⊕ φ_empty]` | none | **event-time rate/dist** | D+d_time | a **timing** target |
| **Cnt. `[mean E ⊕ log1p N]`** | 1b | append scalar log-count; empty `[z_empty ⊕ 0]` | none | none | D+1 | the **cheapest exact-count** target |
| **TS-M. temporal-slot / SetK** | 1b | M equal-**wall-clock** slots (`carve_subwindows`), `[m₁⊕…⊕m_M]`, each empty slot `z_empty` | **coarse slot only** (perm-invariant *within* slot) | **coarse rate profile** | M·D | a **coarse temporal-structure** target (NOT exact order) |

**Pricing (Pi R7 #6):** TS-M has `M·D` input dims vs `D`, `D+d_time`, `D+1`. **Match trained-
head parameter/FLOP budget across arms (or use a fixed bottleneck) and report latent
bytes/dimension** alongside fidelity. Freeze `d_time` and **M=4 primary, M=8 sensitivity**
before dev. **Multiplicity:** freeze a hierarchical comparison order (or a multiplicity
correction) for arm selection; **no best-of-panel selection without paired, corrected CIs.**

**Alt-arms are ceiling-only:** the predictor was trained to predict `mean_embed`, so `D(ẑ)`
and the `D(z⁺) vs D(ẑ)` split exist **only for arm A (Rung 1a)**. Every 1b arm reports the
ceiling `D(z⁺)` only and can therefore only **nominate**. **Deferred entirely** (needs
training / retrofit management): PosTag ordinal, TAP-moment cross term, fixed-k-means VQ,
learned slot-attention / learned VQ — any learned arm would be trained train-only on a
probe-blind objective and frozen before decode.

## 4. Per-property probes, frozen gates, baselines

Per source×horizon cell, **non-empty** windows (empties → §5 falsifier); order/Δt require
**≥2 events**. All baselines / cell-means / NNs / content-priors fit or selected on
**train / internal-train only**; **dev used once** for the verdict; **NN baselines
patient-disjoint** and never conditioned on unavailable future attributes (except explicitly
oracle-conditioned mechanistic probes).

| Property | Frozen gate (headline metric) | Companions / mechanistic | Hard floors (M0 / baselines) | Direct expectation (arm A) |
|---|---|---|---|---|
| (1) marginals | *descriptive* (summary-stat cross-check, not a programme gate): macro-F1, set-Jaccard, cosine/TV | — | cell-marginal freq; matched-NN | HIGH |
| (2) count | **exact-count ≥0.80 per cell AND the readout's paired excess lower-CI > 0.10** | count-support / modal-prevalence report | modal/mean count; context-length rate-prior; NN-copy count | uncertain — real test |
| (3) order | **unconditional exact ordered-seq recon ≥0.70 per cell** | oracle-multiset-conditioned exact order + **tie-aware Kendall-τ** (labelled oracle-assisted, NOT generation metrics) | prevalence-matched perm; **content-prior order** (multiset bigram/precedence); canonical-sort | none beyond multiset; any score = **content-prior** |
| (4) timing | **KS-D upper-95%-CI ≤0.05 per cell (randomized-PIT) AND normalized-CRPS skill lower-CI ≥0.05** | calibration/coverage; NLL (secondary) | cell-marginal Δt; NN Δt | none direct; token-rate proxy only |

**Order metric definition (Pi R7 #2, freeze exactly):** headline = **unconditional** exact
ordered-sequence reconstruction (decode the ordered token sequence from `z⁺`, exact match,
≥2-event windows, **no truncation**, denominator = all evaluable ordered windows). Repeated-
token ties and tournament/cycle resolution defined in the harness (frozen). **Arm A control =
exact permutation-pair invariance test:** same multiset, altered order → identical `z⁺` →
identical decoder output *by construction* — confirms zero instance-order beyond the multiset;
any real-data order score is reported as **content-prior performance**, not latent order. Order-
bearing alternatives use **multiset-matched swaps** (where feasible) + controlled permutation
hard negatives. TS-M is gated on **predeclared slot-assignment / slot-wise multiset fidelity**,
with exact-sequence + τ reported separately — it **cannot** nominate an exact-order target.

**Timing mechanics (Pi R7 #7, freeze exactly):** simultaneous EHR events create a **point
mass at Δt=0** → decile quantiles do not define a calibrated CDF. Use a **hurdle / mixed
timing distribution** (zero-mass + continuous tail) and **randomized PIT** for the discrete
mass/ties; freeze interpolation, tail handling, and quantile-crossing repair. **Bandwidth-
matched** density estimation across arms/baselines. Marginal **and** conditional timing models
fit on **train only**. Gate on the **upper-95%-CI of KS-D ≤0.05** (skeptic-favouring) **and**
the conditional-vs-marginal **normalized CRPS skill** `1 − CRPS_cond/CRPS_marg` with paired
lower-95%-CI **≥0.05** — **both** required in **every** evaluable primary cell (non-
compensatory). NLL secondary unless a complete mixed-density model is frozen.

## 5. Guard battery (floor-adjusted), ranked; non-compensatory flagged

**Tier-1 (must-pass; a failure voids the ceiling claim):**
1. **Per-readout on-manifold swap floor (§2).** Primary floor. Each of M1a/M1b/M2 must beat
   its own wrong-instance-swap score (paired cluster bootstrap) to earn a "decodable" label;
   M2-passes-raw-but-fails-swap ⇒ PRIOR-MASKED. Arm A order uses the permutation-pair
   invariance test instead (a content-changing swap is invalid there).
2. **Hard baselines × excess.** Every gate met by the readout's own floor-adjusted excess
   (count/order excess lower-CI > 0.10; timing CRPS-skill lower-CI ≥0.05), never the raw
   score. Order must beat **content-prior order**; count must beat **modal-count** and the
   **context-length rate-prior**.
3. **Memorisation / NN copy-floor (Cog integrity gate).** Patient-disjoint NN-in-latent;
   shingle-3gram/LCS overlap of `D(z⁺)` vs true / NN / random-matched; **subtract the NN-copy
   baseline** (require CI-positive); rare-token decile audit; output-entropy / near-duplicate
   rate. Aggregate rates only — **no sequences or NN pairs leave the boundary.**
4. **Conditional-vs-marginal timing CRPS skill (§4)** — non-compensatory for any timing claim.

**Tier-2 (evaluability / promotion):**
5. **KS-D sample-size honesty (Pi R7 #4/#7):** cluster-based adequacy floors (§6); gate on
   upper-CI; **all evaluable cells conjunctive** (no pooled rescue); randomized-PIT; hurdle/
   mixed model.
6. **Interpolation-to-mean curve (graded diagnostic, not the floor).** Monotone degradation is
   supportive evidence of latent-reading; the *swap* (Guard 1) is authoritative.
7. **Scissors + circle caveat (mandatory report string).** "`D(z⁺)` is a dev-only upper bound
   inside the encoder's circle; it bounds, does not demonstrate, generation; order/timing
   figures are content-proxy-inflated by construction." Standing scissors alarm: a gate PASS
   with swap-excess ≈ 0 ⇒ discriminability masquerading as fidelity.
8. **G0 hygiene (violation voids the rung).** Readout on train, selected on dev-once, **test
   inaccessible**; report train-vs-dev decode gap; pre-declare the full cell set + comparison
   order; **decoder-capacity saturation sweep** (a latent-bound ceiling saturates in capacity,
   a prior-bound one climbs toward the marginal-achievable); patient/cluster bootstrap only.
9. **Empty-class falsifier (existing, keep).** Empty recall ≥0.95 (+ precision, FPR); count-0
   a real class; order/Δt on non-empty only.

**Diagnostic only (never pass/fail):** mean-vs-sample crispness gap (v0B predictor
deterministic ⇒ N/A this rung); `M2 − M1` magnitude (prior reservoir size); `D(z⁺)` vs `D(ẑ)`
component split (arm A only; full latent-corruption battery is the Rung-3 falsifier).

## 6. Cells, splits, coverage, adequacy (Pi R7 #4)

- **Cells:** SCID {30,90,365,730} d; MIMIC {0.25,0.5,1} d primary **+ 2 d sensitivity**
  (non-gating, inherits the Rung-0 primary-band rule). Worst evaluable cell must clear;
  conjunctive; no pooled-only rescue; no source-specific relaxation.
- **Splits:** probes/decoder/baselines/NNs fit on **TRAIN / internal-train only**; headline on
  **DEV, used once**; **TEST inaccessible this run** — no `--confirm-test` in the ordinary
  driver; a later one-shot test confirmation needs a **separately locked manifest +
  authorization** after target selection (Pi R7 #8).
- **Cluster-based adequacy floors (revised):**
  - *order:* ≥500 **distinct patient/sequence bootstrap clusters with N≥2** (not merely 500
    windows);
  - *count:* ≥500 non-empty clusters + count-support / modal-prevalence report;
  - *timing:* ≥500 clusters with N≥2 **and ≥1,000 inter-event intervals** as the minimum smoke
    floor, **plus a pre-run cluster-bootstrap precision simulation** demonstrating the design
    can certify an upper KS-D of 0.05 (200 is far too small — even the IID 95% DKW scale is
    ≈738 obs *before* clustering).
- **Evaluability vs verdict:** cells **below** a floor = `NOT_EVALUABLE`; cells **above** the
  floor but with a **wide CI** = `INCONCLUSIVE` (not FAIL).
- **Coverage/denominators (report per cell):** n_nonempty, n(≥2-event) + fraction, n_clusters,
  n_intervals, and the per-source `is_outcome`/censoring **refusal denominators** (Pi R3 Q6).
- **Bootstrap:** patient/sequence-cluster percentile CIs, n_boot=2000, seed 20260523; rep-vs-A
  contrasts use synchronous **paired** cluster resampling (Rung-0 idiom).

## 7. Decision rule

Per property, decided **on dev, once**; mirrors the Rung-0 three-way + a NOMINATE branch.
Gate on the **readout's own floor-adjusted excess** (§2/§4), not on `M1−M0`.

- **Rung 1a (incumbent verdict, independent):** for each property, arm A →
  `{DECODABLE(simple|nonlinear) | PRIOR-MASKED | NOT-DECODABLE | INCONCLUSIVE | NOT_EVALUABLE}`
  per cell, combined worst-cell. Expected: count uncertain; order = content-prior only (direct
  none beyond multiset); timing = proxy only. This is the frozen-decode ceiling of `mean_embed`
  and stands on its own.
- **Rung 1b (target nomination, secondary):** an alternative arm whose true-target ceiling
  clears a property's gate (with excess, bandwidth-matched, multiplicity-corrected) →
  **`NOMINATE_TARGET_FOR_RUNG2:<arm>`** for that property (TS-M may nominate only a *coarse
  temporal-structure* target, never exact order). A monotone-but-none-clears ladder ⇒
  **nominate a direction** (per-event/ordinal or learned VQ/seq-of-latents) for Rung 2 to build
  and train. Flat ⇒ **ESCALATE_REDESIGN** (pooling-target family inadequate).

**Rung 2 obligation (explicit):** a nomination is not adoption — Rung 2 must train a predictor
to reach the nominated target and test **prediction-achieved** fidelity before any switch.

## 8. Module / CLI layout

- `clinical_jepa/targets/target_reps.py` — **NEW.** `build_target_rep(name, …)` for
  `{mean_embed, tap_concat, count_concat, temporal_slot}` (parameter-free; frozen `E`;
  `block_spans.read_target_span` + `subwindow_blocks.carve_subwindows`; dimensional empties;
  reports `target_dim` + latent bytes).
- `clinical_jepa/eval/export_target_latents.py` — **NEW bridge.** Per source×horizon×split×arm:
  `z_plus.npy` (fp16) + `target_props.jsonl` (multiset counts, `N`, ordered ids, inter-event
  `Δt` incl. zero-mass, occupancy, cluster id, denominators). Reuses the governed-sidecar +
  censored-exclude + `is_outcome`/endpoint target-span scan pattern. Sidecar root gitignored.
- `clinical_jepa/eval/rung1_probes.py` — **NEW.** M1a analytic `(Eᵀ)⁺`/NNLS; M1b trained-simple
  readouts; per-readout on-manifold swap-floor excess; matched-NN + content-prior baselines;
  hurdle/mixed timing + randomized-PIT KS-D + normalized CRPS skill (bandwidth-matched);
  permutation-pair invariance (arm A); patient/cluster bootstrap.
- `clinical_jepa/eval/rung1_decode.py` — **KEEP + EXTEND.** Empty/count hurdle + recall≥0.95;
  add M2 heads (token-set/order/Δt) with matched parameter budget + capacity sweep.
- `clinical_jepa/eval/rung1_precision_sim.py` — **NEW.** Pre-run cluster-bootstrap precision
  simulation certifying the design can resolve an upper KS-D of 0.05 per timing cell.
- `clinical_jepa/eval/rung1_ceiling.py` — **NEW driver** (dev-only; **no** `--confirm-test`).
- `clinical_jepa/eval/rung1_verdict.py` — **NEW.** §2 classification + §7 rule →
  `rung1-ceiling-manifest.json` (`per_source[source][W][arm][property]` with readout scores,
  excesses, verdicts; separate `rung1a`/`rung1b` blocks). A distinct locked test manifest is a
  later, separately-authorized artifact.
- Config `configs/rung1/sources-config.json` (horizons, primary band, cluster floors, arms,
  `d_time`, M={4,8}, matched head budget, comparison order). Governed root
  `run-workspace/local-governed/rung1/…` (gitignored placeholder only).

## 9. Compute

Export = embedding lookup + masked mean/slot-mean (no transformer) → minutes. Probes: ridge
closed-form; NNLS/logistic/quantile/KS-D/CRPS + cluster bootstraps → seconds–minutes
(vectorised). Precision sim → minutes. M2 heads: few-hundred AdamW steps, matched budget →
minutes each; bounded by `~4 arms × ~4 probes × ~7 cells`. **Total < 1 GPU-hour; no
representation training.**

## 10. Governance / aggregate-only

`z⁺` sidecars + per-window property arrays are **local governed gitignored artifacts**
(Rung-0 status). Published = per source×horizon-cell aggregate metrics, readout scores +
floor-adjusted excesses + CIs, gate/joint/nomination verdicts, calibration/coverage as binned
aggregate arrays, refusal/evaluable/cluster/interval denominators, latent bytes/dim. Guard-3
overlap/copy metrics are population rates only; underlying pairs never leave the boundary. **No
raw tokens, per-patient rows, embeddings, codebooks, checkpoints, or real paths.** Inherited
cross-cutting guards (DATASET source-mask, `is_outcome`/endpoint target-span scan,
source×horizon-matched patient-disjoint NN). **Whole rung DEV-ONLY; TEST inaccessible.**

## Cog imports

- `[[synthesis/orca-and-jepa-representation-space-translation]]` — target-representation quality
  vs later surface rendering. **Changed the plan:** legitimises the 1b target panel, but (per
  Pi) alternatives **nominate, not certify** a new predictor/generator → §3/§7 NOMINATE-only.
- Default principle *separate generator/executor from verifier* — **changed the plan:** the
  fix in §2 requires floor-adjusted controls for **M2 itself**; a weak linear verifier cannot
  veto genuinely nonlinear latent information (the R7 #1 correction).
- `[[synthesis/orca-external-prior-specification-tests]]` — define external/proxy quantities
  before comparing to learned geometry. **Changed the plan:** the §1 language distinguishing
  **direct** order/timing retention from **content-derived proxy** predictability.
- Cog dream (soft) *floor-calibrated memorisation/diversity integrity gate* → Guard 3 as a
  non-compensatory copy-floor subtraction.
- **Not followed:** `[[concepts/target-trial-emulation]]` (no treatment/action contrast in this
  rung); the memorisation dream card kept as soft guidance, not canonical.

**Scout trigger (Pi-endorsed):** the Rung-2 target choice is now a genuine frontier question.
**SUGGEST** a sanitized Cog/Hydra scout — *"bandwidth-matched order/timing-preserving JEPA
targets (per-event/ordinal vs VQ vs seq-of-latents vs time-augmented pooling) and nonlinear
prior-attribution controls"* — **after** the revised Rung-1a/1b contract is frozen (Pi R8).
Per CLAUDE.md, **do not launch** without Chris's explicit authorisation.

## Status / next

Pi R7 = REVISE, all 8 changes + Q2–Q5 rulings folded in. Pre-registration is frozen **pending
a Pi R8 confirm** that the revised contract is faithful (esp. §2 per-readout attribution, §4
order/timing definitions, §6 cluster floors + precision simulation). On R8 GO → build the
harness (Rung 1a first, 1b in the same governed pass).
