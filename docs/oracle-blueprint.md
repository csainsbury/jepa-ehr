---
title: Semi-synthetic clinical-EHR ORACLE blueprint — the external certifier that unlocks governed T4
created: 2026-07-11
status: DRAFT v3 — design panel + Cog + fable5 + Pi oracle-gate REVISE folded in (10 changes); an EXTERNAL-TO-ENCODER synthetic SPECIFICATION TEST + recipe falsifier (NOT a real-EHR certificate); governed T4 locked pending Pi re-gate
reporting: aggregate-only; oracle is FULLY SYNTHETIC (safe-public/committable); real touch = a one-time aggregate-marginal read for calibration targets only
scope: the DEEPEST external anchor in the falsifier ladder — the only place a decision rises from NOMINATE to CERTIFY
---

# Semi-synthetic clinical-EHR oracle

Everything on real dev is inside the encoder's own latent circle → can only **NOMINATE**. This
oracle has a **KNOWN generative order/intensity rule**, so "true order/timing" is defined *outside
any learned encoder* → it can **CERTIFY**. It is the mandatory precondition (Pi) before any governed
**T4** (learned VQ order-target) work: `rung2_t4_stub.guard_t4` already fail-closes until a frozen,
Pi-gated oracle authorization manifest exists. This blueprint requires its **own** Cog preflight +
Pi gate before it may flip `oracle_frozen=True`.

> **★ REAL→SYNTHETIC TRANSFER CAVEAT (load-bearing; on every manifest).** Passing the oracle
> certifies recovery of a **KNOWN synthetic** instance-specific order/intensity signal under declared
> generative-rule families with real-*matched aggregate marginals*. It does **NOT** certify recovery
> of the unknown, unlabelled instance structure in real governed EHR. The oracle is **NECESSARY**
> (it rules out content-prior hallucination and the self-manufactured-codebook ceiling) but **NOT
> SUFFICIENT** for a real-data fidelity claim; real claims remain `NOMINATE`-ceiling (Rung 2). The
> rule families are hypotheses about clinical order/timing, not the true clinical DGP.

## fable5 falsifier pass — folded-in changes
fable5's ruling: **the oracle is the right instrument for one narrow job (falsifying the
self-manufactured-ceiling / null-hallucination failure of a learned target — a question with ground
truth ONLY in synthetic data) but over-sold for everything else; as drafted it CANNOT unlock T4.**
Folded-in **BLOCKING** fixes:

1. **Instance-level knob randomization (B1 — the single highest-value fix).** Per-*cell* nulls make
   null-ness *inferable from the observable data* (residual-order variance ≈0), so a codebook trained
   across the grid can learn a **null-detector** ("if residual variance ≈0 emit content-prior, else
   my learned order") and pass U2 by construction. Fix: draw a per-INSTANCE `λ_order_i` from a
   mixture with a null atom, **interleaving null and positive instances within ONE dataset** (same
   source, same marginals) + **camouflaged nulls** (marginals matched to a positive cell) +
   **off-grid certification knobs** (κ not in the training grid). Label isolation is insufficient —
   null-ness is inferable, not just labelled.
2. **Replace R\* with the CONTEXT-Bayes ceiling `R_bayes = E[order | observable context]` (B2).** R\*
   is handed the true state `h`, so R\* passing is near-tautological (proves label self-consistency,
   not that the task is solvable *from context*). A cell where `h` recovers order but context is
   ~independent of the order-driving `h`-projection is an IMPOSSIBLE test R\* passes. `R_bayes` is
   constructible (the DGP is fully known → `p(h|context)` is closed-form/MC). **Fair gate: `R_bayes`
   must materially beat `R0`; if `R_bayes ≈ R0` the cell is a HIDDEN NULL (order in `h` but not
   inferable from context) → EXCLUDED from positive certification.** R\* demoted to a secondary
   SNR/plumbing check.
3. **Fix the false "independent maps ⇒ no leak" claim + add `R_{h⊥order}` (B3).** Independent maps on
   a SHARED `h` do NOT give independent signals (`v_c·h` and `b·h` are correlated). A method reading
   ANY `h`-correlated channel (timing/count) scores positive order-skill without recovering the
   order functional `v_c`; R0 (no `h`) can't expose it. Fix: a THIRD reference `R_{h⊥order}` (given
   `h`, allowed the non-order projections, **DENIED `v_c`**) that **MUST FAIL** positive; certified
   order-skill is **orthogonalized** against the other-channel-predictable component. This **is** the
   narrowed U4.
4. **Harden the shipped guard (B4 — DONE).** `rung2_contract.t4_governed_allowed` checked only
   `oracle_frozen+pi_gate` (fail-OPEN — a REFUTED manifest would pass); now requires the full
   conjunction (verdict, unlock_checks, reference bounds incl. `R_bayes_beats_R0` /
   `R_h_perp_order_fails_positive`, precision-sim, realism-envelope), fail-closed.
5. **Structural family diversity for LOFO (B5).** All listed families share `h` + a linear readout =
   ONE meta-family; a method learning "recover the linear projection of context onto `h`"
   generalizes A→E trivially. Fix: ≥1 held-out family with **NO `h`-dependence** in the order channel
   (realized-history / exogenous-Markov driven); certify on **≥2 structurally-distinct** held-out
   families; add an `h`-projection **shortcut probe** (a method that only recovers a projection of
   context onto `h` must NOT pass).

**Rename + rescope (fable5 N1/N2, folded in):** the verdict is `synthetic_recovery_CERTIFIED` /
`T4_TRAINING_AUTHORIZED` (NOT "CERTIFIED") and every downstream governed-T4 result carries
`governed_t4_real_output_ceiling = "NOMINATE"` — unlock authorizes *building/training* T4, never
believing its real outputs. **The counterfactual apparatus is SCOPE CREEP for an ORDER target:**
DROP U5 + potential outcomes + propensity + positivity from the order-T4 unlock (a manifest citing
`Y(0)/Y(1)`+`e_i` reads as a causal certification the arc disavows); keep only the narrowed **U4 =
`R_{h⊥order}` orthogonality** and defer the full counterfactual machinery to a separate, later
counterfactual-oracle blueprint. **The word "CERTIFY" does no sufficiency work:** the only claim T4
ever earns is "on the oracle's synthetic distribution, this recipe recovers planted order"; real
claims stay `NOMINATE` forever (there is no real order label — that is why the oracle is synthetic).

## Pi oracle-gate REVISE — folded-in changes (10)
Pi accepted the narrowed purpose (a recipe FALSIFIER) but blocked on four load-bearing
inconsistencies + provenance. **This is an EXTERNAL-TO-ENCODER synthetic SPECIFICATION TEST, not an
external certificate for real EHR** (Pi #10): `synthetic_recovery_CERTIFIED` means only "this frozen
recipe recovered the planted *context-predictable* mechanism under the declared synthetic families
and nulls"; it authorizes a governed *experiment* and raises **no** real output above `NOMINATE`.

1. **Actually remove the counterfactual/policy apparatus (Pi #1).** DELETE action/propensity,
   potential outcomes `Y(0)/Y(1)`, positivity/overlap, treatment-effect knobs, `conf_level`, U5, the
   causal-style support-starved cell, and all TTE language from the generator, knob grid, labels,
   manifests, and imports. Keep only non-causal **nuisance-channel correlation strength** +
   multiset-support density needed for the order/timing falsifiers. A counterfactual oracle is a
   separate future blueprint.
2. **Gate prediction-achieved `D(ẑ)` context-only, NEVER `D(z⁺)` (Pi #2).** The method-under-test has
   a fail-hard prefix-safe API: the target encoder/codebook/decoder may train on synthetic TRAIN
   futures; the **predictor receives observable synthetic context only**; sealed-cert scoring is on
   **sampled `D(ẑ(context))`** vs the known Bayes order/intensity quantities; `D(z⁺)` is a reported
   ceiling/attribution diagnostic that can NEVER satisfy U1/U2/U3/U4/U6. Add
   `assert_predictor_context_only` + future/label perturbation invariance + a **fail-hard test where
   a target encoder perfectly writes realized order into codes while the context predictor has no
   signal — the recipe MUST fail U2/positive** (else the oracle repeats the very self-manufactured-
   ceiling it exists to falsify).
3. **Fix the null under irreducible noise (Pi #3).** `r_j = μ_class + λ_order·(v·h) + σ_r·η_j`, so at
   `λ_order=0` the residual noise `σ_r·η` means realized order ≠ `argsort(μ_class)`. Define the null
   as **zero context-PREDICTABLE residual order**, not zero realized residual; score predicted
   pairwise probabilities against the known **context-Bayes** probabilities. Either set null noise to
   0 (exact equality) or retain it and state `π0` is the Bayes content prior while realized order
   varies irreducibly. **U2 rejects prediction of non-existent *predictable* residual, not
   reconstruction of random future noise.**
4. **Replace the impossible nuisance reference (Pi #4).** Masking the explicit `v_c` coordinate does
   NOT make the correlated projections independent of `v_c·h`, so `R_{h⊥order}` "must fail" is
   incoherent. Use **`R_nuis = E[order | allowed non-order projections/channels]` and require the
   recipe's INCREMENTAL skill over `R_nuis` to clear a practical CI margin** (rename U4 →
   **nuisance-incremental recovery**, freeze the score/margin), PLUS designated Σ-orthogonality cells
   (order projection orthogonal to nuisance — there the nuisance reference must fail) + separate
   correlated-leak stress cells.
5. **Property-specific unlock (Pi #5).** Governed T4 is an ORDER target → its authorization conjuncts
   ONLY order-relevant checks (E-O, null, monotonicity, **nuisance-incremental**, support, bandwidth,
   realism, precision). **E-T1..4 do NOT gate an order-only T4** (an unrelated timing failure must
   not veto order). The continuous-time head emits a **separate** synthetic timing-recovery manifest.
   *(Guard `t4_governed_allowed` now checks only `ORDER_UNLOCK_CHECKS`.)*
6. **Bind authorization to the EXACT recipe (Pi #6).** `codebook_postdates_oracle` is insufficient —
   freeze a **`t4_recipe_hash`** over {target-encoder/codebook/predictor/decoder architectures,
   losses+weights, optimizer/schedule, bit budget, sampling/temperature, training split, seeds/seed
   policy, evaluator commit}. Governed T4 must present the SAME hash; any material change ⇒
   re-certification. Pre-register a finite **recipe registry** before opening sealed-cert; failed
   recipes cannot re-consume the same sealed seeds → rotate a separately frozen certification split.
   *(Guard now takes `presented_recipe_hash` and refuses a mismatch.)*
7. **Harden the schema (Pi #7).** Remove legacy `"CERTIFIED"` (only `synthetic_recovery_CERTIFIED`);
   require `schema_version`, `oracle_mechanism_hash`, `evaluator_commit`, `certified_recipe_hash`,
   `held_out_family_ids`, `sealed_cert_run_id`, `gate_event_ref`, and the property-specific unlock
   block; missing/stale ⇒ fail-closed; `pi_gate="PASS"` is tied to the committed gate artifact +
   expected blueprint hash. *(Guard DONE + tested.)*
8. **Freeze references/OC numerically (Pi #8).** The `R_bayes−R0` margin + MC-error tolerance +
   independent MC seeds; the nuisance-incremental margin; the exact per-instance null decision
   statistic + how FPR aggregates over the many precedence pairs per sequence (the **cluster unit** =
   patient/sequence, family-level conjunction explicit); # train/dev/sealed structural families with
   **≥2 held-out meta-families**; the exact no-`h` family + `h`-projection shortcut method + pass/fail
   thresholds; off-grid knob values, sample sizes, MDE, multiple-testing correction, hidden-null
   handling. ("≥200 null / ≥100 positive seeds" are starts, not the contract.)
9. **Separate calibration from certification (Pi #9).** Real aggregate marginals calibrate the
   generator on a **dedicated calibration stage only**, with the fitting algorithm/tolerances frozen
   *before* reading them; certification families/seeds stay sealed. The realism envelope is an
   **eligibility condition, not transfer evidence**. Record the governance classification of the
   aggregate calibration targets — do NOT commit licensed/sensitive aggregates merely because the
   derived synthetic is safe-public.
10. **Headline language (Pi #10).** External-to-encoder synthetic specification test + recipe
    falsifier — see the frontmatter + top of this section.

**Real-data ruling (Pi — the terminal claim ladder):** **DEV = `NOMINATE`.** After the synthetic
oracle only: **`NOMINATE + synthetic-recovery-certified recipe`.** A *separately pre-registered,
one-shot LOCKED/external/prospective* observation-space evaluation could later confirm **real
held-out OBSERVATIONAL PREDICTIVE fidelity** (held-out sequence likelihood/proper scores, exact/
count/order/timing metrics, calibration, coverage, copying/diversity controls vs strong AR/rate/
content baselines) — but NEVER latent-mechanism or causal truth (those labels are unavailable). If no
locked/external route is authorized, the honest terminal real-data claim is exactly **nomination +
a synthetic-recovery-certified recipe, never a real fidelity certificate.**

## 1. What "semi-synthetic" means here
**Fully-synthetic MECHANISM + latent state; real-marginal CALIBRATION envelope.** One committed
latent state `h_i` generates BOTH the context and the future window, so every certification baseline
(content-prior order, marginal rate, nuisance-channel projections) is a **closed-form committed function
of `h_i` and fixed spec constants** — defined entirely outside any learned encoder (the deepest
anchor). The "semi" = the mechanism's free parameters are **calibrated to published aggregate real
marginals** so it is a fair test, not a toy. A real-context-conditioned variant is admissible only
as a *supporting realism-stress diagnostic* (stays governed, never certifies).

## 2. The GENERATOR — one latent state, three separable known channels
Per sequence (pure numpy, deterministic given seed): source `s_i`, phenotype `k_i`, latent state
`h_i ∈ ℝ⁸ ~ N(m_{k_i},Σ_{k_i})` (the confounder/driver), source parameter block `Θ_s` (wall-clock
scale, base rate, Δt=0 mass, sparsity — honours the ~1870× span gap + 70/98% zero split). `C=6`
ordered abstract classes; synthetic vocab |V|=1050 in per-class banks (its OWN synthetic id/class
map — never the real concept map → safe-public). The three channels share `h_i` (genuine
confounding). Independent generative *maps* do **NOT** give independent *signals* (fable5 B3 —
`v_c·h` and `b·h` correlate through the shared `h`), so a method reading any other `h`-channel scores
spurious order-skill → certification REQUIRES the `R_{h⊥order}` orthogonality reference (§3).

- **Known ORDER.** Per event `r_j = μ_class(c_j) + λ_order·(v_{c_j}·h_i) + σ_r·η_j`; observed order =
  `argsort(r_j)`. The **content-prior-only order** = `argsort(μ_class)` (Bayes-optimal given only
  the class multiset) is a closed-form committed function. The oracle exposes per-pair
  `P(a≺b|state)` and `P(a≺b|multiset)` → the **instance-specific order-skill ceiling is KNOWN
  exactly**. `λ_order=0` ⇒ instance order ≡ content-prior ⇒ **zero** instance skill (a decoder
  claiming skill here is a false positive — this dissolves fable5's self-manufactured-ceiling: T4 is
  scored against the *known rule*, not a co-trained ceiling). Plugs into `order_targets.py` T1–T4.
- **Known TIMING (marked TPP with the Δt=0 mass).** Two-stage cluster process on wall-clock days:
  (a) distinct-timestamp arrivals via inhomogeneous-Poisson × Hawkes `Λ_i(t)=μ_rate(s)·exp(λ_time·(b·h))·[1+Σκe^{-(t-t_j)/ω}]`
  (`λ_time=0` ⇒ homogeneous Poisson at the marginal rate ⇒ zero conditional skill = null); (b)
  simultaneity multiplicity `M ~ ZTNB(mean=μ_M(s)·exp(λ_mult·(d·h)))` tuned so the realised Δt=0
  fraction hits **SCID≈0.70 / MIMIC≈0.98** (`λ_mult=0` ⇒ multiplicity ⟂ state = 4A conditional null;
  p₀≈marginal-zero-fraction is a freebie either way → 4A must key on the multiplicity conditional).
  Within a Δt=0 cluster events keep their `r_j` order (order stays meaningful; the timing head is
  denied it). This yields the exact **timestamp-cluster factorization** (inter-cluster time ·
  multiplicity · marks); raw Δt is generated but never embedded → the genuinely-external quantity.
- **Nuisance-channel correlation + support (NON-causal only — Pi #1: the counterfactual apparatus
  is REMOVED).** A **nuisance-channel correlation strength** knob `ζ_nuis` sets how strongly the
  non-order channels (timing/count) correlate with the order channel through the shared `h` — the
  substrate for the nuisance-incremental reference `R_nuis` (§3) and the correlated-leak stress
  cells. Designated **Σ-orthogonality cells** construct the order projection orthogonal to every
  allowed nuisance projection (there the nuisance reference must fail). A **support-density** knob
  controls multiset duplication so the fixed-multiset swap floor (500) is met on positive cells,
  plus a deliberately support-starved cell where the swap is *genuinely* `NOT_EVALUABLE`. **No
  action, propensity, potential outcomes, positivity, or treatment-effect knobs** — a counterfactual
  oracle is a separate future blueprint.

- **Knob grid (null→strong; NON-causal).** `λ_order/λ_time/λ_mult/λ_count ∈ {0,.25,.5,.75,1(,2)}`,
  `ζ_nuis`, `support_density` — swept from **0 (null)** to strong; `ALL_NULL` (any certification
  firing = false positive) + `ALL_STRONG` corners. **Nulls are INSTANCE-level** (per-instance
  `λ_order_i` from a mixture with a null atom, interleaved within one dataset — fable5 B1), never a
  detectable per-cell block. Each (source × cell × split) is an independent seeded dataset;
  train/dev/**sealed-cert** (the synthetic sealed split is runnable, one-shot, not selectable-on;
  ≥2 held-out structural meta-families).

- **Realism envelope acceptance test** (generator must-not-certify unless matched, per source):
  Δt=0 fraction |Δ|≤0.02; per-class freq TV ≤0.05; seq-length KS ≤0.05; positive-gap KS ≤0.05;
  occupancy/empty per rung-1 horizon |Δ|≤0.03; per-window count KS ≤0.05. A PASS is a recorded
  precondition. (`oracle_realism_test.py`, safe-public.)

## 3. The EVALUATOR — measure RECOVERY against ground truth (not skill-over-a-fitted-baseline)
The generator exposes an **eval-only label bundle** per instance (`π0/π*` order; `λ0/λ*` intensity;
`p0*` simultaneity; `κ_order/κ_time`; `conf_level`; `overlap_flag`; `rule_family`; both potential
outcomes). The method-under-test consumes ONLY observed `(token, Δt)` — identical to governed data.

**Certification metrics (per source×knob cell, cluster-bootstrap CIs; reuse `rung1_probes`):**
- ORDER — **E-O1** beyond-prior order skill over the **EXACT known content-prior** `π0` (not a fitted
  swap; certify >0); **E-O2** recovery calibration slope `β_ord∈[0.8,1.2]`, `|b|≤0.05` (certifies the
  *right magnitude*, not just presence); **E-O3** recovery R²/bias.
- TIMING — **E-T1** beyond-marginal CRPS-skill over the **exact** `λ0` (certify >0); **E-T2** intensity
  recovery calibration; **E-T3** multiplicity recovery (ECE ≤0.05 AND slope of `p0^M` on `p0*` in
  band — p₀ reliability alone cannot pass); **E-T4** positive-tail randomized-PIT KS upper-CI ≤0.05.
  4A(E-T3) ∧ 4B(E-T4) separate + conjunctive.

**Null + positive control battery (the heart of the certifier):**
- **KNOWN-NULL — INSTANCE-level, not cell-level (fable5 B1):** per-instance `λ_order_i` drawn from a
  mixture with a null atom, **null and positive instances interleaved within one dataset** (+
  camouflaged nulls matched to a positive cell's marginals + off-grid certification knobs) so the
  method must decide per-instance and cannot detect-and-suppress a null cell. Any "pass" on a null
  instance is a FALSE POSITIVE. `null FPR ≤ α=0.05` (upper-CI), per property, ≥200 null seeds.
- **KNOWN-POSITIVE (κ>0):** a "fail" where the oracle-informed reference succeeds + precision is
  adequate = `REFUTED`; else `NOT_EVALUABLE`. `power ≥ 0.80 at κ_mid` + report the full power curve +
  the **MDE** (smallest κ with power≥0.80), ≥100 positive seeds.
- **Monotonicity:** Spearman(κ, recovered signal) lower-CI ≥ 0.90.
- **Calibration:** slope ∈[0.8,1.2], |intercept|≤0.05 across the positive grid.
- OC estimates are themselves certified by an **evaluator precision sim** (reuse
  `rung1_precision_sim`: coverage ≥0.95, power ≥0.80), else NOT_EVALUABLE / raise seeds.

**Falsifier-of-the-falsifier (is the oracle a fair anchor?) — the THREE-reference bracket (fable5
B2/B3):**
- **`R_bayes` = context-Bayes ceiling `E[order | observable context]`** (constructible: `p(h|context)`
  is closed-form/MC from the known DGP) — **the FAIR ceiling**. `R_bayes` must **materially beat R0**
  for a cell to be a valid positive; if `R_bayes ≈ R0` the cell is a **HIDDEN NULL** (order lives in
  `h` but is not inferable from context) → **excluded from positive certification** (demanding power
  there is either impossible or gamed via the shared-`h` leak).
- **R0** content-prior-only floor (`π0/λ0` only) **MUST fail** positive + **pass** null; its realised
  FPR = `evaluator_realized_alpha ≤ 0.05`.
- **`R_{h⊥order}`** (given `h`, DENIED the order functional `v_c`, allowed the other `h`-channels)
  **MUST FAIL** positive — else certified "order-skill" is a spurious other-channel `h`-correlate;
  certified skill is orthogonalized against the other-channel-predictable component.
- **R\*** (given `h`) is DEMOTED to a secondary SNR/plumbing check (signal clears the noise floor) —
  it is near-tautological and does NOT establish fairness (that is `R_bayes`'s job).
Plus a realism-sensitivity sweep and the transfer caveat.

## 4. The T4 UNLOCK contract
A learned VQ order-target T4 is authorized for governed data only when, on **sealed-cert seeds of a
HELD-OUT rule family**, it passes ALL:
**PROPERTY-SPECIFIC (order-only; Pi #5): E-T1..4 (timing) do NOT gate this order target — the CT
head emits a SEPARATE timing-recovery manifest.** On the **sampled context-only `D(ẑ)`** (never
`D(z⁺)`; Pi #2):
- **U1_order_recovery** — E-O1∧E-O2 CI-positive, slopes in band, power ≥0.80 at κ_mid;
- **U2_null** — FPR ≤0.05 across instance-level nulls; rejects prediction of non-existent
  *context-predictable* residual order (Pi #3), **the catch for a codebook that writes order into
  codes while the context predictor has no signal** (fails positive/U2);
- **U3_monotone** — Spearman ≥0.90;
- **U4_nuisance_incremental (Pi #4)** — the recipe's INCREMENTAL skill over `R_nuis =
  E[order|allowed non-order channels]` clears the frozen CI margin, AND the nuisance reference fails
  in the Σ-orthogonality cells. (No `R_{h⊥order}`-must-fail; no counterfactual U5/potential-outcomes —
  removed.)
- **U6** bandwidth-fair (beat mean_embed-quantised-to-T4-bits AND a frozen-random-codebook at matched
  bits — the random-codebook control MUST pass null and FAIL positive; if it passes positive, the
  "win" is bandwidth → REFUTED).

## 5. Anti-tailoring (fable5 will scrutinize) — the oracle must not be tailorable to a dev-favoured codebook
1. **Freeze-before-codebook** — generator mechanism + knob grid + rule families + seed split +
   evaluator contract content-hashed (`oracle_mechanism_hash`) and git-committed BEFORE any T4
   codebook exists; manifest carries `oracle_frozen_before_codebook` + `codebook_postdates_oracle`
   (commit-ancestry proof); a codebook predating the freeze voids authorization.
2. **Held-out sealed-cert seeds** — all building/tuning on `dev`; the OC computed on `sealed_cert`
   ONLY, one-shot.
3. **Codebook-cannot-see-the-rule** — ground-truth labels reach the EVALUATOR only; a fail-hard
   `assert_labels_eval_only` guard (analogue of `DATASET:*` masking) enforces no
   `π*/λ*/p0*/κ/conf/overlap` channel touches training.
4. **STRUCTURALLY-DIVERSE rule families + leave-one-family-out (fable5 B5).** The count 3–5 is a red
   herring — all `h`-linear-readout families are ONE meta-family a projection-recovering method
   generalizes across trivially. Require ≥1 held-out family with **NO `h`-dependence** in the order
   channel (order driven by realized earlier events / an exogenous Markov chain, so "invert the
   linear-in-`h` map" cannot transfer), certify on **≥2 structurally-distinct** held-out families, and
   add an **`h`-projection shortcut probe** (a method that only recovers a linear projection of
   context onto `h` must NOT pass).
5. **Realism envelope** (§2) + **metric pre-registration** (evaluator statistic frozen via
   `config_hash` before any cert run — no dev-favoured metric selection).

## 6. Composition + authorization manifest + hardened guard
Flow: **dev NOMINATE (Rung 2) → oracle CERTIFY (this) → governed T4.** E-O1 is the oracle-grade
`PRECEDENCE_SKILL_GATE` (exact `π0` replaces the approximate swap); E-T1/E-T3/E-T4 are the
oracle-grade 4B/4A (exact `λ0` replaces the fitted marginal). The oracle emits an
**authorization manifest** (schema `clinical-jepa-oracle-authorization-v1`) carrying
`oracle_mechanism_hash`, `oracle_frozen`, `pi_gate`, `codebook_postdates_oracle`,
`labels_eval_only_verified`, `held_out_family`, the full OC block per property, the six
`unlock_checks`, the `reference_bounds` (**`R_bayes_beats_R0`**, R0 null-pass, R0 positive-fail,
**`R_h_perp_order_fails_positive`**, `evaluator_realized_alpha`), `precision_sim`, `realism_envelope`,
`verdict ∈ {synthetic_recovery_CERTIFIED, REFUTED, NOT_EVALUABLE}`, `governed_t4_real_output_ceiling
= "NOMINATE"`, and the `transfer_caveat`. **`t4_governed_allowed` is HARDENED (DONE, fable5 B4)** to
require, conjunctively: `verdict==synthetic_recovery_CERTIFIED` ∧ `codebook_postdates_oracle` ∧
`labels_eval_only_verified` ∧ all `unlock_checks==PASS` ∧ the reference bounds ∧
`precision_sim.adequate` ∧ `realism_envelope.within_envelope` — any missing field ⇒ fail-closed.

## 7. Module / CLI layout + compute + governance
`clinical_jepa/targets/oracle_spec.py` (frozen constants + `oracle_mechanism_hash`) ·
`oracle_generator.py` (numpy generator + eval-only labels) · `eval/oracle_realism_test.py` ·
`export_oracle_datasets.py` · `eval/oracle_contract.py` (OC constants + fail-hard guards
`assert_labels_eval_only`/`assert_frozen_before_codebook`/`assert_sealed_cert_disjoint` + hardened
`authorize_governed_t4`) · `eval/oracle_probes.py` (E-O/E-T estimators on `rung1_probes`) ·
`eval/oracle_reference_methods.py` (R*, R0) · `eval/oracle_gate.py`/`oracle_verdict.py` · configs
under `configs/oracle/`. **Compute:** generation = numpy/CPU ≪1 CPU-hr; evaluation ≪1 GPU-hr; the
real cost is T4 training on the oracle (~10–30 GPU-hr across families × knob grid) — the intended
gate. **Governance:** fully synthetic ⇒ generator/spec/hash/datasets are **safe-public/committable**;
sole governed touch = a one-time aggregate-marginal read for calibration targets. Aggregate-only
manifests (no per-instance rows / no synthetic sequences dumped — keep the governed discipline so the
pipeline transfers). Governed-safe simulator with declared assumptions: do NOT describe generated
futures as observed patient data nor the action operator as a treatment-effect estimate.

## Cog imports
- `[[synthesis/orca-external-prior-specification-tests]]` — raw Δt/order as external prior
  specifications; the oracle supplies the **exact** `π0`/`λ0` decomposition, converting Rung-2's
  *approximate* content/rate-matched controls into *exact* known baselines (NOMINATE→CERTIFY).
- `[[synthesis/orca-and-jepa-representation-space-translation]]` (predict the latent of the future) —
  the certified target is the contextualised T4 code; the oracle is where its self-manufactured
  ceiling is externally checked.
- Default *separate generator from verifier* — generator + evaluator designed by separate agents;
  hard-enforced as the label-isolation guard + the two-sided R*/R0 self-calibration.
- **Not followed for the order/timing core:** `[[concepts/target-trial-emulation]]` (no action
  contrast there) — it re-enters only via the confounding/overlap cells (U4/U5), where the oracle's
  known flags certify *abstention*, not a causal effect.

**Scout trigger:** SUGGEST a sanitized Cog scout on *"semi-synthetic marked-TPP EHR oracles with a
dominant Δt=0 point mass, known content-prior-vs-instance order decomposition, nuisance-incremental
references, and leave-one-structural-family-out anti-memorization as an external
certifier for learned VQ order targets"* — do not launch without authorisation; the oracle needs its
own Pi gate.

## Open questions — RULED by the fable5 pass (carried to Pi)
1. **Family structural diversity, not envelope width, is the binding axis** (B5): ≥1 no-`h`-dependence
   held-out family + ≥2 structurally-distinct held-out families + the `h`-projection shortcut probe.
2. **Fully-synthetic is right for its ONE narrow job** (falsifying the self-manufactured-ceiling /
   null-hallucination failure — unanswerable on real EHR). `synthetic_recovery_CERTIFIED` is
   NECESSARY-and-narrow, NOT near-meaningless, IFF the `R_bayes`/R0/`R_{h\u22a5order}` bracket is honest
   and the denotation is disciplined (real claims stay NOMINATE).
3. **U2 IS gameable by null-detection as originally drafted** -> fixed by instance-level knob
   randomization + camouflaged nulls + off-grid knobs (B1).
4. **R0 alone is NOT enough** (the shared-`h` leak) -> added `R_{h\u22a5order}` (B3); R* was circular ->
   `R_bayes` now holds the fairness role (B2).

## Residual open question for Pi
Is there ANY real-data certification route beyond NOMINATE once a governed T4 is trained (fable5:
possibly nonexistent-in-principle)? If not, state plainly that the arc's terminal real-data claim for
learned targets is NOMINATE + a synthetic-recovery-certified RECIPE, never a real fidelity certificate.
