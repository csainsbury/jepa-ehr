---
title: Semi-synthetic clinical-EHR ORACLE — external-to-encoder synthetic SPECIFICATION TEST (v3, authoritative)
created: 2026-07-11
updated: 2026-07-12
status: DRAFT v3 (single authoritative contract; supersedes all earlier layered drafts — see Appendix A) — requires its own Pi gate before it may flip oracle_frozen / unlock governed T4
reporting: aggregate-only; fully synthetic (safe-public); a SEPARATE frozen calibration stage reads one-time aggregate real marginals
scope: a recipe FALSIFIER, NOT a real-EHR certificate — the only place a T4 recipe can be checked against KNOWN ground truth
---

# Semi-synthetic clinical-EHR oracle (v3)

Everything on real DEV is inside the encoder's own latent circle → can only **NOMINATE**. This oracle
is a **fully-synthetic mechanism with KNOWN ground truth**: it can falsify a learned target recipe
that hallucinates instance-order signal or succeeds only through a self-manufactured target ceiling.
It is the mandatory precondition (Pi) before any governed **T4** (learned VQ order-target) work;
`rung2_t4_stub.guard_t4` fail-closes until a frozen, Pi-gated, **recipe-bound** authorization manifest
exists. This is an **external-to-encoder synthetic SPECIFICATION TEST + recipe falsifier, NOT an
external certificate for real EHR.**

> **★ TRANSFER CAVEAT (load-bearing; required on every manifest).** `synthetic_recovery_CERTIFIED`
> means only: *"this frozen recipe recovered the planted **context-predictable** order/intensity
> mechanism under the declared synthetic families and nulls."* It authorizes a governed *experiment*
> and raises **no** real output above `NOMINATE`. It does NOT certify real-EHR instance structure,
> latent mechanism, counterfactuals, or causal effects.

**Real-data ladder (Pi, accepted):** DEV = `NOMINATE`; after the oracle = `NOMINATE +
synthetic-recovery-certified recipe`; only after a **separately pre-registered, one-shot LOCKED/
external/prospective observation-space evaluation** could a claim rise to **real held-out
OBSERVATIONAL PREDICTIVE fidelity confirmed** (held-out likelihood/proper scores, exact/count/order/
timing, calibration, coverage, copying/diversity controls vs strong AR/rate/content baselines) —
never latent-mechanism or causal. If no locked route is authorized, the terminal real-data claim is
exactly nomination + a synthetic-recovery-certified recipe.

## §1 The GENERATOR (fully synthetic; NON-causal only)
One committed latent state `h_i ∈ ℝ⁸ ~ N(m_{k_i},Σ_{k_i})` (phenotype `k_i`) drives BOTH the context
and the future window, so every certification baseline is a closed-form committed function of `h`,
defined outside any learned encoder. Source `s_i`, source block `Θ_s` (wall-clock scale, base rate,
Δt=0 mass, sparsity — honours the ~1870× span gap + 70/98% zero split). `C=6` abstract ordered
classes; synthetic vocab |V|=1050 in per-class banks (its own synthetic id/class map — never the real
concept map). **NO action, propensity, potential outcomes, positivity, treatment-effect knobs, or TTE
anything** — a counterfactual oracle is a separate future blueprint.

- **Known ORDER.** `r_j = μ_class(c_j) + λ_order·(v_{c_j}·h_i) + σ_r·η_j`; observed order = `argsort(r_j)`.
  `π0(a≺b) = P(a≺b | class multiset)` (the **content prior**, Bayes-optimal given only the multiset)
  and `π*(a≺b | h) ` are exposed to the evaluator. **Null (Pi #3):** `λ_order=0` does **NOT** imply
  realized order `≡ argsort(μ_class)` — the residual `σ_r·η` perturbs it irreducibly. The null is
  **zero context-PREDICTABLE residual order**: the Bayes pairwise probability given the observable
  context equals `π0`. Scoring is vs the **known context-Bayes pairwise probabilities**; realized
  order noise is explicit and is NOT a target to reconstruct.
- **Known TIMING (marked-cluster process).** Inter-cluster gap intensity
  `Λ_i(t)=μ_rate(s)·exp(λ_time·(b·h))·[1+Σκe^{-(t-t_j)/ω}]` (`λ_time=0` ⇒ homogeneous Poisson at the
  marginal rate ⇒ conditional-timing null). Simultaneity multiplicity `M ~ ZTNB(μ_M(s)·exp(λ_mult·(d·h)))`
  tuned to the realised Δt=0 fraction (SCID≈0.70 / MIMIC≈0.98); `λ_mult=0` ⇒ multiplicity ⟂ state
  (4A conditional null). The Δt=0 mass is **multiplicity**, the inter-cluster gap is strictly
  positive.
- **NON-causal nuisance + support only.** A **nuisance-correlation** knob `ζ_nuis` sets how strongly
  the timing/count channels correlate with the order channel through the shared `h` (the substrate
  for `R_nuis` and the correlated-leak cells). Designated **Σ-orthogonality cells** construct the
  order projection orthogonal to every allowed nuisance projection (there `R_nuis` MUST fail).
  **Correlated-leak stress cells** keep the channels correlated. A **support-density** knob controls
  multiset duplication so the fixed-multiset floor is met on positive cells, plus a support-starved
  cell where the swap is genuinely `NOT_EVALUABLE`.
- **Structural families + held-outs (Pi #8).** ≥3 train + **≥2 DISTINCT held-out structural
  meta-families**, including at least **one held-out family with NO `h`-dependence in the order
  channel** (order driven by realized earlier events / an exogenous Markov chain — so a method that
  only inverts the linear-in-`h` map cannot transfer). An **`h`-projection shortcut method** (recover
  only a linear projection of context onto `h`) must NOT exceed `ORACLE_SHORTCUT_MAX_SKILL=0.10`.
- **Instance-level nulls (Pi/fable5).** Per-INSTANCE `λ_order_i` from a mixture with a null atom,
  null and positive instances INTERLEAVED within one dataset + camouflaged nulls (marginals matched
  to a positive cell) + **off-grid** certification knobs `κ∈{0.15,0.6}` (not in the training grid).
  Cluster unit = **sequence** (FPR aggregated per sequence, not per precedence-pair).
- **Calibration SEPARATE from certification (Pi #9).** A dedicated calibration stage reads one-time
  aggregate real marginals with the fitting algorithm/tolerances frozen *before* reading; the
  realism envelope (Δt=0 fraction |Δ|≤0.02; per-class TV≤0.05; seq-len/gap/count KS≤0.05; occupancy
  |Δ|≤0.03) is an **ELIGIBILITY condition, not transfer evidence**. Certification families/seeds stay
  sealed. Record the governance classification of the aggregate targets (do not commit licensed/
  sensitive aggregates just because derived synthetic is safe-public).

## §2 The EVALUATOR (score prediction-achieved `D(ẑ)` context-only, NEVER `D(z⁺)`)
**Context-only predictor API (Pi #2, fail-hard).** Target encoder/codebook/decoder may train on
synthetic TRAIN futures; the **predictor receives observable synthetic context only**
(`assert_predictor_context_only` + future/label perturbation invariance). Sealed-cert scoring is on
**sampled `D(ẑ(context))`** vs the known context-Bayes quantities. **`D(z⁺)` is a reported
ceiling/attribution diagnostic only and can NEVER satisfy U1–U4/U6.** A fail-hard test: a target
encoder that perfectly writes realized order into codes while the context predictor has no signal
MUST fail U2/positive.

**Certification metrics (per cell; cluster=sequence bootstrap; on `D(ẑ)`):**
- ORDER — **E-O1** beyond-content-prior order skill over the EXACT `π0` (certify >0); **E-O2**
  recovery calibration slope ∈`ORACLE_CALIB_SLOPE_BAND`, |intercept|≤`ORACLE_CALIB_INTERCEPT_TOL`;
  **E-O3** recovery R²/bias.
- TIMING — **E-T1** beyond-marginal CRPS-skill over the exact `λ0` (certify >0); **E-T2** intensity
  recovery calibration; **E-T3** multiplicity recovery (ECE≤`GATE_4A_ECE` AND slope of `p0^M` on
  `p0*` in band); **E-T4** positive-tail randomized-PIT KS upper-CI ≤`GATE_4B_KS`. (Timing is a
  SEPARATE manifest — see §5.)

**The reference bracket (ONE place; Pi #2/#4):**
- **`R_bayes = E[order | observable context]`** (constructible: `p(h|context)` is closed-form/MC over
  `ORACLE_R_BAYES_MC_SEEDS` seeds, MC error ≤`ORACLE_R_BAYES_MC_TOL`) — the FAIR ceiling. `R_bayes`
  must beat `R0` by lower-CI ≥`ORACLE_R_BAYES_MARGIN`, else the cell is a **HIDDEN NULL** excluded
  from positive certification (`ORACLE_HIDDEN_NULL_RULE`).
- **`R0 = `** content-prior-only (`π0/λ0`): MUST fail positive, MUST pass null; realised
  `evaluator_realized_alpha` ≤ `ORACLE_NULL_ALPHA`.
- **`R_nuis = E[order | allowed non-order channels]`** (Pi #4, replaces the incoherent
  "`R_{h⊥order}` must fail"): the recipe's **INCREMENTAL skill over `R_nuis`** must clear lower-CI
  ≥`ORACLE_NUISANCE_MARGIN`; separately, `R_nuis` MUST fail in the Σ-orthogonality cells.
- **R\*** (given `h`) is a secondary SNR/plumbing diagnostic only — near-tautological, never a
  fairness bound.

**Null + positive OC battery.** KNOWN-NULL (instance-level): any pass is a FALSE POSITIVE →
`null FPR ≤ ORACLE_NULL_ALPHA` (upper-CI), ≥`ORACLE_N_NULL_SEEDS` null seeds. KNOWN-POSITIVE:
`power ≥ ORACLE_POWER_FLOOR` at κ_mid + report the power curve + MDE (`ORACLE_MDE_DEF`),
≥`ORACLE_N_POS_SEEDS` positive seeds. Monotonicity Spearman lower-CI ≥`ORACLE_MONO_SPEARMAN`.
Multiple-testing `ORACLE_MULTIPLE_TESTING`. All OC estimates certified adequate by the evaluator
precision sim (coverage ≥0.95, power ≥0.80) else `NOT_EVALUABLE`.

## §3 Frozen operating-characteristic constants (Pi #8 — the numeric pre-registration)
All defined in `clinical_jepa/eval/rung2_contract.py`, content-hashed:

| Constant | Value | Meaning / unit |
|---|---|---|
| `ORACLE_SCHEMA_VERSION` | `clinical-jepa-oracle-order-authorization-v3` | exact authorization schema |
| `ORACLE_R_BAYES_MARGIN` | 0.05 | R_bayes−R0 lower-CI (order-skill units) |
| `ORACLE_R_BAYES_MC_TOL` / `_MC_SEEDS` | 0.01 / 8 | R_bayes MC error tol / independent seeds |
| `ORACLE_NUISANCE_MARGIN` | 0.05 | recipe incremental skill over R_nuis (lower-CI) |
| `ORACLE_NULL_ALPHA` | 0.05 | per-property null FPR (upper-CI) |
| `ORACLE_POWER_FLOOR` | 0.80 | power at κ_mid |
| `ORACLE_MONO_SPEARMAN` | 0.90 | monotonicity Spearman lower-CI |
| `ORACLE_CALIB_SLOPE_BAND` / `_INTERCEPT_TOL` | (0.8,1.2) / 0.05 | recovery calibration |
| `ORACLE_N_NULL_SEEDS` / `_N_POS_SEEDS` | 200 / 100 | seeds per property |
| `ORACLE_N_HELDOUT_FAMILIES` | 2 | ≥2 DISTINCT held-out structural meta-families |
| `ORACLE_CLUSTER_UNIT` | `sequence` | FPR aggregated per sequence, not per pair |
| `ORACLE_MULTIPLE_TESTING` | `bonferroni_over_evaluable_cells` | correction |
| `ORACLE_OFFGRID_KAPPA` | (0.15, 0.6) | off-grid certification knobs |
| `ORACLE_MDE_DEF` | smallest κ with power≥0.80 | MDE definition |
| `ORACLE_HIDDEN_NULL_RULE` | R_bayes within margin of R0 → excluded | hidden-null handling |
| `ORACLE_SHORTCUT_MAX_SKILL` | 0.10 | h-projection shortcut must not exceed |

## §4 T4 order-unlock contract (property-specific; scored on `D(ẑ)`)
Governed T4 is an ORDER target → its authorization conjuncts ONLY order-relevant checks (Pi #5); the
continuous-time head emits a SEPARATE timing-recovery manifest and CANNOT veto order. On the
**sampled context-only `D(ẑ)`**, on **sealed-cert seeds of a HELD-OUT family**:
- **U1_order_recovery** — E-O1∧E-O2 CI-positive, slopes in band, power ≥`ORACLE_POWER_FLOOR` at κ_mid;
- **U2_null** — FPR ≤`ORACLE_NULL_ALPHA` over instance-level nulls; rejects prediction of non-existent
  *context-predictable* residual order (the catch for a codebook that writes order into codes while
  the context predictor has no signal);
- **U3_monotone** — Spearman ≥`ORACLE_MONO_SPEARMAN`;
- **U4_nuisance_incremental** — incremental skill over `R_nuis` ≥`ORACLE_NUISANCE_MARGIN`, AND
  `R_nuis` fails in the Σ-orthogonality cells;
- **U6_bandwidth_fair** — beat mean_embed-quantised-to-T4-bits AND a frozen-random-codebook at
  matched bits (the random-codebook control MUST pass null + FAIL positive; else the win is bandwidth).

**Recipe registry + sealed-seed rotation (Pi #6).** A finite recipe registry is pre-registered before
opening sealed-cert; each recipe is a `t4_recipe_hash` over {target-encoder/codebook/predictor/
decoder architectures, losses+weights, optimizer/schedule, bit budget, sampling/temperature, training
split, seeds/seed policy, evaluator commit}. A FAILED recipe cannot re-consume the same sealed seeds
→ a separately frozen certification split rotates (state machine: `registered → sealed_cert_assigned
→ {CERTIFIED | REFUTED} → seeds_retired`). Governed T4 must present the SAME `t4_recipe_hash`; any
material change ⇒ re-certification.

## §5 Authorization schema (v3, exact) + separate timing schema
The order-authorization manifest (`ORACLE_SCHEMA_VERSION`) carries, all MANDATORY and exactly checked
by `t4_governed_allowed`: `schema_version`, `oracle_mechanism_hash`, `blueprint_hash`,
`evaluator_commit`, `certified_recipe_hash`, `recipe_registry_id`, `sealed_cert_run_id`,
`gate_event_ref`, `held_out_family_ids` (≥2 distinct), `oracle_frozen=true`, `pi_gate="PASS"`,
`verdict="synthetic_recovery_CERTIFIED"`, `codebook_postdates_oracle=true`,
`labels_eval_only_verified=true`, `governed_t4_real_output_ceiling="NOMINATE"`, `transfer_caveat`,
`unlock_checks` (only `ORDER_UNLOCK_CHECKS`, all PASS), `precision_sim.adequate`,
`realism_envelope.within_envelope`, and `reference_bounds` (`R_bayes_beats_R0`, `R0_null_pass`,
`R0_positive_fail`, `nuisance_incremental_margin_ok`, `evaluator_realized_alpha ≤ ORACLE_NULL_ALPHA`).

**Guard fail-closed (Pi v3 defect fixed + tested):** for governed inputs the caller MUST supply a
non-empty `presented_recipe_hash` AND `expected_blueprint_hash`; **omission is a REFUSAL, not a
skipped check.** `schema_version` must equal the expected exactly; `held_out_family_ids` must have ≥2
DISTINCT entries; `governed_t4_real_output_ceiling` must be `NOMINATE` with a non-empty
`transfer_caveat`. The **timing-recovery** result is a separate `clinical-jepa-oracle-timing-recovery`
manifest (E-T1..4) that does NOT gate the order target.

## §6 Anti-tailoring + falsifier-of-the-falsifier
Freeze-before-codebook (mechanism/knob-grid/families/seed-split/evaluator content-hashed +
git-committed BEFORE any T4 codebook; `codebook_postdates_oracle` commit-ancestry proof);
held-out sealed-cert seeds (build/tune on `dev`, OC on `sealed_cert` one-shot); label isolation
(`assert_labels_eval_only` — no `π*/λ*/p0*/κ/nuisance/family` channel touches training); the ≥2
distinct held-out meta-families incl. a no-`h` family + the `h`-projection shortcut probe; realism
envelope = eligibility; metric pre-registration. The three-reference bracket (`R_bayes` fair ceiling;
`R0` floor; `R_nuis` nuisance-incremental) bounds the test from both sides; the evaluator precision
sim certifies its own FPR/power estimates.

## §7 Module / CLI layout
`clinical_jepa/targets/oracle_spec.py` (frozen constants + `oracle_mechanism_hash`) ·
`oracle_generator.py` (numpy generator: order/timing channels, nuisance+support, instance-level
nulls, Σ-orthogonal + correlated-leak cells, ≥2 held-out families incl. no-`h`; eval-only labels) ·
`eval/oracle_realism_test.py` (eligibility) · `eval/oracle_contract.py` (OC constants re-export +
`assert_predictor_context_only`, `assert_labels_eval_only`, recipe-registry state machine) ·
`eval/oracle_probes.py` (E-O/E-T estimators + **`R_bayes`, `R0`, `R_nuis`** + the h-projection
shortcut probe, on `rung1_probes`) · `eval/oracle_gate.py`/`oracle_verdict.py` (order-authorization +
separate timing-recovery manifests). Governance: fully synthetic ⇒ safe-public/committable;
aggregate-only manifests; the guard `rung2_contract.t4_governed_allowed` is the sole governed-T4
unlock.

## Cog imports
- `[[synthesis/orca-external-prior-specification-tests]]` — one unambiguous prior/reference contract
  (`π0/λ0` exact; `R_bayes`/`R_nuis` incremental) rather than obsolete + revised definitions.
- `[[synthesis/orca-and-jepa-representation-space-translation]]` — keeps prediction-achieved
  context→target recovery distinct from target-side encoding ceilings and from real-EHR claims.
- Default fail-closed / anti-tailoring — mandatory recipe/blueprint identities (not optional
  equality), sealed families, label isolation, property-specific authorization.
- **TTE is OUT OF SCOPE and fully removed** — a counterfactual oracle is a separate future blueprint.

**Scout trigger:** none needed for this consolidation (the search space is fixed); revisit scouting
only after the canonical v3 contract is internally consistent and implemented. No governed T4 is
unlocked by this draft.

---

## Appendix A — revision history (NON-OPERATIVE)
Superseded drafts, for provenance only — NOT part of the contract:
- **v1** — initial generator+evaluator design panel.
- **v2 (fable5 pass)** — added instance-level nulls, the `R_bayes` context-Bayes ceiling, structural
  family diversity, guard hardening, and the rename to `synthetic_recovery_CERTIFIED`. **Superseded
  claim:** the v2 "`R_{h⊥order}` MUST FAIL positive" reference is REPLACED by `R_nuis` nuisance-
  incremental recovery (§2) — do not use the must-fail formulation.
- **v3 (Pi oracle-gate REVISE)** — this document: counterfactual apparatus fully removed;
  prediction-achieved `D(ẑ)` context-only gating; null = zero context-predictable residual; the
  numeric OC frozen (§3); property-specific order unlock; exact recipe/blueprint binding + a
  fail-closed guard; separate order/timing schemas.
