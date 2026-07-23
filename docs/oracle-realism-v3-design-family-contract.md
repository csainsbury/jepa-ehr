# Oracle Realism Verifier — Design Family v3 (rev-7, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-6. Folds Pi rev-2..rev-5 rulings AND the Pi rev-6 implementation
gate (7 items). All work below is development-only, synthetic-only; no calibration build, reference-map draw,
audit/evaluation seed, policy population, or run — Pi authorized the dev work here but NOT any of those.

**Rev-7 changes folded (Pi rev-6 #1–#7):**
1. **Corrected exact-group demo (#1)** — `scripts/oracle_realism_v3_group_power.py` now runs through the engine
   with INDEPENDENT per-(profile,check) map-design samples (no leakage), `exp_id` in every fixture seed
   (independent experiments), the structural-zero 3 within-stratum quotas, honest sparse one-experiment /
   two-primary-cell attribution, all THREE active components, and tied-KS on the 8-dp-rounded support (§ Delivered).
2. **Registered-N boundary preflight (#2)** — `scripts/oracle_realism_v3_regn_preflight.py`: at N=8000 the bounded
   S3_loggap map ISSUES (evaluable) — the dev-scale "structurally un-calibratable" claim is WITHDRAWN; both S3
   exemptions now rest on the DETECTION criterion (bounded detect `0.0 < 0.5`), reported with/without each (§7).
3. **Issuance-complete map (#3)** — `oracle_realism_v3_map.py` binds a full trust root + validator that refuses
   cross-identity application; adequate-support tests prove exact reproduction (not `None==None`) (§9).
4. **Heterogeneous registry-owned dispatcher (#4)** — `oracle_realism_v3_engine.py`: cells bind to registry-owned
   estimators by check (callers pass DATA, never a statfn); mandatory per-map identity + floor-policy, exact cell
   ids/order, executable RNG identity, one trusted assignment path, real permutation-NE test; per-perm recompute
   reproduces the exact v2 estimand to <1e-9 (§3, §8).
5. **Registry identities re-minted (#5)** — S3_tau pilot suffix, S6 estimator text, and audit prose corrected;
   hashes re-minted (§ Delivered).
6. **Single canonical PASS rule + design-rule framing (#7)** — `p_g > α_group` everywhere; `B ≥ K/α_group` is a
   conservative sparse-effect DESIGN RULE, not a theorem — actual group power remains a required falsifier (§3).
7. Wired-engine benchmark with measured generation/serialization + conservative cap margin (#6) — §9.

**Rev-6 changes folded (Pi rev-5 #1–#7):**
1. **Corrected frozen-map builder (#1)** — `scripts/oracle_realism_v3_map.py` keeps the ORIGINAL registered bins,
   GROUPS them via the frozen v2 merge (`coarsen_reference`), preserves per-sequence summaries + equal-weight +
   floors, and fixes S6 to LENGTH_BINS. Reproduces the v2 estimand exactly; the Δ-aligned full-support S3_loggap
   null-exceedance drops **0.526 → 0.0** (§7, §9).
2. **p-value prose corrected to the symmetric code form (#2)** — `p_c^(j) = #{k∈0..B : e^(k) ≥ e^(j)}/(B+1)` (§3).
3. **NOT_EVALUABLE policy (#3)** — no zero-fill: observed cell NE → group NOT_EVALUABLE; permutation NE →
   maximally extreme; wired into the fail-closed gate (§7).
4. **Exact-group power demo (#4)** — the 36-cell burst-timing registry group, product strata, `B ≥ K_g/α_group`,
   Wilson CIs, both variants, tied-KS on unique support; the old 3-cell run is relabelled a mechanical smoke (§ Delivered).
5. **Coarse audit REMOVED (#5)** — level-valid but could not resolve a single-cell effect; main SD level suffices,
   mechanical/exhaustive tests retained (§4).
6. **Per-profile benchmark (#6)** — `Σ_experiment Σ_cell cost(route, profile volume)`; SCID heavier than MIMIC;
   deterministic config identity separate from timing; measured serialization; per-group job plan (§9).
7. **Fail-closed gate wiring (#7)** — `scripts/oracle_realism_v3_gate.py` validates BEFORE any statistic and
   refuses all enumerated malformed inputs; registry binds the coupled-role RNG law per experiment (§7, §8).

**Rev-5 changes folded (Pi rev-4 gate defects #1–#8):**
1. **Benchmark now executes + is route-weighted (#1/#2).** The prior forecast did not run (`NameError`) and
   averaged three surrogate routes; it now runs and is a route-weighted `Σ_route(cells × measured cost/perm)` +
   generation/serialization + a labelled MM upper-bound proxy. The distribution-comparison (KS) routes dominate
   (~92% of a permutation's cost); the honest whole-job cost is far above the earlier "~6h" (§9).
2. **Audit B declared honestly (#3).** The min-p resolution requirement is `B ≥ K_max/α_group` (= **8100** for the
   deepest group); main `B=20,000` meets it, audit `B=2,000` does NOT — so the audit is predeclared as a DISTINCT
   coarser screen (valid level at any B, less power), with an exact false-park proof; the job is split into
   separately-gated stop-on-failure jobs rather than weakening B to fit the cap (§3, §4, §9).
3. **Exact schema-validated registry (#4/#5).** Every cell binds estimator/required-property/direct-group_id/
   map+floor/RNG-law/exact-stratum-quota/expected-status/malformed-input/Δ-provenance and is schema-validated
   (missing/extra/mistyped REFUSE, proven). `Δ` is bound to the LIVE verifier constants with an exact-equality
   test (`S3_loggap` rounding corrected to `log(1.10)`); the Δ-table is hashed (§8, Δ table).
4. **Both boundary-exemption variants emitted (#6).** M0 = 192 with / 194 without the provisional `S3` exemption;
   the Δ-aligned boundary recompute uses the MM-aligned `P[d>Δ @0.5]` via the frozen map (§7).
5. **Real product/stratified finite-B randomization + refusals (#7).** The actual evaluator path (within-stratum
   + product permutation, nested ranks, deadbands/ties, deterministic replay) plus the seven malformed-input
   refusals are implemented and tested against the exhaustive universe (§3, §8).
6. **Frozen-map builder SPECIFIED, not executed (#8).** The reference-owned map schema/derivation/seed-namespace/
   floor/artifact-format/support-failure behaviour is frozen in §9; the one-time draw remains blocked.

## 0. Retained from v2 (works): N=8000, exact control allocation (2667,2667,2666), the three D components
{burst_timing, mark_burst_tie, cluster_size_mark_diversity}, fail-closed runner/exec, manifest deep-equality trust
root, gate-event provenance, integrity-bound resume, exact-count hash+gate, sequential stop-on-fail launch, frozen
schema/certification-boundary/support-policy identities.

## 1. Fresh-draw conditional randomization (Pi §1) — SD cells

Same-distribution (SD) cells (null, candidate-D repeatability, structural-zero, boundary-short) have exchangeable
candidate/reference sequences → an **exact conditional randomization test per draw**. Each fresh SD acceptance draw:
1. compute the observed group statistic `S_g`;
2. permute cand/ref labels **within the frozen exchangeability strata** (§8) on that same pooled draw, `B` times;
3. compute the conservative MC p-value `p_g = (1 + #{permuted S_g at least as extreme as observed}) / (B + 1)`;
4. accept group `g` iff `p_g > α_group` (strict; the single canonical PASS rule — §3, Pi rev-6 #7). Battery
   accepts iff every SD group accepts AND all MM requirements hold.

**Frozen (design identity), NOT draw-specific critical values:** the gate-cell registry, groups + exchangeability
strata, `Δ` table, estimator identities, statistic/tie rules, coarsening rule, `B`, RNG namespaces, `α` split, the
exact p-value inequality direction and inclusion conventions. Draw-specific permutation evidence (the `S_g` null,
`p_g`, per-cell `p_c` vector, argmin) is emitted per evaluation.

## 2. Honest multiplicity (Pi §2) — exact G, groups, M0

- `M0` = exact count of atomic in-scope SD cells (per §8 registry). It is **not** an error divisor.
- Cells are partitioned into `G` groups by `(support-regime × substantive-family)`, each with an exact cell list.
  Within a group of `K_g` weakly dependent cells the min-p null critical value is roughly `α_group/K_g` — grouping
  captures the real cross-cell dependence (via the shared permutation) but does not turn `M0` opportunities into
  `G`. `α_group = α_eval/G` is the ONLY Bonferroni step (over the handful of groups).
- **Group critical values and power are demonstrated under the actual joint permutation null** at design freeze
  (dev seeds), not asserted. No arbitrary merging of independent experiments to inflate thresholds; a group
  spanning independent experiments uses independent per-experiment label permutations within each synchronized MC
  replicate (§8). The rev-2 inconsistent numbers (`M0≈300`, `G≈5–10`, `5–8/group`) are replaced by the exact
  registry (§ Registry); `M0` and `G` follow from it.

## 3. Randomization p-value algorithm (Pi §3 + rev-3 §1 direction fix)

Directions are OPPOSITE at the cell and group levels (Pi rev-3 §1; formula corrected per Pi rev-5 #2 to match the
code exactly):
- **Per-cell, upper tail** (larger discrepancy ⇒ more extreme): with `e_c = (d_c − Δ_c)_+`, for each assignment
  `j ∈ {0..B}` (observed = index 0), the **symmetric** marginal rank counts `j` exactly once:
  **`p_c^(j) = #{k ∈ 0..B : e_c^(k) ≥ e_c^(j)} / (B + 1)`**. (The earlier prose `1 + #{b≠j} + [obs≥j]` double-counts
  for some `j`; for the observed assignment the symmetric form reduces to the usual plus-one expression.)
- **Group min-p, lower tail** (smaller min-p ⇒ more extreme): `S_g = min_c p_c`, and
  **`p_g = (1 + #{b : S_g^(b) ≤ S_g^obs}) / (B + 1)`** (`≤`, NOT `≥`).
- **Tie handling:** the inclusive `≥` (cell) and `≤` (group) as written; all `B+1` assignments symmetric.
- **PASS convention (frozen):** group `g` passes iff `p_g > α_group` (strict `>`).
- **Validation (required before any MC implementation):** exhaustive tiny-enumeration tests over ALL balanced
  candidate/reference label assignments (small `m`, small cell sets) confirm the direction, the nested-rank
  construction, and the tie handling exactly.
- **Synchronized permutations:** one permutation index drives all cells in a group in a replicate (preserves
  cross-cell dependence). For groups spanning independent experiments, independent per-experiment permutations are
  drawn under one synchronized MC index.
- **`B` sizing (exact resolution rule):** the min-p over `K_g` cells shares its floor `1/(B+1)` across ~`K_g`
  permutations (each cell's own extreme), so a single-cell effect resolves below `α_group` only when
  **`B ≥ K_max/α_group`**. With the deepest in-scope group `K_max = 54` and `α_group = 0.04/6`, this is
  **`B ≥ 8100`**. Main **`B = 20,000`** meets it. (The rev-5 coarse `B = 2,000` audit could NOT resolve a
  single-cell effect in the deepest group and is REMOVED — §4.) `B` is frozen by this rule, not a default.
  Fisher/Cauchy combiners are **OUT** for v3.
- **Real evaluator path (implemented + validated).** The product/stratified finite-`B` engine (within-stratum
  label permutation preserving each stratum's quota; independent per-experiment permutation under one synchronized
  MC index; nested cell upper-tail then group lower-tail min-p; deadbands and ties) reproduces the exhaustive
  group `p_g` exactly, stays conservative under a finite-`B` product null, and replays deterministically. All
  seven malformed-input classes (unequal candidate/reference size, wrong stratum quota, duplicate/missing pooled
  index, non-bijection, role-dependent RNG law, `B`/RNG mismatch, truncated cell vector) fail closed BEFORE any
  statistic is computed.

## 4. Error budget — main SD level only; the coarse audit is REMOVED (Pi rev-5 #5)

```
alpha_total_family = 0.05
alpha_main_SD      = 0.04        # main fresh SD draw; alpha_group = 0.04 / G ; unused margin 0.01
```

The rev-5 coarse audit (15 replicates, `B_audit = 2,000`, park ≥4) was **level-valid but functionally blind**:
for the deepest group (`K_max = 54`, `α_group ≈ 0.00667`) a sparse single-cell effect cannot resolve below the
group threshold at `B = 2,000` (`B ≥ K_max/α_group = 8100` is required — §3), so a `B = 2,000` screen cannot detect
the pathology it was meant to guard, and spending ~7.2 h on it is unjustified. Per Pi's preferred option the
**scientific audit gate is REMOVED**. The main conditional-randomization battery already has exact SD level
(`α_main_SD = 0.04 ≤ 0.05`, leaving unused margin); implementation soundness is covered by the **mechanical /
exhaustive tests** (randomization exhaustive-vs-MC equivalence, refusal tests, the fail-closed gate self-tests),
not a low-power re-run. No audit `B`, no audit namespace, no park-on-audit rule. (If a resolving audit were ever
wanted it would need `B ≥ 8100` per group and its own separately-gated jobs with an explicit power/cost case —
not adopted here.)

## 5. Mismatched-arm (MM) requirements — direct effect (Pi §5)

MM arms (candidate_A vs coupled reference) are NOT exchangeable → no permutation p-value. Use the frozen effect
discrepancy directly, over 25 fresh MM replicates:
- **Primary detection:** attributed cell detected iff `d_c > Δ_c`; require **≥20/25**.
- **Specificity:** non-attributed cell passes iff `d_c ≤ Δ_c`; require **≥24/25**.
- Only the predeclared S4↔S7 allowed-sensitive exemptions apply (mark_burst_tie exempts S7; CSMD exempts S4).
- **source_swap** is a separately frozen expected-FAIL negative control (must FAIL its nondegenerate set), not a
  null regime. Permutation diagnostics may be logged but are not the MM guarantee.

## 6. Frozen estimator (Pi §6, Phase-0 complete)

Phase-spanning capped, tie-corrected pooled Kendall tau-b for burst-timing (`T_pool`); exact cap formula
(`m≤6→all; else 6 quantile-spaced indices round(linspace(0,m−1,6))`), standard tau-b `n1`/`n2` (all x/y ties incl.
joint), pair-count-weighted pooling across within-sequence pairs. Frozen + evidenced in the committed pilot (hash
`db885b97…`): formula err vs scipy `0.0`; phase coverage mean-pos 0.086→0.500; concentration MIMIC 0.034→0.989 /
SCID 0.083→1.00; source-wise power full=1.00, bounded (non-Δ proxy) 0.125; S8 no cross-load. Other coarse-null
rank statistics adopt the same pool-within/synchronized-permute pattern (enumerated in the registry).

## 7. Effect deadband, scope, un-calibratable — exact semantics (Pi §7)

- **`e_c = (d_c − Δ_c)_+` is an OBSERVED-discrepancy deadband under the sharp-equality null.** Claimed narrowly:
  exact SD type-I control **when the two generators are identical**; practical effect screening via the observed
  deadband. It is NOT an exact equivalence test for every population discrepancy within `Δ_c` (exchangeability
  holds only at equality).
- **Un-calibratable** is defined by power against a meaningful alternative beyond `Δ_c`, NOT by null mass above
  `Δ_c` (conditional randomization can still control sharp-null rejection). The retain/exempt criterion is
  **Δ-aligned to the MM effect rule (Pi rev-3 §3):** `power = P_dev[ d_c > Δ_c | @0.5 alternative ]`, **retain iff
  power ≥ 0.5**. This 0.5 threshold is a v3 design choice informed by development evidence (not predeclared before
  the pilot). A property-specific support cell failing it is EXEMPT; a **core/full-support cell failing it PARKS**.
  The criterion therefore **cannot finalize an exemption until the numeric `Δ` table exists** (the earlier
  `P[d > matched-null p96]` proxy is not equivalent).
- **Per-statistic exemption status (Δ-aligned, Pi rev-4 #6 — each decided on `P[d>Δ @0.5]` via the frozen map):**
  the pilot now computes the MM-aligned criterion `P_dev[d_c > Δ_c | @0.5]` for BOTH S3 subchecks, using the
  frozen reference-owned coarsening for `S3_loggap` (NOT the earlier `P[d>null p96]` proxy, NOT the v2 adaptive
  PASS/FAIL). Dev evidence: on **bounded** support `S3_tau` detects at **0.15** and `S3_loggap` at **0.10** (both
  ≪ the 0.5 retain criterion → **both provisional-EXEMPT**); on **full** support both detect at **1.00** (checks
  work where supported; full-support burst-timing is RETAINED, non-exemptible). These remain **PROVISIONAL** — the
  final exemption is only confirmed after the reference-owned frozen-map CALIBRATION draw (still blocked). Both M0
  variants (with/without the provisional exemption: 192 / 194) are carried in the registry so the exemption cannot
  manufacture favourable group size or power.
  - *Caveat surfaced honestly:* the `P[d>Δ]` here is the DIRECT two-independent-draw discrepancy used ONLY for the
    boundary exemption decision — NOT the SD gate's type-I control (that is the permutation test). The elevated
    full-support `S3_loggap` null-exceedance under the direct rule is a dev-scale (N=3000, floor=200, max-over-bins)
    artifact and a flag for MM-specificity calibration at the registered N, not a boundary or SD-gate defect.
- The full `Δ` table + required-property matrix is the "Δ / required-property" deliverable (below); values are
  sourced from prior/first-principles before calibration and hashed, and preserve existing v2 practical-effect
  semantics unless a separately-justified change is identified.

## 8. Exact SD/MM registry, exchangeability strata, group structure (Pi §2, §8)

Machine-readable `GateCell` universe, frozen before calibration. Each cell binds:
```
cell_id, class(SD|MM), experiment_id, source/profile, arm/control_condition, statistic/subcheck,
support_regime, estimator_identity, required_property, group_id, delta_c, permutation_scheme,
exchangeability_stratum, scope(in|exempt-degenerate|exempt-uncalibratable), calibration_note
```
- **Each SD cell is a distinct actual experiment** with its own candidate/reference pair. Candidate-D repeatability
  differs by **component AND source** → 3 components × 2 sources = 6 repeatability experiments, each carrying all
  in-scope statistics. Null control = 2 sources. Structural-zero = 1. Boundary-short = 1. (M0 = Σ in-scope
  statistics over these experiments; enumerated exactly in the registry artifact.)
- **Permutation strata:** for controls built with fixed per-stratum quotas `(2667,2667,2666)`, labels are permuted
  **within matching length strata**, never across the pooled mixture (preserves the fixed marginal design).
  Groups spanning independent experiments use **independent label permutations per experiment** within each
  synchronized MC replicate. Group sizes and per-cell exchangeability strata are bound + proved in the registry.
- **Coupling exchangeability proof:** the active coupling construction is verified sequence-wise and identically
  distributed across candidate-D/reference roles (any sample-level role dependence would break exchangeability);
  proof recorded per repeatability experiment.
- Exemption permissions are encoded per cell; **core/full-support cells cannot be exempted** by the classifier.

## 9. Whole-battery permutation feasibility (Pi §9 + rev-3 §2/§6 corrections)

**Independent reference-owned frozen coarsening (Pi rev-3 §2 — replaces the rejected pooled-frozen idea).** The
coarsening/binning map per (profile/regime/check) is derived ONCE from a **separately-namespaced synthetic
reference-DESIGN draw**, frozen + hashed **before** any audit/evaluation, and applied IDENTICALLY to: SD
candidate/reference, every permutation assignment, the MM arms, and later candidate evaluation. Candidate data
never modify the map; a candidate/reference floor failure under the frozen map stays fail-closed `NOT_EVALUABLE`.
Because the map is fixed independently of the tested labels, (a) conditional randomization remains valid, (b) the
v2 anti-masking principle is preserved (candidate cannot influence the bins to hide tail collapse — the defect of a
pooled map), and (c) O(N)-per-permutation recomputation remains possible. The map-building draw is a
verifier-DEFINITION artifact (bind seed/profile/code/hash); it is NOT a transported null threshold and must not be
used to alter `Δ` or scope. **Generating this reference-design draw is NOT authorized by this review** (Pi rev-3 §7).

**Frozen reference-owned map builder — CORRECTED spec + built code (Pi rev-5 #1; the reserved DRAW stays blocked).**
The rev-4 pilot map was wrong (data-driven unique-value edges + pooled adjacency means, and S6 mis-labelled). The
corrected builder (`scripts/oracle_realism_v3_map.py`) preserves the REGISTERED estimand:
- **Original bins retained.** Map-carrying checks and their original bins: `LENGTH_BINS` for `{S1_density, S5_abs,
  S6_tv}` (Pi #1: S6 is length-binned, NOT class-coarsened), `CLUSTER_BINS` for `{S3_loggap, S7_abs}`.
- **Reference-owned GROUPING of original bin indices** via the frozen v2 merge (`coarsen_reference` on the
  reference per-bin sequence counts) — the candidate never influences the bins (anti-masking). It groups original
  bins; it does NOT invent new edges.
- **Registered per-sequence summary + EQUAL-WEIGHT pooling** (`_grouped`) preserved, with every denominator floor
  (sequence floor for all; the extra adjacent-pair floor for `S3_loggap`). Floor breach on either arm ⇒
  `NOT_EVALUABLE` (never zero-filled); a regime whose reference cannot reach the floor (bounded support) makes the
  check un-calibratable → the §7 exemption path.
- **Verified:** the self-test proves `apply_frozen_map` reproduces the v2 check `.value` EXACTLY when the frozen
  grouping equals the per-draw grouping, is candidate-independent, and enforces floors. Re-running the Δ-aligned
  boundary evidence with the corrected estimand drops the full-support `S3_loggap` null-exceedance **0.526 → 0.0**
  (confirming Pi #1: the earlier 0.526 was the wrong estimator, not a harmless artifact).
- **Seed namespace / artifact / hash.** The one-time draw uses a disjoint `map-design` namespace (registered
  `N = 8000`), binds `{check, bins_id, n_original_bins, groups, floor}` + a per-map + map-set hash, and is a
  verifier-DEFINITION artifact that must not alter `Δ`/scope. **The draw itself is NOT authorized** (dev fixtures
  only for now).

**Per-profile route-weighted benchmark (executable; Pi rev-5 #6) + NO audit (Pi rev-5 #5).**
`scripts/oracle_realism_v3_benchmark.py` maps every statistic to one of five recompute route classes and forecasts
`Σ_experiment Σ_cell cost(route, THAT experiment's profile/regime volume) · B_main + serialization + MM_proxy` —
route costs are per-item (profile-independent) but volumes are per-profile, so SCID-scale experiments are heavier
than MIMIC (the rev-5 "one MIMIC cost × M0" understated SCID). The **distribution-comparison (KS) routes dominate**
(≈ 92 % of a permutation's cost; `S9_gap` ≈ 56 ms, `S2_ks`/`positive_gap_ks` ≈ 14 ms). Consequences:
- **No audit term** (§4): the forecast is SD-main + MM only.
- **"SD-main fits one 8 h job" is NOT assumed** — the benchmark emits a **per-group job plan** (`per_group_job_hours`)
  and a `sd_main_fits_one_8h_job` flag from the per-profile sum; if it does not fit, run **separately-gated
  per-group SD jobs** (stop-on-failure) + a separate MM job.
- **Two identities:** a DETERMINISTIC `config_identity` (formula + routes + per-experiment routing + per-profile
  volumes + `B` — all seed-deterministic) and a separate TIMING artifact (route costs / hours, environment-
  dependent, NOT reproducible — Pi's benchmark-hash observation). **Serialization is measured**, not assumed.

**Preflight persistence:** per experiment/group report `sequences, events, clusters, positive-gaps, eligible
pairs, strata, cells, B, statistic recomputation route, wall time, peak RAM, checkpoint size`; aggregate-only
(per-cell/group permutation summaries + hashes; no sequence arrays).

## Deliverables status (rev-6 — all DONE + tested; see "Delivered" below for hashes)
1. Rev-6 contract — **this document**.
2. Exact schema-validated SD/MM registry (M0/G/groups/strata/proofs, both exemption variants) — **DONE**.
3. Δ / required-property / exemption table — **DONE** (Δ bound to live constants, exact-equality test, Δ-hash;
   boundary exemption Δ-aligned via the frozen map).
4. Exact min-p / product-stratified randomization p-value algorithm (strict `p_g > α_group`) + refusals — **DONE** (§3,§4).
5. Δ-aligned phase-spanning pooled-tau pilot (code/seeds/hash) — **DONE** (hash `db885b97…`).
6. Exact MM 20/25 & 24/25 logic — **DONE** (§5; registry MM cells).
7. Route-weighted whole-job benchmark + cap/checkpoint plan — **DONE** (executes; separately-gated jobs; §9).
8. Dev-seed group critical-value + power demonstration — **DONE** (calibrated null + demonstrated power).

## Revised phases (Pi's, adapted)
0. Estimator pilot — DONE. 1. Design freeze — this contract + registry artifact + Δ table + Pi ratification of the
reference-owned original-bin coarsening + G/group/Δ specifics. 2. Calibration-build preflight (benchmark/caps/
manifest; separately authorized). 3. Group-critical-value + power demonstration under the joint permutation null +
freeze the algorithm identity. 4. Fresh evaluation manifest review, empty policy, reproducing v3 hashes. 5.
Sequential evaluation, separately gated (power first; ident only on clean power PASS).

## Ratified by Pi rev-3 (structural elements accepted)
SD/MM split; fresh-draw conditional randomization (no transported critical values); MM direct effects
(≥20/25 primary, ≥24/25 specificity); `α_main_SD=0.04` (the rev-3 15-rep coarse audit is now REMOVED per Pi rev-5
#5 — see §4); conservative finite-`B` with observed assignment included; no Fisher/Cauchy; phase-spanning
pooled-tau CONCEPT + exact ties +
label-permutation-only + explicit equal-sequence replacement; stratum-preserving + independent product permutations;
"exact registry/Δ/property artifacts must precede implementation."

## Delivered (rev-7, Pi rev-6 authorized dev-only scope — all tested)
- **Registry (identities re-minted, Pi #5)** — `scripts/oracle_realism_v3_registry.py`: full per-cell identity +
  strict refusal; `Δ` bound to LIVE thresholds (Δ-hash **`ec6f4dff…`**). S3_tau identity DECOUPLED from the
  fragile whole-pilot aggregate hash to a stable frozen-estimator descriptor; S6 estimator text → LENGTH_BINS;
  audit prose removed; `B≥K/α` reframed as a conservative design rule. Both variants **M0 = 192 / 194**
  (hashes **`9a0ae6a8…` / `e978ecd0…`**); `G=6`, `K_max=54`.
- **Issuance-complete map builder (Pi #3)** — `scripts/oracle_realism_v3_map.py` (map-set hash **`5106ad09…`**):
  original bins + reference-owned `coarsen_reference` grouping + per-sequence means/equal-weight/floors (S6 on
  LENGTH_BINS); a FULL issuance trust root (profile/regime/namespace/seed/N/estimator+bin identity/denominator
  policy) + validator that REFUSES cross-identity application; adequate-support tests prove EXACT v2-value
  reproduction (not `None==None`) + malformed/floor/cross-identity refusals.
- **Registry-owned heterogeneous engine (Pi #4)** — `scripts/oracle_realism_v3_engine.py`: 10 estimators keyed by
  registered `check` (callers pass DATA, never a statfn); the hardened gate validates BEFORE any statistic
  (exact cell ids+order / mandatory per-map identity + floor-policy / alpha_group∈(0,1) / positive-int B / present
  seed / executable per-experiment RNG identity), constructs assignments through one trusted path (IID-with-
  replacement; duplicates valid+bound), and applies the NE policy. Each per-permutation recompute reproduces the
  EXACT v2 estimand to `<1e-9` (S3_tau is the v3 pooled-tau).
- **Registered-N boundary preflight (Pi #2)** — `scripts/oracle_realism_v3_regn_preflight.py` (hash **`9b4024f8…`**):
  at N=8000 the bounded S3_loggap map ISSUES (evaluable) — the dev-scale "structurally un-calibratable" claim is
  WITHDRAWN; both S3 exemptions rest on the DETECTION criterion (bounded detect **0.0 < 0.5**, full/SCID/MIMIC/
  structural-zero **1.0**), reported per regime with/without each independently. Both remain PROVISIONAL.
- **Corrected exact-group demo (Pi #1)** — `scripts/oracle_realism_v3_group_power.py` (hash **`79a6da45…`**), via
  the engine: INDEPENDENT per-(profile,check) map-design samples (no leakage), `exp_id` in every fixture seed,
  structural-zero 3 within-stratum quotas, tied-KS on 8-dp-rounded support, honest sparse one-experiment/two-primary
  attribution. Group-level sensitivity for ALL THREE active components (burst_timing via the burst-timing group;
  mark_burst_tie & cluster_size_mark_diversity via the class-mark group, argmin = the registered primary cell). Dev
  floor/N LABELLED. The rev-5 3-cell run is a MECHANICAL SMOKE (`oracle_realism_v3_group_demo.py`, `29da0411…`).
- **Δ-aligned pilot (dev evidence only)** — `scripts/oracle_realism_v3_phase0_pilot.py` (hash **`24ca0123…`**),
  formula err `0.0`; the boundary decision is DEFERRED to the registered-N preflight (Pi #2); full-support
  S3_loggap null-exceedance `0.0`.
- **Randomization** — `scripts/oracle_realism_v3_randomization.py`: MC == exhaustive `p_g`; conservative finite-`B`;
  7 refusals; strict `p_g > α_group` (Pi #7).
- **Per-profile + wired-engine benchmark (Pi #6)** — `scripts/oracle_realism_v3_benchmark.py`: per-profile
  `Σ_experiment Σ_cell cost(route, profile volume)·B_main`, DETERMINISTIC `config_identity` **`ef7a9280…`**
  separate from the timing artifact; **measured serialization + measured generation**; a **conservative 1.5× cap
  margin** (not merely <8h); a WIRED-ENGINE measurement of the burst-timing group's actual engine costs. Honest:
  SD-main ≈ 10.3 h does NOT fit one 8 h job → separately-gated per-group SD jobs. (Full per-group wired benchmark
  follows once all five groups' estimators are wired.)

## Still open (next, ONLY after this rev-7 package passes review + Pi separately authorizes)
- **Wire the remaining three full-support groups into the engine** — the engine (`oracle_realism_v3_engine.py`)
  binds the burst-timing (4) + class-mark (6) estimators; the length_density / run_size / phase_seam group
  estimators, the full per-group wired benchmark, and an all-component power demonstration at registered scale are
  the next wiring step.
- **Reference-owned frozen-map CALIBRATION draw + calibration-build preflight** — the corrected + issuance-complete
  builder is written and tested on dev fixtures; the one-time reserved-namespace DRAW is **NOT authorized** until
  Pi separately authorizes.

## Cog imports
- **Fixed-evaluator-before-keep/discard** (Cog): freeze the conditional-randomization *algorithm* (not a
  transported quantile) before fresh draws; the permutation null is exact per draw.
- **Generator/verifier separation** (Cog): calibration/coarsening artifacts are verifier-owned; the SD null uses
  only same-distribution exchangeability; candidate behaviour never chooses groups or exemptions.
- **Reward-hacking / stale-assumption cautions** (Cog): exact registry; no shrinking `G`, merging independent
  experiments, or exempting core cells; boundary-S3 exemption is a predeclared property-specific decision from dev
  evidence + a frozen power criterion; over-claim boundary retained (execution PASS qualifies only the synthetic
  verifier).
- **Sensitive-data boundary** (Cog): synthetic-only; the two external statistical consults used a fully abstracted,
  domain-stripped problem statement.
- Not followed: none. Permutation architecture + phase-spanning estimator imported from external statistical
  consult + standard exchangeability/rank theory (not in Cog distillation); flagged as candidate Cog feedback.

## Scout trigger
Resolved by the permutation architecture + Phase-0 pilot + the §9 feasibility benchmark. No scout needed for v3;
revisit only on a genuinely new fork. Launch only on Chris's explicit authorization regardless.

## Stop line
Design + (on Pi ratification) the registry artifact + calibration-build preflight only. No expensive evaluation
run, M3a freeze, M2, governed read, TEST, sealed certification, oracle-T4, policy population, or governed T4 until
Pi re-gates the frozen registry + benchmark.
