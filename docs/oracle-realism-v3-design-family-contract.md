# Oracle Realism Verifier — Design Family v3 (rev-5, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-4. Folds Pi rev-2 (§1–§9), rev-3 structure ruling, AND the Pi
rev-4 deliverable gate (8 defects). All work below is development-only, synthetic-only; no calibration build,
reference-map draw, audit/evaluation seed, policy population, or run — Pi rev-4 authorized the dev work here but
NOT any of those. Written before any calibration/audit/evaluation seed.

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
4. accept group `g` iff `p_g ≥ α_group`. Battery accepts iff every SD group accepts AND all MM requirements hold.

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

Directions are OPPOSITE at the cell and group levels (Pi rev-3 §1):
- **Per-cell, upper tail** (larger discrepancy ⇒ more extreme): with `e_c = (d_c − Δ_c)_+`, for each assignment
  `j ∈ {obs, 1..B}` (observed = index 0), `p_c^(j) = (1 + #{b ≠ j : e_c^(b) ≥ e_c^(j)} + [e_c^(obs) ≥ e_c^(j)]) /
  (B + 1)`, treating all `B+1` assignments symmetrically when building the nested ranks.
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
  **`B ≥ 8100`**. Main **`B = 20,000`** meets it; the coarse audit **`B = 2,000`** does NOT (a distinct coarser
  evaluator — §4). `B` is frozen by this rule, not a default. Fisher/Cauchy combiners are **OUT** for v3.
- **Real evaluator path (implemented + validated).** The product/stratified finite-`B` engine (within-stratum
  label permutation preserving each stratum's quota; independent per-experiment permutation under one synchronized
  MC index; nested cell upper-tail then group lower-tail min-p; deadbands and ties) reproduces the exhaustive
  group `p_g` exactly, stays conservative under a finite-`B` product null, and replays deterministically. All
  seven malformed-input classes (unequal candidate/reference size, wrong stratum quota, duplicate/missing pooled
  index, non-bijection, role-dependent RNG law, `B`/RNG mismatch, truncated cell vector) fail closed BEFORE any
  statistic is computed.

## 4. Error split + independent audit stopping rule (Pi §4)

```
alpha_total_family = 0.05
alpha_main_SD      = 0.04        # main fresh SD draw; alpha_group = 0.04 / G
audit_budget       = 0.01        # independent audit stopping rule (NOT a B CI)
audit_replicates   = 15
audit_park_rule    = park if >= 4 of 15 complete fresh SD batteries reject
```

If each valid SD battery rejects a true null with probability ≤ `alpha_main_SD = 0.04`,
`P(Binom(15, 0.04) ≥ 4) ≈ 0.002454 ≤ 0.01` (exact). Main draw + audit ≤ 0.05 total false-park by union bound. The
audit is a **gross implementation/seed-pathology guard**, not a tail estimate; a miss parks with no retuning.
Audit seeds are a disjoint namespace (§8).

**`B_main` vs `B_audit` — declared honestly (Pi rev-4 #3).** The main SD draw uses `B_main = 20,000`, which meets
the resolution rule `B ≥ K_max/α_group = 8100` (§3). The audit uses `B_audit = 2,000`, which does NOT meet it: the
plus-one randomization test is level-controlled at ANY `B` (a smaller `B` costs power/resolution, not level), so
`B_audit = 2,000` is a **valid but distinctly coarser** pathology screen — predeclared as such, with its own
`B`, and NOT a silently-weakened main evaluator. The false-park proof above holds for the coarse screen because
per-rep false-reject stays ≤ `α`. Because the exact route-weighted forecast (§9) exceeds the 8 h cap as one job,
SD-main / SD-audit / MM run as **separately-gated stop-on-failure jobs**, not a weakened single job.

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
`86dd8ab5…`): formula err vs scipy `0.0`; phase coverage mean-pos 0.086→0.500; concentration MIMIC 0.034→0.989 /
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

**Frozen reference-owned map builder — SPECIFICATION (Pi rev-4 #8; specified, NOT executed).** The one-time
coarsening-map draw stays blocked; its identity is frozen here so that (a) it can be reviewed before any draw and
(b) the Δ-aligned boundary evidence (§7) uses the same algorithm. Map-carrying checks (proved by executable
evidence — reference coarsening in the verifier detail) are `{S1_density, S5_abs, S7_abs, S6_tv, S3_loggap}`.
- **Derivation.** For each `(check, source_profile, support_regime)`, draw ONE synthetic reference-design sample
  in a disjoint `map-design` namespace; coarsen the reference bins (length bins for `S1_density/S5_abs`; cluster-
  size bins for `S3_loggap/S7_abs`; class coarsening for `S6_tv`) by ascending greedy merge until every bin holds
  ≥ the registered floor (`ORACLE_ENV_MIN_DENOM = 500`) adjacencies/sequences in the reference; freeze the bin
  edges. The candidate never influences the bins (anti-masking).
- **Seed namespace + N.** `map-design` seed namespace, disjoint from estimator-dev / locked-calibration / audit /
  evaluation; registered `N = 8000` per profile. Bind `(seed_namespace, profile, regime, check, code_version)`.
- **Floor / tie / support-failure.** Floor breach on either arm in any frozen bin ⇒ `NOT_EVALUABLE` (never
  zero-filled); ties resolved by the standard bin assignment (first closing edge ≥ value); if a frozen map yields
  inadequate candidate OR reference support at evaluation, the cell is `NOT_EVALUABLE` (fail-closed), never
  re-binned. A regime whose reference cannot reach the floor at all (e.g. bounded support collapsing to one bin)
  makes its conditional check un-calibratable → the §7 exemption path, not a re-derived map.
- **Artifact / hash.** Emit `{check, profile, regime, bin_edges, floor, seed_namespace, N, code_version}` and a
  per-map hash + a map-set hash; the frozen set is bound into the verifier identity before audit/evaluation. The
  draw is a verifier-DEFINITION artifact and must NOT alter `Δ` or scope. **The draw itself is NOT authorized.**

**Route-weighted benchmark (executable, committed harness; Pi rev-4 #1/#2).** Every registered statistic maps to
one of five per-permutation recompute route classes; each is measured at the registered `N` and true operating
volume, and the whole job is `(B_main + audits·B_audit)·Σ_inscope_cells cost_perm(route(cell)) + generation/
precompute + serialization + MM_proxy`. Findings (dev, N=8000/side, aggregate-hashed):
- **Naive** whole-battery recompute ≈ **17.6 s/perm** → INFEASIBLE.
- Efficient O(volume) route costs: `tau_pooled` ≈ **0.25 ms/perm** (~0.45 ms at the label unit, NOT µs);
  `tau_source` ≈ 1.3 ms; `marginal`/`frozen_map` ≈ 0.35–0.42 ms; and the **distribution-comparison (KS) routes
  dominate** at the large event/cluster/gap volumes — `S9_gap` ≈ **56 ms/perm**, `S2_ks`/`positive_gap_ks` ≈
  **14 ms/perm** each. **KS routes are ≈ 92 % of one permutation's cost.**
- **Whole-job:** a single combined job is **≈ 16–81 h** (audit at coarse `B=2,000` vs same `B=20,000`) — the
  earlier "~6 h" was the flawed surrogate-average. But **separately-gated stop-on-failure jobs each fit the 8 h
  cap**: SD-main (`B=20,000`) ≈ **4.8 h**, audit (coarse `B=2,000`) ≈ **7.2 h**, MM (v2 upper-bound proxy) ≈
  **4.9 h**. (An alternative lever is optimising the dominant KS routes; flagged, not pursued.) Compute is driven
  by `M0·B`, NOT by `G` (`G` only sets `α_group`).

**Preflight persistence:** per experiment/group report `sequences, events, clusters, positive-gaps, eligible
pairs, strata, cells, B, statistic recomputation route, wall time, peak RAM, checkpoint size`; aggregate-only
(per-cell/group permutation summaries + hashes; no sequence arrays).

## Deliverables status (rev-5 — all DONE + tested; see "Delivered" below for hashes)
1. Rev-5 contract — **this document**.
2. Exact schema-validated SD/MM registry (M0/G/groups/strata/proofs, both exemption variants) — **DONE**.
3. Δ / required-property / exemption table — **DONE** (Δ bound to live constants, exact-equality test, Δ-hash;
   boundary exemption Δ-aligned via the frozen map).
4. Exact min-p / product-stratified randomization p-value algorithm + α/audit split + refusals — **DONE** (§3,§4).
5. Δ-aligned phase-spanning pooled-tau pilot (code/seeds/hash) — **DONE** (hash `86dd8ab5…`).
6. Exact MM 20/25 & 24/25 logic — **DONE** (§5; registry MM cells).
7. Route-weighted whole-job benchmark + cap/checkpoint plan — **DONE** (executes; separately-gated jobs; §9).
8. Dev-seed group critical-value + power demonstration — **DONE** (calibrated null + demonstrated power).

## Revised phases (Pi's, adapted)
0. Estimator pilot — DONE. 1. Design freeze — this contract + registry artifact + Δ table + Pi ratification of the
frozen-pooled-coarsening requirement and the G/group/Δ specifics. 2. Calibration-build preflight (benchmark/caps/
manifest; separately authorized). 3. Group-critical-value + power demonstration under the joint permutation null +
freeze the algorithm identity. 4. Fresh evaluation manifest review, empty policy, reproducing v3 hashes. 5.
Sequential evaluation, separately gated (power first; ident only on clean power PASS).

## Ratified by Pi rev-3 (structural elements accepted)
SD/MM split; fresh-draw conditional randomization (no transported critical values); MM direct effects
(≥20/25 primary, ≥24/25 specificity); `α_main_SD=0.04` + independent 15-rep audit (park ≥4, ≈0.00245); conservative
finite-`B` with observed assignment included; no Fisher/Cauchy; phase-spanning pooled-tau CONCEPT + exact ties +
label-permutation-only + explicit equal-sequence replacement; stratum-preserving + independent product permutations;
"exact registry/Δ/property artifacts must precede implementation."

## Delivered (rev-5, Pi rev-4 authorized dev-only scope — all tested)
- **Exact schema-validated SD/MM registry** — `scripts/oracle_realism_v3_registry.py`. Every cell binds estimator
  identity / required-property / DIRECT group_id / map+floor identity / RNG-law / exact stratum quotas /
  expected-status / malformed-input behaviour / per-Δ provenance, and is **schema-validated** (missing/extra/
  mistyped fields REFUSE — proven by a baked-in negative test). `Δ` is bound to the LIVE verifier thresholds with
  an EXACT float-equality self-test (`S3_loggap` corrected to `log(1.10)`), Δ-table hash **`ec6f4dff…`**.
  **BOTH boundary-exemption variants** are emitted: **M0 = 192** (with provisional `S3` exemption, hash
  **`d547efb2…`**) / **M0 = 194** (without, hash **`2e21ce00…`**). **G = 6** groups (5 full-support family groups
  product-permuted across the 9 full experiments + 1 bounded group); `α_group = 0.04/6 = 0.00667`; deepest group
  `K_max = 54` ⇒ resolution `B ≥ K_max/α_group = 8100`. 117 MM cells. Self-tests: partition `Σ K_g = M0`,
  reachability, core non-exemptible, map-carrying set by executable evidence, source-swap + S4↔S7 matching the
  executable ABLATION_MATRIX.
- **Δ-aligned Phase-0 pilot** — `scripts/oracle_realism_v3_phase0_pilot.py` (hash **`86dd8ab5…`**), formula error
  vs scipy `0.0`. The boundary exemption is now decided on `P_dev[d_c>Δ_c | @0.5]` via the frozen map (§7):
  bounded detection `S3_tau = 0.15`, `S3_loggap = 0.10` (both ≪ 0.5 → provisional-EXEMPT); full-support `1.00`.
- **Product/stratified randomization engine + refusals + exhaustive validation** —
  `scripts/oracle_realism_v3_randomization.py`: MC-with-full-enumeration reproduces the exhaustive `p_g` exactly;
  stratified quotas preserved; finite-`B` product null conservative; deterministic replay; all **seven** malformed-
  input classes fail closed; exact-null ≤ α and correct tail directions retained.
- **Route-weighted benchmark + whole-job forecast** — `scripts/oracle_realism_v3_benchmark.py` (hash
  **`798ae250…`**), executes cleanly (the `NameError` is fixed); KS routes ≈ 92 % of cost; single job 16–81 h,
  separately-gated jobs each < 8 h (§9); exact audit false-park `P(Binom(15,0.04)≥4) = 0.00245`.
- **Dev-seed group critical-value + power demonstration** — `scripts/oracle_realism_v3_group_demo.py`
  (hash `29da0411…`): under the same-distribution null the grouped min-p gate is calibrated (size `0.0` at both
  `0.05` and `α_group`); with a single burst-timing cell coupled at 0.5 the group detects at `α_group` with rate
  `1.00` (power demonstrated — grouping does not dissolve a real single-cell effect); a two-experiment PRODUCT
  permutation is likewise calibrated (`0.0`); group sizes reported for both exemption variants (bounded 12 / 14).

## Still open (next, ONLY after this rev-5 package passes review + Pi separately authorizes)
- **Runtime exchangeability-refusal checks wired into the gate implementation** (the refusal logic is implemented
  and tested in the randomization module; wiring it into the production gate path is the gate-build step).
- **Reference-owned frozen-map CALIBRATION draw + calibration-build preflight** — SPECIFIED (§9) but the draw is
  **NOT authorized**; it stays blocked until `Δ`/exemptions/routes/`B` are frozen and Pi separately authorizes.

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
