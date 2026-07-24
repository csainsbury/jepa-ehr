# Oracle Realism Verifier — Design Family v3 (rev-9, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-8. The NORMATIVE design is §0–§9 + Delivered/Still-open below; the
**Changelog** records how each Pi ruling was folded. All work is development-only, synthetic-only; no calibration
build, reserved map-design draw, audit/evaluation seed, policy population, or run. There is **no production/trusted
readiness claim** — the registered boundary is DEFINED and its refusals are tested, but a real registered run is
BLOCKED (reserved map-set + RNG manifest not drawn).

**Rev-9 changes folded (Pi rev-8 #1–#9):**
1. **Structured strata (#1)** — the engine accepts `experiment → stratum_id → {candidate, reference}`, compares
   counts to the canonical registry, and concatenates in canonical order. The registered structural-zero
   `(2667,2667,2666)` now assembles without the flat-pool divisibility refusal (a synthetic-registered assembly is
   in the engine self-test).
2. **Explicit dev vs registered boundaries (#2)** — `gate_group_dev` (explicit hash-bound floor/B, no module-global
   mutation) and `gate_group_registered` (enforces exact N/arm, `B=20000`, floor 500, EXACT `α=0.04/6`, exact
   stratum quotas, the full registry identity, and the approved map-set + RNG-manifest identities — refusing ANY
   deviation, and BLOCKING the real run until the reserved manifests are drawn). Floor is a parameter, not a global.
3. **Map context binding (#3)** — dev mode binds each map artifact's profile/regime/dev-floor to the canonical cell
   and enforces ONE shared identity per `(profile, regime, check)`; the engine no longer self-derives expected map
   hashes from the supplied artifacts.
4. **Map validator (#4)** — mandatory positive non-bool N/seed/floor; nonempty profile/regime/namespace; groups
   must be ORDERED CONTIGUOUS nonempty disjoint covering `0..n-1` (a v2 merge yields ordered contiguous runs, not
   any partition); the builder verifies `len(reference_sample)==N`. Reproduced acceptances (None seed/N, blank
   profile, non-contiguous groups, empty group) now refuse.
5. **Paired direction (#7)** — the demo evaluates base and its one-experiment perturbation under the SAME
   permutation seed and compares `p_g(perturbed)` to `p_g(base)`.
6. **Canonical controls (#3/#8)** — structural-zero data AND map-design sample use the canonical multiscale
   constructor.
7. **Contract (#9)** — the superseded DONE list is removed; the readiness/provenance overclaims are withdrawn (this
   round separates dev-dispatcher wiring from a registered trusted boundary).
8. **Still open (Pi-authorized, NOT done):** wire length_density/run_size/phase_seam; the generate-only RNG +
   reserved map-set manifests (a real registered run stays blocked); the full `B=20000` per-group benchmark;
   boundary-short canonical map; estimator/precompute/raw-record schemas.

**Rev-8 changes folded (Pi rev-7 #1–#9):**
1. **Trusted engine boundary (#4)** — `scripts/oracle_realism_v3_engine.py::gate_group_trusted` loads the canonical
   group registry INTERNALLY, computes precompute ITSELF from raw pools, and treats any non-finite discrepancy as
   NE (never zero-fill). Reproduced fail-open exploits (all-NaN precompute → PASS; wrong check/Δ → PASS) now
   NE/refuse (adversarial self-tests). No caller-supplied check/Δ/registered/precompute is trusted.
2. **Sparse perturbation (#1)** — the demo perturbs EXACTLY one experiment and asserts (executable, hashed) that
   every other experiment pool is content-identical to its null construction; argmin reported + attributes to the
   perturbed experiment's registered primary cell. The dense-9-experiment claim is withdrawn.
3. **Canonical structural-zero (#2/#3)** — demo + preflight use the registered multiscale control constructor
   (means 18/60/250), not three same-profile draws.
4. **Map provenance + validator (#6)** — artifacts record the EXACT map-design seed/namespace actually used; the
   validator enforces registered estimator/bin/denominator identities, floor-consistent denom policy, positive
   non-bool N/seed/floor, status↔groups consistency, and a complete disjoint bin partition.
5. **Registered B (#3)** — the dev demo B is LABELLED a mechanical development test, distinct from the registered
   `B_main=20000` evaluator.
6. **Contract (#9)** — one normative section + this changelog; the stale §7 pilot detections and the §3 "exact
   resolution rule" wording are corrected.
7. **Still open (Pi-authorized, not yet done):** wire the length_density/run_size/phase_seam estimators (#2 wiring);
   generate—not execute—the reserved map-set manifest (#5); benchmark all wired groups at `B=20000` per exact
   profile/stratum with margin/checkpoint/persistence (#6/#8); executable RNG-provenance manifest (#5).

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

**Changelog — earlier rulings folded (condensed; the normative design is §0–§9):**
- **Rev-6 (Pi rev-5):** corrected reference-owned frozen-map builder (original bins + `coarsen_reference` grouping,
  S6→LENGTH_BINS; full-support S3_loggap null-exceedance 0.526→0.0); symmetric p-value prose; NOT_EVALUABLE policy;
  exact-group demo; coarse audit REMOVED; per-profile benchmark; coupled-role RNG law bound per experiment.
- **Rev-5 (Pi rev-4):** benchmark executes + route-weighted (KS ~92% of a permutation; whole-job ≫ "~6h"); exact
  schema-validated registry with Δ bound to live constants (`S3_loggap=log(1.10)`) + Δ-hash; both boundary-exemption
  variants (M0 192/194); product/stratified finite-B randomization + 7 refusals vs the exhaustive universe.
- **Rev-2..rev-4 (Pi):** SD fresh-draw conditional randomization + MM direct-effect split; honest multiplicity
  (min-p over `K_g`); phase-spanning pooled-tau estimator; α split; exact registry/Δ before implementation.

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
- **`B` sizing (conservative sparse-effect DESIGN RULE, not an exact theorem — Pi rev-6 #7):** the min-p over `K_g` cells shares its floor `1/(B+1)` across ~`K_g`
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
- **Per-statistic exemption status (decided at REGISTERED N=8000, Pi rev-6 #2 + rev-7 #2/#7):** the boundary
  exemption is decided by the registered-N preflight (`scripts/oracle_realism_v3_regn_preflight.py`), NOT the
  dev-scale (N=3000) pilot. At N=8000 the **bounded `S3_loggap` map ISSUES (evaluable)** — the earlier "structurally
  un-calibratable" rationale is **WITHDRAWN**. Both S3 subchecks are instead exempt on the **DETECTION criterion**:
  registered-N bounded detection `P[d>Δ @0.5] = 0.0 < 0.5` for both, while full/SCID/MIMIC/structural-zero detect at
  `1.0` (checks work where supported; full-support burst-timing RETAINED, non-exemptible). Both exemptions remain
  **PROVISIONAL** and are reported **with AND without each independently**; both M0 variants (192 / 194) are carried
  so the exemption cannot manufacture favourable group size/power. **Finalisation requires a separately-authorized,
  freshly-namespaced calibration/power battery using the frozen map (Pi rev-7 #7)** — the preflight is development
  evidence only and does not meet a final power rule. The dev pilot is retained as supporting dev evidence and
  explicitly DEFERS the decision to the preflight.
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

## Delivered (rev-9, Pi rev-8 authorized dev-only scope — all tested)
- **Registry (identities re-minted, Pi #5)** — `scripts/oracle_realism_v3_registry.py`: full per-cell identity +
  strict refusal; `Δ` bound to LIVE thresholds (Δ-hash **`ec6f4dff…`**). S3_tau identity DECOUPLED from the
  fragile whole-pilot aggregate hash to a stable frozen-estimator descriptor; S6 estimator text → LENGTH_BINS;
  audit prose removed; `B≥K/α` reframed as a conservative design rule. Both variants **M0 = 192 / 194**
  (hashes **`9a0ae6a8…` / `e978ecd0…`**); `G=6`, `K_max=54`.
- **Issuance-complete map builder + HARDENED validator (Pi #3/#6)** — `scripts/oracle_realism_v3_map.py` (map-set
  hash **`5106ad09…`**): original bins + reference-owned `coarsen_reference` grouping + per-sequence means/equal-
  weight/floors (S6 on LENGTH_BINS). Full issuance trust root; the validator now enforces registered estimator/bin/
  n_original_bins identities, a **floor-consistent denominator policy** (a dev-floor map cannot record the registered
  floor), positive non-bool N/seed/floor, **status↔groups consistency**, and a **complete disjoint bin partition**;
  it REFUSES cross-identity application. Callers record the EXACT map-design seed/namespace actually used. Adequate-
  support tests prove EXACT v2-value reproduction (not `None==None`).
- **Registry-owned engine + DEV/REGISTERED boundaries (Pi rev-7 #4, rev-8 #1/#2/#3)** —
  `scripts/oracle_realism_v3_engine.py`: cells are built from the canonical registry ONLY (callers pass DATA as
  STRUCTURED arms `experiment → stratum → {candidate, reference}`, never check/Δ/registered/precompute); precompute
  is computed internally; non-finite → NE (Pi's two fail-open exploits — all-NaN precompute, mismatched check+Δ —
  are reproduced as adversarial tests and confirmed NE/REFUSE). Two explicit boundaries, NO module-global mutation:
  **`gate_group_dev`** (explicit hash-bound floor/B; context-bound maps; one identity per (profile,regime,check))
  and **`gate_group_registered`** (enforces exact N/arm=8000, `B=20000`, floor 500, EXACT `α=0.04/6`, exact stratum
  quotas, full registry identity, approved map-set + RNG-manifest identities — refusing ANY deviation; a real run is
  BLOCKED until the reserved manifests are drawn). Structured registered structural-zero `(2667,2667,2666)`
  assembles with NO divisibility refusal (self-test). Each per-permutation recompute reproduces the EXACT v2
  estimand to `<1e-9`. 10 estimators wired (burst-timing + class-mark); the other three groups are the next wiring
  step.
- **Registered-N boundary preflight (Pi #2, canonical constructor Pi #3, provenance Pi #6)** —
  `scripts/oracle_realism_v3_regn_preflight.py` (hash **`da499057…`**): at N=8000 the bounded S3_loggap map ISSUES
  (evaluable) — the dev-scale "structurally un-calibratable" claim is WITHDRAWN; both S3 exemptions rest on the
  DETECTION criterion (bounded detect **0.0 < 0.5**, full/SCID/MIMIC/structural-zero **1.0**), per regime with/
  without each. **structural-zero uses the CANONICAL multiscale constructor** (means 18/60/250), and the map
  artifact records the EXACT seed/namespace. Both PROVISIONAL; finalisation needs a separately-authorized power
  battery (Pi #7).
- **SPARSE + PAIRED dev-boundary demo (Pi rev-7 #1, rev-8 #7)** — `scripts/oracle_realism_v3_group_power.py`
  (hash **`105c4d9c…`**), via `gate_group_dev` with STRUCTURED arms: perturbs EXACTLY one experiment and ASSERTS
  (hashed, timestamps included) every non-target arm is identical to null AND the target arm changed; evaluates
  base AND its perturbation under the SAME permutation seed and compares `p_g(perturbed)` to `p_g(base)`;
  structural-zero data AND map-design sample via the canonical multiscale constructor; maps context-bound to
  (profile,regime,dev-floor). MECHANICAL (B<resolution). The **burst-timing group** demonstrates it end-to-end
  (null PASS; paired perturbed `p_g` < base; argmin attributes to the perturbed primary cell). The **class-mark
  group correctly NEs at dev N=810** (the leakage-free independent map is legitimately stricter) — its
  multi-component sensitivity belongs to the registered-scale run (deferred). The rev-5 3-cell
  run is a MECHANICAL SMOKE (`29da0411…`).
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

## Still open (Pi-authorized next scope; NOT yet done in rev-9)
- **Reserved map-set + RNG-manifest generation** (generate, do NOT execute) — `gate_group_registered` binds to
  `RESERVED_MAP_SET_NOT_DRAWN` / `RESERVED_RNG_MANIFEST_NOT_BOUND`, so a real registered run is BLOCKED. The RNG
  manifest must bind exact per-role/stratum seeds + generator/coupling CODE identities + profile identity +
  content/count hashes + canonical arm order (Pi rev-8 #5/#6); the map-set manifest enumerates every
  `(profile,regime,check)` + seed/namespace/N/floor/builder identity + set-hash rule.
- **Estimator/precompute/raw-record schemas** (Pi rev-8 #5) — `_validate_precompute` rejects Inf only; add
  per-estimator key/shape/owner-index/vector-width/pooled-length + legal-NaN + raw-record finiteness schemas; mark
  the low-level `gate_group(spec)` private/test-only.
- **Boundary-short canonical map** (Pi rev-8 #8) — route boundary-short through its canonical control constructor.
- **Wire the length_density / run_size / phase_seam estimators** into the engine (burst-timing + class-mark are
  wired); then a full per-group **wired benchmark at the registered `B=20000`** per exact profile/stratum, with the
  1.5× margin applied to EVERY group + checkpoint/resume + real evidence persistence, then mint job kinds. (The
  current benchmark's wired measurement uses a mechanical `B=5400` and covers only the burst-timing group; the
  route surrogate already forecasts phase/seam ≈ 8.46 h, so that route needs a predeclared split — Pi rev-7 #8.)
- **Executable RNG-provenance manifest** — bind exact per-role derived seeds + generator/coupling CODE identities
  in a trusted experiment manifest and compare content/count hashes at the engine boundary (the current
  `rng_identity` hashes metadata, not the generator call — Pi rev-7 #5).
- **Reserved map-set MANIFEST** — enumerate (do NOT execute) every required `(profile, regime, check)`, seed,
  namespace, N/floor, builder/code identity, output path, and set-hash rule, with missing/extra/duplicate failing
  closed (Pi rev-7 #6). The final Δ-aligned exemption then needs a separately-authorized calibration/power battery
  using the frozen map (Pi rev-7 #7).
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
