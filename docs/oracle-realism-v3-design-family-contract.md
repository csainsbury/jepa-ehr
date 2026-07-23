# Oracle Realism Verifier — Design Family v3 (rev-4, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-3. Folds Pi rev-2 ruling (§1–§9) AND Pi rev-3 structure ruling
(partial-pass corrections). Phase-0 estimator CONCEPT frozen (`docs/oracle-realism-v3-phase0-estimator-pilot.md`,
committed pilot `scripts/oracle_realism_v3_phase0_pilot.py`) — pending the exact-profile rerun (below). New design
family; no calibration build or run until Pi re-gates. Written before any calibration/audit/evaluation seed.

**Corrections folded (Pi rev-2 + rev-3):**
1. **Per-draw p-value, not transported** — the SD permutation test is computed on each fresh evaluation draw (§1).
2. **Grouping does NOT erase multiplicity** — min-p over `K_g` cells has critical value ~`α_group/K_g`; the "dissolves
   M0" claim is withdrawn; the architecture removes only the independent-calibration-corpus cost (§2).
3. **Group-p direction fixed (rev-3 §1):** `p_g = (1+#{S_g^(b) ≤ S_g^obs})/(B+1)` — smaller min-p is more extreme
   (§3). Per-cell is the opposite (upper) tail.
4. **Coarsening: independent REFERENCE-OWNED frozen map, not pooled (rev-3 §2)** — a pooled map lets the candidate
   influence the bins (anti-masking defect). The reference-design-draw map preserves randomization validity AND
   anti-masking AND O(N) (§9).
5. **Estimator/exemptions provisional (rev-3 §3):** rerun on the EXACT registered profiles; `S3_tau` and `S3_loggap`
   exemptions decided SEPARATELY, Δ-aligned (§7); estimator not fully frozen until that reruns.
6. **Benchmark unit + whole-JOB forecast (rev-3 §6):** ~0.45 ms/perm (not µs); forecast the full job (main + 15
   audit batteries + 25 MM), which may need separately-gated jobs (§9).

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
- **`B` sizing:** frozen by a resolution criterion, not a default. Require the MC granularity `1/(B+1)` ≤
  `α_group/20` and benchmark-bounded; provisional `B = 20,000` pending the exact `G`. Fisher/Cauchy combiners are
  **OUT** for v3.

## 4. Error split + independent audit stopping rule (Pi §4)

```
alpha_total_family = 0.05
alpha_main_SD      = 0.04        # main fresh SD draw; alpha_group = 0.04 / G
audit_budget       = 0.01        # independent audit stopping rule (NOT a B CI)
audit_replicates   = 15
audit_park_rule    = park if >= 4 of 15 complete fresh SD batteries reject
```

If each valid SD battery rejects a true null with probability ≤ `alpha_main_SD = 0.04`,
`P(Binom(15, 0.04) ≥ 4) ≈ 0.002454 ≤ 0.01`. Main draw + audit ≤ 0.05 total false-park by union bound. The audit is
a **gross implementation/seed-pathology guard**, not a tail estimate; a miss parks with no retuning. Audit seeds
are a disjoint namespace (§8).

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
joint), pair-count-weighted pooling across within-sequence pairs. Frozen + evidenced in the pilot doc: formula err
vs scipy `1.7e-16`; phase coverage mean-pos 0.10→0.50; concentration MIMIC 0.045→0.905 / SCID 0.014→0.934;
source-wise power full=1.00, bounded=0.125; S8 no cross-load. Other coarse-null rank statistics, if any, adopt the
same pool-within/synchronized-permute pattern (enumerated in the registry).

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
- **Per-statistic exemption status (Pi rev-3 §3 — each boundary exemption decided separately, on exact-profile +
  Δ-aligned evidence):** boundary-short × `S3_tau` is **provisional but plausible** (dev power 0.125 on the
  earlier non-Δ proxy); boundary-short × `S3_loggap` is **NOT accepted** (no S3_loggap power result exists yet);
  neither is frozen until the exact-registered-profile, Δ-aligned pilot evidence is routed. Full-support
  burst-timing is RETAINED (non-exemptible).
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

**Provisional benchmark (dev, N=8000/side; NOT yet reproducibly committed — committed harness is a deliverable).**
- **Naive** whole-battery recompute per permutation = **20.1 s** → B=20,000 ≈ **112 h/group** → INFEASIBLE.
- **Permutation-efficient route** (precompute per-sequence contributions once; per permutation an O(N) group
  re-summation): pooled-tau re-sum ≈ **0.45 ms/perm** (B=20k ≈ 9 s) [unit corrected per Pi rev-3 §6, was mis-stated
  as µs]; KS via one pooled sort + O(N) cumsum ≈ 0.19 ms/perm (B=20k ≈ 3.8 s); frozen-map group-mean-diff ≈
  0.34 ms/perm (B=20k ≈ 6.8 s). Whole battery (~14 stats, all O(N)) ≈ **74 s/group at B=20k** — PROVISIONAL pending
  the exact registry + executable coverage of every nonlinear route (conditional bins, KS, seam floors) under the
  reference-owned map. Non-additive checks not reducible to precompute+O(N) are excluded from the SD gate / park.

**Whole-JOB wall-time forecast (Pi rev-3 §6 — account for the entire job, not one group):**
`1 main SD battery + 15 audit SD batteries + (all SD experiments/groups × B permutations) + 25 MM
power/specificity replicates + global controls + persistence + verdict aggregation`. v2's MM work alone was
≈ 4.9 h, so audit ×15 + permutations may approach the 8 h cap. The preflight must give the total forecast and
either **fit one job under the cap OR predeclare separately-gated jobs** (without weakening sequential
stop-on-failure semantics). Committed benchmark lands code/environment/seed/raw timing/event-cluster-pair
volumes/extrapolation formula/aggregate hash.

**Preflight persistence:** per experiment/group report `sequences, events, clusters, eligible pairs, strata,
cells, B, statistic recomputation route, wall time, peak RAM, checkpoint size`; aggregate-only (per-cell/group
permutation summaries + hashes; no sequence arrays).

## Deliverables status
1. Rev-3 contract — **this document**.
2. Exact SD/MM registry (M0/G/groups/strata/proofs) — structure frozen (§8); the enumerated machine-readable
   artifact + coupling-exchangeability proofs is the next concrete build step.
3. Δ / required-property / exemption table — framework frozen (§5,§7); boundary-S3 exemption decided; the numeric
   Δ table is sourced + hashed at design freeze (values flagged for Pi/Chris ratification).
4. Exact min-p / randomization p-value algorithm + α/audit — §3,§4.
5. Phase-spanning pooled-tau pilot (code/seeds/hash) — **DONE** (committed script + doc + hash `1e930132…`).
6. Exact MM 20/25 & 24/25 logic — §5.
7. Whole-battery permutation benchmark + cap/checkpoint plan — feasibility PROVEN (§9); the full exact-registry
   benchmark is produced at the calibration-build preflight, separately authorized.

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

## Delivered (Pi rev-3 §4/§5/§6 authorized scope)
- **Exact machine-readable SD/MM registry + self-tests** — `scripts/oracle_realism_v3_registry.py` (hash
  `7adb3369…`). **10 SD experiments** (null×2, candidate-D repeatability per component×source ×6, structural-zero,
  boundary-short); **M0 = 192** in-scope SD cells; **G = 6** groups by (support-regime × substantive-family) — 5
  full-support family groups (product-permutation across the 9 full experiments) + 1 bounded support-control group;
  `α_group = 0.04/6 = 0.00667`; deepest group min-p critical ≈ `α_group/54 ≈ 1.2e-4` (the honest within-group
  multiplicity — grouping is NOT claimed a power gain; dev-seed group critical values + power are demonstrated
  next). 117 MM cells. All self-tests pass: exact partition `Σ K_g = M0`, reachability, explicit exemptions, core
  (full-support) cells non-exemptible, source-swap + S4↔S7 matching the executable ABLATION_MATRIX.
- **`Δ` / exemption table** — `Δ` = the v2 per-check practical-effect thresholds (embedded in the registry);
  boundary exemptions: degenerate `{S1_density,S5_abs,S6_tv,S9_zero,S9_class,S9_gap}` + provisional-uncalibratable
  `{S3_tau,S3_loggap}` (Δ-aligned confirmation pending).
- **Exact-profile Phase-0 pilot rerun** — DONE (`scid_scale_control`/`mimic_scale_control`; `S3_tau` and
  `S3_loggap` measured separately; hash `0e1680fc…`).
- **Randomization p-value algorithm + EXHAUSTIVE tests** — `scripts/oracle_realism_v3_randomization.py`: exact-null
  rejection ≤ α in all configurations (finite-sample exactness), correct opposite tail directions, ties exact.
- **Committed benchmark harness + whole-JOB forecast** — `scripts/oracle_realism_v3_benchmark.py` (hash
  `acd8704f…`): job cost is driven by `M0·B` (NOT `G`). With **main B=20,000 + audit B=2,000** (15 audit reps) the
  whole job ≈ 6 h under the 8 h cap; with audit B=20,000 it exceeds → separately-gated SD/MM jobs.

## Still open (Pi rev-3, next after this package is reviewed)
- **Executable exchangeability proofs at RUNTIME** (§5): the registry binds each experiment's stratum + passes the
  structural self-tests; the gate implementation must additionally refuse before statistics on a malformed stratum
  / unequal group size / duplicate-missing index / non-bijection (flagged; part of the gate build).
- **Dev-seed group critical values + sparse/diffuse power demonstration** under the actual joint permutation null.
- **Reference-design coarsening-map draw + calibration-build preflight** — NOT authorized until this package passes.

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
