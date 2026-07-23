# Oracle Realism Verifier — Design Family v3 (rev-3, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-2. Folds Pi rev-2 ruling (§1–§9). Phase-0 estimator FROZEN
(`docs/oracle-realism-v3-phase0-estimator-pilot.md`, aggregate hash `1e930132…`, committed pilot
`scripts/oracle_realism_v3_phase0_pilot.py`). Whole-battery permutation feasibility EMPIRICALLY PROVEN (§9). New
design family; no calibration build or run until Pi re-gates. Written before any calibration/audit/evaluation seed.

**Corrections to rev-2 that Pi caught (now fixed):**
1. **The permutation p-value is computed on each FRESH evaluation draw**, not calibrated once and transported
   (exactness is conditional on the tested draw's pooled observations). We freeze the *algorithm*, not critical
   values (§1).
2. **Grouping does NOT erase multiplicity.** min-p over `K_g` cells has its critical value ~`α_group/K_g`; the
   architecture removes the independent-calibration-corpus cost, not the statistical multiplicity/power cost. The
   rev-2 "dissolves M0" claim is withdrawn; `M0`/`G`/group-sizes are now exact and consistent (§2).

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

## 3. Randomization p-value algorithm (Pi §3)

- `p_g = (1 + #{b : S_g^(b) ≥ S_g^obs}) / (B + 1)` — observed assignment included; conservative finite-`B`.
- min-p construction: each cell's discrepancy `e_c = (d_c − Δ_c)_+` → its permutation p-value `p_c` via the SAME
  synchronized permutations (inclusion convention: the observed assignment is permutation index 0; ties in `S_g`
  counted with `≥`); `S_g = min_c p_c` (larger −ln p ⇒ more extreme ⇒ the `≥`/`≤` direction is frozen so
  "at least as extreme" = "min-p at most as small").
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
- **Un-calibratable** is defined by **predeclared power against a meaningful alternative beyond `Δ_c`**, not by
  null mass above `Δ_c` (conditional randomization can still control sharp-null rejection). Frozen dev-only power
  criterion: a cell is retained only if its development detection power against the `@0.5` component alternative is
  ≥ 0.5. A property-specific support cell failing this is EXEMPT; a **core/full-support cell failing it PARKS**.
- **Decided now (dev evidence):** boundary-short × burst-timing (S3_tau, S3_loggap) is EXEMPT (power 0.125 < 0.5);
  full-support burst-timing is RETAINED (non-exemptible). Recorded in the registry.
- The full `Δ` table + required-property matrix is the "Δ / required-property" deliverable (below); values are
  sourced from prior/first-principles before calibration and hashed.

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

## 9. Whole-battery permutation feasibility — PROVEN, with a design requirement (Pi §9)

Benchmark (dev, N=8000/side, one MIMIC pooled draw):
- **Naive** whole-battery recompute per permutation = **20.1 s** → B=20,000 ≈ **112 h/group** → INFEASIBLE.
- **Permutation-efficient route** (required): precompute each sequence's contribution once; per permutation do an
  O(N) group re-summation. Measured: pooled-tau re-sum 0.4 µs/perm (B=20k ≈ 9 s); KS via one pooled sort + O(N)
  cumsum 0.19 ms/perm (B=20k ≈ 3.8 s); **frozen-coarsening group-mean-diff 0.34 ms/perm (B=20k ≈ 6.8 s)**. Whole
  battery (~14 stats, all O(N)) ≈ **74 s/group at B=20k** (order minutes).
- **Design requirement (needs Pi ratification):** for the SD gate the **coarsening/binning is FROZEN from the
  POOLED draw** (both groups), making it permutation-invariant — this is what turns the v2 reference-only
  coarsening (which changes per permutation) into an O(N)-per-permutation statistic AND keeps the test an exact
  conditional randomization test (the coarsening is a fixed function of the pooled sample under label permutation).
  KS statistics use one pooled sort + per-permutation cumsum. Non-additive checks not reducible to these forms are
  excluded from the SD permutation gate and, if required, park pending a separate treatment.
- **Preflight (before any calibration build):** per experiment/group report `sequences, events, clusters, eligible
  pairs, strata, cells, B, statistic recomputation route, wall time, peak RAM, checkpoint size`; aggregate-only
  persistence (per-cell/group permutation summaries + hashes; no sequence arrays).

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

## Items needing Pi ratification
- The two rev-2 corrections above (per-draw p-value; multiplicity not dissolved) — adopted.
- **Frozen-pooled-coarsening for the SD gate** (the §9 feasibility+exactness requirement) — a change from v2
  reference-only coarsening.
- `α` split `0.05 = 0.04 (SD, α_group=0.04/G) + 0.01 (audit)`; audit rule `15 reps, park ≥4` (P≈0.002454).
- `G`, the group partition, and the `Δ` table sourcing.
- The SD/MM split and the boundary-S3 exemption (decided from dev evidence).

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
