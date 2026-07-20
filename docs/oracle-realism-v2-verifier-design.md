# Oracle realism v2 — executable-verifier DESIGN FREEZE (rebuild step 2; for Pi's confirmation)

Synthetic-only. The frozen DESIGN that step 3 will implement, codifying your M3a rulings. DEV identity
`m3a_design_dev_hash = 3ec8577d…`; source: `clinical_jepa/eval/oracle_realism_v2_verifier_design.py`. Nothing
sampled/fitted/governed. Scored on **candidate − reference**; admissible claim = "matches the declared
marginal + cross-statistic envelope," never the joint process. Requesting confirm/revise BEFORE I code step 3.

## 1. Input schema (typed, fail-closed)
A sample is N independent full `SequenceRecord`s: `{source, L_total, block_count_B, residual_R, class_ids,
timestamps (nondecreasing), cluster_ids (Δt=0 runs, may span block boundaries), position∈[0,1]}`. Any schema
violation (nonmonotone ts, `R∉[0,7]`, `L_total≠8B+R`, class out of range, empty) raises — never coerced.

## 2. S1–S8 algorithms (per-sequence first, equal-weight sequences; max-norm over bins; conjunction over
checks × sources)
- six marginals candidate-vs-reference: `length_ks`, `class_tv`, `count_ks` (**per-seq cluster-count** KS),
  `occupancy_abs`, `delta_t_zero_abs`, `positive_gap_ks`.
- **S1** K/L_total by length-bin: max-bin abs-diff ≤0.03 + τ-b(L,K) diff ≤0.05.
- **S2** Δt=0 cluster-run-size ECDF KS ≤0.05 (overflow-supported).
- **S3** gap by preceding-cluster-size bin: τ-b diff ≤0.05 AND max-bin |log mean-ratio| ≤log1.10.
- **S4** P(same|same cluster)−P(same|adjacent) diff ≤0.03; eligible-pair equal weight per seq.
- **S5** occupancy by length-bin candidate-vs-reference ≤0.03 (the M0b `min(L,5)/5` cap is a separate
  feasibility assertion, not the comparison target).
- **S6** length-dependent class-mix, max-bin TV(candidate−reference) ≤0.05 (**mandatory**; not raw TV).
- **S7** E[n_distinct/min(C,5)] by cluster-size bin, max-bin abs-diff ≤0.03; large clusters weighted by
  cluster count.
- **S8** cluster density + class mix by position quartile: density max-bin ≤0.03, class TV ≤0.05.
- floors: 500 per the stated `floor_unit`; deterministic adjacent-bin coarsening applied identically to
  reference and candidate; floor-fail-after-merge ⇒ NOT_EVALUABLE. Bins are the frozen overflow bins.

## 3. Numeric synthetic control profiles (closed-form, frozen; never fitted from M2)
Length = discretized lognormal on `L_total≥1` shape-anchored to the cleared quantiles
(`scid_like` μ=ln350 σ=0.90; `mimic_like` μ=ln99 σ=1.00); 5-class prior (scid_like dominant-class to make a
class-TV swap non-trivial); geometric cluster size; lognormal gaps; Bernoulli Δt=0. Profiles: `scid_like`,
`mimic_like`, `interior_low/mid/high` (D coupling 0.10/0.35/0.60), `null_independent`, `boundary_short`
(near L<5 / support-floor edge), `source_swap` (must fail a **non-degenerate** check, not only class-TV), and
per-component `ablation` (one D component at 0.5 — must fail exactly its attribution-mapped check).

## 4. Simulation (rate-based)
25 deterministic seeds (1000–1024), 4000 sequences/source/seed; self/known passes all required checks ≥24/25;
each minimally-misspecified control fails its intended check ≥20/25; ablations pass non-attributed checks at
the specificity rate; report empirical rates + binomial CIs; conjunctive across sources; median secondary.

## 5. Independent fixture generator (your self-certification guard)
A closed-form reference/control constructor `realism_v2_reference_constructor_dev` that samples each field
directly from the profile parameters and composes whole-sequence timestamps/cluster-ids per the block schema.
It shares **no code path** with the future M2 candidate adapter, is frozen before M2, is never scored as a
realism candidate, and uses no TRAIN data.

## 6. Identifiability
Param ranges [0,0.6] per component, logit-scaled + standardized; 3^k joint grid (0.1/0.35/0.6) with marginal
nuisance varied; central FD step 0.02 with CRN seeds; **standardized Jacobian σ_min/σ_max ≥ 1e-3** (abs tol
secondary); recovery ≤0.05 of range and ≤ half a grid step; collision iff ALL standardized cross-stat diffs
within acceptance tol; at `null_independent` use active-subset/one-sided rank.

## 7. Escalation (separate CHECK→D map, active-set identity)
`CHECK_TO_D_COMPONENTS`: S1→burst_count_length, S2→cluster_size_mark_diversity, S3→burst_timing,
S4→mark_burst_tie, **S5/S6→length_class_mix**, S7→cluster_size_mark_diversity, S8→{burst_count_length,
length_class_mix}. Selection = smallest super-set covering failed checks (ties by frozen menu order); active
set expands monotonically; D identity minted over the exact active set; iteration_cap 3 → terminal FAIL/park;
ledger = {parent_identity, m3a_spec_hash, control_profile, failed_statistics, active_component_set,
decision_hash, result_hash, seed_set}.

## Questions for Pi
1. Statistic algorithms + floor_units (esp. S3 dual metric, S4/S7 weighting, S8 quartile density+TV) — as you
   intend?
2. Numeric profiles: are the shape anchors + the scid_like dominant-class + interior 0.1/0.35/0.6 + ablation
   0.5 acceptable synthetic controls, or do you want specific values?
3. Sample size 4000/source/seed sufficient for the floors (500) after coarsening at these bins? (I will confirm
   empirically in step 4; changing size not thresholds.)
4. Identifiability param range [0,0.6], FD step 0.02, grid 0.1/0.35/0.6 — accept?
5. CHECK_TO_D_COMPONENTS mapping (esp. S2/S7 both → cluster_size_mark_diversity; S8 → two components) — accept?

On confirm I implement the executable verifier + independent fixtures (step 3), run the sims (step 4), and
route the implemented verifier for your M3a final review (step 5). M2 stays blocked.
