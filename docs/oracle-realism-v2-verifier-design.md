# Oracle realism v2 — executable-verifier DESIGN FREEZE, rev-2 (Pi REVISE folded)

Synthetic-only. The frozen DESIGN step 3 will implement. DEV identity `m3a_design_dev_hash = 4b9e76cc…`;
source `clinical_jepa/eval/oracle_realism_v2_verifier_design.py`. Re-routed for confirmation before coding.

## What changed from rev-1 (your REVISE, all folded)
- **Two separate routes.** The six REGISTERED marginals keep their EXACT v1 estimands
  (`class_tv` = pooled event-count proportions; `delta_t_zero` = pooled zero-adjacencies/pooled-adjacencies;
  `positive_gap` = pooled ECDF 8dp right-continuous; length/count per-seq ECDFs; occupancy equal-seq mean) on an
  `AggregateStats↔AggregateStats` route — development-seen match is **EXPLORATORY only**. S1–S9 run on a
  separate `SequenceSample↔SequenceSample` route (synthetic recovery). Passing both is **not** a joint-envelope
  claim; the per-sequence sampling-unit rule applies to S-statistics only.
- **Canonical fixture law (fixed the overdetermination).** Sample L; sample maximal run sizes until they sum to
  exactly L (frozen terminal truncation); **derive** K, cluster ids, and Δt=0 from the runs; one positive
  inter-cluster gap per boundary; derive timestamps. `dt0_rate` is now a **derived diagnostic**, not an
  independent Bernoulli. Singletons included in S2.
- **Whole-sequence timing + S9 seam guard.** `V2_BLOCK_COMPOSITION` amended: the timing process is generated
  over the whole sequence independent of block boundaries (seams follow the same law; not forced positive).
  **New mandatory S9 — block-seam invisibility** (zero-gap, same-class, positive-gap seam-vs-nonseam contrasts;
  ≥500 seam + nonseam adjacencies); terminal adequacy guard, **not** a D route.
- **Reference-only coarsening.** The adjacent-bin merge map is derived from the **reference only** and applied
  unchanged to the candidate (candidate floor-fail ⇒ NOT_EVALUABLE); ≥3 retained length + ≥3 cluster bins;
  position quartiles never coarsened.
- **Derive-not-trust input schema.** Trust only {source, class_ids, timestamps}; derive L_total/B/R/positions/
  cluster ids; reconcile any redundant fields exactly; reject NaN/Inf/bool-as-int; 8dp tie convention.
- **S-statistic fixes.** S2 = seq-equal ECDF of all run sizes (not binned+KS); S3 = `|E[log gap]_c − E[log
  gap]_r| ≤ log1.10` + per-seq τ-b; S4 = class-count combinatorics (no O(L²)) with same-cluster + adjacent-pair
  floors; S7 = equal-SEQUENCE weighting; S8 = position-quartile density (cluster-starts/items) + class, all four
  quartiles, terminal; every conditional check declares its floors and subchecks.
- **Escalation map corrected.** Split into subchecks (`S1_density/S1_tau`, `S3_tau/S3_loggap`, …). **Removed
  S2→D** (S2 is the cluster-size marginal). **S8 declared terminal/out-of-model** (no D route) — I did not add
  position D-components; the alternative you offered (`position_cluster_density`/`position_class_mix`) is noted
  for a future round if you'd rather make S8 escalatable. Six marginals + source-swap never trigger D.
- **Operational couplings + ablation matrix.** Each D component has a marginal-preserving operational law
  (pool-neutral reallocation / rank copula / count-preserving relabelling) that moves only its mapped subcheck;
  `ABLATION_MATRIX` gives each component's expected fail/pass row; `source_swap` is operational.
- **Profiles renamed** `scid_scale_control`/`mimic_scale_control` (only length scale anchored; class/timing/burst
  are synthetic, not source estimates); added an explicit `structural_zero_control`; fixture "no governed read;
  uses only cleared development-seen constants."
- **Identifiability.** Affine range standardization (no logit); backward one-sided FD at the top profile (moved
  to 0.55); forward one-sided active-subset at 0; explicit estimator/whitening/covariance-ridge/tie-break/
  collision distance; staged compute budget.
- **Sample size** 4000 is now explicitly PROVISIONAL — step 4 must demonstrate every floor at the reference-
  derived coarsening (≥3 length + ≥3 cluster bins, all 4 quartiles, seam/pair floors, rate criteria) or
  increase N only, with a runtime/memory estimate first.

## Confirm / revise
1. The two-route separation + exact registered estimands — correct?
2. Canonical fixture law + derived dt0 + S9 seam guard — as intended?
3. S8 **terminal** (my choice) vs adding `position_*` D-components (your alternative) — which do you want?
4. The operational coupling laws + ablation matrix — accept, or specify exact transforms you prefer?
5. Escalation subcheck granularity + set-cover semantics — accept?

On confirm I implement the executable verifier + independent fixtures (step 3), run sims (step 4), and route
the implemented verifier for M3a final review (step 5). M2 stays blocked.
