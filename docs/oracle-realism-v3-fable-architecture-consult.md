# Fable consult #2 — acceptance architecture (tail-certification crux)

## Headline
The M0 / tiny-α_i / ~15,000-replicate-corpus / tail-model / N-transport pain is **an artifact of the per-check
k-of-n (≥24/25) architecture**. Under a permutation gate those problems largely **disappear**, because the
same-distribution control makes candidate/reference sequences **exchangeable by construction**, which gives an
**exact finite-sample null for free** (label-permute cand/ref across the pooled sequences, recompute the battery).
Recommendation: **replace the k-of-n battery with a small number of grouped PIT/min-p permutation-calibrated
family gates on one large draw.**

## (1) The architecture — grouped min-p permutation gates
- **Exactness:** calibrating a statistic's threshold to the (1−α_family) quantile of its cand/ref
  **label-permutation** null controls the family same-distribution false-fail rate **exactly** (not asymptotic),
  up to Monte-Carlo error in `B` (# permutations), which is bought cheaply by raising `B`.
- **Statistic — use PIT/min-p, NOT raw studentized max.** Map each cell's discrepancy through its own
  permutation-null CDF to a p-value `p_cell`; gate on `S = min_cell p_cell` (Tippett). Scale-free: no cell can
  dominate via an under-estimated sd (this is the main reason to prefer min-p over studentized-max).
- **Effect deadband survives:** define each cell's discrepancy as `(d_cell − Δ_cell)_+` **before** the PIT →
  cells within their equivalence margin give p≈1 and never drive the min. Effect-size semantics preserved.
- **Grouping (the power fix):** do NOT take one global min over ~300 cells. Use **~5–10 grouped gates**, one per
  (support-regime × substantive-family), each a min-p over its 5–8 coherent cells, with α_family split across the
  handful of groups (mild Bonferroni over G≈5–10, not 300). Limits power dilution; preserves attribution.
- **Calibration:** ONE large N=8000 draw; `B ≥ 20,000` label permutations recompute the WHOLE battery per
  resample; threshold `c` = the α_family-quantile of S's permutation null. Accept iff `min-p ≥ c`.
- **δ_cal budget → choose B**, not ~15k independent draws. Put a binomial order-statistic CI on the permutation
  quantile; tighten by raising B. (This is where the calibration-uncertainty budget is spent.)
- **R=25 dropped as an acceptance mechanism** (the permutation null already models single-draw variability
  exactly). Repurpose a **small replicate set (~5–15 seeds) as a calibration AUDIT** — check empirical
  false-fail ≈ α_family and guard a pathological generator seed. Insurance, not the error-control mechanism.
  One large draw + permutation gate is strictly more efficient than 25 smaller draws under k-of-n.
- **Attribution:** the gate is scalar but you log the full per-cell `p_cell` vector + report the argmin → exact
  attribution of which cell failed. No diagnostic loss vs k-of-n.
- **min-p vs diffuse departures:** min-p/Tippett is optimal against "ONE cell badly off" (the usual acceptance
  failure mode — one specific dependency wrong). Weak against "many cells each slightly off." If diffuse matters,
  add a Cauchy-combination or Fisher (−2Σln p) gate ALONGSIDE min-p within a group and split α; don't replace it.

## (2) Parametric tail model (GPD/POT) — moot under (1)
Only relevant if you keep independent-replicate k-of-n. Even then it buys ~2–3×, not an order of magnitude
(POT needs ~100 exceedances at a 99th-pct threshold → n ≈ 10⁴ anyway). If used: fix ONE family + ONE
threshold-rule + a train/test GoF falsifier, all committed BEFORE the locked corpus; GoF rejection ⇒ cell
un-calibratable ⇒ park (no family/threshold re-selection on the corpus). Recommendation: don't — adopt (1).

## (3) Reduced-N calibration transported to eval N — mostly unnecessary under (1)
`τ(N)=τ(N_cal)·√(N_cal/N)` is defensible ONLY if BOTH scale-invariance (sd ∝ 1/√N_eff) AND **shape-invariance**
(standardized tail shape stable) hold. Breaks on: discreteness at small N_cal (heavier tail → under-predicts τ),
finite-N tail convergence (shrinking skew/kurtosis biases the fixed quantile), and the cap making effective-N ∝ #
sequences only once saturated. Falsifier must be two-part: calibrate at two N_cal, require both the √-scaling AND
a standardized-tail (skew/kurtosis/QQ) match. Under (1) the full-N permutation null is already cheap, so skip.

## (4) Composite joint calibration — same machinery as (1)
Keep "all-K-pass" composites as a grouped min-p gate: permute the group label ONCE per resample, recompute all K
checks together → within-replicate cross-check correlation preserved automatically (no independence assumption, no
Bonferroni). A composite is just a grouped min-p gate → (1) and (4) converge (a good sign the structure is right).

## One-paragraph recommendation (verbatim)
Drop per-check k-of-n and the 24/25 bar. Build ~5–10 grouped min-p (PIT) permutation gates on one large N=8000
draw, with the Δ_cell equivalence deadband applied before the PIT, α_family split across the handful of groups,
and B permutations chosen large enough that δ_cal is spent as Monte-Carlo precision on the permutation quantile
(with a binomial order-statistic CI). Keep min-p for the "one cell badly wrong" alternative; add a Cauchy/Fisher
combiner within a group only if diffuse departures matter. Repurpose a small replicate set as a calibration audit,
not an acceptance bar. This gives exact same-distribution family error control, automatic cross-cell/cross-check
dependence handling, full per-cell attribution by logging, and turns a ~10⁴-replicate-per-regime corpus into B
in-memory resamples of a single draw. The tail-model (2) and N-transport (3) questions are then unnecessary.

## Reconciliation note (added by jepa-ultra)
This CONFLICTS with Pi phase-1 ratified direction 1 ("keep R=25, ≥24/25 expected-PASS bar, ≥20/25 primary-power
bar"). Fable's architecture replaces the k-of-n acceptance mechanism with permutation gates + a small audit.
Adopting it requires Pi's explicit RE-ratification of the acceptance structure — it is a bigger change than the v3
draft assumed, but it dissolves the corpus/compute infeasibility Pi flagged. Effect-size Δ, the frozen pooled-tau
estimator, exact regimes, seed namespaces, and fail-closed machinery all survive unchanged.
