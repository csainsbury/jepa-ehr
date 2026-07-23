# v3 Phase-0 estimator micro-pilot — results and frozen choice (rev-2, phase-spanning)

**Purpose (Pi rev-2 §6):** freeze ONE burst-timing dependence estimator on **development-only seeds**, with an
executable committed artifact, a frozen seed list, and an aggregate hash. No calibration/audit/evaluation seeds are
used. Supersedes the first-6-boundary pilot (rejected: early-sequence estimand, gameable).

**Reproducible artifact:** `scripts/oracle_realism_v3_phase0_pilot.py` — dev namespace `v3-estimator-dev`, frozen
dev seed list `90000..90039`, N=3000/side. Aggregate result hash: `1e930132…` (canonical_hash of the full
aggregate; deterministic given the seeds).

## Frozen estimator: phase-spanning capped, tie-corrected pooled Kendall tau-b

For a sample, per sequence extract positive-gap boundaries as pairs `(x = preceding cluster size, y = gap)`.
**Cap (phase-spanning, frozen):** if `m ≤ 6`, use all boundaries; else use the 6 distinct **quantile-spaced**
indices `sorted({ round(v) : v in linspace(0, m−1, 6) })` — spanning the WHOLE sequence, not the first 6.
Over the within-sequence pairs of the selected boundaries (never across sequences), accumulate the **standard
tau-b** components, summed across sequences:

```
C − D = Σ_pairs sign(x_i−x_j)·sign(y_i−y_j)          # pairs tied in x or y contribute 0
n0    = Σ_s C(m_sel_s, 2)
n1    = Σ_s #{pairs tied in x}    (ALL x-ties, incl. joint ties)     # standard tau-b n1
n2    = Σ_s #{pairs tied in y}    (ALL y-ties, incl. joint ties)     # standard tau-b n2
T_pool(sample) = (C − D) / sqrt((n0 − n1)(n0 − n2))                  # None if either factor ≤ 0
```

Two-sample discrepancy `d = |T_pool(cand) − T_pool(ref)|`. **Property claim (frozen):** a *capped, phase-spanning,
pair-count-weighted within-sequence rank dependence* between inter-cluster gaps and preceding cluster sizes — not
first-6 (early), not uncapped, not equal-per-sequence. For the SD gate, the null is the **per-draw** between-group
sequence-label permutation computed at evaluation (Pi rev-2 §1); this pilot does NOT claim a calibrated tail.

## Evidence (dev seeds; aggregate hash `1e930132…`)

**(a) Formula correctness — reconciled with scipy tau-b incl. joint ties.** Over 500 random tie-heavy `(x,y)`,
`max |T_pool_over_one_seq − scipy.kendalltau(x,y)| = 1.67e-16` (machine epsilon). Fixes the rev-1 "tied in x only /
y only" discrepancy Pi flagged; `n1`/`n2` now count all x/y ties (joint ties included), matching tau-b exactly.

**(b) Phase coverage — the cap now spans the sequence.** MIMIC full support, 2,466 sequences with `m>6`:
mean normalized selected-boundary position is **0.104** for the rejected first-6 cap vs **0.500** for the
phase-spanning cap. The estimand no longer tests only early timing.

**(c) Contribution concentration — fixed on full support, both sources** (inverse-Simpson effective fraction;
top-1% pair share):

| regime | uncapped | phase-spanning cap |
|---|---|---|
| MIMIC-scale | eff-frac 0.045 (top1% 0.384) | **0.905 (0.011)** |
| SCID-scale  | eff-frac 0.014 (top1% 0.495) | **0.934 (0.011)** |

**(d) Source-wise null + power** (40 dev replicates; `d` matched-null vs `burst_timing@0.5`-coupled; power =
fraction of coupled draws exceeding the null 96th percentile — a distribution, not one comparison):

| regime | null mean (sd) | null p96 | coupled mean (min) | power (frac > null_p96) |
|---|---|---|---|---|
| full MIMIC | 0.0083 (0.0064) | 0.0189 | 0.358 (0.337) | **1.00** |
| full SCID  | 0.0084 (0.0070) | 0.0208 | 0.347 (0.317) | **1.00** |
| bounded-short | 0.0196 (0.0161) | 0.0516 | 0.029 (0.002) | **0.125** |

Full support has strong, well-separated power on both sources (every coupled draw's `d` ≫ the null 96th pct).
Bounded-short has essentially none (coupled ≈ null; only 12.5% exceed the null 96th pct).

**(e) S8 phase interaction — no cross-load.** With only burst_timing coupled, `S8_density`/`S8_class` stay PASS at
value 0.0 (vs 0.006 on a matched null): reassigning inter-cluster gaps does not move the class-occupancy phase
check. Burst-timing does not spuriously load S8.

## Predeclared decision: boundary-short × burst-timing (Pi rev-2 §7 — decide now from dev evidence)

**Frozen power criterion (dev-only, predeclared):** an SD (statistic × regime) cell is retained only if its
development detection power against the `@0.5` component alternative is ≥ 0.5 (fraction of coupled draws with
`d` above the matched-null 96th percentile). Applied to burst-timing:
- full MIMIC / full SCID: power 1.00 → **RETAIN**.
- boundary-short: power 0.125 (< 0.5) → the statistic carries no usable burst-timing signal at L≤7 →
  **EXEMPT (property-specific, un-calibratable)** on the boundary-short support only. Boundary-short is a
  predeclared property-specific support control, so this exemption is admissible; it is a property of the support,
  decided from development evidence, NOT a convenience exemption of a FAIL. Core/full-support burst-timing cells
  are non-exemptible and RETAINED.

(S3_loggap, the second burst-timing subcheck, is assessed by the same criterion in the rev-3 registry.)

## Provenance
Development-only (`v3-estimator-dev`, seeds 90000–90039), N=3000, 40 matched replicates, 500 tie-formula checks.
No calibration/audit/evaluation seeds touched. Executable artifact + aggregate hash committed for reproducibility.
