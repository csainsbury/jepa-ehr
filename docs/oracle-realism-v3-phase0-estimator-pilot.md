# v3 Phase-0 estimator micro-pilot — results and estimator choice (rev-3, exact profiles)

**Purpose (Pi rev-2 §6, rev-3 §3):** freeze the burst-timing dependence estimator on **development-only seeds**,
with a committed reproducible script, on the **EXACT registered source profiles** (`scid_scale_control`,
`mimic_scale_control` — single-profile draws, not a 3-mu smoke), measuring `S3_tau` AND `S3_loggap` separately. No
calibration/audit/evaluation seeds. Supersedes the first-6-boundary and 3-mu-smoke pilots.

**Reproducible artifact:** `scripts/oracle_realism_v3_phase0_pilot.py` — dev namespace `v3-estimator-dev`, frozen
seeds `90000..90039`, N=3000/side. Aggregate result hash: `0e1680fc5422cd7770d1bd63b284b9c07978f71c54373f5db0bdb760e8983790`.

## Frozen estimator (burst-timing dependence, `S3_tau` replacement)

Phase-spanning capped, tie-corrected pooled Kendall tau-b. Per sequence, extract positive-gap boundaries as pairs
`(x = preceding cluster size, y = gap)`. **Cap (phase-spanning, frozen):** if `m ≤ 6` use all boundaries; else the
6 distinct quantile-spaced indices `sorted({ round(v) : v in linspace(0, m−1, 6) })` — spanning the WHOLE sequence.
Over within-sequence pairs of the selected boundaries (never across sequences), accumulate the STANDARD tau-b
components summed across sequences:

```
C − D = Σ_pairs sign(x_i−x_j)·sign(y_i−y_j)         # pairs tied in x or y contribute 0
n0    = Σ_s C(m_sel_s, 2);  n1 = Σ_s #{tied in x} (all x-ties incl joint);  n2 = Σ_s #{tied in y}
T_pool(sample) = (C − D) / sqrt((n0 − n1)(n0 − n2))
```

`d = |T_pool(cand) − T_pool(ref)|`. **Property claim (frozen):** a *capped, phase-spanning, pair-count-weighted
within-sequence rank dependence* — not first-6 (early), not uncapped, not equal-per-sequence. For the SD gate the
null is the per-draw between-group sequence-label permutation computed at evaluation (Pi rev-3 §1); this pilot does
NOT claim a calibrated tail.

## Evidence (exact-profile dev seeds; aggregate hash `0e1680fc…`)

**(a) Formula — exact match to scipy tau-b incl. joint ties:** `max |T_pool_1seq − scipy.kendalltau| = 0.0` over
500 random tie-heavy `(x,y)`. Standard `n1`/`n2` (all x/y ties, joint included) reconcile exactly.

**(b) Phase coverage — the cap spans the sequence.** Exact MIMIC, 2,885 sequences with `m>6`: mean normalized
selected-boundary position **0.086** for the rejected first-6 cap vs **0.500** for phase-spanning.

**(c) Contribution concentration — fixed on the exact profiles** (inverse-Simpson effective fraction; top-1% pair
share):

| regime | uncapped | phase-spanning cap |
|---|---|---|
| MIMIC-scale (exact) | 0.034 (0.365) | **0.989 (0.010)** |
| SCID-scale (exact)  | 0.083 (0.255) | **1.000 (0.010)** |

**(d) Source-wise power — `S3_tau` (pooled tau) AND `S3_loggap` (existing coarsened check), each separately**
(40 dev replicates; matched-null vs `burst_timing@0.5`-coupled):

| regime | S3_tau: null p96 → coupled (min) | S3_tau power¹ | S3_loggap: null→coupled status; val | S3_loggap power² |
|---|---|---|---|---|
| full MIMIC | 0.019 → 0.36 | **1.00** | 40 PASS → 40 FAIL; 0.028→1.05 | **1.00** |
| full SCID  | 0.022 → 0.35 | **1.00** | 40 PASS → 40 FAIL; 0.026→1.14 | **1.00** |
| bounded-short | 0.052 → 0.029 (min 0.002) | **0.125** | 33P/2F/5NE → 33P/2F/5NE; 0.053→0.054 | **0.05** |

¹ fraction of coupled draws with `d` above the matched-null 96th pct (proxy — see criterion note).
² fraction of coupled draws detected (status FAIL). On bounded-short the coupled S3_loggap status distribution and
value are **indistinguishable from the null** (and partly NE): no usable burst-timing signal at L≤7.

**(e) S8 phase interaction — no cross-load.** With only burst_timing coupled, `S8_density`/`S8_class` = 0.0 PASS
(vs 0.003/0.005 on a matched null): reassigning inter-cluster gaps does not move the class-occupancy phase check.

## Boundary-short exemptions — PROVISIONAL, Δ-aligned criterion pending (Pi rev-3 §3)

The retain/exempt criterion is Δ-aligned to the MM effect rule: `power = P_dev[ d_c > Δ_c | @0.5 ]`, **retain iff
≥ 0.5** (a v3 design choice informed by dev evidence, not predeclared before the pilot). The numeric `Δ` table does
not exist yet, so exemptions are **provisional** until Δ-aligned evidence is routed. Development evidence, each
subcheck decided SEPARATELY:
- full-support `S3_tau` and `S3_loggap`: power 1.00 → **RETAIN** (non-exemptible, core).
- boundary-short `S3_tau`: proxy power 0.125 → **provisional EXEMPT** (property-specific support).
- boundary-short `S3_loggap`: power 0.05, coupled indistinguishable from null (+ partial NE) → **provisional
  EXEMPT** (property-specific; may additionally qualify as EXEMPT-degenerate where the cluster-bin coarsening
  collapses). Now supported by its own evidence (Pi rev-3 §3), not inherited from S3_tau.

Both boundary exemptions become frozen only when re-expressed against the numeric `Δ` table and routed.

## Provenance
Development-only (`v3-estimator-dev`, seeds 90000–90039), N=3000, 40 replicates, 500 tie-formula checks, exact
registered source profiles. No calibration/audit/evaluation seeds touched. Executable artifact + aggregate hash
committed for reproducibility.
