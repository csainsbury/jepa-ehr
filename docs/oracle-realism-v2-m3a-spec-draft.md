# Oracle realism v2 — M3a verification-spec DRAFT (for Pi's ruling)

Synthetic-only. This is the verifier declared **before** the generator (freeze-before-fit). It is a
**DRAFT / DEV** identity (`m3a_spec_dev_hash = 57ecfc93…`); the FINAL frozen M3a identity is minted only after
Pi rules. Source of truth: `clinical_jepa/eval/oracle_realism_v2_spec.py`. No fitting / sampling / target
comparison is implied. Admissible claim after any pass: *"matches the declared marginal + cross-statistic
envelope,"* never the joint process.

## Frozen bins (shared)
- `LENGTH_BINS = (2-3),(4-5),(6-7),(8)` — used by S1/S5/S6 (L=1 is M0b, excluded).
- `CLUSTER_SIZE_BINS = (1),(2),(3-4),(5-8)` — used by S2/S3/S7.

## Six marginal checks (thresholds reused verbatim from the v1 envelope)
length_ks ≤ 0.05, class_tv ≤ 0.05, count_ks ≤ 0.05, occupancy_abs ≤ 0.03, delta_t_zero_abs ≤ 0.02,
positive_gap_ks ≤ 0.05.

## Cross-statistics S1–S7 (thresholds DRAFT — flagged PROPOSED)
| ID | Definition | Bins | Proposed threshold | Denom floor | Refusal |
|----|-----------|------|--------------------|-------------|---------|
| S1 | E[cluster-count K \| length-bin] + Kendall τ(L,K) | length | \|ΔE[K\|bin]\|≤0.25, \|Δτ\|≤0.05 | 500 | <floor ⇒ NOT_EVALUABLE; coarsen small cells |
| S2 | ECDF of Δt=0 cluster-run sizes, KS | cluster-size | KS≤0.05 | 500 | ″ |
| S3 | mean positive gap \| preceding-cluster-size bin + τ(size,gap) | cluster-size | \|Δτ\|≤0.05 | 500 | ″ |
| S4 | P(same class\|same cluster) − P(same class\|adjacent) | — | \|·\|≤0.03 | 500 | ″ |
| S5 | E[occupancy \| length-bin] vs M0b cap min(L,5)/5 | length | \|·\|≤0.03 | 500 | ″ |
| S6 | class TV between length terciles (**mandatory**) | terciles | TV≤0.05 | 500 | ″ |
| S7 | E[n_distinct/min(C,5) \| cluster-size-bin] | cluster-size | \|·\|≤0.03 | 500 | ″ |

S6 mandatory and S7 added per Pi P-B. S5 compares against the M0b occupancy cap, not 1.0.

## Parameter → statistic attribution (also the escalation component→check map)
- `burst_count_length → S1`; `burst_timing → S3`; `mark_burst_tie → S4`; `cluster_size_mark_diversity → S7`.
- marginal laws: `length_law → length_ks,S5,S6`; `class_law → class_tv,S6`; `cluster_size_law → S2,count_ks`;
  `gap_law → positive_gap_ks,S3`; `dt0_law → delta_t_zero_abs,S2`.

## Identifiability battery (declared here; RUN at M3)
Rule `n_dependence_params ≤ n_independent cross-stat DoF (after constraints/covariance)`, PLUS: full-column-rank
sensitivity **Jacobian** d(cross-stats)/d(dependence-params) at every registered profile (rank_tol 1e-6,
PROPOSED); **grid-recovery** of each dependence param on a joint grid with marginal nuisance varied
(recover_tol abs 0.05, PROPOSED); **collision search** (no two distinct dependence settings → within-threshold
identical cross-stat vectors). Registered profiles: scid_like, mimic_like, independent_A_baseline,
copula_zeroed_null.

## Power (DRAFT)
target_power 0.80; the attribution-mapped mis-specified control must FAIL its check at ≥ power; n_seeds 5;
seed aggregation = median-of-seeds satisfies threshold AND mis-specified control median breaches it.

## Source conjunction
Every check (marginals + S1–S7) must pass for EVERY required source (SCID, MIMIC); conjunction over sources ×
checks; any NOT_EVALUABLE source-check ⇒ whole spec NOT_EVALUABLE (never a pass).

## Escalation + immutable ledger
M2 binds `A_independent` first (baseline). D is controls-driven only: on an attribution-mapped control failure,
escalate the smallest super-set of components covering the failed checks (ties broken by frozen attribution
order), mint a new `D_copula` identity, append ONE immutable ledger entry (round, failed_checks,
escalated_components, new_identity, battery_result_hash, seed_set), and re-run the FULL battery. iteration_cap 3.
Decision basis: known-ground-truth control battery ONLY; TRAIN-target diagnostics non-decisional.

## Specific questions for Pi
1. Cross-stat thresholds/bins (S1–S7): confirm or revise the PROPOSED values (esp. S1 abs-E[K] 0.25 and the τ
   thresholds; and the CLUSTER_SIZE_BINS / LENGTH_BINS edges).
2. Identifiability tolerances: rank_tol 1e-6, grid recover_tol abs 0.05, and the collision-search definition —
   sufficient? Any additional registered profile?
3. Power: target 0.80, n_seeds 5, the worst-of-both seed-aggregation rule — accept or tighten?
4. Escalation: iteration_cap 3 and the smallest-super-set tie rule — accept?
5. Any statistic still missing before the freeze (e.g. a position-nonstationarity guard), or is S1–S7 + the six
   marginals the complete admissible envelope?

On your ruling I apply the revisions, bump to the FINAL frozen identity (`m3a_spec_frozen_v1`), and only then
is M2 Option-A fitting unblocked. Governed stop line holds.
