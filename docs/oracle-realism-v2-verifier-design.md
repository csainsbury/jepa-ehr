# Oracle realism v2 — executable-verifier DESIGN FREEZE, rev-3 (Pi NARROW REVISE folded)

Synthetic-only. DEV identity `m3a_design_dev_hash = 60c85b64…`; source
`clinical_jepa/eval/oracle_realism_v2_verifier_design.py`. Architecture is accepted; rev-3 folds the exactness
corrections. Re-routed for confirmation before coding step 3.

## Folded (your NARROW REVISE)
1. **Exact timestamp/cluster semantics.** Clusters = maximal runs under **exact raw timestamp equality**
   (`dt==0`), matching the registered extractor; only the positive-gap ECDF **support** is rounded to 8dp
   (right-continuous). Every positive adjacency must satisfy `t[i+1]>t[i]` after materialization; a positive
   gap lost under cumulative float addition is rejected/nudged before issuance. `L_total==1`: dt0 undefined —
   excluded from dt0 and the adjacency denominator (never `(L-K)/(L-1)`).
2. **Executable S9 + coarsening.** S9 is a frozen conjunction (zero/class contrasts ≤0.03; three KS gap
   clauses ≤0.05) with per-sequence seam/nonseam probabilities, ≥500 eligible sequences + ≥500 seam/nonseam
   adjacencies + ≥500 positive seam/nonseam gaps. Coarsening is a 6-step algorithm: sparse bins from the
   **reference** denominators only → merge highest-index sparse bin into its left neighbour (bin 0 into bin 1)
   → repeat → refuse if <3 bins → apply the final map unchanged to the candidate (candidate floor-fail ⇒
   NOT_EVALUABLE). S1_tau is **one source-level** τ-b; S2 is `F(x)=mean_i F_i(x)`; S3 adjacent-pair floor is
   **per retained cluster-size bin**.
3. **Honest exact coupling constructions.** Replaced prose with finite-pool constructions —
   `burst_count_length` uses the comonotone-cycle assignment of sorted K to sorted L (preserves the L and K
   multisets exactly). Marginal preservation is **empirically required (tested ≥24/25)**, not asserted "by
   construction"; any preservation failure ⇒ DESIGN FAIL / re-gate, never threshold tuning. Each component
   freezes pre/post-state, integer `s→units`, tie-breaks/refusal, RNG derivation + draw order, composition
   order, and short-sequence / structural-zero behaviour. S4↔S7 cross-loading is **recorded** (not
   orthogonalized) and left to the Jacobian/collision tests. `SOURCE_SWAP` is a concrete pair (mimic length ×
   named scid-scale class/run/gap laws; expected failures enumerated; cannot trigger D).
4. **Ablation orientation.** reference = one component at 0.5; candidate A (null-independent) must **fail** the
   primary row; candidate D-recovery (independent impl at 0.5) must **pass** the full row; non-attributed +
   marginals + S2 + S8 + S9 pass ≥24/25 (allowed-sensitive checks exempt). `burst_count_length` primary-fails
   `S1_density` with `S1_tau` allowed-sensitive.
5. **Identifiability.** Affine standardization (no logit, no epsilon); **central FD at all interior grid points
   incl 0.55** (backward reserved for the actual 0.60 boundary); deterministic nearest-grid recovery
   (whitened-L2, lexicographic menu-order tie-break — no local LS); covariance ridge
   `Σ_λ = Σ + 1e-3·trace(Σ)/d·I`; named `NUISANCE_PROFILES`; explicit compute cap (≤8 CPU-h, ≤32 GB; no
   adaptive grid reduction after results).
6. **Claim wording.** Top-level now carries the **two distinct claims** (never recombined) plus explicit
   negatives: *NO real joint-envelope claim; NO confirmatory realism claim; NO joint-process claim; the
   independent fixture is synthetic-recovery infrastructure, not evidence dev-seen TRAIN has these
   dependencies.*

## Confirm
Requesting ACCEPT (or any remaining narrow correction). On accept I implement the executable verifier +
independent fixtures (step 3), run the sims (step 4), and route the implemented verifier for your M3a final
review (step 5). M2 stays blocked.
