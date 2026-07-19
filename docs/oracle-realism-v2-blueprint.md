# Oracle realism v2 — build blueprint (Option D, scout + Fable-review folded)

Blueprint for the redesigned aggregate-realism layer after v1 falsified on real TRAIN marginals
(`docs/oracle-calibration-result-v1.md`; options in `docs/oracle-realism-redesign-options-v1.md`). Governs a
**synthetic-only** build: no governed data, no HDF5 reopen, no execution against real data until a separate
pre-registered confirmatory gate. Direction: **Option D — the frozen order mechanism wrapped in a
source-conditioned, variable-length, compound-burst copula realism envelope.**

Incorporates Cog scout `20260719T141540Z-jepa-e5b45c1e` (verdict PROMOTE) and a Fable pre-implementation
review (verdict REVISE-before-implementation). Both are folded below; the review's seven required revisions
are all reflected. **Implementation has NOT begun.**

## Novelty frame (scout)

The contribution is the **frozen-contract semi-synthetic order-recovery oracle with a certified realism
envelope** — NOT a generic EHR-JEPA trajectory model. Clin-JEPA (Yang et al. 2026) and the "SMB-Structure"
line are standing prior-art guardrails; differentiate on the frozen order/nuisance certification contract.

## Two generators — the distinction M0 hinges on (Fable #1)

- **META stack** (`oracle_meta_gen.generate_meta_cell`, references in `oracle_meta_bayes`): the C=5
  order-recovery certification references (`r0_pairwise`, `pi_star_pairwise`).
- **LITERAL stack** (`oracle_literal_gen.generate_literal_cell`): the token/event generator that (a) the v1
  realism BASE uses and v2 will MODIFY (`_apply_calibration_layer`), and (b) the certification path itself
  uses (`oracle_verdict`, `oracle_references.eo1_r_bayes` call `generate_literal_cell`; `per_sequence_eo1`
  scores literal `true_order`). Family sets differ (`E_offgrid_heavytail` meta vs `E_offgrid_nonlinear`
  literal).
The variable-length restriction and every seam edit happen on the LITERAL stack.

## Non-negotiable contracts

1. Preserve the frozen ORDER mechanism and NUISANCE-control boundaries.
2. New generator version/schema/identity; never reinterpret the v1 FAIL as a pass.
3. Explicitly, source-conditioned, model: variable length, class composition (structural zeros), cluster
   counts, occupancy, simultaneity (Δt=0), positive-gap distribution.
4. Synthetic-only development; NO governed re-read. Confirmatory realism only via a separate PRE-REGISTERED
   locked/external aggregate gate.
5. No latent-mechanism / transfer / counterfactual / causal / order-certification claim.

## ★ Identifiability — corrected wording (Fable #2)

Two distinct notions: **(i) parameter identifiability WITHIN the declared sparse copula family** — achievable,
and what M3 recovery tests; **(ii) identification of the true joint event process from aggregates** —
impossible from any finite predeclared summary (a sequence-level random-effects mixture reproduces dependence
summaries without genuine coupling; tail dependence / position-nonstationarity beyond the declared summaries
are invisible). The admissible claim after any pass is *"matches the declared marginal + cross-statistic
envelope,"* NEVER "matches the joint process." The predeclared cross-statistics must be jointly sufficient to
recover the declared copula parameters, demonstrated by M3 grid-recovery at predeclared tolerance. Rule:
#dependence-parameters ≤ #independent predeclared cross-statistic DoF.

### Predeclared cross-statistics set (per source; frozen bins/thresholds/denominator floors)

| ID | Statistic | Identifies |
|---|---|---|
| S1 | E[cluster-count K \| length L] on frozen coarse L-bins + Kendall τ(L,K) | burst-count/length coupling |
| S2 | ECDF of Δt=0-run (cluster) sizes, KS check | the compound/burst-size law directly |
| S3 | mean positive gap by preceding-cluster-size class (or τ) | burst-timing coupling (MIMIC 0.89/0.95) |
| S4 | P(same class \| same cluster) − P(same class \| adjacent clusters) | mark–burst tie (same-class panels) |
| S5 | E[occupancy \| L] on the same L-bins | composition–length coupling (occ is L-censored) |
| S6 (opt) | class TV between length terciles | length-dependent class mix |

**Computability (trust the code):** NONE of S1–S6 are in the committed contract — `_AGG_FIELDS` /
`AggregateStats` carry marginals only (only `n_events/n_clusters` = mean cluster size is derivable). For M3
(synthetic) they are new pure functions in the frozen verification spec. For M4 (any real read) each is a NEW
governed aggregate field ⇒ new extraction schema hash / question / run id / clearance, so M4 pre-registration
must include the EXTENDED EXTRACTION CODE, not just thresholds — and a small-cell coarsening / min-cell-count
policy (joint histograms have a higher re-identification surface than marginals) disclosed in the clearance.

## Route decisions (scout)

- **PROMOTE:** compound-burst copula envelope (bursts/clusters first, then tied marks + occupancy);
  verification-first; the order-restriction proof as a hard prerequisite; the frozen-contract novelty frame.
- **VERIFY:** Option A independent marginals as a falsifiable **baseline**; latent-factor point-process as
  possible D guidance; a JEPA-Reasoner-style decoupled latent step as a LATER reader/speaker extension only.
- **PARK:** a JEPA latent realism scorer (until the explicit generator passes its gates); a neural marked TPP
  (until A/D fail controlled tests AND an external confirmatory target exists).
- **REJECT:** latent JEPA as the event-ORDERING mechanism; an autoregressive event head (Option C); generic
  "introduce Clinical-JEPA" novelty; real-sequence bootstrap/replay; treating a TRAIN pass or v1 as confirmation.

## Milestones (each a gate; order corrected per Fable #1c)

### M0 — order-restriction invariance: DONE for META, OPEN for LITERAL (Fable #1)
Proven pairwise-local on the META stack (exact-0.0 full-vs-restricted `r0`/`pi_star`, sub-ranking =
restriction, `invariant_hash` unchanged; `tests/test_oracle_order_restriction.py`). NOT yet established on the
LITERAL stack where v2 edits + verdicts live. **M0 is not complete until a LiteralCell mirror test passes**
(restricted `s_true` sub-ranking = restriction; restricted `nuisance_u` = exact column slice for BOTH nuisance
cells; `eo1_r0`/`eo1_r_bayes` on a restricted literal cell = pair-restriction of the full-cell values) AND the
support-floor half (below) is gated.

### M0b — support-floor / min-length policy (Fable #1b)
A fitted length law yields short and L=1 sequences (real length KS was 1.0 = broad real lengths). Define and
gate: per-cell and per-pair support ≥ `ORDER_SUPPORT_FLOOR` or an explicit `SUPPORT_STARVED` tag (never
silent); L=1 (vacuous order, undefined adjacency) and L<5 (occupancy cap L/5) handling.

### M1 — v2 identity scaffolding (no behaviour change)
Extend the default-path byte-pin (Fable #1e): full-digest hashes of EVERY array field of fixed-seed default
`LiteralCell`s, all 5 families × both nuisance cells — committed BEFORE any v2 edit. New module (do not grow
the literal default path) with `realism_v2_schema_hash()` binding {version, per-source profile (length / class
+ structural-zero mask / cluster-size / gap / Δt=0 laws), sparse copula descriptor, cross-statistics set with
bins/thresholds, required sources}. Bump/branch `CALIB_ADAPTER_VERSION` and fold the schema into the adapter
identity. Which identities MAY move: `extraction_code_identity` (file-byte hash), adapter/`realism_v2` schema
hashes. Which are PINNED: `invariant_hash`, `ORACLE_EVALUATOR_IDENTITY`, `base_schema_hash` /
`generator_fit_schema_hash` / `calibration_schema_hash` (unless an explicit, gated M1 change). Gate: default
byte-pin + literal M0 test + Pi design pre-review (record the agent-room thread id here).

### M3a — FREEZE the evaluator BEFORE any fitting (Fable #1c — reordered ahead of M2)
Freeze, and hash into `realism_v2_schema_hash`, the verification spec: the six marginal checks PLUS S1–S6
cross-statistics (bins/thresholds/floors/refusal), the escalation **attribution map** (component→check, with a
tie rule), the escalation **decision basis = the known-ground-truth control battery ONLY** (TRAIN-target
comparisons are labelled exploratory and CANNOT drive design, per Fable #1d and the v1 result note), an
iteration cap + escalation ledger, and a predeclared **power statement per threshold** (the mis-specified
control must fail it at stated power; multi-seed with a predeclared aggregation rule).

### M2 — Option A baseline (source-conditioned independent marginals)
Generalize the adapter seam (marks + timestamps only). Seam rules (Fable #4), each a test:
- length/subset draws come EXCLUSIVELY from the adapter RNG `arng`, NEVER from `rng` (a new `rng` draw shifts
  the nuisance stream for every cell);
- generate at full L then RESTRICT for order; `future_events` recomputed as `argsort(argsort(s_true[mask]))`,
  never sliced; restricted `nuisance_u` == exact column slice (0.0 tol) for both nuisance cells;
- timing/marks generated FRESH at realized length inside the adapter (deleting items from a full-L cell thins
  clusters and distorts the very stats measured);
- Δt=0 needs a genuinely new parameterization (induced by the S2 cluster-size law), not a widened grid under
  the old 0.9 clip (MIMIC 0.8914 leaves 0.0086 headroom).
Emit per source: length law, class Dirichlet-multinomial with HARD structural zeros, cluster-size + gap +
Δt=0 laws. Treated as a FALSIFIABLE BASELINE. Include M0b support accounting. Gate: default byte-pin still
passes; all order/nuisance/context fields `array_equal` to default across all families; then the M3 battery.

### M3 — run the frozen battery (synthetic-only)
Self-recovery at a known profile; **known-ground-truth recovery** (SCID-like / MIMIC-like ground truths →
recover parameters + pass marginals AND cross-stats); **negative controls** incl. the DEPENDENCE-ablation
teeth (Fable #3): a copula-zeroed profile with correct marginals MUST fail ≥1 cross-stat against a coupled
ground truth; a right-marginals/wrong-copula profile fails ONLY cross-stats (no cross-talk); source-swap must
fail on a NON-degenerate check too (SCID zero-state makes class-TV swap trivial); per-check ablation. Escalate
ONLY the attribution-mapped failed components to the D compound-burst copula; each escalation bumps the v2
identity, re-runs the FULL battery, and is ledgered under the cap.

### M4 — pre-registered locked/external confirmatory gate (governed; BLOCKING open item)
TRAIN targets are development-seen, so freeze + pre-register the v2 generator, thresholds, and EXACT statistics
(marginals + cross-stats) before any locked/external read; the target must NOT be the seen TRAIN marginals.
Pre-registration must include the EXTENDED extraction code (S1–S6 are new governed fields) AND the expected
content digests / target manifest identity committed BEFORE the read (fixes v1's `UNVERIFIED` provenance,
Fable #3). Requires new question/schema/run-id/generator identity, spent-run accounting, and its own Pi
micro-gate + policy-population + result gate. **The locked/external target is UNSPECIFIED — this is a blocking
prerequisite (Chris/Pi decision), not merely open, because escalation must never be TRAIN-target-driven.**

## Claim boundaries (per milestone)

An M3 pass authorizes only M4 pre-registration — no realism claim of any kind. Only an M4 pass yields a
confirmatory claim, and only "matches the declared marginal + cross-statistic envelope on the locked/external
target," never a joint-process, latent-mechanism, or causal claim.

## Open / blocking items

- **BLOCKING:** the M4 locked/external confirmatory target (Chris/Pi decision) — required before escalation is
  allowed to be anything but controls-driven.
- The exact S1–S6 bins/thresholds/power (frozen in M3a).
- The ragged-vs-mask representation decision (recommend mask + full-L internal arrays, restriction only at
  emission/scoring — preserves M0 semantics; rectangular (N,L)/(N,L,L) contracts pervade the code).
- Whether certification cells stay fixed-L=8 (if so, most seam risk collapses — decide + state).
- Literature citations to confirm (Shchur et al. 2020).

## Stop lines

Until an M4 pre-registered gate separately passes: `APPROVED_ORACLE_POLICY` empty; aggregate-read policy
retired; TEST sealed; no sealed certification, oracle-T4 training, manifest issuance, governed T4, or any
governed aggregate read. The v1 FAIL is never reinterpreted as a pass.
