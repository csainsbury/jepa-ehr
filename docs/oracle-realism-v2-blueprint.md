# Oracle realism v2 — build blueprint (Option D, scout + Fable-review folded)

Blueprint for the redesigned aggregate-realism layer after v1 falsified on real TRAIN marginals
(`docs/oracle-calibration-result-v1.md`; options in `docs/oracle-realism-redesign-options-v1.md`). Governs a
**synthetic-only** build: no governed data, no HDF5 reopen, no execution against real data until a separate
pre-registered confirmatory gate. Direction: **Option D — the frozen order mechanism wrapped in a
source-conditioned, variable-length, compound-burst copula realism envelope.**

Incorporates Cog scout `20260719T141540Z-jepa-e5b45c1e` (verdict PROMOTE), a Fable pre-implementation
review (verdict REVISE-before-implementation), and a **Pi M1 design pre-review** (agent-room thread
`thr-20260719T155635Z-fff46633`, artifact `jepa-pi-oracle-realism-v2-m1-prereview.md`; verdict **REVISE
before M3a — no M2 fitting**). All three are folded below; Fable's seven revisions and Pi's four answers +
nine M3a preconditions are all reflected. **v2 generator behaviour / fitting has NOT begun** (proof tests +
identity scaffolding have landed; no sampling law, parameter fit, or target comparison exists).

**Agreed work order (Pi assent-with-reorder, thread `thr-20260719T155635Z-fff46633`):** (1) pins first — v1
calibrated-path golden digests + exact dev-scaffold hash + assert frozen v1 identities; (2) M0 as an
order-core boundary only (fixed-L rejection guard + additive `RestrictedOrderCore` primitive); (3) identity
split (common marginal schema + distinct `A_independent`/`D_copula`, v2 adapter interface/schema STUB only);
(4) M0b; (5) full M3a freeze; (6) only then implement + fit M2 Option A; (7) escalate to D only under frozen
M3 rules + a new identity.

### Pi M1 pre-review — folded conditions ledger (traceability)
Pi accepted the boundary (fixed-L certification + emission-only variable length), the full default byte-pin
as the primary tripwire, and the additive scaffold (no v1 identity moved; independent battery 12 passed;
`realism_v2_schema_hash = ffb4f3d8a8b50bd3…` confirmed). Required corrections, folded into the milestones:
- **P-A** (M0): drop the false "context-only" rationale; add a fail-hard certification rejection guard;
  remove the restricted-literal `eo1_r_bayes` requirement; replace the `_restrict` proof fixture with a
  production restriction primitive. See M0.
- **P-B** (cross-stats): S6 mandatory when length-dependent class mix is modelled; add **S7**
  (cluster-size-conditioned class diversity) or drop+exclude the coupling; count DoF after constraints;
  Jacobian rank + grid-recovery + collision search. See ★ Identifiability and M3a.
- **P-C** (pins): add a v1 **calibrated-path** pin; branch v2 behind a distinct versioned adapter (never
  mutate v1 in place); pin the scaffold hash renamed as a *development* identity. See M1.
- **P-D** (M3a): **separate Option-A and Option-D identities** (common marginal schema + distinct
  `A_independent`/`D_copula` hashes; M2 binds A first). See M2/M3a and the nine preconditions.

### Work-order progress (Pi-assented order)
- **Step 1 (pins) — DONE** (`7df490f`): v1 calibrated-path golden digests (SCID/MIMIC × 5 families × 2
  nuisance), dev-scaffold pin, frozen-v1-identity asserts.
- **Step 2 (M0 order-core boundary) — DONE** (`dae191b`): `RestrictedOrderCore`/`restrict_order_core`
  (emission-free) + `assert_canonical_certification_cell` fail-hard guard at all reference/verdict
  entrypoints, bound into the additive `v2_certification_boundary_hash = b33c2d9f…`. Full suite 518 passed.
- **Step 3 (identity split) — DONE**: shared `V2_MARGINAL_SCHEMA` (`9dc5a2ac…`) + distinct dev/final
  identities `A_independent` (`b299a779…`/`c0d9f136…`) and `D_copula` (`43b55944…`/`4fb65d5e…`), adapter
  INTERFACE stub (`463dd88f…`, behaviour forbidden pre-M3a). The umbrella dev-scaffold hash bumped
  INTENTIONALLY `ffb4f3d8… → 704b079a…` (A/D split + corrected certification rationale); boundary + frozen
  v1 identities unmoved.
- **Step 4 (M0b support/min-length accounting) — DONE**: `account_order_support` / `restricted_core_support`
  → `OrderSupportAccounting` with never-silent statuses (`SUPPORTED`/`SUPPORT_STARVED`/`VACUOUS_ORDER`),
  per-cell + per-pair denominator floors (500), and the L<5 occupancy cap L/5 as an explicit accounting flag;
  bound into the additive `m0b_support_policy_hash = 876bffb6…`. Restricted cores are classifiable here but
  still rejected by the certification guard (complementary roles).
- **Next:** step 5 (full M3a freeze — 6 marginals + S1–S7, identifiability battery, escalation ledger), then
  step 6 (M2 Option A fit). M3a is the BLOCKING freeze before any M2 fitting.

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

**Fixed-L certification boundary — corrected rationale (Pi P-A).** Certification stays fixed-L=8 and variable
length is emission-only. The *reason* is a governance decision, NOT the earlier (wrong) claim that
`GoodContextRecipe.predict_latent` is context-only: `_design` (oracle_recipe.py:99–111) DOES read
`item_features` and forms context×item interaction blocks; the recipe is fixed-L only because of the hard-coded
`self._L` reshape (line 147). Feeding a restricted/variable-L cell to `_design` builds `X` at subset-L then
reshapes to L=8 — i.e. it *breaks*, which is an incidental reshape failure, not a boundary. Therefore the
fixed-L boundary must be enforced by an explicit fail-hard rejection guard (M0), never left to rely on the
reshape happening to fail.

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
envelope,"* NEVER "matches the joint process."

**Sufficiency is more than a parameter count (Pi P-B).** The rule `#dependence-parameters ≤ #independent
cross-statistic DoF` is NECESSARY but NOT sufficient. Before declaring within-family identifiability, freeze an
exact **parameter→statistic attribution table** and establish BOTH (a) local rank — a full-column-rank
sensitivity/Jacobian of the cross-statistics w.r.t. the dependence parameters at every registered profile —
and (b) global recovery / no-collision on a joint grid with the marginal nuisance parameters varied
(grid-recovery + collision search; a single known profile is insufficient). Count independent vector
*components* after constraints/covariance, not the six/seven statistic labels as that many DoF. These recovery
tests are what M3 exercises; they are declared here and frozen in M3a.

### Predeclared cross-statistics set (per source; frozen bins/thresholds/denominator floors)

| ID | Statistic | Identifies |
|---|---|---|
| S1 | E[cluster-count K \| length L] on frozen coarse L-bins + Kendall τ(L,K) | burst-count/length coupling |
| S2 | ECDF of Δt=0-run (cluster) sizes, KS check | the compound/burst-size law directly |
| S3 | mean positive gap by preceding-cluster-size class (or τ) | burst-timing coupling (MIMIC 0.89/0.95) |
| S4 | P(same class \| same cluster) − P(same class \| adjacent clusters) | mark–burst tie (same-class panels) |
| S5 | E[occupancy \| L] on the same L-bins | composition–length coupling (occ is L-censored) |
| S6 | class TV between length terciles | length-dependent class mix |
| S7 | within-cluster class diversity \| cluster-size bin: E[n_distinct_classes / min(C,5) \| size-bin] (or mixed-class-cluster prob vector) | how class diversity is allocated across burst sizes |

**S6 is now MANDATORY, not optional (Pi P-B):** it cannot be optional while a length-dependent class-composition
parameter is claimed identifiable.

**S7 added (Pi P-B):** S4's single same-class contrast plus the sequence-level S5 can BOTH be matched while
class diversity is mis-allocated between large and small bursts — precisely the high-occupancy/burst ambiguity
Option D exists to resolve. Therefore either (a) include S7, OR (b) **remove the cluster-size→mark-diversity
coupling from D and explicitly exclude it from the claim** — the coupling must not be present-but-untested.
Default: include S7.

**Computability (trust the code):** NONE of S1–S7 are in the committed contract — `_AGG_FIELDS` /
`AggregateStats` carry marginals only (only `n_events/n_clusters` = mean cluster size is derivable). For M3
(synthetic) they are new pure functions in the frozen verification spec. For M4 (any real read) each is a NEW
governed aggregate field ⇒ new extraction schema hash / question / run id / clearance, so M4 pre-registration
must include the EXTENDED EXTRACTION CODE, not just thresholds — and a small-cell coarsening / min-cell-count
policy (joint histograms have a higher re-identification surface than marginals) disclosed in the clearance.
Also freeze, per source: bin edges, denominator floors, small-cell coarsening, sequence-clustered uncertainty,
multi-seed aggregation, multiplicity handling, and the source conjunction rule (Pi P-B).

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

### M0 — order-restriction invariance: DONE for META; LITERAL property proven, REVISE-before-complete (Pi P-A)
Proven pairwise-local on the META stack (exact-0.0 full-vs-restricted `r0`/`pi_star`, sub-ranking =
restriction, `invariant_hash` unchanged; `tests/test_oracle_order_restriction.py`). The LITERAL
order-restriction PROPERTY is also demonstrated (`tests/test_oracle_literal_order_restriction.py`, 5 families ×
2 nuisance cells), and Pi accepts the boundary. But Pi requires the following corrections before M0-literal is
marked complete **for the order-restriction property only** (M0b remains open):

1. **Fix the rationale** — see the corrected fixed-L note above; do not lean on `predict_latent` being
   context-only (it reads `item_features`).
2. **Explicit fail-hard certification rejection guard** — a test proving: certification/verdict/reference APIs
   accept ONLY canonical `L_ITEMS=8` cells; a v2 emission mask / restricted cell CANNOT reach `eo1_recipe`,
   `eo1_r_bayes`, or governed certification; v2 restriction metadata is emission/evaluator-realism-only; and
   changing emitted length/items cannot change the fixed-L certification input/artifact. (Rejection, not
   reshape-failure.)
3. **Resolve the blueprint contradiction** — M0 previously required restricted-literal `eo1_r_bayes`, but the
   chosen boundary says variable-L never reaches that fixed-L recipe. That requirement is REMOVED and replaced
   by the rejection/separation test in (2). If variable-L certification is ever wanted, M0 REOPENS and the
   recipe/reference contracts must be redesigned and re-gated.
4. **Replace the `_restrict` proof fixture with a production restriction primitive.** The landed `_restrict`
   blindly slices every array whose second dim = L, leaving `future_events`/`cluster_ids`/`multiplicity`
   semantically stale, and its sub-ranking assertion is tautological (`rc.true_order` is the same slice). The
   production primitive (with test) — an additive `RestrictedOrderCore`/mask object, **NOT a partially sliced
   `LiteralCell`** — must: select items from full-L `s_true`/item/nuisance channels; RECOMPUTE
   `future_events = argsort(argsort(s_true_subset))` (never slice it); verify surviving pair signs against the
   full order; and NEVER re-standardize correlated nuisance. **It must NOT slice OR fabricate emission fields**
   — `future_timestamps`, `cluster_ids`, `multiplicity`, and marks are absent/unmaterialized at this stage
   (generating them fresh at realized length is M2 adapter behaviour under the frozen verifier, per Pi; doing it
   here would be premature generator behaviour before M3a). Exercise representative **contiguous and
   non-contiguous** subsets over lengths **2–8** (L=1 belongs to M0b).

Only after (1)–(4) is M0-literal complete for the order-restriction property; M0b (support floor) is still an
open gate below.

### M0b — support-floor / min-length policy (Fable #1b; Pi P-A owns L=1)
A fitted length law yields short and L=1 sequences (real length KS was 1.0 = broad real lengths). Define and
gate: support accounting with per-cell and per-pair support ≥ `ORDER_SUPPORT_FLOOR` or an explicit
`SUPPORT_STARVED` tag (never silent); **pair-denominator floors**; L=1 (vacuous order, undefined adjacency —
explicitly a M0b case, kept out of the M0 L=2–8 restriction test) and L<5 (occupancy cap L/5) handling.

### M1 — v2 identity scaffolding (no behaviour change) — LANDED, REVISE per Pi P-C
Landed (commit `c9f3b6e`): the full default-path byte-pin (every array field of fixed-seed default
`LiteralCell`s, all 5 families × both nuisance cells, full sha256, committed BEFORE any v2 edit); a new module
`oracle_realism_v2.py` with `realism_v2_schema_hash()` binding {version, per-source profile, sparse copula
descriptor, cross-statistics set, required sources}. Pi accepts the full default pin as the correct PRIMARY
tripwire and confirms no v1 identity moved. Required additions before M2 (Pi P-C):
- **Add a v1 CALIBRATED-path pin** — representative digests of the v1 calibrated path for BOTH source profiles
  across the literal families/nuisance cells. The default-only pin would not catch an edit that keeps
  `calib_knobs=None` on the default path but silently changes the historical v1 adapter path / result
  reproducibility.
- **Branch v2 behind a distinct versioned adapter/module/API with a NEW v2 adapter schema — never mutate the
  existing v1 adapter in place.** Keep tests pinning: v1 default path; v1 calibrated path; v2
  non-order/non-nuisance invariance; unchanged `invariant_hash`, evaluator identity, and every identity M1
  declares pinned.
- **Pin the scaffold hash exactly and rename it a DEVELOPMENT scaffold identity** (`realism_v2_scaffold_dev`),
  explicitly NOT the final frozen M3a verification identity.
Which identities MAY move: `extraction_code_identity` (file-byte hash), the new v2 adapter/`realism_v2` schema
hashes. Which are PINNED: `invariant_hash`, `ORACLE_EVALUATOR_IDENTITY`, `base_schema_hash` /
`generator_fit_schema_hash` / `calibration_schema_hash` (unless an explicit, gated M1 change).

**Gate status (Pi assent):**
- Pi design pre-review: **DONE** (thread `thr-20260719T155635Z-fff46633`, verdict REVISE-before-M3a, assented
  with reorder; conditions folded here).
- Landed default byte-pin + dev scaffold: **DONE** (commit `c9f3b6e`).
- M1 milestone gate: **OPEN** until the v1 calibrated-path pin, the corrected M0 boundary (P-A rejection guard
  + `RestrictedOrderCore` primitive), the `A_independent`/`D_copula` identity split, and the required guards
  all pass.

**Guard-integration condition (Pi).** The fail-hard guard may clarify the existing fixed-shape domain but must
NOT move any valid-path output; it must be bound into the v2 boundary identity and tested at EVERY public
certification/reference/verdict entrypoint (do not rely on the recipe reshape exception). If integrating the
guard would change a frozen evaluator identity or a valid-path artifact, STOP and re-gate that identity
explicitly.

### M3a — FREEZE the evaluator BEFORE any fitting (Fable #1c — reordered ahead of M2)
Freeze, and hash into the FINAL (non-dev) `realism_v2_schema_hash`, the verification spec. The nine Pi P-D
preconditions that must ALL hold before this freeze:
1. **Separate Option-A and Option-D identities.** The dev scaffold currently declares
   `join=sparse_compound_burst_copula` while the escalation says start at independent Option A. Freeze a common
   marginal schema PLUS distinct `A_independent` and `D_copula` variant hashes. M2 initially binds A; a
   controls-driven D escalation creates a NEW identity + ledger entry.
2. Land the explicit fixed-L certification rejection guard (M0 item 2).
3. Land the production restriction primitive (M0 item 4) and complete M0b (support accounting,
   `SUPPORT_STARVED`, L=1, L<5 occupancy cap, pair-denominator floors).
4. Decide rectangular-full-L+mask vs ragged — ACCEPT full-L canonical internals + explicit emission
   mask/restricted output; certification receives ONLY unmasked fixed-L cells.
5. Freeze the six marginal checks PLUS revised **S1–S7** — exact units/bins/thresholds/floors/refusal rules
   and aggregate-safe coarsening.
6. Freeze the dependence-parameter table, the identifiability/rank/recovery tests (Jacobian full-column-rank +
   grid-recovery + collision search), per-check power, seed aggregation, and failure controls.
7. Freeze the component→check escalation attribution, the tie rule, the iteration cap, and an immutable
   escalation ledger. TRAIN-target diagnostics remain non-decisional.
8. Keep the neural marked-TPP and latent-JEPA realism routes parked — no latent/AR replacement of the frozen
   ordering mechanism.
9. Keep M4 blocked until a non-TRAIN locked/external target AND its expected provenance identity are specified
   and separately gated.

Concretely the frozen spec hashes in: the six marginal checks PLUS S1–S7 cross-statistics
(bins/thresholds/floors/refusal), the escalation **attribution map** (component→check, with a tie rule), the
escalation **decision basis = the known-ground-truth control battery ONLY** (TRAIN-target comparisons are
labelled exploratory and CANNOT drive design, per Fable #1d and the v1 result note), an iteration cap +
immutable escalation ledger, and a predeclared **power statement per threshold** (the mis-specified control
must fail it at stated power; multi-seed with a predeclared aggregation rule).

### M2 — Option A baseline (source-conditioned independent marginals)
Binds the **`A_independent` identity** (distinct from `D_copula`, per Pi P-D-1); D is entered ONLY by a
controls-driven escalation in M3, which mints a new identity + ledger entry. Generalize the (new, v2-branched)
adapter seam (marks + timestamps only). Seam rules (Fable #4), each a test:
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
Pre-registration must include the EXTENDED extraction code (S1–S7 are new governed fields) AND the expected
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
- The exact S1–S7 bins/thresholds/power (frozen in M3a).
- **RESOLVED (Pi P-D-4):** ragged-vs-mask → full-L canonical internals + explicit emission mask/restricted
  output; certification receives ONLY unmasked fixed-L cells.
- **RESOLVED (Pi P-A):** certification cells stay fixed-L=8; variable length is emission-only, enforced by the
  fail-hard rejection guard (NOT by the reshape accident).
- The S7-vs-drop-coupling decision (default: include S7).
- Literature citations to confirm (Shchur et al. 2020).

## Stop lines

Until an M4 pre-registered gate separately passes: `APPROVED_ORACLE_POLICY` empty; aggregate-read policy
retired; TEST sealed; no sealed certification, oracle-T4 training, manifest issuance, governed T4, or any
governed aggregate read. The v1 FAIL is never reinterpreted as a pass.
