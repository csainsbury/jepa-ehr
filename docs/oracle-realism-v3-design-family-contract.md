# Oracle Realism Verifier — Design Family v3 (rev-20, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-19 (folds the rest of Pi rev-18 REVISE). The NORMATIVE design is §0–§9 + Delivered/Still-open below; the
**Changelog** records how each Pi ruling was folded. All work is development-only, synthetic-only; no calibration
build, reserved map-design draw, audit/evaluation seed, policy population, or run. There is **no production/trusted
readiness claim.** Registered mode is a **BLOCKED STUB**: its invariants are DECLARED and its structured assembly +
identity refusals are TESTED, but the registered STATISTIC path is UNIMPLEMENTED — it never runs a statistic, never
consumes a map set, never validates an RNG manifest; `B`/floor/`α` are not yet call arguments. It validates what it
can, then unconditionally blocks (reserved map-set + RNG manifest not drawn). Activation is a later reviewed change.

**Rev-20 changes folded (Pi rev-18 REVISE — manifest rework, items #2–#6):**
1. **Full-registry universes (#1/#2)** — the manifests derive from `REG.build_sd_cells` (the full schema-validated
   SD registry), not the wired projection: the RNG manifest carries all **10 experiments / 14 strata**, and the
   map-set carries **BOTH exemption variants independently** — `with_exemption` (16 entries) and `without_exemption`
   (17 entries, +`(boundary_short, bounded, S3_loggap)`). No silent adoption of the favorable 15-entry set.
2. **Role-specific RNG provenance (#3)** — per experiment × stratum × **role**: fixture seed, coupling seed
   (explicit `NONE` for uncoupled controls, `RESERVED` for coupled), per-role content hash, and a count hash binding
   **experiment+stratum+role+count** (distinguishable across entries). Per stratum: coupling strength `0.5` for
   repeatability. Per experiment: constructor-route identity (source-profile / structural-multiscale /
   bounded-control), profile-config identity, canonical arm order. The descriptive `rng_identity` is retained as a
   LAW/semantic identity only — NOT presented as executable seed provenance.
3. **Strict schemas (#4)** — `_strict` enforces EXACT required-field sets + types at every level (RNG top/experiment/
   stratum/role; map-set top/variant/entry; identity bundle), refusing unknown/missing/mistyped/wrong-sentinel-state
   fields (Pi reproduced unknown-field acceptance — now closed). FROZEN is a SEPARATE schema version, not a status
   toggle (versions bumped to `-draft-2`).
4. **Identities + reproducible hash (#5)** — both manifests bind the **full-registry variant identities**
   (`9a0ae6a8`/`e978ecd0`), the **manifest source identity**, and a **schema-definition identity**. A DETERMINISTIC
   `deterministic_schema_identity` (`56147c0f…`) is reported (field defs + versions + registry variants + the four
   SOURCE identity layers + manifest source — **excluding** the env-dependent dependency layer), so it reproduces
   across environments — fixing the rev-18 non-reproducible printed hash (which had folded the dependency layer).
5. **Map-set canonical semantics (#6)** — explicit `sort_key (profile,regime,check)`, set-hash payload rule,
   output-path grammar (relative POSIX, traversal refused), per-entry `expected_status=OK` + builder/apply source
   identities, and an explicit `refused_artifact_rule` (a `REFUSED_reference_coarsening` map neither satisfies an
   entry nor grants a provisional exemption). Still draft-only — `map_set_identity`/`rng_manifest_identity` stay the
   reserved sentinels; `gate_group_registered` stays BLOCKED.

**Rev-19 changes folded (Pi rev-18 REVISE — boundary-short / bounded-support wiring FIRST):**
1. **`G_bounded_support` wired** into `CANONICAL_GROUPS` (the one `boundary_short` experiment — condition=boundary,
   3 strata `(2667,2667,2666)`, all 12 checks already wired; map-carrying = `S7_abs` under the with-exemption
   variant). The canonical-group builder now stores each experiment's **`support_regime`** (`full` | `bounded`).
2. **Map-context regime is no longer hardcoded** — `gate_group_dev` binds the map artifact's regime to the
   experiment's `support_regime` (a bounded cell requires a `regime="bounded"` map; a `full` map is refused). Six
   engine-wired groups now; the drift assertion updates.
3. **Consequence for the manifest universe** — because the manifest derives from `CANONICAL_GROUPS`, wiring
   boundary_short already lifts the RNG universe to the full 10 experiments / 14 strata and adds the
   `(boundary_short, bounded, S7_abs)` map. The remaining Pi rev-18 items — carrying BOTH exemption variants,
   role-specific RNG provenance, strict unknown-field-refusing schemas, full-registry + manifest-source identities,
   and a deterministic schema identity separated from the env-dependent instance hash (the reported-hash fix) — are
   the NEXT increment (the manifest rework). Benchmark `09f413c9…` → `b1f97d1a…`.

**Rev-18 changes (Pi rev-13 authorized — DRAFT-ONLY reserved-manifest schemas):**
1. **New module `scripts/oracle_realism_v3_manifest.py`** defines the STRUCTURE + fail-closed VALIDATORS for the two
   reserved manifests, each binding the FIVE identity layers, and STRICTLY draft-only (nothing drawn/populated/frozen):
   - **RNG manifest** — enumerated from the canonical registry (9 experiments / 11 strata): real per-stratum
     registered quotas, canonical arm order, executable `rng_identity`, count hash; RESERVED (a later authorized
     draw): per-role seeds, drawn-sequence content hash, generator/coupling code identities. Composed identity stays
     `RESERVED_RNG_MANIFEST_NOT_BOUND`.
   - **Map-set manifest** — the 15 required `(profile, regime, check)` triples (5 map-carrying checks × 3 sources):
     registered N/floor + map-builder source identity; RESERVED: reserved-namespace seed, namespace, output path,
     per-map `map_identity`. Composed identity stays `RESERVED_MAP_SET_NOT_DRAWN`.
2. **Validators refuse every corruption** (missing/extra/duplicate, tampered identity layer, wrong quota/rng-identity/
   N/builder-identity, a draft pretending to be populated, a bound-when-reserved identity, a non-DRAFT status); the
   self-test validates a well-formed draft and refuses the battery. Draft schema hashes: RNG `b35f9173…`, map-set
   `4ba9cf1f…`.
3. **Nothing is unblocked:** the reserved identities are unchanged, so `gate_group_registered` stays unconditionally
   BLOCKED. FREEZING/populating the manifests remains a separately-authorized later review. Remaining authorized
   dev-only work: boundary-short canonicalization; the full per-group `B=20000` benchmark/cap/checkpoint plan.

**Rev-17 changes (Pi rev-12/rev-13 authorized wiring — LAST group: phase_seam; ALL groups now wired):**
1. **Five estimators wired** reproducing v2 to `< 1e-9` (v2-consistency self-test): **`S8_density`**/**`S8_class`** —
   within-sequence CENTERED quartile-phase nonstationarity (each sequence's per-quartile cluster-start density /
   class vector MINUS its own whole-sequence value; then max-over-4-quartiles two-sample mean diff, with per-quartile
   sequence + item floors both arms); **`S9_zero`**/**`S9_class`** — per-sequence seam-minus-nonseam `Δt==0` fraction
   / same-class fraction (block seam = adjacency `i` where `(i+1)%8==0`), two-sample mean diff with shared
   sequence/adjacency/positive-gap floors; **`S9_gap`** — seam-vs-nonseam positive-gap KS
   (`max` of within-arm + cross-arm `ks_2samp`). New precompute-schema branches (per-quartile lists, centered
   scalars/vectors, owner-indexed gaps with a bool `contrib` channel).
2. **`G_full_phase_seam` group wired** (45 cells) — added to `CANONICAL_GROUPS`; drift assertion + registry hash
   update. **ALL five full-support groups are now engine-wired (20 estimators total).** The five estimators are
   covered by the v2-consistency, real-precompute-validates, and degenerate-pool self-tests.
3. Source identities re-mint (benchmark `c4970291…` → `09f413c9…`). Remaining authorized dev-only work: draft-only
   manifest schemas (5 identity layers), boundary-short canonicalization, and the full per-group `B=20000` benchmark
   (now feasible with all groups wired).

**Rev-16 changes (Pi rev-12/rev-13 authorized wiring — SECOND group: length_density):**
1. **Four estimators wired** as permutation-friendly precompute/recompute splits reproducing v2 to `< 1e-9`
   (v2-consistency self-test): **`S1_density`** (MAP-carrying, mirrors S5 with per-LENGTH_BINS mean cluster density
   `K/L`; shares the S5 schema branch + exactly-one-length-bin); **`S1_tau`** (SOURCE-LEVEL `|kendalltau_b(L,K)_cand −
   _ref|` with a source-level sequence floor; schema requires `1 ≤ K ≤ L`, integer); **`count_ks`**/**`length_ks`**
   (two-sample KS on per-sequence cluster count `K` / length `L` via a `(n,|support|)` 0/1 ECDF-indicator matrix whose
   max-mean-diff equals scipy `ks_2samp`).
2. **`G_full_length_density` group wired** (36 cells) — added to `CANONICAL_GROUPS`; drift assertion + registry hash
   update. The four estimators are covered by the v2-consistency, RC1 floor-protocol, real-precompute-validates, and
   degenerate-pool self-tests.
3. Incremental: **`phase_seam` (S8/S9 family) is the last remaining group** (Still-open). The registry hash grew and
   the source identities re-mint (benchmark `0e3b5b5c…` → `c4970291…`).

**Rev-15 changes folded (Pi rev-13 GO-WITH-CHANGES — 4 schema/identity refinements):**
1. **Real dtype only (#1)** — `_pc_arr` requires real integer/floating (not `np.number`, which admits complex and
   would lossily cast to float); bool excluded. Complex `occupancy`/`S3_tau` now refuse.
2. **Cross-channel consistency (#2)** — the schema now refuses structurally impossible states each field would pass
   alone: `S3_tau` `n0/n1/n2` integer-valued with `|C−D| ≤ n0` (retaining `n0 ≥ n1,n2`); `S3_loggap` `sm` finite IFF
   `sp>0`; `S7` `sm` finite IFF `cc>0`; `S4` present rows require BOTH pair counts strictly `>0`; `S5`/`S6` each
   sequence present in EXACTLY one LENGTH_BIN (S6 present rows nonneg summing to 1). Support-absence stays NE.
3. **Five explicit identity layers (#3)** via reviewed FULL-MODULE-FILE hashes (transitively covering
   `phase_spanning_indices`, verifier `_runs`, `coarsen_reference`, etc.): (1) estimator semantic, (2) estimator
   implementation-source {engine, v2_verifier, phase0_pilot}, (3) engine canonicalization/schema/gate {engine,
   randomization}, (4) map builder/apply source {map, v2_verifier}, (5) dependency/environment. The deterministic
   config/dev-stable identities bind the four SOURCE layers (`SOURCE_IDENTITY_BUNDLE`); dependency stays in the
   env-dependent timing artifact/dev_config. Walked back "map builder bound per-artifact via `map_identity`":
   `map_identity` binds the map OUTPUT, not the builder implementation source (bound in layer 4).
4. **Explicit profile→skeleton map (#4)** — `_canonicalize_pool` uses `_PROFILE_SKELETON` (all registered SD sources
   enumerated) and REFUSES an unknown profile instead of a `"scid" in name` substring heuristic that mapped every
   unknown string to MIMIC.

**Rev-14 changes (Pi rev-12 authorized estimator wiring — FIRST group: run_size):**
1. **`S2_ks` estimator wired.** `_s2_pre`/`_s2_re` implement the v2 run-size KS (MEAN over sequences of the
   per-sequence run-size ECDF, max-abs over the pool support) as a permutation-friendly precompute/recompute split on
   the SAME keyword-only protocol and the SAME schema pipeline (dtype/shape/[0,1]-range M, nonneg-integer `nruns`,
   sequence + cluster floors both arms). The v2-consistency self-test reproduces `s2()` to `< 1e-9`.
2. **`G_full_run_size` group wired.** Added to `CANONICAL_GROUPS` (9 cells), so the engine can now gate the run-size
   group; the canonical-groups drift assertion and the registry hash update accordingly.
3. Wiring is incremental and verified — `length_density` and `phase_seam` remain (Still-open). The
   `ESTIMATOR_IMPL_SOURCE_IDENTITY` and benchmark `config_identity` re-mint (`f3a75308…` → `b3c1b00a…`) to reflect the
   new estimator source.

**Rev-13 changes folded (Pi rev-12 REVISE — schema/identity corrections):**
1. **Derive-not-trust raw canonicalization (Pi rev-12 #1/#4).** The `SequenceRecord` contract trusts only
   `source`/`class_ids`/`timestamps`; `cluster_ids`/`L_total`/`K`/`positions` are DERIVED. The rev-12 validator
   range-checked the supplied derived fields — Pi reproduced a record with real positive gaps but `K=1`/all-zero
   `cluster_ids` that passed and would silently corrupt burst/run stats. Replaced with `_canonicalize_pool`, which
   REBUILDS each record via the repository's `derive_record(source, class_ids, timestamps)` boundary (discarding
   caller-supplied derived fields) and binds the experiment's expected skeleton source. Runs ONCE per experiment in
   `_gate_core` (not per cell). Malformed TRUSTED fields (bad source, class id ∉ [0,C), non-finite/non-monotone
   timestamps) refuse; inconsistent DERIVED fields are canonicalized, not trusted. Owned: I should have used the
   existing derive-not-trust boundary.
2. **Exact precompute schemas (Pi rev-12 #2).** Fixed the accepted-but-wrong cases Pi reproduced: `S3_tau` exactly
   `(n,4)`; explicit numeric non-bool dtypes (object arrays refuse); S4/S6 rows are absent(all-NaN)-or-complete-finite
   (partial-NaN refuses); legal value ranges (occupancy/S5/S7 ∈ [0,1], S4 contrast ∈ [-1,1], class/dt0/pair/cluster
   count channels nonneg integer-valued, dt0 `L-K ≤ L-1`, tau `n0 ≥ n1,n2`, S6 present rows sum to 1). Crucially,
   **support-absence is a valid `NOT_EVALUABLE` state, not a schema refusal**: `positive_gap_ks` now ACCEPTS the
   empty `nu=0` case (owner/inv empty) and, for `nu>0`, requires inverse coverage of `0..nu-1`. Valid degenerate
   pools (all L=1, one cluster) are tested to VALIDATE.
3. **Complete identity separation (Pi rev-12 #3).** The rev-11/12 "code identity" is renamed
   `ESTIMATOR_IMPL_SOURCE_IDENTITY` and now covers the dispatch table + wrapper lambdas (each recompute callable's
   source), every precompute, the local map reducers, the imported executable dependencies (`_seq_components`,
   `_positive_gaps_and_prev_size`, `_s4_contrast`, `_bin_index`), and the bin definitions — so a check→function swap
   or wrapper change is caught. A separate `ESTIMATOR_DEPENDENCY_IDENTITY` (python/numpy versions + constants) holds
   the environment-dependent part. The DETERMINISTIC config/stable identities now bind SOURCE identities only; the
   dependency identity lives in the env-dependent timing artifact / dev_config, so a "deterministic" config is no
   longer version-bearing. `4c86fbe2` is superseded (it was a partial implementation identity).
4. **Once-per-experiment validation (Pi rev-12 #4).** Canonicalization runs one pass per experiment, then the
   canonical pool is reused for every cell (no 4–6× repeat).

**Rev-12 changes folded (Pi rev-10 authorized scope #3 — estimator/precompute/raw-record schemas):**
1. **Per-estimator precompute schema.** `_validate_precompute(pre, check, n)` replaces the Inf-only check with a
   fail-closed per-estimator schema: exact keys (dicts), shapes/dtypes, pooled length `n`, class-vector width `C`,
   per-bin list lengths (`LENGTH_BINS`/`CLUSTER_BINS`), owner/inverse index ranges (`[0,n)` / `[0,nu)`), and legal
   NaN locations — NaN is permitted ONLY as the per-sequence absent-in-bin sentinel (map/S4); count channels
   (`sp`, `cc`, `class_tv`, `dt0`, S4 pair-columns) must be finite + nonnegative; Inf is never legal.
2. **Raw-record schema.** `_validate_raw_records(pool)` runs in `_gate_core` BEFORE any precompute: each record must
   satisfy the SequenceRecord invariants the estimators rely on — positive non-bool `L_total`, `K∈[1,L_total]`,
   equal-length 1-D arrays, `class_ids` non-bool ints in `[0,C)`, finite nondecreasing float `timestamps`,
   `cluster_ids` non-bool ints in `[0,K)`. A hand-built/corrupt record now refuses at the boundary instead of
   crashing (or silently mis-binning) inside a precompute. Confirmed no-op on the real fixture / structural-zero /
   coupled pools.
3. **Adversarial self-tests.** Every real precompute validates clean; the malformed classes Pi enumerated refuse —
   missing/extra keys, wrong dims/pooled length, wrong class width, owner/inverse length mismatch, out-of-range
   owner/inverse indices, illegal NaN in a finite-only field, negative counts, Inf; and corrupt raw records
   (non-finite timestamp, out-of-range class/cluster id, length mismatch) refuse.
4. **Benchmark map provenance seed (Pi rev-11 Correction 1).** The wired benchmark's timing map is built from `Bs`
   (`draw("mimic_scale_control", 2)`); its artifact now records the EXACT generating seed
   `bseed("mimic_scale_control", 2) = 236460273103544` (not the convenient `seed=1`), is labelled a TIMING-ONLY map
   (namespace `v3-benchmark-timing-map`, distinct from any reserved map-design artifact), and an assertion binds the
   artifact seed to the fixture invocation. Owned: satisfying a mandatory provenance field with a convenient value is
   the exact anti-pattern flagged before.
5. **Semantic vs executable code identity (Pi rev-11 Correction 2).** `ESTIMATOR_PROTOCOL_IDENTITY` (which hashes only
   the protocol + estimand identity STRINGS) is renamed `ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY`, and a distinct
   `ESTIMATOR_CODE_IDENTITY` (hash of the actual estimator implementation source + numpy version) is added. BOTH are
   bound into the benchmark `config_identity` and the dev stable identity. The rev-11 claim that a "positional/altered
   estimator shows as identity drift" is withdrawn for the code case: the semantic hash does NOT catch an
   implementation change with unchanged strings — the code identity does.

**Rev-11 changes folded (Pi rev-10 PASS: required integration repair + RC5 follow-through):**
1. **Required benchmark repair.** The rev-10 keyword-only estimator protocol broke `oracle_realism_v3_benchmark.py`,
   whose wired-engine measurement still called `est["recompute"](pre, m, groups)` positionally (→ `TypeError`) AND
   called `build_frozen_map(...)` without the now-mandatory `seed`. Both are fixed (`recompute(pre, m, groups=groups,
   floor=500)` — registered benchmark floor semantics; `build_frozen_map(..., seed=1, N=N)`); the benchmark reruns
   clean. My rev-10 "all scripts rerun clean" claim was false — I had not executed every downstream caller after the
   protocol change; owned and corrected.
2. **Estimator/protocol identity + benchmark re-mint.** A new estimator identity is bound into the deterministic
   benchmark `config_identity`, which is re-minted (`ef7a9280…` → `34d1a50a…`) rather than cited stale. *(Corrected by
   rev-12 Correction 2: that identity is SEMANTIC — protocol + estimand strings — and is renamed
   `ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY`; the rev-11 claim that "a positional/altered estimator shows as identity
   drift" OVERCLAIMED for the code case. A distinct executable `ESTIMATOR_CODE_IDENTITY` now catches implementation
   changes; both are bound.)*
3. **RC5 follow-through.** The dev stable identity is extended with `ESTIMATOR_PROTOCOL_IDENTITY`; the demo asserts the
   base and perturbed stable identities are EQUAL in every paired trial (same counts/maps/registry/protocol/namespace;
   only sequence content differs). Input content hashes remain separate evidence (deferred to the manifest work).

**Rev-10 changes folded (Pi rev-9 GO-WITH-CHANGES, RC1–RC5):**
1. **Uniform keyword-only estimator protocol (RC1)** — every registry estimator is now called
   `recompute(pre, mask, *, groups, floor)`; `S5_abs`/`S6_tv`/`S7_abs` bind keyword-only wrappers. The rev-9 defect
   bound `_map_re_scalar` positionally, so the dev floor was assigned to `extra_key` and the real floor stayed 500 —
   which is what produced the class-mark `NOT_EVALUABLE`, **not** a legitimately stricter map (that rev-9 claim is
   withdrawn). A controlled self-test exercises every floor-gated estimator at floor 60 (evaluates) and floor 500
   (refuses) and every floor-insensitive estimator identically at both.
2. **Registered mode framed as a blocked stub (RC2)** — the docstring/contract no longer say it "enforces exact
   B/floor/α" as an executable evaluator; it declares invariants and tests structured assembly/refusals, execution
   unimplemented and unconditionally blocked.
3. **Map builder never emits an unvalidated artifact (RC3)** — `seed` and `N` are MANDATORY builder arguments;
   profile/regime/namespace/seed/N/floor are validated BEFORE integer coercion (so `True` is not coerced to `1`);
   the artifact is validated before return. `5106ad09…` is a **development self-test** map-set hash, not the
   reserved registered map-set identity.
4. **Strict paired direction (RC4)** — the mechanical paired check records success only for strict
   `p_g(perturbed) < p_g(base)` (a tie is no direction evidence).
5. **Dev-config semantics hardened (RC5)** — `gate_group_dev` requires a positive non-bool floor and a non-bool
   integer seed, rejects extra map artifacts, and returns a FULL dev-config reproducibility record (per-stratum
   counts, map identities + set identity, registry identity, namespace, seed law) with a stable seed-invariant
   identity — not merely `(floor,B,group)`. The low-level kernel is renamed `_gate_group` (private/test-only).

**Rev-9 changes folded (Pi rev-8 #1–#9):**
1. **Structured strata (#1)** — the engine accepts `experiment → stratum_id → {candidate, reference}`, compares
   counts to the canonical registry, and concatenates in canonical order. The registered structural-zero
   `(2667,2667,2666)` now assembles without the flat-pool divisibility refusal (a synthetic-registered assembly is
   in the engine self-test).
2. **Explicit dev vs registered boundaries (#2)** — `gate_group_dev` (explicit hash-bound floor/B, no module-global
   mutation) and `gate_group_registered`. Floor is a parameter, not a global. *(Superseded by rev-10 RC2: the
   original wording here — that registered mode "enforces exact N/arm, B=20000, floor 500, α=0.04/6 … refusing ANY
   deviation" as an executable evaluator — OVERCLAIMED. Registered mode is a blocked stub that declares invariants
   and tests structured assembly/identity refusals; the statistic path is unimplemented and unconditionally blocked;
   B/floor/α are not call arguments. See the rev-10 section above and the Delivered entry below.)*
3. **Map context binding (#3)** — dev mode binds each map artifact's profile/regime/dev-floor to the canonical cell
   and enforces ONE shared identity per `(profile, regime, check)`; the engine no longer self-derives expected map
   hashes from the supplied artifacts.
4. **Map validator (#4)** — mandatory positive non-bool N/seed/floor; nonempty profile/regime/namespace; groups
   must be ORDERED CONTIGUOUS nonempty disjoint covering `0..n-1` (a v2 merge yields ordered contiguous runs, not
   any partition); the builder verifies `len(reference_sample)==N`. Reproduced acceptances (None seed/N, blank
   profile, non-contiguous groups, empty group) now refuse.
5. **Paired direction (#7)** — the demo evaluates base and its one-experiment perturbation under the SAME
   permutation seed and compares `p_g(perturbed)` to `p_g(base)`.
6. **Canonical controls (#3/#8)** — structural-zero data AND map-design sample use the canonical multiscale
   constructor.
7. **Contract (#9)** — the superseded DONE list is removed; the readiness/provenance overclaims are withdrawn (this
   round separates dev-dispatcher wiring from a registered trusted boundary).
8. **Still open (Pi-authorized, NOT done):** wire length_density/run_size/phase_seam; the generate-only RNG +
   reserved map-set manifests (a real registered run stays blocked); the full `B=20000` per-group benchmark;
   boundary-short canonical map; estimator/precompute/raw-record schemas.

**Rev-8 changes folded (Pi rev-7 #1–#9):**
1. **Engine dispatcher boundary (#4)** — `scripts/oracle_realism_v3_engine.py` loads the canonical group registry
   INTERNALLY, computes precompute ITSELF from raw pools, and treats any non-finite discrepancy as NE (never
   zero-fill). Reproduced fail-open exploits (all-NaN precompute → PASS; wrong check/Δ → PASS) now NE/refuse
   (adversarial self-tests). No caller-supplied check/Δ/registered/precompute is trusted. *(Superseded framing:
   rev-8 called this a "trusted"/production boundary via `gate_group_trusted`; rev-9/rev-10 correct it to a
   DEVELOPMENT dispatcher — `gate_group_dev` — distinct from the registered blocked stub `gate_group_registered`.)*
2. **Sparse perturbation (#1)** — the demo perturbs EXACTLY one experiment and asserts (executable, hashed) that
   every other experiment pool is content-identical to its null construction; argmin reported + attributes to the
   perturbed experiment's registered primary cell. The dense-9-experiment claim is withdrawn.
3. **Canonical structural-zero (#2/#3)** — demo + preflight use the registered multiscale control constructor
   (means 18/60/250), not three same-profile draws.
4. **Map provenance + validator (#6)** — artifacts record the EXACT map-design seed/namespace actually used; the
   validator enforces registered estimator/bin/denominator identities, floor-consistent denom policy, positive
   non-bool N/seed/floor, status↔groups consistency, and a complete disjoint bin partition.
5. **Registered B (#3)** — the dev demo B is LABELLED a mechanical development test, distinct from the registered
   `B_main=20000` evaluator.
6. **Contract (#9)** — one normative section + this changelog; the stale §7 pilot detections and the §3 "exact
   resolution rule" wording are corrected.
7. **Still open (Pi-authorized, not yet done):** wire the length_density/run_size/phase_seam estimators (#2 wiring);
   generate—not execute—the reserved map-set manifest (#5); benchmark all wired groups at `B=20000` per exact
   profile/stratum with margin/checkpoint/persistence (#6/#8); executable RNG-provenance manifest (#5).

**Rev-7 changes folded (Pi rev-6 #1–#7):**
1. **Corrected exact-group demo (#1)** — `scripts/oracle_realism_v3_group_power.py` now runs through the engine
   with INDEPENDENT per-(profile,check) map-design samples (no leakage), `exp_id` in every fixture seed
   (independent experiments), the structural-zero 3 within-stratum quotas, honest sparse one-experiment /
   two-primary-cell attribution, all THREE active components, and tied-KS on the 8-dp-rounded support (§ Delivered).
2. **Registered-N boundary preflight (#2)** — `scripts/oracle_realism_v3_regn_preflight.py`: at N=8000 the bounded
   S3_loggap map ISSUES (evaluable) — the dev-scale "structurally un-calibratable" claim is WITHDRAWN; both S3
   exemptions now rest on the DETECTION criterion (bounded detect `0.0 < 0.5`), reported with/without each (§7).
3. **Issuance-complete map (#3)** — `oracle_realism_v3_map.py` binds a full trust root + validator that refuses
   cross-identity application; adequate-support tests prove exact reproduction (not `None==None`) (§9).
4. **Heterogeneous registry-owned dispatcher (#4)** — `oracle_realism_v3_engine.py`: cells bind to registry-owned
   estimators by check (callers pass DATA, never a statfn); mandatory per-map identity + floor-policy, exact cell
   ids/order, executable RNG identity, one trusted assignment path, real permutation-NE test; per-perm recompute
   reproduces the exact v2 estimand to <1e-9 (§3, §8).
5. **Registry identities re-minted (#5)** — S3_tau pilot suffix, S6 estimator text, and audit prose corrected;
   hashes re-minted (§ Delivered).
6. **Single canonical PASS rule + design-rule framing (#7)** — `p_g > α_group` everywhere; `B ≥ K/α_group` is a
   conservative sparse-effect DESIGN RULE, not a theorem — actual group power remains a required falsifier (§3).
7. Wired-engine benchmark with measured generation/serialization + conservative cap margin (#6) — §9.

**Changelog — earlier rulings folded (condensed; the normative design is §0–§9):**
- **Rev-6 (Pi rev-5):** corrected reference-owned frozen-map builder (original bins + `coarsen_reference` grouping,
  S6→LENGTH_BINS; full-support S3_loggap null-exceedance 0.526→0.0); symmetric p-value prose; NOT_EVALUABLE policy;
  exact-group demo; coarse audit REMOVED; per-profile benchmark; coupled-role RNG law bound per experiment.
- **Rev-5 (Pi rev-4):** benchmark executes + route-weighted (KS ~92% of a permutation; whole-job ≫ "~6h"); exact
  schema-validated registry with Δ bound to live constants (`S3_loggap=log(1.10)`) + Δ-hash; both boundary-exemption
  variants (M0 192/194); product/stratified finite-B randomization + 7 refusals vs the exhaustive universe.
- **Rev-2..rev-4 (Pi):** SD fresh-draw conditional randomization + MM direct-effect split; honest multiplicity
  (min-p over `K_g`); phase-spanning pooled-tau estimator; α split; exact registry/Δ before implementation.

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
4. accept group `g` iff `p_g > α_group` (strict; the single canonical PASS rule — §3, Pi rev-6 #7). Battery
   accepts iff every SD group accepts AND all MM requirements hold.

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

## 3. Randomization p-value algorithm (Pi §3 + rev-3 §1 direction fix)

Directions are OPPOSITE at the cell and group levels (Pi rev-3 §1; formula corrected per Pi rev-5 #2 to match the
code exactly):
- **Per-cell, upper tail** (larger discrepancy ⇒ more extreme): with `e_c = (d_c − Δ_c)_+`, for each assignment
  `j ∈ {0..B}` (observed = index 0), the **symmetric** marginal rank counts `j` exactly once:
  **`p_c^(j) = #{k ∈ 0..B : e_c^(k) ≥ e_c^(j)} / (B + 1)`**. (The earlier prose `1 + #{b≠j} + [obs≥j]` double-counts
  for some `j`; for the observed assignment the symmetric form reduces to the usual plus-one expression.)
- **Group min-p, lower tail** (smaller min-p ⇒ more extreme): `S_g = min_c p_c`, and
  **`p_g = (1 + #{b : S_g^(b) ≤ S_g^obs}) / (B + 1)`** (`≤`, NOT `≥`).
- **Tie handling:** the inclusive `≥` (cell) and `≤` (group) as written; all `B+1` assignments symmetric.
- **PASS convention (frozen):** group `g` passes iff `p_g > α_group` (strict `>`).
- **Validation (required before any MC implementation):** exhaustive tiny-enumeration tests over ALL balanced
  candidate/reference label assignments (small `m`, small cell sets) confirm the direction, the nested-rank
  construction, and the tie handling exactly.
- **Synchronized permutations:** one permutation index drives all cells in a group in a replicate (preserves
  cross-cell dependence). For groups spanning independent experiments, independent per-experiment permutations are
  drawn under one synchronized MC index.
- **`B` sizing (conservative sparse-effect DESIGN RULE, not an exact theorem — Pi rev-6 #7):** the min-p over `K_g` cells shares its floor `1/(B+1)` across ~`K_g`
  permutations (each cell's own extreme), so a single-cell effect resolves below `α_group` only when
  **`B ≥ K_max/α_group`**. With the deepest in-scope group `K_max = 54` and `α_group = 0.04/6`, this is
  **`B ≥ 8100`**. Main **`B = 20,000`** meets it. (The rev-5 coarse `B = 2,000` audit could NOT resolve a
  single-cell effect in the deepest group and is REMOVED — §4.) `B` is frozen by this rule, not a default.
  Fisher/Cauchy combiners are **OUT** for v3.
- **Randomization kernel (implemented + exhaustively validated); registered integrated path BLOCKED (Pi rev-9 RC2).**
  The product/stratified finite-`B` kernel (within-stratum label permutation preserving each stratum's quota;
  independent per-experiment permutation under one synchronized MC index; nested cell upper-tail then group
  lower-tail min-p; deadbands and ties) reproduces the exhaustive group `p_g` exactly, stays conservative under a
  finite-`B` product null, and replays deterministically. All seven malformed-input classes (unequal
  candidate/reference size, wrong stratum quota, duplicate/missing pooled index, non-bijection, role-dependent RNG
  law, `B`/RNG mismatch, truncated cell vector) fail closed BEFORE any statistic is computed. This kernel is
  exercised through the DEVELOPMENT dispatcher (`gate_group_dev`) only; the **registered** integrated path
  (`gate_group_registered` at `B=20000`/floor 500 over a drawn map set + bound RNG manifest) is **not implemented**
  and is unconditionally blocked until those reserved manifests exist and a later activation review passes.

## 4. Error budget — main SD level only; the coarse audit is REMOVED (Pi rev-5 #5)

```
alpha_total_family = 0.05
alpha_main_SD      = 0.04        # main fresh SD draw; alpha_group = 0.04 / G ; unused margin 0.01
```

The rev-5 coarse audit (15 replicates, `B_audit = 2,000`, park ≥4) was **level-valid but functionally blind**:
for the deepest group (`K_max = 54`, `α_group ≈ 0.00667`) a sparse single-cell effect cannot resolve below the
group threshold at `B = 2,000` (`B ≥ K_max/α_group = 8100` is required — §3), so a `B = 2,000` screen cannot detect
the pathology it was meant to guard, and spending ~7.2 h on it is unjustified. Per Pi's preferred option the
**scientific audit gate is REMOVED**. The main conditional-randomization battery already has exact SD level
(`α_main_SD = 0.04 ≤ 0.05`, leaving unused margin); implementation soundness is covered by the **mechanical /
exhaustive tests** (randomization exhaustive-vs-MC equivalence, refusal tests, the fail-closed gate self-tests),
not a low-power re-run. No audit `B`, no audit namespace, no park-on-audit rule. (If a resolving audit were ever
wanted it would need `B ≥ 8100` per group and its own separately-gated jobs with an explicit power/cost case —
not adopted here.)

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
joint), pair-count-weighted pooling across within-sequence pairs. Frozen + evidenced in the committed pilot (hash
`db885b97…`): formula err vs scipy `0.0`; phase coverage mean-pos 0.086→0.500; concentration MIMIC 0.034→0.989 /
SCID 0.083→1.00; source-wise power full=1.00, bounded (non-Δ proxy) 0.125; S8 no cross-load. Other coarse-null
rank statistics adopt the same pool-within/synchronized-permute pattern (enumerated in the registry).

## 7. Effect deadband, scope, un-calibratable — exact semantics (Pi §7)

- **`e_c = (d_c − Δ_c)_+` is an OBSERVED-discrepancy deadband under the sharp-equality null.** Claimed narrowly:
  exact SD type-I control **when the two generators are identical**; practical effect screening via the observed
  deadband. It is NOT an exact equivalence test for every population discrepancy within `Δ_c` (exchangeability
  holds only at equality).
- **Un-calibratable** is defined by power against a meaningful alternative beyond `Δ_c`, NOT by null mass above
  `Δ_c` (conditional randomization can still control sharp-null rejection). The retain/exempt criterion is
  **Δ-aligned to the MM effect rule (Pi rev-3 §3):** `power = P_dev[ d_c > Δ_c | @0.5 alternative ]`, **retain iff
  power ≥ 0.5**. This 0.5 threshold is a v3 design choice informed by development evidence (not predeclared before
  the pilot). A property-specific support cell failing it is EXEMPT; a **core/full-support cell failing it PARKS**.
  The criterion therefore **cannot finalize an exemption until the numeric `Δ` table exists** (the earlier
  `P[d > matched-null p96]` proxy is not equivalent).
- **Per-statistic exemption status (decided at REGISTERED N=8000, Pi rev-6 #2 + rev-7 #2/#7):** the boundary
  exemption is decided by the registered-N preflight (`scripts/oracle_realism_v3_regn_preflight.py`), NOT the
  dev-scale (N=3000) pilot. At N=8000 the **bounded `S3_loggap` map ISSUES (evaluable)** — the earlier "structurally
  un-calibratable" rationale is **WITHDRAWN**. Both S3 subchecks are instead exempt on the **DETECTION criterion**:
  registered-N bounded detection `P[d>Δ @0.5] = 0.0 < 0.5` for both, while full/SCID/MIMIC/structural-zero detect at
  `1.0` (checks work where supported; full-support burst-timing RETAINED, non-exemptible). Both exemptions remain
  **PROVISIONAL** and are reported **with AND without each independently**; both M0 variants (192 / 194) are carried
  so the exemption cannot manufacture favourable group size/power. **Finalisation requires a separately-authorized,
  freshly-namespaced calibration/power battery using the frozen map (Pi rev-7 #7)** — the preflight is development
  evidence only and does not meet a final power rule. The dev pilot is retained as supporting dev evidence and
  explicitly DEFERS the decision to the preflight.
- The full `Δ` table + required-property matrix is the "Δ / required-property" deliverable (below); values are
  sourced from prior/first-principles before calibration and hashed, and preserve existing v2 practical-effect
  semantics unless a separately-justified change is identified.

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

## 9. Whole-battery permutation feasibility (Pi §9 + rev-3 §2/§6 corrections)

**Independent reference-owned frozen coarsening (Pi rev-3 §2 — replaces the rejected pooled-frozen idea).** The
coarsening/binning map per (profile/regime/check) is derived ONCE from a **separately-namespaced synthetic
reference-DESIGN draw**, frozen + hashed **before** any audit/evaluation, and applied IDENTICALLY to: SD
candidate/reference, every permutation assignment, the MM arms, and later candidate evaluation. Candidate data
never modify the map; a candidate/reference floor failure under the frozen map stays fail-closed `NOT_EVALUABLE`.
Because the map is fixed independently of the tested labels, (a) conditional randomization remains valid, (b) the
v2 anti-masking principle is preserved (candidate cannot influence the bins to hide tail collapse — the defect of a
pooled map), and (c) O(N)-per-permutation recomputation remains possible. The map-building draw is a
verifier-DEFINITION artifact (bind seed/profile/code/hash); it is NOT a transported null threshold and must not be
used to alter `Δ` or scope. **Generating this reference-design draw is NOT authorized by this review** (Pi rev-3 §7).

**Frozen reference-owned map builder — CORRECTED spec + built code (Pi rev-5 #1; the reserved DRAW stays blocked).**
The rev-4 pilot map was wrong (data-driven unique-value edges + pooled adjacency means, and S6 mis-labelled). The
corrected builder (`scripts/oracle_realism_v3_map.py`) preserves the REGISTERED estimand:
- **Original bins retained.** Map-carrying checks and their original bins: `LENGTH_BINS` for `{S1_density, S5_abs,
  S6_tv}` (Pi #1: S6 is length-binned, NOT class-coarsened), `CLUSTER_BINS` for `{S3_loggap, S7_abs}`.
- **Reference-owned GROUPING of original bin indices** via the frozen v2 merge (`coarsen_reference` on the
  reference per-bin sequence counts) — the candidate never influences the bins (anti-masking). It groups original
  bins; it does NOT invent new edges.
- **Registered per-sequence summary + EQUAL-WEIGHT pooling** (`_grouped`) preserved, with every denominator floor
  (sequence floor for all; the extra adjacent-pair floor for `S3_loggap`). Floor breach on either arm ⇒
  `NOT_EVALUABLE` (never zero-filled); a regime whose reference cannot reach the floor (bounded support) makes the
  check un-calibratable → the §7 exemption path.
- **Verified:** the self-test proves `apply_frozen_map` reproduces the v2 check `.value` EXACTLY when the frozen
  grouping equals the per-draw grouping, is candidate-independent, and enforces floors. Re-running the Δ-aligned
  boundary evidence with the corrected estimand drops the full-support `S3_loggap` null-exceedance **0.526 → 0.0**
  (confirming Pi #1: the earlier 0.526 was the wrong estimator, not a harmless artifact).
- **Seed namespace / artifact / hash.** The one-time draw uses a disjoint `map-design` namespace (registered
  `N = 8000`), binds `{check, bins_id, n_original_bins, groups, floor}` + a per-map + map-set hash, and is a
  verifier-DEFINITION artifact that must not alter `Δ`/scope. **The draw itself is NOT authorized** (dev fixtures
  only for now).

**Per-profile route-weighted benchmark (executable; Pi rev-5 #6) + NO audit (Pi rev-5 #5).**
`scripts/oracle_realism_v3_benchmark.py` maps every statistic to one of five recompute route classes and forecasts
`Σ_experiment Σ_cell cost(route, THAT experiment's profile/regime volume) · B_main + serialization + MM_proxy` —
route costs are per-item (profile-independent) but volumes are per-profile, so SCID-scale experiments are heavier
than MIMIC (the rev-5 "one MIMIC cost × M0" understated SCID). The **distribution-comparison (KS) routes dominate**
(≈ 92 % of a permutation's cost; `S9_gap` ≈ 56 ms, `S2_ks`/`positive_gap_ks` ≈ 14 ms). Consequences:
- **No audit term** (§4): the forecast is SD-main + MM only.
- **"SD-main fits one 8 h job" is NOT assumed** — the benchmark emits a **per-group job plan** (`per_group_job_hours`)
  and a `sd_main_fits_one_8h_job` flag from the per-profile sum; if it does not fit, run **separately-gated
  per-group SD jobs** (stop-on-failure) + a separate MM job.
- **Two identities:** a DETERMINISTIC `config_identity` (formula + routes + per-experiment routing + per-profile
  volumes + `B` — all seed-deterministic) and a separate TIMING artifact (route costs / hours, environment-
  dependent, NOT reproducible — Pi's benchmark-hash observation). **Serialization is measured**, not assumed.

**Preflight persistence:** per experiment/group report `sequences, events, clusters, positive-gaps, eligible
pairs, strata, cells, B, statistic recomputation route, wall time, peak RAM, checkpoint size`; aggregate-only
(per-cell/group permutation summaries + hashes; no sequence arrays).

## Revised phases (Pi's, adapted)
0. Estimator pilot — DONE. 1. Design freeze — this contract + registry artifact + Δ table + Pi ratification of the
reference-owned original-bin coarsening + G/group/Δ specifics. 2. Calibration-build preflight (benchmark/caps/
manifest; separately authorized). 3. Group-critical-value + power demonstration under the joint permutation null +
freeze the algorithm identity. 4. Fresh evaluation manifest review, empty policy, reproducing v3 hashes. 5.
Sequential evaluation, separately gated (power first; ident only on clean power PASS).

## Ratified by Pi rev-3 (structural elements accepted)
SD/MM split; fresh-draw conditional randomization (no transported critical values); MM direct effects
(≥20/25 primary, ≥24/25 specificity); `α_main_SD=0.04` (the rev-3 15-rep coarse audit is now REMOVED per Pi rev-5
#5 — see §4); conservative finite-`B` with observed assignment included; no Fisher/Cauchy; phase-spanning
pooled-tau CONCEPT + exact ties +
label-permutation-only + explicit equal-sequence replacement; stratum-preserving + independent product permutations;
"exact registry/Δ/property artifacts must precede implementation."

## Delivered (rev-20, Pi rev-9/rev-10/rev-12/rev-13/rev-18 authorized dev-only scope — all tested)
- **Registry (identities re-minted, Pi #5)** — `scripts/oracle_realism_v3_registry.py`: full per-cell identity +
  strict refusal; `Δ` bound to LIVE thresholds (Δ-hash **`ec6f4dff…`**). S3_tau identity DECOUPLED from the
  fragile whole-pilot aggregate hash to a stable frozen-estimator descriptor; S6 estimator text → LENGTH_BINS;
  audit prose removed; `B≥K/α` reframed as a conservative design rule. Both variants **M0 = 192 / 194**
  (hashes **`9a0ae6a8…` / `e978ecd0…`**); `G=6`, `K_max=54`.
- **Issuance-complete map builder + HARDENED validator (Pi #3/#6; RC3)** — `scripts/oracle_realism_v3_map.py`
  (**development self-test** map-set hash **`5106ad09…`** — NOT the reserved registered map-set identity, which is
  not drawn): original bins + reference-owned `coarsen_reference` grouping + per-sequence means/equal-weight/floors
  (S6 on LENGTH_BINS). Full issuance trust root; the validator enforces registered estimator/bin/n_original_bins
  identities, a **floor-consistent denominator policy** (a dev-floor map cannot record the registered floor),
  positive non-bool N/seed/floor, **status↔groups consistency**, and a **complete disjoint bin partition**; it
  REFUSES cross-identity application. **RC3:** `seed` and `N` are MANDATORY builder arguments, all provenance fields
  are validated BEFORE integer coercion (so `True` is not coerced to `1`), and the builder validates the artifact
  before returning it — it can never emit an unvalidated/null-provenance map. Callers record the EXACT map-design
  seed/namespace actually used. Adequate-support tests prove EXACT v2-value reproduction (not `None==None`).
- **Registry-owned engine + DEV dispatcher / REGISTERED blocked stub (Pi rev-7 #4, rev-8 #1/#2/#3; RC1/RC2/RC5)** —
  `scripts/oracle_realism_v3_engine.py`: cells are built from the canonical registry ONLY (callers pass DATA as
  STRUCTURED arms `experiment → stratum → {candidate, reference}`, never check/Δ/registered/precompute); precompute
  is computed internally; non-finite → NE (Pi's two fail-open exploits — all-NaN precompute, mismatched check+Δ —
  are reproduced as adversarial tests and confirmed NE/REFUSE). All estimators use a **uniform keyword-only protocol**
  `recompute(pre, mask, *, groups, floor)` (**RC1**); a controlled self-test exercises every floor-gated estimator at
  floor 60 (evaluates) and floor 500 (refuses). Two explicit boundaries, NO module-global mutation:
  **`gate_group_dev`** — the DEVELOPMENT dispatcher (explicit hash-bound floor/B; positive non-bool floor + non-bool
  integer seed; rejects extra map artifacts; context-bound maps, one identity per (profile,regime,check); returns a
  full dev-config reproducibility record with a stable seed-invariant identity — **RC5**) — and
  **`gate_group_registered`** — a **BLOCKED STUB (RC2)**: it DECLARES the registered invariants (exact N/arm=8000,
  `B=20000`, floor 500, `α=0.04/6`, exact stratum quotas, full registry identity, approved map-set + RNG-manifest
  identities) and TESTS structured assembly + identity refusals, then UNCONDITIONALLY blocks. It never runs a
  statistic, never consumes a map set, never validates an RNG manifest; `B`/floor/`α` are not call arguments and the
  registered statistic path is unimplemented — activation is a later reviewed change after manifest generation. The
  structured registered structural-zero `(2667,2667,2666)` **assembly self-test uses DUMMY sequences and checks
  quotas/order only** (no divisibility refusal), NOT registered statistic execution. Each per-permutation recompute
  reproduces the EXACT v2 estimand to `<1e-9`. The low-level kernel is `_gate_group` (private/test-only). 20
  estimators wired across ALL SIX groups — the five full-support groups + `G_bounded_support` (boundary_short, bounded
  regime; map-context regime is per-experiment `support_regime`, not hardcoded). Estimator + group wiring is COMPLETE.
- **DRAFT-ONLY reserved-manifest schemas (Pi rev-13, reworked Pi rev-18; rev-20)** —
  `scripts/oracle_realism_v3_manifest.py`: STRICT fail-closed schema + validator + self-test for the RNG
  manifest (FULL registry: 10 experiments / 14 strata; per experiment×stratum×ROLE fixture/coupling seeds +
  content/count hashes + constructor-route/profile-config identities; reserved) and the map-set manifest
  (BOTH exemption variants — 16 with / 17 without — with registered N/floor + builder/apply source identities;
  reserved seed/namespace/path/map_identity), each binding the five identity layers + full-registry variant +
  manifest-source + schema-definition identities. Validators refuse unknown fields at every nesting level. A
  DETERMINISTIC `deterministic_schema_identity` (`56147c0f…`, env-independent) is reported separately from any
  env-dependent instance hash. NOTHING drawn/populated/frozen; the reserved manifest identities are unchanged
  so `gate_group_registered` stays BLOCKED. Freezing/populating is a separately-authorized later review.
- **Fail-closed input schemas — DERIVE-NOT-TRUST + exact (Pi rev-8 #5 / rev-10 #3 / rev-12 #1–#4; rev-13)** —
  `_canonicalize_pool` runs ONCE per experiment in `_gate_core` and REBUILDS each record via the repository's
  `derive_record(source, class_ids, timestamps)` boundary (discarding caller-supplied `cluster_ids`/`L_total`/`K`/
  `positions`) + binds the experiment's expected skeleton source — so a record with real gaps but inconsistent
  `K`/`cluster_ids` is canonicalized, not trusted, and malformed TRUSTED fields refuse. `_validate_precompute(pre,
  check, n)` is an EXACT per-estimator schema (explicit numeric non-bool dtypes; `S3_tau` exactly `(n,4)`;
  absent(all-NaN)-or-complete-finite S4/S6 rows; legal ranges — occupancy/S5/S7 ∈ [0,1], S4 contrast ∈ [-1,1], nonneg
  integer count channels, dt0 `L−K ≤ L−1`, tau `n0 ≥ n1,n2`, S6 rows sum to 1) that **distinguishes support-absence
  (`nu=0` → NOT_EVALUABLE) from malformed structure (REFUSE)**. Adversarial + degenerate self-tests: every real
  precompute validates; the wrongly-accepted cases Pi reproduced ((n,5), object dtype, partial-NaN rows,
  out-of-range) now refuse; the valid `nu=0` support-empty state and valid degenerate pools (all L=1, one cluster)
  validate. Confirmed a no-op on real fixture / structural-zero / coupled pools.
- **Registered-N boundary preflight (Pi #2, canonical constructor Pi #3, provenance Pi #6)** —
  `scripts/oracle_realism_v3_regn_preflight.py` (hash **`da499057…`**): at N=8000 the bounded S3_loggap map ISSUES
  (evaluable) — the dev-scale "structurally un-calibratable" claim is WITHDRAWN; both S3 exemptions rest on the
  DETECTION criterion (bounded detect **0.0 < 0.5**, full/SCID/MIMIC/structural-zero **1.0**), per regime with/
  without each. **structural-zero uses the CANONICAL multiscale constructor** (means 18/60/250), and the map
  artifact records the EXACT seed/namespace. Both PROVISIONAL; finalisation needs a separately-authorized power
  battery (Pi #7).
- **SPARSE + PAIRED dev-boundary demo (Pi rev-7 #1, rev-8 #7; RC1/RC4/RC5 + rev-11–rev-14 follow-through)** —
  `scripts/oracle_realism_v3_group_power.py` (hash **`f1d0f3a0…`** — successively
  `…→`09573553`→`f1d0f3a0` as the dev stable identity was rebound to the source identity layers and the
  registry grew by all six wired groups (incl. bounded); the schemas AND the
  derive-not-trust canonicalization are a NO-OP on valid pipeline data — the component p-values/verdicts/evald/argmin
  are unchanged, confirming `sample_fixture`/`apply_coupling` already emit canonical records), via `gate_group_dev`
  with STRUCTURED arms:
  perturbs EXACTLY one experiment and ASSERTS (hashed, timestamps included) every non-target arm is identical to null
  AND the target arm changed; evaluates base AND its perturbation under the SAME permutation seed and compares
  `p_g(perturbed)` to `p_g(base)` with a STRICT `<` (RC4); structural-zero data AND map-design sample via the
  canonical multiscale constructor; maps context-bound to (profile,regime,dev-floor); records the full RC5 dev-config
  identity (now including `ESTIMATOR_PROTOCOL_IDENTITY`) and ASSERTS the base and perturbed stable identities are
  EQUAL in every paired trial (`stable_id_base_eq_perturbed=True`, rev-11 RC5 follow-through). MECHANICAL (B<resolution).
  All three components are **group-sensitive**: perturbing the target experiment drives `p_g` from `1.0` to `~0.02`
  (paired direction `1.0`) in every component. The **burst-timing group** (evald 4/4) attributes cleanly — argmin
  is the perturbed primary cell (attribution `1.0`). **After the RC1 floor fix the class-mark group EVALUATES at dev
  N=810** (evald 2/2 each) — withdrawing the rev-9 "correctly NEs / legitimately stricter map" claim, which was the
  positional-floor bug, not real stricter behaviour: `mark_burst_tie` attributes cleanly to `S4_abs` (argmin `1.0`),
  while `cluster_size_mark_diversity` is group-sensitive (paired `1.0`) but its argmin lands on the **co-moved
  `S4_abs` sibling rather than the nominal `S7_abs`** (attribution `0.0`) — an HONEST mechanical-dev-scale limitation
  (the coupling moves several class-mark statistics at once; surgical single-cell attribution is a registered-scale
  question, not a dev claim). The rev-5 3-cell run is a MECHANICAL SMOKE (`29da0411…`).
- **Δ-aligned pilot (dev evidence only)** — `scripts/oracle_realism_v3_phase0_pilot.py` (hash **`24ca0123…`**),
  formula err `0.0`; the boundary decision is DEFERRED to the registered-N preflight (Pi #2); full-support
  S3_loggap null-exceedance `0.0`.
- **Randomization** — `scripts/oracle_realism_v3_randomization.py`: MC == exhaustive `p_g`; conservative finite-`B`;
  7 refusals; strict `p_g > α_group` (Pi #7).
- **Per-profile + wired-engine benchmark (Pi #6; rev-11 repair, rev-12/rev-13 provenance/identity)** —
  `scripts/oracle_realism_v3_benchmark.py`: per-profile `Σ_experiment Σ_cell cost(route, profile volume)·B_main`,
  DETERMINISTIC `config_identity` **`b1f97d1a…`** (re-minted through `…→ 09f413c9 → b1f97d1a`; it now
  binds the FOUR deterministic SOURCE identity layers (`SOURCE_IDENTITY_BUNDLE`: semantic + impl-source +
  engine-canon/schema/gate + map-source) and is no longer version-bearing; the env-dependent
  `ESTIMATOR_DEPENDENCY_IDENTITY` lives in the timing artifact, Pi rev-12/rev-13 #3). The wired timing map records the EXACT `Bs` fixture seed (`236460273103544`) and is labelled TIMING-ONLY
  (namespace `v3-benchmark-timing-map`, distinct from any reserved map-design artifact), with an assertion binding
  artifact seed to the fixture invocation. **Measured serialization + measured generation**; a **conservative 1.5×
  cap margin** (not merely <8h). Honest: SD-main ≈ 10.3 h does NOT fit one 8 h job → separately-gated per-group SD
  jobs. (Full per-group wired benchmark + re-mint follows as each remaining group's estimators are wired.)

## Still open (Pi-authorized next scope; NOT yet done in rev-20)
- **Reserved map-set + RNG-manifest FREEZE/POPULATION** (the DRAFT SCHEMAS are done — rev-18; freezing needs a separately-authorized draw) — `gate_group_registered` binds to
  `RESERVED_MAP_SET_NOT_DRAWN` / `RESERVED_RNG_MANIFEST_NOT_BOUND`, so a real registered run is BLOCKED. The RNG
  manifest must bind exact per-role/stratum seeds + generator/coupling CODE identities + profile identity +
  content/count hashes + canonical arm order (Pi rev-8 #5/#6, rev-7 #5); the map-set manifest enumerates every
  `(profile,regime,check)` + seed/namespace/N/floor/builder/code identity + output path + set-hash rule, with
  missing/extra/duplicate failing closed (Pi rev-7 #6). The final Δ-aligned exemption then needs a separately-
  authorized calibration/power battery using the frozen map (Pi rev-7 #7). *(This single bullet consolidates the
  previously duplicated RNG-manifest and map-set bullets — Pi rev-9 contract change.)*
- **Boundary-short canonical map** (Pi rev-8 #8) — route boundary-short through its canonical control constructor.
- **Full per-group wired benchmark at the registered `B=20000`** (ALL five groups' estimators are now wired); then a full per-group **wired benchmark at the registered
  `B=20000`** per exact profile/stratum, with the 1.5× margin applied to EVERY group + checkpoint/resume + real
  evidence persistence, then mint job kinds. (The current benchmark's wired measurement uses a mechanical `B=5400`
  and covers only the burst-timing group; the route surrogate already forecasts phase/seam ≈ 8.46 h, so that route
  needs a predeclared split — Pi rev-7 #8.)
- **Reference-owned frozen-map CALIBRATION draw + calibration-build preflight** — the corrected + issuance-complete
  builder is written and tested on dev fixtures; the one-time reserved-namespace DRAW is **NOT authorized** until
  Pi separately authorizes.

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
