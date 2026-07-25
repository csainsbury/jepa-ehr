# Oracle Realism Verifier — Design Family v3 (rev-25, fresh-draw conditional randomization)

**Status:** DRAFT for Pi review. Supersedes rev-22 (Pi rev-22 = GO-WITH-CHANGES for dev work, NOT PASS for schema/compute freeze; ruling 1 + the band correction are folded here, rulings 2-4 and the benchmark rebuild are still open). The NORMATIVE design is §0–§9 + Delivered/Still-open below; the
**Changelog** records how each Pi ruling was folded. All work is development-only, synthetic-only; no calibration
build, reserved map-design draw, audit/evaluation seed, policy population, or run. There is **no production/trusted
readiness claim.** Registered mode is a **BLOCKED STUB**: its invariants are DECLARED and its structured assembly +
identity refusals are TESTED, but the registered STATISTIC path is UNIMPLEMENTED — it never runs a statistic, never
consumes a map set, never validates an RNG manifest; `B`/floor/`α` are not yet call arguments. It validates what it
can, then unconditionally blocks (reserved map-set + RNG manifest not drawn). Activation is a later reviewed change.

**Rev-23 — Pi rev-22 ruling 1 + the band correction folded (rulings 2–4 and the benchmark rebuild remain OPEN).**
Pi rev-22 = **GO-WITH-CHANGES for dev implementation; NOT PASS for schema/compute freeze or registered
qualification** (reviewed `e06d346`; reproduced all self-tests, the 400/400 rank differential battery and all
three schema IDs; stop line unchanged). Folded here: **ruling 1** — `width_proportional` selected and bound as
the registered design, boundary quotas re-minted to (2286,2286,3428), structural-zero unchanged (§8.1/§8.2) —
and the **fail-closed band correction**, a real defect in the rev-22 constructor that let shifted bands admit
`L=8` while still asserting the S9-NE guarantee. Also folded: the **canonical-constructor benchmark rebuild** (the other blocking correction) and **ruling 3**
(whole checkpoint blocks + exact `1..B` partition proof + integrity-gated shard semantics) — see §9.1 — and, at
rev-25, **rulings 2 and 4** (§9.2): the counter-addressable assignment law with an issued root seed and its own
separate manifest, streamed assignment iteration, and the sorted upper-tail rank moved into the trusted
randomization module. **ALL FOUR RULINGS AND BOTH BLOCKING CORRECTIONS ARE NOW FOLDED.** Still open: the
final-assembly validator and the remaining manifest items (structural-zero multiscale binding, executable content
canonicalizer / union hash payload, frozen-vs-draft schemas, issuer path checks). See "Still open".

**Rev-22 — the last three authorized dev-only items (Pi rev-19/20 "authorized next scope" 2, 3 and 4):**

1. **Full per-group registered-scale benchmark** at `N=8000, B=20000` (§9.1,
   `scripts/oracle_realism_v3_registered_benchmark.py`). Replaces the five-route SURROGATE cost model with a
   direct measurement of every in-scope cell's ACTUAL wired estimator at its own experiment's profile volume, and
   prices three registered-scale stages the surrogate never costed. **Withdraws the surrogate per-group hours**
   (wrong in both directions, worst case 52×) and the rev-6 "per-group jobs each fit" claim.
2. **Canonical boundary-short structured constructor** (§8.1, `scripts/oracle_realism_v3_constructors.py`), plus
   the manifest follow-through it exposed (§8.2). The declared `len_i` strata were fictitious and
   `G_bounded_support` was not runnable; the route is now executable and both allocation variants are
   identity-bound.
3. **Union/variant map-identity model** — already PREPARED (not activated) at rev-21 and re-verified here: 17
   union entries drawn once, separately-hashed 16/17 variant projections, structured set-hash payloads, every
   identity `RESERVED`.

**Three findings need a reviewer decision before any registered run can be planned:** two groups bust the 8 h cap;
the current assignment RNG law makes checkpoint/resume impossible; and the bounded-control allocation variant is a
genuine either/or. Nothing was drawn, populated or frozen; `randomization.py`, `gate.py` and `engine.py` are
untouched; `gate_group_registered` stays BLOCKED.

**Rev-21 changes folded (Pi rev-19/20 GO-WITH-CHANGES — manifest-semantics tightening):**
1. **Canonical-VALUE validation (#1)** — `_eq` compares every design-bearing field to its canonical expected value
   (not just type): `schema_version`, `content_hash_algorithm_identity`, `constructor_route`,
   `profile_config_identity`, `rng_law_identity`, `sort_key`, `set_hash_rule`, `output_path_grammar`,
   `refused_artifact_rule`, `expected_status`, per-role `count_hash`, registry/manifest-source/schema-def identities,
   and the reserved sentinels. Adversarial tests tamper each typed-but-unvalidated field Pi reproduced → all refuse.
2. **Real `profile_config_identity` (#2)** — binds the ACTUAL config: exact `PROFILES[source]` payload + source
   skeleton + constructor route + stratum allocation `(2667,2667,2666)` (recomputed and compared), not a bare
   `{profile,regime}` label hash.
3. **Seed-independent RNG-law identity (#3)** — `_rng_law_identity` hashes derivation formulas + role symmetry +
   constructor route + coupling rule with NO seed; the seed-bearing `rng_identity(seed=0)` is no longer presented as
   a law identity. A `reserved_replicate_identity` slot holds the (reserved) final replicate identity.
4. **Union/variant map-identity model (#4)** — one `union_entries` list of the 17 unique maps (drawn once) + two
   variant projections (16/17) with STRUCTURED `set_hash_payload` (`{variant, registry_identity, sorted
   (profile,regime,check,map_identity), builder/apply identities, schema identity}`), not prose; per-variant
   `variant_set_identity` + a `union_set_identity`, all RESERVED until draw. No duplicate physical draws.
5. **Schema-identity naming + real dependency-exclusion (#5)** — separate `rng_schema_identity` (`b56e5e12…`) and
   `mapset_schema_identity` (`41a1a10c…`, now DISTINCT) plus a `combined_manifest_schema_identity` (`74f9a252…`), all
   from a `_deterministic_payload()` that provably excludes the env dependency layer; a self-test asserts no
   dependency field and that mutating a synthetic dependency identity leaves the schema identities UNCHANGED.
6. **Nonempty + non-bool value checks (#6)** — `_strict` rejects empty strings and bool-as-int. Schema versions
   bumped to `-draft-3`. Still draft-only; reserved sentinels + `gate_group_registered` BLOCK unchanged.

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
   (`4d99e8aa`/`26693ca5`), the **manifest source identity**, and a **schema-definition identity**. A DETERMINISTIC
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
   the NEXT increment (the manifest rework). Benchmark `09f413c9…` → `2215f339…`.

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

### 8.1 Boundary-short canonical structured constructor (rev-22, Pi rev-8 #8 / rev-19-20 authorized item 2)

`scripts/oracle_realism_v3_constructors.py` (self-tests pass). The registry declared three length strata for the
boundary-short experiment (`len_0/len_1/len_2`, allocation `(2667,2667,2666)`, "permute WITHIN each length
stratum") and the RNG manifest bound its constructor route as `bounded_length_control` — but **nothing
implemented that route.** Every draw site used the GENERIC single-profile fixture path
(`sample_fixture("MIMIC", PROFILES["boundary_short"], N, seed)`), producing one pooled homogeneous sample. Two
consequences, both now reproduced as self-tests:

- **the declared `len_i` strata were fictitious** — a pooled `uniform_int[1,7]` draw carries no length
  stratification, so labelling its first 2667 records `len_0` bound a design identity the data did not have
  (contrast `structural_zero`, whose `_multiscale` constructor genuinely creates three length strata);
- **`G_bounded_support` was NOT RUNNABLE.** The engine's `_assemble_arms` correctly REFUSES the generic path's
  single-stratum arms against the canonical three — fail-closed behaviour working as designed, but it means the
  bounded group was wired at rev-19 with no constructor able to feed it.

The route is now executable: one `uniform_int` draw per **disjoint length band** partitioning the canonical
support, `BOUNDED_BANDS = ((1,2),(3,4),(5,7))` — same family, per-stratum bounds — exactly the shape of
`_multiscale`, which keeps the length family and varies the per-stratum location. Per-stratum seeds mirror
`_derive_seed(tag, seed, i)` with the experiment id in `tag`. The structural bound `L ≤ 7` is **asserted, not
assumed**, so the boundary-short guarantee that no 8-item block can form (S9 seam checks NE by construction)
provably survives.

**The design decision — RULED BY PI (rev-22 #1): `width_proportional` SELECTED.** Band widths are 2/2/3, so a
three-stratum bounded control **cannot simultaneously** keep the equal-ish allocation and the uniform
`L ~ U{1..7}` pooled marginal. Pi's reasoning: exchangeability strata are a sampling/permutation device and must
not silently change the declared target law, so the tiny integer-rounding residual is preferable to a 0.031774
marginal distortion. Consequences folded at rev-23:

- boundary-short registry quotas re-mint to **(2286, 2286, 3428)**; **structural-zero keeps (2667, 2667, 2666)
  unchanged** — the two stratified sources no longer share an allocation;
- the selected variant is **BOUND as the registered design** (`REGISTERED_BOUNDED_VARIANT`, canonical entry point
  `registered_bounded_arms`), not offered as a caller-selected runtime option;
- `equal_control_alloc` survives **only as a development-labelled comparison route** so the marginal-distortion
  evidence stays reproducible. *(It was named `registered_alloc` at rev-22; after the ruling that name asserted
  the opposite of the truth, so it is renamed.)*
- registry identities re-mint to **`4d99e8aa…` / `26693ca5…`**, canonical-registry `8338a2cf…`, boundary
  `profile_config_identity` `039d4dd9…`, and the manifest schema identities to `c575ced4…` / `6b0f6ce0…` /
  `e0e8d397…`.

**Fail-closed band validation (Pi rev-22 #1 — a real defect in the rev-22 constructor).** The rev-22 validator
derived its "structural bound" from the CALLER's bands (`max(band)`), so shifted bands `((2,3),(4,5),(6,8))`
validated and admitted `L=8` — 26 such sequences in the reproduction — while the route identity still asserted
the S9-NE guarantee. The rev-22 claim that `L ≤ 7` was "asserted, not assumed" was therefore **weaker than
stated**: it asserted `L ≤ max(bands)`. `validate_bands` now enforces, for ANY bands, integer/non-bool, ordered,
contiguous, disjoint, starting at 1, ending at 7 and strictly below the 8-item block size; the canonical
construction additionally requires **exactly** `BOUNDED_BANDS`. The bound is checked against the CANONICAL
support, never the caller's maximum, and a route identity is **refused outright** for any payload permitting
`L ≥ 8`. Eleven adversarial band cases (shifted, expanded, starts-high, ends-low, gap, overlap, inverted,
non-integer, bool, empty, malformed shape) are each refused at validation, at BUILD time, and at
route-identity time.

The two variants as they now stand:

| variant | allocation @ N=8000 | pooled marginal | route identity |
|---|---|---|---|
| `width_proportional` **(SELECTED — REGISTERED)** | **(2286, 2286, 3428)** | uniform to integer-allocation rounding (**2.4e-05** at N=8000; exact iff `n_total` is a multiple of 7) | `95b417b8…` |
| `equal_control_alloc` *(dev comparison only)* | (2667, 2667, 2666) | perturbed: max deviation **0.031774**; long lengths `L∈{5,6,7}` under-represented ~22 % | `5b934389…` |

Self-tests cover: route/skeleton agreement with the manifest and engine; exact allocation partitioning; the
marginal-law claim (including that `width_proportional` is exact only at multiples of 7 — an earlier
"reproduces uniform EXACTLY" claim was **wrong at N=8000** and was corrected by the test, not the other way
round); band containment and the `L ≤ 7` bound for both variants; that the arms now ASSEMBLE through the engine
for `G_bounded_support`; that the OLD generic single-stratum path is **still refused**; four adversarial
refusals (unknown variant, non-positive/bool `n_total`, missing stratum, short count, swapped strata,
non-contiguous bands); and reproducibility with `exp_id` genuinely entering the seed. **The registered
bounded-control DRAW stays RESERVED** (`RESERVED_BOUNDED_CONTROL_NOT_DRAWN`).

### 8.2 Manifest follow-through (rev-22)

Implementing the constructor exposed a defect of the class Pi rev-19/20 #2 targeted, in the code that was
supposed to fix that class: `_profile_config_identity` bound a **hardcoded** `[2667,2667,2666]` allocation into
**every** profile's configuration identity, but only the two stratified sources carry it — the eight
source-profile experiments have a single pooled stratum of 8000. The allocation is now **DERIVED from the
registry** per source (refusing if a source's experiments disagree), and the boundary route binds the **real
executable constructor identities for both allocation variants** with `constructor_allocation_variant` left
`RESERVED_NOT_DRAWN` (the choice is undecided). New self-tests assert the derived allocation matches the
registry, that pooled sources bind one 8000 stratum and no longer the stratified allocation, that pooled and
stratified sources cannot share a configuration identity, that recomputing with the old hardcoded allocation
gives a *different* identity (i.e. the fix bites), and that the two boundary variants do not collapse to one
constructor identity. The misleading constant `_STRATUM_ALLOC` is renamed `_STRATIFIED_CONTROL_ALLOC`.

Schema identities re-mint accordingly (draft, expected): RNG `b56e5e12…` → **`dbfd7bb4…`**, map-set
`41a1a10c…` → **`f2c29973…`**, combined `74f9a252…` → **`e786b4be…`**, schema-definition `cf762bed…` →
**`7b91b7bc…`**. The union/variant map-identity model is unchanged (17 union entries drawn once, 16/17 variant
projections), every reserved field stays `RESERVED_NOT_DRAWN`, both manifest identities stay the engine's
`RESERVED_*` sentinels, and `gate_group_registered` stays BLOCKED.

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

### 9.2 Counter-addressable assignment law, streaming, sorted ranks (rev-25 — Pi rev-22 rulings 2 and 4)

**Assignment law (ruling 2)** — `scripts/oracle_realism_v3_assignment.py`, self-tests pass. The rev-22 engine drew
every mask from ONE sequential `default_rng(seed)` stream, so replicate `j` could not be regenerated without
replaying `1..j-1` — which is why the whole-block split/checkpoint/resume plan of §9.1 was not implementable. The
benchmark's first proposal `sha256(namespace|group|experiment|replicate_index)` was rejected for omitting the
issued assignment ROOT SEED (issuance would not have selected the stream) and for hashing an ambiguous string
concatenation. The adopted law derives each mask from a CANONICAL STRUCTURED, domain-separated payload —
`{law_version, assignment_root_seed_identity, registry_variant_identity, group_id, experiment_id,
replicate_index}` — consumed at the FULL digest width, no truncation. `j = 0` is the deterministic observed split
and consumes **no random draw**; for each `(experiment, j)` exactly ONE mask is generated and reused by every cell
of that experiment, experiments being addressed independently under the shared MC index; IID-with-replacement is
unchanged so **duplicate assignments stay VALID and are never refused**.

All eight required tests pass: deterministic replay; monolithic == arbitrary block order == split execution ==
resume, bit-for-bit; exact index coverage of `1..B`; domain separation on root / registry variant / group /
experiment / index (each independently changes the mask); per-stratum quota preservation on every replicate;
`j=0` consuming no draw; duplicates occurring and accepted on a small support; and refusal of negative,
beyond-`B`, bool, float, blank-identifier and inverted-range inputs. Deriving against the **RESERVED (unissued)
registered root seed REFUSES** — the registered assignment stream cannot be selected while seed issuance is
blocked.

**Separate assignment manifest (ruling 2)** — the draft RNG manifest covers fixture/coupling generation only;
conflating it with assignment provenance would let fixture provenance stand in for the assignment stream. A
distinct `build_draft_assignment_manifest` / `validate_assignment_manifest` pair carries the law version, root
seed (RESERVED), law identity (RESERVED), payload field list, digest rule, observed-index / per-experiment /
replacement / stratum rules, and the per-group block structure (20 blocks of 1000 tiling `1..B` exactly). Eight
adversarial tampers refuse, and a self-test asserts the fixture RNG manifest carries **no** assignment field.

**Streaming assignments (ruling 4)** — `_gate_group` now walks the MC index, builds one mask per experiment,
fills the whole `E[:, j]` column across every cell, then releases the masks. The rev-22 path materialised
`(B+1)·M` masks per experiment up front (320,016,000 bytes each; ≈2.88 GB for a nine-experiment group).

**Rank-benchmark accounting corrected (rev-26, Pi rev-25).** Once ruling 4 made `cell_upper_p` the sorted
form, the benchmark's "reference" arm was timing the PRODUCTION function against itself while still labelling
one side quadratic / 400 MB — the run behind identity `2e68817d…` reported `speedup 1.0x` beside
`400.0 MB transient`, a self-contradiction that shipped because the rank ladder went unread. The benchmark now
times the retained quadratic oracle `_cell_upper_p_quadratic` against the production `cell_upper_p`, restoring
a real comparison (**156.3×** at `A=20001`), and a regression test pins the invariant: the two sides must be
distinct functions, agree exactly, differ ~10× in peak allocation, and show a speedup that GROWS with `A` —
a flat ~1.0× ratio being the exact signature of the bug.

**Assignment source added to the trust root (rev-26, Pi rev-25).** `SOURCE_IDENTITY_BUNDLE` gained an
`assignment_source` layer. Without it an assignment-only change would not have re-minted the gate trust root,
even though it changes which permutations every SD cell is scored against. Verified by construction: a
synthetic edit to the assignment module moves the bundle hash, and reverting restores it exactly.

**Sorted upper-tail ranks (ruling 4)** — the differential proof now lives in the trusted randomization module,
not in the benchmark that first measured it. `cell_upper_p` is the sort/`searchsorted` form; the quadratic
`A × A` version is retained as `_cell_upper_p_quadratic`, the differential oracle. 600 adversarial cases are
bit-identical (heavy ties, all-constant, one `+inf` NE sentinel, mixed NE, and all-NE vectors). Input validation
fails closed: NaN is REFUSED (the frozen NE policy encodes an undefined statistic as `+inf`, never NaN, so NaN
here means a broken precompute reached the ranking stage), as are `-inf`, non-1-D and empty vectors, while a
legal all-`+inf` vector is still accepted with `p = 1.0`.

### 9.1 FULL per-group registered-scale benchmark at `B = 20000` (rev-23, MEASURED on CANONICAL constructors)

`scripts/oracle_realism_v3_registered_benchmark.py` (deterministic `config_identity` **`e441472e…`**; self-tests
pass). Every in-scope cell is priced by measuring **its own engine estimator** at registered scale
(`N=8000`/arm, `M=16000`), and — since rev-23 — on arms assembled through **that experiment's canonical
constructor route** in the engine's canonical pool order.

**rev-23 rebuild (Pi rev-22 blocking correction).** rev-22 drew every arm through the generic
`draw_arm(profile)` and concatenated `cand + ref`. That was wrong twice over: it ran neither the structural-zero
`_multiscale` constructor nor the bounded-length constructor, and the flat concatenation does not match the
engine's canonical pool order `[cand_s0, ref_s0, cand_s1, ref_s1, …]` for the two STRATIFIED experiments.
Measurement is now keyed by **experiment**, not profile — 194 `(experiment, statistic)` measurements — because
repeatability experiments carry a coupling on both roles and the two stratified experiments are built by their
own constructors, all of which move event volume and support cardinality. The effect is large and in both
directions: structural-zero `S9_gap` **179.6 → 64.3 ms** under the true multiscale constructor, while coupling
raises `repeat_burst_timing_scid` `S9_gap` to **569 ms** against `null_scid`'s **463 ms** — a 23 % spread that
per-profile measurement could not see. A self-test proves the benchmark's pool order equals
`_assemble_arms`, and that the two stratified experiments do not run the generic route. The exact constructor and
profile payloads and this module's own source identity are now bound into the deterministic configuration.

Maps used for timing are TIMING-ONLY artifacts in a distinct namespace built from the canonical pool's reference
positions; no reserved draw, calibration/evaluation seed, manifest population or persisted run.

**The surrogate per-group hours are WITHDRAWN.** Measured against surrogate (`with_exemption`, hours):

| group | K | surrogate | MEASURED | error |
|---|---|---|---|---|
| `G_full_phase_seam` | 45 | 6.68 | **14.01** | understated 2.1× |
| `G_full_length_density` | 36 | 0.105 | **5.08** | understated **48×** |
| `G_full_burst_timing` | 36 | 1.70 | **2.59** | understated 1.5× |
| `G_full_class_mark` | 54 | 0.12 | **0.25** | understated 2× |
| `G_full_run_size` | 9 | 1.66 | **0.06** | **overstated 28×** |
| `G_bounded_support` | 12 | 0.03 | **0.04** | ≈ |

The surrogate erred in BOTH directions because its per-item KS route mis-modelled the real estimators: `count_ks`
/ `length_ks` / `S2_ks` recompute over an `(n × |support|)` indicator matrix, so their cost scales with the
**support cardinality**, not with the sequence or cluster count. This is exactly the rebuild Pi rev-5 #6 required
(`Σ_experiment Σ_cell measured_cost`), and it is why a surrogate forecast must not gate a job plan.

**ONE group busts the cap; a second is MARGINAL.** At the 8 h cap with the required 1.5× margin,
`G_full_phase_seam` needs **21.02 h** — over by 2.6× — so the rev-6 "per-group jobs each fit" claim stays
**WITHDRAWN**. `G_full_length_density` now lands at **7.62 h**, i.e. it nominally fits but with only **1.0 %
headroom**. Because these hours are environment-dependent and have moved ~5 % between reruns on this machine, a
"fits" verdict inside that variance is not a safe classification: the benchmark therefore reports an explicit
**MARGINAL** status for any group with less than 10 % cap headroom, and both `length_density` and the ratified
3-shard `phase_seam` split fall into it. Cost is overwhelmingly permutation recompute (phase_seam: 99.9 %),
concentrated in a few cells — SCID `S9_gap` is 2.6–3.2 h **per cell** and phase_seam contains four of them.

**Split rule (normative; amended per Pi rev-22 #3).** The divisible unit is a **WHOLE CHECKPOINT BLOCK** (20
blocks at `B=20000`), never `ceil(B/jobs)` replicates. A group whose single-job forecast exceeds `cap/margin` is
split into the smallest number of **sequentially gated shards** such that
`fixed_overhead + divisible·⌈n_blocks/jobs⌉/n_blocks` fits the budget, plus one final ranking/aggregation job.
Pricing the worst shard at `⌈n_blocks/jobs⌉/n_blocks` rather than an equal fraction matters: the rev-22 equal-third
figure of 6.92 h understated the executable 7/20-block maximum, exactly as Pi said. Fixed overhead (assemble +
canonicalise + precompute) is repaid by every shard; the min-p ranking runs ONCE at the end over the assembled
`E` matrix. `B=20000` still satisfies `B ≥ K_max/α_group = 8100`.

Every plan and self-test **proves an exact partition of `1..B`** — explicit inclusive per-shard replicate ranges,
no gap, overlap, duplicate or extra replicate. The rev-22 check only asserted `jobs × ⌈B/jobs⌉ ≥ B`, which permits
an extra replicate; `verify_block_partition` now rejects both an over-covering and an overlapping partition, and
those rejections are themselves self-tested.

A shard is an **integrity-gated block stage, not a scientific gate**: no within-group PASS/FAIL exists until final
assembly, so "stop-on-failure" within a split means execution / identity / checkpoint failure only. Scientific
stop-on-failure resumes at the complete group verdict.

Measured allocations (`with_exemption`), reporting BOTH the ratified rule and a headroom-safe recommendation:

| group | status | ratified (smallest that fits) | recommended (≥10 % headroom) |
|---|---|---|---|
| `G_full_phase_seam` | OVER cap | `[7,7,6]` @ **7.38 h** (5.6 % headroom) | **`[5,5,5,5]` @ 5.29 h (33.9 %)** |
| `G_full_length_density` | MARGINAL | `[20]` @ **7.62 h** (1.0 % headroom) | **`[10,10]` @ 3.83 h (52.1 %)** |
| `G_full_burst_timing` | fits | `[20]` @ 3.89 h (51.4 %) | `[20]` |
| `G_full_class_mark` / `G_full_run_size` / `G_bounded_support` | fits | `[20]` @ ≤0.38 h | `[20]` |

Pi's provisional `length/density 10+10` is **vindicated** — not because the raw forecast demands it (one shard
nominally fits) but because one shard leaves 1 % headroom. Pi's provisional `phase/seam 7+7+6` is **not enough**:
at three shards the worst shard retains only 5.6 % headroom, so **4 shards `[5,5,5,5]` is recommended**. The
solver still implements the rule as ratified; the headroom-safe allocation is reported alongside it as a proposed
amendment rather than substituted for it.

**Checkpoint / persistence (normative).** Checkpoint unit = permutation-replicate block (1000 replicates, 20
blocks/group). Block state = that block's columns of the per-cell discrepancy matrix `E[K, block]` ONLY (432 KB;
the full deepest-group null matrix is 8.64 MB). A resumed job recomputes only missing blocks and must re-verify the
bound identity set (registry, source identities, map-set identity, RNG-manifest identity, floor, `B`, `α_group`,
group id) and **REFUSE on any mismatch rather than merge across identities**. Persistence stays **aggregate-only**:
the result artifact carries verdict / `p_g` / `α_group` / `B` / `K` / `argmin_cell` / observed per-cell `p` and `e`
/ null-`S` quantiles / the bound identity set / a per-block assignment-digest manifest (1355 bytes measured). Never
persisted: per-permutation `E` columns beyond the live checkpoint, assignment masks, canonicalised pools,
precompute payloads.

**Three registered-scale stages the surrogate never costed** (all invisible at the dev `B ≤ 5400`):

1. **Assignment materialisation — RAM.** `_gate_group` builds `[canonical] + [perm]×B` masks for every experiment
   before any statistic. Stated precisely (Pi rev-22): `(B+1)·M` = **320,016,000 mask bytes per `M=16000`
   experiment**, hence a **≈2.88 GB assignment-mask payload for nine experiments** — the larger **5.4–8.2 GB**
   figures are TOTAL process forecasts including pools, precomputes and other arrays, not masks alone. The masks
   are held across the whole recompute phase. Streaming per replicate is `O(M)` and brings total peak RAM to
   0.03–2.81 GB
   (`G_full_length_density` remains 2.81 GB because its `length_ks`/`count_ks` precompute indicator matrices are
   266 MB and 176 MB per SCID experiment — that part is irreducible without changing the estimator).
2. **Min-p ranking — RAM, not time.** `cell_upper_p` forms an `A × A` boolean matrix (`A = B+1`): at `A = 20001`
   that is an exact **400.0 MB transient per cell**, 216.9 ms/cell measured. A `sort`/`searchsorted` formulation is
   **bit-identical** (proven here over 400 adversarial cases spanning heavy ties, `+inf` NE sentinels, all-constant
   and all-NE vectors) at 1.39 ms and `O(A)` memory — 156×. **Honest scale: ranking is ~10 s per group, so this is
   a memory and hygiene fix, NOT a feasibility blocker.**
3. **Block-addressable assignments — a genuine blocker for the checkpoint plan.** The current law draws every mask
   from ONE sequential `default_rng(seed)` stream, so block *k* cannot be regenerated without replaying blocks
   `0..k-1`. **The split/checkpoint/resume plan above is therefore not implementable under the current law.** The
   measured alternative derives a per-replicate seed from
   `sha256(namespace | group | experiment | replicate_index)`, costing **+1.4 % to +8.9 %** per mask. This CHANGES
   the assignment RNG law that the RNG manifest binds, so it is a **reviewer decision and has NOT been applied.**

**None of the three fixes has been applied** — `randomization.py`, `gate.py` and `engine.py` are untouched by
rev-22. The benchmark measures both the current and the proposed path and reports each separately, so the reviewer
decides. Two identities are emitted as before: the deterministic `config_identity` (formula, per-cell routing,
per-experiment strata/volumes, `B`, cap/margin/block, split rule, source identity layers) and a separate
environment-dependent timing artifact (seconds, hours, RAM — NOT reproducible across machines).

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

## Delivered (rev-23, Pi rev-9/rev-10/rev-12/rev-13/rev-18/rev-19-20/rev-22 authorized dev-only scope — all tested)
- **FULL per-group registered-scale benchmark + cap/checkpoint/persistence plan (rev-22, REBUILT rev-23 on
  canonical constructors, §9.1)** — `scripts/oracle_realism_v3_registered_benchmark.py`, deterministic
  `config_identity` **`e441472e…`**, self-tests pass. **194** measured `(experiment, statistic)` costs at
  `N=8000`/arm on canonically assembled arms in `_assemble_arms` pool order; per-group forecast at `B=20000` for
  BOTH variants; measured ranking / assignment / persistence stages; whole-block split with an exact `1..B`
  partition proof, checkpoint and aggregate-only persistence plans; explicit MARGINAL status below 10 % cap
  headroom. Withdraws the surrogate per-group hours and the rev-6 "per-group jobs each fit" claim:
  `G_full_phase_seam` **14.01 h** is over cap (recommend 4 shards `[5,5,5,5]`) and `G_full_length_density`
  **5.08 h** is MARGINAL at 1.0 % headroom (recommend `[10,10]`, vindicating Pi's provisional allocation). Binds
  the exact constructor/profile payloads and the benchmark's own source identity. The surrogate benchmark is
  unchanged and still reproduces its own `config_identity` **`2215f339…`** — it binds cell routing and measured
  volumes, not the registry quotas, so the boundary re-mint does not move it.
- **Canonical boundary-short constructor (rev-22, §8.1)** — `scripts/oracle_realism_v3_constructors.py`,
  self-tests pass. Makes the declared `bounded_length_control` route executable (three disjoint length bands over
  `L∈[1,7]`), proves the `L≤7` structural bound and the S9 NE guarantee, makes `G_bounded_support` assemblable for
  the first time, keeps the old generic path refused, and identity-binds BOTH allocation variants
  (registered `95b417b8…` / dev-comparison `5b934389…`). Registered draw RESERVED.
- **Manifest allocation/constructor follow-through (rev-22, §8.2)** — `scripts/oracle_realism_v3_manifest.py`:
  `profile_config_identity` now DERIVES each source's stratum allocation from the registry instead of binding one
  hardcoded `(2667,2667,2666)` to every profile, and binds the real boundary constructor identities for both
  variants with the variant choice left RESERVED. Schema identities re-mint to **`dbfd7bb4…`** (RNG) /
  **`f2c29973…`** (map-set) / **`e786b4be…`** (combined) / **`7b91b7bc…`** (schema-definition); union/variant
  model and every RESERVED sentinel unchanged.
- **Registry (identities re-minted, Pi #5)** — `scripts/oracle_realism_v3_registry.py`: full per-cell identity +
  strict refusal; `Δ` bound to LIVE thresholds (Δ-hash **`ec6f4dff…`**). S3_tau identity DECOUPLED from the
  fragile whole-pilot aggregate hash to a stable frozen-estimator descriptor; S6 estimator text → LENGTH_BINS;
  audit prose removed; `B≥K/α` reframed as a conservative design rule. Both variants **M0 = 192 / 194**
  (hashes **`4d99e8aa…` / `26693ca5…`** — re-minted at rev-23 by the boundary allocation ruling); `G=6`, `K_max=54`.
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
  DETERMINISTIC `config_identity` **`2215f339…`** (re-minted through `…→ 09f413c9 → 2215f339`; it now
  binds the FOUR deterministic SOURCE identity layers (`SOURCE_IDENTITY_BUNDLE`: semantic + impl-source +
  engine-canon/schema/gate + map-source) and is no longer version-bearing; the env-dependent
  `ESTIMATOR_DEPENDENCY_IDENTITY` lives in the timing artifact, Pi rev-12/rev-13 #3). The wired timing map records the EXACT `Bs` fixture seed (`236460273103544`) and is labelled TIMING-ONLY
  (namespace `v3-benchmark-timing-map`, distinct from any reserved map-design artifact), with an assertion binding
  artifact seed to the fixture invocation. **Measured serialization + measured generation**; a **conservative 1.5×
  cap margin** (not merely <8h). Honest: SD-main ≈ 10.3 h does NOT fit one 8 h job → separately-gated per-group SD
  jobs. (Full per-group wired benchmark + re-mint follows as each remaining group's estimators are wired.)

## Still open (Pi-authorized next scope; NOT yet done in rev-21)
- **Reserved map-set + RNG-manifest FREEZE/POPULATION** (the DRAFT SCHEMAS are done — rev-18; freezing needs a separately-authorized draw) — `gate_group_registered` binds to
  `RESERVED_MAP_SET_NOT_DRAWN` / `RESERVED_RNG_MANIFEST_NOT_BOUND`, so a real registered run is BLOCKED. The RNG
  manifest must bind exact per-role/stratum seeds + generator/coupling CODE identities + profile identity +
  content/count hashes + canonical arm order (Pi rev-8 #5/#6, rev-7 #5); the map-set manifest enumerates every
  `(profile,regime,check)` + seed/namespace/N/floor/builder/code identity + output path + set-hash rule, with
  missing/extra/duplicate failing closed (Pi rev-7 #6). The final Δ-aligned exemption then needs a separately-
  authorized calibration/power battery using the frozen map (Pi rev-7 #7). *(This single bullet consolidates the
  previously duplicated RNG-manifest and map-set bullets — Pi rev-9 contract change.)*
- **Pi rev-22 rulings — ALL FOUR now folded (1/3 + benchmark rebuild at rev-23/24; 2 and 4 at rev-25).**
  Retained here for the record, with the one open sub-item called out:
  1. **[DONE at rev-25] Ruling 2 — assignment RNG law** (§9.2). Counter-addressable per-replicate derivation, but
     the proposed `sha256(namespace|group|experiment|replicate_index)` **omits the issued assignment root seed**,
     so issuance would not select the stream. Needs a structured, domain-separated canonical payload
     (`law_version`, issued root seed / its manifest-bound identity, registry variant identity, `group_id`,
     `experiment_id`, `replicate_index ∈ 1..B`), full digest or a frozen truncation width, `j=0` as the
     deterministic observed split consuming no draw, one mask per `(experiment, j)` reused by every cell of that
     experiment, IID-with-replacement retained. Requires a **separate assignment-law/root-seed section or a
     distinct assignment RNG manifest** — fixture/coupling provenance is not assignment provenance. Tests:
     monolithic == arbitrary block order == split == resume bit-for-bit; exact index coverage; root/group/
     experiment/index domain separation; quota preservation; deterministic replay; duplicates accepted;
     malformed/out-of-range indices refused.
  2. **[DONE at rev-23] Ruling 3 — whole-block splits** (§9.1): whole checkpoint blocks, worst shard priced at
     `⌈n_blocks/jobs⌉/n_blocks`, exact `1..B` partition proved and its violations self-tested, integrity-gated
     shard semantics recorded. **Open sub-item:** the FINAL ASSEMBLY validator (bind/verify cell order, block
     range, assignment-law/root identity, registry variant, map set, RNG fixture manifest, floor, `B`, `α`, group,
     code identities and per-block content/assignment digests; refuse missing/overlapping/duplicated/foreign
     blocks; keep checkpoint `E` ephemeral) is specified but NOT yet implemented — it needs ruling 2's
     assignment-law identity to bind against.
  3. **[DONE at rev-25] Ruling 4 — streaming + sorted ranks** (§9.2). Iterate by MC index, one mask per experiment, fill the
     whole `E[:,j]` column across cells, release; do not regenerate masks per cell. Move the differential proof
     into the trusted randomization module, refuse NaN/malformed/non-1D input, retain finite/`+inf`/tie/constant/
     all-NE semantics. Both re-mint engine/randomization source identities.
  4. **[DONE at rev-23] Benchmark rebuild** — inputs now run through the exact canonical constructors in
     `_assemble_arms` order, remeasured per EXPERIMENT (194 pairs), forecast reissued, and the exact
     constructor/profile payloads plus the benchmark source identity bound into the deterministic configuration
     (§9.1).
- **Remaining manifest items before freeze review** — `profile_config_identity` must additionally bind the
  structural-zero **multiscale means / executable constructor + source identity / exact seed derivation** (it
  currently binds the base `PROFILES`, a route label and quotas); `_CONTENT_HASH_ALGO` and the union-set hash must
  become **exact executable canonical payload/serialization definitions** rather than English descriptions; plus
  frozen-vs-draft schemas, issuer path/symlink checks, and builder/validator source identities.
- **[RESOLVED at rev-23] Bounded-control ALLOCATION VARIANT decision** — Pi rev-22 #1 selected
  `width_proportional` (2286,2286,3428), now bound as the registered design; `equal_control_alloc` is retained
  only as a development comparison route. See §8.1. The rev-8 #8 "route boundary-short through its canonical
  control constructor" item is DONE. The registered bounded-control **draw** stays RESERVED.
- **Job-kind minting for the split SD jobs** — §9.1 delivers the measured per-group forecast, the cap verdict, the
  split rule and the checkpoint/persistence plan, but the split job kinds / run IDs are NOT minted (that belongs
  with the launch-manifest work, which is blocked).
- **Reviewer decision on the three registered-scale implementation changes** surfaced by §9.1, none of which is
  applied: (a) streamed per-replicate assignments instead of materialising `(B+1)·M` masks (RAM: 5.4–8.2 GB →
  ≤2.81 GB per group); (b) the bit-identical sorted min-p ranking (400 MB transient per cell → `O(A)`);
  (c) **blocking** — the block-addressable assignment RNG law, without which the split/checkpoint/resume plan
  cannot be implemented at all. (c) changes a law the RNG manifest binds and so cannot be adopted unilaterally.
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
- **rev-22 preflight (2026-07-24):** the Cog planning query for this benchmark work (route-identity binding,
  reserved-but-unexecuted draw paths, honest compute forecasting) returned only ORCA/JEPA representation-space
  material — **no relevant distilled guidance**, so §9.1 imports nothing from Cog. Candidate Cog feedback from
  this rev: *a surrogate cost model must not gate a compute plan* — the five-route surrogate erred 52× low on one
  group and 41× high on another, because it modelled cost by sequence/cluster volume while the real estimators
  scale with support cardinality.

## Scout trigger
Resolved by the permutation architecture + Phase-0 pilot + the §9 feasibility benchmark. No scout needed for v3;
revisit only on a genuinely new fork. Launch only on Chris's explicit authorization regardless.

## Stop line
Design + (on Pi ratification) the registry artifact + calibration-build preflight only. No expensive evaluation
run, M3a freeze, M2, governed read, TEST, sealed certification, oracle-T4, policy population, or governed T4 until
Pi re-gates the frozen registry + benchmark.
