# Oracle realism v2 — build blueprint (Option D)

Living blueprint for the redesigned aggregate-realism layer after v1 falsified on real TRAIN marginals
(`docs/oracle-calibration-result-v1.md`; options in `docs/oracle-realism-redesign-options-v1.md`). Governs a
**synthetic-only** build: no governed data, no HDF5 reopen, no execution against real data until a separate
pre-registered confirmatory gate. Recommended shape: **Option D — the frozen order mechanism wrapped in a
source-conditioned, variable-length, joint-marginal realism envelope.**

Status: forks marked *[pending scout]* await Cog scout `20260719T141540Z-jepa-e5b45c1e`
(parametric-vs-neural + JEPA-latent-autoregression). Everything else is stable and buildable now.

## Cog imports (preflight run via `scripts/cog_spoke.py context`)

- **Separate generator/executor roles from verifier roles** — the realism sampler stays strictly separate
  from the frozen order mechanism AND from the envelope verifier. Shapes M1–M4.
- **[[synthesis/orca-external-prior-specification-tests]]** — frame the realism layer as a
  *specification/realism test*, never clinical prediction. Bounds every claim.
- **[[synthesis/orca-and-jepa-representation-space-translation]]** (post JEPA/autoregression batch) — latent
  autoregressive variants are legitimately on the table but need deliberate empirical anchors; keep neural
  options gated by evidence. Clin-JEPA (Yang et al. 2026) occupies the EHR-JEPA trajectory lane → any
  latent-generative direction must differentiate.
- **Watch for reward-hacking / over-claiming / tuning-on-the-target** — drives the pre-registered external
  confirmatory gate (M4), because the six TRAIN targets are now development-seen.
- **Use scout mode while the search space is fuzzy; keep/discard only after a fixed evaluator** — the
  parametric-vs-neural fork is scout territory; the envelope is the fixed evaluator.

## Non-negotiable contracts (carried from the result gate)

1. Preserve the frozen ORDER mechanism and NUISANCE-control boundaries. **Precondition VERIFIED** (M0).
2. The redesigned generator gets a NEW version/schema/identity; it never reinterprets the v1 FAIL as a pass.
3. It must explicitly, source-conditioned, model: variable length, class composition (with structural
   zeros), cluster counts, occupancy, simultaneity (Δt=0), and the positive-gap distribution.
4. Synthetic-only development; NO governed re-read. Any confirmatory realism claim goes through a separate
   PRE-REGISTERED locked/external aggregate gate (fresh run id + schema + clearance + generator identity).
5. No latent-mechanism / transfer / counterfactual / causal / order-certification claim.

## Milestones (each is a gate)

### M0 — Order-restriction invariance (DONE, verified)
Variable length is realized as a deterministic RESTRICTION of the canonical fixed-L=8 ranking to realized
items. Proven pairwise-local: surviving pairs' `r0`/`pi_star` are identical (exact 0.0) between full and
restricted cells across all five families; sub-ranking is the restriction of the full ranking;
`invariant_hash` unchanged. Locked by `tests/test_oracle_order_restriction.py`. **Consequence: the order
mechanism needs no re-freeze for v2 — only the realism layer takes a new identity.**

### M1 — v2 identity + schema scaffolding (no behaviour change yet)
Introduce `calibration_layer_v2` → `_v3`-style versioning for the realism layer: a `realism_v2_schema_hash`
binding the new source-conditioned structure, and a `length_law`/`class_law`/`timing_law`/`join` descriptor.
Keep the default path byte-identical (as the v1 adapter did) so the mechanism/reference/C=5 identities do not
move. Gate: identity plumbing + default-path regression, Pi design pre-review.

### M2 — source-conditioned marginal layer (start at Option A parameterization)
Generalize the post-hoc adapter (the `_apply_calibration_layer` seam — marks + timestamps only, dedicated
RNG, never `s_true`/`future_events`/`nuisance_u`) to emit, per source profile:
- a **variable length** via the M0 restriction (a length law: neg-binomial / discretized log-normal);
- a **class composition** via a Dirichlet-multinomial with HARD structural zeros (e.g. SCID `state`=0);
- a **simultaneity** model (Δt=0 rate, incl. > the old 0.8 grid cap) and a **source positive-gap law**;
- cluster-count and occupancy emerge from length × composition × simultaneity.
Gate: the M3 control battery must pass on synthetic self/known-profile targets.

### M3 — synthetic control battery (the fixed evaluator; no governed data)
- **Self-recovery**: generate at a known profile → extract aggregates → re-fit → recover inside the envelope.
- **Known-profile recovery (identifiability)**: hand-crafted "SCID-like" / "MIMIC-like" ground truths →
  recover both parameters and envelope pass. *This decides A vs D:* if independent marginals cannot reproduce
  the coupled cluster/occupancy/tie law, escalate to the D joint (copula) form. *[fork pending scout]*
- **Negative controls**: source-swap MUST fail cross-source; per-check ablation fails exactly its check with
  no cross-talk; a mis-specified profile fails.
Gate: all controls pass; escalate to D-joint or (only if D fails identifiability) a marked-TPP form
*[pending scout]*.

### M4 — pre-registered confirmatory realism gate (governed; separate authorization)
Because the six TRAIN targets are development-seen, the v2 generator + envelope thresholds + exact statistics
are FROZEN and pre-registered BEFORE any locked/external read. The confirmatory target must NOT be the seen
TRAIN marginals (a held-out/locked real split never read, or an external dataset). Requires: new question,
new schema, new run id, new generator identity, spent-run accounting (as `aggcalib-microgate-run-1`), and its
own Pi micro-gate + policy-population + result gate — the same discipline just completed for v1. No post-hoc
threshold changes.

## Open forks (pending Cog scout `20260719T141540Z`)

- **A vs D vs neural**: can independent source-conditioned marginals reproduce MIMIC's clustered-tie / high-
  occupancy burst structure, or is a joint copula (D) or a marked/neural TPP (B) required? Decided empirically
  at M3-identifiability; scout to anchor the choice and the identifiability boundary.
- **Latent-space autoregression**: whether a Clin-JEPA-style latent-transition/generative direction is a
  viable heavier alternative to token-level marked-process generation, and how to differentiate from the
  occupied EHR-JEPA trajectory lane. Do not commit to it before scout + explicit review.

## Stop lines

Until an M4 pre-registered gate separately passes: `APPROVED_ORACLE_POLICY` empty; aggregate-read policy
retired; TEST sealed; no sealed certification, oracle-T4 training, manifest issuance, governed T4, or any
governed aggregate read. The v1 FAIL is never reinterpreted as a pass.

## Scout trigger

Cog frontier scout `20260719T141540Z-jepa-e5b45c1e` is LAUNCHED (parametric-vs-neural + latent-AR). Fold its
returns into M2/M3 fork decisions before committing to the D-joint vs neural escalation. No further scout
needed unless M3 identifiability is itself ambiguous.
