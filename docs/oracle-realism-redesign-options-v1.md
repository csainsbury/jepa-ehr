# Oracle aggregate-realism redesign — options analysis (post-v1 falsification)

Committed record of the redesign-options analysis commissioned after the v1 synthetic generator failed the
frozen TRAIN aggregate-realism envelope on both SCID and MIMIC (see `docs/oracle-calibration-result-v1.md`).
Produced by a Fable-5 analyst agent from the sanitized result note + the safe-public generator/calibration/
extraction code. **Synthetic-only design strategy — no governed data, no execution, no claims beyond the
recorded falsification.**

## Position — what failed, why structurally

v1 is executable and self-consistent but NOT eligible; it failed the conjunctive envelope on both sources.
This is the only admissible claim; it is not a latent-mechanism, transfer, counterfactual, causal, or
order-certification statement. The failure is structural, not a tuning miss (confirmed against the code):

- **Length is a point mass** — `L_ITEMS = 8` is frozen; every cell emits exactly 8 items, so length KS is
  forced to its ceiling 1.0, unfixable by any knob.
- **Class composition is near-uniform and unmovable** — the class draw is uniform, and `token_freq_temperature`
  is EXPLICITLY FIXED at 1.0 in the canonical fit (and tempering a uniform prior is a no-op). Real composition
  is source-skewed: MIMIC lab-dominated; SCID medication-dominated with zero `state` tokens. Class TV ≈ 0.58.
- **Cluster-count + occupancy are structural**, not functions of any knob.
- **Timing is only partly reachable** — `zero_gap_bias` can set the Δt=0 rate, but the canonical fit grid caps
  at 0.8 (`GEN_FIT_ZGB`), so MIMIC's 0.89 simultaneity lands ~0.091 outside; SCID's 0.65 passes. A single
  pooled gap ECDF cannot match two source-specific gap laws.

Only 1 of 6 checks (SCID Δt=0) was ever reachable — a specification gap correctly caught by the verifier.

## Binding constraint — variable length pushes on the ORDER boundary

The single requirement that touches the frozen order mechanism is variable length: the certified order must
become a **deterministic restriction of a canonical fixed-length ranked template** to the realized items.
Whether that preserves the invariant's semantics and still meets `ORDER_SUPPORT_FLOOR` per cell must be
verified FIRST — if it changes the invariant, the whole certification battery re-freezes and re-runs (highest
cost). Every option plugs into the existing seam that touches only marks + timestamps, never `s_true` /
`future_events` / `nuisance_u`.

## Options (effort/faithfulness spectrum)

- **A — independent source-conditioned parametric marginals (minimal):** per-source length law
  (neg-binomial / discretized log-normal), class Dirichlet-multinomial with hard structural zeros, Δt=0 model,
  per-source positive-gap law. Lowest complexity, fully auditable/hashable, one parameter per check. Risk:
  length/count/occupancy are coupled, so independent marginals may satisfy one check and miss another.
- **D — frozen order wrapped in a JOINT source-conditioned envelope (RECOMMENDED):** same scope as A, but the
  length × composition × cluster × occupancy × simultaneity block is sampled jointly (small copula / compact
  conditional sampler) to capture the coupling A ignores (MIMIC occupancy 0.95 + Δt=0 0.89 ⇒ large
  simultaneous bursts). Moderate complexity, still parametric/hashable, best protects the order boundary;
  fixes the count/occupancy checks A is most likely to miss.
- **B — marked / neural temporal point process:** highest fidelity for clustered bursty simultaneity; highest
  overfit + audit + identity-surface risk (a neural model fit to now-development-seen targets is a strong
  over-fitter). Held in reserve; justified only if the controls prove parametric A/D insufficient, and only
  with a strict external confirmatory gate.
- **C — autoregressive event-sequence head:** defines its own ordering ⇒ worst fit with "don't touch order";
  over-powered for a marginal spec test.
- **Foreclosed non-option:** bootstrap/replay of real per-sequence structure — governance forbids it
  (aggregates only, spent, development-seen). Any redesign must be a synthetic model, not a replay.

## Control / validation + confirmatory gate

Synthetic-only battery (no governed re-read): (1) self-recovery at the native profile; (2) known-profile
recovery — hand-crafted "SCID-like" / "MIMIC-like" ground truths, confirm both parameter and envelope
recovery (this decides A vs D/B — whether independent marginals reproduce the coupled count/occupancy/tie
law); (3) negative controls — a SCID-fit envelope MUST fail the MIMIC target (and vice-versa), per-check
ablation fails exactly its check with no cross-talk, a mis-specified profile fails.

Pre-registered confirmatory gate (targets are development-seen): freeze generator + thresholds + exact
statistics BEFORE any locked/external read; the confirmatory target must NOT be the seen TRAIN marginals (a
held-out/locked split never read, or external); new run id + schema + clearance + generator identity; no
post-hoc threshold changes. Neural options effectively require an external target.

## Literature anchors

- Intensity-free TPPs (Shchur et al. 2020) — log-normal mixture over inter-event gaps + a Δt=0 mixture; core
  for B, informs the A/D gap model.
- Neural / self-attentive / transformer Hawkes (Mei & Eisner 2017; Zuo 2020; Zhang 2020) — bursty marked
  sequences for MIMIC simultaneity if parametric proves insufficient.
- Autoregressive EHR generation (medGAN → EVA → HALO) with explicit length/EOS — reference for C;
  over-powered for a marginal spec test.
- Classical marked point processes + Dirichlet-multinomial (composition) / negative-binomial (length/count) /
  copula (joint) — building blocks for A/D.
- **[open, to scout] JEPA-based autoregressive rollout in embedding/latent space** — predicting future latents
  autoregressively rather than tokens; flagged by Cog's JEPA/autoregression distillation and by Chris. Not yet
  assessed against this realism-layer problem; subject of a Cog frontier scout.

## Recommendation

Build **Option D** (frozen order wrapped in a source-conditioned envelope), starting at the **A**
parameterization and escalating to D's joint structure only where the known-profile controls prove
independent marginals cannot reproduce the coupled count/occupancy/tie law — the decision to go neural is
evidence-gated, not a preference. First milestone: a source-conditioned generalization of the existing
calibration adapter adding a free length law, source class multinomial with structural zeros, source
Δt=0/burst model, and source positive-gap law — validated entirely by the synthetic control battery, no
governed re-read. Neural (B/C) held in reserve.

Constraints most at risk (priority): (1) frozen order/nuisance boundary under variable length — verify FIRST;
(2) tuning-on-the-target (development-seen) — confirmatory only via a separate pre-registered locked/external
gate; (3) new version/identity, never reinterpret the v1 FAIL as a pass.

## Two code refinements (strengthen the falsification position)

- The MIMIC Δt=0 miss is caused by the canonical fit GRID cap (`GEN_FIT_ZGB` = 0.8), not the knob's hard bound
  (0.9) — even the hard bound would not fully solve it, and count/occupancy remain structural regardless.
- `token_freq_temperature` is explicitly fixed at 1.0 in the canonical fit, so class TV is doubly unmovable.
