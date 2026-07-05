---
title: Clinical-JEPA latent-native generation & counterfactual — design + decision record
created: 2026-07-05
status: draft-for-review (NOT approval to run compute)
project: ascend-orca-clinical-jepa
safety_tier: safe_distilled
extends:
  - docs/clinical-jepa-v0-blueprint.md (Part C — action-conditioned latent transition)
  - docs/clinical-jepa-next-experiment-brief.md (generation / readout / TTE readiness)
consult: fable5 (external, pure-abstract ML methods consult; two rounds)
consult_transcript: ascend-flat:coordination/fable5_jepa_thread.md (titan; verbatim two-round thread)
substrate: joint MIMIC+SCI-D corrected 350M flex (step_150000, vocab 1050) — see §1a; supersedes old B1a/85M
Pi-reviewed: GO-WITH-CHANGES (2026-07-05) — required changes incorporated (§4a); verdict in coordination/jepa_pi_thread.md
---

# Clinical-JEPA latent-native generation & counterfactual — design + decision record

## 0. One-line objective

Decide how (and whether) to turn the working Clinical-JEPA latent **future-block predictor** into an engine for **accurate autoregressive generation** and **counterfactual (action-conditioned) generation** of explicit `(token, Δt)` event sequences — and set the cheapest-first experiment ladder that adjudicates the architecture choice rather than committing to it on priors.

This is a **planning / decision artifact**, not approval to access data or run compute. Any rung that touches governed data runs under the existing governance boundary (aggregate-only, prefix-safe, no raw tokens / IDs / checkpoints in reports).

## 1. Where v0 leaves us (established)

v0 established a real, well-controlled result at the **representation/retrieval** level (see `b1a-real-pilot-progress-report.md` / `clinical-jepa-v0-research-narrative.md`):

- The JEPA latent future-block prediction objective learns **non-collapsed** representations on governed re-keyed tokenised EHR.
- Predicted latents **retrieve** the true future block above frozen-FlatASCEND and utilisation baselines under matched/candidate-normalised distractors.
- Retrieval **degrades monotonically with horizon**, survives shuffle / query-shuffle / time-shift controls, and shows MIMIC→INSPECT transfer *(on the retired old-tokeniser B1a substrate — see §1a)*.

What v0 did **not** touch — and what this note is about:

- no decoder/renderer emitting explicit future `(token, Δt)` sequences;
- no JEPA-conditioned autoregressive rollout;
- no plausibility evaluation of *generated* futures;
- action-conditioned / counterfactual latent transition (blueprint Part C) is parked, with the standing guardrail against calling latent shifts "treatment effects."

**Frontier restated:** latent future-state *retrieval* → accurate *autoregressive generation* → *counterfactual* (action-conditioned) generation. The retrieval win does **not** automatically transfer to generation (see P5).

## 1a. Substrate update (2026-07-05) — re-base on the joint MIMIC+SCI-D corrected model

The v0 pilot above ran on the **old tokeniser** (the re-keyed B1a MIMIC+INSPECT bundle + the frozen FlatASCEND-85M `step_100000`). Those are **retired** for the go-forward plan. This note's live substrate is the **from-scratch joint MIMIC+SCI-D corrected 350M "flex" model** (~318M params, checkpoint `step_150000`, 2026-07-02; joint vocab hash `4b57b210…`; concrete checkpoint/substrate paths kept local per the safety boundary):

- **Larger vocabulary: 1,050 tokens** (vigintile-factored; 43 lab measures × 20 vigintiles), with B1-corrected lab attribution, floor-gated richer drug classes, complication-DX presence tokens, and TYPE / DURATION / SIMD anchors — vs the old ~220-token MIMIC tokenisation.
- **Substrate:** the joint corrected build — per-source vigintile boundaries and a `DATASET:SCID / DATASET:MIMIC` source token at sequence start (concrete local paths uncommitted). The flat-token JEPA arm trains on these tokens; any frozen-teacher arm uses the 350M flex model's hidden states (not the 85M).
- **The P4 two-source asset changes: MIMIC (US ICU) ↔ SCI-D (Scottish outpatient diabetes registry)**, replacing MIMIC/INSPECT. This is a *stronger, cleaner behaviour-policy contrast* for fable5's cross-environment invariance test (§3.2 P4 / §6 rung 4), and the `DATASET` token + per-source boundaries make source-stratified counterfactual tests clean. **INSPECT is not in this substrate.**
- **Not the 3-way aggregate.** A newer MIMIC+SCI-D+**CPRD** aggregate (vocab 1071, 2026-07-05) has far more rare-drug power, but CPRD is a **2-month LOCF panel** whose coarse timing would undermine the continuous-time / marked-TPP core (P3). Per-event MIMIC + SCI-D is the better substrate for *this* generation work; revisit the aggregate only if a drug-power-limited counterfactual arm specifically needs it (and then handle CPRD's coarse timing explicitly).

Read every "MIMIC/INSPECT" / "FlatASCEND-85M" reference below as the **retired-substrate record**; the live substrate is the joint MIMIC+SCI-D corrected 350M model and its 1,050-token vocab.

## 2. The five methodological problems (extraction)

Abstract setup: sequences of `(token ∈ V, Δt ∈ ℝ₊)` over irregular continuous time; context encoder `f_θ`; EMA/stop-grad target encoder `f_ξ`; pooled target latent `z⁺`; predictor `g(context, horizon) → ẑ`; loss `1 − cos(ẑ, z⁺)` + var/cov anti-collapse.

- **P1 — Inversion / conditional-mean collapse.** `f` was never trained to be invertible and pooling is many-to-one, so `z⁺` is not identifiable back to a sequence. And `g` is a deterministic regressor to a single target: for multimodal futures its minimiser is the **conditional-mean embedding**, which decodes to blurred, non-committal sequences (the sequence analogue of L2→blurry-video).
- **P2 — Multi-step rollout stability in a learned, non-stationary latent space.** Iterating `ẑ_{t+1}=g(state_t,·)` compounds error: (a) exposure-bias / off-manifold **drift**; (b) EMA-style iterated maps **contract to a mean/attractor**. Compounded by the EMA target being non-stationary during training.
- **P3 — Continuous-time / irregular Δt.** Block pooling discards timing, yet generation needs exact `Δt`; event-count vs wall-clock horizons trade rate-leakage against a variable count; `Δt` is heavy-tailed and zero-inflated; counterfactual actions shift *when* as well as *what*, and rate is itself confounded.
- **P4 — Interventional identifiability from policy-confounded data.** Actions are policy-selected (confounded with state), so `g(c,a)=E[z′|c,a]` is associational; a counterfactual `a*` is off the action-propensity manifold; latent "action directions" may re-encode propensity rather than a genuine transition operator.
- **P5 — Retrieval-as-proxy + validation circularity.** All current evidence is discriminative *retrieval*; the goal is conditional *generation*. High Recall@k can ride coarse cluster identity while lacking decodable fine detail. And target/metric/reward can all live in one encoder's frame → circular validation (the failure mode that sank the sibling DPO effort: reward ≡ eval).

## 3. Consult verdict (fable5, latent-native framing)

External pure-abstract methods consult, two rounds; verbatim thread at `ascend-flat:coordination/fable5_jepa_thread.md`. The consult was run **under the committed constraint** of a full (from-scratch) JEPA pipeline, not reusing the existing FlatASCEND AR speaker.

### 3.1 The organising idea — "the decoupling"

The single latent is overloaded with six roles (representation, generative target, rollout state, counterfactual-operator domain, reward, metric); that overloading causes all five problems. Split it: **(1) frozen representation encoder, (2) distributional latent predictor `p(z|context)`, (3) continuous-time head on the latent, (4) latent action-operator, (5) observation-space metrics at a terminal read-out.** In the latent-native commitment the decoder is confined to **terminal read-out + validation bridge** (not a per-step crutch); rollout stays in latent.

Standing hazard fable5 flagged: a fully-latent process risks **self-referential drift you cannot see** — target, predictor, rollout and metric share one coordinate system, so the model can look successful while diverging from reality. This makes **P2 (rollout stability without an external anchor)** and **P5 (non-circular validation)** load-bearing.

### 3.2 Per-problem headline recommendations

- **P1:** never decode the predictor's *mean* — decode a **sampled** latent (`ctx → p(z|ctx) → sample → read-out`). Make the latent decodable via **sequence-of-latents** or **VQ/discrete** targets (VQ also turns "distribution over latents" into a categorical that is stable against the moving EMA target and kills mean-collapse for free). Cheapest discriminator: freeze encoder, train only a read-out `D`; `D(z⁺)` can't reconstruct ⇒ representation bottleneck (change targets); `D(z⁺)` fine but `D(mean)` blurs while `D(sample)` is crisp ⇒ predictor bottleneck (go distributional).
- **P2 (latent-first):** **stochastic predictor + dynamics-level (k-step, rolled-out) variance regulariser** is the highest-value move (fixes both drift and attractor-collapse); add rollout-in-the-loop consistency to encoded ground truth, a learned latent projection-to-manifold, and/or an invertible/structured operator. **Two-phase: EMA-learn representation → freeze `f_ξ` → learn dynamics/read-out.** Obs-space (decode→re-encode) rollout is a **concession** (defeats the thesis) kept only as a diagnostic upper bound / optional every-K-step re-grounding. Cheapest discriminator: a no-training rollout sweep recording `d_t` (distance to nearest true latent) and `v_t` (ensemble variance) — `d_t`↑ ⇒ drift, `v_t`→0 ⇒ collapse, blows up only under a live EMA ⇒ two-phase freeze.
- **P3:** separate "what/where" (latent) from "when" (a continuous-time head). Prefer **wall-clock horizons** (or "latent state at query timestamp `t`") over **event-count** (which leaks rate and confounds counterfactuals). Head: **marked TPP on the latent** > ZILN/mixture > CT-state (ODE/CT-RNN) > time-tokens. Carry **absolute/cumulative time anchored to a scheduled clock**, not summed `Δt`. Evaluate with the **time-rescaling theorem** (QQ/KS of rescaled inter-event times vs Exp(1)), teacher-forced vs rollout.
- **P4 (where the latent route pays off):** the counterfactual is **only partially identified** — deliver an **overlap-gated, sensitivity-bounded latent operator with an explicit abstention region and validity horizon**. Combine **overlap-restriction (report it) + a structural constraint on `T_a`** (additive/factored/invertible/composable). **Strongest diagnostic = cross-environment invariance** (a genuine operator is invariant across policies; a re-encoded propensity is not) — fit the effect on env A, test counterfactual accuracy on env B; **near-free using the MIMIC (US ICU) / SCI-D (Scottish outpatient) two-source asset** (see §1a; a stronger policy contrast than the retired MIMIC/INSPECT). Off-support **compounds multiplicatively** over rollout → per-step overlap gate with abstain.
- **P5 (load-bearing):** retrieval rewards *between*-cluster discriminability; generation needs *within*-cluster fidelity. Battery: frozen-latent decode ceiling; order/time-perturbation hard-negative retrieval; targeted decodability probes. Non-circular validation axes (strongest→weakest): **forward-prediction of never-encoded raw quantities** > semi-synthetic known-effect env (the only interventional yardstick for P4) > held-out modality > external labels; **anything in the encoder's own latent is inside the circle** (dev signal only). Run the **obs-space-forward-prediction vs latent-retrieval "scissors" as a standing alarm.**

### 3.3 Round 2 — hierarchy, the falsifier, and the refutation

- **Hierarchy (R2.1):** a two-level **coarse-plan-conditions-fine-render** JEPA (with a cycle-consistency + count-consistency interface) is the right latent-native P2 fix — it cuts the **exponent** of multiplicative compounding and re-grounds fine drift to the coarse plan — **but only if a real timescale separation exists**; otherwise it relocates drift to the coarse level. P3's "state at `t`" factors cleanly (coarse = rate/intensity envelope, fine = marked-TPP under it). **Cheapest gate (near-free): coarse-vs-fine horizon-decay curves vs wall-clock — build the hierarchy only if the coarse level decays substantially slower per unit time.**
- **The falsifier, honestly (R2.2):** fable5 **conceded** its round-1 "the only thing outside the circle" was an overclaim. Decode-to-raw is only *partially* outside: the raw comparison target is external, but the co-trained decoder can **mask latent failure via its generative prior**. Escape is asymptotic, via a **ladder**: latent-corruption sensitivity curve (falsifier-of-the-falsifier) → `D(z⁺)` vs `D(ẑ)` ceiling split (component attribution) → **decoder-free summary statistics** (counts, marginals, inter-event-time quantiles predicted straight from the latent) → **semi-synthetic oracle** (the only fully-external anchor; the deepest part of the circle is the encoder itself).

## 4. The premise decision

### 4.1 fable5's refutation (R2.3)

For **generation quality per se**, a flat AR `(token, Δt)` speaker is the stronger, simpler baseline; P1–P3 are difficulties the latent route inflicts on itself. The load-bearing point: **P4 justifies a latent *plan*, not a latent *renderer*.** The operator `T_a` acts at the plan level; nothing about the counterfactual requires that *rendering* also be latent-native. So the honest front-runner is the **hybrid** — JEPA latent plan + operator at the plan level + an AR speaker rendering conditioned on the (possibly counterfactual) plan. The pure-latent bet then rests on **research value / elegance + a possible long-horizon-abstraction edge**, with a hard bar:

> (i) **parity** with a flat AR speaker on raw-data generation metrics (token NLL, time-rescaling calibration, count/type distribution); and
> (ii) a **win** on long-horizon calibration + **validated** counterfactual accuracy (semi-synthetic / cross-environment).
> Passing (i) but not (ii) refutes its raison d'être → switch to the hybrid.

### 4.2 Reconciliation (Claude — independent read)

fable5's refutation is sound and is the most important output of the consult, but its claim that the hybrid is *clean* is **partly overstated**:

- **Counterfactual confounding doesn't vanish in the hybrid; it relocates to the render boundary.** The AR speaker is trained on *observational* (plan → sequence) pairs and its autoregressive prior `p(next|history)` bakes in the behaviour policy. Conditioned on an *off-distribution counterfactual plan* `T_a(c)`, it can "snap back" toward observationally-typical continuations that contradict the plan (the exact failure fable5 named for flat-AR-alone, only mitigated by plan-conditioning). Making the speaker faithful under off-distribution plans is itself nontrivial (plan-dropout / counterfactual-plan augmentation / a faithfulness loss). A **jointly-trained pure-latent renderer can be optimised end-to-end for counterfactual-plan faithfulness** — precisely the P4 payoff. So the pure-latent bet retains a **genuine, measurable** potential edge on *counterfactual-render faithfulness*, narrowing the refutation rather than leaving only "elegance."
- **The premise is cheaply decidable, not a matter of priors.** The R2.1 horizon-decay curve tests whether the abstraction edge exists at all; a counterfactual-render-faithfulness probe tests whether the hybrid's boundary leaks. Measure, don't assume.

### 4.3 Decision

- **Exclusion clarified (Chris, 2026-07-05):** the constraint excludes only **reuse of the existing FlatASCEND** speaker. A **fresh** AR renderer conditioned on the JEPA plan is admissible. Therefore the **hybrid arm and the three-arm benchmark are in-bounds.**
- **Stance: benchmark, don't bet.** Do not commit to pure-latent rendering on priors. The **latent plan + plan-level operator is the common spine** of every arm and is where JEPA earns its keep (P4). The **renderer is the open variable** — pure-latent vs fresh AR speaker — to be settled by the three-arm benchmark against the §4.1 hard bar, augmented with the counterfactual-render-faithfulness probe from §4.2.

## 4a. Gate outcome (Pi, 2026-07-05): GO-WITH-CHANGES — incorporated

Full verdict in `coordination/jepa_pi_thread.md`. Direction endorsed; 8 changes are required before rung 0 and are now binding on this plan:

1. **Rung −1 substrate/eval-readiness gate** — a per-source manifest (patients/sequences/windows, token & wall-clock quantiles, candidate-action frequency, block yield per horizon, split counts); the indexer **fails closed** if a source under-contributes. (See `rung0_1_run_specs.md`.)
2. **MIMIC windowing — decided (Chris, 2026-07-05):** source-specific shorter MIMIC wall-clock windows for representation/generation; **MIMIC↔SCI-D demoted to a *supporting* transportability diagnostic**; the **semi-synthetic oracle is the *primary*** counterfactual yardstick.
3. **Mask source shortcuts** — `DATASET:*` and source-only anchors are unavailable to the encoder/predictor/operator (eval-only, for stratification); source-matched/within-source distractors + a source-prediction probe from latents.
4. **Real `is_outcome` leakage audit now** — exclude `is_outcome==1` and endpoint-proximal positions from context/target/eval, with unit tests + a manifest (replaces the stub).
5. **Latent-space:** fresh minimal v0B JEPA is the primary first latent; frozen 350M states are a comparator/teacher arm only.
6. **Pre-register numeric gates** before looking at results — concrete effect-size/CI for "substantially slower" (rung 0), "adequate decode" (rung 1), non-flat sensitivity, `d_t/v_t` failure, overlap thresholds, abstention horizons.
7. **Pre-register the three arms + metrics now** — flat AR / pure-latent / **hybrid (starts immediately as a design/benchmark arm, not a fallback)**; the AR-renderer/hybrid interface + faithfulness probes designed from the start (heavy training stays gated).
8. **Specify the semi-synthetic oracle before arm training** — known action effects, a confounded behaviour policy, overlap failures, no-effect controls, source-like rate/length distributions.

**Correction to §4.2 (Pi):** counterfactual-render faithfulness is **not** unique to pure-latent — the hybrid can be given it via plan-faithfulness losses + re-encode/contrast probes. So the pure-latent bet must beat that **strengthened** hybrid; the §4.1 bar is raised accordingly.

**Language / scope discipline (Pi):** no "treatment effect" / "causal accuracy" — use "overlap-gated associational operator," "transportability diagnostic," "semi-synthetic known-effect accuracy." **Abstention coverage is a headline metric.** Per-source vigintile effects are **rank-scale, not raw-scale**. Candidate normalisation matches source + wall-clock horizon + length + event rate.

## 5. Architecture direction (common spine + open variable)

- **Latent plan (spine):** distributional predictor `p(z|context)`; VQ/discrete or sequence-of-latents targets for decodability; two-phase EMA-freeze; stochastic predictor + k-step variance regulariser for rollout stability; **hierarchy iff the horizon-decay gate passes.**
- **Time (spine):** **wall-clock horizons** (retire event-count for generation/counterfactual work); "latent state at query `t`" as the native mechanism; **marked-TPP read-out** anchored to a scheduled absolute clock (retire ZILN for the "state at `t`" capability); count-consistency at any coarse/fine interface.
- **Operator (spine):** `T_a` at the plan level; overlap-gated + sensitivity-bounded + abstention region; structural constraint (additive/factored/invertible) for well-behaved multi-step composition; cross-environment invariance as the primary genuine-vs-propensity diagnostic.
- **Renderer (open variable):** pure-latent read-out **vs** fresh AR speaker conditioned on the plan — decided empirically (three-arm benchmark).
- **Validation (spine):** decoder-free summary-statistic heads + latent-corruption sensitivity curve as the trusted falsification channel; obs-space forward-prediction "scissors" as a standing alarm; semi-synthetic oracle for counterfactual claims.

## 6. Experiment ladder (cheapest-first, each rung gates the next)

| Rung | Test | Decides | Gate to proceed |
|---|---|---|---|
| −1 | **Substrate/eval-readiness manifest** (per-source counts, token & wall-clock quantiles, block yield per horizon, split counts; indexer **fails closed** if a source under-contributes) + **leakage guards live** (`DATASET:*` masked to eval-only; real `is_outcome` audit + unit tests) | Is the joint substrate valid & leakage-safe to run on? | Both sources yield adequate matched windows and audits pass; else fix before rung 0 |
| 0 | **Coarse-vs-fine horizon-decay curve vs wall-clock** (reuses existing retrieval eval at two granularities) | Does a timescale separation / abstraction edge exist? | Coarse decays substantially slower per unit time ⇒ hierarchy worth building; else single-scale |
| 1 | **Frozen-decode ceiling** `D(z⁺)` — exact order/count/timing recon | Is the latent decodable at all (P1)? Upper-bounds any generator | `D(z⁺)` recon adequate ⇒ latent is generation-capable; else change targets (VQ / seq-of-latents) |
| 2 | **No-training rollout `d_t` / `v_t` sweep** | Drift vs attractor-collapse vs EMA-nonstationarity (P2) | Signatures identify which stabiliser is needed before training dynamics |
| 3 | **Falsifier ladder** — latent-corruption sensitivity curve + `D(z⁺)`-vs-`D(ẑ)` split + decoder-free summary heads | Is the validation channel trustworthy (R2.2/P5)? | Sensitivity curve non-flat ⇒ falsifier can see degradation; else add summary heads before trusting any generation metric |
| 4 | **Transportability diagnostic** (MIMIC↔SCI-D, *supporting* not primary; source-matched windows + overlap-decay) | Does the operator transport across sources (supporting evidence only) | Not near-free — requires common actions, matched wall-clock horizons, overlap, and source-balanced windows first; the **semi-synthetic oracle (rung 5) is the primary** counterfactual yardstick |
| 5 | **Three-arm benchmark** — flat AR / pure-latent / hybrid — on (a) raw-generation metrics and (b) validated counterfactual accuracy (semi-synthetic + cross-env), plus a **counterfactual-render-faithfulness probe** on the hybrid boundary | Which renderer; is the pure-latent bet justified (§4.1 bar) | Pure-latent must clear parity-plus-counterfactual-win; else adopt the hybrid |

Rung −1 gates all others (fail-closed readiness + live leakage guards). Rungs 0–3 are largely training-free; **rung 0 gates hierarchy only**, not the whole programme. Rung 4 is a *supporting* transportability diagnostic (not near-free — see the row). The **semi-synthetic oracle spec is pre-registered before rung 5** (the only new-training investment), and the **three arms + numeric gates are pre-registered now** (§4a), with the hybrid an immediate design arm.

## 7. Governance / safety boundary

Method-only note; safe_distilled. Any rung on governed data runs under the existing boundary: patient-level split, prefix-safe context, aggregate-only outputs, no raw tokens / source-IDs / embeddings / checkpoints in reports; v0C raw/MEDS-lite and T2 outcome-proximal labels remain gated. The semi-synthetic oracle (rung 5b / P4 validation) must be built as a governed-safe simulator with declared generative assumptions; do not describe pseudo-rendered observed blocks as generated sequences, nor action-conditioned latent rollout as treatment-effect estimation.

## 8. Open questions for adversarial review (Pi)

1. Is **counterfactual-render faithfulness** (§4.2) the genuine crux that keeps the pure-latent bet alive, or is it also expressible as a plan-faithfulness loss on the hybrid's AR speaker (collapsing the distinction)?
2. Is the **horizon-decay pre-test** a sufficient gate for hierarchy, or can a timescale separation exist that per-unit-time retrieval decay fails to reveal?
3. Is a **semi-synthetic oracle** realistic enough to adjudicate counterfactual accuracy, given the realism gap, and what is the minimal oracle that still discriminates the arms?
4. Residual circularity after the falsifier ladder — is decoder-free summary-statistic agreement enough to trust the "scissors" alarm, or is the encoder-level blind spot (shared by every learned channel) large enough to require the oracle as the *primary*, not backup, anchor?
5. Does adding the hybrid arm dilute the programme (three models to build) beyond what the "price the exclusion" value justifies — i.e., should rung 5 start as pure-latent-vs-flat-AR and add the hybrid only if pure-latent clears parity?

## 9. Provenance

- Verbatim two-round consult: `ascend-flat:coordination/fable5_jepa_thread.md` (titan).
- fable5 was a pure-abstract ML methods consult (no domain specifics; it did not see the v0 empirical results, so its suggestions are cross-checked here against the v0 evidence chain and the governance boundary).
- Supersedes nothing; extends blueprint Part C and the next-experiment brief with a decision on the generation/counterfactual architecture question and a gated experiment ladder.
