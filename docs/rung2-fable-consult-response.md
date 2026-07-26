---
title: Fable consult RESPONSE — window-restricted prediction in irregular symbol streams
created: 2026-07-26
status: verbatim external consult response; recorded as received, unedited
brief: docs/rung2-fable-consult-brief.md (domain-stripped)
agent: fable a963917e28bc0c4a2
scope_note: |
  ANSWERED THE BRIEF AS POSED — i.e. conditioned on a MEAN-POOLED window target. Chris subsequently
  reframed the goal to "predict n next events in the right sequence with time intervals", which changes
  the target and therefore INVERTS the (d')-over-(a) ranking below. See the reframe note appended at the
  end for which findings survive the reframe and which do not.
---

# Consult response — window-restricted prediction in irregular symbol streams

Consultant: external, sequence modelling / temporal point processes / estimation theory. Scope as briefed: Q1–Q5 plus adversarial review of the criterion. Brief read in full; no other repository material consulted. All claims below are labelled **[standard]** (adopt), **[judgement]** (my call, needs testing), or **[test]** (cheap experiment).

## The one structural fact that reorganises the whole brief

Your encoder is mean pooling of fixed per-type embeddings. Therefore the window target is

```
y = E^T h / N
```

where `h` is the **count vector by symbol type** in the window, `N = |h|`, and `E` is the frozen embedding matrix. **The pooled window target is a deterministic function of the window's type histogram and nothing else.** Order, within-window timing, and element identity beyond type are annihilated by your own target construction. **[standard — it is algebra, not a result]**

Consequences, developed below: (i) per-element structure is *not* required by the target, only by your block-then-select decomposition; (ii) the "selection is the binding constraint" conclusion is an artifact of that decomposition; (iii) the minimal representation change is a horizon-conditioned prediction of the window's expected type composition — the first moment of a marked TPP — not any of (a)–(c) in full.

## Q1 — The bound construction

**Your construction is sound for a claim you are not making, and unsound for the claim you are making.** A regressor fitted on train and scored held-out gives an *achievability certificate*: an **upper bound on the Bayes risk** ("at least this is achievable on this representation"). It is **not a lower bound on what any map can achieve** — no fitted function class can certify that, because a richer class might do better. So "the trained predictor is 0.016–0.029 from the bound" establishes only that your predictor is near the ridge/RFF family optimum. The inference "the residual is a representation limitation" does not follow from this construction. **[standard]** This directional error is my runner-up candidate for your eighth mistake (the primary candidate is in the criterion section).

What to use instead:

1. **Halved 1-NN target discrepancy** (function-class-free floor estimate). Under smoothness of the regression function, `(1/2n) Σ_i ‖y_i − y_{nn(i)}‖²` — nearest neighbour taken in *context* space, discrepancy measured between *targets* — converges to `E[Var(y|x)]`, the Bayes floor for squared loss. This is the classical difference-based / nearest-neighbour residual-variance estimator (Rice 1984 in 1-D; von Neumann-type; nearest-neighbour multivariate versions, e.g. Devroye et al.). **[standard]** Note the reinterpretation: your 1-NN regression did not "fail as a bound" — it was the wrong statistic. The 1-NN *risk* converges to roughly **twice** the Bayes noise (plus a context-gap bias term), which is exactly why any smoothed predictor beat it. Halve it and it becomes the floor estimate you wanted. Work in squared Euclidean on the unit-normalised vectors, where cos-dist ≈ ½‖Δ‖², so the factor of 2 is exact rather than approximate. The context-gap bias inflates the estimate, so it remains an upper bound on the floor, but a function-class-free one. **[test: one script, no training]**

2. **Capacity-sweep plateau.** Report the held-out bound as a curve over capacity (RFF features ×1, ×2, ×4; kNN over k; a small MLP). If the curve plateaus and the plateau agrees with the halved-1-NN estimate, you have converging evidence for the floor. Agreement of two estimators with different failure modes is the strongest cheap certificate available. **[judgement, cheap]**

3. **Do not use MI / conditional-entropy estimators.** For continuous targets in moderate dimension at n ≈ 10³–10⁴, KSG/MINE-type estimators are high-bias and high-variance, and converting an MI estimate into a risk bound requires Fano-type inequalities that are loose for regression. You would be replacing one fragile bound with a more fragile one. **[judgement, but well-supported]**

4. One more caveat that connects to the criterion: even the true Bayes predictor `E[y|x]` can have **ratio > 1** when conditional entropy is high (small windows). A criterion failure is therefore never, by itself, evidence of predictor *or* representation failure. See the criterion section.

## Q2 — Eligibility / censoring

Name the problem correctly first: this is **outcome-dependent selection under a terminal event**, not censoring in the MAR sense. Eligibility `S = 1{remaining length ≥ m}` is a function of the *future*, and your own measurements show the future content correlates with remaining length given context. IPW with weights `1/P̂(S=1 | context)` is valid only under `S ⫫ Y | context` — which is exactly the assumption your data refutes. **Plain IPCW does not decouple them here, and no reweighting scheme can without untestable assumptions.** **[standard]**

Three distinct estimands, only two identifiable:

1. *Predictability conditional on surviving to horizon h.* Identifiable. This is what your current numbers estimate — legitimate, but must be reported as survival-conditional, and the conditioning set changes with the rule, which is the whole 3–4× "usable horizon" shift. That shift is an **estimand change, not an artifact**.
2. *Unconditional predictability of window content, where termination is content.* Identifiable, and my recommendation. Make sequence termination an explicit event/mark; a window past the end has a well-defined (empty / terminal-marked) target. Every instance is eligible; the confounder is dissolved by construction rather than adjusted away. This is also the TPP-native formulation (termination = absorbing state), which is one of the quiet arguments for the Q3 recommendation.
3. *"Predictability had the sequence continued."* Counterfactual. **Not identifiable** without jointly modelling the length process and assuming independence you cannot check. Do not report this quantity.

Practical reporting: ratio-vs-offset curves **stratified by remaining-length quantile**, with the population mixture stated. The crossing point is well-defined per stratum; a single population "usable horizon" is not a rule-free quantity and should not be reported as one. **[standard]**

One contamination specific to your table: the two arms have n = 1,722 vs 3,775, and the ambient nearest-neighbour denominator **mechanically shrinks as n grows** (min-distance scales ≈ n^(−1/d_int)). Part of the 0.212 → 0.165 drop, and hence part of "the denominator became harder", is pure sample-size effect, not geometry. Recompute both arms with the denominator subsampled to matched n before interpreting that table. **[test: trivial]**

## Q3 — Ranking of representation changes (the main question)

Applying the histogram fact: the target needs the window's **type composition**, horizon-resolved. Nothing more.

**1. (d), reframed — and this is also the "option you missed".** As written, (d) says "abandon instance-specific latent identity". That framing is wrong, and correcting it makes (d) the winner: for a mean-pooled target, the count distribution's first moment *is* the instance-specific latent identity, exactly, via `y = E^T h/N`. The minimal change is a **horizon-conditioned expected-composition head**: predict per-type expected counts in `[t, t+H)` — equivalently the integrated per-type intensity `Λ_k(t, t+H)`, trained with Poisson or multinomial likelihood on *directly observed* window counts — then map through the frozen `E`. No per-element generation, no selection step, no rollout, native time windows, termination handled as a mark (fixing Q2), dense observable supervision instead of latent-space bootstrapping. This is precisely the **first-moment reduction of (c)**, which is all your target can see. **[judgement on whether it beats 0.717 empirically; test below. The sufficiency of the first moment for this target is [standard] algebra.]**

**2. (c) Full marked TPP.** Correct in principle and the principled superset: it is what (d′) is the sufficient reduction of. Adopt the full machinery only if (d′) plateaus or you later need beyond-first-moment queries (calibrated uncertainty over window contents, multi-window joint consistency, sampling). Costs: 1,050-mark intensity estimation under a 200× cross-population density ratio (you will need time rescaling / log-inter-arrival parameterisation to make one model span both populations), and a likelihood objective that sits awkwardly beside your embedding-prediction training. Building (c) *first* would be paying for moments your target provably ignores.

**3. (a) Per-element latents + soft selection.** Validated by your oracle arms and would work, but it is strictly more machinery than the target requires — you would be predicting order and element identity only to immediately mean-pool them away. Rank it as the fallback if (d′)/(c) underperform (which would indicate the factored structure — content composition × timing weights — is easier to learn than the joint, a real possibility at your sample sizes), and as the route if *future* queries need within-window order. If built: non-autoregressive (K learned queries cross-attending to context, emitting per-element type logits + Δt), not autoregressive — your Q5 exposure-gap numbers make autoregressive per-element rollout untenable.

**4. (b) Pooled latent + timing head.** Structurally capped, and your own experiment proves it: selection weights must act **per element**, and a pooled content vector can only be scalar-modulated after pooling has already mixed the composition. The 1.433 vs 0.944 identical-information result is effectively a demonstration that no timing head bolted onto a pooled content latent can recover per-symbol reweighting. The ~0.72 cap is not an optimisation failure; retire (b) to baseline status. **[standard, given your own data]**

**On your decomposition being a dead end:** partially, yes — as a *production route*. "Predict the block, then select" was diagnostically excellent (it proved content-given-timing is recoverable and located the difficulty), but the conclusion "selection is the binding constraint" is a property of the decomposition, not of the problem. The direct composition route has no selection step to be bound by. Keep the decomposition as a diagnostic instrument; do not architect around it.

## Q4 — Is soft selection the point-process expectation, and is there a sufficiency result?

**Yes, with one bias caveat.** Your probability-weighted mean is the plug-in estimate of `E[Σᵢ 1{i∈W} e_i] / E[Σᵢ 1{i∈W}]` under a per-element membership posterior — the ratio-of-expectations approximation to the posterior mean of the pooled window vector, i.e. exactly the first moment of (c). So (c)/(d′) is the principled version of what you found, and it dominates (b) for the structural reason above, not merely statistically. **[standard]** The caveat: at median window count N ≈ 4, `E[A]/E[N] ≠ E[A/N]` appreciably — the ratio-of-expectations bias is second-order in 1/N and your N is 4. Either add the second-order correction or train the head on the *normalised* composition directly. **[test: compare weighted-mean vs expected-normalised targets on your existing oracle arm — no new training]**

**Sufficiency result — yes, and it is exactly the DeepSets boundary. [standard]** Any permutation-invariant query of a set is expressible as `ρ(Σᵢ φ(xᵢ))` for sufficiently rich per-element `φ` and readout `ρ` (Zaheer et al. 2017; width limits in Wagstaff et al. 2019). Window restriction is a symmetric query over (type, time) *pairs*. Therefore: **sum-pooling with time-augmented per-element features and a nonlinear readout is sufficient in principle for window-restricted queries; mean-pooling of time-free embeddings is provably insufficient**, because window membership is not measurable with respect to a statistic from which time has been marginalised out. Your 0.944-clears / 1.433-fails identical-information pair is a clean empirical instance of exactly this theorem boundary. The practical caveat on the positive direction: DeepSets universality can require embedding dimension scaling with set size and pathological `φ`; do not read it as "just widen the pooling" — it justifies (d′)/(a), not a rescue of (b).

## Q5 — Rollout budgeting

First, an audit item: **check the units of the exposure gap.** If 0.049 → 0.099 is in cosine-distance units, divide by your ambient denominators (0.165–0.212): a single free-running step costs ~0.24–0.30 in *ratio* units. Your best soft-selection ratio is 0.717, so the margin to criterion is (1 − 0.717) × 0.165 ≈ **0.047 absolute** — consumed by approximately **one** free-running step, with increments of ~0.012–0.017 per further step. Budget formula: `n* = max n s.t. base_error + gap(n) < ambient`, and with your numbers n* ≈ 1. **The recursive route is not viable for population B at measured gap magnitudes**; this independently reinforces the single-shot (d′) recommendation, which requires zero rollout — the horizon `H` is covered by conditioning, not by stepping. **[test: units audit + arithmetic]**

If rollout is ever unavoidable: (i) reduce steps by using the largest block that clears — your own saturation data says 64 for population B (it reverses at 128); (ii) scheduled sampling / DAgger-style training on self-generated prefixes is the standard mitigation for the gap itself **[standard]**; (iii) **soft selection helps at readout and hurts as state**: averaging over element-level uncertainty reduces terminal variance, but feeding the *expected* (softened) representation back as the next context is feeding in the mean of a multimodal future — an off-manifold input the model was never trained on, which compounds distribution shift. Soft-average only at the final readout; propagate hard samples (or multiple sampled trajectories, averaging at the end) if you must recurse. **[judgement; standard-adjacent — this is the known mean-collapse failure of deterministic latent rollout]**

## Adversarial review of the criterion — and the eighth error

Your four failure modes are real and known in kind. Here are four more; the first is my primary candidate for the eighth error.

**5. Sample-size dependence of the denominator (the eighth error, in my judgement).** `min_j` distance shrinks with corpus size as ≈ n^(−1/d_int). Every cross-arm comparison at different n — including the central eligibility table (1,722 vs 3,775) and any COUNT-vs-TIME comparison where eligibility differs — has a denominator confound *independent of* the selection confound you already found. Some fraction of your "the normaliser fell yet the ratio improved" narrative is mechanical. **[test: subsample all denominators to a common n; recompute every headline ratio. Hours, not days.]**

**6. Ratio-of-means is not per-instance.** Your criterion compares a mean numerator to a mean-of-minima denominator. It can pass while most instances fail (a few isolated targets with large NN distance inflate the denominator) and fail while most instances succeed. The object you actually want is per-instance: **is the own target the nearest target to the prediction?** — i.e. top-1 retrieval accuracy / mean reciprocal rank over the target corpus. This is the standard instance-specificity metric in representation learning, it is scale-free per instance, and it is immune to failure modes 1, 5, and largely 7. **[standard; test: one evaluation script]**

**7. Small-N discreteness and histogram collisions.** A 4-symbol window over 1,050 types with skewed marginals has non-trivial probability of *exact* histogram collision with another instance — identical targets, NN distance ≈ 0. The denominator collapse concentrates on precisely the small-window instances, i.e. the TIME-defined regime. Some of your 0-of-12 TIME failures may be denominator artifacts, not prediction failures. **[test: count duplicate window histograms and plot NN-distance vs window count N]**

**8. It is a point-prediction criterion applied to a distributional problem.** At N ≈ 4 the conditional entropy of the window is large; the Bayes-optimal point prediction `E[y|x]` sits between modes and can have ratio > 1 *even given the true conditional law*. In that regime the criterion measures conditional entropy, not predictor or representation quality, and **no representation change in Q3 can fix it**. For small windows the right object is a proper scoring rule on a *distributional* prediction — energy score in embedding space, or log-loss on the predicted composition (which (d′) gives you for free) — with the ratio retained only as a coarse large-N sanity check.

**Verdict on the criterion:** not fundamentally misconceived, but only valid in its per-instance form (mode 6), at matched denominator n (mode 5), and at window sizes large enough that conditional entropy is not the dominant term (modes 7–8). As currently used — mean-form, unmatched n, small-N TIME windows — it is the wrong object in exactly the regime the project cares most about. Fix the metric before running (d′), or the new architecture will be scored by the old artifact.

## Summary table — standard vs judgement

| Item | Status |
|---|---|
| Fitted-regressor bound is achievability-only; cannot certify representation limits | standard — adopt |
| Halved 1-NN target discrepancy as Bayes-floor estimate | standard — adopt; trivial test |
| IPW invalid here; termination-as-content estimand; stratified reporting; "usable horizon" not rule-free | standard — adopt |
| Pooled window target depends only on type histogram | algebra — adopt |
| Ranking (d′) > (c) > (a) > (b); (b) structurally capped | (b)'s cap and (c)⊃(d′) are standard; the *ordering of (d′) vs (a) empirically* is judgement — test |
| Soft selection = plug-in first-moment TPP posterior; ratio-of-expectations bias at N≈4 | standard; bias test is free on existing oracle arm |
| DeepSets sufficiency boundary (time-augmented sum-pool sufficient; time-free mean-pool insufficient) | standard — adopt |
| Rollout budget n* ≈ 1 for population B; soft at readout, hard in state | arithmetic given your numbers + judgement on state propagation |
| Eighth error: unmatched-n denominators across arms | judgement on which error is "the" eighth; the confound itself is standard — test first, it may move headline numbers |

**Single recommended next experiment:** horizon-conditioned per-type expected-count head (Poisson/multinomial loss on observed window counts, termination as a mark), mapped through the frozen embedding matrix, evaluated with per-instance retrieval rank at matched n — against your 0.717 soft-selection arm re-scored under the same corrected metric. It tests the top-ranked option, the corrected criterion, and the Q2 estimand fix in one run.

---

# REFRAME NOTE (added after the consult returned)

Chris, on reading the consult: *"Don't get hung up on the 30-day element. I'm just interested in predicting
n next events. But they do need to be in the right sequence with time intervals."*

This changes the **target**, and therefore inverts the consult's central ranking. Recorded here so the
response above is not later read as endorsing a route it was never asked about.

## What the reframe does to the ranking

Fable's top-ranked option **(d′) horizon-conditioned expected-composition head** predicts *per-type expected
counts* in a window. By construction it is a **histogram** — it discards order and per-event timing. It was
ranked first *because the briefed target was mean-pooled*, and Fable's own algebra (`y = Eᵗh/N`) shows a
mean-pooled target can see nothing but the histogram. Under the new goal, order and intervals ARE the
deliverable, so (d′) is optimal for a target we no longer want.

Under the reframe the ranking becomes:

1. **(a) per-element latents / decoder head** — order-preserving, per-element type + Δt. Was ranked 3rd only
   because the pooled target would have mean-pooled the order away. It is now the direct expression of the goal.
   Fable's build note applies and is adopted: **non-autoregressive** — K learned queries cross-attending to
   the frozen context, emitting per-position type logits + Δt — *not* autoregressive, because the Q5 budget
   `n* ≈ 1` makes per-element autoregressive rollout untenable at measured exposure-gap magnitudes.
2. **(c) marked TPP** — still the principled superset, and now genuinely needed rather than reducible, since
   the goal asks for beyond-first-moment structure (order, intervals). Reachable as a likelihood upgrade of (1).
3. **(d′)** — no longer on the route. First moment only; annihilates the deliverable.
4. **(b) pooled + timing head** — retired, unchanged.

## What the reframe DISSOLVES (i.e. work now moot)

- The **time-window / wall-clock target line** (0 of 12 cells) — moot; the goal is count-defined.
- The **block-then-select decomposition** and the finding *"selection is the binding constraint"* — moot.
  Selection existed only to convert a count-defined block into a time-defined window. With no window there is
  no cut, so the thing I identified as the principal blocker is not on the path to the stated goal.
- **Most of the criterion problem.** Failure modes 1, 2, 5, 6 and 7 of the ratio criterion are all artifacts of
  the *ambient-nearest-neighbour normaliser* over pooled target vectors. Per-position type **log-loss** and
  interval **CRPS** need no ambient denominator, are per-instance by construction, and are comparable across
  arms. The reframe replaces the fragile metric rather than repairing it.

## What SURVIVES the reframe (all adopted)

- Fitted-regressor bounds are **achievability-only** — cannot certify a representation limit.
- **Halved 1-NN target discrepancy** as the function-class-free Bayes-floor estimate.
- **Eighth error: unmatched-n denominators** — still contaminates the recorded eligibility table
  (n = 1,722 vs 3,775), so that headline stays flagged regardless of route.
- **Termination-as-content**: more load-bearing now, not less. "Predict the next n events" needs n events to
  exist; making sequence-end an explicit mark keeps every instance eligible and predicts termination instead of
  selecting on it. This is the fix for the eligibility confounder under the new goal.
- **DeepSets sufficiency boundary**: time-free mean-pooling is *provably insufficient* for order- and
  time-resolved queries. This upgrades "the pooled v0B latent cannot do this" from an empirical finding to a
  structural one, and is the strongest argument for adding a per-element head.
- **Distributional scoring at small N** — directly satisfied by log-loss on types and a proper score on Δt.

## One line of prior evidence that must NOT be treated as closing the goal

The recorded order-target result — frozen T1/T2/T3 scoring **0.963 / 1.221 / 1.199 vs 0.824 mean-pooled** —
looks like evidence against order-resolved targets. It is not usable as such: those arms were scored with the
ratio criterion **across different target spaces** (256-dim vs 4,224-dim), which is exactly the
cross-space-incomparability and shared-component degeneracy the consult confirms invalidates the comparison,
at unmatched n on top. The line stays open; it was never fairly tested.
