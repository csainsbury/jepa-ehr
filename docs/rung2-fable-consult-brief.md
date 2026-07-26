---
title: Fable consult brief — instance-specific prediction of time-windowed segments in irregular symbol streams
created: 2026-07-26
status: ABSTRACTED problem statement for external consult; domain-stripped, no substrate detail, no records
abstraction: deliberately domain-free — the consult is answerable from sequence modelling / point-process /
  estimation theory alone, and nothing here identifies the data, the field, or the application
---

# Consult brief — predicting time-windowed segments of irregular symbol streams

## The abstract setup

We have a corpus of **sequences of discrete symbols** drawn from a fixed vocabulary of ~1,050 types. Each
symbol carries a **real-valued timestamp**, non-decreasing within a sequence. Inter-symbol intervals are
**highly irregular** and the corpus contains two sub-populations with very different temporal density:

- **Population A** — short, bounded streams. ~32 symbols span ≈1 unit of time.
- **Population B** — long streams. ~32 symbols span ≈210 units of time (median), i.e. ~200× sparser per symbol.

A **frozen encoder** maps any set of symbols to a vector by mean-pooling their learned embeddings. A small
predictor maps a context vector to a predicted target vector. Both are trained with a
predict-the-future-representation objective (no reconstruction, no token-level likelihood).

**Task.** From a context prefix, predict a summary vector of a *future segment* of the same sequence. Two
segment definitions matter:

- **COUNT-defined**: the next *K* symbols.
- **TIME-defined**: all symbols falling in `[t_query, t_query + H)` for a fixed horizon `H`.

## The evaluation criterion (and our doubts about it)

For a predicted vector `p`, its own target `y`, and the corpus of other instances' targets `{y_j}`:

```
ratio = mean cos_dist(p, y) / mean_j min_{j != i} cos_dist(y_i, y_j)
```

i.e. **prediction error normalised by the ambient nearest-neighbour distance among targets**. `ratio < 1`
means the prediction is closer to its own target than the nearest *other* instance's target — a strict
instance-specificity criterion.

We have empirically found **four failure modes** of this criterion and would like to know whether they are
known, and whether a better criterion exists:

1. **Shared-component degeneracy.** If the target representation contains a large component common to all
   instances (we hit this with a deterministic positional code concatenated to every target), the ambient
   distance collapses (0.65 → 0.13) and the ratio becomes unreadable — it reported a spuriously excellent
   0.775 on two unrelated datasets to three decimals, which is what exposed it.
2. **Cross-space incomparability.** We initially assumed the ratio is comparable across different target
   representations because it is self-normalising. That is false in the presence of (1).
3. **Sample-size sensitivity of any fitted "ceiling"** (below).
4. **Eligibility confounding** (below) — the largest effect we have measured.

## The three constructions we used, and the two that failed

To ask "is the residual error a *predictor* limitation or a *representation* limitation", we tried to bound
what any map from context to target could achieve.

- **1-NN regression** (borrow the nearest other context's target). **Failed as a bound** — the trained
  predictor beat it in every arm, because a single-neighbour estimate is high-variance while a fitted
  predictor smooths.
- **In-sample fitted ridge / random-Fourier ridge.** **Failed** — held out, the apparent bound evaporated
  (in-sample 0.879–0.913 vs held-out 1.127–1.198 for the same family).
- **Train-fitted ridge/RFF, scored on held-out data, with more fitting rows than the predictor itself saw.**
  This one **passes its own checks**: it bounds the trained predictor, is not interpolating
  (features/rows ≈ 0.05), and the target geometry is non-degenerate. It puts the trained predictor
  0.016–0.029 from the bound.

**Question 1.** Is train-fitted-held-out regression the right way to bound "achievable by any map on a fixed
representation"? Is there a sharper standard construction — a conditional-entropy / mutual-information
estimate, or a nonparametric regression-function bound — that would be less sensitive to fitting sample size
and to the choice of function class?

## The eligibility confounder — our most consequential measurement artefact

Because a TIME-defined or far-offset target needs the sequence to *continue* far enough, every experiment
carries an eligibility rule ("the instance must have ≥ *m* future symbols / ≥ *h* future time"). That rule is
a **selection on how much sequence remains**, and it dominates results. Holding target definition, predictor
and metric fixed and changing only the rule:

| eligibility | n | ratio |
|---|---|---|
| ≥ 32 future symbols | 1,722 | **1.134** (fails) |
| ≥ 256 future symbols | 3,775 | **0.824** (clears) |

The ambient normaliser *fell* (0.212 → 0.165), so the denominator became harder and the ratio still improved.
In a fine sweep, the offset at which population B crosses the criterion moves from **~33** symbols under one
rule to **~110–118** under the other — a 3–4× difference in "usable horizon" from eligibility alone.

**Question 2.** This looks like informative right-censoring: instances with more remaining sequence are both
*more eligible* and *more predictable*. What is the correct way to report predictability as a function of
horizon when eligibility and horizon are entangled? Is there an inverse-probability-of-selection or
competing-risks treatment that would decouple them, and is the "usable horizon" even identifiable without
modelling the sequence-length process jointly?

## The substantive result, and where it stalls

- **COUNT-defined targets are predictable.** Best ratios ≈0.61–0.67. There is a **granularity floor**: 4-symbol
  windows score 2.15/1.72 (worse than chance-adjacent), 8 symbols still fail, **16 symbols is where the
  criterion is crossed**, and the gain **saturates** by 64–128 symbols (population B actually reverses,
  0.675 → 0.609 → 0.634).
- **TIME-defined targets fail completely** — 0 of 12 (offset × horizon) cells clear, for either population
  (best 1.284 and 1.064). This is *consistent* with the floor: a 30-unit window for population B contains a
  **median of 4 symbols**, and even a 365-unit window only 40.
- **A decomposition rescues it.** Rather than predicting the time-window directly, predict the next *K*
  symbols (which works) and then **select the sub-prefix falling inside `H`**. Coverage is not the obstacle:
  32 symbols contain population B's entire 30-unit window in **99.6 %** of instances. Given the *true* symbol
  block, the time-window target is recoverable (0.385 / 0.944) versus 0.907 / 2.077 from context alone.
- **The selection is the binding constraint.** Perfect content + perfect selection reproduces the target
  exactly (0.000). Substituting a learned selector costs everything observed.
- **Soft selection strongly beats hard selection.** Predicting a per-symbol probability of being inside the
  window and taking a probability-weighted mean gives 0.717, versus 0.970 for committing to a boundary index —
  our single largest improvement. Mechanically clear: with ~4 symbols in the window, a boundary off by one is
  a 25 % content error.
- **Mean pooling cannot express a temporal prefix.** Feeding a selector the mean-pooled block gives 1.433
  (fails); feeding it per-symbol identity + relative time gives 0.944 (clears) on *identical information*. The
  timing-preserving arm is **flat in K** (it selects); the pooled arm **degrades in K** (it dilutes).

## The blocker we want the consult to address

Every arm above grants the **true future symbol block** and only learns the selection. But selecting a
temporal prefix requires **per-element** predictions. A predictor that emits a single **pooled latent** for a
future segment cannot be cut at an element boundary — there is nothing to select over. So the validated route
implicitly requires an **element-level generator** plus a timing model, which is a different architecture from
the pooled-latent predictor we have. An earlier result in this project already rejected per-instance exact
count/order/timing fidelity for the pooled latent.

**Question 3 (the main one).** Given:
- pooled-latent prediction of a count-defined segment works well;
- the quantity actually wanted is a *time-windowed* segment;
- selecting a temporal prefix needs per-element structure;

what is the **minimal** representation change? Candidate directions we can see, and we would value both a
ranking and anything we have missed:

- (a) predict a **sequence of per-element latents** (order-preserving) instead of one pooled latent, and select;
- (b) keep the pooled latent but add a **separate timing head** predicting the arrival-time distribution, and
  select in expectation — essentially what we prototyped, which caps at ~0.72;
- (c) model the stream as a **marked temporal point process** and integrate the intensity over `[t, t+H)`,
  making the time-window target native rather than something recovered by selection;
- (d) predict a **set/multiset-valued** summary of the window directly (a distribution over symbol counts)
  rather than a mean-pooled vector, i.e. abandon instance-specific latent identity as the objective.

**Question 4.** Our soft-selection gain (probability per element, weighted mean) resembles an expectation
under a point-process posterior. Is (c) simply the principled version of what we found empirically, and would
it be expected to dominate (b)? Is there a known result on when a pooled-summary objective is *sufficient* for
window-restricted queries and when it is provably not?

**Question 5.** Compounding. In recursive multi-step rollout we measure an **exposure gap** — free-running
minus teacher-forced same-instance error — growing 0.049 → 0.099 over 4 steps for population B, while
teacher-forced error stays flat. If the route requires rolling forward until `H` is covered, how should the
number of steps be budgeted against that growth, and does soft selection help or hurt under compounding?

## What we are *not* asking

We are not asking for help with the criterion's threshold, the corpus, or any domain interpretation. The
consult is scoped to: the soundness of the bound construction (Q1), the eligibility/censoring entanglement
(Q2), the minimal representation change for window-restricted prediction (Q3/Q4), and rollout budgeting (Q5).

## Honest statement of our own reliability

Seven distinct methodological errors arose in producing the results above, and **every one was caught by a
control rather than by foresight**: a bound the model beat, an in-sample bound, a false cross-representation
comparability assumption, a data-starved bound, the eligibility confounder, a pooled oracle arm that produced
a false negative, and a grid choice that made one population's selection task trivial. Two of our headline
conclusions were stated and then retracted. Please treat every number above as provisional and say so if a
construction still looks unsound.
