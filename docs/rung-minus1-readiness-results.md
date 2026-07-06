---
title: Clinical-JEPA rung −1 readiness — RESULTS (path A, first governed touch)
created: 2026-07-05
status: rung −1 FULLY CLEARED on real data — substrate half PASS + audit half PASS, both independently verified
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050, vigintile-factored)
gate: rung0_1_run_specs.md §"Rung −1"; guards per cross-cutting guards
reporting: aggregate-only (counts + quantiles; no sequence ids / tokens / real paths)
---

# Rung −1 readiness — results

First governed-data touch (path A: event-index readiness smoke on the real joint
substrate). Read-only, aggregate-only, no training, no integration. Ran the two
committed CLIs against a **local gitignored** config (real h5 paths never
committed): `clinical_jepa.splits.build_index` → `clinical_jepa.splits.readiness_manifest`.

## Verdict — `gate_status = PASS` (exit 0)

Full substrate, all three splits. Both expected sources present in every required
split; nothing under the 500-matched-windows/source/split floor; nothing missing.

| source | split | n_seqs | feasible T0 | valid matched | median tok | **median span (d)** | med-freq |
|--------|-------|-------:|------------:|--------------:|-----------:|--------------------:|---------:|
| SCID | train | 97 490 | 89 950 | 89 947 | 354 | 6 332 | 0.78 |
| SCID | dev | 14 798 | 13 620 | 13 617 | 344 | 6 321 | 0.78 |
| SCID | test | 14 624 | 13 492 | 13 490 | 344 | 6 323 | 0.78 |
| MIMIC | train | 348 717 | 348 717 | 348 717 | 102 | 3.44 | 0.97 |
| MIMIC | dev | 42 966 | 42 966 | 42 964 | 101 | 3.38 | 0.97 |
| MIMIC | test | 44 221 | 44 221 | 44 218 | 103 | 3.47 | 0.97 |

Total sequences 562 816 (train 446 207 / dev 57 764 / test 58 845). Zero
source-token mismatches on real data (every `tok[0]` matched its declared source;
`source_prefix_len=2`, DATASET:SCID=1048 / DATASET:MIMIC=1049 + [BOS]=1).

## Two headline findings

1. **The "MIMIC vanishes under event-count" prior is WRONG.** MIMIC is *more*
   event-index-feasible than SCID (100% vs ~92% primary_T0), because MIMIC uses a
   smaller per-source window (8/min-ctx 4) vs SCID (32/min-ctx 8). The
   source-specific window earns its keep: at a *common* window-32, MIMIC feasibility
   drops to ~86% (block_yield event_window_32 ≈ 300 660 / 348 717) — a naive shared
   window would silently drop ~14% of MIMIC.

2. **~1 870× cross-source wall-clock-span gap** (SCID median 6 321 d ≈ 17 yr vs
   MIMIC median 3.38 d), independently recomputed off the raw index. The event-index
   floor passes, but the two sources live on wholly incomparable wall-clock horizons
   → a naive common-horizon cross-source comparison would be a domain/length
   classifier. This quantitatively vindicates: wall-clock windowing, source-specific
   windows, and demoting MIMIC↔SCI-D to a *supporting transportability diagnostic*
   (semi-synthetic oracle = primary counterfactual yardstick).

## Independent adversarial verification (3 lenses, all CONFIRMED)

- **gate-fidelity** — PASS faithfully implements the fail-closed floor+presence
  gate; absence of any expected source/required split fails (no silent-drop path);
  the 500 floor is applied to the post-eligibility matched count.
- **independent-recompute** — a from-scratch recount off `dev.index.jsonl`
  reproduced feasible_windows (SCID 13 620, MIMIC 42 966) and span medians
  (6 321.0 / 3.3802) *exactly*. Emitter is faithful.
- **governance-leak** — manifest aggregate-only; clean tree; nothing governed
  staged/tracked; `run-workspace/local-governed/` gitignored; template still
  `/PLACEHOLDER`.

## Honest caveats (carry to Pi)

1. **The floor gate is deliberately weak.** With `min_matched_candidates=2` and
   coarse (≤8 length × 7 rate) bins at N=1e4–1e5, the length+rate matching filter is
   near-vacuous (`matched_yield ≈ 1.000`; the rate axis is near-degenerate — MIMIC
   all high-rate, SCID all low-rate), so the gate reduces to ≈ raw feasibility ≥ 500.
   This is *as designed* (Pi: "a minimum sanity floor, not a power guarantee") — PASS
   means "substrate is not pathologically thin," **not** "rung 0 is well-powered."
   Per-action / per-overlap power floors come later.

2. **`gate_status="pass"` is only the substrate half of the readiness manifest.** The
   spec gate is "floor met **+ audits green** ⇒ proceed to rung 0." The
   leakage / `is_outcome` / DATASET-mask fail-hard audit half is a *separate*
   pipeline (`clinical_jepa.audit.run_leakage_audit`), **not** AND-combined in the
   readiness code — so a consumer must run both. Both are now green on real data
   (audit half below). _(Design note: the two CLIs could later be AND-combined behind
   a single rung −1 driver so "PASS" can't be read as complete on the substrate half
   alone.)_

## Audit half — real-data leakage audit PASS (path A1)

Ran `extract_blocks` (real, T0, event_index) → `run_leakage_audit` on the full
substrate. `extract_blocks` emitted **535 252** T0 blocks (train 425 067 / dev
54 533 / test 55 652); `run_leakage_audit` → **`overall_status = pass` (exit 0)**.

- **Source-shortcut mask** — `configured+required`, `source_prefix_len=2 ≥ min 2`
  (fail-hard config satisfied), `blocks_checked=535 252`, prefix-position hits **0**,
  violating blocks **0**. Enforced on every real block, not pass-by-absence.
- **`is_outcome` label separation** — `mode=h5_channel` (re-read the real
  `is_outcome_label` channel), `blocks_checked=535 252`, leaked positions **0**,
  violating blocks **0**. Plus `patient_overlap` / `window_inheritance` /
  `horizon_boundary` / `forbidden_tokens` / `duplicate_windows` all pass.

**The leakage guard demonstrably fires, source-asymmetrically (independently
reproduced off the raw h5):** of feasible T0 windows, **SCID 15.13%** were refused
because `is_outcome`==1 (MACE) fell in the context span (long chronic trajectories →
midpoint context often includes a labelled position), vs **MIMIC 0.0016%** (7
windows; per-admission labels are terminal/sparse). Not a no-op — it removes ~1-in-7
SCID context-contaminated windows. Arithmetic reconciles exactly: emitted 535 252 =
feasible 552 966 − refused 17 714; emitted + skipped (27 564) = 562 816 processed.

## Local artifacts (gitignored, not committed)

- `run-workspace/local-governed/dataset.joint.local.yaml`, `arms.joint.local.yaml`,
  `split-manifest.joint.local.json`
- `run-workspace/local-governed/joint_index/{train,dev,test}.index.jsonl`
- `run-workspace/local-governed/readiness_full/rung-minus1-readiness-manifest.json`
- `run-workspace/local-governed/target_blocks_T0/target-block-manifest.json` (383 MB;
  carries seq ids + real paths — governed-local, same class as the index)
- `run-workspace/local-governed/leakage_audit_T0.json` (aggregate-only)

## Rung −1 status: FULLY CLEARED (both halves, independently verified)

Substrate floor+presence PASS + real-data leakage/mask audit PASS. Not "well-powered"
(the floor is a sanity floor by design), but the substrate is confirmed adequate and
leakage-clean to proceed to rung 0.

## Next (per restart plan)

1. Route the rung −1 packet (this doc) to Pi; freeze the rung-1 KS-D timing tolerance
   (proposed ≤ 0.05).
2. Downstream `-1` (empty-target) integration for the wall-clock rung −1; T1
   wall-clock is future work.
3. Rung 0 (horizon-decay pre-test) at common wall-clock horizons — gated on the
   wall-clock rung −1 (needs the empty-target integration first).
