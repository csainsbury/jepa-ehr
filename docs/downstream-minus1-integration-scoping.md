---
title: Downstream −1 (empty-target) integration — SCOPING
created: 2026-07-06
status: scoping (not yet implemented) — gates the within-source wall-clock rung 0
depends_on: wall-clock target blocks (extract_blocks, merged 1ef918e); Pi R3 Q7 attack surface
---

# Downstream `-1` empty-target integration — scoping

## Problem
Wall-clock T0 blocks encode a zero-event target as the sentinel
`target_start_ref = target_end_ref = EMPTY_TARGET_REF = -1` with `empty_target: true`,
`n_target_events: 0` (Pi item 3: encode, don't drop). The event-index downstream
consumers were written before wall-clock mode and read target refs with
`t0 = max(0, int(target_start_ref))` → `max(0, -1) = 0`. So an **empty future is
silently misread as "the start of the sequence"** (the DATASET/BOS tokens), or
triggers a false boundary violation. This must be fixed before the within-source
wall-clock rung 0 (Pi R3 Q7: "empty wall-clock targets: encoded or explicitly
counted as incomplete, never silently dropped").

## Decision (Chris, 2026-07-06): Option A (encode-empty) — build it now
Chris chose **A** ("if A is the right long-term design we should do this now") over the
interim B. A dedicated design blueprint for the encode-empty *mechanism* (collapse-safe,
decodable, comparator-honest in the MeanToken JEPA) is being produced and will be routed
to Pi as a blueprint gate before implementation. The B write-up below is retained as the
fallback/contrast only.

---

## (Superseded) The interim decision — B vs A
How should an empty wall-clock target be treated in the first within-source rung?

- **(B) skip-with-count [RECOMMENDED, interim]** — exclude empty-target blocks from
  training/eval but **count + report the empty-target rate per source × horizon**,
  and treat the wall-clock rung as conditional/incomplete. This is exactly what Pi
  pre-authorized (item 3: "if empty targets cannot yet be encoded, report their rate
  and treat the wall-clock rung as conditional/incomplete"). Minimal, unblocks rung 0.
- **(A) encode-empty [deferred, full]** — represent "nothing happens in W days" as a
  genuine target (special empty latent / no-event token) so the predictor learns
  silence. The right long-term design; a real modeling addition — defer past rung 0.

Recommendation: **B now, A later.** Empty rates from the feasibility grid are moderate
in the discriminating bands (SCID ~15% at 30 d falling to ~0% by 365 d; MIMIC ~0% at
1–3 d), so skip-with-count loses little and is honestly reported.

## Misread surface (file:line, severity, fix under option B)

| # | Site | Current `-1` behaviour | Sev | Fix |
|---|------|------------------------|-----|-----|
| 1 | `audit/run_leakage_audit.py:264` | `context_end_ref >= target_start_ref` → `10 >= -1` **True** → false `horizon_boundary` violation → **audit FAILS on any empty-target run** | **BUG/HIGH** | skip empty-target blocks in the boundary/duplicate loop; count them separately |
| 2 | `arms/v0b/train_minimal_jepa.py:114,130-135` | filter keeps `-1` (`is not None`); `t0=max(0,-1)=0` → target = `arr[0:max_target]` = **sequence start** → corrupts training | HIGH | route through helper; skip empty + count |
| 3 | `arms/v0e/train_transformer_autoreg.py:21,35-37` | same as v0b | HIGH | same |
| 4 | `arms/v0a/extract_flatascend_embeddings.py:149-151,194` | target span `arr[0:0]` empty/bogus embedding; `target_len = -1-(-1)+1 = 1` → pollutes retrieval bins + rollout eval | MED | helper; empty → skip+count; `target_len=0` for empty |
| 5 | `eval/export_mean_token_rollouts.py:198-200` & `eval/export_transformer_autoreg_rollouts.py:174-176` | `t0=max(0,-1)=0` → misread | MED | helper; skip empty + count |
| 6 | `arms/v0a/train_predictor.py:79-80` | `t0=0, t1=-1` → `arr[0:0]` empty (near-harmless) | LOW | helper; skip empty + count |
| 7 | `tte/scan_feasibility.py:70` | `target_len = max(0, -1-(-1)+1) = 1` for empty | LOW | return 0 when `empty_target` / refs `== -1` |
| 8 | `arms/v0d/train_query_baseline.py:89-93` | **already safe** (`if end < start: return []`) but does not count | LOW | add empty count for reporting only |
| — | `eval/retrieval.py` | **indirect** — bins on `target_len` from the embedding cache (site 4) | — | fixed upstream by site 4 |
| — | `splits/readiness_manifest.py` | event-index `t0_feasible`; does **not** read `-1` refs | — | separate: add wall-clock readiness (below) |

## Design — shared helper (single source of truth)
Add to `clinical_jepa/targets/extract_blocks.py` (or a new `block_spans.py`, imported by all consumers):

```python
def is_empty_target(block) -> bool:
    return bool(block.get("empty_target")) or int(block.get("target_start_ref", 0)) == EMPTY_TARGET_REF

def read_target_span(block, arr) -> tuple[np.ndarray, bool]:
    """Return (token_ids_slice, is_empty). Empty => (empty array, True); never
    misread -1 as position 0."""
```

Every consumer: `if is_empty_target(b): empty_skipped += 1; continue` (option B), and
each manifest/report grows an `empty_target_skipped` count + rate (Pi item 3). No
consumer computes a span from `-1` again.

## Wall-clock readiness (separate, small)
`readiness_manifest.py` uses event-index `t0_feasible`; the within-source wall-clock
rung needs a wall-clock readiness variant that reports per source × horizon:
feasible / non-empty / saturated / valid-matched + **empty-target rate** — reuse the
feasibility-grid logic already written (`wallclock_feasibility_grid.py`). Fold that into
the readiness path (or a sibling) so the composite rung −1 driver can gate the
wall-clock run too.

## Test plan
- `is_empty_target` / `read_target_span` unit tests (empty vs populated).
- Per consumer: a synthetic wall-clock block manifest with a mix of populated +
  empty targets → assert (a) no empty target read as `arr[0:]`, (b) empty blocks
  counted not silently dropped, (c) **`run_leakage_audit` no longer emits a false
  `horizon_boundary` violation** on empty targets (regression for the bug).
- Wall-clock leakage audit end-to-end on a synthetic empty-heavy manifest → PASS with
  reported empty rate.

## Sequencing
1. Shared helper + `run_leakage_audit` boundary bug fix (site 1) — unblocks correctness.
2. Arms/eval consumers (sites 2–6) skip-with-count + report.
3. scan_feasibility / v0d count (sites 7–8).
4. Wall-clock readiness variant + composite-driver hook.
5. Then: within-source wall-clock rung 0.

Each step independently verified before merge (project discipline); route the empty-rate
reporting design back to Pi (it bears on his Q6 refusal/coverage-denominator ask too).
