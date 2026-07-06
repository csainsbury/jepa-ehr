---
title: Wall-clock feasibility grid — RESULTS (Pi R3 Q5)
created: 2026-07-06
status: no usable common cross-source horizon → rung 0 re-scoped to within-source (Pi Q5 branch)
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); split=dev (SCID 14,798 / MIMIC 42,966)
reporting: aggregate-only (per source × horizon counts/rates/quantiles; no seq ids)
---

# Wall-clock feasibility grid — results

Pi R3 Q5 required a wall-clock feasibility grid **before** building the wall-clock rung:
"do not assume common horizons." For each source × candidate horizon W (days), we
carve a wall-clock T0 target (mirroring `extract_blocks._t0_wall_clock_block`:
`context_start=source_prefix_len=2`; W-independent midpoint `context_end`; `t_query =
cumulative_days[context_end]`, gap 0; half-open `[t_query, t_query+W)`) and measure
feasibility + degeneracy on the real dev split.

**"Resolves within-source variation"** = neither mostly-empty NOR mostly-saturated
(target ≠ "the entire remaining sequence"), so horizon-decay is measurable. Adequate(W,
source) = valid-matched ≥ 500 AND empty-rate ≤ 0.5 AND saturated-rate ≤ 0.5.

## Verdict — no usable common cross-source horizon

| W (d) | SCID empty% | SCID satur% | SCID adeq | MIMIC empty% | MIMIC satur% | MIMIC adeq |
|------:|------------:|------------:|:---------:|-------------:|-------------:|:----------:|
| 1 | 38.3 | 0.0 | ✓* | 0.4 | 22.2 | ✓* |
| 3 | 35.7 | 0.0 | ✓ | 0.0 | 63.0 | ✗ |
| 7 | 31.9 | 0.1 | ✓ | 0.0 | 90.0 | ✗ |
| 14 | 26.2 | 0.1 | ✓ | 0.0 | 97.3 | ✗ |
| 30 | 15.0 | 0.2 | ✓ | 0.0 | 99.6 | ✗ |
| 90 | 5.2 | 0.7 | ✓ | 0.0 | 100 | ✗ |
| 365 | 1.1 | 3.9 | ✓ | 0.0 | 100 | ✗ |
| 730 | 0.3 | 12.7 | ✓ | 0.0 | 100 | ✗ |
| 1825 | 0.0 | 49.7 | ✓ | 0.0 | 100 | ✗ |
| 3650 | 0.0 | 97.8 | ✗ | 0.0 | 100 | ✗ |

- **SCID discriminating band ≈ 30–730 d** — target occupancy grows monotonically
  (median 1→2→4→10→39→75→151), empty-rate falls, saturation stays low until ~1825 d.
- **MIMIC discriminating band ≈ 1 d only** — dense short admissions (median span 3.4 d)
  → ≥63% saturated by 3 d, ≥90% by 7 d; occupancy **frozen at median 49** from 7 d on
  (any wider window just grabs the rest of the admission — the horizon stops
  discriminating).
- **Bands do not overlap.** The only technically-common horizon is **W = 1 d**, and it
  is degenerate on both sides (MIMIC 22% saturated; SCID 38% empty with median-1-event
  targets) — marked ✓* above. Not a comparable, well-resolved common horizon.

## Consequence (Pi Q5 branch taken)

Rung 0's cross-source common-horizon hierarchy claim is **not viable**. Rung 0 re-scopes
to **within-source** hierarchy tests (SCID across ~30–730 d; MIMIC across ~1–3 d);
**MIMIC↔SCI-D stays a transportability diagnostic only** — never a common-horizon
hierarchy claim. Recorded in `rung0_1_run_specs.md` (rung 0 re-scope + wall-clock
definition). This is the quantitative form of the ~1870× wall-clock-span gap found at
rung −1: the two sources occupy disjoint temporal regimes.

## Local artifacts (gitignored)
- `run-workspace/local-governed/wallclock_feasibility_grid.py`
- `run-workspace/local-governed/wallclock_feasibility_grid_dev.json`

## R3 update (Pi ruled 2026-07-06) — frozen within-source horizons

Pi **confirmed the within-source re-scope** (cross-source common-horizon hierarchy
NOT promoted; MIMIC↔SCI-D = transportability diagnostic only) and required tighter
horizon discipline. Folded in + re-run on **train + dev** (test held out — horizons
are frozen before any test look):

- **Sub-day horizons added** (`cumulative_days` is sub-day resolved: 98.6% of MIMIC
  seqs fractional, median min inter-event gap 0.67 h): grid now spans 0.25 d … 3650 d.
- **Strengthened "resolves-variation" rule** (Pi): adequate = matched ≥ 500 AND
  empty ≤ 0.5 AND saturated ≤ 0.5 **AND non-empty median occupancy ≥ 2** (no
  median-1-only targets as sole evidence).
- **Train/dev stable** (numbers near-identical), so horizons are **PRE-FROZEN**:
  - **SCID hierarchy horizons: {30, 90, 365, 730} d** (band 30–730 d; empty ≤15%,
    non-empty median occ 4→78, saturation ≤13%).
  - **MIMIC hierarchy horizons: {0.25, 0.5, 1} d** primary; **2 d = saturation-
    boundary sensitivity only** (empty ≤0.4%, saturation 6→22% at 0.25–1 d, 45% at
    2 d; non-empty median occ 13→23).

**Transparency note for Pi:** with sub-day horizons a *technical* common-adequate
band [0.25–2 d] appears on both splits — but it is driven by SCID *barely* clearing
the loose thresholds (38% empty, non-empty median exactly 2, near both degeneracy
boundaries), which is precisely the median-near-1 / high-empty regime the strengthened
rule warns against. Per Pi's ruling (cross-source rejected on acute-vs-chronic process
grounds; "the next gate is not 'common horizon found'") we do **not** promote a
cross-source claim; the sub-day common band is recorded, not used. Flag if you want it
revisited.

**Next gate (Pi):** not "common horizon found" — "chosen within-source horizons are
stable [✓ train/dev], non-degenerate [✓ strengthened rule], leakage-clean, and
empty-handled consistently [encode-empty, R4]." Local artifacts now include
`wallclock_feasibility_grid_{train,dev}.json`.
