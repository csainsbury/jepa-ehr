# Oracle aggregate-real calibration — result note (v1 generator)

Durable, sanitized record of the one-time TRAIN aggregate-real calibration run and its result-gate ruling.
Aggregate-only; no governed paths, HDF5, group keys, tokens, timestamps, patient rows, or identifiers.

## Run identity (spent — not re-runnable)

| field | value |
|---|---|
| run id | `aggcalib-microgate-run-1` (SPENT) |
| reviewed implementation commit | `6d565b3` |
| policy-population commit | `3e6e33f` (data-only) |
| test-repair commit | `348cbf6` (test-only) |
| micro-gate PASS event | `evt-20260718T170631Z-6c149959` |
| result-gate event | `evt-20260719T012528Z-8e5f96c9` |
| run state | `COMPLETE` |
| result sha256 | `bd64a29e2d517df347941ed4575574ada64d6667e0e17a92dcb5e17dba02b163` |
| state sha256 | `b3f18d1911aa2c953dbc703c24a343fd7559d43d23314ca6564b640dd03713ae` |
| SCID extracted-content digest | `9a8ffb40ff56abc63deeace9b6c0e3e7d4d18806d2d347e3e6f0a4409e197f1d` |
| MIMIC extracted-content digest | `d3c23a06a2207d30e1c36d05d6209f749942b5e99ee6c9db6aa6027ca0f7943d` |

Provenance is `UNVERIFIED` (no pre-read expected digest existed); the result is a route-local TRAIN
diagnostic only. The local `COMPLETE` state/result artifacts under `state/aggregate-calib/aggcalib-microgate-run-1/`
(gitignored) are preserved and must not be deleted or overwritten.

## Verdict (result gate)

- **Execution integrity: PASS** — ran once, cleanly.
- **Aggregate-realism eligibility: FAIL / NOT ELIGIBLE** — both SCID and MIMIC fail the conjunctive envelope.
- **Authorization consequence: NONE.**

## Aggregate marginals read (TRAIN)

| source | n_sequences | n_events | n_clusters | n_positive_gaps | Δt=0 | occupancy |
|---|---:|---:|---:|---:|---:|---:|
| SCID | 97,490 | 45,996,039 | 16,232,979 | 16,135,489 | 0.6485 | 0.7304 |
| MIMIC | 348,717 | 61,428,993 | 6,983,005 | 6,634,288 | 0.8914 | 0.9511 |

Class counts `[demographic, diagnosis, lab, medication, state]`: SCID `[304314, 310399, 11474407, 33906919, 0]`;
MIMIC `[697434, 828493, 47644761, 5073583, 7184722]`. No-content sequences: 0 for both.

## Envelope checks (canonical generator fit)

| check | SCID | MIMIC |
|---|---|---|
| zero-gap abs error | 0.000880 **pass** | 0.091444 fail |
| class TV | 0.584333 fail | 0.579253 fail |
| length KS | 1.0 fail | 1.0 fail |
| cluster-count KS | 0.985314 fail | 0.893475 fail |
| occupancy abs error | 0.102765 fail | 0.119227 fail |
| positive-gap KS | 0.659183 fail | 0.335995 fail |

Failures are **structural**: fixed synthetic length 8 vs broad real lengths; near-uniform C=5 class prior vs
source-specific lab/medication/state composition; incompatible cluster-count and occupancy; positive-gap
mismatch; MIMIC simultaneity beyond the frozen fit grid.

## Claim boundary (only admissible claim)

> The registered v1 synthetic generator was executable and self-consistent but failed the frozen TRAIN
> aggregate-realism envelope on both SCID and MIMIC; it is not eligible for promotion.

Explicitly NOT claimed: latent-mechanism recovery on real data, transfer, counterfactual validity, causal
validity, or real-world order certification.

## Governance state after this run

- The one-time read policy is RETIRED (`APPROVED_AGGREGATE_READ_POLICY` empty / fail-closed); the spent run
  is recorded in `oracle_aggregate_policy_data.SPENT_RUNS`.
- No further governed read is warranted now (re-running after observing the targets would be post-hoc
  tailoring). Any new real aggregate fields require a new question, schema, run id, and a separate gate.
- Bounded synthetic-only development diagnostics using these (now development-seen) TRAIN targets are
  permitted without reopening HDF5, labelled post-result exploratory. A future confirmatory realism claim
  needs a separately preregistered locked/external aggregate gate.
- TEST remains sealed; `APPROVED_ORACLE_POLICY` remains empty; no sealed certification / oracle-T4 /
  manifest / governed T4 is authorized.
