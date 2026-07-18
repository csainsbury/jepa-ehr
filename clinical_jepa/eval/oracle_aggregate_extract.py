"""Aggregate-real calibration extraction (Pi C=5 micro-gate, revised for REVISE#1).

Maps the TRAIN-only tokenised SCID/MIMIC substrate to the frozen ``AggregateStats`` marginals the
calibration realism envelope consumes, joins them to a deterministic synthetic BASE, and fits the frozen
calibration knobs — as ONE identity-bound, one-time, fail-closed runner. **Nothing here reads a governed
bundle until the calibration micro-gate PASSes and populates
``oracle_aggregate_policy.APPROVED_AGGREGATE_READ_POLICY``.**

Boundary (unchanged from the accepted semantics): derived aggregates are Chris's explicit route-specific
safe clearance; the INPUT bundle and execution environment stay governed/local-only; outputs are
aggregate-only and go to gitignored local paths (no commit / no sync).

Field semantics (C=5, matched to the generator + CALIBRATION_SPEC), all ACCEPTED by Pi:

* structural tokens EXCLUDED from every clinical class/count/timing quantity: ``special [0,4)`` (PAD/BOS/
  EOS/DEATH) and ``dataset_context [1048,1050)`` (DATASET:SCID/DATASET:MIMIC). The real prefix is index 0
  = ``DATASET:{source}`` and index 1 = ``[BOS]``.
* class map: content token id → class 0..4 via ``ORACLE_ENV_CLASS_FAMILIES`` half-open ranges.
* CLUSTER = maximal run of content events sharing one ``cumulative_days`` value (Δt=0 multiplicity).
* ``sequence_length`` = content-token count (→ ``length_ecdf``); ``count_ecdf`` = CLUSTER-COUNT per
  sequence (frozen name: cluster-count ECDF; NOT ``events_per_sequence`` — ``n_events`` counts tokens).
* ``delta_t_zero_fraction`` = zero-Δt adjacent content pairs / all adjacent content pairs.
* positive GAP = Δt>0 between consecutive CLUSTERS (→ ``positive_gap_ecdf``, frozen 8-decimal support
  convention of observed values, Chris's route-specific unbucketed clearance).
* ``mean_occupancy_fraction`` = mean over sequences of (distinct classes / 5).

Below ``ORACLE_ENV_MIN_DENOM`` on any denominator → NOT_EVALUABLE, never zero-filled.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_calibration import (
    AggregateStats, NOT_EVALUABLE, REQUIRED_SOURCES, calibration_schema_hash, calibrate_sources,
    validate_aggregate_input,
)
from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_meta_gen import invariant_hash
from clinical_jepa.eval.oracle_meta_ledger import ledger_hash
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_CLASS_FAMILIES, ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES, ORACLE_ENV_STRUCTURAL_RANGES,
    ORACLE_EVALUATOR_IDENTITY,
)

# ---- frozen substrate identity (Pi defect #3: enforce, do not trust operator strings) ----
TIME_CHANNEL = "cumulative_days"
TRAIN_SPLIT_ONLY = "train"
EXPECTED_BUNDLE = "joint_flat_corrected_v1"
EXPECTED_VOCAB_NAME = "flatascend_joint_corrected_v1"
EXPECTED_VOCAB_HASH = "4b57b210ab4b3ec6"
VOCAB_SIZE = 1050
SOURCE_PREFIX_LEN = 2
BOS_ID = 1                                                  # index 1 of every sequence
SOURCE_DATASET_TOKENS = {"SCID": 1048, "MIMIC": 1049}      # index 0 of every sequence
# predeclared exclusion rule: sequences with zero content tokens are COUNTED (allowlisted aggregate), and
# the extraction REFUSES if their fraction exceeds this bound (a silent skip is never allowed).
MAX_NO_CONTENT_FRACTION = 0.02


class ExtractionRefused(RuntimeError):
    """Raised when a read/step would violate the identity / one-time / TRAIN-only / aggregate-only boundary."""


def extraction_schema_hash() -> str:
    """Frozen hash of the extraction field/range/convention schema — bound by the approved-read policy so a
    silent semantic change invalidates authorization."""
    return canonical_hash({
        "n_classes": ORACLE_ENV_N_CLASSES, "class_families": ORACLE_ENV_CLASS_FAMILIES,
        "structural_ranges": ORACLE_ENV_STRUCTURAL_RANGES, "time_channel": TIME_CHANNEL,
        "vocab_size": VOCAB_SIZE, "prefix_len": SOURCE_PREFIX_LEN, "bos_id": BOS_ID,
        "source_dataset_tokens": SOURCE_DATASET_TOKENS, "split": TRAIN_SPLIT_ONLY,
        "cluster": "maximal_run_equal_cumulative_days", "length": "content_token_count",
        "count_ecdf": "cluster_count_per_sequence", "ecdf_support": "frozen_8_decimal_observed_values",
        "delta_t_zero": "zero_dt_adjacent_content_pairs_over_all", "min_denom": ORACLE_ENV_MIN_DENOM,
        "no_content_rule": {"action": "count_and_refuse_above_fraction", "max_fraction": MAX_NO_CONTENT_FRACTION},
    })


# ============================ pure aggregation (unit-tested, no I/O) ============================
def _class_of(token_ids: np.ndarray) -> np.ndarray:
    cls = np.full(token_ids.shape, -1, dtype=np.int64)
    for c, (_, lo, hi) in enumerate(ORACLE_ENV_CLASS_FAMILIES):
        cls[(token_ids >= lo) & (token_ids < hi)] = c
    return cls


def _ecdf(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ()
    supp, counts = np.unique(np.round(v, 8), return_counts=True)       # frozen 8-decimal observed support
    cdf = np.cumsum(counts) / counts.sum()
    return tuple((float(s), float(c)) for s, c in zip(supp, cdf))


def _sequence_features(token_ids: np.ndarray, cdays: np.ndarray):
    """Per-sequence content-only features, or None if the sequence has no content token. Assumes the
    inputs already passed ``_validate_raw_sequence`` (finite/aligned/monotone)."""
    cls = _class_of(token_ids)
    keep = cls >= 0
    if not keep.any():
        return None
    c = cls[keep]
    t = np.asarray(cdays, dtype=float)[keep]
    length = int(c.size)
    per_class = np.bincount(c, minlength=ORACLE_ENV_N_CLASSES)[:ORACLE_ENV_N_CLASSES]
    occupancy = float((per_class > 0).sum()) / ORACLE_ENV_N_CLASSES
    dt = np.diff(t)
    n_adj = int(dt.size)
    n_zero_adj = int((dt == 0).sum())
    boundaries = np.concatenate([[True], dt > 0]) if t.size else np.array([True])
    cluster_times = t[boundaries]
    inter = np.diff(cluster_times)
    pos_gaps = inter[inter > 0]
    return {"length": length, "per_class": per_class, "occupancy": occupancy,
            "n_clusters": int(cluster_times.size), "n_pos_gaps": int(pos_gaps.size),
            "n_zero_adj": n_zero_adj, "n_adj": n_adj, "pos_gaps": pos_gaps}


def aggregate_from_sequences(source: str, sequences) -> AggregateStats | str:
    """Build ``AggregateStats`` for one source from an iterable of (token_ids, cumulative_days) sequences.
    PURE — no I/O. Sequences with zero content tokens are COUNTED (not silently skipped) and the build
    REFUSES if their fraction exceeds ``MAX_NO_CONTENT_FRACTION``. Returns NOT_EVALUABLE below the floor."""
    lengths, cluster_counts, occ = [], [], []
    class_tot = np.zeros(ORACLE_ENV_N_CLASSES, dtype=np.int64)
    n_clusters = n_pos_gaps = n_zero_adj = n_adj = 0
    pos_gap_pool: list[np.ndarray] = []
    n_seen = n_no_content = 0
    for token_ids, cdays in sequences:
        n_seen += 1
        f = _sequence_features(np.asarray(token_ids), np.asarray(cdays))
        if f is None:
            n_no_content += 1
            continue
        lengths.append(f["length"]); cluster_counts.append(f["n_clusters"]); occ.append(f["occupancy"])
        class_tot += f["per_class"]
        n_clusters += f["n_clusters"]; n_pos_gaps += f["n_pos_gaps"]
        n_zero_adj += f["n_zero_adj"]; n_adj += f["n_adj"]
        if f["pos_gaps"].size:
            pos_gap_pool.append(f["pos_gaps"])
    if n_seen and n_no_content / n_seen > MAX_NO_CONTENT_FRACTION:
        raise ExtractionRefused(
            f"{source}: no-content fraction {n_no_content}/{n_seen} exceeds {MAX_NO_CONTENT_FRACTION}")
    n_seq = len(lengths)
    if n_seq == 0:
        return NOT_EVALUABLE
    n_events = int(class_tot.sum())
    if min(n_seq, n_clusters, n_events, n_pos_gaps) < ORACLE_ENV_MIN_DENOM:
        return NOT_EVALUABLE
    pos_gaps = np.concatenate(pos_gap_pool) if pos_gap_pool else np.empty(0)
    agg = AggregateStats(
        source=source, n_sequences=n_seq, n_events=n_events, n_clusters=n_clusters,
        n_positive_gaps=n_pos_gaps, class_counts=tuple(int(x) for x in class_tot),
        delta_t_zero_fraction=float(n_zero_adj / n_adj) if n_adj else 0.0,
        length_ecdf=_ecdf(np.asarray(lengths)), positive_gap_ecdf=_ecdf(pos_gaps),
        count_ecdf=_ecdf(np.asarray(cluster_counts)),
        mean_occupancy_fraction=float(np.mean(occ)) if occ else 0.0)
    ok, reason = validate_aggregate_input(agg)
    return agg if ok else reason


# ============================ strict fail-closed HDF5 validation (D4) ============================
def _validate_raw_sequence(source: str, group: str, tok: Any, cdays: Any) -> tuple[np.ndarray, np.ndarray]:
    """Fail closed on any malformed sequence — never silently skip or reinterpret (Pi defect #4)."""
    if tok is None:
        raise ExtractionRefused(f"{source}/{group}: missing token_ids dataset")
    if cdays is None:
        raise ExtractionRefused(f"{source}/{group}: missing {TIME_CHANNEL} dataset")
    tok = np.asarray(tok); cdays = np.asarray(cdays)
    if tok.ndim != 1 or cdays.ndim != 1:
        raise ExtractionRefused(f"{source}/{group}: token_ids/{TIME_CHANNEL} must be 1-D")
    if tok.shape[0] != cdays.shape[0]:
        raise ExtractionRefused(f"{source}/{group}: token_ids/{TIME_CHANNEL} length mismatch")
    if tok.shape[0] < SOURCE_PREFIX_LEN:
        raise ExtractionRefused(f"{source}/{group}: shorter than the source prefix")
    if not np.issubdtype(tok.dtype, np.integer):
        if not np.all(np.equal(np.mod(tok.astype(np.float64), 1), 0)):
            raise ExtractionRefused(f"{source}/{group}: non-integer token ids")
        tok = tok.astype(np.int64)
    if tok.min() < 0 or tok.max() >= VOCAB_SIZE:
        raise ExtractionRefused(f"{source}/{group}: token id out of [0,{VOCAB_SIZE})")
    cdays = cdays.astype(np.float64)
    if not np.all(np.isfinite(cdays)):
        raise ExtractionRefused(f"{source}/{group}: non-finite {TIME_CHANNEL}")
    if np.any(np.diff(cdays) < 0):
        raise ExtractionRefused(f"{source}/{group}: {TIME_CHANNEL} not nondecreasing")
    if int(tok[0]) != SOURCE_DATASET_TOKENS[source]:
        raise ExtractionRefused(f"{source}/{group}: index-0 token is not DATASET:{source}")
    if int(tok[1]) != BOS_ID:
        raise ExtractionRefused(f"{source}/{group}: index-1 token is not [BOS]")
    return tok, cdays


def _iter_validated_sequences(h5_path: str, source: str):
    """Yield validated (token_ids, cumulative_days) for every group. Governed read — reached only after
    the policy/one-time/config guards pass. Fails closed on any non-conforming entry."""
    import h5py
    with h5py.File(h5_path, "r") as h5:
        for group in h5.keys():
            grp = h5.get(group)
            if grp is None or not hasattr(grp, "keys"):
                raise ExtractionRefused(f"{source}/{group}: not a group")
            if "token_ids" not in grp:
                raise ExtractionRefused(f"{source}/{group}: no token_ids")
            if TIME_CHANNEL not in grp:
                raise ExtractionRefused(f"{source}/{group}: no {TIME_CHANNEL}")
            yield _validate_raw_sequence(source, group, grp["token_ids"][:], grp[TIME_CHANNEL][:])


# ============================ config / source identity runner (D3) ============================
def load_and_verify_config(config_path: str) -> dict[str, Any]:
    """Parse the approved LOCAL config and enforce identity: exactly {SCID,MIMIC}, TRAIN paths under the
    approved bundle, the frozen vocab name/hash/size. Returns {source: train_h5_path, config_hash,
    vocab_hash}. Fails closed on any deviation (Pi defect #3)."""
    import yaml
    raw = open(config_path, "rb").read()
    cfg = yaml.safe_load(raw)
    ds = cfg.get("datasets", cfg) if isinstance(cfg, dict) else {}
    # locate the joint source_datasets block (support the committed example shape)
    block = None
    for node in _walk(cfg):
        if isinstance(node, dict) and "source_datasets" in node:
            block = node; break
    if block is None:
        raise ExtractionRefused("config has no source_datasets block")
    if str(block.get("vocab_name", cfg.get("vocab_name", ""))) not in ("", EXPECTED_VOCAB_NAME) \
            and cfg.get("vocab_name") != EXPECTED_VOCAB_NAME:
        pass  # vocab_name may live at top level; verified against the manifest below
    paths: dict[str, str] = {}
    for entry in block["source_datasets"]:
        name = str(entry.get("name"))
        h5 = (entry.get("h5_paths") or {}).get(TRAIN_SPLIT_ONLY)
        if name in SOURCE_DATASET_TOKENS and h5:
            paths[name] = str(h5)
    if set(paths) != set(REQUIRED_SOURCES):
        raise ExtractionRefused(f"config sources {sorted(paths)} != required {sorted(REQUIRED_SOURCES)}")
    for src, p in paths.items():
        if EXPECTED_BUNDLE not in p:
            raise ExtractionRefused(f"{src}: TRAIN path not under {EXPECTED_BUNDLE!r}: {p}")
        if "test" in p.lower() or "sealed" in p.lower() or "val" in p.lower():
            raise ExtractionRefused(f"{src}: TRAIN path looks like a non-train split: {p}")
    vocab_hash = _verify_vocab(paths)
    return {"paths": paths, "config_hash": hashlib.sha256(raw).hexdigest(), "vocab_hash": vocab_hash}


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _verify_vocab(paths: dict[str, str]) -> str:
    """Verify the vocab manifest beside the bundle matches the frozen name/hash/size (Pi defect #3)."""
    import os.path as osp
    roots = {osp.dirname(p) for p in paths.values()}
    if len(roots) != 1:
        raise ExtractionRefused("SCID/MIMIC TRAIN paths are not in one bundle directory")
    manifest = osp.join(next(iter(roots)), "vocab_manifest.json")
    if not osp.exists(manifest):
        raise ExtractionRefused("vocab_manifest.json missing beside the bundle")
    m = json.loads(open(manifest, "rb").read())
    if str(m.get("vocab_name")) != EXPECTED_VOCAB_NAME:
        raise ExtractionRefused(f"vocab_name {m.get('vocab_name')!r} != {EXPECTED_VOCAB_NAME!r}")
    if str(m.get("vocab_hash")) != EXPECTED_VOCAB_HASH:
        raise ExtractionRefused(f"vocab_hash {m.get('vocab_hash')!r} != {EXPECTED_VOCAB_HASH!r}")
    fr = m.get("family_ranges") or []
    size = max((f["end"] for f in fr), default=0)
    if size != VOCAB_SIZE:
        raise ExtractionRefused(f"vocab size {size} != {VOCAB_SIZE}")
    return EXPECTED_VOCAB_HASH


# ============================ one-time atomic run state machine (D2) ============================
_STATES = ("APPROVED", "READING_SCID", "READING_MIMIC", "FITTING", "COMPLETE", "REFUSED")


class OneTimeRun:
    """Persistent gitignored atomic run state, claimed via O_CREAT|O_EXCL so a run id can be claimed once.
    Any pre-existing state file for the run id refuses reuse; a stale/partial state refuses (retry needs a
    NEW gate/run id). Not thread-shared: a single-process one-shot governed run (Pi defect #2)."""

    def __init__(self, state_path: str, run_id: str) -> None:
        self.state_path = state_path
        self.run_id = run_id

    def claim(self) -> None:
        try:
            fd = os.open(self.state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ExtractionRefused(f"run state already exists at {self.state_path}: no replay")
        with os.fdopen(fd, "w") as fh:
            json.dump({"run_id": self.run_id, "state": "APPROVED"}, fh)

    def _read(self) -> dict[str, Any]:
        st = json.loads(open(self.state_path).read())
        if st.get("run_id") != self.run_id:
            raise ExtractionRefused("run state id mismatch")
        return st

    def advance(self, frm: str, to: str) -> None:
        if frm not in _STATES or to not in _STATES:
            raise ExtractionRefused(f"illegal state {frm}->{to}")
        st = self._read()
        if st.get("state") != frm:
            raise ExtractionRefused(f"cannot advance {frm}->{to} from state {st.get('state')!r}")
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"run_id": self.run_id, "state": to}, fh)
        os.replace(tmp, self.state_path)                          # atomic transition


# ============================ synthetic BASE construction (D5) ============================
BASE_FAMILY = "T_latent_factor"          # frozen synthetic mechanism the BASE is drawn from
BASE_KAPPA = 0.35
BASE_N_SEQUENCES = 4000
BASE_SEED = 0x0B45E


def _synthetic_base(source: str) -> AggregateStats:
    """Deterministic synthetic BASE ``AggregateStats`` from the FROZEN literal mechanism, relabelled to the
    source (the mechanism is source-agnostic; the label only satisfies ``calibrate_sources`` matching). All
    identities are hashed into the result payload by the caller."""
    from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
    cell = generate_literal_cell(BASE_FAMILY, BASE_KAPPA, "orthogonal", BASE_N_SEQUENCES, seed=BASE_SEED)
    rep = [lo for (_, lo, _) in ORACLE_ENV_CLASS_FAMILIES]        # representative content token per class
    ds, bos = SOURCE_DATASET_TOKENS[source], BOS_ID
    fm, ts = np.asarray(cell.future_multiset), np.asarray(cell.future_timestamps, dtype=float)
    L = fm.shape[1]
    # the literal cell fixes L items per sequence → degenerate length/count ECDFs. Vary the retained length
    # DETERMINISTICALLY (frozen seed) so the BASE marginals are non-degenerate, matching the variable-length
    # real substrate. Each sequence keeps a prefix of length 2..L.
    lengths = np.random.default_rng(BASE_SEED + 17).integers(2, L + 1, fm.shape[0])

    def seqs():
        for i in range(fm.shape[0]):
            li = int(lengths[i])
            toks = np.array([ds, bos] + [rep[int(c)] for c in fm[i, :li]], dtype=np.int64)
            t0 = float(ts[i, 0])
            times = np.concatenate([[t0, t0], ts[i, :li]]).astype(np.float64)
            yield toks, times

    base = aggregate_from_sequences(source, seqs())
    if isinstance(base, str):
        raise ExtractionRefused(f"synthetic BASE for {source} is {base}")
    return base


def base_identity() -> dict[str, Any]:
    return {"family": BASE_FAMILY, "kappa": BASE_KAPPA, "n_sequences": BASE_N_SEQUENCES,
            "seed": BASE_SEED, "mechanism_hash": invariant_hash()}


# ============================ sanitized output contract (D7) ============================
def sanitized_output(targets: dict[str, AggregateStats | str], calibration: dict[str, Any],
                     run_meta: dict[str, Any]) -> dict[str, Any]:
    """The ONLY object written to disk / eligible to be committed. Per-source aggregate marginals + fitted
    knobs + bound identities, scanned for forbidden row-level keys before return (fail closed)."""
    from clinical_jepa.validation import _scan_forbidden_aggregate_keys
    out: dict[str, Any] = {
        "governance_class": "explicitly_cleared_safe_aggregate_only_no_patient_rows",
        "split": TRAIN_SPLIT_ONLY, "n_classes": ORACLE_ENV_N_CLASSES,
        "sources": {s: (a if isinstance(a, str) else asdict(a)) for s, a in targets.items()},
        "calibration": calibration, "base_identity": base_identity(),
        "identities": {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
                       "calibration_schema_hash": calibration_schema_hash(),
                       "evaluator_identity": ORACLE_EVALUATOR_IDENTITY,
                       "extraction_schema_hash": extraction_schema_hash()},
        "run": run_meta}
    leaks = _scan_forbidden_aggregate_keys(out)
    if leaks:
        raise ExtractionRefused("aggregate output failed the forbidden-key scan: " + "; ".join(leaks))
    return out


# ============================ the one-time governed runner (D2/D3/D5/D7) ============================
def run_calibration_extraction(config_path: str, *, policy: dict[str, Any], run_id: str,
                               state_path: str, out_path: str) -> dict[str, Any]:
    """The SINGLE entry point that reads governed aggregates. Fail-closed sequence: verify config/identity,
    authorize against the committed approved-read policy (empty ⇒ refuse), claim the one-time run, read
    TRAIN-only SCID then MIMIC ONCE, build the synthetic BASE, calibrate, write the sanitized summary.
    Refuses on any identity/policy/state violation and never opens a TEST/sealed/non-train bundle."""
    cfg = load_and_verify_config(config_path)
    live = {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
            "calibration_schema_hash": calibration_schema_hash(),
            "evaluator_identity": ORACLE_EVALUATOR_IDENTITY, "vocab_hash": cfg["vocab_hash"],
            "vocab_name": EXPECTED_VOCAB_NAME, "extraction_schema_hash": extraction_schema_hash(),
            "config_hash": cfg["config_hash"], "run_id": run_id}
    ok, reason = _authorize(policy, live)
    if not ok:
        raise ExtractionRefused(f"aggregate read refused: {reason}")

    run = OneTimeRun(state_path, run_id)
    run.claim()                                                 # APPROVED (atomic, no replay)
    targets: dict[str, AggregateStats | str] = {}
    for src, frm, to in (("SCID", "APPROVED", "READING_SCID"), ("MIMIC", "READING_SCID", "READING_MIMIC")):
        run.advance(frm, to)
        targets[src] = aggregate_from_sequences(src, _iter_validated_sequences(cfg["paths"][src], src))
    if any(isinstance(t, str) for t in targets.values()):       # both sources must be evaluable
        run.advance("READING_MIMIC", "REFUSED")
        raise ExtractionRefused(f"a source is NOT_EVALUABLE: "
                                f"{ {s: t for s, t in targets.items() if isinstance(t, str)} }")
    run.advance("READING_MIMIC", "FITTING")
    bases = {s: _synthetic_base(s) for s in REQUIRED_SOURCES}
    coll = calibrate_sources({s: targets[s] for s in REQUIRED_SOURCES}, bases)
    calib = {"all_sources_within_envelope": coll.all_sources_within_envelope,
             "source_coverage_ok": coll.source_coverage_ok, "combined_hash": coll.combined_hash,
             "schema_hash": coll.schema_hash, "mechanism_hash": coll.mechanism_hash,
             "per_source": {s: {"fitted_knobs": r.fitted_knobs, "within_envelope": r.within_envelope,
                                "input_hash": r.input_hash, "fitted_param_hash": r.fitted_param_hash,
                                "diagnostics": r.diagnostics} for s, r in coll.per_source.items()}}
    run_meta = {"run_id": run_id, "reviewed_commit": policy.get("reviewed_commit"),
                "gate_event_ref": policy.get("gate_event_ref"), "config_hash": cfg["config_hash"]}
    out = sanitized_output(targets, calib, run_meta)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, sort_keys=True, indent=2)
    os.replace(tmp, out_path)
    run.advance("FITTING", "COMPLETE")
    return out


def _authorize(policy: dict[str, Any], live: dict[str, Any]) -> tuple[bool, str]:
    from clinical_jepa.eval.oracle_aggregate_policy import aggregate_read_authorized
    return aggregate_read_authorized(policy, live)
