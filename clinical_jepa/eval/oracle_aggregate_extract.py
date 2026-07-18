"""Aggregate-real calibration extraction (Pi C=5 micro-gate, REVISE#2).

One identity-bound, one-time, fail-closed runner: TRAIN-only SCID/MIMIC tokenised aggregates → declared
synthetic BASE → frozen calibration fit → post-fit regeneration check → sanitized summary. **Nothing reads
a governed bundle until the micro-gate PASSes and populates
``oracle_aggregate_policy.APPROVED_AGGREGATE_READ_POLICY``.** The governed entry point takes NO caller
policy and NO caller output paths — it loads the committed policy internally and derives canonical paths
from the approved run id under one fixed gitignored root.

Accepted semantics (Pi): C=5 ranges + structural exclusion; content-token length vs cluster-count ECDF;
finite/aligned/nondecreasing ``cumulative_days``; frozen 8-decimal support; route-specific safe-output
clearance with inputs staying governed/local. Prefix: index 0 = ``DATASET:{source}``, index 1 = ``[BOS]``.
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
    realism_envelope, validate_aggregate_input,
)
from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_meta_gen import KAPPA_TRAIN_GRID, TRAIN_FAMILIES, invariant_hash
from clinical_jepa.eval.oracle_meta_ledger import ledger_hash
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_CLASS_FAMILIES, ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES, ORACLE_ENV_STRUCTURAL_RANGES,
    ORACLE_EVALUATOR_IDENTITY, ORACLE_POWER_KAPPA_MID,
)

# ---- frozen substrate identity (enforced, never trusted from the operator) ----
TIME_CHANNEL = "cumulative_days"
TRAIN_SPLIT_ONLY = "train"
EXPECTED_BUNDLE = "joint_flat_corrected_v1"
EXPECTED_VOCAB_NAME = "flatascend_joint_corrected_v1"
EXPECTED_VOCAB_HASH = "4b57b210ab4b3ec6"
VOCAB_SIZE = 1050
SOURCE_PREFIX_LEN = 2
BOS_ID = 1                                                  # index 1 of every sequence
SOURCE_DATASET_TOKENS = {"SCID": 1048, "MIMIC": 1049}      # index 0 of every sequence
EXPECTED_TRAIN_FILENAMES = {"SCID": "scid_train.h5", "MIMIC": "mimic_train.h5"}
# the EXACT vocab family partition the manifest must declare (name, start, end) — not merely max-end 1050.
EXPECTED_FAMILY_RANGES = (
    ("special", 0, 4), ("demographic", 4, 51), ("diagnosis", 51, 91), ("lab", 91, 951),
    ("medication", 951, 1032), ("state", 1032, 1048), ("dataset_context", 1048, 1050),
)
MAX_NO_CONTENT_FRACTION = 0.02             # predeclared, schema-bound; counts are EMITTED, never silent
FIXED_STATE_ROOT = os.path.join("state", "aggregate-calib")   # gitignored; canonical run root


class ExtractionRefused(RuntimeError):
    """Raised when a step would violate the identity / one-time / TRAIN-only / aggregate-only boundary.
    Messages carry only ordinals + aggregate-safe reason codes — never a raw group key or governed path."""


# ============================ frozen identities ============================
def extraction_schema_hash() -> str:
    return canonical_hash({
        "n_classes": ORACLE_ENV_N_CLASSES, "class_families": ORACLE_ENV_CLASS_FAMILIES,
        "structural_ranges": ORACLE_ENV_STRUCTURAL_RANGES, "time_channel": TIME_CHANNEL,
        "vocab_size": VOCAB_SIZE, "prefix_len": SOURCE_PREFIX_LEN, "bos_id": BOS_ID,
        "source_dataset_tokens": SOURCE_DATASET_TOKENS, "split": TRAIN_SPLIT_ONLY,
        "family_ranges": EXPECTED_FAMILY_RANGES, "train_filenames": EXPECTED_TRAIN_FILENAMES,
        "cluster": "maximal_run_equal_cumulative_days", "length": "content_token_count",
        "count_ecdf": "cluster_count_per_sequence", "ecdf_support": "frozen_8_decimal_observed_values",
        "no_content_rule": {"action": "count_emit_and_refuse_above_fraction",
                            "max_fraction": MAX_NO_CONTENT_FRACTION}})


def extraction_code_identity() -> str:
    """Hash of the EXECUTABLE logic (this module + calibration + spec). The policy binds this, so any later
    code change invalidates authorization — while a policy-only edit (the ``oracle_aggregate_policy`` data
    module, excluded here) does NOT, avoiding a circular commit hash (Pi REVISE#2 #1)."""
    import clinical_jepa.eval.oracle_calibration as _cal
    import clinical_jepa.eval.oracle_spec as _spec
    h = hashlib.sha256()
    for mod in (__file__, _cal.__file__, _spec.__file__):
        with open(mod, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


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
    cls = _class_of(token_ids)
    keep = cls >= 0
    if not keep.any():
        return None
    c = cls[keep]
    t = np.asarray(cdays, dtype=float)[keep]
    per_class = np.bincount(c, minlength=ORACLE_ENV_N_CLASSES)[:ORACLE_ENV_N_CLASSES]
    dt = np.diff(t)
    boundaries = np.concatenate([[True], dt > 0]) if t.size else np.array([True])
    cluster_times = t[boundaries]
    inter = np.diff(cluster_times)
    pos_gaps = inter[inter > 0]
    return {"length": int(c.size), "per_class": per_class,
            "occupancy": float((per_class > 0).sum()) / ORACLE_ENV_N_CLASSES,
            "n_clusters": int(cluster_times.size), "n_pos_gaps": int(pos_gaps.size),
            "n_zero_adj": int((dt == 0).sum()), "n_adj": int(dt.size), "pos_gaps": pos_gaps}


def aggregate_from_sequences(source: str, sequences) -> tuple[AggregateStats | str, dict[str, int]]:
    """Build ``AggregateStats`` for one source plus an ALLOWLISTED audit-count dict
    ({n_input_sequences, n_no_content_sequences}). No-content sequences are counted and emitted (never
    silently skipped); the build REFUSES above ``MAX_NO_CONTENT_FRACTION``. Returns (agg|NOT_EVALUABLE,
    counts)."""
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
    counts = {"n_input_sequences": n_seen, "n_no_content_sequences": n_no_content}
    if n_seen and n_no_content / n_seen > MAX_NO_CONTENT_FRACTION:
        raise ExtractionRefused(f"{source}: no-content fraction exceeds {MAX_NO_CONTENT_FRACTION}")
    n_seq = len(lengths)
    if n_seq == 0:
        return NOT_EVALUABLE, counts
    n_events = int(class_tot.sum())
    if min(n_seq, n_clusters, n_events, n_pos_gaps) < ORACLE_ENV_MIN_DENOM:
        return NOT_EVALUABLE, counts
    pos_gaps = np.concatenate(pos_gap_pool) if pos_gap_pool else np.empty(0)
    agg = AggregateStats(
        source=source, n_sequences=n_seq, n_events=n_events, n_clusters=n_clusters,
        n_positive_gaps=n_pos_gaps, class_counts=tuple(int(x) for x in class_tot),
        delta_t_zero_fraction=float(n_zero_adj / n_adj) if n_adj else 0.0,
        length_ecdf=_ecdf(np.asarray(lengths)), positive_gap_ecdf=_ecdf(pos_gaps),
        count_ecdf=_ecdf(np.asarray(cluster_counts)),
        mean_occupancy_fraction=float(np.mean(occ)) if occ else 0.0)
    ok, reason = validate_aggregate_input(agg)
    return (agg if ok else reason), counts


# ============================ strict fail-closed HDF5 validation (R4/R3) ============================
def _validate_raw_sequence(source: str, ordinal: int, tok: Any, cdays: Any):
    """Fail closed on any malformed sequence. Error strings carry the ORDINAL, never the raw group key
    (which may be a patient/sequence id, Pi REVISE#2 #4)."""
    tag = f"{source}#seq{ordinal}"
    if tok is None or cdays is None:
        raise ExtractionRefused(f"{tag}: missing token_ids/{TIME_CHANNEL}")
    tok = np.asarray(tok); cdays = np.asarray(cdays)
    if tok.ndim != 1 or cdays.ndim != 1:
        raise ExtractionRefused(f"{tag}: token_ids/time not 1-D")
    if tok.shape[0] != cdays.shape[0]:
        raise ExtractionRefused(f"{tag}: length mismatch")
    if tok.shape[0] < SOURCE_PREFIX_LEN:
        raise ExtractionRefused(f"{tag}: shorter than prefix")
    if not np.issubdtype(tok.dtype, np.integer):
        if not np.all(np.equal(np.mod(tok.astype(np.float64), 1), 0)):
            raise ExtractionRefused(f"{tag}: non-integer token ids")
        tok = tok.astype(np.int64)
    if tok.min() < 0 or tok.max() >= VOCAB_SIZE:
        raise ExtractionRefused(f"{tag}: token id out of range")
    cdays = cdays.astype(np.float64)
    if not np.all(np.isfinite(cdays)):
        raise ExtractionRefused(f"{tag}: non-finite time")
    if np.any(np.diff(cdays) < 0):
        raise ExtractionRefused(f"{tag}: time not nondecreasing")
    if int(tok[0]) != SOURCE_DATASET_TOKENS[source]:
        raise ExtractionRefused(f"{tag}: index-0 not DATASET:{source} (source/path swap?)")
    if int(tok[1]) != BOS_ID:
        raise ExtractionRefused(f"{tag}: index-1 not [BOS]")
    return tok, cdays


def _iter_validated_sequences(h5_path: str, source: str):
    """Yield validated (token_ids, cumulative_days). Rejects soft/external/virtual links so a nominal TRAIN
    file cannot open another (Pi REVISE#2 #3). Wraps h5py errors into sanitized refusals (#4)."""
    import h5py
    try:
        h5 = h5py.File(h5_path, "r")
    except Exception as e:                                  # sanitize: no governed path in the message
        raise ExtractionRefused(f"{source}: cannot open TRAIN bundle ({type(e).__name__})")
    with h5:
        for ordinal, group in enumerate(h5.keys()):
            link = h5.get(group, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ExtractionRefused(f"{source}#seq{ordinal}: non-hard link rejected")
            grp = h5.get(group)
            if grp is None or not isinstance(grp, h5py.Group):
                raise ExtractionRefused(f"{source}#seq{ordinal}: not a group")
            for ds in ("token_ids", TIME_CHANNEL):
                if ds not in grp:
                    raise ExtractionRefused(f"{source}#seq{ordinal}: missing {ds}")
                if not isinstance(grp.get(ds, getlink=True), h5py.HardLink):
                    raise ExtractionRefused(f"{source}#seq{ordinal}: {ds} non-hard link rejected")
                if getattr(grp[ds], "is_virtual", False):
                    raise ExtractionRefused(f"{source}#seq{ordinal}: {ds} virtual dataset rejected")
            try:
                tok, cd = grp["token_ids"][:], grp[TIME_CHANNEL][:]
            except Exception as e:
                raise ExtractionRefused(f"{source}#seq{ordinal}: unreadable dataset ({type(e).__name__})")
            yield _validate_raw_sequence(source, ordinal, tok, cd)


# ============================ exact config / source identity (R3) ============================
def load_and_verify_config(config_path: str) -> dict[str, Any]:
    """Parse the approved LOCAL config and enforce EXACT identity (Pi REVISE#2 #3): exactly one TRAIN entry
    each for SCID and MIMIC, exact TRAIN filenames under the approved bundle, canonical non-symlink paths,
    and the exact manifest family partition + vocab name/hash/size. Fails closed on any deviation."""
    import yaml
    try:
        raw = open(config_path, "rb").read()
        cfg = yaml.safe_load(raw)
    except Exception as e:
        raise ExtractionRefused(f"config unreadable ({type(e).__name__})")
    block = next((n for n in _walk(cfg) if isinstance(n, dict) and "source_datasets" in n), None)
    if block is None:
        raise ExtractionRefused("config has no source_datasets block")
    seen: dict[str, str] = {}
    for entry in block["source_datasets"]:
        name = str(entry.get("name"))
        if name not in SOURCE_DATASET_TOKENS:
            continue
        if name in seen:
            raise ExtractionRefused(f"duplicate source entry: {name}")
        h5 = (entry.get("h5_paths") or {}).get(TRAIN_SPLIT_ONLY)
        if not h5:
            raise ExtractionRefused(f"{name}: no TRAIN h5 path")
        seen[name] = str(h5)
    if set(seen) != set(REQUIRED_SOURCES):
        raise ExtractionRefused(f"config sources != required set {sorted(REQUIRED_SOURCES)}")
    tc = str(block.get("time_channel", cfg.get("time_channel", "")))
    if tc != TIME_CHANNEL:
        raise ExtractionRefused(f"config time_channel {tc!r} != {TIME_CHANNEL!r}")
    real_paths: dict[str, str] = {}
    for src, p in seen.items():
        rp = os.path.realpath(p)
        if os.path.islink(p):
            raise ExtractionRefused(f"{src}: TRAIN path is a symlink")
        if os.path.basename(rp) != EXPECTED_TRAIN_FILENAMES[src]:
            raise ExtractionRefused(f"{src}: TRAIN filename != {EXPECTED_TRAIN_FILENAMES[src]}")
        if EXPECTED_BUNDLE not in rp.split(os.sep):
            raise ExtractionRefused(f"{src}: not under bundle {EXPECTED_BUNDLE}")
        real_paths[src] = rp
    vocab_hash = _verify_manifest(real_paths)
    return {"paths": real_paths, "config_hash": hashlib.sha256(raw).hexdigest(), "vocab_hash": vocab_hash}


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _verify_manifest(real_paths: dict[str, str]) -> str:
    roots = {os.path.dirname(p) for p in real_paths.values()}
    if len(roots) != 1:
        raise ExtractionRefused("SCID/MIMIC TRAIN paths not in one bundle directory")
    manifest = os.path.join(next(iter(roots)), "vocab_manifest.json")
    if os.path.islink(manifest) or not os.path.exists(manifest):
        raise ExtractionRefused("vocab_manifest.json missing or symlink")
    try:
        m = json.loads(open(manifest, "rb").read())
    except Exception as e:
        raise ExtractionRefused(f"vocab manifest unreadable ({type(e).__name__})")
    if str(m.get("vocab_name")) != EXPECTED_VOCAB_NAME or str(m.get("vocab_hash")) != EXPECTED_VOCAB_HASH:
        raise ExtractionRefused("vocab name/hash mismatch")
    fr = tuple((str(f["family"]), int(f["start"]), int(f["end"])) for f in (m.get("family_ranges") or []))
    if fr != EXPECTED_FAMILY_RANGES:                          # EXACT partition, not just max-end==1050
        raise ExtractionRefused("vocab family_ranges != frozen C=5 partition")
    return EXPECTED_VOCAB_HASH


# ============================ one-time atomic run state (R2) ============================
_STATES = ("APPROVED", "READING_SCID", "READING_MIMIC", "FITTING", "REGEN_CHECK", "COMPLETE", "REFUSED")


def canonical_run_paths(run_id: str) -> tuple[str, str]:
    """Canonical state + output paths DERIVED from the run id under the fixed gitignored root — never
    caller-supplied (Pi REVISE#2 #2). run_id is constrained to a safe slug."""
    if not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ExtractionRefused("run_id must be a nonempty [A-Za-z0-9-_] slug")
    d = os.path.join(FIXED_STATE_ROOT, run_id)
    return os.path.join(d, "state.json"), os.path.join(d, "result.json")


class OneTimeRun:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.state_path, self.out_path = canonical_run_paths(run_id)

    def claim(self) -> None:
        d = os.path.dirname(self.state_path)
        os.makedirs(d, exist_ok=True)
        for p in (self.state_path, self.out_path):
            if os.path.islink(p) or os.path.exists(p):
                raise ExtractionRefused("run artifact already exists: no replay")
        try:
            fd = os.open(self.state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ExtractionRefused("run state already exists: no replay")
        with os.fdopen(fd, "w") as fh:
            json.dump({"run_id": self.run_id, "state": "APPROVED"}, fh)

    def advance(self, frm: str, to: str) -> None:
        if frm not in _STATES or to not in _STATES:
            raise ExtractionRefused("illegal state transition")
        st = json.loads(open(self.state_path).read())
        if st.get("run_id") != self.run_id or st.get("state") != frm:
            raise ExtractionRefused("state precondition failed")
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"run_id": self.run_id, "state": to}, fh)
        os.replace(tmp, self.state_path)


# ============================ declared synthetic BASE + regeneration (R5) ============================
BASE_NUISANCE = "orthogonal"
BASE_N_PER_CELL = 800
BASE_SEED = 0x0B45E
# eligible synthetic TRAIN population: every TRAIN family × every TRAIN-grid κ EXCEPT the power/MDE
# midpoint (0.35 is a designated OC point, not a calibration population). Uniform mixture (Chris's ruling).
BASE_KAPPAS = tuple(k for k in KAPPA_TRAIN_GRID if abs(k - ORACLE_POWER_KAPPA_MID) > 1e-9)
BASE_CELLS = tuple((fam, float(k)) for fam in TRAIN_FAMILIES for k in BASE_KAPPAS)
_REP = tuple(lo for (_, lo, _) in ORACLE_ENV_CLASS_FAMILIES)   # representative content token per class


def base_schema_hash() -> str:
    return canonical_hash({"families": TRAIN_FAMILIES, "kappas": [float(k) for k in BASE_KAPPAS],
                           "excluded_kappa": ORACLE_POWER_KAPPA_MID, "weights": "uniform",
                           "nuisance": BASE_NUISANCE, "n_per_cell": BASE_N_PER_CELL, "seed": BASE_SEED,
                           "class_representatives": list(_REP), "length_law": "native_literal_fixed_L",
                           "n_classes": ORACLE_ENV_N_CLASSES, "mechanism_hash": invariant_hash()})


def _base_multiset_and_times():
    """Deterministic pooled (future_multiset, future_timestamps) over the frozen uniform mixture."""
    from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
    fms, tss = [], []
    for j, (fam, k) in enumerate(BASE_CELLS):
        c = generate_literal_cell(fam, k, BASE_NUISANCE, BASE_N_PER_CELL, seed=BASE_SEED + 101 * j)
        fms.append(np.asarray(c.future_multiset)); tss.append(np.asarray(c.future_timestamps, float))
    return np.concatenate(fms), np.concatenate(tss)


def _seqs_from(fm: np.ndarray, ts: np.ndarray, source: str):
    ds, bos = SOURCE_DATASET_TOKENS[source], BOS_ID
    for i in range(fm.shape[0]):
        toks = np.array([ds, bos] + [_REP[int(x)] for x in fm[i]], dtype=np.int64)
        t0 = float(ts[i, 0])
        yield toks, np.concatenate([[t0, t0], ts[i]]).astype(np.float64)


def synthetic_base(source: str) -> AggregateStats:
    """The declared synthetic BASE aggregate (native fixed length — the length ECDF is a legitimate point
    mass; its KS against real spread may fail honestly)."""
    fm, ts = _base_multiset_and_times()
    agg, _ = aggregate_from_sequences(source, _seqs_from(fm, ts, source))
    if isinstance(agg, str):
        raise ExtractionRefused(f"synthetic BASE for {source} is {agg}")
    return agg


def regenerate_with_knobs(source: str, knobs: dict[str, float], seed: int) -> AggregateStats | str:
    """Apply fitted knobs at the SEQUENCE level to the frozen BASE population and RE-AGGREGATE — the
    generator-output regeneration check (Pi REVISE#2 #5), distinct from the analytic ``_forward_aggregate``
    surrogate. token_freq_temperature reshapes class draws; zero_gap_bias sets Δt=0 adjacencies;
    timing_rate_scale/gap_dispersion transform the positive gaps."""
    from clinical_jepa.eval.oracle_meta_gen import INVARIANT
    fm, ts = _base_multiset_and_times()
    rng = np.random.default_rng(seed)
    temp = float(np.clip(knobs.get("token_freq_temperature", 1.0), 0.5, 2.0))
    zgb = float(np.clip(knobs.get("zero_gap_bias", 0.0), 0.0, 0.9))
    rate = float(np.clip(knobs.get("timing_rate_scale", 1.0), 0.5, 2.0))
    disp = float(np.clip(knobs.get("gap_dispersion", 1.0), 0.5, 2.0))
    base_p = np.bincount(fm.ravel(), minlength=ORACLE_ENV_N_CLASSES)[:ORACLE_ENV_N_CLASSES].astype(float)
    base_p = base_p / base_p.sum()
    p = base_p ** (1.0 / temp); p = p / p.sum()

    def regen():
        for i in range(fm.shape[0]):
            L = fm.shape[1]
            classes = rng.choice(ORACLE_ENV_N_CLASSES, size=L, p=p)         # tempered class draw
            toks = np.array([SOURCE_DATASET_TOKENS[source], BOS_ID] + [_REP[int(x)] for x in classes],
                            dtype=np.int64)
            # rebuild content timestamps: median-anchored gap transform + zero-gap resampling
            _pg = np.diff(ts[i])[np.diff(ts[i]) > 0] if L > 1 else np.empty(0)
            med = float(np.median(_pg)) if _pg.size else 1.0
            t = [float(ts[i, 0])]
            for j in range(1, L):
                same = rng.random() < zgb
                if same:
                    t.append(t[-1])
                else:
                    g0 = max(1e-6, float(ts[i, j] - ts[i, j - 1]))
                    g = max(1e-6, (med + (g0 - med) * disp) / max(1e-6, rate))
                    t.append(t[-1] + g)
            times = np.concatenate([[t[0], t[0]], np.array(t, float)])
            yield toks, times

    agg, _ = aggregate_from_sequences(source, regen())
    return agg


# ============================ sanitized output (R4/R7) ============================
def sanitized_output(targets, target_counts, bases, calibration, regen, run_meta) -> dict[str, Any]:
    from clinical_jepa.validation import _scan_forbidden_aggregate_keys
    out: dict[str, Any] = {
        "governance_class": "explicitly_cleared_safe_aggregate_only_no_patient_rows",
        "split": TRAIN_SPLIT_ONLY, "n_classes": ORACLE_ENV_N_CLASSES,
        "sources": {s: (a if isinstance(a, str) else asdict(a)) for s, a in targets.items()},
        "audit_counts": target_counts,
        "base": {s: (b if isinstance(b, str) else asdict(b)) for s, b in bases.items()},
        "calibration": calibration, "regeneration_check": regen,
        "identities": {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
                       "calibration_schema_hash": calibration_schema_hash(),
                       "extraction_schema_hash": extraction_schema_hash(),
                       "base_schema_hash": base_schema_hash(),
                       "extraction_code_identity": extraction_code_identity(),
                       "evaluator_identity": ORACLE_EVALUATOR_IDENTITY},
        "run": run_meta}
    leaks = _scan_forbidden_aggregate_keys(out)
    if leaks:
        raise ExtractionRefused("aggregate output failed the forbidden-key scan")
    return out


# ============================ the one-time governed runner (R1/R2/R5) ============================
def _live_identities(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
            "calibration_schema_hash": calibration_schema_hash(),
            "evaluator_identity": ORACLE_EVALUATOR_IDENTITY, "vocab_hash": cfg["vocab_hash"],
            "vocab_name": EXPECTED_VOCAB_NAME, "extraction_schema_hash": extraction_schema_hash(),
            "code_identity": extraction_code_identity(), "config_hash": cfg["config_hash"],
            "run_id": run_id}


def run_calibration_extraction(config_path: str, run_id: str) -> dict[str, Any]:
    """The SINGLE governed entry point. Takes ONLY the approved config location and run id — NO caller
    policy, NO caller paths (Pi REVISE#2 #1/#2). Loads the committed policy internally; refuses if empty or
    on any identity mismatch; derives canonical one-time paths; reads TRAIN SCID then MIMIC once; builds the
    declared BASE; calibrates; runs the generator-output regeneration check; writes the sanitized summary."""
    from clinical_jepa.eval.oracle_aggregate_policy import aggregate_read_authorized, load_policy
    cfg = load_and_verify_config(config_path)
    live = _live_identities(cfg, run_id)
    ok, reason = aggregate_read_authorized(load_policy(), live)
    if not ok:
        raise ExtractionRefused(f"aggregate read refused: {reason}")

    run = OneTimeRun(run_id)
    run.claim()
    targets: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for src, frm, to in (("SCID", "APPROVED", "READING_SCID"), ("MIMIC", "READING_SCID", "READING_MIMIC")):
        run.advance(frm, to)
        targets[src], counts[src] = aggregate_from_sequences(src, _iter_validated_sequences(cfg["paths"][src], src))
    if any(isinstance(t, str) for t in targets.values()):
        run.advance("READING_MIMIC", "REFUSED")
        raise ExtractionRefused("a source is NOT_EVALUABLE")
    run.advance("READING_MIMIC", "FITTING")
    bases = {s: synthetic_base(s) for s in REQUIRED_SOURCES}
    coll = calibrate_sources({s: targets[s] for s in REQUIRED_SOURCES}, bases)
    calib = {"all_sources_within_envelope": coll.all_sources_within_envelope,
             "source_coverage_ok": coll.source_coverage_ok, "combined_hash": coll.combined_hash,
             "per_source": {s: {"fitted_knobs": r.fitted_knobs, "within_envelope": r.within_envelope,
                                "input_hash": r.input_hash, "fitted_param_hash": r.fitted_param_hash,
                                "diagnostics": r.diagnostics} for s, r in coll.per_source.items()}}
    run.advance("FITTING", "REGEN_CHECK")
    regen = {}                                               # envelope on ACTUAL knob-applied generator output
    for s in REQUIRED_SOURCES:
        rq = regenerate_with_knobs(s, coll.per_source[s].fitted_knobs, seed=BASE_SEED + 7)
        env = None if isinstance(rq, str) else realism_envelope(rq, targets[s]).within_envelope
        regen[s] = {"regenerated_within_envelope": env,
                    "surrogate_within_envelope": coll.per_source[s].within_envelope,
                    "agrees": (env == coll.per_source[s].within_envelope)}
    run_meta = {"run_id": run_id, "reviewed_commit": load_policy().get("reviewed_commit"),
                "gate_event_ref": load_policy().get("gate_event_ref"), "config_hash": cfg["config_hash"]}
    out = sanitized_output(targets, counts, bases, calib, regen, run_meta)
    tmp = run.out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, sort_keys=True, indent=2)
    os.replace(tmp, run.out_path)
    run.advance("REGEN_CHECK", "COMPLETE")
    return out


def main(argv: list[str] | None = None) -> int:
    """Deterministic CLI: only --config and --run-id; never caller policy or output destinations."""
    import argparse
    ap = argparse.ArgumentParser(description="One-time aggregate-real calibration extraction (post micro-gate PASS)")
    ap.add_argument("--config", required=True, help="approved LOCAL config path (real TRAIN paths)")
    ap.add_argument("--run-id", required=True, help="approved one-time run id (canonical paths derived from it)")
    args = ap.parse_args(argv)
    try:
        run_calibration_extraction(args.config, args.run_id)
    except ExtractionRefused as e:
        print(json.dumps({"ok": False, "refused": str(e)}))
        return 2
    print(json.dumps({"ok": True, "run_id": args.run_id}))
    return 0
