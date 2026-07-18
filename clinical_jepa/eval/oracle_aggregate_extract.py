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


def _repo_root() -> str:
    """Absolute repository root, anchored to the installed package (not the process cwd) — so the one-time
    state cannot be re-claimed from a different working directory (Pi REVISE#3 #1)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def state_root() -> str:
    return os.path.join(_repo_root(), "state", "aggregate-calib")   # gitignored; absolute canonical root


def state_root_identity() -> str:
    return hashlib.sha256(state_root().encode()).hexdigest()


class ExtractionRefused(RuntimeError):
    """Raised when a step would violate the identity / one-time / TRAIN-only / aggregate-only boundary.
    Messages carry only ordinals + aggregate-safe reason codes — never a raw group key or governed path."""


# ============================ frozen identities ============================
PROVENANCE_PROCEDURE = "single_traversal_content_digest_v1"


def provenance_procedure_hash() -> str:
    """Frozen identity of the EXTRACTED-CONTENT provenance procedure (Pi REVISE#4 #1) — NOT a whole-HDF5
    file digest. Declares the digest version, required sources, the pre-read UNVERIFIED status, and the
    mandatory result gate. Bound as a policy anchor."""
    return canonical_hash({"procedure": PROVENANCE_PROCEDURE, "sources": sorted(REQUIRED_SOURCES),
                           "pre_read_status": "UNVERIFIED", "result_gate": "required",
                           "extraction_schema_hash": extraction_schema_hash(),
                           "digest": "domain|source|schema; per-ordinal len+dtype markers; final count"})


class _ContentDigest:
    """Canonical extracted-content digest (Pi REVISE#4 #1): domain/source/schema prefix, per-ordinal
    sequence + token/time length+dtype markers, and a final sequence count — so it uniquely encodes the
    extracted partition (boundary/order/count sensitive). Raw group keys are never hashed or exposed."""

    def __init__(self, source: str) -> None:
        self._h = hashlib.sha256()
        self._h.update(f"{PROVENANCE_PROCEDURE}|{source}|{extraction_schema_hash()}".encode())
        self._n = 0

    def add(self, tok: np.ndarray, cd: np.ndarray) -> None:
        t = np.ascontiguousarray(tok, dtype=np.int64); c = np.ascontiguousarray(cd, dtype=np.float64)
        self._h.update(f"|SEQ{self._n}|tok:{t.shape[0]}:i8|".encode()); self._h.update(t.tobytes())
        self._h.update(f"|time:{c.shape[0]}:f8|".encode()); self._h.update(c.tobytes())
        self._n += 1

    def hexdigest(self) -> str:
        h = self._h.copy()
        h.update(f"|COUNT|{self._n}".encode())
        return h.hexdigest()


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
    """Hash of the COMPLETE executable logic closure that can affect what is read, how the BASE is
    generated, how authorization is decided, or how output is leak-scanned (Pi REVISE#3 #5): this module,
    calibration, spec, the literal + meta generators, contracts, the policy LOGIC (data split out), and the
    validation scanner. The policy DATA module is excluded, so a policy-population edit does not change the
    identity (no circular hash) while any executable change does."""
    import clinical_jepa.eval.oracle_calibration as _cal
    import clinical_jepa.eval.oracle_spec as _spec
    import clinical_jepa.eval.oracle_literal_gen as _lit
    import clinical_jepa.eval.oracle_meta_gen as _meta
    import clinical_jepa.eval.oracle_contracts as _con
    import clinical_jepa.eval.oracle_aggregate_policy as _pol
    import clinical_jepa.eval.rung2_contract as _r2c
    import clinical_jepa.validation as _val
    h = hashlib.sha256()
    for mod in sorted((__file__, _cal.__file__, _spec.__file__, _lit.__file__, _meta.__file__,
                       _con.__file__, _pol.__file__, _r2c.__file__, _val.__file__)):
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


def _iter_validated_sequences(h5_path: str, source: str, digest: "_ContentDigest | None" = None):
    """Yield validated (token_ids, cumulative_days). Rejects soft/external/virtual links AND datasets with
    external raw storage so a nominal TRAIN file cannot pull bytes from another (Pi REVISE#3 #3). Wraps
    h5py errors into sanitized refusals with an ORDINAL, never a raw group key (#4). If ``digest`` is given,
    folds each validated sequence's bytes into it — the provenance digest computed DURING the one permitted
    traversal (Pi REVISE#3 #4); raw keys are never hashed or exposed."""
    import h5py
    try:
        h5 = h5py.File(h5_path, "r")
    except Exception as e:                                  # sanitize: no governed path in the message
        raise ExtractionRefused(f"{source}: cannot open TRAIN bundle ({type(e).__name__})")
    with h5:
        for ordinal, group in enumerate(h5.keys()):
            if not isinstance(h5.get(group, getlink=True), h5py.HardLink):
                raise ExtractionRefused(f"{source}#seq{ordinal}: non-hard link rejected")
            grp = h5.get(group)
            if grp is None or not isinstance(grp, h5py.Group):
                raise ExtractionRefused(f"{source}#seq{ordinal}: not a group")
            for ds in ("token_ids", TIME_CHANNEL):
                if ds not in grp:
                    raise ExtractionRefused(f"{source}#seq{ordinal}: missing {ds}")
                if not isinstance(grp.get(ds, getlink=True), h5py.HardLink):
                    raise ExtractionRefused(f"{source}#seq{ordinal}: {ds} non-hard link rejected")
                d = grp[ds]
                if getattr(d, "is_virtual", False):
                    raise ExtractionRefused(f"{source}#seq{ordinal}: {ds} virtual dataset rejected")
                try:
                    ext = d.external
                except Exception:
                    ext = None
                if ext:
                    raise ExtractionRefused(f"{source}#seq{ordinal}: {ds} external raw storage rejected")
            try:
                tok, cd = grp["token_ids"][:], grp[TIME_CHANNEL][:]
            except Exception as e:
                raise ExtractionRefused(f"{source}#seq{ordinal}: unreadable dataset ({type(e).__name__})")
            tok, cd = _validate_raw_sequence(source, ordinal, tok, cd)
            if digest is not None:                             # canonical content digest (never raw keys)
                digest.add(tok, cd)
            yield tok, cd


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
    # the ENTIRE source-dataset entry list must be exactly one SCID and one MIMIC — no unknown, no
    # duplicate (Pi REVISE#3 #3: an ignored extra source must not pass).
    names = [str(e.get("name")) for e in block["source_datasets"]]
    if sorted(names) != sorted(REQUIRED_SOURCES):
        raise ExtractionRefused("config source entries must be exactly one SCID and one MIMIC")
    seen = {str(e["name"]): str((e.get("h5_paths") or {}).get(TRAIN_SPLIT_ONLY) or "")
            for e in block["source_datasets"]}
    if any(not p for p in seen.values()):
        raise ExtractionRefused("a source has no TRAIN h5 path")
    tc = str(block.get("time_channel", cfg.get("time_channel", "")))
    if tc != TIME_CHANNEL:
        raise ExtractionRefused(f"config time_channel {tc!r} != {TIME_CHANNEL!r}")
    real_paths: dict[str, str] = {}
    for src, p in seen.items():
        _refuse_symlinked_components(src, p)                    # reject a symlink in ANY path component
        rp = os.path.realpath(p)
        if os.path.basename(rp) != EXPECTED_TRAIN_FILENAMES[src]:
            raise ExtractionRefused(f"{src}: TRAIN filename != {EXPECTED_TRAIN_FILENAMES[src]}")
        if EXPECTED_BUNDLE not in rp.split(os.sep):
            raise ExtractionRefused(f"{src}: not under bundle {EXPECTED_BUNDLE}")
        real_paths[src] = rp
    vocab_hash = _verify_manifest(real_paths)
    return {"paths": real_paths, "config_hash": hashlib.sha256(raw).hexdigest(), "vocab_hash": vocab_hash}


def _refuse_symlinked_components(src: str, path: str) -> None:
    """Reject a symlink in ANY component of the path, not just the final one (Pi REVISE#3 #3)."""
    cur = path if os.path.isabs(path) else os.path.abspath(path)
    parts: list[str] = []
    while True:
        if os.path.islink(cur):
            raise ExtractionRefused(f"{src}: symlinked path component")
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        parts.append(cur); cur = parent


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
    """Canonical state + output paths DERIVED from the run id under the ABSOLUTE repo-anchored gitignored
    root — never caller-supplied and never cwd-relative (Pi REVISE#3 #1/#2). run_id is a safe slug (no `.`
    so `../escape` refuses)."""
    if not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ExtractionRefused("run_id must be a nonempty [A-Za-z0-9-_] slug")
    d = os.path.join(state_root(), run_id)
    return os.path.join(d, "state.json"), os.path.join(d, "result.json")


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _excl_write_json(path: str, obj: dict) -> None:
    """Exclusive, symlink-refusing create + write (O_EXCL|O_NOFOLLOW): a pre-existing file or a symlink at
    ``path`` refuses rather than being followed (Pi REVISE#3 #1)."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, sort_keys=True)


def _atomic_replace_json(path: str, obj: dict) -> None:
    """Write to a fresh exclusive tmp then atomically replace. A pre-existing tmp or symlink is REFUSED, not
    unlinked (Pi REVISE#4 #5) — a squatted tmp is an anomaly, not something to silently overwrite."""
    tmp = path + ".tmp"
    if os.path.islink(tmp) or os.path.exists(tmp):
        raise ExtractionRefused("pre-existing run tmp artifact: refusing")
    _excl_write_json(tmp, obj)
    os.replace(tmp, path)


def _refuse_symlinked_run_ancestors(path: str) -> None:
    """Reject a symlink in ANY component from the repo root down to the run path (Pi REVISE#4 #5), and
    verify the run path is under the intended absolute repo-anchored state root."""
    root = os.path.realpath(state_root())
    if os.path.realpath(path).split(os.sep)[:len(root.split(os.sep))] != root.split(os.sep):
        raise ExtractionRefused("run path escapes the canonical state root")
    cur = os.path.abspath(path)
    stop = os.path.dirname(_repo_root())
    while cur and cur != stop and cur != os.path.dirname(cur):
        if os.path.islink(cur):
            raise ExtractionRefused("symlinked state-path ancestor")
        cur = os.path.dirname(cur)


class OneTimeRun:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.state_path, self.out_path = canonical_run_paths(run_id)

    def claim(self) -> None:
        d = os.path.dirname(self.state_path)
        os.makedirs(d, exist_ok=True)
        _refuse_symlinked_run_ancestors(self.state_path)       # reject symlink in ANY ancestor (Pi #5)
        for p in (self.state_path, self.out_path):
            if os.path.islink(p) or os.path.exists(p):
                raise ExtractionRefused("run artifact already exists: no replay")
        try:
            _excl_write_json(self.state_path, {"run_id": self.run_id, "state": "APPROVED"})
        except FileExistsError:
            raise ExtractionRefused("run state already exists: no replay")

    def advance(self, frm: str, to: str) -> None:
        if frm not in _STATES or to not in _STATES:
            raise ExtractionRefused("illegal state transition")
        st = json.loads(open(self.state_path).read())
        if st.get("run_id") != self.run_id or st.get("state") != frm:
            raise ExtractionRefused("state precondition failed")
        _atomic_replace_json(self.state_path, {"run_id": self.run_id, "state": to})


# ============================ declared synthetic BASE + regeneration (R5) ============================
BASE_NUISANCE = "orthogonal"
BASE_N_PER_CELL = 800
BASE_SEED = 0x0B45E
# eligible synthetic TRAIN population: every TRAIN family × every TRAIN-grid κ EXCEPT the power/MDE
# midpoint (0.35 is a designated OC point, not a calibration population). Uniform mixture (Chris's ruling).
BASE_KAPPAS = tuple(k for k in KAPPA_TRAIN_GRID if abs(k - ORACLE_POWER_KAPPA_MID) > 1e-9)
BASE_CELLS = tuple((fam, float(k)) for fam in TRAIN_FAMILIES for k in BASE_KAPPAS)
_REP = tuple(lo for (_, lo, _) in ORACLE_ENV_CLASS_FAMILIES)   # representative content token per class


BASE_CELL_SEED_SCHEDULE = "BASE_SEED + 101*j"          # per-cell seed, bound in base_schema_hash


def base_schema_hash() -> str:
    return canonical_hash({"families": TRAIN_FAMILIES, "kappas": [float(k) for k in BASE_KAPPAS],
                           "excluded_kappa": ORACLE_POWER_KAPPA_MID, "weights": "uniform",
                           "nuisance": BASE_NUISANCE, "n_per_cell": BASE_N_PER_CELL, "root_seed": BASE_SEED,
                           "per_cell_seed_schedule": BASE_CELL_SEED_SCHEDULE,
                           "cell_order": [(f, float(k)) for (f, k) in BASE_CELLS],
                           "class_representatives": list(_REP), "length_law": "native_literal_fixed_L",
                           "adapter": "generate_literal_cell.calib_knobs",
                           "n_classes": ORACLE_ENV_N_CLASSES, "mechanism_hash": invariant_hash()})


# coarse deterministic knob grid the GENERATOR fit searches (Pi REVISE#4 #4: fit against regenerated
# outputs). token_freq_temperature stays 1.0 — the synthetic class prior is uniform, so tempering it is a
# no-op and the class marginal falsifies honestly; only the timing knobs are calibratable.
GEN_FIT_ZGB = (0.05, 0.2, 0.4, 0.6, 0.8)
GEN_FIT_RATE = (0.6, 0.85, 1.0, 1.3, 1.7)
GEN_FIT_DISP = (0.6, 0.85, 1.0, 1.3, 1.7)


def _base_cells(calib_knobs: dict | None = None, calib_context: dict | None = None):
    """The frozen uniform-mixture BASE cells, optionally regenerated THROUGH the calibration adapter — the
    single generator route by which fitted knobs alter generated cells (Pi #6/#4). Deterministic per-cell
    seed ``BASE_SEED + 101*j``."""
    from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
    fms, tss = [], []
    for j, (fam, k) in enumerate(BASE_CELLS):
        c = generate_literal_cell(fam, k, BASE_NUISANCE, BASE_N_PER_CELL, seed=BASE_SEED + 101 * j,
                                  calib_knobs=calib_knobs, calib_context=calib_context)
        fms.append(np.asarray(c.future_multiset)); tss.append(np.asarray(c.future_timestamps, float))
    return np.concatenate(fms), np.concatenate(tss)


def calib_context() -> dict[str, Any]:
    """POOLED calibration context (Pi REVISE#4 #4): the class prior and global positive-gap median of the
    DEFAULT (uncalibrated) BASE population. Computed once; the adapter uses it so it is coherent, not a
    per-cell surrogate mismatch. Deterministic."""
    fm, ts = _base_cells(None)
    prior = (np.bincount(fm.ravel(), minlength=ORACLE_ENV_N_CLASSES)[:ORACLE_ENV_N_CLASSES]
             / max(1, fm.size)).astype(float)
    d = np.diff(ts, axis=1)
    pos = d[d > 0]
    return {"pooled_class_prior": [float(x) for x in prior],
            "global_gap_median": float(np.median(pos)) if pos.size else 1.0}


def _seqs_from(fm: np.ndarray, ts: np.ndarray, source: str):
    ds, bos = SOURCE_DATASET_TOKENS[source], BOS_ID
    for i in range(fm.shape[0]):
        toks = np.array([ds, bos] + [_REP[int(x)] for x in fm[i]], dtype=np.int64)
        t0 = float(ts[i, 0])
        yield toks, np.concatenate([[t0, t0], ts[i]]).astype(np.float64)


def synthetic_base(source: str) -> AggregateStats:
    """The declared synthetic BASE aggregate (default, uncalibrated)."""
    fm, ts = _base_cells(None)
    agg, _ = aggregate_from_sequences(source, _seqs_from(fm, ts, source))
    if isinstance(agg, str):
        raise ExtractionRefused(f"synthetic BASE for {source} is {agg}")
    return agg


def regenerate_via_generator(source: str, knobs: dict[str, float], context: dict) -> AggregateStats | str:
    """Regenerate the BASE population THROUGH the generator adapter with the POOLED context — the actual
    route certification would use — and re-aggregate (Pi REVISE#4 #4)."""
    fm, ts = _base_cells(dict(knobs), context)
    agg, _ = aggregate_from_sequences(source, _seqs_from(fm, ts, source))
    return agg


def _timing_objective(regen: AggregateStats | str, target: AggregateStats) -> float:
    """Calibratable-marginal distance (positive-gap KS + |Δt=0 diff|) between a REGENERATED aggregate and
    the target — the fit's objective on the knobs that can actually move (timing)."""
    from clinical_jepa.eval.oracle_calibration import _ks
    if isinstance(regen, str):
        return float("inf")
    return (_ks(regen.positive_gap_ecdf, target.positive_gap_ecdf)
            + abs(regen.delta_t_zero_fraction - target.delta_t_zero_fraction))


def generator_fit(source: str, target: AggregateStats, context: dict):
    """CANONICAL calibration fit (Pi REVISE#4 #4): deterministic grid search over the timing knobs, each
    evaluated by REGENERATING the population through the generator adapter and scoring against the target —
    NOT the analytic surrogate. Returns (fitted_knobs, regenerated_aggregate, envelope_result)."""
    from clinical_jepa.eval.oracle_calibration import realism_envelope
    best_knobs, best_obj = None, float("inf")
    for zgb in GEN_FIT_ZGB:
        for rate in GEN_FIT_RATE:
            for disp in GEN_FIT_DISP:
                knobs = {"token_freq_temperature": 1.0, "zero_gap_bias": zgb,
                         "timing_rate_scale": rate, "gap_dispersion": disp}
                obj = _timing_objective(regenerate_via_generator(source, knobs, context), target)
                if obj < best_obj - 1e-12:
                    best_obj, best_knobs = obj, knobs
    regen = regenerate_via_generator(source, best_knobs, context)
    env = None if isinstance(regen, str) else realism_envelope(regen, target)
    return best_knobs, regen, env


# ============================ sanitized output (R4/R7) ============================
def _agg_hash(a: AggregateStats | str) -> str:
    return canonical_hash(a if isinstance(a, str) else asdict(a))


def sanitized_output(payload: dict[str, Any]) -> dict[str, Any]:
    from clinical_jepa.validation import _scan_forbidden_aggregate_keys
    out = {"governance_class": "explicitly_cleared_safe_aggregate_only_no_patient_rows",
           "split": TRAIN_SPLIT_ONLY, "n_classes": ORACLE_ENV_N_CLASSES, **payload,
           "identities": {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
                          "calibration_schema_hash": calibration_schema_hash(),
                          "extraction_schema_hash": extraction_schema_hash(),
                          "base_schema_hash": base_schema_hash(),
                          "extraction_code_identity": extraction_code_identity(),
                          "state_root_identity": state_root_identity(),
                          "evaluator_identity": ORACLE_EVALUATOR_IDENTITY}}
    leaks = _scan_forbidden_aggregate_keys(out)
    if leaks:
        raise ExtractionRefused("aggregate output failed the forbidden-key scan")
    return out


# ============================ the one-time governed runner (R1/R2/R4/R6) ============================
def _live_identities(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
            "calibration_schema_hash": calibration_schema_hash(),
            "evaluator_identity": ORACLE_EVALUATOR_IDENTITY, "vocab_hash": cfg["vocab_hash"],
            "vocab_name": EXPECTED_VOCAB_NAME, "extraction_schema_hash": extraction_schema_hash(),
            "base_schema_hash": base_schema_hash(), "code_identity": extraction_code_identity(),
            "state_root_identity": state_root_identity(),
            "provenance_procedure_hash": provenance_procedure_hash(), "config_hash": cfg["config_hash"],
            "run_id": run_id}


def run_calibration_extraction(config_path: str, run_id: str) -> dict[str, Any]:
    """The SINGLE governed entry point. Takes ONLY the approved config location and run id — NO caller
    policy, NO caller paths (Pi #1/#2). Loads ONE immutable policy snapshot internally; refuses if empty or
    on any identity mismatch; derives canonical one-time paths; reads TRAIN SCID then MIMIC once (folding a
    CANONICAL content digest DURING the traversal, #1/#4); the CANONICAL provisional eligibility comes from
    the GENERATOR-ADAPTER fit (#4, analytic surrogate demoted to diagnostic). ``provenance_verified`` and
    ``overall_authorization_eligible`` remain FALSE — this runner cannot authorize anything (#2)."""
    from clinical_jepa.eval.oracle_aggregate_policy import aggregate_read_authorized, load_policy
    policy = load_policy()                                    # ONE immutable snapshot for the whole run (#6)
    cfg = load_and_verify_config(config_path)
    ok, reason = aggregate_read_authorized(policy, _live_identities(cfg, run_id))
    if not ok:
        raise ExtractionRefused(f"aggregate read refused: {reason}")

    run = OneTimeRun(run_id)
    run.claim()
    targets: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    content_digests: dict[str, str] = {}
    for src, frm, to in (("SCID", "APPROVED", "READING_SCID"), ("MIMIC", "READING_SCID", "READING_MIMIC")):
        run.advance(frm, to)
        dig = _ContentDigest(src)                            # canonical extracted-content digest (#1)
        targets[src], counts[src] = aggregate_from_sequences(
            src, _iter_validated_sequences(cfg["paths"][src], src, digest=dig))
        content_digests[src] = dig.hexdigest()
    if any(isinstance(t, str) for t in targets.values()):
        run.advance("READING_MIMIC", "REFUSED")
        raise ExtractionRefused("a source is NOT_EVALUABLE")
    run.advance("READING_MIMIC", "FITTING")
    ctx = calib_context()                                     # POOLED adapter context (#4)
    bases = {s: synthetic_base(s) for s in REQUIRED_SOURCES}
    surrogate = calibrate_sources({s: targets[s] for s in REQUIRED_SOURCES}, bases)  # DIAGNOSTIC ONLY
    run.advance("FITTING", "REGEN_CHECK")
    canonical: dict[str, Any] = {}
    envelope_eligible = True
    for s in REQUIRED_SOURCES:
        knobs, regen, env = generator_fit(s, targets[s], ctx)     # CANONICAL fit vs regenerated output (#4)
        within = None if env is None else bool(env.within_envelope)
        checks = None if env is None else {k: [float(v), bool(ok)] for k, (v, ok) in env.checks.items()}
        envelope_eligible = envelope_eligible and bool(within)
        canonical[s] = {"fitted_knobs": knobs, "regenerated_within_envelope": within,
                        "regenerated_aggregate": (regen if isinstance(regen, str) else asdict(regen)),
                        "regenerated_hash": _agg_hash(regen), "envelope_checks": checks,
                        "adapter_source_profile": s,             # source-conditioning identity (#4)
                        "calibration_adapter_hash": _adapter_hash(knobs, ctx)}
    payload = {
        "sources": {s: (a if isinstance(a, str) else asdict(a)) for s, a in targets.items()},
        "audit_counts": counts,
        "provenance": {"procedure_hash": provenance_procedure_hash(),
                       "extracted_content_digests": content_digests, "status": "UNVERIFIED"},
        "base": {s: asdict(bases[s]) for s in REQUIRED_SOURCES},
        "calibration_generator_canonical": canonical,
        "calibration_surrogate_diagnostic": {                # analytic surrogate = DIAGNOSTIC ONLY (#4)
            "source_coverage_ok": surrogate.source_coverage_ok,
            "per_source": {s: {"fitted_knobs": surrogate.per_source[s].fitted_knobs,
                               "within_envelope": surrogate.per_source[s].within_envelope,
                               "diagnostics": surrogate.per_source[s].diagnostics} for s in REQUIRED_SOURCES}},
        # THREE distinct flags (Pi #2): provisional statistical result / provenance / authorization.
        "calibration_envelope_eligible_provisional": bool(envelope_eligible and surrogate.source_coverage_ok),
        "provenance_verified": False,                         # false until the separate result gate
        "overall_authorization_eligible": False,             # this runner NEVER authorizes anything
        "run": {"run_id": run_id, "reviewed_commit": policy.get("reviewed_commit"),
                "gate_event_ref": policy.get("gate_event_ref"), "config_hash": cfg["config_hash"]}}
    out = sanitized_output(payload)
    _atomic_replace_json(run.out_path, out)
    run.advance("REGEN_CHECK", "COMPLETE")
    return out


def _adapter_hash(knobs: dict, ctx: dict) -> str:
    from clinical_jepa.eval.oracle_literal_gen import calibration_adapter_hash
    return calibration_adapter_hash(knobs, ctx)


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


if __name__ == "__main__":               # functional entrypoint (Pi REVISE#3 #2)
    raise SystemExit(main())
