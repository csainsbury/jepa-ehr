"""Aggregate-real extraction — SYNTHETIC temp HDF5/config fixtures ONLY, NO governed read (Pi micro-gate
REVISE#2 adversarial battery). Frozen integer seeds only (no per-process hash()). Establishes the claimed
fail-closed properties rather than trusting the submission summary.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock as mock

import numpy as np

from clinical_jepa.eval import oracle_aggregate_extract as X
from clinical_jepa.eval import oracle_aggregate_policy as P
from clinical_jepa.eval.oracle_calibration import AggregateStats, NOT_EVALUABLE, REQUIRED_SOURCES
from clinical_jepa.eval.rung2_contract import ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES

_CLS_TOK = (10, 60, 100, 1000, 1040)


def _prefix(source):
    return [X.SOURCE_DATASET_TOKENS[source], X.BOS_ID]


def _seq(source, classes, days):
    toks = _prefix(source) + [_CLS_TOK[c] for c in classes]
    t = [float(days[0]), float(days[0])] + [float(d) for d in days]
    return np.array(toks, dtype=np.int64), np.array(t, dtype=float)


def _healthy(source, n, seed):
    rng = np.random.default_rng(seed)                        # FROZEN integer seed (no hash())
    for _ in range(n):
        L = int(rng.integers(6, 14))
        classes = list(rng.integers(0, ORACLE_ENV_N_CLASSES, size=L))
        days = list(np.cumsum(rng.integers(0, 3, size=L)).astype(float))
        yield _seq(source, classes, days)


def _write_h5(path, source, sequences):
    import h5py
    with h5py.File(path, "w") as h5:
        for i, (tok, cd) in enumerate(sequences):
            g = h5.create_group(f"seq{i}")
            g.create_dataset("token_ids", data=np.asarray(tok))
            g.create_dataset(X.TIME_CHANNEL, data=np.asarray(cd, dtype=float))


def _manifest(bad_family=False, bad_hash=False):
    fr = list(X.EXPECTED_FAMILY_RANGES)
    if bad_family:
        fr = [{"family": "x", "start": 0, "end": X.VOCAB_SIZE}]
    else:
        fr = [{"family": f, "start": s, "end": e} for (f, s, e) in fr]
    return {"vocab_name": X.EXPECTED_VOCAB_NAME,
            "vocab_hash": "deadbeef" if bad_hash else X.EXPECTED_VOCAB_HASH, "family_ranges": fr}


def _make_bundle(tmp, sources=REQUIRED_SOURCES, split="train", filenames=None, bad_family=False,
                 bad_hash=False, dup=False, time_channel=X.TIME_CHANNEL, write=False, seed=1):
    import yaml
    bdir = os.path.join(tmp, X.EXPECTED_BUNDLE)
    os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(_manifest(bad_family, bad_hash)))
    fnames = filenames or {s: f"{s.lower()}_{split}.h5" for s in sources}
    entries = []
    for s in sources:
        p = os.path.join(bdir, fnames[s])
        if write:
            _write_h5(p, s, _healthy(s, ORACLE_ENV_MIN_DENOM + 400, seed=seed + hash("") * 0 + len(s)))
        entries.append({"name": s, "h5_paths": {split: p}})
        if dup:
            entries.append({"name": s, "h5_paths": {split: p}})
    cfg = {"datasets": {"time_channel": time_channel, "source_datasets": entries}}
    cpath = os.path.join(tmp, "cfg.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
    return cpath


class PureAggregationTests(unittest.TestCase):
    def test_structural_excluded(self) -> None:
        tok, days = _seq("SCID", [0, 0, 1, 2], [0.0, 0.0, 1.0, 3.0])
        f = X._sequence_features(tok, days)
        self.assertEqual((f["length"], f["n_clusters"], f["n_zero_adj"]), (4, 3, 1))
        self.assertEqual(list(f["per_class"]), [2, 1, 1, 0, 0])

    def test_audit_counts_emitted(self) -> None:
        seqs = list(_healthy("SCID", ORACLE_ENV_MIN_DENOM + 100, seed=3))
        agg, counts = X.aggregate_from_sequences("SCID", seqs)
        self.assertIsInstance(agg, AggregateStats)
        self.assertEqual(counts["n_input_sequences"], len(seqs))
        self.assertEqual(counts["n_no_content_sequences"], 0)

    def test_no_content_refused_above_fraction(self) -> None:
        good = list(_healthy("SCID", 100, seed=4))
        empties = [(np.array(_prefix("SCID"), dtype=np.int64), np.array([0.0, 0.0])) for _ in range(20)]
        with self.assertRaises(X.ExtractionRefused):
            X.aggregate_from_sequences("SCID", good + empties)


class StrictValidationTests(unittest.TestCase):
    def _v(self, source, tok, cd):
        return X._validate_raw_sequence(source, 0, np.asarray(tok), np.asarray(cd))

    def test_malformed_cases(self) -> None:
        cases = [
            (None, [0.0]), (np.zeros((2, 2)), np.zeros((2, 2))),
            ([1048, 1, 10], [0.0, 0.0]), ([1048], [0.0]),
            ([1048, 1, 99999], [0.0, 0.0, 1.0]), (np.array([1048.0, 1.0, 10.5]), np.array([0.0, 0.0, 1.0])),
            ([1048, 1, 10], [0.0, 0.0, np.nan]), ([1048, 1, 10, 60], [0.0, 0.0, 2.0, 1.0]),
            ([1049, 1, 10], [0.0, 0.0, 1.0]),      # MIMIC token in SCID file (source/path swap)
            ([1048, 2, 10], [0.0, 0.0, 1.0]),      # index1 not BOS
        ]
        for tok, cd in cases:
            with self.assertRaises(X.ExtractionRefused):
                self._v("SCID", tok, cd)

    def test_error_message_has_no_group_key(self) -> None:
        try:
            X._validate_raw_sequence("SCID", 42, np.array([1049, 1, 10]), np.array([0.0, 0.0, 1.0]))
        except X.ExtractionRefused as e:
            self.assertIn("seq42", str(e))              # ordinal, not a raw group key


class ConfigIdentityTests(unittest.TestCase):
    def test_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = X.load_and_verify_config(_make_bundle(tmp))
            self.assertEqual(set(info["paths"]), set(REQUIRED_SOURCES))

    def test_extra_source_ignored_but_dup_and_missing_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(_make_bundle(tmp, sources=("SCID",)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(_make_bundle(tmp, dup=True))

    def test_family_range_mismatch_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:                 # fake [0,1050) single family must fail
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(_make_bundle(tmp, bad_family=True))

    def test_vocab_hash_time_channel_filename_split_refused(self) -> None:
        for kw in ({"bad_hash": True}, {"time_channel": "wall_clock"}, {"split": "test"},
                   {"filenames": {"SCID": "scid_train.h5", "MIMIC": "mimic_val.h5"}}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(X.ExtractionRefused):
                    X.load_and_verify_config(_make_bundle(tmp, **kw))

    def test_symlinked_train_path_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            bdir = os.path.join(tmp, X.EXPECTED_BUNDLE); os.makedirs(bdir)
            open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(_manifest()))
            real = os.path.join(bdir, "scid_train.h5"); open(real, "w").write("x")
            link = os.path.join(bdir, "mimic_train.h5"); os.symlink(real, link)
            cfg = {"datasets": {"time_channel": X.TIME_CHANNEL, "source_datasets": [
                {"name": "SCID", "h5_paths": {"train": real}},
                {"name": "MIMIC", "h5_paths": {"train": link}}]}}
            cpath = os.path.join(tmp, "c.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(cpath)


class PolicyAndCodeIdentityTests(unittest.TestCase):
    def test_empty_policy_refuses(self) -> None:
        ok, reason = P.aggregate_read_authorized(P.load_policy(), {})
        self.assertFalse(ok); self.assertEqual(reason, "aggregate_read_policy_empty")

    def test_every_anchor_must_match(self) -> None:
        live = {k: k for k in ("invariant_hash", "ledger_hash", "calibration_schema_hash",
                               "evaluator_identity", "vocab_hash", "vocab_name", "extraction_schema_hash",
                               "code_identity", "config_hash", "run_id")}
        pol = {**live, "gate_event_ref": "e", "reviewed_commit": "c",
               "sources": list(REQUIRED_SOURCES), "split": "train"}
        self.assertTrue(P.aggregate_read_authorized(pol, live)[0])
        for k in live:                                             # mutating ANY anchor refuses
            self.assertFalse(P.aggregate_read_authorized({**pol, k: "X"}, live)[0])

    def test_code_identity_is_deterministic_hash_of_logic_files(self) -> None:
        cid = X.extraction_code_identity()
        self.assertEqual(len(cid), 64)
        self.assertEqual(cid, X.extraction_code_identity())        # deterministic
        # it is a hash of real on-disk logic files, so it cannot be computed if they are unreadable
        with mock.patch("builtins.open", side_effect=OSError):
            with self.assertRaises(OSError):
                X.extraction_code_identity()


class OneTimeRunTests(unittest.TestCase):
    def test_canonical_paths_and_no_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(X, "FIXED_STATE_ROOT", os.path.join(tmp, "st")):
                run = X.OneTimeRun("runA")
                self.assertTrue(run.state_path.endswith(os.path.join("runA", "state.json")))
                run.claim()
                with self.assertRaises(X.ExtractionRefused):
                    X.OneTimeRun("runA").claim()                   # replay refused
                run.advance("APPROVED", "READING_SCID")
                with self.assertRaises(X.ExtractionRefused):
                    run.advance("APPROVED", "READING_SCID")        # wrong 'from' refused

    def test_bad_run_id_refused(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.canonical_run_paths("../escape")


class SyntheticBaseTests(unittest.TestCase):
    def test_base_mixture_evaluable_and_deterministic(self) -> None:
        b1, b2 = X.synthetic_base("SCID"), X.synthetic_base("SCID")
        self.assertIsInstance(b1, AggregateStats)
        self.assertEqual(b1.class_counts, b2.class_counts)
        self.assertEqual(len(b1.length_ecdf), 1)                   # native fixed length => point mass
        self.assertEqual(X.base_schema_hash(), X.base_schema_hash())

    def test_base_excludes_midpoint_kappa(self) -> None:
        self.assertNotIn(0.35, [float(k) for k in X.BASE_KAPPAS])

    def test_regeneration_returns_aggregate(self) -> None:
        knobs = {"token_freq_temperature": 1.2, "zero_gap_bias": 0.3,
                 "timing_rate_scale": 1.1, "gap_dispersion": 1.0}
        rq = X.regenerate_with_knobs("MIMIC", knobs, seed=1)
        self.assertTrue(isinstance(rq, AggregateStats) or rq == NOT_EVALUABLE)


class EndToEndTests(unittest.TestCase):
    def _policy(self, cfg_path, run_id):
        cfg = X.load_and_verify_config(cfg_path)
        live = X._live_identities(cfg, run_id)
        return {**live, "gate_event_ref": "evt-TEST", "reviewed_commit": "TESTCOMMIT",
                "sources": list(REQUIRED_SOURCES), "split": "train"}

    def test_round_trip_regen_and_replay_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True, seed=11)
            pol = self._policy(cpath, "runE")
            with mock.patch.object(X, "FIXED_STATE_ROOT", os.path.join(tmp, "st")), \
                 mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy", return_value=pol):
                out = X.run_calibration_extraction(cpath, "runE")
                self.assertEqual(set(out["sources"]), set(REQUIRED_SOURCES))
                self.assertIn("regeneration_check", out)
                self.assertIn("audit_counts", out)
                self.assertEqual(out["identities"]["base_schema_hash"], X.base_schema_hash())
                sp, _ = X.canonical_run_paths("runE")
                self.assertEqual(json.loads(open(sp).read())["state"], "COMPLETE")
                with self.assertRaises(X.ExtractionRefused):       # canonical-path replay refused
                    X.run_calibration_extraction(cpath, "runE")

    def test_two_clean_runs_produce_equal_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True, seed=11)
            pol1 = self._policy(cpath, "r1"); pol2 = {**pol1, "run_id": "r2"}
            with mock.patch.object(X, "FIXED_STATE_ROOT", os.path.join(tmp, "st")):
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy", return_value=pol1):
                    o1 = X.run_calibration_extraction(cpath, "r1")
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy", return_value=pol2):
                    o2 = X.run_calibration_extraction(cpath, "r2")
            self.assertEqual(o1["sources"], o2["sources"])         # deterministic aggregates

    def test_empty_committed_policy_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True, seed=11)
            with mock.patch.object(X, "FIXED_STATE_ROOT", os.path.join(tmp, "st")):
                with self.assertRaises(X.ExtractionRefused):        # committed policy is EMPTY
                    X.run_calibration_extraction(cpath, "rX")

    def test_partial_run_reentry_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(X, "FIXED_STATE_ROOT", os.path.join(tmp, "st")):
                run = X.OneTimeRun("rP"); run.claim(); run.advance("APPROVED", "READING_SCID")
                cpath = _make_bundle(tmp, write=True, seed=11)
                pol = self._policy(cpath, "rP")
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy", return_value=pol):
                    with self.assertRaises(X.ExtractionRefused):    # claim() sees existing state => no replay
                        X.run_calibration_extraction(cpath, "rP")


class OutputContractTests(unittest.TestCase):
    def test_sanitized_output_scans_forbidden_keys(self) -> None:
        good = X.sanitized_output({"SCID": NOT_EVALUABLE}, {"SCID": {"n_input_sequences": 0}},
                                  {"SCID": NOT_EVALUABLE}, {"x": 1}, {"SCID": {}}, {"run_id": "r"})
        self.assertEqual(good["governance_class"],
                         "explicitly_cleared_safe_aggregate_only_no_patient_rows")
        with self.assertRaises(X.ExtractionRefused):
            X.sanitized_output({"SCID": NOT_EVALUABLE}, {}, {}, {"sequence_id": "leak"}, {}, {"run_id": "r"})


if __name__ == "__main__":
    unittest.main()
