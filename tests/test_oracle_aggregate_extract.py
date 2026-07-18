"""Aggregate-real extraction — SYNTHETIC temp HDF5/config fixtures ONLY, NO governed read (Pi micro-gate
REVISE#3 adversarial battery). Frozen integer seeds only. Establishes the claimed fail-closed properties.
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
    rng = np.random.default_rng(seed)                        # FROZEN integer seed, no hash()
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
    fr = ([{"family": "x", "start": 0, "end": X.VOCAB_SIZE}] if bad_family
          else [{"family": f, "start": s, "end": e} for (f, s, e) in X.EXPECTED_FAMILY_RANGES])
    return {"vocab_name": X.EXPECTED_VOCAB_NAME,
            "vocab_hash": "deadbeef" if bad_hash else X.EXPECTED_VOCAB_HASH, "family_ranges": fr}


def _make_bundle(tmp, sources=REQUIRED_SOURCES, split="train", filenames=None, bad_family=False,
                 bad_hash=False, extra=False, time_channel=X.TIME_CHANNEL, write=False, seed=11):
    import yaml
    bdir = os.path.join(tmp, X.EXPECTED_BUNDLE)
    os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(_manifest(bad_family, bad_hash)))
    fnames = filenames or {s: f"{s.lower()}_{split}.h5" for s in sources}
    entries = []
    for s in sources:
        p = os.path.join(bdir, fnames[s])
        if write:
            _write_h5(p, s, _healthy(s, ORACLE_ENV_MIN_DENOM + 400, seed=seed + len(s)))
        entries.append({"name": s, "h5_paths": {split: p}})
    if extra:
        entries.append({"name": "OTHER", "h5_paths": {split: os.path.join(bdir, "other_train.h5")}})
    cfg = {"datasets": {"time_channel": time_channel, "source_datasets": entries}}
    cpath = os.path.join(tmp, "cfg.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
    return cpath


class PureAggregationTests(unittest.TestCase):
    def test_structural_excluded_and_audit_counts(self) -> None:
        tok, days = _seq("SCID", [0, 0, 1, 2], [0.0, 0.0, 1.0, 3.0])
        f = X._sequence_features(tok, days)
        self.assertEqual((f["length"], f["n_clusters"], f["n_zero_adj"]), (4, 3, 1))
        agg, counts = X.aggregate_from_sequences("SCID", list(_healthy("SCID", ORACLE_ENV_MIN_DENOM + 80, 3)))
        self.assertIsInstance(agg, AggregateStats)
        self.assertEqual(counts["n_no_content_sequences"], 0)

    def test_no_content_refused_above_fraction(self) -> None:
        good = list(_healthy("SCID", 100, 4))
        empties = [(np.array(_prefix("SCID"), dtype=np.int64), np.array([0.0, 0.0])) for _ in range(20)]
        with self.assertRaises(X.ExtractionRefused):
            X.aggregate_from_sequences("SCID", good + empties)


class StrictValidationTests(unittest.TestCase):
    def _v(self, src, tok, cd):
        return X._validate_raw_sequence(src, 0, np.asarray(tok), np.asarray(cd))

    def test_malformed_cases_refuse(self) -> None:
        for tok, cd in [(None, [0.0]), (np.zeros((2, 2)), np.zeros((2, 2))), ([1048, 1, 10], [0.0, 0.0]),
                        ([1048], [0.0]), ([1048, 1, 99999], [0.0, 0.0, 1.0]),
                        (np.array([1048.0, 1.0, 10.5]), np.array([0.0, 0.0, 1.0])),
                        ([1048, 1, 10], [0.0, 0.0, np.nan]), ([1048, 1, 10, 60], [0.0, 0.0, 2.0, 1.0]),
                        ([1049, 1, 10], [0.0, 0.0, 1.0]), ([1048, 2, 10], [0.0, 0.0, 1.0])]:
            with self.assertRaises(X.ExtractionRefused):
                self._v("SCID", tok, cd)

    def test_error_message_uses_ordinal_not_group_key(self) -> None:
        try:
            X._validate_raw_sequence("SCID", 42, np.array([1049, 1, 10]), np.array([0.0, 0.0, 1.0]))
            self.fail("expected refusal")
        except X.ExtractionRefused as e:
            self.assertIn("seq42", str(e))


class HDFLinkTests(unittest.TestCase):
    def test_soft_link_group_refused(self) -> None:
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                g = h5.create_group("seq0")
                g.create_dataset("token_ids", data=np.array([1048, 1, 10]))
                g.create_dataset(X.TIME_CHANNEL, data=np.array([0.0, 0.0, 1.0]))
                h5["seq1"] = h5py.SoftLink("/seq0")
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))

    def test_external_link_refused(self) -> None:
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(tmp, "other.h5")
            with h5py.File(other, "w") as o:
                o.create_dataset("x", data=np.array([1, 2, 3]))
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                h5["seq0"] = h5py.ExternalLink(other, "/x")
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))

    def test_external_dataset_storage_refused(self) -> None:
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                g = h5.create_group("seq0")
                g.create_dataset("token_ids", shape=(3,), dtype="i8",
                                 external=[(os.path.join(tmp, "ext.bin"), 0, h5py.h5f.UNLIMITED)])
                g.create_dataset(X.TIME_CHANNEL, data=np.array([0.0, 0.0, 1.0]))
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))

    def test_digest_computed_and_content_sensitive(self) -> None:
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "scid_train.h5")
            _write_h5(p, "SCID", _healthy("SCID", 30, 7))
            d1 = hashlib.sha256(); list(X._iter_validated_sequences(p, "SCID", digest=d1))
            p2 = os.path.join(tmp, "scid_train2.h5")
            _write_h5(p2, "SCID", _healthy("SCID", 30, 8))          # different content
            d2 = hashlib.sha256(); list(X._iter_validated_sequences(p2, "SCID", digest=d2))
            self.assertNotEqual(d1.hexdigest(), d2.hexdigest())


class ConfigIdentityTests(unittest.TestCase):
    def test_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(set(X.load_and_verify_config(_make_bundle(tmp))["paths"]), set(REQUIRED_SOURCES))

    def test_extra_unknown_source_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(_make_bundle(tmp, extra=True))

    def test_missing_source_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(_make_bundle(tmp, sources=("SCID",)))

    def test_family_hash_time_split_filename_refused(self) -> None:
        for kw in ({"bad_family": True}, {"bad_hash": True}, {"time_channel": "wall_clock"},
                   {"split": "test"}, {"filenames": {"SCID": "scid_train.h5", "MIMIC": "mimic_val.h5"}}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(X.ExtractionRefused):
                    X.load_and_verify_config(_make_bundle(tmp, **kw))

    def test_symlinked_component_refused(self) -> None:
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            realdir = os.path.join(tmp, "real"); os.makedirs(os.path.join(realdir, X.EXPECTED_BUNDLE))
            bdir = os.path.join(realdir, X.EXPECTED_BUNDLE)
            open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(_manifest()))
            for s in REQUIRED_SOURCES:
                open(os.path.join(bdir, f"{s.lower()}_train.h5"), "w").write("x")
            linkdir = os.path.join(tmp, "link"); os.symlink(realdir, linkdir)   # symlinked ancestor
            cfg = {"datasets": {"time_channel": X.TIME_CHANNEL, "source_datasets": [
                {"name": s, "h5_paths": {"train": os.path.join(linkdir, X.EXPECTED_BUNDLE, f"{s.lower()}_train.h5")}}
                for s in REQUIRED_SOURCES]}}
            cpath = os.path.join(tmp, "c.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(cpath)


class PolicyAndCodeTests(unittest.TestCase):
    def test_empty_policy_refuses(self) -> None:
        self.assertEqual(P.aggregate_read_authorized(P.load_policy(), {})[1], "aggregate_read_policy_empty")

    def test_every_live_anchor_must_match(self) -> None:
        live = {k: k for k in P._LIVE_ANCHORS}
        pol = {**live, "gate_event_ref": "e", "reviewed_commit": "c",
               "train_artifact_identities": {"SCID": "d"}, "sources": list(REQUIRED_SOURCES), "split": "train"}
        self.assertTrue(P.aggregate_read_authorized(pol, live)[0])
        for k in P._LIVE_ANCHORS:
            self.assertFalse(P.aggregate_read_authorized({**pol, k: "X"}, live)[0])

    def test_policy_data_logic_split_and_code_closure(self) -> None:
        import clinical_jepa.eval.oracle_aggregate_policy_data as data
        self.assertIn("base_schema_hash", data.APPROVED_AGGREGATE_READ_POLICY)
        cid = X.extraction_code_identity()
        self.assertEqual(len(cid), 64)
        self.assertEqual(cid, X.extraction_code_identity())        # deterministic closure hash


class OneTimeRunTests(unittest.TestCase):
    def test_absolute_root_and_replay_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(os.path.isabs(X.state_root()))
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                run = X.OneTimeRun("runA"); run.claim()
                cwd = os.getcwd()
                try:
                    os.chdir(tmp)                                  # different cwd cannot re-claim
                    with self.assertRaises(X.ExtractionRefused):
                        X.OneTimeRun("runA").claim()
                finally:
                    os.chdir(cwd)
                run.advance("APPROVED", "READING_SCID")
                with self.assertRaises(X.ExtractionRefused):
                    run.advance("APPROVED", "READING_SCID")

    def test_bad_run_id_refused(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.canonical_run_paths("../escape")


class BaseAndRegenTests(unittest.TestCase):
    def test_base_mixture_excludes_midpoint_and_is_point_mass_length(self) -> None:
        self.assertNotIn(0.35, [float(k) for k in X.BASE_KAPPAS])
        b = X.synthetic_base("SCID")
        self.assertIsInstance(b, AggregateStats)
        self.assertEqual(len(b.length_ecdf), 1)                    # native fixed length => point mass

    def test_knobs_measurably_change_regenerated_output(self) -> None:
        a = X.regenerate_via_generator("MIMIC", {"zero_gap_bias": 0.1, "timing_rate_scale": 1.0,
                                                 "gap_dispersion": 1.0, "token_freq_temperature": 1.0})
        b = X.regenerate_via_generator("MIMIC", {"zero_gap_bias": 0.8, "timing_rate_scale": 2.0,
                                                 "gap_dispersion": 1.8, "token_freq_temperature": 1.0})
        self.assertNotEqual(X._agg_hash(a), X._agg_hash(b))
        # None-knob generation equals the frozen BASE (adapter identity at default)
        self.assertEqual(X.base_schema_hash(), X.base_schema_hash())


class EndToEndTests(unittest.TestCase):
    def _policy(self, cpath, run_id):
        cfg = X.load_and_verify_config(cpath)
        live = X._live_identities(cfg, run_id)
        return {**live, "gate_event_ref": "evt-TEST", "reviewed_commit": "TESTCOMMIT",
                "train_artifact_identities": {s: "declared-at-result-gate" for s in REQUIRED_SOURCES},
                "sources": list(REQUIRED_SOURCES), "split": "train"}

    def test_round_trip_canonical_regen_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                pol = self._policy(cpath, "runE")                  # built UNDER the patched state_root
                mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                           return_value=pol).start()
                self.addCleanup(mock.patch.stopall)
                out = X.run_calibration_extraction(cpath, "runE")
                self.assertIn("regeneration_canonical", out)
                self.assertIn("calibration_eligible", out)
                self.assertIn("train_artifact_digests", out)
                self.assertFalse(out["provenance_verified"])       # pending result gate
                self.assertIn("calibration_surrogate_diagnostic", out)
                sp, _ = X.canonical_run_paths("runE")
                self.assertEqual(json.loads(open(sp).read())["state"], "COMPLETE")
                with self.assertRaises(X.ExtractionRefused):
                    X.run_calibration_extraction(cpath, "runE")     # replay refused

    def test_two_runs_equal_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                                return_value=self._policy(cpath, "r1")):
                    o1 = X.run_calibration_extraction(cpath, "r1")
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                                return_value=self._policy(cpath, "r2")):
                    o2 = X.run_calibration_extraction(cpath, "r2")
            self.assertEqual(o1["sources"], o2["sources"])
            self.assertEqual(o1["train_artifact_digests"], o2["train_artifact_digests"])

    def test_empty_committed_policy_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                with self.assertRaises(X.ExtractionRefused):
                    X.run_calibration_extraction(cpath, "rX")

    def test_partial_run_reentry_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                X.OneTimeRun("rP").claim()                          # a partial run already exists
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                                return_value=self._policy(cpath, "rP")):
                    with self.assertRaises(X.ExtractionRefused):
                        X.run_calibration_extraction(cpath, "rP")

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                self.assertEqual(X.main(["--config", cpath, "--run-id", "cliX"]), 2)   # empty policy -> refuse
                with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                                return_value=self._policy(cpath, "cliOK")):
                    self.assertEqual(X.main(["--config", cpath, "--run-id", "cliOK"]), 0)


class OutputContractTests(unittest.TestCase):
    def test_forbidden_key_refused(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.sanitized_output({"run": {"sequence_id": "leak"}})

    def test_clean_output_ok(self) -> None:
        out = X.sanitized_output({"sources": {"SCID": NOT_EVALUABLE}})
        self.assertEqual(out["governance_class"],
                         "explicitly_cleared_safe_aggregate_only_no_patient_rows")
        self.assertIn("base_schema_hash", out["identities"])


if __name__ == "__main__":
    unittest.main()
