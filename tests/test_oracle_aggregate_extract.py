"""Aggregate-real extraction — SYNTHETIC temp HDF5/config fixtures ONLY, NO governed read (Pi micro-gate
REVISE#1 battery). Covers strict validation, config/identity enforcement, the empty-policy refusal, the
one-time run state machine, the synthetic BASE, the full extraction→BASE→calibration→serialization
round-trip, and malformed-output rejection — every case Pi required.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from clinical_jepa.eval import oracle_aggregate_extract as X
from clinical_jepa.eval import oracle_aggregate_policy as P
from clinical_jepa.eval.oracle_calibration import AggregateStats, NOT_EVALUABLE, REQUIRED_SOURCES
from clinical_jepa.eval.rung2_contract import ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES

_CLS_TOK = (10, 60, 100, 1000, 1040)          # one content token per class range
_BOS = X.BOS_ID


def _prefix(source):
    return [X.SOURCE_DATASET_TOKENS[source], _BOS]           # index0 = DATASET:X, index1 = [BOS]


def _seq(source, classes, days):
    toks = _prefix(source) + [_CLS_TOK[c] for c in classes]
    t = [float(days[0]), float(days[0])] + [float(d) for d in days]
    return np.array(toks, dtype=np.int64), np.array(t, dtype=float)


def _healthy_sequences(source, n, seed=0):
    rng = np.random.default_rng(seed)
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


class PureAggregationTests(unittest.TestCase):
    def test_structural_excluded_and_counts(self) -> None:
        tok, days = _seq("SCID", [0, 0, 1, 2], [0.0, 0.0, 1.0, 3.0])
        f = X._sequence_features(tok, days)
        self.assertEqual(f["length"], 4)
        self.assertEqual(f["n_clusters"], 3)
        self.assertEqual(f["n_zero_adj"], 1)
        self.assertEqual(list(f["per_class"]), [2, 1, 1, 0, 0])

    def test_not_evaluable_below_floor(self) -> None:
        seqs = [_seq("SCID", [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) for _ in range(3)]
        self.assertEqual(X.aggregate_from_sequences("SCID", seqs), NOT_EVALUABLE)

    def test_well_formed_above_floor(self) -> None:
        agg = X.aggregate_from_sequences("MIMIC", _healthy_sequences("MIMIC", ORACLE_ENV_MIN_DENOM + 300))
        self.assertIsInstance(agg, AggregateStats)
        self.assertEqual(agg.n_events, sum(agg.class_counts))
        self.assertEqual(len(agg.class_counts), ORACLE_ENV_N_CLASSES)


class StrictValidationTests(unittest.TestCase):
    def _v(self, source, tok, cd):
        return X._validate_raw_sequence(source, "g", np.asarray(tok), np.asarray(cd))

    def test_missing_or_wrong_rank(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            self._v("SCID", None, [0.0])
        with self.assertRaises(X.ExtractionRefused):
            self._v("SCID", np.zeros((2, 2)), np.zeros((2, 2)))

    def test_length_mismatch_and_short(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            self._v("SCID", [1048, 1, 10], [0.0, 0.0])
        with self.assertRaises(X.ExtractionRefused):
            self._v("SCID", [1048], [0.0])

    def test_bad_tokens(self) -> None:
        with self.assertRaises(X.ExtractionRefused):                 # out of vocab
            self._v("SCID", [1048, 1, 99999], [0.0, 0.0, 1.0])
        with self.assertRaises(X.ExtractionRefused):                 # non-integer
            self._v("SCID", np.array([1048.0, 1.0, 10.5]), np.array([0.0, 0.0, 1.0]))

    def test_bad_time(self) -> None:
        with self.assertRaises(X.ExtractionRefused):                 # non-finite
            self._v("SCID", [1048, 1, 10], [0.0, 0.0, np.nan])
        with self.assertRaises(X.ExtractionRefused):                 # decreasing
            self._v("SCID", [1048, 1, 10, 60], [0.0, 0.0, 2.0, 1.0])

    def test_prefix_and_source_token(self) -> None:
        with self.assertRaises(X.ExtractionRefused):                 # index0 not DATASET:SCID
            self._v("SCID", [1049, 1, 10], [0.0, 0.0, 1.0])          # MIMIC token in a SCID file
        with self.assertRaises(X.ExtractionRefused):                 # index1 not [BOS]
            self._v("SCID", [1048, 2, 10], [0.0, 0.0, 1.0])
        tok, cd = self._v("SCID", [1048, 1, 10, 60], [0.0, 0.0, 1.0, 2.0])   # valid
        self.assertEqual(int(tok[0]), 1048)

    def test_no_content_fraction_refused(self) -> None:
        # >2% no-content sequences must refuse, never silently skip
        good = list(_healthy_sequences("SCID", 100, seed=1))
        empties = [(np.array(_prefix("SCID"), dtype=np.int64), np.array([0.0, 0.0])) for _ in range(20)]
        with self.assertRaises(X.ExtractionRefused):
            X.aggregate_from_sequences("SCID", good + empties)


class ConfigIdentityTests(unittest.TestCase):
    def _bundle(self, tmp, sources=("SCID", "MIMIC"), bundle=X.EXPECTED_BUNDLE, vocab_ok=True, split="train"):
        import yaml
        bdir = os.path.join(tmp, bundle)
        os.makedirs(bdir, exist_ok=True)
        manifest = {"vocab_name": X.EXPECTED_VOCAB_NAME,
                    "vocab_hash": X.EXPECTED_VOCAB_HASH if vocab_ok else "deadbeef",
                    "family_ranges": [{"family": "x", "start": 0, "end": X.VOCAB_SIZE}]}
        open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(manifest))
        entries = []
        for s in sources:
            entries.append({"name": s, "h5_paths": {split: os.path.join(bdir, f"{s.lower()}_{split}.h5")}})
        cfg = {"datasets": {"source_datasets": entries}}
        cpath = os.path.join(tmp, "cfg.yaml")
        open(cpath, "w").write(yaml.safe_dump(cfg))
        return cpath

    def test_valid_config_resolves_two_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = X.load_and_verify_config(self._bundle(tmp))
            self.assertEqual(set(info["paths"]), set(REQUIRED_SOURCES))
            self.assertEqual(info["vocab_hash"], X.EXPECTED_VOCAB_HASH)

    def test_missing_source_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(self._bundle(tmp, sources=("SCID",)))

    def test_wrong_bundle_or_vocab_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(self._bundle(tmp, bundle="some_other_bundle"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(self._bundle(tmp, vocab_ok=False))

    def test_non_train_path_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(self._bundle(tmp, split="test"))


class PolicyAndOneTimeTests(unittest.TestCase):
    def test_empty_policy_refuses(self) -> None:
        ok, reason = P.aggregate_read_authorized(P.load_policy(), {})
        self.assertFalse(ok)
        self.assertEqual(reason, "aggregate_read_policy_empty")

    def test_populated_policy_authorizes_only_on_matching_identities(self) -> None:
        live = {"invariant_hash": "i", "ledger_hash": "l", "calibration_schema_hash": "c",
                "evaluator_identity": "e", "vocab_hash": "v", "vocab_name": "n",
                "extraction_schema_hash": "x", "config_hash": "cf", "run_id": "r1"}
        pol = {**live, "gate_event_ref": "evt", "reviewed_commit": "abc",
               "sources": list(REQUIRED_SOURCES), "split": "train"}
        ok, _ = P.aggregate_read_authorized(pol, live)
        self.assertTrue(ok)
        bad = P.aggregate_read_authorized({**pol, "invariant_hash": "OTHER"}, live)
        self.assertFalse(bad[0])                                     # stale/mismatched identity refused

    def test_one_time_run_refuses_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "run.json")
            run = X.OneTimeRun(sp, "run1")
            run.claim()
            with self.assertRaises(X.ExtractionRefused):
                X.OneTimeRun(sp, "run1").claim()                     # no replay
            run.advance("APPROVED", "READING_SCID")
            with self.assertRaises(X.ExtractionRefused):
                run.advance("APPROVED", "READING_SCID")              # wrong 'from' state refused


class SyntheticBaseTests(unittest.TestCase):
    def test_base_is_evaluable_and_deterministic(self) -> None:
        b1 = X._synthetic_base("SCID")
        b2 = X._synthetic_base("SCID")
        self.assertIsInstance(b1, AggregateStats)
        self.assertEqual(b1.class_counts, b2.class_counts)           # deterministic
        self.assertEqual(len(b1.class_counts), ORACLE_ENV_N_CLASSES)


class EndToEndSyntheticTests(unittest.TestCase):
    """Full extraction→BASE→calibration→serialization on SYNTHETIC h5+config, repeated deterministically,
    with a populated TEST-ONLY policy. NO governed data."""

    def _setup(self, tmp):
        import yaml
        bdir = os.path.join(tmp, X.EXPECTED_BUNDLE)
        os.makedirs(bdir)
        open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(
            {"vocab_name": X.EXPECTED_VOCAB_NAME, "vocab_hash": X.EXPECTED_VOCAB_HASH,
             "family_ranges": [{"family": "x", "start": 0, "end": X.VOCAB_SIZE}]}))
        for s in REQUIRED_SOURCES:
            _write_h5(os.path.join(bdir, f"{s.lower()}_train.h5"), s,
                      _healthy_sequences(s, ORACLE_ENV_MIN_DENOM + 400, seed=hash(s) % 100))
        cfg = {"datasets": {"source_datasets": [
            {"name": s, "h5_paths": {"train": os.path.join(bdir, f"{s.lower()}_train.h5")}}
            for s in REQUIRED_SOURCES]}}
        cpath = os.path.join(tmp, "cfg.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
        return cpath

    def _policy_for(self, cpath, run_id):
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_meta_ledger import ledger_hash
        from clinical_jepa.eval.oracle_calibration import calibration_schema_hash
        from clinical_jepa.eval.rung2_contract import ORACLE_EVALUATOR_IDENTITY
        cfg = X.load_and_verify_config(cpath)
        return {"gate_event_ref": "evt-TEST", "reviewed_commit": "TESTCOMMIT",
                "invariant_hash": invariant_hash(), "ledger_hash": ledger_hash(),
                "calibration_schema_hash": calibration_schema_hash(),
                "evaluator_identity": ORACLE_EVALUATOR_IDENTITY, "vocab_hash": cfg["vocab_hash"],
                "vocab_name": X.EXPECTED_VOCAB_NAME, "extraction_schema_hash": X.extraction_schema_hash(),
                "config_hash": cfg["config_hash"], "sources": list(REQUIRED_SOURCES),
                "split": "train", "run_id": run_id}

    def test_round_trip_and_refuses_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._setup(tmp)
            pol = self._policy_for(cpath, "runA")
            sp, op = os.path.join(tmp, "state.json"), os.path.join(tmp, "out.json")
            out = X.run_calibration_extraction(cpath, policy=pol, run_id="runA", state_path=sp, out_path=op)
            self.assertEqual(set(out["sources"]), set(REQUIRED_SOURCES))
            self.assertIn("calibration", out)
            self.assertEqual(out["identities"]["invariant_hash"], pol["invariant_hash"])
            self.assertEqual(json.loads(open(sp).read())["state"], "COMPLETE")
            # replay with the same run/state path refuses
            with self.assertRaises(X.ExtractionRefused):
                X.run_calibration_extraction(cpath, policy=pol, run_id="runA", state_path=sp, out_path=op)

    def test_empty_committed_policy_refuses_a_governed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._setup(tmp)
            with self.assertRaises(X.ExtractionRefused):
                X.run_calibration_extraction(cpath, policy=P.load_policy(), run_id="r",
                                             state_path=os.path.join(tmp, "s.json"),
                                             out_path=os.path.join(tmp, "o.json"))


class OutputContractTests(unittest.TestCase):
    def test_sanitized_output_is_clean_and_scanner_wired(self) -> None:
        out = X.sanitized_output({"SCID": NOT_EVALUABLE, "MIMIC": NOT_EVALUABLE}, {"x": 1}, {"run_id": "r"})
        self.assertEqual(out["governance_class"],
                         "explicitly_cleared_safe_aggregate_only_no_patient_rows")
        from dataclasses import fields
        from clinical_jepa.validation import FORBIDDEN_AGGREGATE_KEYS, _scan_forbidden_aggregate_keys
        self.assertFalse({f.name for f in fields(AggregateStats)} & FORBIDDEN_AGGREGATE_KEYS)
        self.assertTrue(_scan_forbidden_aggregate_keys({"a": {"token_ids": [1]}}))

    def test_forbidden_key_in_output_refused(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.sanitized_output({"SCID": NOT_EVALUABLE}, {"sequence_id": "leak"}, {"run_id": "r"})


if __name__ == "__main__":
    unittest.main()
