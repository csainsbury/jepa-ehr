"""Aggregate-real extraction — SYNTHETIC temp HDF5/config fixtures ONLY, NO governed read (Pi micro-gate
REVISE#4 adversarial battery). Frozen integer seeds only. Establishes the claimed fail-closed properties.
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
# frozen None-path regression hashes (Pi REVISE#4 #7: default generation must not drift)
_NONE_PATH_HASHES = {"T_latent_factor": "8dc16af2c37e3682", "T_hmm_markov": "44ae0d375764a35a",
                     "T_realized_history": "1a8eec81548b953c"}


def _prefix(src):
    return [X.SOURCE_DATASET_TOKENS[src], X.BOS_ID]


def _seq(src, classes, days):
    toks = _prefix(src) + [_CLS_TOK[c] for c in classes]
    t = [float(days[0]), float(days[0])] + [float(d) for d in days]
    return np.array(toks, dtype=np.int64), np.array(t, dtype=float)


def _healthy(src, n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        L = int(rng.integers(6, 14))
        classes = list(rng.integers(0, ORACLE_ENV_N_CLASSES, size=L))
        days = list(np.cumsum(rng.integers(0, 3, size=L)).astype(float))
        yield _seq(src, classes, days)


def _write_h5(path, src, sequences):
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


def _fast_generator():
    """Shrink the BASE + fit grid so generator-based end-to-end tests stay quick (no governed data)."""
    return [mock.patch.object(X, "BASE_N_PER_CELL", 150),
            mock.patch.object(X, "GEN_FIT_ZGB", (0.1, 0.6)),
            mock.patch.object(X, "GEN_FIT_RATE", (0.85, 1.3)),
            mock.patch.object(X, "GEN_FIT_DISP", (0.85, 1.3))]


class GeneratorAdapterTests(unittest.TestCase):
    def test_default_none_path_regression(self) -> None:
        import hashlib
        from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
        for fam, want in _NONE_PATH_HASHES.items():
            c = generate_literal_cell(fam, 0.3, "orthogonal", 40, seed=7)
            m = hashlib.sha256()
            for a in (c.future_multiset, c.future_timestamps, c.nuisance_u, c.true_order):
                m.update(np.ascontiguousarray(a).tobytes())
            self.assertEqual(m.hexdigest()[:16], want)             # default generation must not drift
            self.assertIsNone(c.calibration_adapter_hash)

    def test_adapter_touches_only_class_and_timing(self) -> None:
        from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
        ctx = {"pooled_class_prior": [0.2] * 5, "global_gap_median": 1.0,
               "pooled_positive_gap_ecdf": [(0.5, 0.4), (1.0, 0.8), (2.0, 1.0)]}
        a = generate_literal_cell("T_latent_factor", 0.3, "orthogonal", 50, seed=7)
        c = generate_literal_cell("T_latent_factor", 0.3, "orthogonal", 50, seed=7,
                                  calib_knobs={"zero_gap_bias": 0.7, "timing_rate_scale": 1.6,
                                               "gap_dispersion": 1.3, "token_freq_temperature": 1.0},
                                  calib_context=ctx)
        for field in ("nuisance_u", "true_order", "item_features", "context_features", "future_events"):
            self.assertTrue(np.array_equal(getattr(a, field), getattr(c, field)), field)
        self.assertFalse(np.array_equal(a.future_timestamps, c.future_timestamps))
        self.assertIsNotNone(c.calibration_adapter_hash)

    def test_fitted_layer_identity_changes_with_knobs(self) -> None:
        from clinical_jepa.eval.oracle_literal_gen import calibration_adapter_hash
        ctx = {"pooled_class_prior": [0.2] * 5, "global_gap_median": 1.0,
               "pooled_positive_gap_ecdf": [(0.5, 0.4), (1.0, 0.8), (2.0, 1.0)]}
        h1 = calibration_adapter_hash({"zero_gap_bias": 0.2}, ctx)
        h2 = calibration_adapter_hash({"zero_gap_bias": 0.8}, ctx)
        self.assertNotEqual(h1, h2)

    def test_calib_knobs_requires_context(self) -> None:
        from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
        with self.assertRaises(ValueError):
            generate_literal_cell("T_latent_factor", 0.3, "orthogonal", 20, seed=1,
                                  calib_knobs={"zero_gap_bias": 0.5})


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

    def test_error_uses_ordinal(self) -> None:
        with self.assertRaises(X.ExtractionRefused) as e:
            X._validate_raw_sequence("SCID", 42, np.array([1049, 1, 10]), np.array([0.0, 0.0, 1.0]))
        self.assertIn("seq42", str(e.exception))

    def test_no_content_refused_above_fraction(self) -> None:
        good = list(_healthy("SCID", 100, 4))
        empties = [(np.array(_prefix("SCID"), dtype=np.int64), np.array([0.0, 0.0])) for _ in range(20)]
        with self.assertRaises(X.ExtractionRefused):
            X.aggregate_from_sequences("SCID", good + empties)


class HDFLinkTests(unittest.TestCase):
    def test_soft_link_refused(self) -> None:
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
            other = os.path.join(tmp, "o.h5")
            with h5py.File(other, "w") as o:
                o.create_dataset("x", data=np.array([1, 2, 3]))
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                h5["seq0"] = h5py.ExternalLink(other, "/x")
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))

    def test_external_storage_refused(self) -> None:
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                g = h5.create_group("seq0")
                g.create_dataset("token_ids", shape=(3,), dtype="i8",
                                 external=[(os.path.join(tmp, "e.bin"), 0, h5py.h5f.UNLIMITED)])
                g.create_dataset(X.TIME_CHANNEL, data=np.array([0.0, 0.0, 1.0]))
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))

    def test_virtual_dataset_refused(self) -> None:
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.h5")
            with h5py.File(src, "w") as s:
                s.create_dataset("d", data=np.array([1048, 1, 10], dtype="i8"))
            p = os.path.join(tmp, "s.h5")
            with h5py.File(p, "w") as h5:
                g = h5.create_group("seq0")
                layout = h5py.VirtualLayout(shape=(3,), dtype="i8")
                layout[:] = h5py.VirtualSource(src, "d", shape=(3,))
                g.create_virtual_dataset("token_ids", layout)
                g.create_dataset(X.TIME_CHANNEL, data=np.array([0.0, 0.0, 1.0]))
            with self.assertRaises(X.ExtractionRefused):
                list(X._iter_validated_sequences(p, "SCID"))


class ContentDigestTests(unittest.TestCase):
    def _digest(self, seqs, src="SCID"):
        d = X._ContentDigest(src)
        for tok, cd in seqs:
            d.add(np.asarray(tok), np.asarray(cd))
        return d.hexdigest()

    def test_order_and_count_and_boundary_sensitive(self) -> None:
        s1 = [_seq("SCID", [0, 1], [0.0, 1.0]), _seq("SCID", [2, 3], [0.0, 2.0])]
        s2 = [_seq("SCID", [2, 3], [0.0, 2.0]), _seq("SCID", [0, 1], [0.0, 1.0])]   # reordered
        s3 = s1 + [_seq("SCID", [4], [0.0])]                                        # extra count
        base = self._digest(s1)
        self.assertNotEqual(base, self._digest(s2))              # order sensitive
        self.assertNotEqual(base, self._digest(s3))              # count sensitive
        self.assertNotEqual(base, self._digest(s1, src="MIMIC"))  # source sensitive
        self.assertEqual(base, self._digest(s1))                 # deterministic


class ConfigIdentityTests(unittest.TestCase):
    def test_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(set(X.load_and_verify_config(_make_bundle(tmp))["paths"]), set(REQUIRED_SOURCES))

    def test_extra_or_missing_source_refused(self) -> None:
        for kw in ({"extra": True}, {"sources": ("SCID",)}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(X.ExtractionRefused):
                    X.load_and_verify_config(_make_bundle(tmp, **kw))

    def test_family_hash_time_split_filename_refused(self) -> None:
        for kw in ({"bad_family": True}, {"bad_hash": True}, {"time_channel": "wall_clock"},
                   {"split": "test"}, {"filenames": {"SCID": "scid_train.h5", "MIMIC": "mimic_val.h5"}}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(X.ExtractionRefused):
                    X.load_and_verify_config(_make_bundle(tmp, **kw))

    def test_symlinked_component_refused(self) -> None:
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            realdir = os.path.join(tmp, "real"); bdir = os.path.join(realdir, X.EXPECTED_BUNDLE)
            os.makedirs(bdir)
            open(os.path.join(bdir, "vocab_manifest.json"), "w").write(json.dumps(_manifest()))
            for s in REQUIRED_SOURCES:
                open(os.path.join(bdir, f"{s.lower()}_train.h5"), "w").write("x")
            linkdir = os.path.join(tmp, "link"); os.symlink(realdir, linkdir)
            cfg = {"datasets": {"time_channel": X.TIME_CHANNEL, "source_datasets": [
                {"name": s, "h5_paths": {"train": os.path.join(linkdir, X.EXPECTED_BUNDLE, f"{s.lower()}_train.h5")}}
                for s in REQUIRED_SOURCES]}}
            cpath = os.path.join(tmp, "c.yaml"); open(cpath, "w").write(yaml.safe_dump(cfg))
            with self.assertRaises(X.ExtractionRefused):
                X.load_and_verify_config(cpath)


class PolicyTests(unittest.TestCase):
    def test_empty_refuses(self) -> None:
        self.assertEqual(P.aggregate_read_authorized(P.load_policy(), {})[1], "aggregate_read_policy_empty")

    def test_every_live_anchor_and_dup_source(self) -> None:
        live = {k: k for k in P._LIVE_ANCHORS}
        pol = {**live, "gate_event_ref": "e", "reviewed_commit": "c",
               "sources": list(REQUIRED_SOURCES), "split": "train"}
        self.assertTrue(P.aggregate_read_authorized(pol, live)[0])
        for k in P._LIVE_ANCHORS:
            self.assertFalse(P.aggregate_read_authorized({**pol, k: "X"}, live)[0])
        dup = {**pol, "sources": list(REQUIRED_SOURCES) + ["SCID"]}
        self.assertFalse(P.aggregate_read_authorized(dup, live)[0])   # duplicate source refused

    def test_provenance_anchor_present_and_data_logic_split(self) -> None:
        import clinical_jepa.eval.oracle_aggregate_policy_data as data
        self.assertIn("provenance_procedure_hash", data.APPROVED_AGGREGATE_READ_POLICY)
        self.assertNotIn("train_artifact_identities", P._LIVE_ANCHORS)


class OneTimeRunTests(unittest.TestCase):
    def test_absolute_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(os.path.isabs(X.state_root()))
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                run = X.OneTimeRun("runA"); run.claim()
                with self.assertRaises(X.ExtractionRefused):
                    X.OneTimeRun("runA").claim()
                run.advance("APPROVED", "READING_SCID")
                with self.assertRaises(X.ExtractionRefused):
                    run.advance("APPROVED", "READING_SCID")

    def test_pre_existing_tmp_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                run = X.OneTimeRun("rT"); run.claim()
                open(run.state_path + ".tmp", "w").write("squat")
                with self.assertRaises(X.ExtractionRefused):
                    run.advance("APPROVED", "READING_SCID")

    def test_bad_run_id(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.canonical_run_paths("../escape")


class BaseAndFitTests(unittest.TestCase):
    def test_base_excludes_midpoint_point_mass(self) -> None:
        self.assertNotIn(0.35, [float(k) for k in X.BASE_KAPPAS])
        with mock.patch.object(X, "BASE_N_PER_CELL", 150):
            b = X.synthetic_base("SCID")
            self.assertEqual(len(b.length_ecdf), 1)

    def test_generator_fit_and_surrogate_are_distinct(self) -> None:
        with mock.patch.object(X, "BASE_N_PER_CELL", 150), mock.patch.object(X, "GEN_FIT_ZGB", (0.1, 0.6)), \
             mock.patch.object(X, "GEN_FIT_RATE", (0.85, 1.3)), mock.patch.object(X, "GEN_FIT_DISP", (1.0,)):
            ctx = X.calib_context()
            tgt, _ = X.aggregate_from_sequences("SCID", _healthy("SCID", ORACLE_ENV_MIN_DENOM + 200, 5))
            knobs, regen, env = X.generator_fit("SCID", tgt, ctx)
            self.assertIn("zero_gap_bias", knobs)
            self.assertTrue(isinstance(regen, AggregateStats) or regen == NOT_EVALUABLE)


class EndToEndTests(unittest.TestCase):
    def _policy(self, cpath, run_id):
        cfg = X.load_and_verify_config(cpath)
        live = X._live_identities(cfg, run_id)
        return {**live, "gate_event_ref": "evt-TEST", "reviewed_commit": "TESTCOMMIT",
                "sources": list(REQUIRED_SOURCES), "split": "train"}

    def _run(self, tmp, run_id):
        cpath = _make_bundle(tmp, write=True)
        patches = _fast_generator() + [mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st"))]
        for p in patches:
            p.start()
        pol = self._policy(cpath, run_id)
        patches.append(mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy", return_value=pol))
        patches[-1].start()
        self.addCleanup(mock.patch.stopall)
        return X.run_calibration_extraction(cpath, run_id), cpath

    def test_round_trip_flags_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, cpath = self._run(tmp, "runE")
            self.assertFalse(out["overall_authorization_eligible"])   # runner NEVER authorizes
            self.assertFalse(out["provenance_verified"])
            self.assertIn("calibration_envelope_eligible_provisional", out)
            self.assertIn("calibration_generator_canonical", out)
            self.assertIn("calibration_surrogate_diagnostic", out)
            self.assertEqual(out["provenance"]["procedure_hash"], X.provenance_procedure_hash())
            with self.assertRaises(X.ExtractionRefused):
                X.run_calibration_extraction(cpath, "runE")           # replay refused

    def test_empty_committed_policy_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            with mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st")):
                with self.assertRaises(X.ExtractionRefused):
                    X.run_calibration_extraction(cpath, "rX")

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpath = _make_bundle(tmp, write=True)
            for p in _fast_generator() + [mock.patch.object(X, "state_root", lambda: os.path.join(tmp, "st"))]:
                p.start()
            self.addCleanup(mock.patch.stopall)
            self.assertEqual(X.main(["--config", cpath, "--run-id", "cliX"]), 2)   # empty policy
            with mock.patch("clinical_jepa.eval.oracle_aggregate_policy.load_policy",
                            return_value=self._policy(cpath, "cliOK")):
                self.assertEqual(X.main(["--config", cpath, "--run-id", "cliOK"]), 0)


class OutputContractTests(unittest.TestCase):
    def test_forbidden_key_refused(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.sanitized_output({"run": {"sequence_id": "leak"}})

    def test_clean_ok(self) -> None:
        out = X.sanitized_output({"sources": {"SCID": NOT_EVALUABLE}})
        self.assertIn("base_schema_hash", out["identities"])
        self.assertIn("state_root_identity", out["identities"])


class Revise5ControlTests(unittest.TestCase):
    def test_base_as_target_self_recovery_passes(self) -> None:
        # DECISIVE positive control (Pi REVISE#5): the fit must place its own BASE inside the envelope.
        with mock.patch.object(X, "BASE_N_PER_CELL", 400):
            ctx = X.calib_context()
            target = X.synthetic_base("SCID")
            knobs, regen, env = X.generator_fit("SCID", target, ctx)
            self.assertIsNotNone(env)
            self.assertTrue(env.within_envelope)                   # self-recovery PASSES
            self.assertEqual(knobs["zero_gap_bias"], 0.35)         # recovers the native profile
            self.assertEqual((knobs["timing_rate_scale"], knobs["gap_dispersion"]), (1.0, 1.0))

    def test_impossible_length_target_fails_provisionally(self) -> None:
        # a target with a spread length distribution cannot be matched by the fixed-length synthetic
        with mock.patch.object(X, "BASE_N_PER_CELL", 300):
            ctx = X.calib_context()
            tgt, _ = X.aggregate_from_sequences("SCID", _healthy("SCID", ORACLE_ENV_MIN_DENOM + 200, 9))
            _, _, env = X.generator_fit("SCID", tgt, ctx)
            self.assertFalse(env.within_envelope)                  # length_ks fails; provisional, no crash

    def test_streaming_equals_batch_aggregate(self) -> None:
        seqs = list(_healthy("SCID", ORACLE_ENV_MIN_DENOM + 300, 12))
        agg, _ = X.aggregate_from_sequences("SCID", seqs)
        # rebuild the ECDFs the OLD batch way and compare
        feats = [X._sequence_features(t, c) for t, c in seqs]
        feats = [f for f in feats if f is not None]
        lengths = np.array([f["length"] for f in feats]); counts = np.array([f["n_clusters"] for f in feats])
        gaps = np.concatenate([f["pos_gaps"] for f in feats if f["pos_gaps"].size])
        self.assertEqual(agg.length_ecdf, X._ecdf(lengths))
        self.assertEqual(agg.count_ecdf, X._ecdf(counts))
        self.assertEqual(agg.positive_gap_ecdf, X._ecdf(gaps))

    def test_content_digest_frozen_little_endian_vector(self) -> None:
        d = X._ContentDigest("SCID")
        d.add(np.array([1048, 1, 10, 60], dtype=np.int64), np.array([0.0, 0.0, 1.0, 3.0]))
        d.add(np.array([1048, 1, 100], dtype=np.int64), np.array([0.0, 0.0, 2.5]))
        self.assertEqual(d.hexdigest(),
                         "4a87994c1e26118b81b2dca6a233368333d6bb178089d495913b8fecf1f0496d")

    def test_adapter_hash_binds_executed_defaults_and_source(self) -> None:
        from clinical_jepa.eval.oracle_literal_gen import calibration_adapter_hash, ZERO_GAP_RATE
        ctx = {"pooled_class_prior": [0.2] * 5, "global_gap_median": 1.0,
               "pooled_positive_gap_ecdf": [(0.5, 0.4), (1.0, 1.0)]}
        # a MISSING zero_gap_bias must bind to the executed default ZERO_GAP_RATE, not 0.0
        h_missing = calibration_adapter_hash({"timing_rate_scale": 1.0}, ctx)
        h_native = calibration_adapter_hash({"zero_gap_bias": ZERO_GAP_RATE, "timing_rate_scale": 1.0}, ctx)
        h_zero = calibration_adapter_hash({"zero_gap_bias": 0.0, "timing_rate_scale": 1.0}, ctx)
        self.assertEqual(h_missing, h_native)
        self.assertNotEqual(h_missing, h_zero)
        # source profile is part of the identity
        self.assertNotEqual(calibration_adapter_hash({}, ctx, source_profile="SCID"),
                            calibration_adapter_hash({}, ctx, source_profile="MIMIC"))

    def test_symlinked_state_ancestor_refused_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            realroot = os.path.join(tmp, "real"); os.makedirs(realroot)
            linkroot = os.path.join(tmp, "linkroot"); os.symlink(realroot, linkroot)
            with mock.patch.object(X, "state_root", lambda: os.path.join(linkroot, "aggregate-calib")):
                with self.assertRaises(X.ExtractionRefused):
                    X.OneTimeRun("rS").claim()

    def test_schema_anchors_present(self) -> None:
        for a in ("generator_fit_schema_hash", "calibration_adapter_schema_hash"):
            self.assertIn(a, P._LIVE_ANCHORS)
        self.assertEqual(len(X.generator_fit_schema_hash()), 64)
        self.assertEqual(len(X.calibration_adapter_schema_hash()), 64)


if __name__ == "__main__":
    unittest.main()
