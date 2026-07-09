from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import (  # noqa: E402
    build_source_h5s,
    joint_arms_config,
    joint_dataset_config,
    write_tiny_vocab,
    write_yaml,
)

from clinical_jepa.splits.build_index import main as build_index_main  # noqa: E402
from clinical_jepa.splits.readiness_manifest import build_readiness_manifest  # noqa: E402
from clinical_jepa.splits.rung_minus1_driver import (  # noqa: E402
    CompositeGateError,
    assert_composite_or_raise,
    build_composite_gate,
    main as driver_main,
)
from clinical_jepa.utils import write_json  # noqa: E402


def _readiness(td: Path, *, scid_lens, mimic_lens, min_valid_windows):
    paths = build_source_h5s(td, scid_lens=scid_lens, mimic_lens=mimic_lens)
    vocab = write_tiny_vocab(td / "vocab.json")
    idx = td / "idx"
    cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx, min_valid_windows=min_valid_windows)
    build_index_main(["--dataset-config", write_yaml(td / "dataset.yaml", cfg), "--output-dir", str(idx)])
    index_paths = {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")}
    manifest = build_readiness_manifest(cfg, joint_arms_config(), index_paths)
    return cfg, manifest


_LEAK_AUDIT_KEYS = (
    "patient_overlap", "window_inheritance", "horizon_boundary", "forbidden_tokens",
    "cached_embeddings", "duplicate_windows", "label_feature_separation",
)


def _leak(status="pass", sources=("SCID", "MIMIC"), aggregate_only=True):
    """Schema-valid synthetic leakage-audit report; ``status`` drives the
    label_feature_separation audit + overall_status (all others pass)."""
    audits = {k: {"status": "pass", "violations": 0} for k in _LEAK_AUDIT_KEYS}
    audits["label_feature_separation"] = {"status": status, "violations": 0 if status == "pass" else 1}
    return {
        "schema_version": "clinical-jepa-leakage-audit-v0",
        "created_utc": "2026-07-06T00:00:00Z",
        "dataset": "joint-test",
        "target_blocks": ["T0"],
        "audits": audits,
        "overall_status": status,
        "blocks_by_source": {s: 10 for s in sources},
        "aggregate_only": aggregate_only,
    }


class CompositeDriverTests(unittest.TestCase):
    def test_composite_passes_when_both_halves_green(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            self.assertEqual(readiness["gate_status"], "pass")
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "pass")
            self.assertEqual(m["failed_checks"], [])
            self.assertEqual(m["component_status"], {"readiness_gate": "pass", "leakage_audit": "pass", "governance_scan": "pass", "wallclock_readiness": "not_provided"})
            assert_composite_or_raise(m)  # does not raise

    def test_fails_closed_when_readiness_under_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20], min_valid_windows=3)
            self.assertEqual(readiness["gate_status"], "fail")
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("readiness_gate_pass", m["failed_checks"])
            with self.assertRaises(CompositeGateError):
                assert_composite_or_raise(m)

    def test_fails_when_leakage_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            m = build_composite_gate(readiness, _leak("fail"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("leakage_audit_pass", m["failed_checks"])

    def test_fails_on_governance_violation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            m = build_composite_gate(readiness, _leak("pass", aggregate_only=False), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("governance_aggregate_only", m["failed_checks"])

    def test_fails_on_source_prefix_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            cfg["mask"]["source_prefix_len"] = 3  # readiness computed with 2
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("source_prefix_consistency", m["failed_checks"])

    def test_fails_closed_without_dataset_config(self) -> None:
        # A governance gate must not pass without provenance verification.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=None)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("provenance_verified", m["failed_checks"])

    def test_fails_when_critical_audit_not_configured(self) -> None:
        # The gameable "pass by being absent" hole: overall_status=pass but a
        # leakage-critical audit merely not_configured must NOT pass the composite.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            for absent in ("forbidden_tokens", "label_feature_separation"):
                leak = _leak("pass")
                leak["audits"][absent] = {"status": "not_configured", "violations": 0}
                m = build_composite_gate(readiness, leak, dataset_cfg=cfg)
                self.assertEqual(m["composite_status"], "fail", absent)
                self.assertIn("leakage_critical_audits_verified", m["failed_checks"])

    def test_fails_when_noncritical_audit_not_configured(self) -> None:
        # Defense-in-depth: a non-critical audit reported not_configured (a
        # configured check that didn't run) must not launder into PASS.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            leak = _leak("pass")
            leak["audits"]["duplicate_windows"] = {"status": "not_configured", "violations": 0}
            m = build_composite_gate(readiness, leak, dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("leakage_no_unconfigured_audits", m["failed_checks"])

    def test_cached_embeddings_not_applicable_is_allowed(self) -> None:
        # not_applicable is legitimate ONLY for cached_embeddings.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            leak = _leak("pass")
            leak["audits"]["cached_embeddings"] = {"status": "not_applicable", "violations": 0}
            m = build_composite_gate(readiness, leak, dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "pass")

    def test_fails_when_any_audit_failed_despite_overall_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            leak = _leak("pass")
            leak["audits"]["patient_overlap"] = {"status": "fail", "violations": 9}  # overall_status stays "pass"
            m = build_composite_gate(readiness, leak, dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("leakage_no_failed_audits", m["failed_checks"])

    def test_fails_when_gate_pass_but_under_floor_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            readiness["under_floor"] = [{"source": "MIMIC", "split": "dev", "valid_matched_windows": 1}]  # inconsistent
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("readiness_gate_pass", m["failed_checks"])

    def test_fails_on_zero_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            readiness["source_prefix_len"] = 0  # no source mask at all
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("source_prefix_floor", m["failed_checks"])

    def test_fails_on_pii_key_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            readiness["mrn"] = "12345"  # planted clinical PII
            m = build_composite_gate(readiness, _leak("pass"), dataset_cfg=cfg)
            self.assertEqual(m["composite_status"], "fail")
            self.assertIn("governance_aggregate_only", m["failed_checks"])

    def test_cli_fails_closed_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, readiness = _readiness(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20], min_valid_windows=3)
            rpath = td / "readiness.json"
            write_json(rpath, readiness)
            lpath = td / "leak.json"
            write_json(lpath, _leak("pass"))
            out = td / "composite"
            with self.assertRaises(CompositeGateError):
                driver_main([
                    "--readiness-manifest", str(rpath),
                    "--leakage-audit", str(lpath),
                    "--output-dir", str(out),
                ])
            self.assertTrue((out / "rung-minus1-composite-gate.json").exists())


if __name__ == "__main__":
    unittest.main()
