"""Wall-clock readiness tests (Pi R4/R3 Step 7): per source x horizon empty/censored
split + the composite rung -1 driver hook that gates the wall-clock rung."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_H5PY = importlib.util.find_spec("h5py") is not None

from synth_fixtures import build_source_h5s, joint_arms_config, joint_dataset_config, write_tiny_vocab  # noqa: E402

from clinical_jepa.splits.rung_minus1_driver import build_composite_gate  # noqa: E402
from clinical_jepa.validation import validate_artifact  # noqa: E402


def _wc_manifest(adequate: dict) -> dict:
    """A minimal schema-valid wall-clock readiness manifest for the driver hook."""
    return {
        "schema_version": "clinical-jepa-wallclock-readiness-v0",
        "created_utc": "2026-07-09T00:00:00Z", "split": "dev", "horizons_days": [1.0, 30.0],
        "per_source": {s: {"per_horizon": {}} for s in adequate},
        "adequate_horizons_by_source": adequate, "aggregate_only": True,
    }


def _readiness_pass():
    return {
        "schema_version": "clinical-jepa-rung-minus1-readiness-v0", "created_utc": "x",
        "gate_status": "pass", "min_valid_windows_per_source_per_split": 500,
        "source_prefix_len": 2, "split_counts": {}, "per_source": {"SCID": {}, "MIMIC": {}},
        "under_floor": [], "missing": [], "aggregate_only": True,
    }


_LEAK_KEYS = ("patient_overlap", "window_inheritance", "horizon_boundary", "forbidden_tokens",
              "cached_embeddings", "duplicate_windows", "label_feature_separation")


def _leak_pass():
    audits = {k: {"status": "pass", "violations": 0} for k in _LEAK_KEYS}
    return {"schema_version": "clinical-jepa-leakage-audit-v0", "created_utc": "x", "dataset": "d",
            "target_blocks": ["T0"], "audits": audits, "overall_status": "pass",
            "blocks_by_source": {"SCID": 5, "MIMIC": 5}, "aggregate_only": True}


@unittest.skipUnless(HAS_H5PY, "h5py required")
class WallclockReadinessBuildTests(unittest.TestCase):
    def test_empty_populated_saturated_split(self) -> None:
        from clinical_jepa.splits.wallclock_readiness import build_wallclock_readiness

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = build_source_h5s(td, scid_lens=[30] * 10, mimic_lens=[30] * 10)
            vocab = write_tiny_vocab(td / "vocab.json")
            cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=td / "idx")
            # cumulative_days = arange(len) => context_end 16; W=1 empty, W=5 populated,
            # W=30 saturated (gap=0 => censored is always 0, consistent with Step 1).
            m = build_wallclock_readiness(cfg, joint_arms_config(), "dev", [1.0, 5.0, 30.0], floor=2)
            self.assertEqual(validate_artifact("wallclock-readiness", m, raise_on_error=False), [])
            scid = m["per_source"]["SCID"]["per_horizon"]
            self.assertGreater(scid["1.0"]["empty_target_windows"], 0)     # W=1 empty
            self.assertEqual(scid["1.0"]["censored_target_windows"], 0)    # gap=0 => no censoring
            self.assertGreater(scid["5.0"]["nonempty_target_windows"], 0)  # W=5 populated
            self.assertGreaterEqual(scid["5.0"]["nonempty_median_occupancy"], 2)
            self.assertGreater(scid["30.0"]["saturated_target_rate"], 0.5)  # W=30 saturated
            self.assertFalse(scid["30.0"]["adequate"])                     # saturated -> not adequate
            self.assertIn("SCID", m["adequate_horizons_by_source"])


class CompositeWallclockHookTests(unittest.TestCase):
    def test_composite_passes_with_adequate_wallclock(self) -> None:
        wc = _wc_manifest({"SCID": [30.0], "MIMIC": [1.0]})
        m = build_composite_gate(_readiness_pass(), _leak_pass(), wallclock_readiness=wc)
        # (dataset_cfg omitted -> provenance fails, but wallclock component must be pass.)
        self.assertEqual(m["component_status"]["wallclock_readiness"], "pass")
        self.assertIn("wallclock_horizons_adequate", [c["name"] for c in m["checks"]])

    def test_composite_fails_when_a_source_has_no_adequate_horizon(self) -> None:
        wc = _wc_manifest({"SCID": [30.0], "MIMIC": []})  # MIMIC: no usable horizon
        m = build_composite_gate(_readiness_pass(), _leak_pass(), wallclock_readiness=wc)
        self.assertEqual(m["component_status"]["wallclock_readiness"], "fail")
        self.assertIn("wallclock_horizons_adequate", m["failed_checks"])

    def test_wallclock_not_provided_is_neutral(self) -> None:
        m = build_composite_gate(_readiness_pass(), _leak_pass())
        self.assertEqual(m["component_status"]["wallclock_readiness"], "not_provided")


if __name__ == "__main__":
    unittest.main()
