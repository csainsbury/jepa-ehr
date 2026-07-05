"""Wall-clock target-block tests (Pi spec, rung0_1_run_specs.md 2026-07-05).

Synthetic-only: low-level block builders are exercised on hand-crafted
``cumulative_days`` arrays, and the full ``_real_blocks`` path on tiny synthetic
h5 fixtures. No governed data, no training, no committed data/tokens.

Coverage:
  - half-open interval [t_query, t_query + W) membership (lower-bound INCLUDED,
    upper-bound EXCLUDED)
  - scheduled t_query = context-end day + fixed gap (NOT next event after context)
  - empty-interval ENCODED + flagged, never dropped (direct + real-path rate)
  - boundary-respect: a block must not cross a declared segment/admission boundary
  - monotonicity rejection (negative reset)
  - simultaneous-event determinism (equal cumulative_days => contiguous, stable)
  - common-horizon metadata + schema validity
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import (  # noqa: E402
    MIMIC_TOKEN,
    SCID_TOKEN,
    joint_arms_config,
    joint_dataset_config,
    make_sequence,
    write_h5,
    write_tiny_vocab,
    write_yaml,
)

from clinical_jepa.splits.build_index import main as build_index_main  # noqa: E402
from clinical_jepa.targets.extract_blocks import (  # noqa: E402
    EMPTY_TARGET_REF,
    _real_blocks,
    _resolve_wall_clock_params,
    _t0_wall_clock_block,
    is_monotone_nondecreasing,
    wall_clock_feasible,
)
from clinical_jepa.validation import validate_artifact  # noqa: E402

# For seq_len=22, context_start=2, min_context=8 the deterministic context_end is
# midpoint = (2 + 22) // 2 = 12; for seq_len=20 it is 11; for seq_len=30 it is 16.
CTX_START = 2
MIN_CTX = 8


def _wc_args(arms_cfg_path: str, *, targets=("T0",), segment_channel=None) -> argparse.Namespace:
    return argparse.Namespace(
        arms_config=arms_cfg_path,
        targets=list(targets),
        t0_gap_events=0,
        endpoint_proximal_margin=0,
        max_real_train=0,
        max_real_dev=0,
        max_real_test=0,
        t1_anchors_per_sequence=1,
        source_role="primary",
        unit="wall_clock",
        segment_channel=segment_channel,
    )


def _reference_membership(cdays, context_end: int, t_query: float, window_days: float, seq_len: int):
    """Independent reference: half-open [t_query, t_query+W), indices > context_end."""
    hi = t_query + window_days
    return [i for i in range(context_end + 1, seq_len) if t_query <= float(cdays[i]) < hi]


class HalfOpenMembershipTests(unittest.TestCase):
    def test_lower_bound_included_upper_bound_excluded(self) -> None:
        # indices 0..12 = day 0..12; index13 ties day 12 (== t_query, simultaneous);
        # then 13,14,16 in window; index17 sits exactly on the upper bound (17).
        cdays = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=np.float32,
        )
        block = _t0_wall_clock_block(
            "seq", "dev", len(cdays), "SCID", 0, cdays, window_days=5.0, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX,
        )
        self.assertIsNotNone(block)
        assert block is not None
        ce = block["context_end_ref"]
        self.assertEqual(ce, 12)
        self.assertEqual(block["t_query"], 12.0)
        expected = _reference_membership(cdays, ce, block["t_query"], 5.0, len(cdays))
        self.assertEqual(expected, [13, 14, 15, 16])  # day 12(==t_query) IN, day 17(==hi) OUT
        self.assertEqual(block["target_start_ref"], 13)  # lower-bound tie included
        self.assertEqual(block["target_end_ref"], 16)    # day 16 < 17 included
        self.assertEqual(block["n_target_events"], 4)
        self.assertFalse(block["empty_target"])
        # The event at exactly t_query + W (index 17, day 17) is excluded.
        self.assertNotIn(17, range(block["target_start_ref"], block["target_end_ref"] + 1))


class ScheduledTQueryTests(unittest.TestCase):
    def test_t_query_is_context_end_plus_gap_not_next_event(self) -> None:
        cdays = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=np.float32,
        )
        gap = 2.0
        block = _t0_wall_clock_block(
            "seq", "dev", len(cdays), "SCID", 0, cdays, window_days=5.0, gap_days=gap,
            context_start=CTX_START, min_context=MIN_CTX,
        )
        assert block is not None
        ce = block["context_end_ref"]
        self.assertEqual(ce, 12)
        # Scheduled: t_query = cumulative_days[context_end] + gap = 12 + 2 = 14.
        self.assertEqual(block["t_query"], float(cdays[ce]) + gap)
        self.assertEqual(block["t_query"], 14.0)
        # The immediate next events after context (idx13 day12, idx14 day13) are
        # BEFORE the scheduled t_query and are correctly skipped — proving t_query
        # is scheduled, not "the next event after context".
        self.assertEqual(block["gap_days"], gap)
        self.assertEqual(block["target_start_ref"], 15)  # first index with day >= 14
        expected = _reference_membership(cdays, ce, 14.0, 5.0, len(cdays))
        self.assertEqual(expected, [15, 16, 17, 18])


class EmptyIntervalTests(unittest.TestCase):
    def test_empty_target_encoded_not_dropped_direct(self) -> None:
        cdays = np.arange(22, dtype=np.float32)  # context_end=12, day 12
        block = _t0_wall_clock_block(
            "seq", "dev", len(cdays), "SCID", 0, cdays, window_days=0.5, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX,
        )
        # Block is RETURNED (not None) even though [12, 12.5) contains no event.
        self.assertIsNotNone(block)
        assert block is not None
        self.assertTrue(block["empty_target"])
        self.assertEqual(block["n_target_events"], 0)
        self.assertEqual(block["target_start_ref"], EMPTY_TARGET_REF)
        self.assertEqual(block["target_end_ref"], EMPTY_TARGET_REF)

    def test_empty_target_counted_in_real_path_rate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Per split: one dense SCID (non-empty), one SCID whose post-context
            # days jump far beyond the 90d window (empty), one dense MIMIC.
            empty_cdays = list(range(22)) + [1021.0] * 18  # monotone; jump at idx22
            scid_paths, mimic_paths = {}, {}
            for split in ("train", "dev", "test"):
                scid_groups = {
                    f"scid_{split}_dense": make_sequence(SCID_TOKEN, 40),
                    f"scid_{split}_empty": make_sequence(SCID_TOKEN, 40, cumulative_days=empty_cdays),
                }
                mimic_groups = {f"mimic_{split}_dense": make_sequence(MIMIC_TOKEN, 20)}
                scid_paths[split] = write_h5(td / f"scid_{split}.h5", scid_groups)
                mimic_paths[split] = write_h5(td / f"mimic_{split}.h5", mimic_groups)
            blocks, report = self._extract(td, {"scid": scid_paths, "mimic": mimic_paths})

            wc = report["wall_clock"]
            # 3 splits x 1 empty SCID = 3 empty blocks, all RETAINED + flagged.
            emitted_empty = [b for b in blocks if b.get("empty_target")]
            self.assertEqual(len(emitted_empty), 3)
            self.assertEqual(sum(wc["empty_target_blocks"].values()), 3)
            self.assertGreater(wc["empty_target_rate"], 0.0)
            for b in emitted_empty:
                self.assertEqual(b["target_start_ref"], EMPTY_TARGET_REF)
                self.assertEqual(b["unit"], "wall_clock_days")

    def _extract(self, td: Path, source_paths):
        vocab = write_tiny_vocab(td / "vocab.json")
        idx = td / "idx"
        cfg = joint_dataset_config(td, source_paths, vocab_path=vocab, index_dir=idx)
        dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
        arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
        build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
        split_manifest = {
            "dataset": "joint-test",
            "source_index_paths": {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")},
            "source_h5_paths": {},
        }
        blocks, details = _real_blocks(_wc_args(arms_cfg_path), cfg, split_manifest)
        return blocks, details["report"]


class BoundaryRespectTests(unittest.TestCase):
    def _seq_with_boundary(self):
        # seq_len 30 -> context_end 16 (segment 0). Boundary at index 20.
        cdays = np.arange(30, dtype=np.float32)
        segment_ids = np.array([0] * 20 + [1] * 10, dtype=np.int64)
        return cdays, segment_ids

    def test_block_does_not_cross_declared_boundary(self) -> None:
        cdays, segment_ids = self._seq_with_boundary()
        block = _t0_wall_clock_block(
            "seq", "dev", 30, "SCID", 0, cdays, window_days=90.0, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX, segment_ids=segment_ids,
        )
        assert block is not None
        self.assertEqual(block["context_end_ref"], 16)
        # Window [16, 106) would reach index 29, but the segment boundary at index
        # 20 stops the target: only same-segment indices 17,18,19 are included.
        self.assertEqual(block["target_end_ref"], 19)
        self.assertEqual(block["n_target_events"], 3)
        self.assertTrue(block["boundary_respect"])

    def test_control_without_boundary_crosses(self) -> None:
        cdays, _segment_ids = self._seq_with_boundary()
        block = _t0_wall_clock_block(
            "seq", "dev", 30, "SCID", 0, cdays, window_days=90.0, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX, segment_ids=None,
        )
        assert block is not None
        # No declared boundary => within-sequence default; target spans to the end.
        self.assertEqual(block["target_end_ref"], 29)
        self.assertEqual(block["n_target_events"], 13)
        self.assertFalse(block["boundary_respect"])

    def test_boundary_hook_config_driven_via_real_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cdays, segment_ids = self._seq_with_boundary()
            scid_paths, mimic_paths = {}, {}
            for split in ("train", "dev", "test"):
                scid_groups = {
                    f"scid_{split}_seg": make_sequence(
                        SCID_TOKEN, 30, cumulative_days=cdays, segment_ids=segment_ids
                    )
                }
                mimic_groups = {f"mimic_{split}_dense": make_sequence(MIMIC_TOKEN, 20)}
                scid_paths[split] = write_h5(td / f"scid_{split}.h5", scid_groups)
                mimic_paths[split] = write_h5(td / f"mimic_{split}.h5", mimic_groups)
            source_paths = {"scid": scid_paths, "mimic": mimic_paths}

            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, source_paths, vocab_path=vocab, index_dir=idx)
            dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
            arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
            build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
            split_manifest = {
                "dataset": "joint-test",
                "source_index_paths": {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")},
                "source_h5_paths": {},
            }
            # With the boundary channel configured, SCID blocks stop at the boundary.
            blocks, _ = _real_blocks(
                _wc_args(arms_cfg_path, segment_channel="segment_ids"), cfg, split_manifest
            )
            scid_blocks = [b for b in blocks if b["source_dataset"] == "SCID"]
            self.assertTrue(scid_blocks)
            for b in scid_blocks:
                self.assertEqual(b["target_end_ref"], 19)
                self.assertTrue(b["boundary_respect"])


class MonotonicityTests(unittest.TestCase):
    def test_monotone_helper(self) -> None:
        self.assertTrue(is_monotone_nondecreasing([0, 1, 1, 2, 3]))  # ties allowed
        self.assertFalse(is_monotone_nondecreasing([0, 1, 2, 1, 3]))  # negative reset

    def test_negative_reset_rejected_direct(self) -> None:
        cdays = np.arange(40, dtype=np.float32)
        cdays[25] = 5.0  # negative reset after the context region
        block = _t0_wall_clock_block(
            "seq", "dev", 40, "SCID", 0, cdays, window_days=90.0, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX,
        )
        self.assertIsNone(block)

    def test_negative_reset_counted_in_real_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bad_cdays = list(range(40))
            bad_cdays[25] = 5.0  # reset
            scid_paths, mimic_paths = {}, {}
            for split in ("train", "dev", "test"):
                scid_groups = {
                    f"scid_{split}_good": make_sequence(SCID_TOKEN, 40),
                    f"scid_{split}_bad": make_sequence(SCID_TOKEN, 40, cumulative_days=bad_cdays),
                }
                mimic_groups = {f"mimic_{split}_dense": make_sequence(MIMIC_TOKEN, 20)}
                scid_paths[split] = write_h5(td / f"scid_{split}.h5", scid_groups)
                mimic_paths[split] = write_h5(td / f"mimic_{split}.h5", mimic_groups)
            source_paths = {"scid": scid_paths, "mimic": mimic_paths}
            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, source_paths, vocab_path=vocab, index_dir=idx)
            dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
            arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
            build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
            split_manifest = {
                "dataset": "joint-test",
                "source_index_paths": {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")},
                "source_h5_paths": {},
            }
            blocks, details = _real_blocks(_wc_args(arms_cfg_path), cfg, split_manifest)
            wc = details["report"]["wall_clock"]
            # One bad SCID sequence per split rejected; not silently dropped.
            self.assertEqual(sum(wc["monotonicity_violations"].values()), 3)
            # Only the good SCID sequences (+ MIMIC) yield blocks.
            n_scid = sum(1 for b in blocks if b["source_dataset"] == "SCID")
            self.assertEqual(n_scid, 3)


class SimultaneousEventDeterminismTests(unittest.TestCase):
    def test_simultaneous_run_contiguous_and_deterministic(self) -> None:
        # seq_len 20 -> context_end 11 (day 11). Three simultaneous events at day 13.
        cdays = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 20, 21, 22, 23],
            dtype=np.float32,
        )
        kwargs = dict(context_start=CTX_START, min_context=MIN_CTX)
        block_a = _t0_wall_clock_block(
            "seq", "dev", len(cdays), "SCID", 0, cdays, window_days=5.0, gap_days=0.0, **kwargs
        )
        block_b = _t0_wall_clock_block(
            "seq", "dev", len(cdays), "SCID", 0, cdays, window_days=5.0, gap_days=0.0, **kwargs
        )
        assert block_a is not None and block_b is not None
        self.assertEqual(block_a["context_end_ref"], 11)
        # [11, 16): day12(idx12), day13(idx13,14,15). The full simultaneous run is
        # a contiguous, stably ordered index span.
        self.assertEqual(block_a["target_start_ref"], 12)
        self.assertEqual(block_a["target_end_ref"], 15)
        self.assertEqual(block_a["n_target_events"], 4)
        for i in (13, 14, 15):
            self.assertTrue(block_a["target_start_ref"] <= i <= block_a["target_end_ref"])
        # Determinism: identical inputs => byte-identical block (incl. block_id).
        self.assertEqual(block_a["block_id"], block_b["block_id"])
        self.assertEqual(block_a["target_start_ref"], block_b["target_start_ref"])
        self.assertEqual(block_a["target_end_ref"], block_b["target_end_ref"])


class CommonHorizonMetadataTests(unittest.TestCase):
    def test_block_records_wall_clock_metadata(self) -> None:
        cdays = np.arange(40, dtype=np.float32)
        block = _t0_wall_clock_block(
            "seq", "dev", 40, "SCID", 0, cdays, window_days=90.0, gap_days=1.0,
            context_start=CTX_START, min_context=MIN_CTX, common_horizons=[30.0, 90.0],
        )
        assert block is not None
        for key in ("unit", "window_days", "gap_days", "t_query", "empty_target", "source_dataset"):
            self.assertIn(key, block)
        self.assertEqual(block["unit"], "wall_clock_days")
        self.assertEqual(block["window_days"], 90.0)
        self.assertEqual(block["gap_days"], 1.0)
        self.assertEqual(block["source_dataset"], "SCID")
        # 90 is one of the common horizons; a non-common window is flagged False.
        self.assertTrue(block["is_common_horizon"])
        self.assertEqual(block["common_horizons_days"], [30.0, 90.0])
        off = _t0_wall_clock_block(
            "seq", "dev", 40, "SCID", 0, cdays, window_days=45.0, gap_days=0.0,
            context_start=CTX_START, min_context=MIN_CTX, common_horizons=[30.0, 90.0],
        )
        assert off is not None
        self.assertFalse(off["is_common_horizon"])

    def test_resolve_params_source_specific_windows_plus_common_horizons(self) -> None:
        common = joint_arms_config()["common"]
        scid = _resolve_wall_clock_params(common, "T0", "SCID")
        mimic = _resolve_wall_clock_params(common, "T0", "MIMIC")
        # Source-specific windows for yield; shared common horizons for the
        # cross-source hierarchy comparison (rung 0).
        self.assertEqual(scid["window_days"], 90.0)
        self.assertEqual(mimic["window_days"], 5.0)
        self.assertEqual(scid["common_horizons_days"], [30.0, 90.0])
        self.assertEqual(mimic["min_context"], 4)

    def test_shipped_example_arms_config_resolves(self) -> None:
        # The example arms config's wall_clock shape must match the resolver
        # (per_source at the T0 level, wall_clock nested under each source).
        from clinical_jepa.utils import load_yaml

        arms_path = Path(__file__).resolve().parents[1] / "configs" / "v0" / "arms.example.yaml"
        common = load_yaml(str(arms_path)).get("common", {})
        scid = _resolve_wall_clock_params(common, "T0", "SCID")
        mimic = _resolve_wall_clock_params(common, "T0", "MIMIC")
        self.assertEqual(scid["window_days"], 90.0)
        self.assertEqual(mimic["window_days"], 7.0)
        self.assertEqual(mimic["min_context"], 4)
        self.assertEqual(scid["common_horizons_days"], [30.0, 90.0, 180.0])

    def test_wall_clock_manifest_validates_against_schema(self) -> None:
        cdays = np.arange(40, dtype=np.float32)
        blocks = []
        for i, w in enumerate((30.0, 90.0)):
            b = _t0_wall_clock_block(
                f"seq{i}", "dev", 40, "SCID", 0, cdays, window_days=w, gap_days=0.0,
                context_start=CTX_START, min_context=MIN_CTX, common_horizons=[30.0, 90.0],
            )
            assert b is not None
            blocks.append(b)
        manifest = {
            "schema_version": "clinical-jepa-target-block-manifest-v0",
            "created_utc": "2026-07-05T00:00:00Z",
            "targets": ["T0"],
            "counts": {"dev": {"T0": len(blocks)}},
            "blocks": blocks,
        }
        errors = validate_artifact("target-block-manifest", manifest, raise_on_error=False)
        self.assertEqual(errors, [])


class FeasibilityTests(unittest.TestCase):
    def test_wall_clock_feasible_requires_context_plus_future_slot(self) -> None:
        # Enough context + a future slot.
        self.assertTrue(wall_clock_feasible(20, context_start=2, min_context=4))
        # Too short: lowest context end exceeds the last future-eligible index.
        self.assertFalse(wall_clock_feasible(9, context_start=2, min_context=8))


if __name__ == "__main__":
    unittest.main()
