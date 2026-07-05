"""Fail-hard guard tests (Pi round 2 hardening).

Each test proves a guard *fails* rather than passing by absence:
  - is_outcome configured-but-unchecked (blocks_checked == 0) -> FAIL
  - is_outcome endpoint-facing target/eval-span leak -> FAIL (context clean)
  - source mask required but source_prefix_len < 2 / not_configured -> FAIL
  - the actual v0B dataloader tensor slice excludes seq positions 0-1 / source token
  - missing expected source or required split -> ReadinessGateError (fail-closed)
  - source-prediction probe is alarm/report-only, never a pass/fail gate

Synthetic data only (no governed h5).
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
    BENIGN_TOKEN,
    BOS,
    MIMIC_TOKEN,
    SCID_TOKEN,
    build_source_h5s,
    joint_arms_config,
    joint_dataset_config,
    make_sequence,
    write_h5,
    write_tiny_vocab,
    write_yaml,
)

from clinical_jepa.audit.run_leakage_audit import main as audit_main  # noqa: E402
from clinical_jepa.splits.build_index import main as build_index_main  # noqa: E402
from clinical_jepa.splits.readiness_manifest import (  # noqa: E402
    ReadinessGateError,
    assert_gate_or_raise,
    build_readiness_manifest,
)
from clinical_jepa.targets.extract_blocks import _real_blocks  # noqa: E402


def _block(block_id: str, c0: int, c1: int, t0: int, t1: int, *, h5_path=None, group=None, **extra) -> dict:
    b = {
        "block_id": block_id,
        "patient_hash": "h" + block_id,
        "sequence_id": group or block_id,
        "sequence_group": group,
        "split": "train",
        "target_type": "T0",
        "context_start_ref": c0,
        "context_end_ref": c1,
        "target_start_ref": t0,
        "target_end_ref": t1,
        "horizon_descriptor": "event_gap_0_window_8",
        "source_dataset": "SCID",
    }
    if h5_path is not None:
        b["sequence_file"] = h5_path
    b.update(extra)
    return b


def _write_manifest(path: Path, blocks: list[dict]) -> None:
    path.write_text(json.dumps({
        "schema_version": "clinical-jepa-target-block-manifest-v0",
        "created_utc": "2026-07-05T00:00:00Z",
        "targets": ["T0"],
        "counts": {"train": {"T0": len(blocks)}},
        "blocks": blocks,
    }))


def _run_audit(root: Path, cfg_path: str, blocks: list[dict], *, extra_argv: list[str] | None = None):
    manifest = root / "blocks.json"
    _write_manifest(manifest, blocks)
    split = root / "split.json"
    split.write_text(json.dumps({"dataset": "joint-test"}))
    out = root / "audit.json"
    argv = [
        "--dataset-config", cfg_path,
        "--split-manifest", str(split),
        "--target-blocks", str(manifest),
        "--output", str(out),
    ] + (extra_argv or [])
    rc = audit_main(argv)
    return rc, json.loads(out.read_text())


class OutcomeAuditFailHardTests(unittest.TestCase):
    def test_configured_channel_but_no_readable_channel_fails(self) -> None:
        # Block references a real sequence, channel configured, but the group has
        # NO is_outcome_label dataset -> blocks_checked == 0 -> FAIL (not pass).
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"source_prefix_len": 2},
                "leakage": {"outcome_label_dataset": "is_outcome_label"},
            })
            arr = np.full(40, BENIGN_TOKEN, dtype=np.int64)
            arr[0] = SCID_TOKEN
            arr[1] = BOS
            h5 = write_h5(root / "seq.h5", {"g": {"token_ids": arr, "time_deltas": np.ones(40, dtype=np.float32)}})
            rc, report = _run_audit(root, cfg, [_block("no_channel", 2, 10, 11, 18, h5_path=h5, group="g")])
            self.assertEqual(rc, 2)
            self.assertEqual(report["audits"]["label_feature_separation"]["status"], "fail")
            self.assertEqual(report["outcome_label_separation"]["blocks_checked"], 0)
            self.assertEqual(report["outcome_label_separation"]["mode"], "h5_channel")

    def test_configured_channel_but_no_sequences_or_annotations_fails(self) -> None:
        # Configured channel, blocks carry neither sequence refs nor annotations
        # -> unverifiable -> FAIL.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"source_prefix_len": 2},
                "leakage": {"outcome_label_dataset": "is_outcome_label"},
            })
            rc, report = _run_audit(root, cfg, [_block("bare", 2, 10, 11, 18)])
            self.assertEqual(rc, 2)
            self.assertEqual(report["audits"]["label_feature_separation"]["status"], "fail")
            self.assertEqual(report["outcome_label_separation"]["mode"], "unverifiable")
            self.assertEqual(report["outcome_label_separation"]["blocks_checked"], 0)

    def test_endpoint_facing_target_span_leak_is_caught(self) -> None:
        # is_outcome==1 inside the TARGET span [11,18] (context [2,10] clean).
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"source_prefix_len": 2},
                "leakage": {"outcome_label_dataset": "is_outcome_label"},
            })
            h5 = write_h5(root / "seq.h5", {"g": make_sequence(SCID_TOKEN, 40, outcome_at=[15])})
            blk = [_block("tgt_leak", 2, 10, 11, 18, h5_path=h5, group="g")]

            # Non-endpoint-facing: context+margin scan is clean -> PASS.
            rc, report = _run_audit(root, cfg, blk)
            self.assertEqual(rc, 0)
            self.assertEqual(report["audits"]["label_feature_separation"]["status"], "pass")

            # Endpoint-facing: target span scanned -> FAIL.
            rc2, report2 = _run_audit(root, cfg, blk, extra_argv=["--endpoint-facing"])
            self.assertEqual(rc2, 2)
            self.assertEqual(report2["audits"]["label_feature_separation"]["status"], "fail")
            self.assertGreaterEqual(report2["outcome_label_separation"]["target_span_leaked_positions"], 1)
            self.assertTrue(report2["endpoint_facing"])


class SourceMaskFailHardTests(unittest.TestCase):
    def _clean_block_seq(self, root: Path):
        h5 = write_h5(root / "seq.h5", {"g": make_sequence(SCID_TOKEN, 40)})
        return [_block("clean", 2, 10, 11, 18, h5_path=h5, group="g")]

    def test_required_but_prefix_too_short_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"source_prefix_len": 1, "require_source_mask": True},
            })
            rc, report = _run_audit(root, cfg, self._clean_block_seq(root))
            self.assertEqual(rc, 2)
            self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "fail")
            self.assertEqual(report["source_shortcut"]["source_prefix_len"], 1)
            self.assertTrue(report["source_shortcut"]["required"])

    def test_required_but_not_configured_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"require_source_mask": True},  # no source_prefix_len
            })
            rc, report = _run_audit(root, cfg, self._clean_block_seq(root))
            self.assertEqual(rc, 2)
            self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "fail")
            self.assertFalse(report["source_shortcut"]["configured"])

    def test_required_and_prefix_two_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {
                "mask": {"source_prefix_len": 2, "require_source_mask": True},
            })
            rc, report = _run_audit(root, cfg, self._clean_block_seq(root))
            self.assertEqual(rc, 0)
            self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "pass")

    def test_not_required_absent_mask_still_not_configured(self) -> None:
        # Backward compatibility: without require_source_mask an absent mask is
        # not_configured (the dry-run scaffold path), not a failure.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = write_yaml(root / "dataset.yaml", {})
            rc, report = _run_audit(root, cfg, self._clean_block_seq(root))
            self.assertEqual(rc, 0)
            self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "not_configured")


class DataloaderTensorMaskTests(unittest.TestCase):
    """Assert the ACTUAL v0B dataloader tensor slice excludes seq idx 0-1."""

    def test_read_examples_tensors_exclude_source_prefix(self) -> None:
        from clinical_jepa.arms.v0b.train_minimal_jepa import _read_examples

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = build_source_h5s(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], med_at=[100])
            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx)
            dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
            arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
            build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])

            split_manifest = {
                "dataset": "joint-test",
                "source_index_paths": {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")},
                "source_h5_paths": {},
            }
            args = argparse.Namespace(
                arms_config=arms_cfg_path, targets=["T0"], t0_gap_events=0,
                endpoint_proximal_margin=0, max_real_train=0, max_real_dev=0,
                max_real_test=0, t1_anchors_per_sequence=1, source_role="primary",
            )
            blocks, _details = _real_blocks(args, cfg, split_manifest)
            self.assertTrue(blocks)

            # Load the real dataloader input tensors (context + targets).
            examples = _read_examples(blocks, 0, 128, 4, seed=0, horizon_count=1)
            self.assertTrue(examples)
            forbidden = {SCID_TOKEN, MIMIC_TOKEN, BOS}
            for _split, ctx, targets in examples:
                self.assertGreater(len(ctx), 0)
                # In the fixtures the source token lives only at seq idx 0 and
                # [BOS] only at idx 1, so their absence in the tensor proves the
                # encoder/predictor input never sees positions 0-1 / source meta.
                self.assertFalse(forbidden & set(int(x) for x in ctx.tolist()))
                for t in targets:
                    self.assertFalse(forbidden & set(int(x) for x in t.tolist()))


class ReadinessMissingFailClosedTests(unittest.TestCase):
    def _prepare(self, td: Path, *, min_valid_windows=2):
        paths = build_source_h5s(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], med_at=[100])
        vocab = write_tiny_vocab(td / "vocab.json")
        idx = td / "idx"
        cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx, min_valid_windows=min_valid_windows)
        dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
        arms = joint_arms_config()
        build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
        index_paths = {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")}
        return cfg, arms, index_paths, idx

    def test_missing_required_split_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, arms, index_paths, _ = self._prepare(td)
            # Drop the 'test' split entirely from the passed index paths.
            partial = {"train": index_paths["train"], "dev": index_paths["dev"]}
            manifest = build_readiness_manifest(cfg, arms, partial)
            self.assertEqual(manifest["gate_status"], "fail")
            self.assertTrue(any(m["split"] == "test" for m in manifest["missing"]))
            with self.assertRaises(ReadinessGateError):
                assert_gate_or_raise(manifest)

    def test_missing_source_in_a_split_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, arms, index_paths, _ = self._prepare(td)
            # Rewrite dev index to SCID-only (MIMIC absent from dev).
            dev = Path(index_paths["dev"])
            scid_only = [
                line for line in dev.read_text().splitlines()
                if line and json.loads(line).get("source_dataset") == "SCID"
            ]
            dev.write_text("\n".join(scid_only) + "\n")
            manifest = build_readiness_manifest(cfg, arms, index_paths)
            self.assertEqual(manifest["gate_status"], "fail")
            self.assertTrue(any(m["source"] == "MIMIC" and m["split"] == "dev" for m in manifest["missing"]))
            with self.assertRaises(ReadinessGateError):
                assert_gate_or_raise(manifest)


class ReadinessMatchedWindowTests(unittest.TestCase):
    def test_matched_counts_below_feasible_and_action_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Two SCID sequences share a (length, rate) group; the third (much
            # longer) is a singleton -> matched (2) < feasible (3).
            paths = build_source_h5s(td, scid_lens=[220, 220, 700], mimic_lens=[20, 20, 20], med_at=[100])
            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx, min_valid_windows=2, min_matched_candidates=2)
            dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
            build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
            index_paths = {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")}

            manifest = build_readiness_manifest(cfg, joint_arms_config(), index_paths)
            scid_train = manifest["per_source"]["SCID"]["train"]
            self.assertEqual(scid_train["feasible_windows"], 3)
            self.assertEqual(scid_train["valid_matched_windows"], 2)
            self.assertAlmostEqual(scid_train["matched_candidate_yield"], 2 / 3, places=5)
            # Candidate-action frequency surfaced from the index (MED planted at 100).
            self.assertTrue(scid_train["candidate_action_available"])
            self.assertEqual(scid_train["candidate_action_frequency"], 1.0)
            self.assertGreaterEqual(scid_train["candidate_action_mean_per_sequence"], 1.0)
            # MIMIC (len 20) has no candidate action in range 100.
            mimic_train = manifest["per_source"]["MIMIC"]["train"]
            self.assertEqual(mimic_train["candidate_action_frequency"], 0.0)


class SourceProbeAlarmOnlyTests(unittest.TestCase):
    def test_probe_marked_report_only_and_never_gates(self) -> None:
        from clinical_jepa.eval.source_probe import main as probe_main
        from clinical_jepa.eval.source_probe import source_prediction_probe

        rng = np.random.default_rng(0)
        n = 120
        sources = ["SCID"] * (n // 2) + ["MIMIC"] * (n // 2)
        shift = np.array([[3.0 if s == "SCID" else -3.0] for s in sources])
        latents = (rng.normal(size=(n, 6)) * 0.3 + shift).astype("float32")
        report = source_prediction_probe(latents, sources, seed=1)
        self.assertEqual(report["role"], "alarm_report_only")
        self.assertFalse(report["is_pass_fail_gate"])
        self.assertFalse(report["near_base_rate_required"])
        # Highly predictable source (leaky) must NOT cause a non-zero exit: the
        # probe is a report, not a gate.
        self.assertFalse(report["near_base_rate"])

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            np.save(td / "lat.npy", latents)
            (td / "index.jsonl").write_text("".join(json.dumps({"source_dataset": s}) + "\n" for s in sources))
            rc = probe_main([
                "--latents", str(td / "lat.npy"),
                "--index", str(td / "index.jsonl"),
                "--output-dir", str(td / "out"),
            ])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
