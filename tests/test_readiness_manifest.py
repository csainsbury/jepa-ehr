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
from clinical_jepa.splits.readiness_manifest import (  # noqa: E402
    ReadinessGateError,
    assert_gate_or_raise,
    build_readiness_manifest,
)
from clinical_jepa.splits.readiness_manifest import main as readiness_main  # noqa: E402


def _prepare(td: Path, *, scid_lens, mimic_lens, min_valid_windows):
    paths = build_source_h5s(td, scid_lens=scid_lens, mimic_lens=mimic_lens)
    vocab = write_tiny_vocab(td / "vocab.json")
    idx = td / "idx"
    cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx, min_valid_windows=min_valid_windows)
    dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
    arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
    build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])
    index_paths = {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")}
    return cfg, joint_arms_config(), index_paths, dataset_cfg_path, arms_cfg_path, idx


class ReadinessManifestTests(unittest.TestCase):
    def test_gate_passes_with_adequate_windows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg, arms, index_paths, *_ = _prepare(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20], min_valid_windows=2)
            manifest = build_readiness_manifest(cfg, arms, index_paths)
            self.assertEqual(manifest["gate_status"], "pass")
            # Per-source, per-split structure with quantiles + block yield.
            self.assertIn("SCID", manifest["per_source"])
            self.assertIn("MIMIC", manifest["per_source"])
            mimic_train = manifest["per_source"]["MIMIC"]["train"]
            self.assertEqual(mimic_train["n_sequences"], 3)
            self.assertEqual(mimic_train["valid_matched_windows"], 3)
            self.assertIn("median", mimic_train["token_count_quantiles"])
            self.assertIn("primary_T0", mimic_train["block_yield"])
            # Short MIMIC yields nothing under the 32-event window.
            self.assertEqual(mimic_train["block_yield"].get("event_window_32", 0), 0)
            assert_gate_or_raise(manifest)  # does not raise

    def test_gate_fails_closed_when_source_under_contributes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Only 2 MIMIC sequences per split, floor 3 -> MIMIC under-contributes.
            cfg, arms, index_paths, *_ = _prepare(td, scid_lens=[220, 220, 220], mimic_lens=[20, 20], min_valid_windows=3)
            manifest = build_readiness_manifest(cfg, arms, index_paths)
            self.assertEqual(manifest["gate_status"], "fail")
            self.assertTrue(any(u["source"] == "MIMIC" for u in manifest["under_floor"]))
            with self.assertRaises(ReadinessGateError):
                assert_gate_or_raise(manifest)

    def test_cli_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _cfg, _arms, _index_paths, dataset_cfg_path, arms_cfg_path, idx = _prepare(
                td, scid_lens=[220, 220, 220], mimic_lens=[20, 20], min_valid_windows=3,
            )
            out = td / "readiness"
            with self.assertRaises(ReadinessGateError):
                readiness_main([
                    "--dataset-config", dataset_cfg_path,
                    "--arms-config", arms_cfg_path,
                    "--index-dir", str(idx),
                    "--output-dir", str(out),
                ])
            # Diagnostic manifest is still persisted before failing closed.
            self.assertTrue((out / "rung-minus1-readiness-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
