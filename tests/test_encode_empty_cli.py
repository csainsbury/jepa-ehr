"""Encode-empty CLI smoke test: _encode_empty_run drives the full path (helper-
routed reader -> hybrid training -> manifests) on synthetic wall-clock blocks,
writing valid artifacts with the frozen prototype unmoved. The LEARNING invariants
are covered separately by test_encode_empty_train."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH_H5PY = (
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("h5py") is not None
)

from synth_fixtures import write_h5  # noqa: E402


def _args(**over) -> argparse.Namespace:
    base = dict(
        max_blocks=0, max_context_tokens=32, max_target_tokens=16, embedding_dim=16,
        autoregression_mode="recursive", max_horizons=1, empty_prototype_seed=0,
        learning_rate=5e-3, empty_fraction_cap=0.5, batch_size=8, real_steps=20, cpu=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


@unittest.skipUnless(HAS_TORCH_H5PY, "torch and h5py required")
class EncodeEmptyCliTests(unittest.TestCase):
    def _blocks_and_h5(self, td: Path):
        # 12 groups of 30 small tokens (< vocab 64); a mix of populated + empty
        # wall-clock T0 blocks across train/dev.
        groups = {f"g{i}": {"token_ids": np.array([1] + list(range(10, 39)), dtype=np.int64)} for i in range(12)}
        h5_path = write_h5(td / "seqs.h5", groups)
        blocks = []
        i = 0
        for split, n_each in (("train", 4), ("dev", 2)):
            for _ in range(n_each):
                # populated
                blocks.append({
                    "block_id": f"p{i}", "sequence_file": h5_path, "sequence_group": f"g{i % 12}",
                    "split": split, "target_type": "T0", "source_dataset": "SCID", "unit": "wall_clock_days",
                    "context_start_ref": 2, "context_end_ref": 12, "target_start_ref": 13, "target_end_ref": 20,
                    "empty_target": False, "censored": False, "n_target_events": 8,
                })
                i += 1
                # empty (silence)
                blocks.append({
                    "block_id": f"e{i}", "sequence_file": h5_path, "sequence_group": f"g{i % 12}",
                    "split": split, "target_type": "T0", "source_dataset": "SCID", "unit": "wall_clock_days",
                    "context_start_ref": 2, "context_end_ref": 12, "target_start_ref": -1, "target_end_ref": -1,
                    "empty_target": True, "censored": False, "n_target_events": 0,
                })
                i += 1
        return {"targets": ["T0"], "blocks": blocks}

    def test_encode_empty_run_writes_valid_artifacts(self) -> None:
        from clinical_jepa.arms.v0b.train_minimal_jepa import _encode_empty_run
        from clinical_jepa.validation import validate_artifact

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            targets = self._blocks_and_h5(td)
            dataset = {"run": {"seed": 20260523}, "vocabulary": {"vocab_size": 64}}
            outdir = td / "out"
            outdir.mkdir()
            rc = _encode_empty_run(_args(), {}, dataset, targets, outdir)
            self.assertEqual(rc, 0)

            tm = json.loads((outdir / "train-manifest.json").read_text())
            self.assertEqual(validate_artifact("v0b-train-manifest", tm, raise_on_error=False), [])
            self.assertTrue(tm["architecture"]["encode_empty"])
            self.assertGreater(tm["n_train_empty"], 0)

            diag = json.loads((outdir / "collapse-diagnostics.json").read_text())
            self.assertEqual(diag["mode"], "encode_empty")
            self.assertTrue(diag["prototype_unmoved_during_training"])  # frozen buffer never moved
            self.assertIn("separation_check", diag)
            self.assertIn("occupancy_auc", diag["collapse_diagnostics"])
            self.assertIn("brier", diag["calibration_natural_prevalence"])
            self.assertTrue((outdir / "minimal-jepa-v0b-encode-empty.pt").exists())

    def test_censored_blocks_excluded_from_examples(self) -> None:
        from clinical_jepa.arms.v0b.train_minimal_jepa import _read_encode_empty_examples

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            targets = self._blocks_and_h5(td)
            # Add a censored block; it must not become a training example.
            targets["blocks"].append({
                "block_id": "cens", "sequence_file": str(next(iter({b["sequence_file"] for b in targets["blocks"]}))),
                "sequence_group": "g0", "split": "train", "target_type": "T0", "source_dataset": "SCID",
                "context_start_ref": 2, "context_end_ref": 12, "target_start_ref": -1, "target_end_ref": -1,
                "empty_target": False, "censored": True, "n_target_events": 0,
            })
            examples = _read_encode_empty_examples(targets["blocks"], 0, 32, 16, seed=0)
            self.assertTrue(examples)
            # No example may originate from the censored block (all are p*/e* blocks).
            n_empty = sum(1 for e in examples if e[3])
            n_pop = sum(1 for e in examples if not e[3])
            self.assertEqual(n_empty, 6)   # 6 empty blocks (4 train + 2 dev)
            self.assertEqual(n_pop, 6)     # censored excluded -> exactly the 6 populated


if __name__ == "__main__":
    unittest.main()
