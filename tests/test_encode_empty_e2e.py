"""End-to-end empty-heavy audit (Pi R4 required change #3): before any real
wall-clock use, drive the whole encode-empty stack on empty-heavy synthetic data —
extract wall-clock blocks (with empties) -> leakage audit (NO false horizon_boundary
on empties) -> encode-empty training (prototype frozen) -> composite gate."""
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


def _wc_args(arms_cfg_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        arms_config=arms_cfg_path, targets=["T0"], t0_gap_events=0, endpoint_proximal_margin=0,
        max_real_train=0, max_real_dev=0, max_real_test=0, t1_anchors_per_sequence=1,
        source_role="primary", unit="wall_clock", segment_channel=None,
    )


def _ee_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_blocks=0, max_context_tokens=32, max_target_tokens=16, embedding_dim=16,
        autoregression_mode="recursive", max_horizons=1, empty_prototype_seed=0,
        learning_rate=5e-3, empty_fraction_cap=0.5, batch_size=8, real_steps=20, cpu=True,
    )


@unittest.skipUnless(HAS_TORCH_H5PY, "torch and h5py required")
class EncodeEmptyEndToEndTests(unittest.TestCase):
    def test_empty_heavy_pipeline(self) -> None:
        from clinical_jepa.arms.v0b.train_minimal_jepa import _encode_empty_run
        from clinical_jepa.audit.run_leakage_audit import main as audit_main
        from clinical_jepa.splits.build_index import main as build_index_main
        from clinical_jepa.targets.extract_blocks import _real_blocks

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # SCID: sparse events every 200 days -> the 90d window is EMPTY (silence,
            # fully observed). MIMIC: dense (arange days) -> the 5d window is populated.
            sparse = [i * 200.0 for i in range(30)]
            scid_paths, mimic_paths = {}, {}
            for split in ("train", "dev", "test"):
                scid_groups = {
                    f"scid_{split}_{i}": make_sequence(SCID_TOKEN, 30, cumulative_days=sparse)
                    for i in range(8)
                }
                mimic_groups = {f"mimic_{split}_{i}": make_sequence(MIMIC_TOKEN, 30) for i in range(8)}
                scid_paths[split] = write_h5(td / f"scid_{split}.h5", scid_groups)
                mimic_paths[split] = write_h5(td / f"mimic_{split}.h5", mimic_groups)

            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, {"scid": scid_paths, "mimic": mimic_paths}, vocab_path=vocab, index_dir=idx)
            ds_path = write_yaml(td / "dataset.yaml", cfg)
            arms_path = write_yaml(td / "arms.yaml", joint_arms_config())
            build_index_main(["--dataset-config", ds_path, "--output-dir", str(idx)])
            split_manifest = {
                "dataset": "joint-test",
                "source_index_paths": {s: str(idx / f"{s}.index.jsonl") for s in ("train", "dev", "test")},
                "source_h5_paths": {},
            }

            # 1) Extract wall-clock blocks -> mix of empty (SCID) + populated (MIMIC).
            blocks, details = _real_blocks(_wc_args(arms_path), cfg, split_manifest)
            n_empty = sum(1 for b in blocks if b.get("empty_target"))
            n_pop = sum(1 for b in blocks if not b.get("empty_target"))
            self.assertGreater(n_empty, 0, "expected empty (silence) blocks")
            self.assertGreater(n_pop, 0, "expected populated blocks")
            self.assertGreater(details["report"]["wall_clock"]["empty_target_rate"], 0.0)

            # 2) Leakage audit on the empty-heavy manifest: NO false horizon_boundary.
            sm = td / "split.json"; sm.write_text(json.dumps({"dataset": "joint-test"}))
            tb = td / "target-blocks.json"
            tb.write_text(json.dumps({"targets": ["T0"], "blocks": blocks}))
            audit_out = td / "leak.json"
            audit_main(["--dataset-config", ds_path, "--split-manifest", str(sm),
                        "--target-blocks", str(tb), "--output", str(audit_out)])
            leak = json.loads(audit_out.read_text())
            self.assertEqual(leak["audits"]["horizon_boundary"]["status"], "pass")  # the bug we fixed
            self.assertEqual(leak["overall_status"], "pass")

            # 3) Encode-empty training consumes the empties (routed to z_empty); the
            #    frozen prototype must not move.
            out = td / "ee"; out.mkdir()
            rc = _encode_empty_run(_ee_args(), {}, {"run": {"seed": 1}, "vocabulary": {"vocab_size": 1050}}, {"blocks": blocks}, out)
            self.assertEqual(rc, 0)
            diag = json.loads((out / "collapse-diagnostics.json").read_text())
            self.assertTrue(diag["prototype_unmoved_during_training"])
            self.assertGreater(json.loads((out / "train-manifest.json").read_text())["n_train_empty"], 0)


if __name__ == "__main__":
    unittest.main()
