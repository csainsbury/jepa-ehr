from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH_H5PY = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("h5py") is not None


@unittest.skipUnless(HAS_TORCH_H5PY, "torch and h5py are required for rollout export tests")
class ExportMeanTokenRolloutsTests(unittest.TestCase):
    def test_cli_exports_local_sidecar_without_ids(self) -> None:
        import h5py
        import torch
        import torch.nn as nn

        class MeanJEPA(nn.Module):
            def __init__(self, vocab: int, d: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab, d, padding_idx=0)
                self.predictor = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            h5_path = root / "seq.h5"
            with h5py.File(h5_path, "w") as h5:
                grp = h5.create_group("seq0")
                grp.create_dataset("token_ids", data=np.arange(1, 121, dtype=np.int64) % 20 + 1)
            manifest = {
                "blocks": [{
                    "block_id": "block-secret-0",
                    "patient_hash": "patient-secret-0",
                    "split": "dev",
                    "target_type": "T0",
                    "sequence_file": str(h5_path),
                    "sequence_group": "seq0",
                    "context_start_ref": 0,
                    "context_end_ref": 15,
                    "target_start_ref": 16,
                    "target_end_ref": 31,
                    "horizon_descriptor": "synthetic",
                    "source_dataset": "synthetic",
                }]
            }
            manifest_path = root / "target-blocks.json"
            manifest_path.write_text(json.dumps(manifest))
            ckpt_path = root / "ckpt.pt"
            model = MeanJEPA(32, 8)
            torch.save({"model_state_dict": model.state_dict(), "vocab_size": 32, "embedding_dim": 8}, ckpt_path)
            out = root / "out"
            subprocess.run([
                sys.executable,
                "-m",
                "clinical_jepa.eval.export_mean_token_rollouts",
                "--checkpoint",
                str(ckpt_path),
                "--target-blocks",
                str(manifest_path),
                "--output-dir",
                str(out),
                "--splits",
                "dev",
                "--target-types",
                "T0",
                "--horizon-count",
                "2",
                "--target-window-events",
                "8",
                "--horizon-stride-events",
                "8",
            ], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report = json.loads((out / "rollout-export-manifest.json").read_text())
            self.assertEqual(report["aggregate_only"], True)
            self.assertEqual(report["rows_exported"], 1)
            pred = np.load(out / "predicted-rollout.fp16.npy")
            obs = np.load(out / "observed-rollout.fp16.npy")
            self.assertEqual(pred.shape, (1, 2, 8))
            self.assertEqual(obs.shape, (1, 2, 8))
            sidecar = (out / "rollout-index.local.jsonl").read_text()
            self.assertNotIn("block-secret", sidecar)
            self.assertNotIn("patient-secret", sidecar)


if __name__ == "__main__":
    unittest.main()
