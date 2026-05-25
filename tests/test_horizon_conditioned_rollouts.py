from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval.autoregression_readiness import compute_autoregression_readiness_report
from clinical_jepa.eval.export_mean_token_rollouts import export_mean_token_rollouts

HAS_TORCH_H5PY = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("h5py") is not None


@unittest.skipUnless(HAS_TORCH_H5PY, "torch and h5py are required for horizon-conditioned rollout tests")
class HorizonConditionedRolloutTests(unittest.TestCase):
    def test_horizon_conditioned_checkpoint_separates_time_shift_control(self) -> None:
        import h5py
        import torch

        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA

        n_rows = 8
        dim = n_rows * 2
        vocab = 64
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            h5_path = root / "synthetic-seq.h5"
            manifest_rows = []
            with h5py.File(h5_path, "w") as h5:
                for i in range(n_rows):
                    group = f"seq{i}"
                    ctx_token = 1 + i
                    h0_token = 20 + i
                    h1_token = 40 + i
                    token_ids = np.asarray([ctx_token, ctx_token, h0_token, h0_token, h1_token, h1_token], dtype=np.int64)
                    h5.create_group(group).create_dataset("token_ids", data=token_ids)
                    manifest_rows.append({
                        "block_id": f"block-secret-{i}",
                        "patient_hash": f"patient-secret-{i}",
                        "split": "dev",
                        "target_type": "T0",
                        "sequence_file": str(h5_path),
                        "sequence_group": group,
                        "context_start_ref": 0,
                        "context_end_ref": 1,
                        "target_start_ref": 2,
                        "target_end_ref": 3,
                        "horizon_descriptor": "synthetic_two_horizon",
                        "source_dataset": "synthetic",
                    })
            manifest_path = root / "target-blocks.json"
            manifest_path.write_text(json.dumps({"blocks": manifest_rows}))

            model = MeanTokenJEPA(vocab, dim, autoregression_mode="horizon_conditioned", max_horizons=2)
            with torch.no_grad():
                model.embedding.weight.zero_()
                for i in range(n_rows):
                    model.embedding.weight[1 + i, i] = 1.0
                    model.embedding.weight[1 + i, n_rows + i] = 1.0
                    model.embedding.weight[20 + i, i] = 1.0
                    model.embedding.weight[40 + i, n_rows + i] = 1.0
                for head in model.horizon_heads:
                    head.weight.zero_()
                    head.bias.zero_()
                model.horizon_heads[0].weight[:n_rows, :n_rows] = torch.eye(n_rows)
                model.horizon_heads[1].weight[n_rows:, n_rows:] = torch.eye(n_rows)
            ckpt_path = root / "horizon-conditioned.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab,
                "embedding_dim": dim,
                "autoregression_mode": "horizon_conditioned",
                "horizon_count_trained": 2,
                "max_horizons": 2,
                "architecture": model.architecture_metadata(),
            }, ckpt_path)

            out = root / "rollout"
            export_report = export_mean_token_rollouts(
                checkpoint_path=ckpt_path,
                target_blocks_path=manifest_path,
                output_dir=out,
                splits=("dev",),
                target_types=("T0",),
                horizon_count=2,
                target_window_events=2,
                horizon_stride_events=2,
                batch_size=4,
            )
            self.assertEqual(export_report["autoregression_mode"], "horizon_conditioned")
            self.assertEqual(export_report["horizon_conditioning"], "horizon_specific_linear_heads")
            pred = np.load(out / "predicted-rollout.fp16.npy")
            target = np.load(out / "observed-rollout.fp16.npy")
            index = [json.loads(line) for line in (out / "rollout-index.local.jsonl").read_text().splitlines()]
            report = compute_autoregression_readiness_report(
                pred,
                target,
                index,
                distractor_policy="same_split_target_type",
                control_mode="all",
            )
            time_rows = report["shift_controls"]["controls"]["time_shift"]["per_horizon"]
            self.assertEqual(len(time_rows), 2)
            self.assertGreater(min(float(row["cosine_aligned_minus_control"]) for row in time_rows), 0.95)
            self.assertGreater(report["transition_dynamics"][0]["direction_cosine_mean"], 0.99)
            dumped_report = json.dumps(report)
            self.assertNotIn("block-secret", dumped_report)
            self.assertNotIn("patient-secret", dumped_report)


if __name__ == "__main__":
    unittest.main()
