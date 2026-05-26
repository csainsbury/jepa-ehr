from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval.autoregression_readiness import compute_autoregression_readiness_report
from clinical_jepa.eval.export_transformer_autoreg_rollouts import export_transformer_autoreg_rollouts

HAS_TORCH_H5PY = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("h5py") is not None


@unittest.skipUnless(HAS_TORCH_H5PY, "torch and h5py are required for transformer autoreg rollout tests")
class TransformerAutoregRolloutTests(unittest.TestCase):
    def _write_synthetic_fixture(self, root: Path, *, degenerate: bool = False) -> tuple[Path, Path]:
        import h5py
        import torch

        from clinical_jepa.arms.v0e.transformer_autoreg import (
            TransformerAutoregConfig,
            TransformerHorizonAutoregressor,
            checkpoint_metadata,
        )

        n_rows = 8
        horizons = 3
        dim = n_rows * horizons
        vocab = 96
        h5_path = root / "synthetic-transformer-seq.h5"
        manifest_rows = []
        with h5py.File(h5_path, "w") as h5:
            for i in range(n_rows):
                group = f"seq{i}"
                ctx_token = 1 + i
                h0_token = 20 + i
                h1_token = 40 + i
                h2_token = 60 + i
                token_ids = np.asarray(
                    [ctx_token, ctx_token, h0_token, h0_token, h1_token, h1_token, h2_token, h2_token],
                    dtype=np.int64,
                )
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
                    "horizon_descriptor": "synthetic_transformer_three_horizon",
                    "source_dataset": "synthetic",
                })
        manifest_path = root / "target-blocks.json"
        manifest_path.write_text(json.dumps({"blocks": manifest_rows}))

        config = TransformerAutoregConfig(
            vocab_size=vocab,
            embedding_dim=dim,
            max_horizons=horizons,
            encoder_layers=0,
            heads=1,
            max_len=8,
            dropout=0.0,
            use_layer_norm=False,
            predictor_hidden_mult=0,
        )
        model = TransformerHorizonAutoregressor(config)
        with torch.no_grad():
            model.encoder.token_embedding.weight.zero_()
            model.encoder.position_embedding.weight.zero_()
            for i in range(n_rows):
                model.encoder.token_embedding.weight[1 + i, i] = 1.0
                model.encoder.token_embedding.weight[1 + i, n_rows + i] = 1.0
                model.encoder.token_embedding.weight[1 + i, 2 * n_rows + i] = 1.0
                model.encoder.token_embedding.weight[20 + i, i] = 1.0
                model.encoder.token_embedding.weight[40 + i, n_rows + i] = 1.0
                model.encoder.token_embedding.weight[60 + i, 2 * n_rows + i] = 1.0
            for head in model.horizon_heads:
                head.weight.zero_()
                head.bias.zero_()
            if not degenerate:
                model.horizon_heads[0].weight[:n_rows, :n_rows] = torch.eye(n_rows)
                model.horizon_heads[1].weight[n_rows : 2 * n_rows, n_rows : 2 * n_rows] = torch.eye(n_rows)
                model.horizon_heads[2].weight[2 * n_rows :, 2 * n_rows :] = torch.eye(n_rows)
        ckpt_path = root / ("degenerate-transformer-autoreg.pt" if degenerate else "transformer-autoreg.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            **checkpoint_metadata(config, horizon_count_trained=horizons),
            "target_window_events": 2,
            "horizon_stride_events": 2,
        }, ckpt_path)
        return ckpt_path, manifest_path

    def test_transformer_autoreg_checkpoint_beats_time_shift_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_path, manifest_path = self._write_synthetic_fixture(root)
            out = root / "rollout"
            export_report = export_transformer_autoreg_rollouts(
                checkpoint_path=ckpt_path,
                target_blocks_path=manifest_path,
                output_dir=out,
                splits=("dev",),
                target_types=("T0",),
                horizon_count=3,
                target_window_events=2,
                horizon_stride_events=2,
                batch_size=4,
            )
            self.assertEqual(export_report["model_family"], "transformer_autoregressive_latent_v0e")
            pred = np.load(out / "predicted-rollout.fp16.npy")
            target = np.load(out / "observed-rollout.fp16.npy")
            index = [json.loads(line) for line in (out / "rollout-index.local.jsonl").read_text().splitlines()]
            report = compute_autoregression_readiness_report(
                pred,
                target,
                index,
                distractor_policy="same_split_target_type",
                control_mode="all",
                time_shift_mode="noncyclic_forward",
                time_shift_distances=(1, 2),
            )
            time_rows = report["shift_controls"]["controls"]["time_shift"]["per_horizon"]
            self.assertEqual({row["distance"] for row in time_rows}, {1, 2})
            self.assertGreater(min(float(row["cosine_aligned_minus_control"]) for row in time_rows), 0.95)
            self.assertGreater(min(float(row["matched_random_control"]["cosine_delta"]) for row in report["per_horizon"]), 0.70)
            dumped = json.dumps(report)
            self.assertNotIn("block-secret", dumped)
            self.assertNotIn("patient-secret", dumped)

    def test_degenerate_predictor_is_detected_by_effective_rank_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_path, manifest_path = self._write_synthetic_fixture(root, degenerate=True)
            out = root / "degenerate-rollout"
            export_transformer_autoreg_rollouts(
                checkpoint_path=ckpt_path,
                target_blocks_path=manifest_path,
                output_dir=out,
                splits=("dev",),
                target_types=("T0",),
                horizon_count=3,
                target_window_events=2,
                horizon_stride_events=2,
                batch_size=4,
            )
            pred = np.load(out / "predicted-rollout.fp16.npy")
            target = np.load(out / "observed-rollout.fp16.npy")
            index = [json.loads(line) for line in (out / "rollout-index.local.jsonl").read_text().splitlines()]
            report = compute_autoregression_readiness_report(
                pred,
                target,
                index,
                distractor_policy="same_split_target_type",
                control_mode="all",
                time_shift_mode="noncyclic_forward",
                time_shift_distances=(1, 2),
            )
            self.assertLess(max(float(row["pred_effective_rank"]) for row in report["per_horizon"]), 1.0)
            self.assertLess(max(abs(float(row["matched_random_control"]["cosine_delta"])) for row in report["per_horizon"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
