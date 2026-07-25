"""Rung-2 sub-gate 1 exposure gap — the teacher-forced arm and its absent-input guard.

The exposure gap g_h = d_self_free(h) - d_self_tf(h) needs BOTH rollout arms. Before this was wired,
no producer emitted the teacher-forced arm, so `exposure_gap_slope_lo` was necessarily 0, the
DRIFT_DOMINANT branch could never fire, and `classify_signature` fell through to HEALTHY — a confident
wrong answer on the exact question sub-gate 1 exists to answer. These tests pin both halves: absent
input must REFUSE, and the producer must emit a well-formed second arm.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval import rung2_rollout_diag as RD
from clinical_jepa.eval import rung2_rollout_export as RE
from clinical_jepa.eval.rung2_contract import NOT_EVALUABLE, TRANSITION_META_KEY

RECURSIVE_META = {"autoregression_mode": "recursive", "horizon_count_trained": 3,
                  "horizon_stride_tokens": 8, "max_target_tokens": 8}


class TestAbsentExposureGapRefuses(unittest.TestCase):
    def test_missing_gap_does_not_silently_report_healthy(self):
        """The regression this guards: absent gap scored as absence of drift."""
        sig = RD.classify_signature(dself_over_nn_point=0.40, exposure_gap_slope_lo=0.0,
                                    dself_slope_hi=0.10, transition_evaluable=True,
                                    exposure_gap_available=False)
        self.assertEqual(sig, NOT_EVALUABLE)

    def test_same_inputs_with_gap_available_are_drift(self):
        sig = RD.classify_signature(dself_over_nn_point=0.40, exposure_gap_slope_lo=0.30,
                                    dself_slope_hi=0.10, transition_evaluable=True,
                                    exposure_gap_available=True)
        self.assertEqual(sig, "DRIFT_DOMINANT")

    def test_collapse_still_asserted_without_the_gap(self):
        """COLLAPSE does not consume the gap, so it must survive the guard."""
        sig = RD.classify_signature(dself_over_nn_point=0.95, exposure_gap_slope_lo=0.0,
                                    dself_slope_hi=0.0, transition_evaluable=True,
                                    exposure_gap_available=False)
        self.assertEqual(sig, "COLLAPSE_DOMINANT")

    def test_export_refuses_a_signature_without_teacher_forcing(self):
        r = RE.recursive_transition_metrics(RECURSIVE_META, dself_over_nn_point=0.40,
                                            exposure_gap_slope_lo=0.0, dself_slope_hi=0.10)
        self.assertFalse(r["exposure_gap_available"])
        self.assertEqual(r["signature"], NOT_EVALUABLE)
        self.assertIn("teacher-forced", r["reason"])

    def test_export_scores_normally_with_both_arms(self):
        r = RE.recursive_transition_metrics(RECURSIVE_META, dself_free=np.full(40, 0.55),
                                            dself_tf=np.full(40, 0.20), dself_over_nn_point=0.40,
                                            exposure_gap_slope_lo=0.30, dself_slope_hi=0.10)
        self.assertTrue(r["exposure_gap_available"])
        self.assertEqual(r["signature"], "DRIFT_DOMINANT")
        self.assertAlmostEqual(r["exposure_gap_mean"], 0.35, places=6)


class TestTeacherForcedProducer(unittest.TestCase):
    def setUp(self):
        try:
            import h5py  # noqa: F401
            import torch  # noqa: F401
        except ImportError:                                     # pragma: no cover
            self.skipTest("torch/h5py not available")

    def _export(self, mode: str):
        import h5py
        import torch

        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA
        from clinical_jepa.eval.export_mean_token_rollouts import export_mean_token_rollouts

        V, D, K, W = 1050, 32, 3, 8
        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(11)
        model = MeanTokenJEPA(vocab_size=V, embedding_dim=D, autoregression_mode=mode, max_horizons=K)
        ck = tmp / "ckpt.pt"
        torch.save({"model_state_dict": model.state_dict(), "vocab_size": V, "embedding_dim": D,
                    "architecture": {"vocab_size": V, "embedding_dim": D, "autoregression_mode": mode},
                    "autoregression_mode": mode, "horizon_count_trained": K,
                    "horizon_stride_tokens": W, "max_target_tokens": W, "max_horizons": K,
                    TRANSITION_META_KEY: mode == "recursive"}, ck)
        h5p = tmp / "seqs.h5"
        blocks = []
        with h5py.File(h5p, "w") as f:
            for i in range(16):
                g = f.create_group(f"s{i}")
                g.create_dataset("token_ids", data=rng.integers(1, V, size=200).astype(np.int64))
                blocks.append({"block_id": f"b{i}", "split": "dev", "target_type": "T0",
                               "sequence_file": str(h5p), "sequence_group": f"s{i}",
                               "context_start_ref": 0, "context_end_ref": 40, "target_start_ref": 50})
        mp = tmp / "blocks.json"
        mp.write_text(json.dumps({"blocks": blocks}))
        rep = export_mean_token_rollouts(
            checkpoint_path=ck, target_blocks_path=mp, output_dir=tmp / "out", splits=("dev",),
            target_types=("T0",), horizon_count=K, target_window_events=W, horizon_stride_events=W,
            max_context_tokens=64, batch_size=8, cpu=True)
        return tmp / "out", rep

    def test_recursive_arm_emits_teacher_forced_rollouts(self):
        out, rep = self._export("recursive")
        self.assertTrue(rep["teacher_forcing_available"])
        self.assertEqual(rep["teacher_forced_rows"], rep["rows_exported"])
        free = np.load(out / "predicted-rollout.fp16.npy")
        tf = np.load(out / "teacher-forced-rollout.fp16.npy")
        obs = np.load(out / "observed-rollout.fp16.npy")
        self.assertEqual(free.shape, tf.shape)
        self.assertEqual(free.shape, obs.shape)

    def test_step_zero_is_identical_in_both_arms(self):
        """Both arms predict step 0 from the TRUE context latent, so g_0 is 0 by construction."""
        out, _ = self._export("recursive")
        free = np.load(out / "predicted-rollout.fp16.npy").astype(np.float64)
        tf = np.load(out / "teacher-forced-rollout.fp16.npy").astype(np.float64)
        obs = np.load(out / "observed-rollout.fp16.npy").astype(np.float64)
        np.testing.assert_allclose(free[:, 0, :], tf[:, 0, :])
        gap0 = RD.exposure_gap(RD.cos_dist(free[:, 0, :], obs[:, 0, :]),
                               RD.cos_dist(tf[:, 0, :], obs[:, 0, :]))
        np.testing.assert_allclose(gap0, 0.0, atol=1e-12)

    def test_arms_diverge_after_the_first_step(self):
        out, _ = self._export("recursive")
        free = np.load(out / "predicted-rollout.fp16.npy").astype(np.float64)
        tf = np.load(out / "teacher-forced-rollout.fp16.npy").astype(np.float64)
        self.assertFalse(np.allclose(free[:, 1:, :], tf[:, 1:, :]))

    def test_non_recursive_arm_declares_the_gap_absent(self):
        """A horizon-conditioned head has no previous-step latent to force — must not fake an arm."""
        _out, rep = self._export("horizon_conditioned")
        self.assertFalse(rep["teacher_forcing_available"])
        self.assertEqual(rep["teacher_forced_rows"], 0)
        self.assertIn("NOT produced", rep["teacher_forcing_note"])


if __name__ == "__main__":
    unittest.main()
