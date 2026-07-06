"""Encode-empty training-core tests (Pi R4): the hybrid loss must LEARN silence
(occupancy beats base-rate; empties pulled toward the frozen z_empty; positive
empty-vs-populated margin) without collapsing — on synthetic separable data."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.arms.v0b.encode_empty_train import (  # noqa: E402
    binary_auc,
    calibration_report,
    collapse_diagnostics,
    encode_empty_loss,
    maybe_reseed_prototype,
    occupancy_targets,
)

DIM = 16
VOCAB = 64


def _make_examples(n_empty: int, n_pop: int):
    """Separable synthetic batch: type-A contexts {10,11,12} -> EMPTY; type-B
    contexts {20,21,22} -> non-empty targets (count 1..3). Occupancy is learnable
    from context; empties route their target latent to the frozen z_empty."""
    import torch

    ctx, tgt, is_empty, count = [], [], [], []
    for _ in range(n_empty):
        ctx.append([10, 11, 12, 10, 11])
        tgt.append([0, 0, 0])            # unused for empties (target_latent -> z_empty)
        is_empty.append(True)
        count.append(0)
    for i in range(n_pop):
        ctx.append([20, 21, 22, 20, 21])
        k = 1 + (i % 3)                  # counts 1,2,3 -> includes 1-event (Pi Q3)
        tgt.append([30, 31, 32][:k] + [0] * (3 - k))
        is_empty.append(False)
        count.append(k)
    return (
        torch.tensor(ctx, dtype=torch.long),
        torch.tensor(tgt, dtype=torch.long),
        torch.tensor(is_empty, dtype=torch.bool),
        torch.tensor(count, dtype=torch.long),
    )


@unittest.skipUnless(HAS_TORCH, "torch required")
class LossComponentTests(unittest.TestCase):
    def test_occupancy_targets(self) -> None:
        import torch

        is_empty = torch.tensor([True, False, False])
        count = torch.tensor([0, 3, 1])
        y_occ, y_logcount = occupancy_targets(is_empty, count)
        self.assertEqual(y_occ.tolist(), [0.0, 1.0, 1.0])          # 1 = occupied
        self.assertAlmostEqual(float(y_logcount[1]), float(torch.log1p(torch.tensor(3.0))), places=5)

    def test_binary_auc_known(self) -> None:
        import torch

        s = torch.tensor([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(binary_auc(s, torch.tensor([1, 1, 0, 0])), 1.0, places=6)
        self.assertAlmostEqual(binary_auc(s, torch.tensor([0, 0, 1, 1])), 0.0, places=6)
        self.assertEqual(binary_auc(s, torch.tensor([1, 1, 1, 1])), 0.5)  # class absent


@unittest.skipUnless(HAS_TORCH, "torch required")
class LearnsSilenceTests(unittest.TestCase):
    def test_training_learns_occupancy_and_pulls_empties_to_prototype(self) -> None:
        import torch

        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA

        torch.manual_seed(0)
        model = MeanTokenJEPA(VOCAB, DIM, encode_empty=True)
        ctx, tgt, is_empty, count = _make_examples(n_empty=24, n_pop=24)

        # Init separation check: reseed z_empty away from the non-empty sample (Pi Q2).
        sep = maybe_reseed_prototype(model, ctx[~is_empty], threshold=0.15)
        self.assertLessEqual(sep["abs_cos"], 0.5)  # exists a reasonable seed

        opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
        model.train()
        proto_before = model.empty_prototype.clone()
        for _ in range(400):
            loss, parts = encode_empty_loss(model, ctx, tgt, is_empty, count)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        diag = collapse_diagnostics(model, ctx, is_empty, count)
        # Occupancy is learned (separable data) and beats the base-rate Brier.
        self.assertGreater(diag["occupancy_auc"], 0.9)
        self.assertTrue(diag["beats_marginal"])
        self.assertGreater(diag["empty_recall"], 0.8)
        self.assertLess(diag["empty_false_positive_rate"], 0.2)
        # Empties are pulled toward z_empty MORE than populated rows (positive margin).
        self.assertGreater(diag["empty_vs_populated_margin"], 0.1)
        # empty-vs-1-event is separable here (Pi Q3 metric is defined + informative).
        self.assertGreater(diag["empty_vs_one_event_auc"], 0.8)
        # Anti-collapse invariant: the frozen prototype never moved during training.
        self.assertTrue(torch.equal(model.empty_prototype, proto_before))

    def test_calibration_report_on_natural_prevalence(self) -> None:
        import torch

        # A well-calibrated-ish predictor: prob≈label.
        y = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        p = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        rep = calibration_report(p, y)
        self.assertLess(rep["brier"], 0.1)
        self.assertGreaterEqual(rep["ece"], 0.0)
        self.assertAlmostEqual(rep["prevalence_occupied"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
