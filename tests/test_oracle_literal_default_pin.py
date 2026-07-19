"""Comprehensive default-path byte-pin for the LITERAL generator (blueprint M1; Fable review #1e).

The existing `test_default_none_path_regression` pins only 3 families / 4 arrays / truncated digests /
orthogonal nuisance. Before any v2 realism-layer edit lands, the FULL default cell (every array field, all 5
families, BOTH nuisance cells, full sha256) must be pinned so a v2 change cannot silently perturb the frozen
default generation the certification path depends on.
"""
from __future__ import annotations

import dataclasses
import hashlib
import unittest

import numpy as np

from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell

# full sha256 of every field of a fixed-seed (seed=7, n=40, kappa=0.35) default LiteralCell.
_DEFAULT_CELL_DIGESTS = {
    ("T_hmm_markov", "orthogonal"): "fe7f2c75356199eabf5a822872437168c561a02cc88cfcb6aac963c2dd68dae9",
    ("T_hmm_markov", "correlated_leak"): "409c6b383abbec2c60ae0d0252fc123abc192855620b6d3286aac777045ecd75",
    ("T_realized_history", "orthogonal"): "8638b07a0e2aa484fab639c2c57d86bcfdb9a979e65fb3163ba414be85f843fb",
    ("T_realized_history", "correlated_leak"): "05d344402c94f6b5f1559e5bd8cf65243c886536f56139fd3b0899dd3b96b900",
    ("T_latent_factor", "orthogonal"): "50baea3eadc171e39ebd22515f4494db39ec54ebe8ce7668dccc9c52b0643f30",
    ("T_latent_factor", "correlated_leak"): "77297ea061983725477f5bfc84cd0ddf83e73ef5e7241c6248cd660cf586d28d",
    ("E_no_h_exogenous", "orthogonal"): "5761ba69bae0d9f30a88519508241a314caeb717c69f2bd5ebc567bf76edfc3e",
    ("E_no_h_exogenous", "correlated_leak"): "66d5d728b4f2ec4b65cb93e73c3c2e1da0c628d824a8c99aba049e4b457f1396",
    ("E_offgrid_nonlinear", "orthogonal"): "f132c069447d4f74a116658365b81fbd64480095c3fad59548aca6e4edecfb3b",
    ("E_offgrid_nonlinear", "correlated_leak"): "7d7914b66f5eb24ee18091f916ce20b9dab32b59bb945e8e309a5111a4475f32",
}


def _cell_digest(c) -> str:
    h = hashlib.sha256()
    for f in sorted(dataclasses.fields(c), key=lambda x: x.name):
        v = getattr(c, f.name)
        h.update(f.name.encode())
        if v is None:
            h.update(b"None")
        elif isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).tobytes())
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


class DefaultPathBytePin(unittest.TestCase):
    def test_default_literal_cells_are_byte_pinned(self) -> None:
        for (fam, nu), want in _DEFAULT_CELL_DIGESTS.items():
            c = generate_literal_cell(fam, 0.35, nu, 40, seed=7)
            self.assertEqual(_cell_digest(c), want, f"default cell drifted: {fam}/{nu}")
            self.assertIsNone(c.calibration_adapter_hash, f"{fam}/{nu} default must be uncalibrated")


if __name__ == "__main__":
    unittest.main()
