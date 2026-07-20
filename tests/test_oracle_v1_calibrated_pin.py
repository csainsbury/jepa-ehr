"""Step-1 pins for the realism-v2 build (Pi assent, thread thr-20260719T155635Z-fff46633).

Before any v2 seam work, freeze three tripwires that the corrected M0/M1/identity-split work must not disturb:

1. **v1 CALIBRATED-path golden digests** — full sha256 of every field of a fixed-seed calibrated `LiteralCell`
   across BOTH source profiles (SCID / MIMIC) x all 5 literal families x both nuisance cells. The existing
   `test_oracle_literal_default_pin` covers only the default (uncalibrated) path; Pi P-C requires a calibrated
   pin so an edit that keeps `calib_knobs=None` on the default path but silently changes the historical v1
   adapter path is caught.
2. **dev-scaffold hash pin** — `realism_v2_schema_hash()` is pinned EXACTLY and labelled a DEVELOPMENT scaffold
   identity, explicitly NOT the final frozen M3a verification identity.
3. **frozen v1 identity asserts** — `invariant_hash`, `ORACLE_EVALUATOR_IDENTITY`, `base_schema_hash`,
   `generator_fit_schema_hash`, `calibration_schema_hash` must be unchanged.

The two source profiles are FIXED SYNTHETIC golden inputs (knobs + calib_context), NOT derived from any
governed data — this is a deterministic-reproducibility tripwire, not a real calibration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import unittest

import numpy as np

from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell

# --- fixed synthetic source profiles (golden inputs; NOT governed data) ------------------------------
_PROFILES = {
    "SCID": {
        "knobs": {"token_freq_temperature": 1.1, "zero_gap_bias": 0.5,
                  "timing_rate_scale": 1.2, "gap_dispersion": 0.9},
        "ctx": {"pooled_class_prior": [0.30, 0.25, 0.20, 0.15, 0.10], "global_gap_median": 1.0,
                "pooled_positive_gap_ecdf": [(0.5, 0.4), (1.0, 0.8), (2.0, 1.0)]},
    },
    "MIMIC": {
        "knobs": {"token_freq_temperature": 1.0, "zero_gap_bias": 0.8,
                  "timing_rate_scale": 1.6, "gap_dispersion": 1.3},
        "ctx": {"pooled_class_prior": [0.10, 0.15, 0.20, 0.25, 0.30], "global_gap_median": 1.5,
                "pooled_positive_gap_ecdf": [(0.4, 0.3), (0.9, 0.7), (1.8, 0.95), (3.0, 1.0)]},
    },
}

# full sha256 of every field of a fixed-seed (seed=7, n=40, kappa=0.35) CALIBRATED LiteralCell.
_CALIBRATED_CELL_DIGESTS = {
    ("SCID", "T_hmm_markov", "orthogonal"): "99d25fd2bbc2554f48142b026e881f924e905ada3644da3e6faf02939faaff66",
    ("SCID", "T_hmm_markov", "correlated_leak"): "dfa18aeb5b425699131c175b12a6c3d6931917e14e2251b18d1b22c64fc894df",
    ("SCID", "T_realized_history", "orthogonal"): "79ee618a51550b97118fa16e3b49a178e761f681f8c6136c34cec99a3952cf51",
    ("SCID", "T_realized_history", "correlated_leak"): "22a5e28d720431a37311db0d0a152c76919188b4d7fbe4337e2fe562844175cd",
    ("SCID", "T_latent_factor", "orthogonal"): "9023fe33e53aa257f2a59771c16d97cf73cf6c440d7b956a4609a35fa07f30a0",
    ("SCID", "T_latent_factor", "correlated_leak"): "c9b52cf7700c8f1f235a2749e27e24f6547576cd7a550249e23bcd3b61ca5c24",
    ("SCID", "E_no_h_exogenous", "orthogonal"): "015ff380c3d68a9b3d407ca491a4f52be9566a8ade423a358fabb67440d53704",
    ("SCID", "E_no_h_exogenous", "correlated_leak"): "a9904ce289f6f48da778f6465d8a6b4f1705d371c77686688a938f4eeaa596a5",
    ("SCID", "E_offgrid_nonlinear", "orthogonal"): "83e7da0f598b7fa8170999a6d8e7309671c2520518422895094b31506bf87548",
    ("SCID", "E_offgrid_nonlinear", "correlated_leak"): "ead676e825a900dc890fa2e831331a8531caf14ef13c5088b755fc7c0d713ebf",
    ("MIMIC", "T_hmm_markov", "orthogonal"): "28f6bf8d7048423bdded746f2c869891e950916f62e6491fafc096b5f853e070",
    ("MIMIC", "T_hmm_markov", "correlated_leak"): "a6d13cb58a318997809fb5e2cbcadfb8059c36e1408800e7b7ceccafb661793f",
    ("MIMIC", "T_realized_history", "orthogonal"): "d32acb5baec6646d7e04e0798f3a26fd1ad4c3c4344de69d8ce93f1d8c587216",
    ("MIMIC", "T_realized_history", "correlated_leak"): "033517becdfa15373fd1976339c6bffe57f8b93d099834191ff4cb92aa05b46e",
    ("MIMIC", "T_latent_factor", "orthogonal"): "5b7e67736a3fedabac449d5da974363ce902bacde503f6fcf5fbe33f53055cad",
    ("MIMIC", "T_latent_factor", "correlated_leak"): "1d7f5ff737414680bcdd426ea87286fc2a0ec93bd99a69f220cdcc59df306f6c",
    ("MIMIC", "E_no_h_exogenous", "orthogonal"): "9cb1e6c8da61f3a4a4110898948cfe31fd189cd791667cdb6131d915be3373ce",
    ("MIMIC", "E_no_h_exogenous", "correlated_leak"): "12d62862f0fb88cfc862bd1dd2465621e9776e05550b9b725d1b5d5af60cc08f",
    ("MIMIC", "E_offgrid_nonlinear", "orthogonal"): "0d7434b20afadd659e0754a06837170219fd63849f0aa73cb24761111c3b4544",
    ("MIMIC", "E_offgrid_nonlinear", "correlated_leak"): "53ffc99ed8288b90a2f176695d0049e3fcdc830e6198beaeff758ac2eb46e9e8",
}

# frozen v1 identities (Pi P-C: these must NOT move during step-1 pins / M0 boundary / identity split).
_FROZEN_IDENTITIES = {
    "invariant_hash": "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0",
    "calibration_schema_hash": "f4f86336fff104d89ec5b589e5bfc7368b2f97efac48d6b00e5b686b250d7aab",
    "base_schema_hash": "13a0b4dedfd1ec773f29f680b5e752b2d6c7111cbc57d396704e6e75a619c8be",
    "generator_fit_schema_hash": "b6aa74e1fd3ddc0565957328c8c7f489c87dd372eef1321af9344aea147b180f",
    "evaluator_identity": "oracle_meta_eval_v5",
}

# DEVELOPMENT scaffold identity — explicitly NOT the final frozen M3a verification identity.
# Bumped intentionally at step 3 (A/D identity split + corrected certification rationale); still a dev hash.
_DEV_SCAFFOLD_HASH = "704b079a6137a9dbadbf9938b031b4af5cedfbaadb81fbfa9d872c6519cd6f85"


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


class V1CalibratedPathPin(unittest.TestCase):
    def test_calibrated_cells_are_byte_pinned(self) -> None:
        for (prof, fam, nu), want in _CALIBRATED_CELL_DIGESTS.items():
            spec = _PROFILES[prof]
            c = generate_literal_cell(fam, 0.35, nu, 40, seed=7,
                                      calib_knobs=spec["knobs"], calib_context=spec["ctx"],
                                      calib_source_profile=prof)
            self.assertIsNotNone(c.calibration_adapter_hash, f"{prof}/{fam}/{nu} must be calibrated")
            self.assertEqual(_cell_digest(c), want, f"v1 calibrated path drifted: {prof}/{fam}/{nu}")

    def test_source_profiles_are_distinct(self) -> None:
        # SCID and MIMIC must not collapse to the same calibrated output (guards a profile-erasing edit).
        for fam in ("T_hmm_markov", "E_offgrid_nonlinear"):
            for nu in ("orthogonal", "correlated_leak"):
                self.assertNotEqual(_CALIBRATED_CELL_DIGESTS[("SCID", fam, nu)],
                                    _CALIBRATED_CELL_DIGESTS[("MIMIC", fam, nu)], f"{fam}/{nu}")


class DevScaffoldHashPin(unittest.TestCase):
    def test_scaffold_hash_is_pinned_as_development_identity(self) -> None:
        from clinical_jepa.eval.oracle_realism_v2 import realism_v2_schema_hash, REALISM_V2_VERSION
        self.assertEqual(realism_v2_schema_hash(), _DEV_SCAFFOLD_HASH)
        # label discipline: the pinned hash is a DEV scaffold, not the final M3a verification identity.
        self.assertTrue(REALISM_V2_VERSION.endswith("_scaffold_dev"),
                        "scaffold identity must remain a development version until the M3a freeze")


class FrozenV1Identities(unittest.TestCase):
    def test_frozen_v1_identities_unchanged(self) -> None:
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_calibration import calibration_schema_hash
        from clinical_jepa.eval.oracle_aggregate_extract import base_schema_hash, generator_fit_schema_hash
        from clinical_jepa.eval.rung2_contract import ORACLE_EVALUATOR_IDENTITY
        got = {
            "invariant_hash": invariant_hash(),
            "calibration_schema_hash": calibration_schema_hash(),
            "base_schema_hash": base_schema_hash(),
            "generator_fit_schema_hash": generator_fit_schema_hash(),
            "evaluator_identity": ORACLE_EVALUATOR_IDENTITY,
        }
        self.assertEqual(got, _FROZEN_IDENTITIES)


if __name__ == "__main__":
    unittest.main()
