"""Smallest J04c bridge: make TRAIN nuisance independent without changing S/X/targets."""
from __future__ import annotations

import numpy as np

from clinical_jepa.eval.j04c_falsifier import (
    NUIS_COMP_0, NUIS_COMP_1, NUIS_ORDER_0, NUIS_ORDER_1, TIME_NUIS,
    SyntheticFactorSplit, _validate_split,
)


def independent_train_nuisance(split: SyntheticFactorSplit, generator_seed: int) -> SyntheticFactorSplit:
    _validate_split(split)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([generator_seed, 1, 7101])))
    nuisance = (rng.random(split.N.shape) < 0.5).astype(np.uint8)
    types = split.prefix_type_ids.copy()
    times = split.prefix_intervals.copy()
    types[:, 4] = np.where(nuisance[:, 0] == 0, NUIS_COMP_0, NUIS_COMP_1)
    types[:, 5] = np.where(nuisance[:, 1] == 0, NUIS_ORDER_0, NUIS_ORDER_1)
    types[:, 6] = TIME_NUIS
    times[:, 6] = np.where(nuisance[:, 2] == 0, 1.0, 4.0)
    result = SyntheticFactorSplit(
        types, times, split.target_type_ids.copy(), split.target_intervals.copy(),
        split.S.copy(), split.X.copy(), nuisance, split.L_after.copy(),
    )
    _validate_split(result)
    if not (np.array_equal(result.S, split.S) and np.array_equal(result.X, split.X)
            and np.array_equal(result.target_type_ids, split.target_type_ids)
            and np.array_equal(result.target_intervals, split.target_intervals)
            and np.array_equal(result.L_after, split.L_after)):
        raise RuntimeError("nuisance bridge changed scientific truth or targets")
    return result
