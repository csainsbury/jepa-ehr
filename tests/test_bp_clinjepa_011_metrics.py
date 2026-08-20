from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from clinical_jepa.eval.next_event_metrics import (
    balanced_accuracy,
    effective_rank,
    gaussian_crps,
    gaussian_interval_nll,
    inverse_median_raw_time,
    off_diagonal_cosine,
    order_pair_mask,
    population_feature_variance,
    position_marginal_probabilities,
    recurrence_probabilities,
    tie_permutation,
    type_cross_entropy,
)


def test_hand_computed_ce_nll_and_crps_equations():
    logits = torch.tensor([[[0.0] + [-math.inf] * 16]])
    assert type_cross_entropy(logits, torch.tensor([[1]]), torch.tensor([[True]])).item() == 0.0
    y, mean, raw = torch.tensor([1.25]), torch.tensor([0.5]), torch.tensor([-0.2])
    sigma = torch.nn.functional.softplus(raw) + 1e-4
    expected_nll = torch.log(sigma) + 0.5 * ((y - mean) / sigma).square() + 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(gaussian_interval_nll(y, mean, raw), expected_nll)
    z = (y - mean) / sigma
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    expected_crps = sigma * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    torch.testing.assert_close(gaussian_crps(y, mean, raw), expected_crps)


def test_inverse_median_and_zero_safe_rule():
    value = inverse_median_raw_time(np.array([0.0, 1.0]), train_mu=math.log(2), train_sigma=0.5, unit=1.0)
    np.testing.assert_allclose(value, np.maximum(0, np.expm1(math.log(2) + 0.5 * np.array([0.0, 1.0]))))
    assert inverse_median_raw_time(np.array([-100.0]), train_mu=0.0, train_sigma=1.0)[0] == 0.0


def test_balanced_accuracy_and_missing_class_failure():
    assert balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75
    with pytest.raises(ValueError, match="both"):
        balanced_accuracy([1, 1], [1, 0])


def test_marginal_and_recurrence_primitives_use_non_pad_class_mapping():
    ids = np.array([[1, 2], [1, 3]])
    mask = np.ones_like(ids, dtype=bool)
    marginal = position_marginal_probabilities(ids, mask, class_count=3, pseudocount=0.5)
    np.testing.assert_allclose(marginal[0], np.array([2.5, 0.5, 0.5]) / 3.5)
    recurrence = recurrence_probabilities(marginal, [2, 2], pseudocount=0.5)
    context = np.array([0.5, 2.5, 0.5]) / 3.5
    np.testing.assert_allclose(recurrence, 0.5 * marginal + 0.5 * context)


def test_exact_variance_effective_rank_eligibility_and_cosine_zero_vector_rules():
    x = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert population_feature_variance(x) == pytest.approx(0.5)  # feature variances [1,0]
    eligible_rank_one = np.tile(x, (16, 1))
    assert effective_rank(eligible_rank_one) == pytest.approx(1.0)
    assert effective_rank(np.ones((32, 2))) == 0.0
    with pytest.raises(ValueError, match="at least 32"):
        effective_rank(np.ones((31, 16)))
    assert off_diagonal_cosine(np.array([[0.0, 0.0], [1.0, 0.0]])) == 0.0
    assert off_diagonal_cosine(np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(-1 / 3)
    with pytest.raises(ValueError):
        off_diagonal_cosine(np.ones((1, 2)))


def test_exact_blake2b256_pcg64_tie_permutation_excludes_type_and_value():
    namespace, key, timestamp, occurrence, size = "clinjepa-j04-tie-v1", "synthetic-sequence-7", "4", 2, 7
    message = f"{namespace}|{key}|{timestamp}|{occurrence}|{size}".encode("utf-8")
    digest = hashlib.blake2b(message, digest_size=32).digest()
    seed64 = int.from_bytes(digest[:8], "little")
    expected = np.random.Generator(np.random.PCG64(seed64)).permutation(size)
    actual = tie_permutation(key, timestamp, occurrence, size)
    np.testing.assert_array_equal(actual, expected)
    # Type/value arrays are intentionally not arguments and cannot affect the key.
    np.testing.assert_array_equal(actual, tie_permutation(key, timestamp, occurrence, size))
    assert not np.array_equal(actual, tie_permutation(key, timestamp, occurrence, size, namespace="clinjepa-j04-tie-v1-sensitivity"))


def test_zero_interval_value_nll_gradients_and_inverse_median_are_finite():
    delta = torch.tensor([0.0])
    train_mu, train_sigma = 0.5, 0.75
    y = (torch.log1p(delta) - train_mu) / train_sigma
    mean, raw = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    nll = gaussian_interval_nll(y, mean, raw).sum()
    nll.backward()
    assert torch.isfinite(y).all() and torch.isfinite(nll)
    assert torch.isfinite(mean.grad).all() and torch.isfinite(raw.grad).all()
    assert np.isfinite(inverse_median_raw_time(np.array([0.0]), train_mu=train_mu, train_sigma=train_sigma)).all()


def test_order_pair_mask_retains_zero_interval_events_but_excludes_ties():
    timestamps = [0.0, 0.0, 2.0, 3.0]
    mask = order_pair_mask(timestamps, [True, True, True, False])
    assert not mask[0, 1]  # within-tie order excluded
    assert mask[0, 2] and mask[1, 2]
    assert not mask[:, 3].any()
    assert mask.sum() == 2
