from __future__ import annotations

import logging
import unittest

import numpy as np

from clinical_jepa.arms.v0b.train_minimal_jepa import (
    DEFAULT_VOCAB_SIZE,
    max_token_id_in_examples,
    resolve_vocab_size,
)


class VocabSizeResolutionTests(unittest.TestCase):
    def test_reads_vocab_size_from_config(self) -> None:
        self.assertEqual(resolve_vocab_size({"vocabulary": {"vocab_size": 1050}}), 1050)

    def test_warns_and_defaults_when_missing(self) -> None:
        with self.assertLogs("clinical_jepa.arms.v0b.train_minimal_jepa", level=logging.WARNING) as cm:
            vs = resolve_vocab_size({})
        self.assertEqual(vs, DEFAULT_VOCAB_SIZE)
        self.assertTrue(any("vocab_size" in m for m in cm.output))

    def test_max_token_id_across_examples(self) -> None:
        examples = [
            ("train", np.array([4, 90, 951], dtype=np.int64), [np.array([1049], dtype=np.int64)]),
            ("dev", np.array([1, 2], dtype=np.int64), [np.array([3], dtype=np.int64)]),
        ]
        self.assertEqual(max_token_id_in_examples(examples), 1049)

    def test_empty_examples_return_minus_one(self) -> None:
        self.assertEqual(max_token_id_in_examples([]), -1)


if __name__ == "__main__":
    unittest.main()
