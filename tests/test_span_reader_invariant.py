"""Single-span-reader invariant (Pi R4 required change #1).

The shared block_spans helper must be the ONLY place a target span is read from a
raw ``target_start_ref``. Any consumer that clamps the -1 empty/censored sentinel
(``max(0, int(... target_start_ref ...))``) or computes a length as
``target_end_ref - target_start_ref + 1`` silently reintroduces the "empty target
read as arr[0:]" / "len-1 miscount" bug upstream of the encoder, where the model
cannot detect it. This grep-style guard fails if any such idiom reappears.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

CLINICAL_JEPA = Path(__file__).resolve().parents[1] / "clinical_jepa"
# block_spans.py is THE reader (it legitimately handles the sentinel); extract_blocks
# CONSTRUCTS the refs (it never slices with them).
ALLOWED = {"block_spans.py"}

# The two dangerous idioms.
CLAMP_SENTINEL = re.compile(r"max\(\s*0\s*,\s*int\([^)]*target_start_ref")
LEN_MISCOUNT = re.compile(r"target_end_ref[^\n]*-[^\n]*target_start_ref[^\n]*\+\s*1")
RAW_SLICE = re.compile(r"(?:arr|token_ids)\[[^\]]*target_start_ref")


class SpanReaderInvariantTests(unittest.TestCase):
    def _offending_lines(self, pattern: re.Pattern) -> list[str]:
        hits: list[str] = []
        for py in CLINICAL_JEPA.rglob("*.py"):
            if py.name in ALLOWED:
                continue
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{py.relative_to(CLINICAL_JEPA)}:{i}: {line.strip()}")
        return hits

    def test_no_sentinel_clamp(self) -> None:
        hits = self._offending_lines(CLAMP_SENTINEL)
        self.assertEqual(hits, [], "raw target_start_ref clamp (empty -> arr[0:]) found:\n" + "\n".join(hits))

    def test_no_len_miscount(self) -> None:
        hits = self._offending_lines(LEN_MISCOUNT)
        self.assertEqual(hits, [], "target_end_ref - target_start_ref + 1 (len-1 for empty) found:\n" + "\n".join(hits))

    def test_no_raw_target_slice(self) -> None:
        hits = self._offending_lines(RAW_SLICE)
        self.assertEqual(hits, [], "raw arr[... target_start_ref ...] slice found:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
