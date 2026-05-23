#!/usr/bin/env python3
"""Preflight checks for a Clinical-JEPA tokenised data bundle.

This script inspects filenames, aggregate sizes, and required metadata files only.
It does not print patient-level records.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DENY_NAME_RE = re.compile(r"(\.env$|secret|credential|oauth|private[-_ ]?key|password|token\.json|id_rsa|id_ed25519)", re.I)
DIRECT_ID_HINT_RE = re.compile(r"(nhs|mrn|date[_-]?of[_-]?birth|dob|ssn|social[_-]?security)", re.I)
WARN_EXT = {".txt", ".doc", ".docx", ".rtf", ".pdf"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    problems: list[str] = []
    warnings: list[str] = []
    if not root.exists() or not root.is_dir():
        problems.append(f"root is not a directory: {root}")
    manifest = root / "manifest.json"
    card = root / "dataset-card.md"
    if args.strict:
        if not manifest.exists():
            problems.append("missing manifest.json")
        if not card.exists():
            problems.append("missing dataset-card.md")
        for d in ["schema", "index", "sequences"]:
            if not (root / d).exists():
                problems.append(f"missing required directory: {d}/")
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            for key in ["contains_raw_data", "contains_free_text", "contains_direct_identifiers"]:
                if data.get(key) is True:
                    problems.append(f"manifest says {key}=true")
        except Exception as e:
            problems.append(f"manifest.json is not valid JSON: {e}")
    total = 0
    count = 0
    examples = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        srel = str(rel)
        if p.is_dir():
            continue
        count += 1
        total += p.stat().st_size
        if p.name.startswith("._") or DENY_NAME_RE.search(p.name) or DIRECT_ID_HINT_RE.search(srel):
            problems.append(f"denied/suspicious path: {srel}")
        if p.suffix.lower() in WARN_EXT and p.name != "dataset-card.md":
            warnings.append(f"text/document-like file present: {srel}")
        if len(examples) < 20:
            examples.append(srel)
    report = {
        "root": str(root),
        "file_count": count,
        "total_bytes": total,
        "strict": args.strict,
        "problems": problems,
        "warnings": warnings[:50],
        "ok": not problems,
        "example_paths": examples,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["root", "file_count", "total_bytes", "strict", "problems", "warnings", "ok"]}, indent=2))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
