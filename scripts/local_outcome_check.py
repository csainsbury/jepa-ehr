#!/usr/bin/env python3
"""Deterministic local checks for the first Clinical-JEPA protocol workspace."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REQUIRED = [
    "v0/protocol.md",
    "v0/arms.md",
    "v0/metrics.md",
    "splits/split-spec.md",
    "leakage-rules/leakage-audit-plan.md",
    "data-access-checklist.md",
]
KEYWORDS = {
    "v0/protocol.md": ["v0A", "v0B", "v0C", "v0D", "target"],
    "v0/arms.md": ["FlatASCEND", "raw", "query", "EMA"],
    "v0/metrics.md": ["retrieval", "baseline", "effective rank", "utilisation"],
    "splits/split-spec.md": ["Patient-level", "No patient overlap"],
    "leakage-rules/leakage-audit-plan.md": ["forbidden", "endpoint", "context"],
    "data-access-checklist.md": ["Raw", "Tokenised", "Backblaze"],
}
DENIED_RE = re.compile(r"(patient\s*id\s*[:=]|nhs\s*number|date\s*of\s*birth|aws_secret_access_key\s*=|b2_application_key\s*=\s*\S+|private[-_ ]key\s*[:=]|password\s*[:=]\s*\S+)", re.I)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--spec", default=None)
    args = ap.parse_args()
    root = Path(args.artifact_root).resolve()
    problems=[]
    passed=[]
    for rel in REQUIRED:
        p=root/rel
        if not p.exists():
            problems.append(f"missing required artifact: {rel}")
            continue
        txt=p.read_text(errors="ignore")
        if DENIED_RE.search(txt):
            problems.append(f"sensitive/secret-like text found in {rel}")
        missing=[k for k in KEYWORDS.get(rel,[]) if k.lower() not in txt.lower()]
        if missing:
            problems.append(f"{rel} missing expected terms: {missing}")
        else:
            passed.append(rel)
    report={"artifact_root":str(root),"passed":passed,"problems":problems,"ok":not problems}
    out=root/"reviews/outcome-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if not problems else 1

if __name__=="__main__":
    sys.exit(main())
