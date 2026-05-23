#!/usr/bin/env python3
"""Sanity checks for the Clinical-JEPA autonomous run package."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

DENY_PARTS = {"raw", ".git", ".venv", "node_modules", "__pycache__"}
SECRET_HINTS = ("secret", "credential", "oauth", "token", "password", "private_key")
REQUIRED = [
    "blueprints/atomic-blueprint.md",
    "outcome/outcome-spec-draft.md",
    "physarum/physarum-brief.md",
    "physarum/route-cards.md",
    "prompts/AUTONOMOUS_AGENT_PROMPT.md",
    "scripts/bootstrap_vast.sh",
]

def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
        return 0, out.strip()
    except Exception as e:
        return 1, str(e)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--min-free-gb", type=float, default=100.0)
    ap.add_argument("--require-gpu", action="store_true", help="fail if nvidia-smi is unavailable")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    problems: list[str] = []
    for rel in REQUIRED:
        if not (root / rel).exists():
            problems.append(f"missing required file: {rel}")
    usage = shutil.disk_usage(root)
    free_gb = usage.free / 1e9
    if free_gb < args.min_free_gb:
        problems.append(f"low disk: {free_gb:.1f}GB free < {args.min_free_gb:.1f}GB")
    code, gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if code != 0 and args.require_gpu:
        problems.append("nvidia-smi unavailable")
    denied = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        parts = set(rel.parts)
        name = p.name.lower()
        if parts & DENY_PARTS:
            denied.append(str(rel))
        if any(h in name for h in SECRET_HINTS):
            # .env.example is allowed; actual secret-like paths are not.
            if str(rel) != ".env.example":
                denied.append(str(rel))
    if denied:
        problems.append("denied/suspicious packaged paths: " + ", ".join(denied[:20]))
    report = {
        "root": str(root),
        "free_gb": round(free_gb, 2),
        "gpu": gpu if code == 0 else None,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
