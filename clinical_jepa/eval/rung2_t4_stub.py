"""Rung-2 T4 (learned VQ order-target) CLI STUB — FAIL-CLOSED (Pi v2 #6/#7).

T4 is barred on governed data until the semi-synthetic oracle is separately blueprinted, Cog-
preflighted, Pi-gated, and frozen. This stub exists so the T4 entry point REFUSES governed inputs
without a frozen oracle authorization manifest — safe-public/synthetic scaffolding only.
"""
from __future__ import annotations

import argparse
from typing import Any

from clinical_jepa.eval.rung2_contract import T4_TARGET, requires_oracle, t4_governed_allowed


def guard_t4(*, inputs_are_governed: bool, oracle_authorization: dict[str, Any] | None,
             presented_recipe_hash: str | None = None, expected_blueprint_hash: str | None = None) -> None:
    """Raise unless the T4 run is permitted (synthetic always OK; governed needs a frozen, Pi-gated
    oracle manifest that CERTIFIES the EXACT presented recipe + blueprint — both mandatory)."""
    assert requires_oracle(T4_TARGET)
    if not t4_governed_allowed(inputs_are_governed, oracle_authorization,
                               presented_recipe_hash=presented_recipe_hash,
                               expected_blueprint_hash=expected_blueprint_hash):
        raise PermissionError(
            "T4 learned-target work on GOVERNED data is barred until a frozen, Pi-gated semi-"
            "synthetic oracle authorization manifest exists (oracle_frozen + pi_gate=PASS).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-2 T4 learned-VQ order target (BARRED on governed data)")
    ap.add_argument("--governed", action="store_true", help="inputs are governed real data")
    ap.add_argument("--oracle-manifest", default=None, help="path to a frozen Pi-gated oracle authorization manifest")
    ap.add_argument("--recipe-hash", default=None, help="the exact T4 recipe hash being run (MANDATORY for governed)")
    ap.add_argument("--blueprint-hash", default=None, help="the expected oracle blueprint hash (MANDATORY for governed)")
    args = ap.parse_args(argv)
    auth = None
    if args.oracle_manifest:
        import json
        from pathlib import Path
        auth = json.loads(Path(args.oracle_manifest).read_text())
    guard_t4(inputs_are_governed=bool(args.governed), oracle_authorization=auth,
             presented_recipe_hash=args.recipe_hash, expected_blueprint_hash=args.blueprint_hash)
    print("T4 permitted (synthetic/safe-public or oracle-authorized). No governed training runs in this build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
