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
             presented_recipe_hash: str | None = None) -> None:
    """Raise unless the T4 run is permitted (synthetic always OK; governed needs a frozen, Pi-gated
    oracle manifest that CERTIFIES the EXACT presented recipe AND matches the TRUSTED COMMITTED
    oracle policy). The blueprint/gate/mechanism trust anchors come from the committed policy — NOT
    from the caller — so only the presented recipe hash is a run input (Pi consolidated #1)."""
    assert requires_oracle(T4_TARGET)
    if not t4_governed_allowed(inputs_are_governed, oracle_authorization,
                               presented_recipe_hash=presented_recipe_hash):
        raise PermissionError(
            "T4 learned-target work on GOVERNED data is barred until a frozen, Pi-gated semi-"
            "synthetic oracle authorization manifest matches the committed approved-oracle policy.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-2 T4 learned-VQ order target (BARRED on governed data)")
    ap.add_argument("--governed", action="store_true", help="inputs are governed real data")
    ap.add_argument("--oracle-manifest", default=None, help="path to a frozen Pi-gated oracle authorization manifest")
    ap.add_argument("--recipe-hash", default=None, help="the exact T4 recipe hash being run (MANDATORY for governed)")
    # NOTE: no --blueprint-hash. The blueprint/gate/mechanism trust anchors are read from the
    # committed clinical_jepa.eval.oracle_policy, never accepted from the run operator (Pi #1).
    args = ap.parse_args(argv)
    auth = None
    if args.oracle_manifest:
        import json
        from pathlib import Path
        auth = json.loads(Path(args.oracle_manifest).read_text())
    guard_t4(inputs_are_governed=bool(args.governed), oracle_authorization=auth,
             presented_recipe_hash=args.recipe_hash)
    print("T4 permitted (synthetic/safe-public or oracle-authorized). No governed training runs in this build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
