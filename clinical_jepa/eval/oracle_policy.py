"""The TRUSTED, COMMITTED approved-oracle policy (Pi consolidated re-gate #1/#2).

The blueprint/gate trust anchor must NOT be selectable by the run operator: checking a manifest's
blueprint hash against a value the SAME operator supplies proves string equality, not approval. This
committed, read-only allowlist is populated ONLY after the mandatory implementation gate passes and
the oracle blueprint is frozen. While EMPTY (as it is now) every governed T4 is refused — fail-closed.

The PRESENTED RECIPE HASH stays a run input (it names the actual governed recipe, matched against the
manifest's `certified_recipe_hash`); everything else here is the trusted identity the guard checks
the manifest against.
"""
from __future__ import annotations

from typing import Any

# EMPTY until the implementation gate + blueprint freeze. Do NOT populate without a Pi gate.
APPROVED_ORACLE_POLICY: dict[str, Any] = {
    "blueprint_hash": None,          # exact frozen oracle-blueprint artifact hash
    "gate_event_ref": None,          # the approved Pi gate event / thread reference
    "oracle_mechanism_hash": None,   # frozen generator mechanism hash
    "evaluator_commits": [],         # approved evaluator code commit(s)
    "recipe_registry_ids": [],       # approved recipe-registry id(s)
    "sealed_cert_run_ids": [],       # approved sealed-cert run id(s)
    "schema_version": None,          # exact approved authorization schema version
}


def load_approved_oracle_policy() -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in APPROVED_ORACLE_POLICY.items()}


def policy_is_populated(policy: dict[str, Any] | None = None) -> bool:
    """A usable policy has the three scalar anchors set (blueprint, gate ref, mechanism hash)."""
    p = policy if policy is not None else APPROVED_ORACLE_POLICY
    return all(isinstance(p.get(k), str) and p[k] for k in ("blueprint_hash", "gate_event_ref", "oracle_mechanism_hash"))


def manifest_matches_policy(manifest: dict[str, Any], policy: dict[str, Any] | None = None) -> bool:
    """Fail-closed: the manifest's trust-anchor fields must EXACTLY match / be MEMBERS OF the
    committed policy (not merely non-empty; not caller-supplied)."""
    p = policy if policy is not None else APPROVED_ORACLE_POLICY
    if not policy_is_populated(p):
        return False
    m = manifest or {}
    if m.get("blueprint_hash") != p["blueprint_hash"]:
        return False
    if m.get("gate_event_ref") != p["gate_event_ref"]:
        return False
    if m.get("oracle_mechanism_hash") != p["oracle_mechanism_hash"]:
        return False
    if p.get("schema_version") is not None and m.get("schema_version") != p["schema_version"]:
        return False
    if m.get("evaluator_commit") not in (p.get("evaluator_commits") or []):
        return False
    if m.get("recipe_registry_id") not in (p.get("recipe_registry_ids") or []):
        return False
    if m.get("sealed_cert_run_id") not in (p.get("sealed_cert_run_ids") or []):
        return False
    return True
