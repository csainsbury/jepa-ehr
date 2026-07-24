#!/usr/bin/env python3
"""Oracle realism v3 — DRAFT-ONLY reserved-manifest SCHEMAS (Pi rev-13 authorized scope).

Defines the STRUCTURE + fail-closed VALIDATORS for the two reserved manifests the registered evaluation will
eventually need, each binding the FIVE identity layers (Pi rev-13 #3):

  * the RNG manifest — per experiment x stratum: role/stratum seeds, generator/coupling CODE identities, profile
    identity, content/count hashes, and canonical arm order (Pi rev-8 #5/#6);
  * the reserved map-set manifest — every required (profile, regime, check): seed/namespace/N/floor/builder
    identity + output path + set-hash rule, with missing/extra/duplicate failing closed (Pi rev-7 #6, rev-8 #5/#6).

These are SCHEMAS ONLY. NOTHING is drawn, populated, or frozen: every not-yet-drawn field carries the
`RESERVED_NOT_DRAWN` sentinel, and the composed manifest identities stay `RESERVED_*_NOT_DRAWN` — identical to the
engine's `REGISTERED` placeholders — so `gate_group_registered` remains unconditionally BLOCKED regardless of any
draft built here. A later, separately-authorized review is required to draw the seeds/maps and freeze the manifests.

Development-only. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_manifest.py
"""
from __future__ import annotations

import copy
import json

from clinical_jepa.eval.oracle_contracts import canonical_hash
from scripts.oracle_realism_v3_randomization import RefusalError
import scripts.oracle_realism_v3_engine as ENG

RESERVED = "RESERVED_NOT_DRAWN"
DRAFT = "DRAFT"
_ARM_ORDER = ["candidate", "reference"]


def _identity_layers():
    """The five identity layers bound into every manifest: the four DETERMINISTIC source layers + the
    environment-dependent dependency layer."""
    return {**ENG.SOURCE_IDENTITY_BUNDLE, "dependency": ENG.ESTIMATOR_DEPENDENCY_IDENTITY}


_LAYER_KEYS = ("estimator_semantic", "estimator_impl_source", "engine_canon_schema_gate", "map_source", "dependency")


def _check_identity_layers(m, where):
    layers = m.get("identity_layers")
    if not isinstance(layers, dict) or set(layers) != set(_LAYER_KEYS):
        raise RefusalError(f"{where}: identity_layers must be exactly {sorted(_LAYER_KEYS)}")
    if layers != _identity_layers():
        raise RefusalError(f"{where}: identity_layers do not match the current engine identities")


# ---------------------------------------------------------------------------------------------------------------
# RNG manifest — per experiment x stratum RNG provenance (DRAFT: seeds / content / generator identities reserved).
# ---------------------------------------------------------------------------------------------------------------
def build_draft_rng_manifest():
    """Enumerate the reserved RNG manifest STRUCTURE from the canonical registry. Real, checkable now: the exact
    experiment/stratum set, per-stratum registered quotas, canonical arm order, executable per-experiment
    `rng_identity`, and the count hash. RESERVED (a later authorized draw): the actual per-role seeds, the drawn-
    sequence content hash, and the generator/coupling code identities. The composed manifest identity stays the
    engine's reserved sentinel, so nothing here unblocks a registered run."""
    experiments = {}
    for canon in ENG.CANONICAL_GROUPS.values():
        for e, meta in canon["experiments"].items():
            if e in experiments:                                    # experiments are shared across groups — dedup
                continue
            strata = []
            for sid, (qc, qr) in zip(meta["stratum_ids"], meta["registered_quota"]):
                strata.append({
                    "stratum_id": sid, "n_candidate": int(qc), "n_reference": int(qr),
                    "candidate_seed": RESERVED, "reference_seed": RESERVED,       # drawn under a later authorized review
                    "rng_identity": ENG.rng_identity(meta["source"], 0, meta["coupled_component"]),
                    "content_hash": RESERVED,                                     # hash of the actual drawn sequences
                    "count_hash": canonical_hash({"n_candidate": int(qc), "n_reference": int(qr)}),
                })
            experiments[e] = {"source_profile": meta["source"], "coupled_component": meta["coupled_component"],
                              "canonical_arm_order": list(_ARM_ORDER), "strata": strata}
    return {
        "schema_version": "v3-rng-manifest-draft-1", "status": DRAFT,
        "identity_layers": _identity_layers(),
        "registry_identity": ENG.CANONICAL_REGISTRY_HASH,
        "generator_code_identity": RESERVED,                        # bound when the generator/coupling modules are pinned
        "coupling_code_identity": RESERVED,
        "experiments": experiments,
        "rng_manifest_identity": ENG.REGISTERED["rng_manifest_identity"],   # RESERVED_RNG_MANIFEST_NOT_BOUND
    }


def validate_rng_manifest(m):
    """Fail-closed structural validation of a DRAFT RNG manifest (a FROZEN manifest is a later, separately-reviewed
    schema). Refuses missing/extra/duplicate experiments or strata, wrong quotas/order, a wrong per-experiment
    `rng_identity`, a wrong `count_hash`, mismatched identity layers, or a draft that pretends to be populated."""
    if not isinstance(m, dict):
        raise RefusalError("rng manifest is not a dict")
    if m.get("schema_version") != "v3-rng-manifest-draft-1":
        raise RefusalError("rng manifest schema_version mismatch")
    if m.get("status") != DRAFT:
        raise RefusalError("only DRAFT rng manifests are defined here (freezing is a later reviewed step)")
    _check_identity_layers(m, "rng manifest")
    if m.get("registry_identity") != ENG.CANONICAL_REGISTRY_HASH:
        raise RefusalError("rng manifest registry_identity mismatch")
    # DRAFT: seeds / content / generator identities are NOT drawn; the composed identity is the reserved sentinel.
    if m.get("generator_code_identity") != RESERVED or m.get("coupling_code_identity") != RESERVED:
        raise RefusalError("DRAFT rng manifest must not bind generator/coupling code identities yet")
    if m.get("rng_manifest_identity") != ENG.REGISTERED["rng_manifest_identity"]:
        raise RefusalError("DRAFT rng manifest identity must be the reserved (not-bound) sentinel")
    exps = m.get("experiments")
    if not isinstance(exps, dict):
        raise RefusalError("rng manifest experiments must be a dict")
    canon_meta = {}
    for canon in ENG.CANONICAL_GROUPS.values():
        canon_meta.update(canon["experiments"])
    if set(exps) != set(canon_meta):
        raise RefusalError(f"rng manifest experiments {sorted(exps)} != canonical {sorted(canon_meta)}")
    for e, meta in canon_meta.items():
        em = exps[e]
        if em.get("source_profile") != meta["source"] or em.get("coupled_component") != meta["coupled_component"]:
            raise RefusalError(f"rng manifest experiment {e} source/coupling mismatch")
        if em.get("canonical_arm_order") != _ARM_ORDER:
            raise RefusalError(f"rng manifest experiment {e} arm order must be {_ARM_ORDER}")
        strata = em.get("strata")
        if not isinstance(strata, list) or [s.get("stratum_id") for s in strata] != list(meta["stratum_ids"]):
            raise RefusalError(f"rng manifest experiment {e} stratum ids/order mismatch")
        for s, (qc, qr) in zip(strata, meta["registered_quota"]):
            if (s.get("n_candidate"), s.get("n_reference")) != (int(qc), int(qr)):
                raise RefusalError(f"rng manifest {e}/{s.get('stratum_id')} quota mismatch")
            if s.get("rng_identity") != ENG.rng_identity(meta["source"], 0, meta["coupled_component"]):
                raise RefusalError(f"rng manifest {e}/{s.get('stratum_id')} rng_identity mismatch")
            if s.get("count_hash") != canonical_hash({"n_candidate": int(qc), "n_reference": int(qr)}):
                raise RefusalError(f"rng manifest {e}/{s.get('stratum_id')} count_hash mismatch")
            if s.get("candidate_seed") != RESERVED or s.get("reference_seed") != RESERVED or s.get("content_hash") != RESERVED:
                raise RefusalError(f"rng manifest {e}/{s.get('stratum_id')} DRAFT seeds/content must be reserved")
    return m


# ---------------------------------------------------------------------------------------------------------------
# Map-set manifest — every required (profile, regime, check) frozen map (DRAFT: seeds/paths/map identities reserved).
# ---------------------------------------------------------------------------------------------------------------
def _required_map_triples():
    """Every (source_profile, regime, check) the registered evaluation needs a frozen coarsening map for — the
    map-carrying cells across ALL wired groups, deduped + sorted (canonical enumeration; missing/extra fail-closed)."""
    triples = set()
    for canon in ENG.CANONICAL_GROUPS.values():
        for cc in canon["cells"]:
            if cc["map_carrying"]:
                triples.add((canon["experiments"][cc["exp"]]["source"], "full", cc["check"]))
    return sorted(triples)


def build_draft_map_set_manifest():
    """Enumerate the reserved map-set manifest STRUCTURE. Real now: the exact required (profile,regime,check) set,
    registered N/floor, and the map-builder SOURCE identity. RESERVED: the reserved-namespace seed, the map-design
    namespace, the output path, and each drawn map's `map_identity`. The composed set identity stays the engine's
    reserved sentinel."""
    entries = []
    for (profile, regime, check) in _required_map_triples():
        entries.append({
            "profile": profile, "regime": regime, "check": check,
            "seed": RESERVED, "namespace": RESERVED,
            "N": int(ENG.REGISTERED["N_per_arm"]), "floor": int(ENG.REGISTERED["floor"]),
            "builder_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
            "output_path": RESERVED, "map_identity": RESERVED,     # drawn/bound under a later authorized review
        })
    return {
        "schema_version": "v3-map-set-manifest-draft-1", "status": DRAFT,
        "identity_layers": _identity_layers(),
        "entries": entries,
        "set_hash_rule": "map_set_identity = canonical_hash([e.map_identity for e in sorted entries]) once all maps drawn",
        "map_set_identity": ENG.REGISTERED["map_set_identity"],     # RESERVED_MAP_SET_NOT_DRAWN
    }


def validate_map_set_manifest(m):
    """Fail-closed structural validation of a DRAFT map-set manifest. Refuses missing/extra/duplicate triples, wrong
    N/floor/builder identity, mismatched identity layers, or a draft that pretends to be populated/frozen."""
    if not isinstance(m, dict):
        raise RefusalError("map-set manifest is not a dict")
    if m.get("schema_version") != "v3-map-set-manifest-draft-1":
        raise RefusalError("map-set manifest schema_version mismatch")
    if m.get("status") != DRAFT:
        raise RefusalError("only DRAFT map-set manifests are defined here (freezing is a later reviewed step)")
    _check_identity_layers(m, "map-set manifest")
    if m.get("map_set_identity") != ENG.REGISTERED["map_set_identity"]:
        raise RefusalError("DRAFT map-set manifest identity must be the reserved (not-drawn) sentinel")
    entries = m.get("entries")
    if not isinstance(entries, list):
        raise RefusalError("map-set manifest entries must be a list")
    got = [(e.get("profile"), e.get("regime"), e.get("check")) for e in entries]
    if len(got) != len(set(got)):
        raise RefusalError("map-set manifest has duplicate (profile,regime,check) entries")
    if sorted(got) != _required_map_triples():
        raise RefusalError("map-set manifest triples != required (missing/extra)")
    for e in entries:
        if e.get("N") != int(ENG.REGISTERED["N_per_arm"]) or e.get("floor") != int(ENG.REGISTERED["floor"]):
            raise RefusalError(f"map-set entry {e.get('check')}@{e.get('profile')} N/floor != registered")
        if e.get("builder_identity") != ENG.SOURCE_IDENTITY_BUNDLE["map_source"]:
            raise RefusalError(f"map-set entry {e.get('check')}@{e.get('profile')} builder_identity mismatch")
        if any(e.get(k) != RESERVED for k in ("seed", "namespace", "output_path", "map_identity")):
            raise RefusalError(f"map-set entry {e.get('check')}@{e.get('profile')} DRAFT fields must be reserved")
    return m


# --- self-tests ------------------------------------------------------------------------------------------------
def selftest():
    errs = []
    rng = build_draft_rng_manifest(); mapset = build_draft_map_set_manifest()
    for label, (m, v) in (("rng", (rng, validate_rng_manifest)), ("map-set", (mapset, validate_map_set_manifest))):
        try:
            v(m)
        except RefusalError as ex:
            errs.append(f"{label}: valid DRAFT manifest wrongly refused: {ex}")

    def refused(base, v, mutate):
        m = copy.deepcopy(base); mutate(m)
        try:
            v(m); return False
        except RefusalError:
            return True

    first_exp = next(iter(rng["experiments"]))
    rng_bad = {
        "drop_experiment": lambda m: m["experiments"].pop(first_exp),
        "extra_experiment": lambda m: m["experiments"].__setitem__("BOGUS_EXP", m["experiments"][first_exp]),
        "tamper_layer": lambda m: m["identity_layers"].__setitem__("estimator_semantic", "X"),
        "wrong_registry": lambda m: m.__setitem__("registry_identity", "X"),
        "bad_quota": lambda m: m["experiments"][first_exp]["strata"][0].__setitem__("n_candidate", 1),
        "bad_rng_identity": lambda m: m["experiments"][first_exp]["strata"][0].__setitem__("rng_identity", "X"),
        "populated_seed_in_draft": lambda m: m["experiments"][first_exp]["strata"][0].__setitem__("candidate_seed", 7),
        "bound_manifest_identity": lambda m: m.__setitem__("rng_manifest_identity", "BOUND"),
        "wrong_arm_order": lambda m: m["experiments"][first_exp].__setitem__("canonical_arm_order", ["reference", "candidate"]),
        "not_draft": lambda m: m.__setitem__("status", "FROZEN"),
    }
    for name, mut in rng_bad.items():
        if not refused(rng, validate_rng_manifest, mut):
            errs.append(f"rng manifest did NOT refuse {name}")

    mapset_bad = {
        "drop_entry": lambda m: m["entries"].pop(),
        "duplicate_entry": lambda m: m["entries"].append(copy.deepcopy(m["entries"][0])),
        "extra_triple": lambda m: m["entries"].append({**m["entries"][0], "check": "NOT_A_CHECK"}),
        "tamper_layer": lambda m: m["identity_layers"].__setitem__("map_source", "X"),
        "wrong_N": lambda m: m["entries"][0].__setitem__("N", 4000),
        "bad_builder_identity": lambda m: m["entries"][0].__setitem__("builder_identity", "X"),
        "populated_map_identity": lambda m: m["entries"][0].__setitem__("map_identity", "abcd"),
        "bound_set_identity": lambda m: m.__setitem__("map_set_identity", "BOUND"),
        "not_draft": lambda m: m.__setitem__("status", "FROZEN"),
    }
    for name, mut in mapset_bad.items():
        if not refused(mapset, validate_map_set_manifest, mut):
            errs.append(f"map-set manifest did NOT refuse {name}")
    return errs, rng, mapset


def main():
    errs, rng, mapset = selftest()
    out = {
        "purpose": "DRAFT-ONLY reserved-manifest schemas + validators; nothing drawn/frozen/populated.",
        "identity_layers_bound": sorted(_LAYER_KEYS),
        "rng_manifest": {"experiments": sorted(rng["experiments"]),
                         "n_strata_total": sum(len(v["strata"]) for v in rng["experiments"].values()),
                         "identity": rng["rng_manifest_identity"]},
        "map_set_manifest": {"n_entries": len(mapset["entries"]),
                             "required_triples": _required_map_triples(),
                             "identity": mapset["map_set_identity"]},
        "selftests_pass": not errs, "selftest_errors": errs,
        "authorization": "dev-only DRAFT schemas; reserved seeds/maps NOT drawn; manifests NOT frozen/populated; "
                         "gate_group_registered stays BLOCKED. Freezing needs a separately-authorized review.",
    }
    print(json.dumps(out, indent=2, default=str))
    print("\nDRAFT_RNG_SCHEMA_HASH:", canonical_hash(build_draft_rng_manifest()))
    print("DRAFT_MAPSET_SCHEMA_HASH:", canonical_hash(build_draft_map_set_manifest()))
    assert not errs, f"manifest schema self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
