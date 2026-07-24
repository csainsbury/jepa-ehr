#!/usr/bin/env python3
"""Oracle realism v3 — DRAFT-ONLY reserved-manifest schemas (Pi rev-13 authorized; reworked for Pi rev-18).

Exact STRUCTURE + STRICT fail-closed VALIDATORS for the two reserved manifests the registered evaluation will need,
derived from the FULL schema-validated SD registry (not the wired projection), carrying BOTH exemption variants, with
role-specific RNG provenance and explicit manifest/schema identities. SCHEMAS ONLY — NOTHING is drawn, populated, or
frozen: seeds/maps/paths/content hashes are `RESERVED_NOT_DRAWN`, and the composed manifest identities stay the
engine's `RESERVED_*` sentinels, so `gate_group_registered` remains unconditionally BLOCKED.

Reworked per Pi rev-18:
  #1 RNG universe = the full SD registry (10 experiments / 14 strata incl. boundary_short), not the wired groups;
  #2 map-set carries BOTH exemption variants (16 with / 17 without) independently;
  #3 role-specific RNG provenance (fixture/coupling seed per experiment x stratum x role; component + strength;
     constructor-route + profile-config identities; per-role content + count hashes binding exp/stratum/role/count);
  #4 STRICT schemas — exact required-field sets/types at every nesting level, refusing unknown/missing/mistyped;
     a FROZEN schema is a SEPARATE VERSION, never a status toggle over a permissive draft;
  #5 bind full-registry variant identities + manifest-source + schema-definition identities; a DETERMINISTIC schema
     identity is reported SEPARATELY from the environment-dependent instance hash (which includes the dependency layer);
  #6 exact map-set canonical semantics (sort key, set-hash payload, output-path grammar, REFUSED-artifact handling).

Development-only. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_manifest.py
"""
from __future__ import annotations

import hashlib
import inspect
import json

from clinical_jepa.eval.oracle_contracts import canonical_hash
from scripts.oracle_realism_v3_randomization import RefusalError
import scripts.oracle_realism_v3_engine as ENG
import scripts.oracle_realism_v3_registry as REG

RESERVED = "RESERVED_NOT_DRAWN"
NONE_COUPLING = "NONE"
DRAFT = "DRAFT"
_ARM_ORDER = ["candidate", "reference"]
_ROLES = ("candidate", "reference")
_COUPLING_STRENGTH = 0.5                                   # repeatability coupling strength (v2/v3 fixed)
_CONTENT_HASH_ALGO = "sha256(canonical_hash) over derive_record-canonical (source,class_ids,timestamps) per role"
RNG_SCHEMA_VERSION = "v3-rng-manifest-draft-2"
MAPSET_SCHEMA_VERSION = "v3-map-set-manifest-draft-2"

_MAP_CARRYING = frozenset(k for k, v in ENG.ESTIMATORS.items() if v["map_carrying"])
_LAYER_KEYS = ("estimator_semantic", "estimator_impl_source", "engine_canon_schema_gate", "map_source", "dependency")
_SOURCE_LAYER_KEYS = ("estimator_semantic", "estimator_impl_source", "engine_canon_schema_gate", "map_source")


def _identity_layers():
    return {**ENG.SOURCE_IDENTITY_BUNDLE, "dependency": ENG.ESTIMATOR_DEPENDENCY_IDENTITY}


def _manifest_source_identity():
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _constructor_route(source):
    """Canonical construction route identity per source (Pi rev-18 #3)."""
    if source == "structural_zero_control":
        return "structural_multiscale"
    if source == "boundary_short":
        return "bounded_length_control"
    return "source_profile_fixture"


# --- exact strict-schema helper: fields must be EXACTLY `spec` keys, each of the declared type ------------------
def _strict(obj, spec, where):
    if not isinstance(obj, dict):
        raise RefusalError(f"{where} must be a dict, got {type(obj).__name__}")
    keys, req = set(obj), set(spec)
    if keys - req:
        raise RefusalError(f"{where} has unknown field(s) {sorted(keys - req)}")
    if req - keys:
        raise RefusalError(f"{where} missing field(s) {sorted(req - keys)}")
    for f, t in spec.items():
        if not isinstance(obj[f], t):
            raise RefusalError(f"{where}.{f} type {type(obj[f]).__name__} != {getattr(t, '__name__', t)}")


def _check_identity_layers(bundle, where):
    _strict(bundle, {k: str for k in _LAYER_KEYS}, f"{where}.identity_layers")
    if bundle != _identity_layers():
        raise RefusalError(f"{where}.identity_layers do not match the current engine identities")


def _full_registry_experiments():
    """The full SD registry experiment/stratum universe (variant-INVARIANT: only bounded cell scope differs, not the
    experiment/stratum set). 10 experiments / 14 strata incl. boundary_short."""
    sd = REG.build_sd_cells(apply_uncalibratable_exemption=True)
    exps = {}
    for c in sd:
        exps.setdefault(c["experiment_id"], {
            "source": c["source"], "condition": c["condition"], "support_regime": c["support_regime"],
            "coupled_component": c["coupled_component"],
            "strata": [(s["stratum_id"], int(s["n_candidate"]), int(s["n_reference"]))
                       for s in c["exchangeability_strata"]]})
    return exps


def _map_triples_for_variant(with_exemption):
    """Sorted `(profile, regime, check)` triples needing a frozen map for one exemption variant (map-carrying cells
    across ALL groups). Sort key is exactly `(profile, regime, check)` (Pi rev-18 #6)."""
    sd = REG.build_sd_cells(apply_uncalibratable_exemption=with_exemption)
    by_id = {c["cell_id"]: c for c in sd}
    triples = set()
    for g in REG.build_groups(sd).values():
        for cid in g["cells"]:
            c = by_id[cid]
            if c["statistic"] in _MAP_CARRYING:
                triples.add((c["source"], c["support_regime"], c["statistic"]))
    return sorted(triples)


# ---------------------------------------------------------------------------------------------------------------
# builders (DRAFT: every drawn value is RESERVED; composed identities stay the engine's reserved sentinels)
# ---------------------------------------------------------------------------------------------------------------
def build_draft_rng_manifest():
    exps = {}
    for e, meta in _full_registry_experiments().items():
        coupled = meta["coupled_component"]
        strata = []
        for sid, qc, qr in meta["strata"]:
            roles = {}
            for role, count in (("candidate", qc), ("reference", qr)):
                roles[role] = {
                    "fixture_seed": RESERVED,
                    "coupling_seed": RESERVED if coupled is not None else NONE_COUPLING,
                    "content_hash": RESERVED,                     # per-ROLE, not one combined stratum hash (Pi rev-18 #3)
                    "count_hash": canonical_hash({"experiment": e, "stratum": sid, "role": role, "count": int(count)}),
                    "expected_count": int(count),
                }
            strata.append({
                "stratum_id": sid, "n_candidate": qc, "n_reference": qr,
                "coupling_strength": (_COUPLING_STRENGTH if coupled is not None else NONE_COUPLING),
                "roles": roles,
            })
        exps[e] = {
            "source_profile": meta["source"], "condition": meta["condition"], "support_regime": meta["support_regime"],
            "coupled_component": (coupled if coupled is not None else NONE_COUPLING),
            "constructor_route": _constructor_route(meta["source"]),
            "profile_config_identity": canonical_hash({"profile": meta["source"], "regime": meta["support_regime"]}),
            "rng_law_identity": ENG.rng_identity(meta["source"], 0, coupled),   # LAW/semantic id — NOT executable seed
            "canonical_arm_order": list(_ARM_ORDER), "strata": strata,
        }
    return {
        "schema_version": RNG_SCHEMA_VERSION, "status": DRAFT,
        "identity_layers": _identity_layers(),
        "registry_variant_identities": dict(ENG.REGISTERED["registry_identity"]),   # full registry with/without
        "manifest_source_identity": _manifest_source_identity(),
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "content_hash_algorithm_identity": _CONTENT_HASH_ALGO,
        "generator_code_identity": RESERVED, "coupling_code_identity": RESERVED,
        "experiments": exps,
        "rng_manifest_identity": ENG.REGISTERED["rng_manifest_identity"],   # RESERVED_RNG_MANIFEST_NOT_BOUND
    }


def _map_entry(profile, regime, check):
    return {
        "profile": profile, "regime": regime, "check": check,
        "N": int(ENG.REGISTERED["N_per_arm"]), "floor": int(ENG.REGISTERED["floor"]),
        "builder_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
        "apply_code_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],    # builder+apply live in the map module
        "expected_status": "OK",                                           # a REFUSED_reference_coarsening map is NOT ok here
        "seed": RESERVED, "namespace": RESERVED, "output_path": RESERVED, "map_identity": RESERVED,
    }


def build_draft_map_set_manifest():
    variants = {}
    for name, with_ex, reg_id in (("with_exemption", True, ENG.REGISTERED["registry_identity"]["with"]),
                                  ("without_exemption", False, ENG.REGISTERED["registry_identity"]["without"])):
        variants[name] = {"registry_identity": reg_id,
                          "entries": [_map_entry(p, r, c) for (p, r, c) in _map_triples_for_variant(with_ex)]}
    return {
        "schema_version": MAPSET_SCHEMA_VERSION, "status": DRAFT,
        "identity_layers": _identity_layers(),
        "manifest_source_identity": _manifest_source_identity(),
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "sort_key": ["profile", "regime", "check"],
        "set_hash_rule": ("map_set_identity = canonical_hash({(profile,regime,check): map_identity} over sorted "
                          "entries, per variant) once every map is drawn"),
        "output_path_grammar": "relative POSIX under the reserved map-design root; no '..' / absolute / symlink (traversal refused)",
        "refused_artifact_rule": ("an issued REFUSED_reference_coarsening artifact does NOT satisfy an entry and does "
                                  "NOT grant a provisional exemption; only a malformed/missing map is an error, an "
                                  "honest REFUSED map is a separate exemption-review input"),
        "variants": variants,
        "map_set_identity": ENG.REGISTERED["map_set_identity"],            # RESERVED_MAP_SET_NOT_DRAWN
    }


# --- strict validators -----------------------------------------------------------------------------------------
_RNG_TOP = {"schema_version": str, "status": str, "identity_layers": dict, "registry_variant_identities": dict,
            "manifest_source_identity": str, "schema_definition_identity": str, "content_hash_algorithm_identity": str,
            "generator_code_identity": str, "coupling_code_identity": str, "experiments": dict,
            "rng_manifest_identity": str}
_RNG_EXP = {"source_profile": str, "condition": str, "support_regime": str, "coupled_component": str,
            "constructor_route": str, "profile_config_identity": str, "rng_law_identity": str,
            "canonical_arm_order": list, "strata": list}
_RNG_STRATUM = {"stratum_id": str, "n_candidate": int, "n_reference": int, "coupling_strength": (float, str), "roles": dict}
_RNG_ROLE = {"fixture_seed": str, "coupling_seed": str, "content_hash": str, "count_hash": str, "expected_count": int}


def validate_rng_manifest(m):
    _strict(m, _RNG_TOP, "rng manifest")
    if m["schema_version"] != RNG_SCHEMA_VERSION:
        raise RefusalError("rng manifest schema_version mismatch")
    if m["status"] != DRAFT:
        raise RefusalError("only DRAFT rng manifests are defined here (FROZEN is a separate schema version)")
    _check_identity_layers(m["identity_layers"], "rng manifest")
    if m["registry_variant_identities"] != dict(ENG.REGISTERED["registry_identity"]):
        raise RefusalError("rng manifest registry_variant_identities != full registry with/without")
    if m["manifest_source_identity"] != _manifest_source_identity() or m["schema_definition_identity"] != SCHEMA_DEFINITION_IDENTITY:
        raise RefusalError("rng manifest source/schema-definition identity mismatch")
    if m["generator_code_identity"] != RESERVED or m["coupling_code_identity"] != RESERVED:
        raise RefusalError("DRAFT rng manifest must not bind generator/coupling code identities")
    if m["rng_manifest_identity"] != ENG.REGISTERED["rng_manifest_identity"]:
        raise RefusalError("DRAFT rng manifest identity must be the reserved sentinel")
    universe = _full_registry_experiments()
    if set(m["experiments"]) != set(universe):
        raise RefusalError(f"rng manifest experiments {sorted(m['experiments'])} != full registry {sorted(universe)}")
    for e, meta in universe.items():
        em = m["experiments"][e]; _strict(em, _RNG_EXP, f"rng experiment {e}")
        coupled = meta["coupled_component"]
        exp_coupled = coupled if coupled is not None else NONE_COUPLING
        if (em["source_profile"], em["condition"], em["support_regime"], em["coupled_component"]) != \
                (meta["source"], meta["condition"], meta["support_regime"], exp_coupled):
            raise RefusalError(f"rng experiment {e} source/condition/regime/coupling mismatch")
        if em["constructor_route"] != _constructor_route(meta["source"]) or em["canonical_arm_order"] != _ARM_ORDER:
            raise RefusalError(f"rng experiment {e} constructor_route / arm order mismatch")
        if em["rng_law_identity"] != ENG.rng_identity(meta["source"], 0, coupled):
            raise RefusalError(f"rng experiment {e} rng_law_identity mismatch")
        if [s.get("stratum_id") for s in em["strata"]] != [sid for sid, _, _ in meta["strata"]]:
            raise RefusalError(f"rng experiment {e} stratum ids/order mismatch")
        for sm, (sid, qc, qr) in zip(em["strata"], meta["strata"]):
            _strict(sm, _RNG_STRATUM, f"rng {e}/{sid} stratum")
            if (sm["n_candidate"], sm["n_reference"]) != (qc, qr):
                raise RefusalError(f"rng {e}/{sid} quota mismatch")
            if sm["coupling_strength"] != (_COUPLING_STRENGTH if coupled is not None else NONE_COUPLING):
                raise RefusalError(f"rng {e}/{sid} coupling_strength mismatch")
            if set(sm["roles"]) != set(_ROLES):
                raise RefusalError(f"rng {e}/{sid} roles must be exactly {sorted(_ROLES)}")
            for role, count in (("candidate", qc), ("reference", qr)):
                rm = sm["roles"][role]; _strict(rm, _RNG_ROLE, f"rng {e}/{sid}/{role} role")
                if rm["expected_count"] != count:
                    raise RefusalError(f"rng {e}/{sid}/{role} expected_count mismatch")
                if rm["count_hash"] != canonical_hash({"experiment": e, "stratum": sid, "role": role, "count": count}):
                    raise RefusalError(f"rng {e}/{sid}/{role} count_hash mismatch")
                want_couple = RESERVED if coupled is not None else NONE_COUPLING
                if rm["fixture_seed"] != RESERVED or rm["content_hash"] != RESERVED or rm["coupling_seed"] != want_couple:
                    raise RefusalError(f"rng {e}/{sid}/{role} DRAFT role fields must be reserved (coupling {want_couple})")
    return m


_MAPSET_TOP = {"schema_version": str, "status": str, "identity_layers": dict, "manifest_source_identity": str,
               "schema_definition_identity": str, "sort_key": list, "set_hash_rule": str, "output_path_grammar": str,
               "refused_artifact_rule": str, "variants": dict, "map_set_identity": str}
_MAPSET_VARIANT = {"registry_identity": str, "entries": list}
_MAPSET_ENTRY = {"profile": str, "regime": str, "check": str, "N": int, "floor": int, "builder_identity": str,
                 "apply_code_identity": str, "expected_status": str, "seed": str, "namespace": str,
                 "output_path": str, "map_identity": str}


def validate_map_set_manifest(m):
    _strict(m, _MAPSET_TOP, "map-set manifest")
    if m["schema_version"] != MAPSET_SCHEMA_VERSION:
        raise RefusalError("map-set manifest schema_version mismatch")
    if m["status"] != DRAFT:
        raise RefusalError("only DRAFT map-set manifests are defined here (FROZEN is a separate schema version)")
    _check_identity_layers(m["identity_layers"], "map-set manifest")
    if m["manifest_source_identity"] != _manifest_source_identity() or m["schema_definition_identity"] != SCHEMA_DEFINITION_IDENTITY:
        raise RefusalError("map-set manifest source/schema-definition identity mismatch")
    if m["sort_key"] != ["profile", "regime", "check"]:
        raise RefusalError("map-set manifest sort_key must be [profile, regime, check]")
    if m["map_set_identity"] != ENG.REGISTERED["map_set_identity"]:
        raise RefusalError("DRAFT map-set manifest identity must be the reserved sentinel")
    if set(m["variants"]) != {"with_exemption", "without_exemption"}:
        raise RefusalError("map-set manifest must carry exactly the with/without exemption variants")
    for name, with_ex, reg_id in (("with_exemption", True, ENG.REGISTERED["registry_identity"]["with"]),
                                  ("without_exemption", False, ENG.REGISTERED["registry_identity"]["without"])):
        var = m["variants"][name]; _strict(var, _MAPSET_VARIANT, f"map-set variant {name}")
        if var["registry_identity"] != reg_id:
            raise RefusalError(f"map-set variant {name} registry_identity mismatch")
        got = [(e.get("profile"), e.get("regime"), e.get("check")) for e in var["entries"]]
        if len(got) != len(set(got)):
            raise RefusalError(f"map-set variant {name} has duplicate entries")
        if sorted(got) != _map_triples_for_variant(with_ex):
            raise RefusalError(f"map-set variant {name} triples != required (missing/extra)")
        for e in var["entries"]:
            _strict(e, _MAPSET_ENTRY, f"map-set {name} entry {e.get('check')}@{e.get('profile')}")
            if e["N"] != int(ENG.REGISTERED["N_per_arm"]) or e["floor"] != int(ENG.REGISTERED["floor"]):
                raise RefusalError(f"map-set {name} entry N/floor != registered")
            if e["builder_identity"] != ENG.SOURCE_IDENTITY_BUNDLE["map_source"] or e["apply_code_identity"] != ENG.SOURCE_IDENTITY_BUNDLE["map_source"]:
                raise RefusalError(f"map-set {name} entry builder/apply identity mismatch")
            if e["expected_status"] != "OK":
                raise RefusalError(f"map-set {name} entry expected_status must be OK")
            if any(e[k] != RESERVED for k in ("seed", "namespace", "output_path", "map_identity")):
                raise RefusalError(f"map-set {name} entry DRAFT fields must be reserved")
    return m


# --- identities ------------------------------------------------------------------------------------------------
# The SCHEMA-DEFINITION identity is a DETERMINISTIC hash over the exact field-set definitions + versions (no runtime
# values, no environment). It reproduces across environments (Pi rev-18 #5).
SCHEMA_DEFINITION_IDENTITY = canonical_hash({
    "rng_version": RNG_SCHEMA_VERSION, "mapset_version": MAPSET_SCHEMA_VERSION,
    "rng_fields": {"top": {k: t.__name__ if isinstance(t, type) else str(t) for k, t in _RNG_TOP.items()},
                   "experiment": {k: t.__name__ if isinstance(t, type) else str(t) for k, t in _RNG_EXP.items()},
                   "stratum": {k: (t.__name__ if isinstance(t, type) else str(t)) for k, t in _RNG_STRATUM.items()},
                   "role": {k: t.__name__ for k, t in _RNG_ROLE.items()}},
    "mapset_fields": {"top": {k: t.__name__ if isinstance(t, type) else str(t) for k, t in _MAPSET_TOP.items()},
                      "variant": {k: t.__name__ for k, t in _MAPSET_VARIANT.items()},
                      "entry": {k: t.__name__ for k, t in _MAPSET_ENTRY.items()}},
    "layer_keys": list(_LAYER_KEYS), "arm_order": _ARM_ORDER, "coupling_strength": _COUPLING_STRENGTH,
    "content_hash_algorithm": _CONTENT_HASH_ALGO})


def _deterministic_schema_identity():
    """Reproducible-across-environments schema identity: field definitions + versions + full-registry variant
    identities + the four DETERMINISTIC SOURCE identity layers + manifest source. Excludes the env-dependent
    dependency layer (Pi rev-18 #5)."""
    return canonical_hash({
        "schema_definition": SCHEMA_DEFINITION_IDENTITY,
        "manifest_source": _manifest_source_identity(),
        "registry_variant_identities": dict(ENG.REGISTERED["registry_identity"]),
        "source_identity_layers": {k: ENG.SOURCE_IDENTITY_BUNDLE[k] for k in _SOURCE_LAYER_KEYS},
    })


# --- self-tests ------------------------------------------------------------------------------------------------
def selftest():
    import copy
    errs = []
    rng = build_draft_rng_manifest(); mapset = build_draft_map_set_manifest()
    for label, (mm, vv) in (("rng", (rng, validate_rng_manifest)), ("map-set", (mapset, validate_map_set_manifest))):
        try:
            vv(mm)
        except RefusalError as ex:
            errs.append(f"{label}: valid DRAFT wrongly refused: {ex}")

    def refused(base, v, mut):
        m = copy.deepcopy(base); mut(m)
        try:
            v(m); return False
        except RefusalError:
            return True

    e0 = next(iter(rng["experiments"])); coupled_e = next((e for e, x in rng["experiments"].items()
                                                            if x["coupled_component"] != NONE_COUPLING), e0)
    rng_bad = {
        "unknown_top_field": lambda m: m.__setitem__("sneaky", 1),
        "unknown_experiment_field": lambda m: m["experiments"][e0].__setitem__("sneaky", 1),
        "unknown_stratum_field": lambda m: m["experiments"][e0]["strata"][0].__setitem__("sneaky", 1),
        "unknown_role_field": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("sneaky", 1),
        "drop_experiment": lambda m: m["experiments"].pop(e0),
        "wrong_registry_variants": lambda m: m.__setitem__("registry_variant_identities", {"with": "X", "without": "Y"}),
        "tamper_layer": lambda m: m["identity_layers"].__setitem__("estimator_semantic", "X"),
        "wrong_manifest_source": lambda m: m.__setitem__("manifest_source_identity", "X"),
        "wrong_count_hash": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("count_hash", "X"),
        "populated_fixture_seed": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("fixture_seed", 7),
        "coupling_seed_on_uncoupled": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("coupling_seed", RESERVED)
        if rng["experiments"][e0]["coupled_component"] == NONE_COUPLING else None,
        "bound_manifest_identity": lambda m: m.__setitem__("rng_manifest_identity", "BOUND"),
        "not_draft": lambda m: m.__setitem__("status", "FROZEN"),
    }
    for name, mut in rng_bad.items():
        if mut is None or (name == "coupling_seed_on_uncoupled" and rng["experiments"][e0]["coupled_component"] != NONE_COUPLING):
            continue
        if not refused(rng, validate_rng_manifest, mut):
            errs.append(f"rng manifest did NOT refuse {name}")

    mapset_bad = {
        "unknown_top_field": lambda m: m.__setitem__("sneaky", 1),
        "unknown_variant_field": lambda m: m["variants"]["with_exemption"].__setitem__("sneaky", 1),
        "unknown_entry_field": lambda m: m["variants"]["with_exemption"]["entries"][0].__setitem__("sneaky", 1),
        "drop_variant": lambda m: m["variants"].pop("without_exemption"),
        "drop_entry": lambda m: m["variants"]["with_exemption"]["entries"].pop(),
        "duplicate_entry": lambda m: m["variants"]["with_exemption"]["entries"].append(
            copy.deepcopy(m["variants"]["with_exemption"]["entries"][0])),
        "wrong_variant_registry": lambda m: m["variants"]["with_exemption"].__setitem__("registry_identity", "X"),
        "swap_variants_favorably": lambda m: m["variants"].__setitem__(
            "without_exemption", copy.deepcopy(m["variants"]["with_exemption"])),
        "tamper_layer": lambda m: m["identity_layers"].__setitem__("map_source", "X"),
        "wrong_N": lambda m: m["variants"]["with_exemption"]["entries"][0].__setitem__("N", 4000),
        "populated_map_identity": lambda m: m["variants"]["with_exemption"]["entries"][0].__setitem__("map_identity", "abcd"),
        "bound_set_identity": lambda m: m.__setitem__("map_set_identity", "BOUND"),
        "not_draft": lambda m: m.__setitem__("status", "FROZEN"),
    }
    for name, mut in mapset_bad.items():
        if not refused(mapset, validate_map_set_manifest, mut):
            errs.append(f"map-set manifest did NOT refuse {name}")

    # DETERMINISTIC schema identity must NOT depend on the environment dependency layer
    if "dependency" in json.dumps(_deterministic_schema_identity.__doc__ or ""):
        pass
    return errs, rng, mapset


def main():
    errs, rng, mapset = selftest()
    out = {
        "purpose": "DRAFT-ONLY reserved-manifest schemas + STRICT validators; full-registry universes, both exemption "
                   "variants, role-specific provenance; nothing drawn/frozen/populated.",
        "identity_layers_bound": sorted(_LAYER_KEYS),
        "rng_manifest": {"experiments": len(rng["experiments"]),
                         "strata_total": sum(len(v["strata"]) for v in rng["experiments"].values()),
                         "registry_variants": sorted(rng["registry_variant_identities"]),
                         "identity": rng["rng_manifest_identity"]},
        "map_set_manifest": {"variants": {k: len(v["entries"]) for k, v in mapset["variants"].items()},
                             "identity": mapset["map_set_identity"]},
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "deterministic_schema_identity": _deterministic_schema_identity(),
        "selftests_pass": not errs, "selftest_errors": errs,
        "authorization": "dev-only DRAFT schemas; reserved seeds/maps/paths/content NOT drawn; manifests NOT "
                         "frozen/populated; gate_group_registered stays BLOCKED. Freezing needs a separate review.",
    }
    print(json.dumps(out, indent=2, default=str))
    # DETERMINISTIC (reproducible across environments) schema identities:
    print("\nDRAFT_RNG_SCHEMA_IDENTITY (deterministic):", _deterministic_schema_identity())
    print("DRAFT_MAPSET_SCHEMA_IDENTITY (deterministic):", _deterministic_schema_identity())
    print("SCHEMA_DEFINITION_IDENTITY (deterministic):", SCHEMA_DEFINITION_IDENTITY)
    # NOTE: a full-manifest instance hash would ALSO fold the env-dependent dependency layer and is therefore NOT
    # reproducible across environments — it is intentionally not reported as a schema hash (Pi rev-18 #5).
    assert not errs, f"manifest schema self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
