#!/usr/bin/env python3
"""Oracle realism v3 — DRAFT-ONLY reserved-manifest schemas (Pi rev-13 authorized; reworked Pi rev-18; tightened Pi
rev-19/20).

Exact STRUCTURE + STRICT fail-closed VALIDATORS for the two reserved manifests the registered evaluation will need,
derived from the FULL schema-validated SD registry, carrying BOTH exemption variants, with role-specific RNG
provenance, CANONICAL-VALUE validation of every design-bearing field, and a union/variant map-identity model.
SCHEMAS ONLY — NOTHING is drawn/populated/frozen: seeds/maps/paths/content hashes are `RESERVED_NOT_DRAWN`, and the
composed manifest identities stay the engine's `RESERVED_*` sentinels, so `gate_group_registered` stays BLOCKED.

Pi rev-19/20 tightening folded:
  #1 validators compare every design-bearing scalar/string/hash to a CANONICAL EXPECTED value (not just type-check),
     with adversarial tests per field;
  #2 `profile_config_identity` binds the ACTUAL profile configuration (exact `PROFILES[...]` payload + source
     skeleton + constructor route + stratum allocation), recomputed and compared;
  #3 a SEED-INDEPENDENT RNG-law identity (derivation formulas + role symmetry + constructor route + coupling rule),
     distinct from the seed-bearing `rng_identity`;
  #4 a union/variant map-identity model — one union of the 17 unique maps (drawn once) + separately-hashed 16/17
     variant projections, with STRUCTURED set-hash payloads (all `map_identity` still RESERVED);
  #5 separate RNG / map-set schema identities + a combined bundle identity, with a REAL dependency-exclusion proof;
  #6 nonempty + non-bool value checks (reject bool-as-int).

Development-only. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_manifest.py
"""
from __future__ import annotations

import copy
import hashlib
import json

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from scripts.oracle_realism_v3_randomization import RefusalError
import scripts.oracle_realism_v3_engine as ENG
import scripts.oracle_realism_v3_registry as REG

RESERVED = "RESERVED_NOT_DRAWN"
NONE_COUPLING = "NONE"
DRAFT = "DRAFT"
_ARM_ORDER = ["candidate", "reference"]
_ROLES = ("candidate", "reference")
_COUPLING_STRENGTH = 0.5
_STRATIFIED_CONTROL_ALLOC = [2667, 2667, 2666]   # the STRATIFIED (structural-zero / boundary) control allocation only
_CONTENT_HASH_ALGO = "sha256(canonical_hash) over derive_record-canonical (source,class_ids,timestamps) per role"
_SET_HASH_RULE = ("variant_set_identity = canonical_hash({variant, registry_identity, sorted "
                  "[(profile,regime,check,map_identity)], builder_identity, apply_code_identity, "
                  "schema_definition_identity}); union_set_identity over the 17 unique drawn maps")
_OUTPUT_PATH_GRAMMAR = "relative POSIX under the reserved map-design root; no '..'/absolute/symlink (traversal refused)"
_REFUSED_ARTIFACT_RULE = ("an issued REFUSED_reference_coarsening map neither satisfies an entry nor grants a "
                          "provisional exemption; it is parked for exemption review; no set identity is issued")
RNG_SCHEMA_VERSION = "v3-rng-manifest-draft-3"
MAPSET_SCHEMA_VERSION = "v3-map-set-manifest-draft-3"

_MAP_CARRYING = frozenset(k for k, v in ENG.ESTIMATORS.items() if v["map_carrying"])
_LAYER_KEYS = ("estimator_semantic", "estimator_impl_source", "engine_canon_schema_gate", "map_source", "dependency")
_SOURCE_LAYER_KEYS = ("estimator_semantic", "estimator_impl_source", "engine_canon_schema_gate", "map_source")


def _identity_layers():
    return {**ENG.SOURCE_IDENTITY_BUNDLE, "dependency": ENG.ESTIMATOR_DEPENDENCY_IDENTITY}


def _manifest_source_identity():
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _constructor_route(source):
    if source == "structural_zero_control":
        return "structural_multiscale"
    if source == "boundary_short":
        return "bounded_length_control"
    return "source_profile_fixture"


def _stratum_alloc_for(source):
    """The source's ACTUAL registered stratum allocation, DERIVED from the registry (rev-22).

    The rev-21 implementation bound the hardcoded `[2667,2667,2666]` allocation into every profile's
    configuration identity, but only the two STRATIFIED sources (structural-zero, boundary-short) carry that
    allocation: the source-profile experiments have a single pooled stratum of 8000. Binding the same wrong
    allocation for those was the same defect class Pi rev-19/20 #2 targeted — a design-bearing field that does
    not reflect the actual configuration. Refuses if a source's experiments disagree on their strata."""
    seen = {}
    for c in REG.build_sd_cells(apply_uncalibratable_exemption=False):
        if c["scope"] != "in" or c["source"] != source:
            continue
        key = tuple((s["stratum_id"], s["n_candidate"], s["n_reference"]) for s in c["exchangeability_strata"])
        seen.setdefault(key, set()).add(c["experiment_id"])
    if not seen:
        raise RefusalError(f"no in-scope registry cells for source {source!r}")
    if len(seen) > 1:
        raise RefusalError(f"source {source!r} has inconsistent registered strata across experiments: "
                           f"{ {k: sorted(v) for k, v in seen.items()} }")
    strata = next(iter(seen))
    return {"stratum_ids": [s[0] for s in strata],
            "n_candidate": [int(s[1]) for s in strata], "n_reference": [int(s[2]) for s in strata]}


def _boundary_constructor_identities():
    """The REAL bounded-length control constructor identities (rev-22), one per allocation variant.

    Pi rev-19/20 #2 requires the boundary constructor/allocation identity to be bound, not merely the route
    LABEL. Now that `oracle_realism_v3_constructors` implements the `bounded_length_control` route, its
    deterministic route identity is bound here. BOTH allocation variants are bound because the choice between
    them is an open reviewer decision (band widths 2/2/3 cannot preserve both the registered allocation and the
    uniform pooled length marginal), exactly as both exemption variants are carried elsewhere."""
    import scripts.oracle_realism_v3_constructors as CON      # local import: CON imports this module in its tests
    return {v: CON.constructor_route_identity(v) for v in CON.ALLOC_VARIANTS}


def _profile_config_identity(source, regime):
    """The ACTUAL profile configuration identity (Pi rev-19/20 #2): the exact PROFILES payload + source skeleton +
    constructor route + REGISTRY-DERIVED stratum allocation (+ the executable constructor identities for the
    bounded route) — not a bare {profile,regime} label and not a hardcoded allocation."""
    payload = {
        "profile": source, "regime": regime,
        "profile_payload": PROFILES[source],
        "skeleton": ENG._PROFILE_SKELETON[source],
        "constructor_route": _constructor_route(source),
        "stratum_allocation": _stratum_alloc_for(source),
    }
    if source == "boundary_short":
        payload["constructor_route_identities"] = _boundary_constructor_identities()
        payload["constructor_allocation_variant"] = RESERVED       # the variant choice is not yet decided
    return canonical_hash(payload)


def _rng_law_identity(source, coupled):
    """A SEED-INDEPENDENT RNG-law semantic identity (Pi rev-19/20 #3): derivation formulas + role symmetry +
    constructor route + coupling rule. Contains NO seed value; distinct from the seed-bearing `rng_identity`."""
    return canonical_hash({
        "fixture_law": "per (experiment,stratum,role) role-symmetric derived seed -> numpy default_rng",
        "coupling_law": (f"apply_coupling(component={coupled}, strength={_COUPLING_STRENGTH}) per role"
                         if coupled is not None else NONE_COUPLING),
        "constructor_route": _constructor_route(source),
        "role_symmetric": True, "canonical_arm_order": list(_ARM_ORDER),
    })


# --- strict schema helpers -------------------------------------------------------------------------------------
def _strict(obj, spec, where):
    if not isinstance(obj, dict):
        raise RefusalError(f"{where} must be a dict, got {type(obj).__name__}")
    keys, req = set(obj), set(spec)
    if keys - req:
        raise RefusalError(f"{where} has unknown field(s) {sorted(keys - req)}")
    if req - keys:
        raise RefusalError(f"{where} missing field(s) {sorted(req - keys)}")
    for f, t in spec.items():
        v = obj[f]
        if t is int and isinstance(v, bool):                       # reject bool-as-int (Pi rev-19/20 #6)
            raise RefusalError(f"{where}.{f} is a bool, not an int")
        if not isinstance(v, t):
            raise RefusalError(f"{where}.{f} type {type(v).__name__} != {getattr(t, '__name__', t)}")
        if isinstance(v, str) and not v.strip():                   # nonempty strings (Pi rev-19/20 #6)
            raise RefusalError(f"{where}.{f} must be a nonempty string")


def _eq(obj, field, expected, where):
    """Compare a design-bearing value to its CANONICAL expected value (Pi rev-19/20 #1)."""
    if obj.get(field) != expected:
        raise RefusalError(f"{where}.{field} != canonical expected value")


def _check_identity_layers(bundle, where):
    _strict(bundle, {k: str for k in _LAYER_KEYS}, f"{where}.identity_layers")
    if bundle != _identity_layers():
        raise RefusalError(f"{where}.identity_layers do not match the current engine identities")


def _full_registry_experiments():
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
    sd = REG.build_sd_cells(apply_uncalibratable_exemption=with_exemption)
    by_id = {c["cell_id"]: c for c in sd}
    triples = set()
    for g in REG.build_groups(sd).values():
        for cid in g["cells"]:
            c = by_id[cid]
            if c["statistic"] in _MAP_CARRYING:
                triples.add((c["source"], c["support_regime"], c["statistic"]))
    return sorted(triples)


def _union_triples():
    """The union of unique map triples across BOTH variants (drawn ONCE; the without-exemption superset). Pi #4."""
    return sorted(set(_map_triples_for_variant(True)) | set(_map_triples_for_variant(False)))


# --- builders (DRAFT: drawn values RESERVED; composed identities are the engine's reserved sentinels) -----------
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
                    "content_hash": RESERVED,
                    "count_hash": canonical_hash({"experiment": e, "stratum": sid, "role": role, "count": int(count)}),
                    "expected_count": int(count),
                }
            strata.append({"stratum_id": sid, "n_candidate": qc, "n_reference": qr,
                           "coupling_strength": (_COUPLING_STRENGTH if coupled is not None else NONE_COUPLING),
                           "roles": roles})
        exps[e] = {
            "source_profile": meta["source"], "condition": meta["condition"], "support_regime": meta["support_regime"],
            "coupled_component": (coupled if coupled is not None else NONE_COUPLING),
            "constructor_route": _constructor_route(meta["source"]),
            "profile_config_identity": _profile_config_identity(meta["source"], meta["support_regime"]),
            "rng_law_identity": _rng_law_identity(meta["source"], coupled),
            "reserved_replicate_identity": RESERVED,               # the reserved final evaluation/replicate identity
            "canonical_arm_order": list(_ARM_ORDER), "strata": strata,
        }
    return {
        "schema_version": RNG_SCHEMA_VERSION, "status": DRAFT,
        "identity_layers": _identity_layers(),
        "registry_variant_identities": dict(ENG.REGISTERED["registry_identity"]),
        "manifest_source_identity": _manifest_source_identity(),
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "content_hash_algorithm_identity": _CONTENT_HASH_ALGO,
        "generator_code_identity": RESERVED, "coupling_code_identity": RESERVED,
        "experiments": exps,
        "rng_manifest_identity": ENG.REGISTERED["rng_manifest_identity"],
    }


def _union_entry(profile, regime, check):
    return {"profile": profile, "regime": regime, "check": check,
            "N": int(ENG.REGISTERED["N_per_arm"]), "floor": int(ENG.REGISTERED["floor"]),
            "builder_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
            "apply_code_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
            "expected_status": "OK",
            "seed": RESERVED, "namespace": RESERVED, "output_path": RESERVED, "map_identity": RESERVED}


def _set_hash_payload(variant_name, reg_id, triples):
    """STRUCTURED set-hash payload (Pi rev-19/20 #4) — not prose. map_identity stays RESERVED until draw."""
    return {"variant": variant_name, "registry_identity": reg_id,
            "entries": [{"profile": p, "regime": r, "check": c, "map_identity": RESERVED} for (p, r, c) in triples],
            "builder_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
            "apply_code_identity": ENG.SOURCE_IDENTITY_BUNDLE["map_source"],
            "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY}


def build_draft_map_set_manifest():
    variants = {}
    for name, with_ex, reg_id in (("with_exemption", True, ENG.REGISTERED["registry_identity"]["with"]),
                                  ("without_exemption", False, ENG.REGISTERED["registry_identity"]["without"])):
        triples = _map_triples_for_variant(with_ex)
        variants[name] = {"registry_identity": reg_id, "triples": [list(t) for t in triples],
                          "set_hash_payload": _set_hash_payload(name, reg_id, triples),
                          "variant_set_identity": RESERVED}
    return {
        "schema_version": MAPSET_SCHEMA_VERSION, "status": DRAFT,
        "identity_layers": _identity_layers(),
        "manifest_source_identity": _manifest_source_identity(),
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "sort_key": ["profile", "regime", "check"],
        "set_hash_rule": _SET_HASH_RULE,
        "output_path_grammar": _OUTPUT_PATH_GRAMMAR,
        "refused_artifact_rule": _REFUSED_ARTIFACT_RULE,
        "union_entries": [_union_entry(*t) for t in _union_triples()],   # 17 unique maps drawn ONCE
        "union_set_identity": RESERVED,
        "variants": variants,
        "map_set_identity": ENG.REGISTERED["map_set_identity"],
    }


# --- strict + value validators ---------------------------------------------------------------------------------
_RNG_TOP = {"schema_version": str, "status": str, "identity_layers": dict, "registry_variant_identities": dict,
            "manifest_source_identity": str, "schema_definition_identity": str, "content_hash_algorithm_identity": str,
            "generator_code_identity": str, "coupling_code_identity": str, "experiments": dict,
            "rng_manifest_identity": str}
_RNG_EXP = {"source_profile": str, "condition": str, "support_regime": str, "coupled_component": str,
            "constructor_route": str, "profile_config_identity": str, "rng_law_identity": str,
            "reserved_replicate_identity": str, "canonical_arm_order": list, "strata": list}
_RNG_STRATUM = {"stratum_id": str, "n_candidate": int, "n_reference": int, "coupling_strength": (float, str), "roles": dict}
_RNG_ROLE = {"fixture_seed": str, "coupling_seed": str, "content_hash": str, "count_hash": str, "expected_count": int}


def validate_rng_manifest(m):
    _strict(m, _RNG_TOP, "rng manifest")
    _eq(m, "schema_version", RNG_SCHEMA_VERSION, "rng manifest")
    _eq(m, "status", DRAFT, "rng manifest")
    _eq(m, "content_hash_algorithm_identity", _CONTENT_HASH_ALGO, "rng manifest")
    _check_identity_layers(m["identity_layers"], "rng manifest")
    _eq(m, "registry_variant_identities", dict(ENG.REGISTERED["registry_identity"]), "rng manifest")
    _eq(m, "manifest_source_identity", _manifest_source_identity(), "rng manifest")
    _eq(m, "schema_definition_identity", SCHEMA_DEFINITION_IDENTITY, "rng manifest")
    _eq(m, "generator_code_identity", RESERVED, "rng manifest")
    _eq(m, "coupling_code_identity", RESERVED, "rng manifest")
    _eq(m, "rng_manifest_identity", ENG.REGISTERED["rng_manifest_identity"], "rng manifest")
    universe = _full_registry_experiments()
    if set(m["experiments"]) != set(universe):
        raise RefusalError("rng manifest experiments != full registry")
    for e, meta in universe.items():
        em = m["experiments"][e]; _strict(em, _RNG_EXP, f"rng experiment {e}")
        coupled = meta["coupled_component"]
        _eq(em, "source_profile", meta["source"], f"rng experiment {e}")
        _eq(em, "condition", meta["condition"], f"rng experiment {e}")
        _eq(em, "support_regime", meta["support_regime"], f"rng experiment {e}")
        _eq(em, "coupled_component", (coupled if coupled is not None else NONE_COUPLING), f"rng experiment {e}")
        _eq(em, "constructor_route", _constructor_route(meta["source"]), f"rng experiment {e}")
        _eq(em, "profile_config_identity", _profile_config_identity(meta["source"], meta["support_regime"]), f"rng experiment {e}")
        _eq(em, "rng_law_identity", _rng_law_identity(meta["source"], coupled), f"rng experiment {e}")
        _eq(em, "reserved_replicate_identity", RESERVED, f"rng experiment {e}")
        _eq(em, "canonical_arm_order", _ARM_ORDER, f"rng experiment {e}")
        if [s.get("stratum_id") for s in em["strata"]] != [sid for sid, _, _ in meta["strata"]]:
            raise RefusalError(f"rng experiment {e} stratum ids/order mismatch")
        for sm, (sid, qc, qr) in zip(em["strata"], meta["strata"]):
            _strict(sm, _RNG_STRATUM, f"rng {e}/{sid} stratum")
            if (sm["n_candidate"], sm["n_reference"]) != (qc, qr):
                raise RefusalError(f"rng {e}/{sid} quota mismatch")
            _eq(sm, "coupling_strength", (_COUPLING_STRENGTH if coupled is not None else NONE_COUPLING), f"rng {e}/{sid}")
            if set(sm["roles"]) != set(_ROLES):
                raise RefusalError(f"rng {e}/{sid} roles must be exactly {sorted(_ROLES)}")
            for role, count in (("candidate", qc), ("reference", qr)):
                rm = sm["roles"][role]; _strict(rm, _RNG_ROLE, f"rng {e}/{sid}/{role}")
                _eq(rm, "expected_count", count, f"rng {e}/{sid}/{role}")
                _eq(rm, "count_hash", canonical_hash({"experiment": e, "stratum": sid, "role": role, "count": count}),
                    f"rng {e}/{sid}/{role}")
                _eq(rm, "fixture_seed", RESERVED, f"rng {e}/{sid}/{role}")
                _eq(rm, "content_hash", RESERVED, f"rng {e}/{sid}/{role}")
                _eq(rm, "coupling_seed", RESERVED if coupled is not None else NONE_COUPLING, f"rng {e}/{sid}/{role}")
    return m


_MAPSET_TOP = {"schema_version": str, "status": str, "identity_layers": dict, "manifest_source_identity": str,
               "schema_definition_identity": str, "sort_key": list, "set_hash_rule": str, "output_path_grammar": str,
               "refused_artifact_rule": str, "union_entries": list, "union_set_identity": str, "variants": dict,
               "map_set_identity": str}
_MAPSET_VARIANT = {"registry_identity": str, "triples": list, "set_hash_payload": dict, "variant_set_identity": str}
_MAPSET_ENTRY = {"profile": str, "regime": str, "check": str, "N": int, "floor": int, "builder_identity": str,
                 "apply_code_identity": str, "expected_status": str, "seed": str, "namespace": str,
                 "output_path": str, "map_identity": str}


def validate_map_set_manifest(m):
    _strict(m, _MAPSET_TOP, "map-set manifest")
    _eq(m, "schema_version", MAPSET_SCHEMA_VERSION, "map-set manifest")
    _eq(m, "status", DRAFT, "map-set manifest")
    _check_identity_layers(m["identity_layers"], "map-set manifest")
    _eq(m, "manifest_source_identity", _manifest_source_identity(), "map-set manifest")
    _eq(m, "schema_definition_identity", SCHEMA_DEFINITION_IDENTITY, "map-set manifest")
    _eq(m, "sort_key", ["profile", "regime", "check"], "map-set manifest")
    _eq(m, "set_hash_rule", _SET_HASH_RULE, "map-set manifest")
    _eq(m, "output_path_grammar", _OUTPUT_PATH_GRAMMAR, "map-set manifest")
    _eq(m, "refused_artifact_rule", _REFUSED_ARTIFACT_RULE, "map-set manifest")
    _eq(m, "union_set_identity", RESERVED, "map-set manifest")
    _eq(m, "map_set_identity", ENG.REGISTERED["map_set_identity"], "map-set manifest")
    # union entries == the 17 unique triples, each drawn once
    union = _union_triples()
    if [(e.get("profile"), e.get("regime"), e.get("check")) for e in m["union_entries"]] != union:
        raise RefusalError("map-set union_entries != the unique union triples (missing/extra/duplicate/order)")
    for e in m["union_entries"]:
        _strict(e, _MAPSET_ENTRY, f"map-set union entry {e.get('check')}@{e.get('profile')}")
        if (e["N"], e["floor"]) != (int(ENG.REGISTERED["N_per_arm"]), int(ENG.REGISTERED["floor"])):
            raise RefusalError("map-set union entry N/floor != registered")
        _eq(e, "builder_identity", ENG.SOURCE_IDENTITY_BUNDLE["map_source"], "map-set union entry")
        _eq(e, "apply_code_identity", ENG.SOURCE_IDENTITY_BUNDLE["map_source"], "map-set union entry")
        _eq(e, "expected_status", "OK", "map-set union entry")
        for k in ("seed", "namespace", "output_path", "map_identity"):
            _eq(e, k, RESERVED, "map-set union entry")
    if set(m["variants"]) != {"with_exemption", "without_exemption"}:
        raise RefusalError("map-set variants must be exactly with/without exemption")
    for name, with_ex, reg_id in (("with_exemption", True, ENG.REGISTERED["registry_identity"]["with"]),
                                  ("without_exemption", False, ENG.REGISTERED["registry_identity"]["without"])):
        var = m["variants"][name]; _strict(var, _MAPSET_VARIANT, f"map-set variant {name}")
        _eq(var, "registry_identity", reg_id, f"map-set variant {name}")
        _eq(var, "variant_set_identity", RESERVED, f"map-set variant {name}")
        triples = _map_triples_for_variant(with_ex)
        if [tuple(t) for t in var["triples"]] != triples:
            raise RefusalError(f"map-set variant {name} triples != required (missing/extra/order)")
        if var["set_hash_payload"] != _set_hash_payload(name, reg_id, triples):
            raise RefusalError(f"map-set variant {name} set_hash_payload != canonical structured payload")
    return m


# --- identities (Pi rev-19/20 #5): SEPARATE per-manifest schema identities + a combined bundle; DETERMINISTIC ----
def _deterministic_payload():
    """The exact env-INDEPENDENT payload hashed into the schema identities. Provably free of the dependency layer:
    only the four SOURCE layers + registry variants + field definitions + manifest source."""
    return {
        "schema_definition": SCHEMA_DEFINITION_IDENTITY,
        "manifest_source": _manifest_source_identity(),
        "registry_variant_identities": dict(ENG.REGISTERED["registry_identity"]),
        "source_identity_layers": {k: ENG.SOURCE_IDENTITY_BUNDLE[k] for k in _SOURCE_LAYER_KEYS},
    }


def _rng_schema_identity():
    return canonical_hash({"kind": "rng", "version": RNG_SCHEMA_VERSION, **_deterministic_payload()})


def _mapset_schema_identity():
    return canonical_hash({"kind": "map_set", "version": MAPSET_SCHEMA_VERSION, **_deterministic_payload()})


def _combined_schema_identity():
    return canonical_hash({"rng": _rng_schema_identity(), "map_set": _mapset_schema_identity(),
                           **_deterministic_payload()})


SCHEMA_DEFINITION_IDENTITY = canonical_hash({
    "rng_version": RNG_SCHEMA_VERSION, "mapset_version": MAPSET_SCHEMA_VERSION,
    "rng_fields": {"top": {k: getattr(t, "__name__", str(t)) for k, t in _RNG_TOP.items()},
                   "experiment": {k: getattr(t, "__name__", str(t)) for k, t in _RNG_EXP.items()},
                   "stratum": {k: getattr(t, "__name__", str(t)) for k, t in _RNG_STRATUM.items()},
                   "role": {k: t.__name__ for k, t in _RNG_ROLE.items()}},
    "mapset_fields": {"top": {k: getattr(t, "__name__", str(t)) for k, t in _MAPSET_TOP.items()},
                      "variant": {k: getattr(t, "__name__", str(t)) for k, t in _MAPSET_VARIANT.items()},
                      "entry": {k: t.__name__ for k, t in _MAPSET_ENTRY.items()}},
    "layer_keys": list(_LAYER_KEYS), "arm_order": _ARM_ORDER, "coupling_strength": _COUPLING_STRENGTH,
    "stratified_control_allocation": _STRATIFIED_CONTROL_ALLOC,
    "content_hash_algorithm": _CONTENT_HASH_ALGO,
    "set_hash_rule": _SET_HASH_RULE, "output_path_grammar": _OUTPUT_PATH_GRAMMAR,
    "refused_artifact_rule": _REFUSED_ARTIFACT_RULE})


# --- self-tests ------------------------------------------------------------------------------------------------
def selftest():
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

    e0 = next(iter(rng["experiments"]))
    rng_bad = {
        "unknown_top_field": lambda m: m.__setitem__("sneaky", 1),
        "unknown_experiment_field": lambda m: m["experiments"][e0].__setitem__("sneaky", 1),
        "unknown_role_field": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("sneaky", 1),
        "drop_experiment": lambda m: m["experiments"].pop(e0),
        "empty_string_field": lambda m: m.__setitem__("content_hash_algorithm_identity", "  "),
        "bool_as_int_count": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("expected_count", True),
        # value tampers (Pi rev-19/20 #1) — typed-but-tampered must refuse:
        "tamper_profile_config": lambda m: m["experiments"][e0].__setitem__("profile_config_identity", "X"),
        "tamper_content_hash_algo": lambda m: m.__setitem__("content_hash_algorithm_identity", "other"),
        "tamper_constructor_route": lambda m: m["experiments"][e0].__setitem__("constructor_route", "other"),
        "tamper_rng_law": lambda m: m["experiments"][e0].__setitem__("rng_law_identity", "X"),
        "tamper_schema_version": lambda m: m.__setitem__("schema_version", "vX"),
        "wrong_count_hash": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("count_hash", "X"),
        "populated_fixture_seed": lambda m: m["experiments"][e0]["strata"][0]["roles"]["candidate"].__setitem__("fixture_seed", "7"),
        "populated_replicate": lambda m: m["experiments"][e0].__setitem__("reserved_replicate_identity", "bound"),
        "bound_manifest_identity": lambda m: m.__setitem__("rng_manifest_identity", "BOUND"),
    }
    for name, mut in rng_bad.items():
        if not refused(rng, validate_rng_manifest, mut):
            errs.append(f"rng manifest did NOT refuse {name}")

    mapset_bad = {
        "unknown_variant_field": lambda m: m["variants"]["with_exemption"].__setitem__("sneaky", 1),
        "unknown_union_entry_field": lambda m: m["union_entries"][0].__setitem__("sneaky", 1),
        "drop_variant": lambda m: m["variants"].pop("without_exemption"),
        "drop_union_entry": lambda m: m["union_entries"].pop(),
        "duplicate_union_entry": lambda m: m["union_entries"].append(copy.deepcopy(m["union_entries"][0])),
        "swap_variants_favorably": lambda m: m["variants"].__setitem__("without_exemption",
                                                                       copy.deepcopy(m["variants"]["with_exemption"])),
        "tamper_set_hash_rule": lambda m: m.__setitem__("set_hash_rule", "other"),
        "tamper_output_path_grammar": lambda m: m.__setitem__("output_path_grammar", "anything goes"),
        "tamper_refused_rule": lambda m: m.__setitem__("refused_artifact_rule", "auto-exempt"),
        "tamper_variant_payload": lambda m: m["variants"]["with_exemption"]["set_hash_payload"].__setitem__("variant", "X"),
        "wrong_N": lambda m: m["union_entries"][0].__setitem__("N", 4000),
        "populated_map_identity": lambda m: m["union_entries"][0].__setitem__("map_identity", "abcd"),
        "populated_union_set_identity": lambda m: m.__setitem__("union_set_identity", "bound"),
        "bound_set_identity": lambda m: m.__setitem__("map_set_identity", "BOUND"),
    }
    for name, mut in mapset_bad.items():
        if not refused(mapset, validate_map_set_manifest, mut):
            errs.append(f"map-set manifest did NOT refuse {name}")

    # Pi rev-19/20 #5: REAL dependency-exclusion proof — the deterministic payload has no dependency layer, and a
    # synthetic change to the dependency identity leaves the schema identities UNCHANGED.
    payload = _deterministic_payload()
    if "dependency" in payload.get("source_identity_layers", {}):
        errs.append("deterministic payload leaks the dependency layer")
    before = (_rng_schema_identity(), _mapset_schema_identity(), _combined_schema_identity())
    saved = ENG.ESTIMATOR_DEPENDENCY_IDENTITY
    try:
        ENG.ESTIMATOR_DEPENDENCY_IDENTITY = "SYNTHETIC_DIFFERENT_DEPENDENCY"
        after = (_rng_schema_identity(), _mapset_schema_identity(), _combined_schema_identity())
    finally:
        ENG.ESTIMATOR_DEPENDENCY_IDENTITY = saved
    if before != after:
        errs.append("deterministic schema identities change when the dependency identity changes (not env-independent)")
    if _rng_schema_identity() == _mapset_schema_identity():
        errs.append("rng and map-set schema identities are not distinct")

    # rev-22: the stratum allocation bound into profile_config_identity must be DERIVED from the registry, not
    # a hardcoded constant applied to every profile.
    reg_strata = {}
    for c in REG.build_sd_cells(apply_uncalibratable_exemption=False):
        if c["scope"] == "in":
            reg_strata.setdefault(c["source"], [(s["stratum_id"], s["n_candidate"], s["n_reference"])
                                                for s in c["exchangeability_strata"]])
    for src, strata in reg_strata.items():
        got = _stratum_alloc_for(src)
        want = {"stratum_ids": [s[0] for s in strata], "n_candidate": [int(s[1]) for s in strata],
                "n_reference": [int(s[2]) for s in strata]}
        if got != want:
            errs.append(f"derived stratum allocation for {src} {got} != registry {want}")
    # the two POOLED sources must bind a single 8000 stratum, NOT the stratified control allocation
    for src in ("scid_scale_control", "mimic_scale_control"):
        a = _stratum_alloc_for(src)
        if a["stratum_ids"] != ["pooled_source"] or a["n_candidate"] != [8000]:
            errs.append(f"{src} should bind one pooled 8000 stratum, got {a}")
        if a["n_candidate"] == _STRATIFIED_CONTROL_ALLOC:
            errs.append(f"{src} still binds the stratified control allocation (the rev-21 defect)")
    # Pi rev-22 ruling 1: the two stratified sources no longer SHARE an allocation — structural-zero keeps the
    # equal-ish control allocation, boundary-short uses the selected width-proportional one.
    if _stratum_alloc_for("structural_zero_control")["n_candidate"] != _STRATIFIED_CONTROL_ALLOC:
        errs.append("structural_zero_control should bind the equal-ish stratified control allocation")
    if _stratum_alloc_for("boundary_short")["n_candidate"] != list(REG.BOUNDARY_ALLOC):
        errs.append(f"boundary_short should bind the selected width-proportional allocation "
                    f"{list(REG.BOUNDARY_ALLOC)}, got {_stratum_alloc_for('boundary_short')['n_candidate']}")
    if _stratum_alloc_for("boundary_short")["n_candidate"] == _STRATIFIED_CONTROL_ALLOC:
        errs.append("boundary_short still binds the equal-ish allocation the ruling replaced")
    # the fix must BITE: a pooled source and a stratified source must not share a configuration identity, and
    # recomputing with the old hardcoded allocation must give a DIFFERENT identity for the pooled sources.
    if _profile_config_identity("scid_scale_control", "full") == _profile_config_identity("structural_zero_control",
                                                                                          "full"):
        errs.append("pooled and stratified sources share a profile_config_identity")
    old_style = canonical_hash({"profile": "scid_scale_control", "regime": "full",
                                "profile_payload": PROFILES["scid_scale_control"],
                                "skeleton": ENG._PROFILE_SKELETON["scid_scale_control"],
                                "constructor_route": _constructor_route("scid_scale_control"),
                                "stratum_allocation": list(_STRATIFIED_CONTROL_ALLOC)})
    if old_style == _profile_config_identity("scid_scale_control", "full"):
        errs.append("profile_config_identity unchanged by the allocation fix (the defect is not actually fixed)")

    # rev-22: the boundary route must bind the REAL executable constructor identities, both variants, and the
    # variant choice must remain RESERVED (undecided).
    try:
        import scripts.oracle_realism_v3_constructors as CON
        bound = _boundary_constructor_identities()
        if set(bound) != set(CON.ALLOC_VARIANTS):
            errs.append(f"boundary constructor identities {sorted(bound)} != variants {sorted(CON.ALLOC_VARIANTS)}")
        for v in CON.ALLOC_VARIANTS:
            if bound[v] != CON.constructor_route_identity(v):
                errs.append(f"bound boundary constructor identity for {v} does not recompute")
        if len(set(bound.values())) != len(bound):
            errs.append("the two boundary allocation variants collapse to one constructor identity")
        if CON.CANONICAL_ROUTE["boundary_short"] != _constructor_route("boundary_short"):
            errs.append("constructor module and manifest disagree on the boundary route id")
    except ImportError as ex:                                   # pragma: no cover
        errs.append(f"cannot import the constructor module to bind the boundary route: {ex}")

    return errs, rng, mapset


def main():
    errs, rng, mapset = selftest()
    out = {
        "purpose": "DRAFT-ONLY reserved-manifest schemas with CANONICAL-VALUE strict validators, full-registry "
                   "universes, both exemption variants, role-specific provenance, union/variant map-identity model. "
                   "Nothing drawn/frozen/populated.",
        "rng_manifest": {"experiments": len(rng["experiments"]),
                         "strata_total": sum(len(v["strata"]) for v in rng["experiments"].values()),
                         "identity": rng["rng_manifest_identity"]},
        "map_set_manifest": {"union_entries": len(mapset["union_entries"]),
                             "variants": {k: len(v["triples"]) for k, v in mapset["variants"].items()},
                             "identity": mapset["map_set_identity"]},
        "rng_schema_identity": _rng_schema_identity(),
        "mapset_schema_identity": _mapset_schema_identity(),
        "combined_manifest_schema_identity": _combined_schema_identity(),
        "schema_definition_identity": SCHEMA_DEFINITION_IDENTITY,
        "selftests_pass": not errs, "selftest_errors": errs,
        "authorization": "dev-only DRAFT schemas; reserved seeds/maps/paths/content NOT drawn; NOT frozen/populated; "
                         "gate_group_registered stays BLOCKED. Freezing needs a separate review.",
    }
    print(json.dumps(out, indent=2, default=str))
    print("\nRNG_SCHEMA_IDENTITY (deterministic):", _rng_schema_identity())
    print("MAPSET_SCHEMA_IDENTITY (deterministic):", _mapset_schema_identity())
    print("COMBINED_MANIFEST_SCHEMA_IDENTITY (deterministic):", _combined_schema_identity())
    assert not errs, f"manifest schema self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
