"""Control / ablation battery for the realism-v2 verifier (rebuild step 3; Pi F3 + run-contract fold).

FAIL-CLOSED: a required check is satisfied ONLY on PASS; NOT_EVALUABLE is non-passing and reported separately.
For each active D component the REFERENCE carries the component at 0.5; a null-independent `candidate_A` must
FAIL every primary subcheck and PASS every required non-attributed check; an independent-sample
`candidate_D` (same frozen constructor, distinct RNG) establishes KNOWN-PROFILE REPEATABILITY (not
implementation recovery — that is the M2 adapter comparison later). Controls: null/self, boundary-short,
structural-zero, source-swap (negative, never D). A rate-based multi-seed harness reports PER-CHECK
PASS/FAIL/NOT_EVALUABLE rates with Wilson CIs and a source-wise CONJUNCTIVE verdict against the registered
thresholds — the step-4 power object.

Source blocks are GENUINELY DISTINCT canonical controls (design `PROFILES`); fixture and coupling RNG are
derived deterministically from (source, profile, replicate_seed, role) / (source, component, replicate_seed,
role). The registered per-source/profile/seed sample size is 4000 (design); tests exercise the machinery at a
small mechanical size. Synthetic-only; no candidate adapter, no governed read.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, FAIL, PASS, NOT_EVALUABLE,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX, PROFILES

_ABL_S = 0.5
REGISTERED_N = 4000                                   # design per-source/profile/seed sample size
SOURCE_PROFILES = ("scid_scale_control", "mimic_scale_control")
PRIMARY_FAIL_MIN = 20                                 # misspecified control FAILs its primary >=20/25
SPECIFICITY_MIN = 24                                  # non-attributed / repeatability / null PASS >=24/25
_NONDEGENERATE = ["count_ks", "positive_gap_ks", "class_tv", "S1_density", "S1_tau"]
# boundary-short PREDECLARED refusals: length-conditioned checks (a single length bin) AND the block-seam
# checks (sequences shorter than an 8-item block have no seams). Everything else must PASS.
_BOUNDARY_EXPECTED_NE = {"S1_density", "S5_abs", "S6_tv", "S9_zero", "S9_class", "S9_gap"}

# Authoritative universe of check keys the two verifier routes emit (keys ALWAYS present; status may be NE).
# A record missing any of these is truncated/tampered — used to require the FULL expected set (Pi §4). A
# conformance test (test_oracle_v2_battery) asserts the emitted keys equal this registry, catching drift.
ALL_CHECK_KEYS = frozenset({
    "S1_density", "S1_tau", "S2_ks", "S3_tau", "S3_loggap", "S4_abs", "S5_abs", "S6_tv", "S7_abs",
    "S8_class", "S8_density", "S9_zero", "S9_class", "S9_gap",
    "class_tv", "count_ks", "delta_t_zero_abs", "length_ks", "occupancy_abs", "positive_gap_ks",
})


def _derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def _source_key(profile_name: str) -> str:
    return "SCID" if "scid" in profile_name else "MIMIC"


def _evidence_map(cand, ref) -> dict:
    """Full reviewable evidence per check: status + scalar value + the CheckResult.detail (denominator and
    coarsening maps). This is what a reviewer audits — WHY a check passed/failed, not just the label."""
    out = {}
    for k, v in {**sequence_route_checks(cand, ref), **marginal_route_checks(cand, ref)}.items():
        out[k] = {"status": v.status, "value": v.value, "threshold": v.threshold, "detail": v.detail}
    return out


def _status_map(cand, ref) -> dict:
    return {k: e["status"] for k, e in _evidence_map(cand, ref).items()}


@dataclasses.dataclass(frozen=True)
class AblationOutcome:
    component: str
    replicate_seed: int
    source_profile: str
    A_status: dict
    D_status: dict
    A_fails_primary: bool                  # every primary subcheck == FAIL
    A_specificity: dict                    # per non-attributed check: True iff PASS (fail-closed)
    A_specificity_ok: bool
    known_profile_repeatability: bool      # candidate_D: every required check == PASS (NOT recovery)
    A_evidence: dict = dataclasses.field(default_factory=dict)   # full CheckResult evidence (value + detail)
    D_evidence: dict = dataclasses.field(default_factory=dict)


def component_ablation(component, replicate_seed, *, base_sampler, source_profile) -> AblationOutcome:
    if component not in V2_D_COMPONENT_MENU:
        raise KeyError(component)
    primary = ABLATION_MATRIX[component]["primary_fail"]
    allowed = set(ABLATION_MATRIX[component]["allowed_sensitive"])
    src = _source_key(source_profile)

    def cseed(role):
        return _derive_seed("coupling", src, component, replicate_seed, role)
    reference = apply_coupling(base_sampler(source_profile, replicate_seed, "reference"),
                               component, _ABL_S, seed=cseed("reference"))
    candidate_A = base_sampler(source_profile, replicate_seed, "candidate_A")
    candidate_D = apply_coupling(base_sampler(source_profile, replicate_seed, "candidate_D"),
                                 component, _ABL_S, seed=cseed("candidate_D"))
    A_ev = _evidence_map(candidate_A, reference)
    D_ev = _evidence_map(candidate_D, reference)
    A = {k: e["status"] for k, e in A_ev.items()}
    D = {k: e["status"] for k, e in D_ev.items()}
    A_fails_primary = all(A[p] == FAIL for p in primary)
    non_attr = [k for k in A if k not in primary and k not in allowed]
    A_spec = {k: A[k] == PASS for k in non_attr}
    return AblationOutcome(component, replicate_seed, source_profile, A, D, A_fails_primary, A_spec,
                           all(A_spec.values()), all(v == PASS for v in D.values()),
                           A_evidence=A_ev, D_evidence=D_ev)


# ---------------------------------------------------------------------------------------------------
# controls (fail-closed, self-contained, with exact expected-status maps)
# ---------------------------------------------------------------------------------------------------
from math import log as _log

_MIMIC_PROF = {"length": {"family": "discretized_lognormal", "mu": _log(60), "sigma": 0.35, "min": 1},
               "class_prior": [0.10, 0.15, 0.20, 0.25, 0.30], "structural_zero_classes": [],
               "cluster_size": {"family": "geometric", "p": 0.45},
               "gap": {"family": "lognormal", "mu": _log(1.5), "sigma": 0.85}, "dependence": {}}
_SWAP_PROF = {**_MIMIC_PROF, "class_prior": [0.55, 0.20, 0.15, 0.07, 0.03],
              "cluster_size": {"family": "geometric", "p": 0.55},
              "gap": {"family": "lognormal", "mu": _log(1.0), "sigma": 0.85}}   # mimic length x SCID class/run/gap
_ZERO_PROF = {**_MIMIC_PROF, "class_prior": [0.40, 0.35, 0.25, 0.0, 0.0], "structural_zero_classes": [3, 4]}
_SHORT_PROF = {**_MIMIC_PROF, "length": {"family": "discretized_lognormal", "mu": _log(6), "sigma": 0.3, "min": 1}}


def _multiscale(prof, source_key, tag, seed, n_each):
    recs = []
    for i, mu in enumerate((_log(18), _log(60), _log(250))):
        p = dict(prof, length={**prof["length"], "mu": mu})
        recs += sample_fixture(source_key, p, n_each, seed=_derive_seed(tag, seed, i))
    return recs


def null_control(replicate_seed, *, base_sampler, source_profile) -> dict:
    ev = _evidence_map(base_sampler(source_profile, replicate_seed, "null_cand"),
                       base_sampler(source_profile, replicate_seed, "null_ref"))
    st = {k: e["status"] for k, e in ev.items()}
    return {"all_pass": all(v == PASS for v in st.values()), "status": st, "evidence": ev,
            "fails": [k for k, v in st.items() if v == FAIL],
            "not_evaluable": [k for k, v in st.items() if v == NOT_EVALUABLE]}


def boundary_control(replicate_seed, *, n_each=600) -> dict:
    """Boundary-short (single-scale short) self-recovery: predeclared length-conditioned refusals
    (NOT_EVALUABLE) are successful; every OTHER check must PASS (no broad 'anything except FAIL')."""
    ref = [r for i in range(3) for r in sample_fixture("MIMIC", _SHORT_PROF, n_each, seed=_derive_seed("b_ref", replicate_seed, i))]
    cand = [r for i in range(3) for r in sample_fixture("MIMIC", _SHORT_PROF, n_each, seed=_derive_seed("b_cand", replicate_seed, i))]
    ev = _evidence_map(cand, ref)
    st = {k: e["status"] for k, e in ev.items()}
    unexpected = [k for k in st if (k in _BOUNDARY_EXPECTED_NE and st[k] == FAIL)
                  or (k not in _BOUNDARY_EXPECTED_NE and st[k] != PASS)]
    return {"ok": not unexpected, "status": st, "evidence": ev, "unexpected": unexpected}


def structural_zero_control(replicate_seed, *, n_each=600) -> dict:
    ref = _multiscale(_ZERO_PROF, "MIMIC", "z_ref", replicate_seed, n_each)
    cand = _multiscale(_ZERO_PROF, "MIMIC", "z_cand", replicate_seed, n_each)
    ev = _evidence_map(cand, ref)
    st = {k: e["status"] for k, e in ev.items()}
    present = set(np.unique(np.concatenate([r.class_ids for r in ref + cand])).tolist())
    required_pass = all(v == PASS for v in st.values())     # normal lengths => every check must PASS
    return {"zeros_absent": present.issubset({0, 1, 2}), "required_pass": required_pass, "status": st,
            "evidence": ev, "ok": present.issubset({0, 1, 2}) and required_pass}


def source_swap_control(replicate_seed, *, n_each=600) -> dict:
    ref = _multiscale(_MIMIC_PROF, "MIMIC", "swap_ref", replicate_seed, n_each)
    cand = _multiscale(_SWAP_PROF, "MIMIC", "swap_cand", replicate_seed, n_each)
    ev = _evidence_map(cand, ref)
    st = {k: e["status"] for k, e in ev.items()}
    return {"fails_nondegenerate": any(st.get(k) == FAIL for k in _NONDEGENERATE),
            "fails": [k for k, v in st.items() if v == FAIL], "status": st, "evidence": ev}


# ---------------------------------------------------------------------------------------------------
# rate-based aggregation + conjunctive verdict (step-4 power object)
# ---------------------------------------------------------------------------------------------------
def rate_battery(components, seeds, *, base_sampler, source_profiles=SOURCE_PROFILES) -> dict:
    out = {}
    for comp in components:
        primary = ABLATION_MATRIX[comp]["primary_fail"]
        per_source = {}
        for sp in source_profiles:
            res = [component_ablation(comp, s, base_sampler=base_sampler, source_profile=sp) for s in seeds]
            per_source[sp] = {
                "primary_fail_per_check": _per_check_rate([{p: r.A_status.get(p) == FAIL for p in primary}
                                                           for r in res]),
                "specificity_per_check": _per_check_rate([r.A_specificity for r in res]),
                "repeatability_rate": _rate([r.known_profile_repeatability for r in res]),
                "not_evaluable_per_check": _per_check_ne([r.A_status for r in res]),
            }
        out[comp] = {"per_source": per_source, "verdict": _component_verdict(comp, per_source, source_profiles)}
    return out


def _component_verdict(comp, per_source, source_profiles) -> dict:
    primary = ABLATION_MATRIX[comp]["primary_fail"]
    def ok_source(sp):
        ps = per_source[sp]
        prim = all(ps["primary_fail_per_check"][p]["k"] >= PRIMARY_FAIL_MIN for p in primary)
        spec = all(v["k"] >= SPECIFICITY_MIN for v in ps["specificity_per_check"].values())
        rep = ps["repeatability_rate"]["k"] >= SPECIFICITY_MIN
        return prim and spec and rep
    per = {sp: ok_source(sp) for sp in source_profiles}
    return {"per_source_ok": per, "conjunctive_pass": all(per.values())}


def _per_check_rate(list_of_maps) -> dict:
    keys = set().union(*[set(m) for m in list_of_maps]) if list_of_maps else set()
    return {k: _rate([m.get(k, False) for m in list_of_maps]) for k in sorted(keys)}


def _per_check_ne(list_of_status_maps) -> dict:
    keys = set().union(*[set(m) for m in list_of_status_maps]) if list_of_status_maps else set()
    return {k: sum(1 for m in list_of_status_maps if m.get(k) == NOT_EVALUABLE) for k in sorted(keys)}


def _rate(bools) -> dict:
    k = int(sum(bools)); n = len(bools)
    p = k / n if n else 0.0
    z = 1.959963984540054
    denom = 1 + z * z / n if n else 1
    centre = (p + z * z / (2 * n)) / denom if n else 0.0
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom if n else 0.0
    return {"k": k, "n": n, "rate": round(p, 4), "ci95": [round(centre - half, 4), round(centre + half, 4)]}


def forecast(base_sampler, *, source_profile="mimic_scale_control", seed=1000,
             secs_per_million_events=None) -> dict:
    """Cost forecast scaled by measured EVENT / CLUSTER / adjacent-PAIR volume (not sequence count alone)."""
    s = base_sampler(source_profile, seed, "forecast")
    n = len(s)
    events = int(sum(r.L_total for r in s))
    clusters = int(sum(r.K for r in s))
    pairs = int(sum(max(0, r.K - 1) for r in s))       # adjacent-cluster pairs
    per_call = (events / 1e6) * secs_per_million_events if secs_per_million_events else None
    return {"n_sequences": n, "total_events": events, "total_clusters": clusters, "adjacent_pairs": pairs,
            "est_secs_per_verifier_call": None if per_call is None else round(per_call, 1)}


# --- samplers -------------------------------------------------------------------------------------
def registered_base_sampler(n: int = REGISTERED_N):
    """Registered per-source-PROFILE sampler at N (default 4000). Draws from the hashed design PROFILES with a
    deterministic seed derived from (source, profile, replicate_seed, role) — genuinely distinct source blocks."""
    def sampler(source_profile, replicate_seed, role):
        prof = PROFILES[source_profile]
        seed = _derive_seed("fixture", source_profile, replicate_seed, role)
        return sample_fixture(_source_key(source_profile), prof, n, seed=seed)
    return sampler


def multiscale_smoke_sampler(n_each: int = 600):
    """SMALL mechanical sampler for the machinery smoke — three length strata so length-conditioned checks can
    evaluate cheaply. Distinct per (source_profile, replicate_seed, role). NOT the registered run contract."""
    from math import log
    def sampler(source_profile, replicate_seed, role):
        prof = PROFILES[source_profile]
        recs = []
        for i, mu in enumerate((log(18), log(60), log(250))):
            p = dict(prof, length={**prof["length"], "mu": mu})
            recs += sample_fixture(_source_key(source_profile), p, n_each,
                                   seed=_derive_seed("smoke", source_profile, replicate_seed, role, i))
        return recs
    return sampler


BATTERY_IMPL = {
    "name": "realism_v2_battery_dev",
    "fail_closed": "required check satisfied ONLY on PASS; NOT_EVALUABLE non-passing, reported separately",
    "orientation": "reference has component@0.5; candidate_A(null) FAILs every primary + PASSes non-attributed; "
                   "candidate_D establishes KNOWN-PROFILE REPEATABILITY (same constructor, distinct RNG) — NOT "
                   "implementation recovery (that is the later M2 adapter comparison)",
    "components": list(V2_D_COMPONENT_MENU),
    "controls": ["null", "boundary_short", "structural_zero", "source_swap"],
    "boundary_expected_not_evaluable": sorted(_BOUNDARY_EXPECTED_NE),
    "ablation_strength": _ABL_S,
    "registered_n": REGISTERED_N,
    "source_profiles": list(SOURCE_PROFILES),
    "rng_derivation": "fixture: (source,profile,replicate_seed,role); coupling: (source,component,seed,role)",
    "verdict": {"primary_fail_min": PRIMARY_FAIL_MIN, "specificity_min": SPECIFICITY_MIN,
                "source_conjunction": "each source independently satisfies its criterion",
                "not_evaluable": "always non-passing, reported separately"},
    "source_swap_never_D": True,
    "forecast": "scaled by event/cluster/adjacent-pair volume",
}


def battery_impl_identity() -> str:
    return canonical_hash(BATTERY_IMPL)
