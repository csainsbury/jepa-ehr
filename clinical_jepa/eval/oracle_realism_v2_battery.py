"""Control / ablation battery for the realism-v2 verifier (rebuild step 3, final increment; Pi F3 fold).

FAIL-CLOSED (Pi): a required check is satisfied ONLY on PASS; NOT_EVALUABLE is non-passing and reported
separately. For each active D component the REFERENCE carries the component at 0.5; a null-independent
`candidate_A` must FAIL every primary subcheck and PASS every required non-attributed check; an independent
`candidate_D_recovery` (component at 0.5, distinct seed) must PASS every required check. Controls: null/self,
boundary-short, structural-zero, source-swap (negative, never D). Rate-based multi-seed harness reports
PER-CHECK PASS/FAIL/NOT_EVALUABLE rates with Wilson CIs and a source-wise conjunction — the step-4 power object.

Synthetic-only; uses the INDEPENDENT fixture + coupling constructions; no candidate adapter, no governed read.
The base sampler is injected (source, seed) so step 4 registers exact profiles/sizes without changing logic.
"""
from __future__ import annotations

import dataclasses
from math import log

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture, C
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, FAIL, PASS, NOT_EVALUABLE,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX

_ABL_S = 0.5             # ablation strength — the design fixes "one component at 0.5"
_NONDEGENERATE = ["count_ks", "positive_gap_ks", "class_tv", "S1_density", "S1_tau"]


def _status_map(cand, ref) -> dict:
    out = {k: v.status for k, v in sequence_route_checks(cand, ref).items()}
    out.update({k: v.status for k, v in marginal_route_checks(cand, ref).items()})
    return out


@dataclasses.dataclass(frozen=True)
class AblationOutcome:
    component: str
    seed: int
    source: str
    A_status: dict                 # candidate_A vs reference — per-check status
    D_status: dict                 # candidate_D_recovery vs reference — per-check status
    A_fails_primary: bool          # every primary subcheck == FAIL
    A_specificity: dict            # per non-attributed check: True iff PASS (fail-closed)
    A_specificity_ok: bool         # every non-attributed check == PASS
    D_recovers: bool               # every required check == PASS


def component_ablation(component, seed, *, base_sampler, source="MIMIC", coupling_seed=7) -> AblationOutcome:
    if component not in V2_D_COMPONENT_MENU:
        raise KeyError(component)
    primary = ABLATION_MATRIX[component]["primary_fail"]
    allowed = set(ABLATION_MATRIX[component]["allowed_sensitive"])
    reference = apply_coupling(base_sampler(source, seed), component, _ABL_S, seed=coupling_seed)
    candidate_A = base_sampler(source, seed + 10_000)
    candidate_D = apply_coupling(base_sampler(source, seed + 20_000), component, _ABL_S, seed=coupling_seed + 1)
    A = _status_map(candidate_A, reference)
    D = _status_map(candidate_D, reference)
    A_fails_primary = all(A[p] == FAIL for p in primary)                # PASS/NOT_EVALUABLE both fail this
    non_attr = [k for k in A if k not in primary and k not in allowed]
    A_spec = {k: A[k] == PASS for k in non_attr}                        # fail-closed: NOT_EVALUABLE is False
    D_recovers = all(v == PASS for v in D.values())                    # every required check PASS only
    return AblationOutcome(component, seed, source, A, D, A_fails_primary, A_spec,
                           all(A_spec.values()), D_recovers)


def null_control(seed, *, base_sampler, source="MIMIC") -> dict:
    """Independent vs independent — every required check must PASS (fail-closed false-positive control)."""
    st = _status_map(base_sampler(source, seed + 30_000), base_sampler(source, seed))
    return {"all_pass": all(v == PASS for v in st.values()), "status": st,
            "fails": [k for k, v in st.items() if v == FAIL],
            "not_evaluable": [k for k, v in st.items() if v == NOT_EVALUABLE]}


def boundary_control(seed, *, source="MIMIC", n_each=600) -> dict:
    """Boundary-short self-recovery: must NOT FALSELY FAIL (NOT_EVALUABLE is acceptable at the support edge)."""
    prof = _profile(log(6), [0.3, 0.25, 0.2, 0.15, 0.1], 0.5, log(1.2), sigma=0.3)
    ref = _multiscale_from(prof, seed, n_each); cand = _multiscale_from(prof, seed + 50_000, n_each)
    st = _status_map(cand, ref)
    return {"no_false_fail": not any(v == FAIL for v in st.values()), "status": st}


def structural_zero_control(seed, *, n_each=600) -> dict:
    """Structural-zero self-recovery: no false FAIL AND the zeroed classes never appear."""
    prof = _profile(log(60), [0.40, 0.35, 0.25, 0.0, 0.0], 0.5, log(1.2)); prof["structural_zero_classes"] = [3, 4]
    ref = _multiscale_from(prof, seed, n_each); cand = _multiscale_from(prof, seed + 60_000, n_each)
    st = _status_map(cand, ref)
    present = set(np.unique(np.concatenate([r.class_ids for r in ref + cand])).tolist())
    return {"no_false_fail": not any(v == FAIL for v in st.values()),
            "zeros_absent": present.issubset({0, 1, 2}), "status": st}


def source_swap_control(seed, *, n_each=600) -> dict:
    """Reference = mimic-scale marginals; candidate = mimic length x SCID-scale class/run/gap. Must FAIL a
    NON-degenerate check; NEVER triggers D (negative control, no coupling)."""
    mimic = _profile(log(60), [0.10, 0.15, 0.20, 0.25, 0.30], 0.45, log(1.5))
    swap = _profile(log(60), [0.55, 0.20, 0.15, 0.07, 0.03], 0.55, log(1.0))
    st = _status_map(_multiscale_from(swap, seed + 40_000, n_each), _multiscale_from(mimic, seed, n_each))
    return {"fails_nondegenerate": any(st.get(k) == FAIL for k in _NONDEGENERATE),
            "fails": [k for k, v in st.items() if v == FAIL], "status": st}


# ---------------------------------------------------------------------------------------------------
# rate-based aggregation (step-4 power object) — PER-CHECK rates + Wilson CI + source conjunction
# ---------------------------------------------------------------------------------------------------
def rate_battery(components, seeds, *, base_sampler, sources=("MIMIC", "SCID")) -> dict:
    out = {}
    for comp in components:
        per_source = {}
        for src in sources:
            res = [component_ablation(comp, s, base_sampler=base_sampler, source=src) for s in seeds]
            primary = ABLATION_MATRIX[comp]["primary_fail"]
            per_source[src] = {
                "A_fails_primary_rate": _rate([r.A_fails_primary for r in res]),
                "A_specificity_per_check": _per_check_rate([r.A_specificity for r in res]),
                "D_recovery_rate": _rate([r.D_recovers for r in res]),
                "primary": primary,
            }
        out[comp] = {"per_source": per_source,
                     "conjunction_A_fails_primary": all(per_source[s]["A_fails_primary_rate"]["k"] ==
                                                        per_source[s]["A_fails_primary_rate"]["n"] for s in sources)}
    return out


def _per_check_rate(list_of_maps) -> dict:
    keys = set().union(*[set(m) for m in list_of_maps]) if list_of_maps else set()
    return {k: _rate([m.get(k, False) for m in list_of_maps]) for k in sorted(keys)}


def _rate(bools) -> dict:
    k = int(sum(bools)); n = len(bools)
    p = k / n if n else 0.0
    z = 1.959963984540054
    denom = 1 + z * z / n if n else 1
    centre = (p + z * z / (2 * n)) / denom if n else 0.0
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom if n else 0.0
    return {"k": k, "n": n, "rate": round(p, 4), "ci95": [round(centre - half, 4), round(centre + half, 4)]}


def forecast(base_sampler, *, source="MIMIC", seed=1, per_call_secs_at_4k=13.0) -> dict:
    """Event-volume / runtime forecast for the full run (Pi): sequences, mean length, total events, and a
    verifier-call estimate scaled from the measured ~13s/4000-sequence cost."""
    s = base_sampler(source, seed)
    n = len(s); mean_L = float(np.mean([r.L_total for r in s]))
    events = int(n * mean_L)
    est = per_call_secs_at_4k * (n / 4000.0)
    return {"n_sequences": n, "mean_length": round(mean_L, 1), "total_events": events,
            "est_secs_per_verifier_call": round(est, 1)}


# --- profile helpers (synthetic; only the length scale anchored) -----------------------------------
def _profile(length_mu, class_prior, cluster_p, gap_mu, *, sigma=0.35):
    return {"length": {"family": "discretized_lognormal", "mu": length_mu, "sigma": sigma, "min": 1},
            "class_prior": class_prior, "structural_zero_classes": [],
            "cluster_size": {"family": "geometric", "p": cluster_p},
            "gap": {"family": "lognormal", "mu": gap_mu, "sigma": 0.85}, "dependence": {}}


def _multiscale_from(profile, seed, n_each):
    recs = []
    src = "MIMIC" if profile.get("_src", "MIMIC") == "MIMIC" else "SCID"
    for i, mu in enumerate((log(18), log(60), log(250))):
        p = dict(profile, length={**profile["length"], "mu": mu})
        recs += sample_fixture(src, p, n_each, seed=seed * 10 + i)
    return recs


def default_base_sampler(n_each: int = 600):
    """Multiscale null-independent base (3 length scales) so length-conditioned checks can evaluate. Injected
    as (source, seed) -> sample so the real step-4 battery can register per-source profiles/sizes."""
    base = _profile(log(60), [0.3, 0.25, 0.2, 0.15, 0.1], 0.5, log(1.2))

    def sampler(source, seed):
        p = dict(base, _src=source)
        return _multiscale_from(p, seed, n_each)
    return sampler


BATTERY_IMPL = {
    "name": "realism_v2_battery_dev",
    "fail_closed": "required check satisfied ONLY on PASS; NOT_EVALUABLE is non-passing, reported separately",
    "orientation": "reference has component@0.5; candidate_A(null) FAILs every primary + PASSes non-attributed; "
                   "candidate_D_recovery(component@0.5, distinct seed) PASSes every required check",
    "components": list(V2_D_COMPONENT_MENU),
    "controls": ["null", "boundary_short", "structural_zero", "source_swap"],
    "ablation_strength": _ABL_S,
    "source_conjunction": True,
    "rate_rule": "per-check PASS/FAIL/NOT_EVALUABLE rates; self/known >=24/25; misspec-fail >=20/25; "
                 ">=24/25 specificity PER non-attributed check; Wilson CI; source-wise conjunction",
    "source_swap_never_D": True,
    "forecast": "per-source sequences + mean length + total events + verifier-call runtime estimate",
}


def battery_impl_identity() -> str:
    return canonical_hash(BATTERY_IMPL)
