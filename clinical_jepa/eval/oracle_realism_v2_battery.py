"""Control / ablation battery for the realism-v2 verifier (rebuild step 3, final increment).

Implements the frozen ablation ORIENTATION (design `ABLATION_ORIENTATION` / `ABLATION_MATRIX`): for each active
D component, the REFERENCE carries that component at 0.5; a null-independent `candidate_A` must FAIL the
component's primary subcheck(s) and pass every non-attributed check; an independent `candidate_D_recovery` (same
component at 0.5, distinct seed) must PASS the full row. Plus null / boundary / source-swap controls. A
rate-based multi-seed harness (`rate_battery`) produces empirical pass/fail RATES + binomial CIs — the
step-4 power object; step 3 exercises the machinery at a small seed count.

Synthetic-only. Uses the INDEPENDENT fixture + coupling constructions; no candidate adapter, no governed read.
The base sampler is injected so step 4 can register the exact profile/sizes without changing this logic.
"""
from __future__ import annotations

import dataclasses
from math import log

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, FAIL, PASS, NOT_EVALUABLE,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX

_ABL_S = 0.5     # ablation strength — the design ABLATION_ORIENTATION fixes "one component at 0.5"


def _all_checks(cand, ref) -> dict:
    out = dict(sequence_route_checks(cand, ref))
    out.update(marginal_route_checks(cand, ref))
    return out


@dataclasses.dataclass(frozen=True)
class AblationOutcome:
    component: str
    seed: int
    A_fails_primary: bool          # candidate_A FAILs every primary subcheck
    A_specificity_ok: bool         # candidate_A does NOT FAIL any non-attributed check
    D_recovers: bool               # candidate_D_recovery does NOT FAIL anything
    detail: dict = dataclasses.field(default_factory=dict)


def component_ablation(component: str, seed: int, *, base_sampler, coupling_seed: int = 7) -> AblationOutcome:
    """One ablation replicate for ``component`` at seed. reference has the component at 0.5; candidate_A is a
    matched null fixture; candidate_D_recovery is an independent fixture with the component at 0.5."""
    if component not in V2_D_COMPONENT_MENU:
        raise KeyError(component)
    primary = ABLATION_MATRIX[component]["primary_fail"]
    allowed = set(ABLATION_MATRIX[component]["allowed_sensitive"])
    reference = apply_coupling(base_sampler(seed), component, _ABL_S, seed=coupling_seed)
    candidate_A = base_sampler(seed + 10_000)                                  # null-independent
    candidate_D = apply_coupling(base_sampler(seed + 20_000), component, _ABL_S, seed=coupling_seed + 1)
    ca = _all_checks(candidate_A, reference)
    cd = _all_checks(candidate_D, reference)
    A_fails_primary = all(ca[p].status == FAIL for p in primary)
    non_attr = [k for k in ca if k not in primary and k not in allowed]
    A_specificity_ok = all(ca[k].status != FAIL for k in non_attr)            # PASS or NOT_EVALUABLE
    D_recovers = all(v.status != FAIL for v in cd.values())
    return AblationOutcome(
        component, seed, A_fails_primary, A_specificity_ok, D_recovers,
        {"primary": primary, "A_primary_status": {p: ca[p].status for p in primary},
         "A_fails_nonattr": [k for k in non_attr if ca[k].status == FAIL],
         "D_fails": [k for k, v in cd.items() if v.status == FAIL]})


def null_control(seed: int, *, base_sampler) -> dict:
    """Independent vs independent — nothing should FAIL (a false-positive control)."""
    c = _all_checks(base_sampler(seed + 30_000), base_sampler(seed))
    return {"ok": all(v.status != FAIL for v in c.values()),
            "fails": [k for k, v in c.items() if v.status == FAIL]}


def source_swap_control(seed: int, *, n_each: int = 600) -> dict:
    """Reference = mimic-scale marginals; candidate = mimic length x SCID-scale class/run/gap. Must FAIL a
    NON-degenerate check; NEVER triggers D (it is a control, no coupling)."""
    mimic = _profile(log(60), [0.10, 0.15, 0.20, 0.25, 0.30], 0.45, log(1.5))
    swap = _profile(log(60), [0.55, 0.20, 0.15, 0.07, 0.03], 0.55, log(1.0))   # mimic length, SCID class/run/gap
    ref = _multiscale_from(mimic, seed, n_each)
    cand = _multiscale_from(swap, seed + 40_000, n_each)
    c = _all_checks(cand, ref)
    nondeg = ["count_ks", "positive_gap_ks", "class_tv", "S1_density", "S1_tau"]
    return {"fails_nondegenerate": any(c[k].status == FAIL for k in nondeg if k in c),
            "fails": [k for k, v in c.items() if v.status == FAIL]}


def rate_battery(components, seeds, *, base_sampler) -> dict:
    """Rate-based aggregation over seeds (step-4 power object). Reports per-component empirical rates + a
    Wilson 95% CI. self/known >=24/25 and misspec-fail >=20/25 are checked by the caller against these rates."""
    out = {}
    for comp in components:
        res = [component_ablation(comp, s, base_sampler=base_sampler) for s in seeds]
        n = len(res)
        out[comp] = {
            "n": n,
            "A_fails_primary_rate": _rate([r.A_fails_primary for r in res]),
            "A_specificity_rate": _rate([r.A_specificity_ok for r in res]),
            "D_recovery_rate": _rate([r.D_recovers for r in res]),
        }
    return out


def _rate(bools) -> dict:
    k = int(sum(bools)); n = len(bools)
    p = k / n if n else 0.0
    # Wilson 95% CI
    z = 1.959963984540054
    denom = 1 + z * z / n if n else 1
    centre = (p + z * z / (2 * n)) / denom if n else 0.0
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom if n else 0.0
    return {"k": k, "n": n, "rate": round(p, 4), "ci95": [round(centre - half, 4), round(centre + half, 4)]}


# --- profile helpers (synthetic; only the length scale anchored) -----------------------------------
def _profile(length_mu, class_prior, cluster_p, gap_mu):
    return {"length": {"family": "discretized_lognormal", "mu": length_mu, "sigma": 0.35, "min": 1},
            "class_prior": class_prior, "structural_zero_classes": [],
            "cluster_size": {"family": "geometric", "p": cluster_p},
            "gap": {"family": "lognormal", "mu": gap_mu, "sigma": 0.85}, "dependence": {}}


def _multiscale_from(profile, seed, n_each):
    recs = []
    for i, mu in enumerate((log(18), log(60), log(250))):
        p = dict(profile, length={**profile["length"], "mu": mu})
        recs += sample_fixture("MIMIC", p, n_each, seed=seed * 10 + i)
    return recs


def default_base_sampler(n_each: int = 600):
    """Multiscale null-independent base (3 length scales) so length-conditioned checks can evaluate."""
    base = _profile(log(60), [0.3, 0.25, 0.2, 0.15, 0.1], 0.5, log(1.2))
    return lambda seed: _multiscale_from(base, seed, n_each)


BATTERY_IMPL = {
    "name": "realism_v2_battery_dev",
    "orientation": "reference has component@0.5; candidate_A(null) FAILs primary + passes non-attributed; "
                   "candidate_D_recovery(component@0.5, distinct seed) passes the full row",
    "components": list(V2_D_COMPONENT_MENU),
    "controls": ["null", "source_swap"],
    "ablation_strength": _ABL_S,
    "rate_rule": "self/known >=24/25; misspec-fail >=20/25; specificity per non-attributed check; Wilson CI",
    "source_swap_never_D": True,
}


def battery_impl_identity() -> str:
    return canonical_hash(BATTERY_IMPL)
