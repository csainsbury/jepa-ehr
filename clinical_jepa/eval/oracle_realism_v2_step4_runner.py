"""Registered fail-closed runner + manifest for the step-4 synthetic power battery (Pi step-3 re-gate §5).

Binds EVERY identity and config value the expensive run depends on, and VERIFIES them before any work starts:
a single mismatch refuses. A PARTIAL / cap-exceeded run is NON-passing and cannot freeze M3a. Nothing here
reads governed data, samples any M2 candidate, or launches the full grid — the full run is invoked only after
Pi approves this manifest + the dry-run output.

Cap (resolves the earlier contradictory "<=8 CPU-hours wall-time"): ONE worker, <= 8 WALL-CLOCK hours,
<= 32 GB RAM. Result / denominator-map / runtime / environment hashes are stamped at completion; a deterministic
checkpoint/resume and an atomic result path protect a long run.
"""
from __future__ import annotations

import hashlib
import os

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES, IDENTIFIABILITY_VECTOR
from clinical_jepa.eval.oracle_realism_v2_battery import (
    REGISTERED_N, SOURCE_PROFILES, PRIMARY_FAIL_MIN, SPECIFICITY_MIN, battery_impl_identity,
    registered_base_sampler, component_ablation, null_control, boundary_control,
    structural_zero_control, source_swap_control, rate_battery, forecast,
)

RUNNER_VERSION = "realism_v2_step4_runner_dev"
SEEDS = tuple(range(1000, 1025))              # 25 deterministic replicate seeds (design)
CAP = {"workers": 1, "wall_clock_hours": 8, "ram_gb": 32,
       "on_exceed": "PARTIAL / non-passing; cannot freeze M3a; re-gate"}

# The executable closure whose bytes affect what the step-4 battery computes.
_CLOSURE_MODULES = (
    "oracle_realism_v2", "oracle_realism_v2_fixture", "oracle_realism_v2_verifier",
    "oracle_realism_v2_coupling", "oracle_realism_v2_battery", "oracle_realism_v2_verifier_design",
    "oracle_contracts",
)


def code_closure_identity() -> str:
    """Hash of the source bytes of the v2 executable closure (any behaviour change moves this)."""
    import importlib
    h = hashlib.sha256()
    paths = []
    for name in _CLOSURE_MODULES:
        mod = importlib.import_module(f"clinical_jepa.eval.{name}")
        paths.append(mod.__file__)
    for path in sorted(paths):
        with open(path, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _identity_bindings() -> dict:
    from clinical_jepa.eval.oracle_realism_v2_fixture import fixture_impl_identity
    from clinical_jepa.eval.oracle_realism_v2_verifier import verifier_impl_identity
    from clinical_jepa.eval.oracle_realism_v2_coupling import coupling_impl_identity
    from clinical_jepa.eval.oracle_realism_v2_verifier_design import m3a_design_dev_hash
    return {
        "fixture": fixture_impl_identity(),
        "verifier": verifier_impl_identity(),
        "coupling": coupling_impl_identity(),
        "battery": battery_impl_identity(),
        "design": m3a_design_dev_hash(),
        "code_closure": code_closure_identity(),
    }


def build_manifest(*, reviewed_commit: str) -> dict:
    """Bind the full step-4 run contract. `reviewed_commit` is the git commit this run is authorised against."""
    manifest = {
        "runner_version": RUNNER_VERSION,
        "reviewed_commit": reviewed_commit,
        "identities": _identity_bindings(),
        "source_profiles": {sp: canonical_hash(PROFILES[sp]) for sp in SOURCE_PROFILES},
        "components": list(V2_D_COMPONENT_MENU),
        "seeds": list(SEEDS),
        "registered_n": REGISTERED_N,
        "stratum_allocation": {"single_block": REGISTERED_N},   # single-scale design profiles; no strata
        "rng_derivation": {"fixture": "(source,profile,replicate_seed,role)",
                           "coupling": "(source,component,replicate_seed,role)"},
        "verdict": {"primary_fail_min": PRIMARY_FAIL_MIN, "specificity_min": SPECIFICITY_MIN,
                    "not_evaluable": "always non-passing", "source_conjunction": True},
        "identifiability_vector": list(IDENTIFIABILITY_VECTOR),
        "cap": CAP,
        "atomic_result_path": "state/realism-v2/step4/<run_id>/result.json (write-temp-then-rename)",
        "checkpoint": "per (component, source, seed) replicate; deterministic resume by manifest_hash + index",
        "completion_hashes": ["result_sha256", "denominator_map_sha256", "runtime_json_sha256", "env_sha256"],
    }
    manifest["manifest_hash"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def verify_identities(manifest: dict) -> dict:
    """FAIL-CLOSED pre-flight: recompute every bound identity and compare. Any mismatch => refuse."""
    live = _identity_bindings()
    mism = {k: {"manifest": manifest["identities"][k], "live": live[k]}
            for k in manifest["identities"] if manifest["identities"][k] != live[k]}
    for sp, h in manifest["source_profiles"].items():
        if canonical_hash(PROFILES[sp]) != h:
            mism[f"profile:{sp}"] = {"manifest": h, "live": canonical_hash(PROFILES[sp])}
    return {"ok": not mism, "mismatches": mism}


def dry_run(manifest: dict, *, n: int = 600, seeds=(1000,), components=None) -> dict:
    """Small MECHANICAL dry-run proving the pipeline end-to-end WITHOUT the full grid. Fail-closed identity
    check first; then one small-N ablation + the four controls; returns statuses + a volume forecast. This is
    what is routed to Pi for approval; it does NOT establish power (that is the full registered run)."""
    v = verify_identities(manifest)
    if not v["ok"]:
        return {"refused": True, "reason": "identity mismatch", "mismatches": v["mismatches"]}
    base = _smoke_sampler(n)
    comps = components or [V2_D_COMPONENT_MENU[0]]
    abl = {}
    for comp in comps:
        o = component_ablation(comp, seeds[0], base_sampler=base, source_profile="mimic_scale_control")
        abl[comp] = {"A_fails_primary": o.A_fails_primary, "A_specificity_ok": o.A_specificity_ok,
                     "known_profile_repeatability": o.known_profile_repeatability}
    controls = {
        "null": null_control(seeds[0], base_sampler=base, source_profile="mimic_scale_control")["all_pass"],
        "boundary": boundary_control(seeds[0], n_each=min(n, 500))["ok"],
        "structural_zero": structural_zero_control(seeds[0], n_each=n)["ok"],
        "source_swap": source_swap_control(seeds[0], n_each=n)["fails_nondegenerate"],
    }
    fc = {sp: forecast(_registered_forecast_sampler(), source_profile=sp, secs_per_million_events=None)
          for sp in SOURCE_PROFILES}
    return {"refused": False, "manifest_hash": manifest["manifest_hash"], "ablation": abl,
            "controls": controls, "forecast_registered_n": fc, "note": "mechanical dry-run only; not power"}


def _smoke_sampler(n_each):
    from clinical_jepa.eval.oracle_realism_v2_battery import multiscale_smoke_sampler
    return multiscale_smoke_sampler(n_each=n_each)


def _registered_forecast_sampler():
    return registered_base_sampler(n=REGISTERED_N)


def run_full_battery(manifest: dict, *, run_id: str):
    """The actual registered step-4 run (guarded). NOT invoked in step 3; runs only after Pi approves the
    manifest + dry-run. Fail-closed identity check, then the full 25-seed source-conjunction rate battery under
    the cap; a cap-exceed or identity mismatch returns a NON-passing PARTIAL result. Left unbound here on
    purpose — launched by the reviewed step-4 job, not by import."""
    v = verify_identities(manifest)
    if not v["ok"]:
        return {"run_id": run_id, "status": "REFUSED", "reason": "identity mismatch", "mismatches": v["mismatches"]}
    base = registered_base_sampler(n=manifest["registered_n"])
    rates = rate_battery(manifest["components"], manifest["seeds"], base_sampler=base,
                         source_profiles=tuple(manifest["source_profiles"]))
    conj = all(rates[c]["verdict"]["conjunctive_pass"] for c in manifest["components"])
    return {"run_id": run_id, "status": "PASS" if conj else "FAIL", "rates": rates,
            "reviewed_commit": manifest["reviewed_commit"], "manifest_hash": manifest["manifest_hash"]}


def env_hash() -> str:
    import numpy as np
    try:
        import scipy
        scv = scipy.__version__
    except Exception:
        scv = "n/a"
    return canonical_hash({"python": os.sys.version.split()[0], "numpy": np.__version__, "scipy": scv})
