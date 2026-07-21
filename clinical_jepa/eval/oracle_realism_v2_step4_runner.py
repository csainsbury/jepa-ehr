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
    "oracle_realism_v2_identifiability", "oracle_realism_v2_step4_runner", "oracle_realism_v2_step4_exec",
    "oracle_contracts",
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _git_head() -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# STATIC in-code trust root: a manifest is trusted ONLY if the WHOLE thing matches these registered constants
# (never caller-supplied fields). Any deviation refuses (Pi §1 — the old preflight was fail-OPEN).
TRUSTED_JOB_KINDS = ("m3a-step4-power-v1", "m3a-step4-ident-v1")
_TRUSTED_VERDICT = {"primary_fail_min": PRIMARY_FAIL_MIN, "specificity_min": SPECIFICITY_MIN,
                    "not_evaluable": "always non-passing", "source_conjunction": True}
_TRUSTED_RNG = {"fixture": "(source,profile,replicate_seed,role)",
                "coupling": "(source,component,replicate_seed,role)"}


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
    from clinical_jepa.eval.oracle_realism_v2_identifiability import identifiability_impl_identity
    return {
        "fixture": fixture_impl_identity(),
        "verifier": verifier_impl_identity(),
        "coupling": coupling_impl_identity(),
        "battery": battery_impl_identity(),
        "design": m3a_design_dev_hash(),
        "identifiability": identifiability_impl_identity(),
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
        "identifiability": _identifiability_binding(),
        "cap": CAP,
        "atomic_result_path": "state/realism-v2/step4/<run_id>/result.json (write-temp-then-rename)",
        "checkpoint": "per (component, source, seed) replicate; deterministic resume by manifest_hash + index",
        "completion_hashes": ["result_sha256", "denominator_map_sha256", "runtime_json_sha256", "env_sha256"],
    }
    manifest["manifest_hash"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def _identifiability_binding() -> dict:
    from clinical_jepa.eval.oracle_realism_v2_identifiability import cost_forecast, GRID, RANK_MIN
    return {"grid": list(GRID), "rank_min": RANK_MIN, "cost_forecast": cost_forecast(n=REGISTERED_N),
            "cap_note": "full grid may exceed the cap => PARTIAL / non-pass / re-gate; never silent reduction"}


def verify_identities(manifest: dict) -> dict:
    """Identity sub-check: recompute every bound identity + source-profile hash and compare (fail-closed)."""
    live = _identity_bindings()
    mism = {k: {"manifest": manifest["identities"].get(k), "live": live[k]}
            for k in live if manifest["identities"].get(k) != live[k]}
    for sp in SOURCE_PROFILES:
        if manifest.get("source_profiles", {}).get(sp) != canonical_hash(PROFILES[sp]):
            mism[f"profile:{sp}"] = {"manifest": manifest.get("source_profiles", {}).get(sp),
                                     "live": canonical_hash(PROFILES[sp])}
    return {"ok": not mism, "mismatches": mism}


def verify_manifest(manifest: dict, *, require_git_head: bool = True) -> dict:
    """FAIL-CLOSED preflight against the STATIC in-code trust root (Pi §1). Verifies the WHOLE manifest —
    manifest_hash integrity, every registered field against trusted constants, identities, and (optionally) that
    live git HEAD equals the reviewed commit. ANY deviation refuses. Caller-supplied fields are NEVER trusted."""
    problems = []
    # (a) manifest_hash integrity
    recomputed = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if manifest.get("manifest_hash") != recomputed:
        problems.append("manifest_hash")
    # (b) whole manifest vs trusted registered constants
    if manifest.get("seeds") != list(SEEDS):
        problems.append("seeds")
    if manifest.get("registered_n") != REGISTERED_N:
        problems.append("registered_n")
    if manifest.get("components") != list(V2_D_COMPONENT_MENU):
        problems.append("components")
    if set(manifest.get("source_profiles", {})) != set(SOURCE_PROFILES):
        problems.append("source_profiles")
    if manifest.get("verdict") != _TRUSTED_VERDICT:
        problems.append("verdict")
    if manifest.get("cap") != CAP:
        problems.append("cap")
    if manifest.get("rng_derivation") != _TRUSTED_RNG:
        problems.append("rng_derivation")
    if manifest.get("identifiability_vector") != list(IDENTIFIABILITY_VECTOR):
        problems.append("identifiability_vector")
    # (c) git HEAD == reviewed commit (short or full)
    if require_git_head:
        head = _git_head()
        rc = str(manifest.get("reviewed_commit", ""))
        if not head or not rc or not (head == rc or head.startswith(rc)):
            problems.append("reviewed_commit_vs_git_head")
    # (d) identities + source-profile hashes
    idv = verify_identities(manifest)
    if not idv["ok"]:
        problems.append("identities")
    return {"ok": not problems, "problems": problems, "identity_mismatches": idv["mismatches"]}


def dry_run(manifest: dict, *, n: int = 600, seeds=(1000,), components=None, require_git_head: bool = True) -> dict:
    """Small MECHANICAL dry-run proving the pipeline end-to-end WITHOUT the full grid. FAIL-CLOSED manifest
    verification first (whole manifest vs trust root); then one small-N ablation + the four controls; returns
    statuses + a volume forecast. It does NOT establish power (that is the full registered run)."""
    v = verify_manifest(manifest, require_git_head=require_git_head)
    if not v["ok"]:
        return {"refused": True, "reason": "manifest verification failed", "problems": v["problems"]}
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


def run_full_battery(manifest: dict, *, run_id: str, out_base: str):
    """The actual registered power/control run (guarded — launched only by the reviewed step-4 job, not by
    import). Delegates to the fail-closed execution engine: manifest verification (git HEAD on), then the full
    25-seed source-conjunction battery + controls under the cap, with per-replicate checkpoint/resume, atomic
    result writing, and persisted denominator/runtime hashes. A cap-exceed or resume mismatch yields
    NON-passing PARTIAL."""
    from clinical_jepa.eval.oracle_realism_v2_step4_exec import execute
    return execute(manifest, run_id, out_base,
                   base_sampler=registered_base_sampler(n=manifest["registered_n"]),
                   seeds=manifest["seeds"], sources=tuple(manifest["source_profiles"]),
                   components=manifest["components"],
                   cap_hours=manifest["cap"]["wall_clock_hours"], cap_gb=manifest["cap"]["ram_gb"],
                   verify=lambda mm: verify_manifest(mm, require_git_head=True))


def env_hash() -> str:
    import numpy as np
    try:
        import scipy
        scv = scipy.__version__
    except Exception:
        scv = "n/a"
    return canonical_hash({"python": os.sys.version.split()[0], "numpy": np.__version__, "scipy": scv})
