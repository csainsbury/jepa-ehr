"""Registered fail-closed runner + manifest for the step-4 synthetic power battery (Pi step-3 re-gate §5).

Binds EVERY identity and config value the expensive run depends on, and VERIFIES them before any work starts:
a single mismatch refuses. A PARTIAL / cap-exceeded run is NON-passing and cannot freeze M3a. Nothing here
reads governed data, samples any M2 candidate, or launches the full grid — the full run is invoked only after
Pi approves this manifest + the dry-run output.

Cap (resolves the earlier contradictory "<=8 CPU-hours wall-time"): ONE worker, <= 8 WALL-CLOCK hours,
<= 32 GB RAM. Result / evidence / runtime / environment hashes are stamped at completion (the evidence hash
covers the full CheckResult values + denominator/coarsening detail, not just status strings); a deterministic
checkpoint/resume and an atomic result path protect a long run.
"""
from __future__ import annotations

import hashlib
import os
import re as _re

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES, IDENTIFIABILITY_VECTOR
from clinical_jepa.eval.oracle_realism_v2_battery import (
    REGISTERED_N, CONTROL_N, CONTROL_ALLOC, SOURCE_PROFILES, PRIMARY_FAIL_MIN, SPECIFICITY_MIN, battery_impl_identity,
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
    "oracle_realism_v2_identifiability", "oracle_realism_v2_ident_runner", "oracle_realism_v2_step4_runner",
    "oracle_realism_v2_step4_exec", "oracle_contracts",
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
TRUSTED_JOB_KINDS = ("m3a-step4-power-v2", "m3a-step4-ident-v2")   # re-minted at the step-4 result-gate (Pi); v1 spent
_CONTROL_ROUTING = {"source_scoped": ["null"], "global": ["boundary", "structural_zero", "source_swap"]}
_TRUSTED_VERDICT = {"primary_fail_min": PRIMARY_FAIL_MIN, "specificity_min": SPECIFICITY_MIN,
                    "not_evaluable": "always non-passing", "source_conjunction": True,
                    "control_routing": _CONTROL_ROUTING}
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


def build_manifest(*, reviewed_commit: str, job_kind: str = "m3a-step4-power-v2") -> dict:
    """Bind ONE step-4 JOB's run contract (Pi §6). `job_kind` is one of TRUSTED_JOB_KINDS; the two jobs
    (power / ident) get DISTINCT manifest hashes and separate checkpoints/results. `reviewed_commit` is the git
    commit this run is authorised against."""
    if job_kind not in TRUSTED_JOB_KINDS:
        raise ValueError(f"unknown job_kind {job_kind!r}; expected one of {TRUSTED_JOB_KINDS}")
    manifest = {
        "runner_version": RUNNER_VERSION,
        "job_kind": job_kind,
        "reviewed_commit": reviewed_commit,
        "identities": _identity_bindings(),
        "source_profiles": {sp: canonical_hash(PROFILES[sp]) for sp in SOURCE_PROFILES},
        "components": list(V2_D_COMPONENT_MENU),
        "seeds": list(SEEDS),
        "registered_n": REGISTERED_N,
        "control_n": CONTROL_N,                                 # global controls at the registered N (Pi re-gate §3)
        "control_alloc": list(CONTROL_ALLOC),                  # EXACT per-stratum allocation summing to N (#1)
        "boundary_fixture": {"bound": "L<=7", "family": "uniform_int", "min": 1, "max": 7,   # bounded support (§4)
                             "canonical_profile_hash": canonical_hash(PROFILES["boundary_short"])},   # design==executed (#3)
        "stratum_allocation": {"single_block": REGISTERED_N},   # single-scale design profiles; no strata
        "rng_derivation": {"fixture": "(source,profile,replicate_seed,role)",
                           "coupling": "(source,component,replicate_seed,role)"},
        "verdict": {"primary_fail_min": PRIMARY_FAIL_MIN, "specificity_min": SPECIFICITY_MIN,
                    "not_evaluable": "always non-passing", "source_conjunction": True,
                    "control_routing": _CONTROL_ROUTING},
        "identifiability_vector": list(IDENTIFIABILITY_VECTOR),
        "identifiability": _identifiability_binding(),
        "cap": CAP,
        "atomic_result_path": "state/realism-v2/step4/<run_id>/result.json (write-temp-then-rename)",
        "checkpoint": "per (component, source, seed) replicate; deterministic resume by manifest_hash + index",
        "completion_hashes": ["result_sha256", "evidence_sha256", "runtime_json_sha256", "env_sha256"],
    }
    manifest["manifest_hash"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def _identifiability_binding() -> dict:
    from clinical_jepa.eval.oracle_realism_v2_identifiability import (
        cost_forecast, GRID, RANK_MIN, NUISANCE_PROFILES, ACCEPT_TOL, RECOVER_TOL,
    )
    return {"grid": list(GRID), "rank_min": RANK_MIN, "recover_tol": RECOVER_TOL,
            "nuisance_profiles": list(NUISANCE_PROFILES), "accept_tol": [round(float(x), 6) for x in ACCEPT_TOL],
            # structured-computation params bound in the manifest (Pi §1) — caller may NOT override:
            "ref_seed": 1000, "heldout_seed": 1024, "cov_seeds": list(SEEDS),
            "grid_values": list(GRID), "rank_point_values": list(GRID),
            "cost_forecast": cost_forecast(n=REGISTERED_N),
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
    """FAIL-CLOSED WHOLE-MANIFEST verification (Pi §1). Reconstructs the COMPLETE expected manifest from the
    trusted constants (via build_manifest at the manifest's reviewed_commit + job_kind) and requires DEEP
    EQUALITY — rejecting missing, extra, or tampered fields (no field allowlist). Also verifies live git HEAD
    == reviewed_commit. Caller-supplied fields are NEVER trusted. Does NOT authorize launch — see
    verify_launch."""
    problems = []
    jk = manifest.get("job_kind")
    if jk not in TRUSTED_JOB_KINDS:
        return {"ok": False, "problems": ["job_kind"], "diff": {"job_kind": jk}}
    rc = str(manifest.get("reviewed_commit", ""))
    expected = build_manifest(reviewed_commit=rc, job_kind=jk)
    # deep equality over the full manifest (both keys): rejects missing/extra/tampered fields
    keys = set(manifest) | set(expected)
    diff = {k: {"manifest": manifest.get(k, "<MISSING>"), "expected": expected.get(k, "<UNEXPECTED>")}
            for k in keys if manifest.get(k) != expected.get(k)}
    if diff:
        problems.append("manifest_not_equal_to_trusted_reconstruction")
    if require_git_head:
        head = _git_head()
        if not head or not rc or not (head == rc or head.startswith(rc)):
            problems.append("reviewed_commit_vs_git_head")
    return {"ok": not problems, "problems": problems, "diff": diff}


_GATE_EVENT_RE = _re.compile(r"^evt-[0-9A-Za-z_.-]+$")


def verify_launch(manifest: dict, *, run_id: str, job_kind: str) -> dict:
    """LAUNCH gate: whole-manifest verification PLUS the policy-data approval map (Pi §1). The (run_id, job_kind,
    reviewed_commit, manifest_hash) must be present in APPROVED_STEP4_JOBS (empty => nothing launches), the
    requested job_kind must match the manifest, AND the approval must carry a non-empty canonical ARR `gate_event`
    (Pi §3 — an entry lacking reviewed-gate provenance must NOT authorize execution). Any mismatch REFUSES. On
    success returns the approval (incl. gate_event) so the runner can persist it as run provenance."""
    from clinical_jepa.eval.oracle_realism_v2_step4_policy import APPROVED_STEP4_JOBS
    v = verify_manifest(manifest, require_git_head=True)
    problems = list(v["problems"])
    if manifest.get("job_kind") != job_kind:
        problems.append("job_kind_mismatch")
    appr = APPROVED_STEP4_JOBS.get(run_id)
    if appr is None:
        problems.append("run_id_not_approved")
    else:
        if not (appr.get("job_kind") == job_kind
                and appr.get("reviewed_commit") == manifest.get("reviewed_commit")
                and appr.get("manifest_hash") == manifest.get("manifest_hash")):
            problems.append("approval_mismatch")
        gate = str(appr.get("gate_event", ""))
        if not _GATE_EVENT_RE.match(gate):
            problems.append("gate_event_missing")
    return {"ok": not problems, "problems": problems,
            "gate_event": (appr or {}).get("gate_event") if not problems else None}


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
        "boundary": boundary_control(seeds[0], alloc=(min(n, 500),) * 3)["ok"],
        "structural_zero": structural_zero_control(seeds[0], alloc=(n, n, n))["ok"],
        "source_swap": source_swap_control(seeds[0], alloc=(n, n, n))["fails_nondegenerate"],
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


_STATE_ROOT = os.path.join(_REPO_ROOT, "state", "realism-v2", "step4")
_RUN_ID_RE = {"m3a-step4-power-v2": _re.compile(r"^m3a-step4-power-v2-run\d+$"),
              "m3a-step4-ident-v2": _re.compile(r"^m3a-step4-ident-v2-run\d+$")}


def _validate_run_id(run_id: str, job_kind: str) -> None:
    """Strict safe run-id + realpath containment under state/realism-v2/step4/ (Pi §2). No caller out_base."""
    if not _RUN_ID_RE[job_kind].match(run_id):
        raise ValueError(f"run_id {run_id!r} does not match the approved pattern for {job_kind}")
    resolved = os.path.realpath(os.path.join(_STATE_ROOT, run_id))
    root = os.path.realpath(_STATE_ROOT)
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError("run_id path traversal")


def run_full_battery(manifest: dict, *, run_id: str):
    """The registered power/control run (guarded). Output root is DERIVED internally from the repo + approved
    run_id (no caller out_base); the exact job kind is enforced; verify_launch requires the (run_id, job_kind,
    commit, manifest_hash) to be in the policy-data approval map. Delegates to the fail-closed execution
    engine."""
    job = "m3a-step4-power-v2"
    v = verify_launch(manifest, run_id=run_id, job_kind=job)
    if not v["ok"]:
        return {"run_id": run_id, "status": "REFUSED", "problems": v["problems"]}
    _validate_run_id(run_id, job)
    from clinical_jepa.eval.oracle_realism_v2_step4_exec import execute
    return execute(manifest, run_id, _REPO_ROOT,
                   base_sampler=registered_base_sampler(n=manifest["registered_n"]),
                   seeds=manifest["seeds"], sources=tuple(manifest["source_profiles"]),
                   components=manifest["components"],
                   cap_hours=manifest["cap"]["wall_clock_hours"], cap_gb=manifest["cap"]["ram_gb"],
                   verify=lambda mm: verify_manifest(mm, require_git_head=True), gate_event=v["gate_event"])


def run_full_identifiability(manifest: dict, *, run_id: str):
    """The registered identifiability run (guarded). Output root DERIVED internally; exact job kind enforced;
    verify_launch requires policy-data approval. The structured-computation params (ref/heldout/cov seeds, grid,
    rank points) come from the MANIFEST, not caller overrides."""
    job = "m3a-step4-ident-v2"
    v = verify_launch(manifest, run_id=run_id, job_kind=job)
    if not v["ok"]:
        return {"run_id": run_id, "status": "REFUSED", "problems": v["problems"]}
    _validate_run_id(run_id, job)
    from clinical_jepa.eval.oracle_realism_v2_ident_runner import execute_identifiability
    idb = manifest["identifiability"]
    from clinical_jepa.eval.oracle_realism_v2_ident_runner import ident_grid, interior_rank_points
    return execute_identifiability(manifest, run_id, _REPO_ROOT,
                                   base_sampler=registered_base_sampler(n=manifest["registered_n"]),
                                   cov_seeds=idb["cov_seeds"], ref_seed=idb["ref_seed"],
                                   heldout_seed=idb["heldout_seed"], grid=ident_grid(),
                                   rank_points=interior_rank_points(),
                                   cap_hours=manifest["cap"]["wall_clock_hours"], cap_gb=manifest["cap"]["ram_gb"],
                                   verify=lambda mm: verify_manifest(mm, require_git_head=True),
                                   gate_event=v["gate_event"])


def benchmark(*, source_profile: str = "mimic_scale_control", seed: int = 1000) -> dict:
    """§4 benchmark-BOUND forecast evidence: record the benchmark commit + hardware, the DETERMINISTIC
    event/cluster/adjacent-pair volume at N per source, and a MEASURED seconds-per-verifier-call. The per-job
    wall-clock forecast is then derived from volume x measured rate (a prose estimate is not a manifest field).
    Timing is evidence, routed alongside the manifest — NOT hashed into the trusted manifest."""
    import time
    from clinical_jepa.eval.oracle_realism_v2_verifier import sequence_route_checks, marginal_route_checks
    base = registered_base_sampler(n=REGISTERED_N)
    vol = {}
    for sp in SOURCE_PROFILES:
        s = base(sp, seed, "benchmark")
        vol[sp] = {"n": len(s), "events": int(sum(r.L_total for r in s)),
                   "clusters": int(sum(r.K for r in s)), "adjacent_pairs": int(sum(max(0, r.K - 1) for r in s))}
    ref = base(source_profile, seed, "benchmark"); cand = base(source_profile, seed + 1, "benchmark")
    t0 = time.monotonic(); sequence_route_checks(cand, ref); marginal_route_checks(cand, ref)
    secs = round(time.monotonic() - t0, 2)
    import platform
    return {"git_head": _git_head(), "environment_hash": env_hash(),   # py/numpy/scipy (Pi §7: not hardware id)
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "processor": platform.processor(), "python": platform.python_version()},
            "registered_n": REGISTERED_N, "volume_per_source": vol, "benchmark_source": source_profile,
            "seconds_per_verifier_call": secs, "workers": 1}


def env_hash() -> str:
    import numpy as np
    try:
        import scipy
        scv = scipy.__version__
    except Exception:
        scv = "n/a"
    return canonical_hash({"python": os.sys.version.split()[0], "numpy": np.__version__, "scipy": scv})
