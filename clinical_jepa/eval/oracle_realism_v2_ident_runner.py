"""Step-4 identifiability runner (Pi step-4 §5) — fail-closed, checkpointed, cap-enforced, atomic.

Executes the identifiability battery per NUISANCE profile (SCID/MIMIC/structural-zero — boundary-short is a
terminal support control, NOT a grid nuisance): a STRICT 25-row null covariance (any NOT_EVALUABLE row => the
profile is non-pass); whitened grid vectors over the 3^4 grid; held-out recovery (nearest-grid under the
whitening, recovered iff every component within recover_tol); a standardized-Jacobian rank check at the interior
rank points; and a fixed-tolerance collision search. Per-profile verdict conjuncts to the job verdict. Wall-clock
+ RSS cap at profile boundaries; per-profile checkpoint/resume; atomic result. A cap-exceed / resume mismatch
yields PARTIAL / non-pass. The full grid runs only under the reviewed `m3a-step4-ident-v1` job.
"""
from __future__ import annotations

import json
import os
import time
from itertools import product

from clinical_jepa.eval.oracle_realism_v2_identifiability import (
    COMPONENTS, GRID, NUISANCE_PROFILES, null_covariance, whitening_matrix, f_theta, jacobian,
    standardized_rank, nearest_grid_recovery, recovered_within_tol, collision_search,
)
from clinical_jepa.eval.oracle_realism_v2_step4_exec import run_dir, _atomic_write, _sha, _rss_gb


def ident_grid():
    """The 3^4 grid of theta dicts (component -> strength)."""
    return [dict(zip(COMPONENTS, combo)) for combo in product(GRID, repeat=len(COMPONENTS))]


def interior_rank_points():
    """Interior low/mid/high uniform-theta points for the rank check (all components at the same grid value)."""
    return [{c: v for c in COMPONENTS} for v in GRID]


def _profile_result(base_sampler, prof, *, cov_seeds, ref_seed, heldout_seed, grid, rank_points) -> dict:
    """Evaluate one nuisance profile end-to-end. Returns a serialisable record (status + evidence)."""
    Sigma = null_covariance(base_sampler, cov_seeds, source_profile=prof)
    if Sigma is None:
        return {"profile": prof, "status": "NOT_EVALUABLE", "reason": "strict null covariance (a seed refused)"}
    W = whitening_matrix(Sigma)
    gvecs = []
    for gt in grid:
        v = f_theta(gt, base_sampler=base_sampler, source_profile=prof, seed=ref_seed)
        if v is None:
            return {"profile": prof, "status": "NOT_EVALUABLE", "reason": "grid reference vector refused"}
        gvecs.append(v)
    recovered = total = 0
    for gt in grid:
        hv = f_theta(gt, base_sampler=base_sampler, source_profile=prof, seed=heldout_seed)
        if hv is None:
            return {"profile": prof, "status": "NOT_EVALUABLE", "reason": "held-out vector refused"}
        rec = nearest_grid_recovery(hv, grid, gvecs, W=W)
        recovered += int(recovered_within_tol(gt, rec)); total += 1
    rank_ok = True
    for gt in rank_points:
        J = jacobian(gt, base_sampler=base_sampler, source_profile=prof, seed=ref_seed)
        if J is None or not standardized_rank(J, Sigma)["rank_ok"]:
            rank_ok = False
    collisions = collision_search(grid, gvecs)
    prof_pass = rank_ok and total > 0 and recovered == total and len(collisions) == 0
    return {"profile": prof, "status": "PASS" if prof_pass else "FAIL", "rank_ok": rank_ok,
            "recovery": f"{recovered}/{total}", "n_collisions": len(collisions),
            "grid_vector_hash": _sha(json.dumps([[round(float(x), 8) for x in v] for v in gvecs]))}


def execute_identifiability(manifest, run_id, out_base, *, base_sampler, nuisance_profiles=NUISANCE_PROFILES,
                            cov_seeds=None, ref_seed=1000, heldout_seed=1024, grid=None, rank_points=None,
                            cap_hours, cap_gb, clock=time.monotonic, verify=None):
    """Fail-closed identifiability execution with per-profile checkpoint/resume + cap. `cov_seeds` are the seeds
    for the STRICT null covariance (default the manifest seeds). Writes result.json + per-profile records
    atomically; a cap-exceed at a profile boundary or a resume mismatch yields PARTIAL / non-pass."""
    if verify is not None:
        v = verify(manifest)
        if not v.get("ok"):
            return {"run_id": run_id, "status": "REFUSED", "reason": "manifest verification failed",
                    "problems": v.get("problems")}
    cov_seeds = list(cov_seeds if cov_seeds is not None else manifest["seeds"])
    grid = grid if grid is not None else ident_grid()
    rank_points = rank_points if rank_points is not None else interior_rank_points()
    rd = run_dir(out_base, run_id)
    os.makedirs(os.path.join(rd, "profiles"), exist_ok=True)
    ckpt = os.path.join(rd, "checkpoint.json")
    mhash = manifest["manifest_hash"]
    done, records = set(), {}
    if os.path.exists(ckpt):
        prev = json.load(open(ckpt))
        if prev.get("manifest_hash") != mhash:
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume manifest mismatch"}
            _atomic_write(os.path.join(rd, "result.json"), json.dumps(res)); return res
        done = set(prev.get("done", []))
        for p in done:
            records[p] = json.load(open(os.path.join(rd, "profiles", p + ".json")))
    t0 = clock()
    for prof in nuisance_profiles:
        if prof in done:
            continue
        if (clock() - t0) > cap_hours * 3600 or _rss_gb() > cap_gb:      # cap AT the profile boundary
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded", "manifest_hash": mhash,
                   "done": sorted(done), "cap": {"hours": cap_hours, "gb": cap_gb}}
            _atomic_write(os.path.join(rd, "result.json"), json.dumps(res)); return res
        rec = _profile_result(base_sampler, prof, cov_seeds=cov_seeds, ref_seed=ref_seed,
                              heldout_seed=heldout_seed, grid=grid, rank_points=rank_points)
        _atomic_write(os.path.join(rd, "profiles", prof + ".json"), json.dumps(rec))
        records[prof] = rec
        done.add(prof)
        _atomic_write(ckpt, json.dumps({"manifest_hash": mhash, "done": sorted(done)}))
    conj = all(records[p]["status"] == "PASS" for p in nuisance_profiles)    # NOT_EVALUABLE/FAIL => non-pass
    result = {"run_id": run_id, "status": "PASS" if conj else "FAIL", "manifest_hash": mhash,
              "reviewed_commit": manifest.get("reviewed_commit"), "per_profile": records,
              "profile_conjunction": conj, "runtime_secs": round(clock() - t0, 2)}
    _atomic_write(os.path.join(rd, "result.json"), json.dumps(result))
    return result
