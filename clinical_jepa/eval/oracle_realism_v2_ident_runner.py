"""Step-4 identifiability runner (Pi step-4 §5/§6) — fail-closed, INTRA-profile checkpoint/cap, atomic, evidence.

Executes the identifiability battery per NUISANCE profile (SCID/MIMIC/structural-zero — boundary-short is a
terminal support control, NOT a grid nuisance) as a STAGED, checkpointable state machine so the cap is honoured
and progress is preserved at FINE boundaries (Pi §5): every cov-seed, every grid reference vector, every held-out
recovery, and every rank Jacobian is its own cap/checkpoint unit. A cap-exceed mid-profile persists the partial
state and yields PARTIAL / non-pass; resume continues from the exact unit.

Each profile records FULL reviewable evidence (Pi §6): the ridge covariance + eigenvalues + whitening; the grid
ordering + reference vectors + any NOT_EVALUABLE locations; per-held-out truth/predicted/per-component error;
per-rank-point singular values + sigma_min/max; and the exact colliding pairs. The strict rule holds: any
NOT_EVALUABLE (a refused cov-seed / grid / held-out vector) makes the profile non-pass, never silently dropped.

Wall-clock is CUMULATIVE across resume; the cap is re-checked before declaring a verdict. The full grid runs
only under the reviewed `m3a-step4-ident-v2` job.
"""
from __future__ import annotations

import json
import os
import time
from itertools import product

import numpy as np

from clinical_jepa.eval.oracle_realism_v2_identifiability import (
    COMPONENTS, GRID, NUISANCE_PROFILES, covariance_from_rows, whitening_matrix, f_theta, jacobian,
    standardized_rank, nearest_grid_recovery, recovered_within_tol, collision_search,
)
from clinical_jepa.eval.oracle_realism_v2_step4_exec import (
    run_dir, _atomic_write, _sha, _rss_gb, _dumps, _environment,
)

_ZERO = {c: 0.0 for c in COMPONENTS}


def ident_grid():
    """The 3^3 grid of theta dicts (component -> strength)."""
    return [dict(zip(COMPONENTS, combo)) for combo in product(GRID, repeat=len(COMPONENTS))]


def interior_rank_points():
    """Interior low/mid/high uniform-theta points for the rank check (all components at the same grid value)."""
    return [{c: v for c in COMPONENTS} for v in GRID]


def _r8v(v):
    return [round(float(x), 8) for x in v]


def _r8m(M):
    return [[round(float(x), 8) for x in row] for row in np.asarray(M)]


def _theta_key(theta):
    return [round(float(theta[c]), 8) for c in COMPONENTS]


# --------------------------------------------------------------------------------------------------
# per-profile staged state machine (each step() advances ONE cap/checkpoint unit)
# --------------------------------------------------------------------------------------------------

def _init_state(prof, grid, rank_points, cov_seeds) -> dict:
    return {"profile": prof, "stage": "covariance", "next_index": 0,
            "cov_rows": [], "cov_ne_seeds": [],
            "Sigma": None, "eigenvalues": None, "whitening": None,
            "gvecs": [], "grid_ne": [],
            "heldout": [], "heldout_ne": [], "recovered": 0, "total": 0,
            "rank": [], "collisions": None,
            "grid_order": [_theta_key(gt) for gt in grid],
            "result": None}


def _finish_ne(state, reason, location=None) -> dict:
    state["stage"] = "done"
    state["result"] = _profile_record(state, "NOT_EVALUABLE", reason=reason, location=location)
    return state


def _profile_record(state, status, *, reason=None, location=None) -> dict:
    rec = {
        "profile": state["profile"], "status": status,
        "covariance": {"Sigma": state["Sigma"], "eigenvalues": state["eigenvalues"],
                       "whitening": state["whitening"], "ne_seeds": state["cov_ne_seeds"]},
        "grid": {"order": state["grid_order"], "vectors": state["gvecs"], "not_evaluable": state["grid_ne"]},
        "heldout": {"recovery": f"{state['recovered']}/{state['total']}", "recovered": state["recovered"],
                    "total": state["total"], "per_point": state["heldout"], "not_evaluable": state["heldout_ne"]},
        "rank": state["rank"],
        "collisions": {"n": len(state["collisions"] or []), "pairs": state["collisions"] or []},
        "grid_vector_hash": _sha(_dumps(state["gvecs"], sort_keys=True)),
    }
    if reason is not None:
        rec["reason"] = reason
    if location is not None:
        rec["location"] = location
    return rec


def _step_profile(state, base_sampler, prof, *, grid, rank_points, cov_seeds, ref_seed, heldout_seed) -> dict:
    """Advance ONE unit of work for `prof` and return the updated state. Never runs more than one cap boundary."""
    stage, i = state["stage"], state["next_index"]

    if stage == "covariance":
        if i < len(cov_seeds):
            row = f_theta(_ZERO, base_sampler=base_sampler, source_profile=prof, seed=cov_seeds[i])
            if row is None:                                  # strict: a refused NULL row => profile non-pass
                state["cov_ne_seeds"].append(cov_seeds[i])
                return _finish_ne(state, "strict null covariance: a cov-seed refused",
                                  location={"stage": "covariance", "cov_seed": cov_seeds[i]})
            state["cov_rows"].append(_r8v(row)); state["next_index"] = i + 1
            return state
        Sigma = covariance_from_rows(np.asarray(state["cov_rows"]))
        if Sigma is None:
            return _finish_ne(state, "null covariance under-determined (<2 rows)",
                              location={"stage": "covariance"})
        w, _V = np.linalg.eigh(Sigma)
        state["Sigma"] = _r8m(Sigma)
        state["eigenvalues"] = _r8v(w)
        state["whitening"] = _r8m(whitening_matrix(Sigma))
        state["stage"], state["next_index"] = "grid_refs", 0
        return state

    if stage == "grid_refs":
        if i < len(grid):
            v = f_theta(grid[i], base_sampler=base_sampler, source_profile=prof, seed=ref_seed)
            if v is None:
                state["grid_ne"].append({"index": i, "theta": _theta_key(grid[i])})
                return _finish_ne(state, "grid reference vector refused",
                                  location={"stage": "grid_refs", "index": i, "theta": _theta_key(grid[i])})
            state["gvecs"].append(_r8v(v)); state["next_index"] = i + 1
            return state
        state["stage"], state["next_index"] = "heldout", 0
        return state

    if stage == "heldout":
        if i < len(grid):
            hv = f_theta(grid[i], base_sampler=base_sampler, source_profile=prof, seed=heldout_seed)
            if hv is None:
                state["heldout_ne"].append({"index": i, "theta": _theta_key(grid[i])})
                return _finish_ne(state, "held-out vector refused",
                                  location={"stage": "heldout", "index": i, "theta": _theta_key(grid[i])})
            W = np.asarray(state["whitening"])
            gvecs = [np.asarray(g) for g in state["gvecs"]]
            rec_theta = nearest_grid_recovery(np.asarray(hv), grid, gvecs, W=W)
            ok = recovered_within_tol(grid[i], rec_theta)
            state["heldout"].append({
                "index": i, "truth": {c: round(float(grid[i][c]), 8) for c in COMPONENTS},
                "predicted": {c: round(float(rec_theta[c]), 8) for c in COMPONENTS},
                "error": {c: round(abs(float(grid[i][c]) - float(rec_theta[c])), 8) for c in COMPONENTS},
                "recovered": bool(ok)})
            state["recovered"] += int(ok); state["total"] += 1; state["next_index"] = i + 1
            return state
        state["stage"], state["next_index"] = "rank", 0
        return state

    if stage == "rank":
        if i < len(rank_points):
            J = jacobian(rank_points[i], base_sampler=base_sampler, source_profile=prof, seed=ref_seed)
            if J is None:                                    # a refused Jacobian is a rank FAILURE (non-pass)
                state["rank"].append({"theta": _theta_key(rank_points[i]), "singular_values": None,
                                      "sigma_min_over_max": None, "rank_ok": False, "reason": "jacobian refused"})
            else:
                sr = standardized_rank(J, np.asarray(state["Sigma"]))
                state["rank"].append({"theta": _theta_key(rank_points[i]),
                                      "singular_values": sr["singular_values"],
                                      "sigma_min_over_max": sr["sigma_min_over_max"], "rank_ok": sr["rank_ok"]})
            state["next_index"] = i + 1
            return state
        state["stage"], state["next_index"] = "collision", 0
        return state

    if stage == "collision":
        gvecs = [np.asarray(g) for g in state["gvecs"]]
        pairs = collision_search(grid, gvecs)
        state["collisions"] = [{"a": _theta_key(a), "b": _theta_key(b)} for a, b in pairs]
        state["stage"], state["next_index"] = "finalize", 0
        return state

    if stage == "finalize":
        rank_ok_all = bool(state["rank"]) and all(r["rank_ok"] for r in state["rank"])
        prof_pass = (rank_ok_all and state["total"] > 0 and state["recovered"] == state["total"]
                     and len(state["collisions"] or []) == 0)
        state["result"] = _profile_record(state, "PASS" if prof_pass else "FAIL")
        state["stage"] = "done"
        return state

    raise ValueError(f"unknown stage {stage!r}")


# --------------------------------------------------------------------------------------------------
# fail-closed driver: cumulative cap, intra-profile checkpoint/resume, atomic result, evidence hashes
# --------------------------------------------------------------------------------------------------

def _ident_completion_hashes(records, run_id, mhash, reviewed_commit, conj) -> dict:
    evidence = {p: records[p] for p in sorted(records)}                 # full per-profile evidence (Pi §6)
    evidence_sha = _sha(_dumps(evidence, sort_keys=True))
    summary = {p: {"status": records[p]["status"],
                   "recovery": records[p].get("heldout", {}).get("recovery"),
                   "n_collisions": records[p].get("collisions", {}).get("n")} for p in sorted(records)}
    result_core = {"run_id": run_id, "manifest_hash": mhash, "reviewed_commit": reviewed_commit,
                   "profile_conjunction": conj, "summary": summary}
    return {"result_sha256": _sha(_dumps(result_core, sort_keys=True)), "evidence_sha256": evidence_sha}


def _resume_profile_record(rd, prof, expected_sha):
    """Load a resumed FINAL profile record with integrity binding (Pi §4): bytes must hash to the recorded sha
    and the record's profile name must match. Returns (rec, None) or (None, why)."""
    path = os.path.join(rd, "profiles", prof + ".json")
    if not os.path.exists(path):
        return None, f"missing:{prof}"
    raw = open(path).read()
    if not expected_sha or _sha(raw) != expected_sha:
        return None, f"hash_mismatch:{prof}"
    rec = json.loads(raw)
    if str(rec.get("profile")) != prof:
        return None, f"metadata_mismatch:{prof}"
    return rec, None


def execute_identifiability(manifest, run_id, out_base, *, base_sampler, nuisance_profiles=NUISANCE_PROFILES,
                            cov_seeds=None, ref_seed=1000, heldout_seed=1024, grid=None, rank_points=None,
                            cap_hours, cap_gb, clock=time.monotonic, verify=None, gate_event=None):
    """Fail-closed identifiability execution with INTRA-profile checkpoint/resume + CUMULATIVE cap. `cov_seeds`
    seed the strict null covariance (default the manifest seeds). `gate_event` is the reviewed ARR approval,
    persisted into the result. Writes result.json + per-profile evidence records atomically; a cap-exceed at any
    unit boundary or a resume mismatch yields PARTIAL / non-pass. Resumed profile records AND the intra-profile
    progress state are integrity-bound to the checkpoint (content sha); any mismatch => PARTIAL / non-pass."""
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
    os.makedirs(os.path.join(rd, "progress"), exist_ok=True)
    ckpt = os.path.join(rd, "checkpoint.json")
    mhash = manifest["manifest_hash"]
    done, records, timings, cum_prev = set(), {}, {}, 0.0
    hashes_ck, progress_ck = {}, {}
    if os.path.exists(ckpt):
        prev = json.load(open(ckpt))
        if prev.get("manifest_hash") != mhash:
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume manifest mismatch"}
            _atomic_write(os.path.join(rd, "result.json"), _dumps(res)); return res
        done = set(prev.get("done", []))
        timings = dict(prev.get("timings", {}))
        hashes_ck = dict(prev.get("hashes", {}))             # per-profile final-record shas
        progress_ck = dict(prev.get("progress_hashes", {}))  # per-profile intra-profile state shas
        cum_prev = float(prev.get("cum_elapsed", 0.0))
        for p in done:
            rec, why = _resume_profile_record(rd, p, hashes_ck.get(p))
            if why is not None:
                res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume evidence integrity failure",
                       "detail": why, "manifest_hash": mhash}
                _atomic_write(os.path.join(rd, "result.json"), _dumps(res)); return res
            records[p] = rec

    def _write_ckpt():
        _atomic_write(ckpt, _dumps({"manifest_hash": mhash, "done": sorted(done), "timings": timings,
                                    "hashes": hashes_ck, "progress_hashes": progress_ck,
                                    "cum_elapsed": round(cum_prev + (clock() - t0), 2)}))

    t0 = clock()
    for prof in nuisance_profiles:
        if prof in done:
            continue
        sf = os.path.join(rd, "progress", prof + ".json")
        if os.path.exists(sf):                               # resume a partial profile — verify its integrity
            raw = open(sf).read()
            if _sha(raw) != progress_ck.get(prof):
                res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume progress integrity failure",
                       "detail": f"progress_hash_mismatch:{prof}", "manifest_hash": mhash}
                _atomic_write(os.path.join(rd, "result.json"), _dumps(res)); return res
            state = json.loads(raw)
        else:
            state = _init_state(prof, grid, rank_points, cov_seeds)
        while state["stage"] != "done":
            cum_now = cum_prev + (clock() - t0)
            if cum_now > cap_hours * 3600 or _rss_gb() > cap_gb:      # cumulative cap AT the unit boundary
                # persist the partial STATE only; the checkpoint's cum_elapsed already reflects COMPLETED units
                # (do NOT re-stamp it here — that would bank the cap-detection instant as elapsed work)
                _atomic_write(sf, _dumps(state))
                res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded", "manifest_hash": mhash,
                       "done": sorted(done), "cum_elapsed": round(cum_now, 2),
                       "in_progress": {"profile": prof, "stage": state["stage"], "next_index": state["next_index"]},
                       "cap": {"hours": cap_hours, "gb": cap_gb}}
                _atomic_write(os.path.join(rd, "result.json"), _dumps(res)); return res
            unit = f"{prof}|{state['stage']}|{state['next_index']}"
            t_u = clock()
            state = _step_profile(state, base_sampler, prof, grid=grid, rank_points=rank_points,
                                  cov_seeds=cov_seeds, ref_seed=ref_seed, heldout_seed=heldout_seed)
            timings[unit] = round(clock() - t_u, 4)
            ptext = _dumps(state)
            _atomic_write(sf, ptext); progress_ck[prof] = _sha(ptext); _write_ckpt()
        rec = state["result"]
        rtext = _dumps(rec)
        _atomic_write(os.path.join(rd, "profiles", prof + ".json"), rtext)
        hashes_ck[prof] = _sha(rtext)
        records[prof] = rec
        done.add(prof)
        progress_ck.pop(prof, None)
        try:
            os.remove(sf)                        # progress consumed; profiles/ holds only final evidence records
        except OSError:
            pass
        _write_ckpt()

    cum_total = round(cum_prev + (clock() - t0), 2)
    if cum_total > cap_hours * 3600 or _rss_gb() > cap_gb:            # Pi §5: re-check before declaring a verdict
        res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded before verdict", "manifest_hash": mhash,
               "done": sorted(done), "cum_elapsed": cum_total, "cap": {"hours": cap_hours, "gb": cap_gb}}
        _atomic_write(os.path.join(rd, "result.json"), _dumps(res)); return res
    conj = all(records[p]["status"] == "PASS" for p in nuisance_profiles)   # NOT_EVALUABLE/FAIL => non-pass
    hashes = _ident_completion_hashes(records, run_id, mhash, manifest.get("reviewed_commit"), conj)
    runtime = {"total_secs": cum_total, "per_unit_secs": timings, "n_profiles": len(records),
               "max_rss_gb": round(_rss_gb(), 3), "cap": {"hours": cap_hours, "gb": cap_gb}}
    env = _environment()
    hashes["runtime_json_sha256"] = _sha(_dumps(runtime, sort_keys=True))
    hashes["env_sha256"] = _sha(_dumps(env, sort_keys=True))
    _atomic_write(os.path.join(rd, "runtime.json"), _dumps(runtime))
    _atomic_write(os.path.join(rd, "environment.json"), _dumps(env))
    result = {"run_id": run_id, "status": "PASS" if conj else "FAIL", "manifest_hash": mhash,
              "reviewed_commit": manifest.get("reviewed_commit"), "gate_event": gate_event,
              "per_profile": records, "profile_conjunction": conj, "completion_hashes": hashes,
              "runtime_secs": cum_total}
    _atomic_write(os.path.join(rd, "result.json"), _dumps(result))
    return result
