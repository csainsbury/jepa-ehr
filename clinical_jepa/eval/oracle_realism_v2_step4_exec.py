"""Step-4 execution engine (Pi step-4 §2/§3) — fail-closed, checkpointed, cap-enforced, atomic.

Runs a step-4 job's replicates one at a time with: per-replicate checkpoint/resume; atomic result writing;
wall-clock (time.monotonic) + RSS enforcement at deterministic replicate boundaries; per-replicate
serialisation of every CheckResult status + VALUE + detail (the denominator/coarsening maps, so the
denominator-map hash can be produced); partial-state persistence; and a cap-exceed / resume-mismatch that
atomically yields PARTIAL / non-pass. Aggregates the per-check per-source verdict (§3).

Synthetic-only; no governed read, no candidate sampling. The full 25-seed / N=4000 run is launched only by the
reviewed job; this module is exercised at a tiny mechanical scale (checkpoint/resume/cap correctness).
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time

from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, PASS, FAIL, NOT_EVALUABLE,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX
from clinical_jepa.eval.oracle_realism_v2_battery import (
    component_ablation, null_control, boundary_control, structural_zero_control, source_swap_control,
    PRIMARY_FAIL_MIN, SPECIFICITY_MIN, _BOUNDARY_EXPECTED_NE,
)

CONTROLS = ("null", "boundary", "structural_zero", "source_swap")


def run_dir(out_base: str, run_id: str) -> str:
    return os.path.join(out_base, "state", "realism-v2", "step4", run_id)


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # linux ru_maxrss is KB


def _status_value_detail(cand, ref) -> dict:
    out = {}
    for k, v in {**sequence_route_checks(cand, ref), **marginal_route_checks(cand, ref)}.items():
        out[k] = {"status": v.status, "value": v.value, "detail": v.detail}
    return out


def _replicate_keys(components, sources, seeds, controls):
    keys = []
    for src in sources:
        for s in seeds:
            for comp in components:
                keys.append(("ablation", comp, src, s))
            for c in controls:
                keys.append(("control", c, src, s))
    return keys


def _run_replicate(kind, name, src, seed, *, base_sampler) -> dict:
    """One replicate -> a serialisable record (statuses + values + details). No pass/fail decision here."""
    if kind == "ablation":
        o = component_ablation(name, seed, base_sampler=base_sampler, source_profile=src)
        return {"kind": kind, "name": name, "source": src, "seed": seed,
                "A_status": {k: o.A_status[k] for k in o.A_status},
                "D_status": {k: o.D_status[k] for k in o.D_status},
                "A_fails_primary": o.A_fails_primary, "A_specificity": o.A_specificity,
                "A_specificity_ok": o.A_specificity_ok, "known_profile_repeatability": o.known_profile_repeatability}
    if name == "null":
        r = null_control(seed, base_sampler=base_sampler, source_profile=src)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "all_pass": r["all_pass"],
                "status": r["status"]}
    if name == "boundary":
        r = boundary_control(seed, n_each=500)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "ok": r["ok"], "status": r["status"]}
    if name == "structural_zero":
        r = structural_zero_control(seed, n_each=600)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "ok": r["ok"],
                "zeros_absent": r["zeros_absent"], "status": r["status"]}
    r = source_swap_control(seed, n_each=600)
    return {"kind": kind, "name": name, "source": src, "seed": seed,
            "fails_nondegenerate": r["fails_nondegenerate"], "status": r["status"]}


def execute(manifest, run_id, out_base, *, base_sampler, seeds, sources, components=None, controls=CONTROLS,
            cap_hours, cap_gb, clock=time.monotonic, verify=None):
    """Fail-closed execution with checkpoint/resume + cap enforcement. `verify` (callable manifest->{ok,...})
    is the runner's fail-closed manifest check; a failure REFUSES before any work. Returns a result dict and
    writes it atomically to run_dir/result.json (and per-replicate records + checkpoint.json)."""
    if verify is not None:
        v = verify(manifest)
        if not v.get("ok"):
            return {"run_id": run_id, "status": "REFUSED", "reason": "manifest verification failed",
                    "problems": v.get("problems")}
    components = list(components or V2_D_COMPONENT_MENU)
    rd = run_dir(out_base, run_id)
    os.makedirs(os.path.join(rd, "replicates"), exist_ok=True)
    ckpt = os.path.join(rd, "checkpoint.json")
    mhash = manifest["manifest_hash"]
    done, records = set(), {}
    if os.path.exists(ckpt):
        prev = json.load(open(ckpt))
        if prev.get("manifest_hash") != mhash:               # resume-mismatch => PARTIAL non-pass
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume manifest mismatch"}
            _atomic_write(os.path.join(rd, "result.json"), json.dumps(res))
            return res
        done = set(prev.get("done", []))
        for kid in done:
            records[kid] = json.load(open(os.path.join(rd, "replicates", kid + ".json")))
    t0 = clock()
    for key in _replicate_keys(components, sources, seeds, controls):
        kid = "|".join(map(str, key))
        if kid in done:
            continue
        if (clock() - t0) > cap_hours * 3600 or _rss_gb() > cap_gb:   # cap check AT the replicate boundary
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded",
                   "manifest_hash": mhash, "done": sorted(done), "cap": {"hours": cap_hours, "gb": cap_gb}}
            _atomic_write(os.path.join(rd, "result.json"), json.dumps(res))
            return res
        rec = _run_replicate(*key, base_sampler=base_sampler)
        _atomic_write(os.path.join(rd, "replicates", kid + ".json"), json.dumps(rec))
        records[kid] = rec
        done.add(kid)
        _atomic_write(ckpt, json.dumps({"manifest_hash": mhash, "done": sorted(done)}))
    verdict = aggregate(records, components, sources, seeds)
    denom_hash = _sha(json.dumps({k: records[k] for k in sorted(records)
                                  if records[k]["kind"] == "ablation"}, sort_keys=True))
    result = {"run_id": run_id, "status": "PASS" if verdict["conjunctive_pass"] else "FAIL",
              "manifest_hash": mhash, "reviewed_commit": manifest.get("reviewed_commit"),
              "verdict": verdict, "denominator_map_sha256": denom_hash,
              "runtime_secs": round(clock() - t0, 2), "n_replicates": len(records)}
    _atomic_write(os.path.join(rd, "result.json"), json.dumps(result))
    return result


def aggregate(records, components, sources, seeds) -> dict:
    """§3 verdict from persisted replicate records: per-check per-source rates + conjunction. NOT_EVALUABLE is
    non-passing. Boundary refusals must be EXACT (predeclared NE, else PASS); structural-zero absence + PASS;
    source-swap non-degenerate; source-wise conjunction."""
    n = len(seeds)
    per_component = {}
    for comp in components:
        primary = ABLATION_MATRIX[comp]["primary_fail"]
        allowed = set(ABLATION_MATRIX[comp]["allowed_sensitive"])
        per_source = {}
        for src in sources:
            recs = [records[f"ablation|{comp}|{src}|{s}"] for s in seeds if f"ablation|{comp}|{src}|{s}" in records]
            prim = {p: sum(1 for r in recs if r["A_status"].get(p) == FAIL) for p in primary}
            non_attr = sorted({k for r in recs for k in r["A_status"]} - set(primary) - allowed)
            spec = {k: sum(1 for r in recs if r["A_status"].get(k) == PASS) for k in non_attr}
            rep = sum(1 for r in recs if r["known_profile_repeatability"])
            ok = (all(prim[p] >= PRIMARY_FAIL_MIN for p in primary)
                  and all(spec[k] >= SPECIFICITY_MIN for k in non_attr)
                  and rep >= SPECIFICITY_MIN and len(recs) == n)
            per_source[src] = {"n": len(recs), "primary_fail_counts": prim, "specificity_counts": spec,
                               "repeatability_count": rep, "source_ok": ok}
        per_component[comp] = {"per_source": per_source,
                               "conjunctive_pass": all(per_source[s]["source_ok"] for s in sources)}
    ctrl = _aggregate_controls(records, sources, seeds)
    conj = all(per_component[c]["conjunctive_pass"] for c in components) and ctrl["controls_ok"]
    return {"per_component": per_component, "controls": ctrl, "conjunctive_pass": conj}


def _aggregate_controls(records, sources, seeds) -> dict:
    out = {}
    for src in sources:
        null_ok = sum(1 for s in seeds if records.get(f"control|null|{src}|{s}", {}).get("all_pass"))
        # boundary: EXACT expected-status map — predeclared NE else PASS — per seed
        bnd = 0
        for s in seeds:
            st = records.get(f"control|boundary|{src}|{s}", {}).get("status", {})
            if st and all((v == NOT_EVALUABLE) if k in _BOUNDARY_EXPECTED_NE else (v == PASS) for k, v in st.items()):
                bnd += 1
        sz = sum(1 for s in seeds if records.get(f"control|structural_zero|{src}|{s}", {}).get("ok"))
        sw = sum(1 for s in seeds if records.get(f"control|source_swap|{src}|{s}", {}).get("fails_nondegenerate"))
        out[src] = {"null_pass": null_ok, "boundary_exact": bnd, "structural_zero_ok": sz,
                    "source_swap_nondegenerate": sw, "n": len(seeds)}
    controls_ok = all(o["null_pass"] >= SPECIFICITY_MIN and o["boundary_exact"] >= SPECIFICITY_MIN
                      and o["structural_zero_ok"] >= SPECIFICITY_MIN and o["source_swap_nondegenerate"] >= PRIMARY_FAIL_MIN
                      for o in out.values()) if len(seeds) >= 25 else None
    return {"per_source": out, "controls_ok": controls_ok}
