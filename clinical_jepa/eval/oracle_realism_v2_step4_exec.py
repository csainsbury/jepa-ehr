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
from clinical_jepa.eval.oracle_realism_v2_verifier import PASS, FAIL, NOT_EVALUABLE
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX
from clinical_jepa.eval.oracle_realism_v2_battery import (
    component_ablation, null_control, boundary_control, structural_zero_control, source_swap_control,
    PRIMARY_FAIL_MIN, SPECIFICITY_MIN, _BOUNDARY_EXPECTED_NE,
)

CONTROLS = ("null", "boundary", "structural_zero", "source_swap")
# Control routing (Pi §4): `null` is a SOURCE-scoped same-profile self-consistency check (run per source).
# boundary / structural_zero / source_swap are MIMIC-DEFINED and identical across the source key, so they are
# GLOBAL controls — computed ONCE per seed and EXCLUDED from the per-source conjunction (routed as global gates).
SOURCE_CONTROLS = ("null",)
GLOBAL_CONTROLS = ("boundary", "structural_zero", "source_swap")
GLOBAL_SRC = "GLOBAL"


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


def _json_default(o):
    import numpy as np
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def _dumps(obj, *, sort_keys=False) -> str:
    return json.dumps(obj, sort_keys=sort_keys, default=_json_default)


def _environment() -> dict:
    """Run-time environment stamped as completion evidence (NOT part of the trusted manifest)."""
    import platform
    import sys
    import numpy as np
    try:
        import scipy
        scv = scipy.__version__
    except Exception:
        scv = "n/a"
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scv,
            "platform": platform.platform(), "machine": platform.machine(), "workers": 1}


def _replicate_keys(components, sources, seeds, controls):
    keys = []
    src_ctrls = [c for c in controls if c in SOURCE_CONTROLS]
    glob_ctrls = [c for c in controls if c in GLOBAL_CONTROLS]
    for src in sources:
        for s in seeds:
            for comp in components:
                keys.append(("ablation", comp, src, s))
            for c in src_ctrls:                          # source-scoped controls: per (source, seed)
                keys.append(("control", c, src, s))
    for s in seeds:                                      # global controls: once per seed (not per source)
        for c in glob_ctrls:
            keys.append(("control", c, GLOBAL_SRC, s))
    return keys


def _run_replicate(kind, name, src, seed, *, base_sampler) -> dict:
    """One replicate -> a serialisable record (statuses + values + details). No pass/fail decision here."""
    if kind == "ablation":
        o = component_ablation(name, seed, base_sampler=base_sampler, source_profile=src)
        return {"kind": kind, "name": name, "source": src, "seed": seed,
                "A_status": {k: o.A_status[k] for k in o.A_status},
                "D_status": {k: o.D_status[k] for k in o.D_status},
                "A_fails_primary": o.A_fails_primary, "A_specificity": o.A_specificity,
                "A_specificity_ok": o.A_specificity_ok, "known_profile_repeatability": o.known_profile_repeatability,
                "A_evidence": o.A_evidence, "D_evidence": o.D_evidence}   # full CheckResult evidence (Pi §3)
    if name == "null":
        r = null_control(seed, base_sampler=base_sampler, source_profile=src)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "all_pass": r["all_pass"],
                "status": r["status"], "evidence": r["evidence"]}
    if name == "boundary":
        r = boundary_control(seed, n_each=500)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "ok": r["ok"], "status": r["status"],
                "evidence": r["evidence"]}
    if name == "structural_zero":
        r = structural_zero_control(seed, n_each=600)
        return {"kind": kind, "name": name, "source": src, "seed": seed, "ok": r["ok"],
                "zeros_absent": r["zeros_absent"], "status": r["status"], "evidence": r["evidence"]}
    r = source_swap_control(seed, n_each=600)
    return {"kind": kind, "name": name, "source": src, "seed": seed,
            "fails_nondegenerate": r["fails_nondegenerate"], "status": r["status"], "evidence": r["evidence"]}


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
    done, records, timings, cum_prev = set(), {}, {}, 0.0
    if os.path.exists(ckpt):
        prev = json.load(open(ckpt))
        if prev.get("manifest_hash") != mhash:               # resume-mismatch => PARTIAL non-pass
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "resume manifest mismatch"}
            _atomic_write(os.path.join(rd, "result.json"), _dumps(res))
            return res
        done = set(prev.get("done", []))
        timings = dict(prev.get("timings", {}))              # per-replicate secs survive resume
        cum_prev = float(prev.get("cum_elapsed", 0.0))       # cumulative wall-clock survives resume (Pi §3/§5)
        for kid in done:
            records[kid] = json.load(open(os.path.join(rd, "replicates", kid + ".json")))
    t0 = clock()
    for key in _replicate_keys(components, sources, seeds, controls):
        kid = "|".join(map(str, key))
        if kid in done:
            continue
        cum_now = cum_prev + (clock() - t0)
        if cum_now > cap_hours * 3600 or _rss_gb() > cap_gb:   # CUMULATIVE cap check AT the replicate boundary
            res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded",
                   "manifest_hash": mhash, "done": sorted(done), "cum_elapsed": round(cum_now, 2),
                   "cap": {"hours": cap_hours, "gb": cap_gb}}
            _atomic_write(os.path.join(rd, "result.json"), _dumps(res))
            return res
        t_rep = clock()
        rec = _run_replicate(*key, base_sampler=base_sampler)
        timings[kid] = round(clock() - t_rep, 4)
        _atomic_write(os.path.join(rd, "replicates", kid + ".json"), _dumps(rec))
        records[kid] = rec
        done.add(kid)
        _atomic_write(ckpt, _dumps({"manifest_hash": mhash, "done": sorted(done), "timings": timings,
                                    "cum_elapsed": round(cum_prev + (clock() - t0), 2)}))
    cum_total = round(cum_prev + (clock() - t0), 2)
    if cum_total > cap_hours * 3600 or _rss_gb() > cap_gb:   # Pi §5: re-check the cap BEFORE declaring a verdict
        res = {"run_id": run_id, "status": "PARTIAL", "reason": "cap exceeded before verdict",
               "manifest_hash": mhash, "done": sorted(done), "cum_elapsed": cum_total,
               "cap": {"hours": cap_hours, "gb": cap_gb}}
        _atomic_write(os.path.join(rd, "result.json"), _dumps(res))
        return res
    verdict = aggregate(records, components, sources, seeds)
    status = "PASS" if verdict["conjunctive_pass"] else "FAIL"
    hashes = _completion_hashes(records, run_id, mhash, manifest.get("reviewed_commit"), verdict)
    runtime = {"total_secs": cum_total, "per_replicate_secs": timings, "n_replicates": len(records),
               "max_rss_gb": round(_rss_gb(), 3), "cap": {"hours": cap_hours, "gb": cap_gb}}
    env = _environment()
    hashes["runtime_json_sha256"] = _sha(_dumps(runtime, sort_keys=True))
    hashes["env_sha256"] = _sha(_dumps(env, sort_keys=True))
    _atomic_write(os.path.join(rd, "runtime.json"), _dumps(runtime))
    _atomic_write(os.path.join(rd, "environment.json"), _dumps(env))
    result = {"run_id": run_id, "status": status,
              "manifest_hash": mhash, "reviewed_commit": manifest.get("reviewed_commit"),
              "verdict": verdict, "completion_hashes": hashes,
              "runtime_secs": cum_total, "n_replicates": len(records)}
    _atomic_write(os.path.join(rd, "result.json"), _dumps(result))
    return result


# reviewable content persisted per replicate is the FULL CheckResult evidence (value + threshold + detail,
# i.e. the denominator/coarsening maps). The completion hashes bind that evidence and the decision separately:
#   evidence_sha256 — the statistical evidence (values + denominators), deterministic across resume
#   result_sha256   — the decision content (verdict + per-replicate status summary), deterministic across resume
#   runtime/env     — observed timing + environment (machine-specific, stamped at completion; not manifest-bound)
_STATUS_KEYS = ("kind", "name", "source", "seed", "A_status", "D_status", "A_fails_primary",
                "A_specificity_ok", "known_profile_repeatability", "all_pass", "ok", "zeros_absent",
                "fails_nondegenerate", "status")


def _completion_hashes(records, run_id, mhash, reviewed_commit, verdict) -> dict:
    evidence, status_summary = {}, {}
    for kid in sorted(records):
        rec = records[kid]
        if rec["kind"] == "ablation":
            evidence[kid] = {"A_evidence": rec.get("A_evidence", {}), "D_evidence": rec.get("D_evidence", {})}
        else:
            evidence[kid] = {"evidence": rec.get("evidence", {})}
        status_summary[kid] = {k: rec[k] for k in _STATUS_KEYS if k in rec}
    evidence_sha = _sha(_dumps(evidence, sort_keys=True))
    result_core = {"run_id": run_id, "manifest_hash": mhash, "reviewed_commit": reviewed_commit,
                   "verdict": verdict, "replicate_status": status_summary}
    result_sha = _sha(_dumps(result_core, sort_keys=True))
    return {"result_sha256": result_sha, "evidence_sha256": evidence_sha}


def _per_check_rates(status_maps) -> dict:
    """Per-check PASS/FAIL/NE counts over a list of status maps (Pi §4 known-profile diagnostic)."""
    keys = sorted({k for m in status_maps for k in m})
    return {k: {"PASS": sum(1 for m in status_maps if m.get(k) == PASS),
                "FAIL": sum(1 for m in status_maps if m.get(k) == FAIL),
                "NE": sum(1 for m in status_maps if m.get(k) == NOT_EVALUABLE)} for k in keys}


def aggregate(records, components, sources, seeds) -> dict:
    """§3/§4 verdict from persisted replicate records: per-check per-source ablation rates + a source-wise
    conjunction, PLUS routed controls. NOT_EVALUABLE is non-passing. `null` is a SOURCE-scoped self-consistency
    control folded into the per-source conjunction; boundary / structural_zero / source_swap are GLOBAL controls
    (computed once, gated globally). Per-check known-profile PASS/FAIL/NE diagnostics are reported for review."""
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
            # source-scoped null control folds into this source's conjunction
            null_recs = [records[f"control|null|{src}|{s}"] for s in seeds if f"control|null|{src}|{s}" in records]
            null_ok = sum(1 for r in null_recs if r.get("all_pass"))
            ok = (all(prim[p] >= PRIMARY_FAIL_MIN for p in primary)
                  and all(spec[k] >= SPECIFICITY_MIN for k in non_attr)
                  and rep >= SPECIFICITY_MIN and null_ok >= SPECIFICITY_MIN and len(recs) == n)
            per_source[src] = {"n": len(recs), "primary_fail_counts": prim, "specificity_counts": spec,
                               "repeatability_count": rep, "null_pass": null_ok, "source_ok": ok,
                               "known_profile_rates": _per_check_rates([r["A_status"] for r in recs])}
        per_component[comp] = {"per_source": per_source,
                               "conjunctive_pass": all(per_source[s]["source_ok"] for s in sources)}
    ctrl = _aggregate_controls(records, sources, seeds)
    conj = all(per_component[c]["conjunctive_pass"] for c in components) and bool(ctrl["controls_ok"])
    return {"per_component": per_component, "controls": ctrl, "conjunctive_pass": conj}


def _boundary_exact(rec) -> bool:
    """Boundary control PASS: EXACT expected-status map — predeclared NE else PASS (no broad 'anything but FAIL')."""
    st = rec.get("status", {})
    return bool(st) and all((v == NOT_EVALUABLE) if k in _BOUNDARY_EXPECTED_NE else (v == PASS)
                            for k, v in st.items())


def _aggregate_controls(records, sources, seeds) -> dict:
    """§4 routed-control aggregation. Source-scoped `null` per source (also folded into per-source conjunction);
    global boundary / structural_zero / source_swap computed once and gated globally (NOT per source)."""
    n = len(seeds)
    per_source = {}
    for src in sources:
        null_recs = [records[f"control|null|{src}|{s}"] for s in seeds if f"control|null|{src}|{s}" in records]
        per_source[src] = {"null_pass": sum(1 for r in null_recs if r.get("all_pass")), "n": len(null_recs),
                           "known_profile_rates": _per_check_rates([r.get("status", {}) for r in null_recs])}
    bnd = sum(1 for s in seeds if _boundary_exact(records.get(f"control|boundary|{GLOBAL_SRC}|{s}", {})))
    sz = sum(1 for s in seeds if records.get(f"control|structural_zero|{GLOBAL_SRC}|{s}", {}).get("ok"))
    sw = sum(1 for s in seeds if records.get(f"control|source_swap|{GLOBAL_SRC}|{s}", {}).get("fails_nondegenerate"))
    glob = {"boundary_exact": bnd, "structural_zero_ok": sz, "source_swap_nondegenerate": sw, "n": n}
    controls_ok = None
    if n >= 25:
        source_ok = all(o["null_pass"] >= SPECIFICITY_MIN for o in per_source.values())
        global_ok = (bnd >= SPECIFICITY_MIN and sz >= SPECIFICITY_MIN and sw >= PRIMARY_FAIL_MIN)
        controls_ok = source_ok and global_ok
    return {"per_source": per_source, "global": glob, "controls_ok": controls_ok}
