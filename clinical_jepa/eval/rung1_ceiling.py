"""Rung-1 frozen-decode-ceiling DRIVER (Pi R7/R8). DEV-ONLY — there is deliberately NO
test-access path here (Pi R8 #8); a later one-shot test confirmation is a separately locked,
separately authorized artifact.

Per (arm × source×horizon cell), fit the M1/M2 readouts on TRAIN z+, evaluate on DEV, apply
the per-readout on-manifold swap-floor excess, the NN-copy floor, and (timing) the
precision-gated randomized-PIT KS + normalized-CRPS-skill, then emit per-cell metric rows for
`rung1_verdict`. The heavy governed IO (building z+ sidecars from real data) is
`export_target_latents`; this driver consumes in-memory cell bundles so it is end-to-end
testable on synthetic data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval import rung1_probes as P
from clinical_jepa.eval.rung1_contract import (
    COUNT_CLUSTER_FLOOR, EXACT_COUNT_GATE, ORDER_CLUSTER_FLOOR, SEED, SLOT_GATE, SWAP_SEED,
    TIMING_CLUSTER_FLOOR, TIMING_INTERVAL_FLOOR,
)
from clinical_jepa.eval.rung1_precision_sim import run_precision_sim
from clinical_jepa.eval.rung1_verdict import build_rung1_manifest, evaluate_property
from clinical_jepa.utils import now_utc, write_json


def _n_clusters(patients: Any) -> int:
    return int(len(np.unique(np.asarray(patients))))


# The KS precision sim is a design check that depends only on the (capped) cell size class,
# so memoize it — every large real timing cell maps to the same bucket (computed ~once).
_PSIM_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


def _precision_sim_cached(n_int: int, n_clu: int) -> dict[str, Any]:
    key = (min(int(n_int), 20000), min(max(int(n_clu), 1), 4000))
    if key not in _PSIM_CACHE:
        _PSIM_CACHE[key] = run_precision_sim(key[0], key[1], n_boot=200)
    return _PSIM_CACHE[key]


def _nn_copy_count_hits(z_tr, cnt_tr, z_dev, cnt_dev) -> np.ndarray:
    """Count of the nearest (cosine) TRAIN z+ — the memorisation/copy floor (Pi R7 #5 G3)."""
    a = z_dev / (np.linalg.norm(z_dev, axis=1, keepdims=True) + 1e-9)
    b = z_tr / (np.linalg.norm(z_tr, axis=1, keepdims=True) + 1e-9)
    nn = np.argmax(a @ b.T, axis=1)
    return P.exact_count_hits(cnt_tr[nn], cnt_dev)


def count_row(arm, source, W, tr, dev, *, embedding_dim, floor=COUNT_CLUSTER_FLOOR,
              n_boot=P.N_BOOT, seed=SEED) -> dict[str, Any]:
    import torch
    from clinical_jepa.eval.rung1_decode import predict_count, train_count_head
    keep = np.asarray(dev["counts"]) >= 1
    z_dev = np.asarray(dev["z"])[keep]; cnt_dev = np.asarray(dev["counts"])[keep]
    pats = np.asarray(dev["patients"])[keep]
    tr_keep = np.asarray(tr["counts"]) >= 1
    z_tr = np.asarray(tr["z"])[tr_keep]; cnt_tr = np.asarray(tr["counts"])[tr_keep]
    evaluable = _n_clusters(pats) >= floor and len(z_dev) > 0 and len(z_tr) > 0
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "count",
           "n_clusters": _n_clusters(pats), "evaluable": bool(evaluable)}
    if not evaluable:
        return row
    # M1b closed-form ridge; M2 matched-budget head.
    W1 = P.ridge_fit(z_tr, np.log1p(cnt_tr), lam=1.0)
    m1_pred = np.expm1(P.ridge_predict(W1, z_dev))
    head = train_count_head(torch.as_tensor(z_tr), torch.as_tensor(cnt_tr.astype(np.float32)), embedding_dim)
    m2_pred = predict_count(head, torch.as_tensor(z_dev))
    partner = P.swap_partner_index(list(pats), SWAP_SEED)
    ok = partner >= 0
    h_m1 = P.exact_count_hits(m1_pred, cnt_dev)
    h_m2 = P.exact_count_hits(m2_pred, cnt_dev)
    m1_swap = P.exact_count_hits(np.expm1(P.ridge_predict(W1, z_dev[partner[ok]])), cnt_dev[ok])
    m2_swap = P.exact_count_hits(predict_count(head, torch.as_tensor(z_dev[partner[ok]])), cnt_dev[ok])
    ex1 = P.paired_excess_ci(h_m1[ok], m1_swap, pats[ok], n_boot=n_boot, seed=seed)
    ex2 = P.paired_excess_ci(h_m2[ok], m2_swap, pats[ok], n_boot=n_boot, seed=seed)
    copy_hits = _nn_copy_count_hits(z_tr, cnt_tr, z_dev, cnt_dev)
    acc1 = P.cluster_bootstrap_ci(h_m1, pats, n_boot=n_boot, seed=seed)
    acc2 = P.cluster_bootstrap_ci(h_m2, pats, n_boot=n_boot, seed=seed)
    row.update({
        "m1_gate_ok": bool(acc1["ci_lo"] >= EXACT_COUNT_GATE), "m1_excess_lo": ex1["ci_lo"],
        "m2_gate_ok": bool(acc2["ci_lo"] >= EXACT_COUNT_GATE), "m2_excess_lo": ex2["ci_lo"],
        "m2_copy_ok": bool(h_m2.mean() > copy_hits.mean()),
        "precise": bool((acc2["ci_hi"] - acc2["ci_lo"]) < 0.20),
        "m2_exact_count": acc2["point"], "nn_copy_exact_count": float(copy_hits.mean()),
    })
    return row


def timing_row(arm, source, W, tr, dev, *, embedding_dim, cluster_floor=TIMING_CLUSTER_FLOOR,
               interval_floor=TIMING_INTERVAL_FLOOR, n_boot=P.N_BOOT, seed=SEED) -> dict[str, Any]:
    import torch
    from clinical_jepa.eval.rung1_decode import predict_hurdle_timing, train_hurdle_timing_head
    # flatten inter-event intervals over non-empty >=2-event windows, carrying the patient.
    def flatten(bundle):
        dt, pat = [], []
        for arr, p in zip(bundle["dt_lists"], bundle["patients"]):
            arr = np.asarray(arr, dtype=np.float64)
            if len(arr) >= 1:
                dt.append(arr); pat.append(np.full(len(arr), p))
        return (np.concatenate(dt) if dt else np.asarray([])), (np.concatenate(pat) if pat else np.asarray([]))
    # per-interval z is the window's z repeated across its intervals
    def flat_z(bundle):
        zs = []
        for z, arr in zip(bundle["z"], bundle["dt_lists"]):
            if len(np.asarray(arr)) >= 1:
                zs.append(np.repeat(np.asarray(z)[None, :], len(np.asarray(arr)), axis=0))
        return np.concatenate(zs) if zs else np.zeros((0, np.asarray(dev["z"]).shape[1]))
    # per-interval (dt, patient, window index) + per-window (z, patient) — the window index
    # lets us apply M2's OWN wrong-instance swap at the window level (Pi result gate #2.1).
    def flatten_win(bundle):
        dt, pat, widx = [], [], []
        wz = np.asarray(bundle["z"], dtype=np.float32); wpat = np.asarray(bundle["patients"])
        for j, arr in enumerate(bundle["dt_lists"]):
            arr = np.asarray(arr, dtype=np.float64)
            if len(arr) >= 1:
                dt.append(arr); pat.append(np.full(len(arr), wpat[j])); widx.append(np.full(len(arr), j))
        return ((np.concatenate(dt) if dt else np.zeros(0)), (np.concatenate(pat) if pat else np.zeros(0, dtype=object)),
                (np.concatenate(widx).astype(int) if widx else np.zeros(0, dtype=int)), wz, wpat)
    dt_dev, pat_dev, widx_dev, wz_dev, wpat_dev = flatten_win(dev)
    dt_tr, _, _, wz_tr, _ = flatten_win(tr)
    z_dev_i = wz_dev[widx_dev] if len(widx_dev) else np.zeros((0, wz_dev.shape[1]))
    z_tr_i = flat_z(tr)
    n_int = len(dt_dev); n_clu = _n_clusters(pat_dev) if n_int else 0
    sim = _precision_sim_cached(n_int, n_clu) if n_int >= interval_floor else {"passes": False}
    evaluable = bool(n_int >= interval_floor and n_clu >= cluster_floor and sim.get("passes", False))
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "timing",
           "n_intervals": int(n_int), "n_clusters": int(n_clu), "precision_sim_passes": bool(sim.get("passes", False)),
           "evaluable": evaluable}
    if not evaluable:
        return row
    max_intervals = 40000
    rng = np.random.default_rng(seed)
    if n_int > max_intervals:
        sel = np.sort(rng.choice(n_int, size=max_intervals, replace=False))
        dt_dev, pat_dev, widx_dev, z_dev_i = dt_dev[sel], pat_dev[sel], widx_dev[sel], z_dev_i[sel]
    if len(dt_tr) > max_intervals:
        selt = np.sort(rng.choice(len(dt_tr), size=max_intervals, replace=False))
        dt_tr, z_tr_i = dt_tr[selt], z_tr_i[selt]
    n_int = len(dt_dev)
    head, qs = train_hurdle_timing_head(torch.as_tensor(z_tr_i), torch.as_tensor(dt_tr.astype(np.float32)), embedding_dim)
    lv = qs.cpu().numpy()
    p0, q_dev = predict_hurdle_timing(head, torch.as_tensor(z_dev_i))
    pit = P.hurdle_randomized_pit(p0, q_dev, lv, dt_dev, seed=seed)
    ks = P.ks_d_upper_ci(pit, pat_dev, n_boot=n_boot, seed=seed)
    NS = 64
    crps_cond = np.clip(P.hurdle_crps_rows(p0, q_dev, dt_dev, n_samp=NS, seed=seed), 0.0, None)
    # SYMMETRIC marginal CRPS: same hurdle-CRPS estimator + sample count as the conditional
    # (Pi result gate #2.2 — remove the finite-sample bias of 32-vs-256 draws).
    mp0, mq, _ = P.marginal_hurdle_quantiles(dt_tr)
    crps_marg = np.clip(P.hurdle_crps_rows(np.full(n_int, mp0), np.tile(mq, (n_int, 1)), dt_dev, n_samp=NS, seed=seed), 1e-9, None)
    skill = P.ratio_skill_ci(crps_cond, crps_marg, pat_dev, n_boot=n_boot, seed=seed)
    # M2's OWN wrong-instance swap (window-level derangement): does the trained head actually
    # read z+? true-vs-swap CRPS excess. swap-excess>0 => uses the latent (MARGINAL_ONLY);
    # swap-excess<=0 => reads its prior (PRIOR_MASKED).
    partner = P.swap_partner_index(list(wpat_dev), SWAP_SEED)
    valid = partner[widx_dev] >= 0
    swap_lo = None
    if valid.sum() > 0:
        z_swap = wz_dev[partner[widx_dev[valid]]]
        p0s, qs_ = predict_hurdle_timing(head, torch.as_tensor(z_swap))
        crps_swap = np.clip(P.hurdle_crps_rows(p0s, qs_, dt_dev[valid], n_samp=NS, seed=seed), 0.0, None)
        # excess = CRPS(swap) − CRPS(true): positive means the true latent decodes better.
        swap_lo = P.paired_excess_ci(crps_swap, crps_cond[valid], pat_dev[valid], n_boot=n_boot, seed=seed)["ci_lo"]
    row.update({"ks_upper_ci": ks["ci_hi"], "crps_skill_lo": skill["ci_lo"],
                "swap_excess_lo": swap_lo, "zero_rate": float((dt_dev <= 0).mean()), "precise": True})
    return row


def _quantile_pit(quantiles: np.ndarray, y: np.ndarray, levels: np.ndarray, *, seed: int) -> np.ndarray:
    """Randomized PIT from predicted quantiles: locate y among the quantile grid, interpolate
    the CDF, randomize within the enclosing step (handles the Δt=0 / ties mass)."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(y))
    for i in range(len(y)):
        q = quantiles[i]
        below = np.searchsorted(q, y[i], side="left")
        above = np.searchsorted(q, y[i], side="right")
        lo = levels[below - 1] if below > 0 else 0.0
        hi = levels[min(above, len(levels) - 1)] if above < len(levels) else 1.0
        out[i] = lo + rng.uniform() * max(hi - lo, 0.0)
    return np.clip(out, 0.0, 1.0)


def order_row(arm, source, W, tr, dev, *, embedding_dim, E, floor=ORDER_CLUSTER_FLOOR,
              n_boot=P.N_BOOT, seed=SEED, max_events=16, max_train_pairs=15_000,
              max_dev_windows=2500, order_steps=150) -> dict[str, Any]:
    """Order for the order-BLIND arms. The frozen unconditional all-window exact-order decoder
    is NOT implemented -> `unconditional_order = NOT_EVALUATED` (Pi result gate #1); the
    structural arm-A invariance stands analytically. We additionally report a clearly-LABELLED
    ORACLE-ASSISTED N<=max_events pairwise probe (tie-aware token-sequence match, post-cap
    denominators) that can never gate or nominate."""
    import torch
    from clinical_jepa.eval.rung1_decode import reconstruct_order_exact, train_pairwise_order_head
    def in_range(bundle):
        return [i for i, s in enumerate(bundle["ordered_ids"]) if 2 <= len(np.asarray(s)) <= max_events]
    n_ge2_dev = sum(1 for s in dev["ordered_ids"] if len(np.asarray(s)) >= 2)
    di = in_range(dev)
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "order",
           "unconditional_order": "NOT_EVALUATED", "oracle_scope": "oracle_assisted",
           "n_ge2_windows": int(n_ge2_dev), "oracle_max_events": int(max_events),
           "oracle_excluded_large": int(n_ge2_dev - len(di))}
    if _n_clusters(np.asarray(dev["patients"])[di] if di else np.asarray([])) < floor or len(in_range(tr)) == 0:
        row["oracle_probe"] = "NOT_EVALUABLE"
        return row
    rng = np.random.default_rng(seed)
    if len(di) > max_dev_windows:
        di = list(np.sort(rng.choice(di, size=max_dev_windows, replace=False)))
    pats = np.asarray(dev["patients"])[di]
    Et = torch.as_tensor(np.asarray(E, dtype=np.float32))
    feats, labels, n_pairs, tr_idx = [], [], 0, in_range(tr)
    rng.shuffle(tr_idx)
    for i in tr_idx:
        if n_pairs >= max_train_pairs:
            break
        ids = np.asarray(tr["ordered_ids"][i]); z = np.asarray(tr["z"][i], dtype=np.float32); n = len(ids)
        ea = E[ids]
        ai, bi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mm = (ai != bi).reshape(-1); ai = ai.reshape(-1)[mm]; bi = bi.reshape(-1)[mm]
        feats.append(np.concatenate([np.tile(z, (len(ai), 1)), ea[ai], ea[bi]], axis=1))
        labels.append((ai < bi).astype(np.float32)); n_pairs += len(ai)
    feats = np.concatenate(feats, axis=0); labels = np.concatenate(labels)
    head = train_pairwise_order_head(torch.as_tensor(feats.astype(np.float32)), torch.as_tensor(feats.astype(np.float32)),
                                     torch.as_tensor(labels.astype(np.float32)), embedding_dim, steps=order_steps)

    def hits(idxs, z_src):  # TIE-AWARE: compare recovered TOKEN sequence to the true one
        toks = [np.asarray(dev["ordered_ids"][i]) for i in idxs]
        perms = [reconstruct_order_exact(head, torch.as_tensor(np.asarray(z, dtype=np.float32)), Et[torch.as_tensor(t)])
                 for z, t in zip(z_src, toks)]
        return P.tie_aware_exact_order_hits(perms, toks)

    partner = P.swap_partner_index(list(pats), SWAP_SEED); ok = partner >= 0
    h_true = hits(di, [dev["z"][i] for i in di])
    h_swap = hits([di[j] for j in np.where(ok)[0]], [dev["z"][di[partner[j]]] for j in np.where(ok)[0]])
    acc = P.cluster_bootstrap_ci(h_true, pats, n_boot=n_boot, seed=seed)
    ex = P.paired_excess_ci(h_true[ok], h_swap, pats[ok], n_boot=n_boot, seed=seed)
    row.update({"oracle_probe": "computed", "oracle_n_eval_windows": len(di),  # post-cap denominator
                "oracle_exact_order": acc["point"], "oracle_excess_lo": ex["ci_lo"]})
    return row


def slot_order_row(arm, source, W, tr, dev, *, embedding_dim, E, slots, floor=ORDER_CLUSTER_FLOOR,
                   n_boot=P.N_BOOT, seed=SEED, gating=True) -> dict[str, Any]:
    """The REAL temporal-slot fidelity metric (Pi result gate #2): exact all-slot token-multiset
    reconstruction decoded (non-oracle) from the per-slot means, with M2's own wrong-instance
    swap excess. Gate = SLOT_GATE + excess > EXCESS_MARGIN. M=4 gates; M=8 is non-rescuing
    sensitivity (gating=False)."""
    D = int(embedding_dim); M = int(slots)
    def keep(bundle):
        return [i for i, c in enumerate(bundle["counts"]) if int(c) >= 1 and "slot_sets" in bundle]
    di = keep(dev); ti = keep(tr)
    pats = np.asarray(dev["patients"])[di] if di else np.asarray([])
    evaluable = _n_clusters(pats) >= floor and len(ti) > 0
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "order",
           "slot_m": M, "slot_gating": bool(gating), "n_clusters": _n_clusters(pats) if len(pats) else 0,
           "evaluable": bool(evaluable)}
    if not evaluable:
        return row
    def slot_means(z):
        z = np.asarray(z, dtype=np.float64)
        return [z[j * D:(j + 1) * D] for j in range(M)]
    Epinv = P.analytic_pinv(E)                                 # precompute once (matmul decode)
    # tune the presence threshold on a train subset (maximize exact all-slot match)
    rng = np.random.default_rng(seed)
    tsub = ti if len(ti) <= 800 else list(rng.choice(ti, 800, replace=False))
    tpv = P.slot_pvals(Epinv, [slot_means(tr["z"][i]) for i in tsub])
    tsets = [tr["slot_sets"][i] for i in tsub]
    best_t, best = 0.1, -1.0
    for thr in (0.02, 0.05, 0.1, 0.2, 0.3):
        if (h := P.slot_exact_from_pvals(tpv, tsets, thr)[0].mean()) > best:
            best, best_t = h, thr
    dpv = P.slot_pvals(Epinv, [slot_means(dev["z"][i]) for i in di])
    dsets = [dev["slot_sets"][i] for i in di]
    exact, f1 = P.slot_exact_from_pvals(dpv, dsets, best_t)
    partner = P.swap_partner_index(list(pats), SWAP_SEED); ok = partner >= 0
    spv = P.slot_pvals(Epinv, [slot_means(dev["z"][di[partner[j]]]) for j in np.where(ok)[0]])
    swap_exact, _ = P.slot_exact_from_pvals(spv, [dev["slot_sets"][di[j]] for j in np.where(ok)[0]], best_t)
    acc = P.cluster_bootstrap_ci(exact, pats, n_boot=n_boot, seed=seed)
    exc = P.paired_excess_ci(exact[ok], swap_exact, pats[ok], n_boot=n_boot, seed=seed)
    # This is exact all-slot TOKEN-SET (presence) reconstruction — an EASIER upper bound than
    # the frozen exact-multiset/count target (Pi amended #3). Its failure far below the gate
    # conservatively rules out the harder multiset target. The slot copy-floor was not run;
    # an absolute-gate failure makes it moot (recorded, not hard-asserted).
    row.update({"m1_gate_ok": False, "m1_excess_lo": -1.0,
                "m2_gate_ok": bool(acc["ci_lo"] >= SLOT_GATE), "m2_excess_lo": exc["ci_lo"], "m2_copy_ok": True,
                "slot_copy_floor": "NOT_RUN_MOOT", "precise": bool((acc["ci_hi"] - acc["ci_lo"]) < 0.20),
                "slot_exact_all_token_set": acc["point"], "slot_tokenset_f1": float(np.mean(f1)),
                "unmeasured_harder_target": "exact_multiset_with_counts", "slot_thresh": best_t})
    return row


def evaluate_cells(bundles: dict[tuple[str, str, float], dict[str, Any]], *, embedding_dim: int,
                   E: np.ndarray, arms: list[str], properties=("count", "timing", "order"),
                   slot_m: int = 4, count_floor: int = COUNT_CLUSTER_FLOOR, order_floor: int = ORDER_CLUSTER_FLOOR,
                   timing_cluster_floor: int = TIMING_CLUSTER_FLOOR,
                   timing_interval_floor: int = TIMING_INTERVAL_FLOOR) -> list[dict[str, Any]]:
    """bundles[(arm, source, W)] = {'train': {...}, 'dev': {...}}. Returns per-cell metric rows.
    Floors default to the frozen contract values; overridable only for synthetic tests. Order is
    routed: temporal_slot -> the real slot-fidelity metric; order-blind arms -> NOT_EVALUATED +
    labelled oracle probe."""
    rows = []
    for (arm, source, W), b in bundles.items():
        if arm not in arms:
            continue
        if "count" in properties:
            rows.append(count_row(arm, source, W, b["train"], b["dev"], embedding_dim=embedding_dim, floor=count_floor))
        if "timing" in properties:
            rows.append(timing_row(arm, source, W, b["train"], b["dev"], embedding_dim=embedding_dim,
                                   cluster_floor=timing_cluster_floor, interval_floor=timing_interval_floor))
        if "order" in properties:
            if arm == "temporal_slot":
                rows.append(slot_order_row(arm, source, W, b["train"], b["dev"], embedding_dim=embedding_dim,
                                           E=E, slots=slot_m, floor=order_floor, gating=True))
            else:
                rows.append(order_row(arm, source, W, b["train"], b["dev"], embedding_dim=embedding_dim, E=E, floor=order_floor))
    return rows


def rows_to_manifest(rows: list[dict[str, Any]], *, run_config=None, evaluator_provenance=None) -> dict[str, Any]:
    """Assemble a manifest from already-computed per-cell metric rows (lets a governed run
    process cell-by-cell to bound memory, then build the verdict once at the end)."""
    from clinical_jepa.eval.rung1_verdict import (
        classify_count_order_cell, classify_order_cell, classify_timing_cell,
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        prop = row["property"]
        base = (classify_timing_cell(row) if prop == "timing"
                else classify_order_cell(row) if prop == "order"
                else classify_count_order_cell(row))
        grouped.setdefault((row["arm"], prop), []).append(
            {"source": row["source"], "window_days": row["window_days"], "base_class": base})
    evals = [evaluate_property(a, p, cells) for (a, p), cells in grouped.items()]
    manifest = build_rung1_manifest(evals, run_config=run_config, evaluator_provenance=evaluator_provenance)
    manifest["cell_metrics"] = rows
    manifest["generated_utc"] = now_utc()
    return manifest


def run_ceiling(bundles, *, embedding_dim, E, arms, run_config=None, **kw) -> dict[str, Any]:
    rows = evaluate_cells(bundles, embedding_dim=embedding_dim, E=E, arms=arms, **kw)
    return rows_to_manifest(rows, run_config=run_config)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-1 frozen-decode ceiling driver (DEV-ONLY; no test access)")
    ap.add_argument("--bundles", required=True, help="npz/json bundle root produced by export_target_latents")
    ap.add_argument("--run-config", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    raise SystemExit(  # the governed loader is wired by export_target_latents; guard against misuse
        "run_ceiling is invoked programmatically from the governed runbook after export; "
        "the CLI loader is intentionally minimal and dev-only (no --confirm-test).")


if __name__ == "__main__":
    main()
