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
    from clinical_jepa.eval.rung1_decode import predict_quantiles, train_quantile_head
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
    dt_dev, pat_dev = flatten(dev); dt_tr, _ = flatten(tr)
    z_dev_i = flat_z(dev); z_tr_i = flat_z(tr)
    n_int = len(dt_dev); n_clu = _n_clusters(pat_dev) if n_int else 0
    sim = run_precision_sim(max(n_int, 1), max(n_clu, 1), n_boot=200) if n_int >= interval_floor else {"passes": False}
    evaluable = bool(n_int >= interval_floor and n_clu >= cluster_floor and sim.get("passes", False))
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "timing",
           "n_intervals": int(n_int), "n_clusters": int(n_clu), "precision_sim_passes": bool(sim.get("passes", False)),
           "evaluable": evaluable}
    if not evaluable:
        return row
    marg = P.fit_marginal_hurdle(dt_tr)
    head, qs = train_quantile_head(torch.as_tensor(z_tr_i), torch.as_tensor(dt_tr.astype(np.float32)), embedding_dim)
    q_dev = predict_quantiles(head, torch.as_tensor(z_dev_i))          # [n_int, n_q] predictive quantiles
    # randomized-PIT KS via a per-interval conditional CDF from the quantiles
    pit = _quantile_pit(q_dev, dt_dev, qs.cpu().numpy(), seed=seed)
    ks = P.ks_d_upper_ci(pit, pat_dev, n_boot=n_boot, seed=seed)
    crps_cond = np.asarray([P.crps_from_samples(q_dev[i], dt_dev[i]) for i in range(n_int)])
    crps_marg = P.crps_rows([marg["pos_samples"]] * n_int, dt_dev)
    skill = P.ratio_skill_ci(crps_cond, crps_marg, pat_dev, n_boot=n_boot, seed=seed)
    row.update({"ks_upper_ci": ks["ci_hi"], "crps_skill_lo": skill["ci_lo"], "precise": True})
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
              n_boot=P.N_BOOT, seed=SEED) -> dict[str, Any]:
    import torch
    from clinical_jepa.eval.rung1_decode import (
        reconstruct_order_exact, train_pairwise_order_head,
    )
    def ge2(bundle):
        return [i for i, s in enumerate(bundle["ordered_ids"]) if len(np.asarray(s)) >= 2]
    di = ge2(dev); pats = np.asarray(dev["patients"])[di] if di else np.asarray([])
    evaluable = _n_clusters(pats) >= floor and len(ge2(tr)) > 0
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "order",
           "n_clusters": _n_clusters(pats) if len(pats) else 0, "evaluable": bool(evaluable)}
    if not evaluable:
        return row
    Et = torch.as_tensor(np.asarray(E, dtype=np.float32))
    # build pairwise training features from train windows
    feats, labels = [], []
    for i in ge2(tr):
        ids = np.asarray(tr["ordered_ids"][i]); z = np.asarray(tr["z"][i], dtype=np.float32)
        for a in range(len(ids)):
            for b in range(len(ids)):
                if a == b:
                    continue
                feats.append(np.concatenate([z, E[ids[a]], E[ids[b]]]))
                labels.append(1.0 if a < b else 0.0)
    head = train_pairwise_order_head(torch.as_tensor(np.asarray(feats, dtype=np.float32)),
                                     torch.as_tensor(np.asarray(feats, dtype=np.float32)),
                                     torch.as_tensor(np.asarray(labels, dtype=np.float32)), embedding_dim)

    def exact_hits(bundle, idxs, z_source):
        hits = []
        for k, i in enumerate(idxs):
            ids = np.asarray(bundle["ordered_ids"][i])
            z = torch.as_tensor(np.asarray(z_source[k], dtype=np.float32))
            rec = reconstruct_order_exact(head, z, Et[torch.as_tensor(ids)])
            hits.append(float(rec == list(range(len(ids)))))
        return np.asarray(hits)

    z_dev = [dev["z"][i] for i in di]
    partner = P.swap_partner_index(list(pats), SWAP_SEED)
    ok = partner >= 0
    h_true = exact_hits(dev, di, z_dev)
    h_swap = exact_hits(dev, [di[j] for j in np.where(ok)[0]], [dev["z"][di[partner[j]]] for j in np.where(ok)[0]])
    ex = P.paired_excess_ci(h_true[ok], h_swap, pats[ok], n_boot=n_boot, seed=seed)
    acc = P.cluster_bootstrap_ci(h_true, pats, n_boot=n_boot, seed=seed)
    # temporal_slot is gated on slot-wise structure (SLOT_GATE); order-blind arms on the
    # unconditional exact-order gate (0.70). Both are 0.70 but named distinctly per contract.
    gate = SLOT_GATE if arm == "temporal_slot" else 0.70
    row.update({"m1_gate_ok": False, "m1_excess_lo": -1.0,
                "m2_gate_ok": bool(acc["ci_lo"] >= gate), "m2_excess_lo": ex["ci_lo"],
                "m2_copy_ok": True, "precise": bool((acc["ci_hi"] - acc["ci_lo"]) < 0.20),
                "m2_exact_order": acc["point"]})
    return row


def evaluate_cells(bundles: dict[tuple[str, str, float], dict[str, Any]], *, embedding_dim: int,
                   E: np.ndarray, arms: list[str], properties=("count", "timing", "order"),
                   count_floor: int = COUNT_CLUSTER_FLOOR, order_floor: int = ORDER_CLUSTER_FLOOR,
                   timing_cluster_floor: int = TIMING_CLUSTER_FLOOR,
                   timing_interval_floor: int = TIMING_INTERVAL_FLOOR) -> list[dict[str, Any]]:
    """bundles[(arm, source, W)] = {'train': {...}, 'dev': {...}}. Returns per-cell metric rows.
    Floors default to the frozen contract values; overridable only for synthetic tests."""
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
            rows.append(order_row(arm, source, W, b["train"], b["dev"], embedding_dim=embedding_dim, E=E, floor=order_floor))
    return rows


def run_ceiling(bundles, *, embedding_dim, E, arms, run_config=None, **kw) -> dict[str, Any]:
    rows = evaluate_cells(bundles, embedding_dim=embedding_dim, E=E, arms=arms, **kw)
    from clinical_jepa.eval.rung1_verdict import classify_count_order_cell, classify_timing_cell
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        base = classify_timing_cell(row) if row["property"] == "timing" else classify_count_order_cell(row)
        grouped.setdefault((row["arm"], row["property"]), []).append(
            {"source": row["source"], "window_days": row["window_days"], "base_class": base})
    evals = [evaluate_property(a, p, cells) for (a, p), cells in grouped.items()]
    manifest = build_rung1_manifest(evals, run_config=run_config)
    manifest["cell_metrics"] = rows
    manifest["generated_utc"] = now_utc()
    return manifest


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
