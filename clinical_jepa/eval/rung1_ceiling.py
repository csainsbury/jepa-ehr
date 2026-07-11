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
    dt_dev, pat_dev = flatten(dev); dt_tr, _ = flatten(tr)
    z_dev_i = flat_z(dev); z_tr_i = flat_z(tr)
    n_int = len(dt_dev); n_clu = _n_clusters(pat_dev) if n_int else 0
    # Precision sim certifies the DESIGN at the cell size class (memoized); certifying at a
    # capped-smaller n is conservative — a larger real n only helps.
    sim = _precision_sim_cached(n_int, n_clu) if n_int >= interval_floor else {"passes": False}
    evaluable = bool(n_int >= interval_floor and n_clu >= cluster_floor and sim.get("passes", False))
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "timing",
           "n_intervals": int(n_int), "n_clusters": int(n_clu), "precision_sim_passes": bool(sim.get("passes", False)),
           "evaluable": evaluable}
    if not evaluable:
        return row
    # subsample intervals to bound quantile-head / KS / CRPS cost (adequacy already met)
    max_intervals = 40000
    rng = np.random.default_rng(seed)
    if n_int > max_intervals:
        sel = np.sort(rng.choice(n_int, size=max_intervals, replace=False))
        dt_dev, pat_dev, z_dev_i = dt_dev[sel], pat_dev[sel], z_dev_i[sel]
    if len(dt_tr) > max_intervals:
        selt = np.sort(rng.choice(len(dt_tr), size=max_intervals, replace=False))
        dt_tr, z_tr_i = dt_tr[selt], z_tr_i[selt]
    n_int = len(dt_dev)
    marg = P.fit_marginal_hurdle(dt_tr)
    # CONDITIONAL hurdle timing model (Pi R8 #7): models the Δt=0 point mass so the randomized
    # PIT reflects the latent, not the evaluator. z+-conditional p0 + positive-tail quantiles.
    head, qs = train_hurdle_timing_head(torch.as_tensor(z_tr_i), torch.as_tensor(dt_tr.astype(np.float32)), embedding_dim)
    p0, q_dev = predict_hurdle_timing(head, torch.as_tensor(z_dev_i))
    lv = qs.cpu().numpy()
    pit = P.hurdle_randomized_pit(p0, q_dev, lv, dt_dev, seed=seed)
    ks = P.ks_d_upper_ci(pit, pat_dev, n_boot=n_boot, seed=seed)
    # Conditional CRPS from the hurdle predictive; marginal CRPS from the shared hurdle marginal
    # (zero mass + positive tail), both vectorized.
    crps_cond = np.clip(P.hurdle_crps_rows(p0, q_dev, dt_dev, seed=seed), 0.0, None)
    marg_p0 = float(marg["p0"]); marg_pos = np.asarray(marg["pos_samples"], dtype=np.float64)
    marg_samples = np.where(np.random.default_rng(seed).uniform(size=min(len(marg_pos), 256)) < marg_p0,
                            0.0, marg_pos[:min(len(marg_pos), 256)])
    crps_marg = np.clip(P.crps_marginal_rows(marg_samples, dt_dev, seed=seed), 1e-9, None)
    skill = P.ratio_skill_ci(crps_cond, crps_marg, pat_dev, n_boot=n_boot, seed=seed)
    row.update({"ks_upper_ci": ks["ci_hi"], "crps_skill_lo": skill["ci_lo"],
                "zero_rate": float((dt_dev <= 0).mean()), "precise": True})
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
    """Exact ordered-sequence reconstruction from a single pooled vector is only conceivable
    for short windows; we bound the pairwise cost to windows with 2..max_events events (the
    excluded-large fraction is reported — Pi R8: log any cap) and subsample training pairs."""
    import torch
    from clinical_jepa.eval.rung1_decode import (
        reconstruct_order_exact, train_pairwise_order_head,
    )
    def in_range(bundle):
        return [i for i, s in enumerate(bundle["ordered_ids"]) if 2 <= len(np.asarray(s)) <= max_events]
    n_ge2_dev = sum(1 for s in dev["ordered_ids"] if len(np.asarray(s)) >= 2)
    di = in_range(dev); pats = np.asarray(dev["patients"])[di] if di else np.asarray([])
    excluded_large = int(n_ge2_dev - len(di))
    evaluable = _n_clusters(pats) >= floor and len(in_range(tr)) > 0
    row = {"arm": arm, "source": source, "window_days": float(W), "property": "order",
           "n_clusters": _n_clusters(pats) if len(pats) else 0,
           "order_max_events": int(max_events), "excluded_large_windows": excluded_large,
           "n_eval_windows": len(di), "evaluable": bool(evaluable)}
    if not evaluable:
        return row
    rng = np.random.default_rng(seed)
    if len(di) > max_dev_windows:                              # bound reconstruction cost
        di = list(np.sort(rng.choice(di, size=max_dev_windows, replace=False)))
        pats = np.asarray(dev["patients"])[di]
    Et = torch.as_tensor(np.asarray(E, dtype=np.float32))
    # build pairwise training features from train windows (VECTORIZED per window; subsampled
    # to a pair budget). All ordered pairs (a,b), a!=b, built via broadcasting.
    feats, labels = [], []
    n_pairs = 0
    tr_idx = in_range(tr)
    rng.shuffle(tr_idx)
    for i in tr_idx:
        if n_pairs >= max_train_pairs:
            break
        ids = np.asarray(tr["ordered_ids"][i]); z = np.asarray(tr["z"][i], dtype=np.float32)
        n = len(ids)
        ea = E[ids]                                          # [n, D]
        ai, bi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        m = (ai != bi).reshape(-1)
        ai = ai.reshape(-1)[m]; bi = bi.reshape(-1)[m]
        feats.append(np.concatenate([np.tile(z, (len(ai), 1)), ea[ai], ea[bi]], axis=1))
        labels.append((ai < bi).astype(np.float32))
        n_pairs += len(ai)
    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, len(np.asarray(E)[0]) + 2 * len(np.asarray(E)[0])))
    labels = np.concatenate(labels) if len(labels) else np.zeros(0, dtype=np.float32)
    head = train_pairwise_order_head(torch.as_tensor(np.asarray(feats, dtype=np.float32)),
                                     torch.as_tensor(np.asarray(feats, dtype=np.float32)),
                                     torch.as_tensor(np.asarray(labels, dtype=np.float32)), embedding_dim,
                                     steps=order_steps)

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


def rows_to_manifest(rows: list[dict[str, Any]], *, run_config=None) -> dict[str, Any]:
    """Assemble a manifest from already-computed per-cell metric rows (lets a governed run
    process cell-by-cell to bound memory, then build the verdict once at the end)."""
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
