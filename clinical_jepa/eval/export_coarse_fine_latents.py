"""Export coarse/fine latents for rung 0 (Pi R5) — the bridge to real data.

For each wall-clock T0 block (one governed horizon W), emit, per granularity:
  - COARSE   : query ẑ_coarse (CONTEXT-ONLY), target z_coarse (full-W pool / z_empty).
  - FINE     : per sub-window k, query ẑ_k (context-only), target z_k.
  - COARSE_B : query ẑ_coarse, target = pool of a frozen B-event subsample of W.
  - FINE_B   : per k, query ẑ_k, target = pool of a B-event subsample of sub-window k.
  - k1 (opt) : the K=1 harness null (fine ≡ coarse).
Plus predicted (context-only) + observed counts for the raw-count corroboration route.

CRITICAL (C1): every QUERY is produced by the context-only functions in
``coarse_fine_latents`` (``coarse_query`` / ``fine_queries``) — no observed future
count/event ever weights a query. Governed local sidecars (fp16 .npy + .jsonl index);
never published.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.coarse_fine_latents import (
    budget_subsample_ids,
    coarse_query,
    fine_queries,
    predicted_counts,
)
from clinical_jepa.targets.block_spans import is_censored, is_empty_target, read_target_span
from clinical_jepa.targets.subwindow_blocks import annotate_block_subwindows
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, write_json


def _events(ids: np.ndarray) -> np.ndarray:
    a = np.asarray(ids)
    return a[a != 0]


def _read_block(block: dict[str, Any], token_ids: np.ndarray, cumulative_days: np.ndarray,
                *, delta_days: float, mode: str, max_context: int) -> dict[str, Any] | None:
    """Read context + full-W + sub-window token ids for one block. None if censored."""
    if is_censored(block):
        return None
    c0 = max(0, int(block.get("context_start_ref", 0)))
    c1 = min(len(token_ids) - 1, int(block["context_end_ref"]))
    if c1 < c0:
        return None
    ctx = np.asarray(token_ids[c0 : c1 + 1][-max_context:], dtype=np.int64)
    full_ids, full_empty = read_target_span(block, token_ids)
    ann = annotate_block_subwindows(block, cumulative_days, delta_days=delta_days, mode=mode,
                                    seq_len=len(token_ids))
    subs = []
    for s in ann["subwindows"]:
        # Each sub-window dict carries target_start_ref/target_end_ref/empty_target, so
        # the SAME -1-safe reader handles it (single-span-reader invariant).
        sub_ids, sub_empty = read_target_span(s, token_ids)
        subs.append({"ids": np.asarray(sub_ids, dtype=np.int64), "is_empty": bool(sub_empty),
                     "subwindow_k": s["subwindow_k"], "n": int(s["n_target_events"])})
    return {"ctx": ctx, "full_ids": np.asarray(full_ids, dtype=np.int64), "full_empty": bool(full_empty),
            "K": int(ann["K"]), "subwindows": subs, "block": block}


def _pad(rows: list[np.ndarray], device: Any) -> Any:
    import torch
    mx = max((len(r) for r in rows), default=1)
    out = torch.zeros((len(rows), max(1, mx)), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        if len(r):
            out[i, : len(r)] = torch.as_tensor(r, dtype=torch.long, device=device)
    return out


def export_blocks(model: Any, items: list[dict[str, Any]], *, W: float, delta_days: float,
                  mode: str, budget_B: int, seed: int, device: Any = None) -> dict[str, dict[str, Any]]:
    """items = [{ctx, full_ids, full_empty, K, subwindows, block}] (all same K for a
    fixed W under fixed_width_delta). Returns {granularity: {queries, targets, index}}."""
    import torch

    if not items:
        return {}
    device = device or torch.device("cpu")
    K = items[0]["K"]
    ctx = _pad([it["ctx"] for it in items], device)
    with torch.no_grad():
        zc_query = coarse_query(model, ctx).cpu().numpy()                 # [N,D] CONTEXT-ONLY
        zf_query = fine_queries(model, ctx, K).cpu().numpy()             # [N,K,D] CONTEXT-ONLY
        counts = predicted_counts(model, ctx, K)
        z_coarse = model.target_latent(
            _pad([it["full_ids"] if len(it["full_ids"]) else np.zeros(1, np.int64) for it in items], device),
            torch.tensor([it["full_empty"] for it in items], dtype=torch.bool, device=device),
        ).cpu().numpy()

    def _idx(it: dict[str, Any], granularity: str, subwindow_k: int | None, n_obs: int) -> dict[str, Any]:
        b = it["block"]
        return {"block_id": b.get("block_id"), "patient_hash": b.get("patient_hash"),
                "source_dataset": b.get("source_dataset"), "split": b.get("split"),
                "window_days": float(W), "target_type": "T0", "granularity": granularity,
                "subwindow_k": subwindow_k, "occupancy_class_hint": ("empty" if n_obs == 0 else "populated"),
                "n_target_events": int(n_obs), "context_len": int(len(it["ctx"]))}

    out: dict[str, dict[str, Any]] = {g: {"queries": [], "targets": [], "index": []} for g in
                                      ("coarse", "coarse_B", "fine", "fine_B", "k1")}

    # coarse + coarse_B
    rng = np.random.default_rng(seed)
    for i, it in enumerate(items):
        out["coarse"]["queries"].append(zc_query[i]); out["coarse"]["targets"].append(z_coarse[i])
        out["coarse"]["index"].append(_idx(it, "coarse", None, len(_events(it["full_ids"]))))
        subB = budget_subsample_ids(it["full_ids"], budget_B, int(rng.integers(1 << 30)))
        if subB is not None:
            with torch.no_grad():
                zb = model.target_latent(_pad([subB], device), torch.tensor([False], dtype=torch.bool, device=device)).cpu().numpy()[0]
            out["coarse_B"]["queries"].append(zc_query[i]); out["coarse_B"]["targets"].append(zb)
            out["coarse_B"]["index"].append(_idx(it, "coarse_B", None, len(subB)))

    # fine + fine_B + k1
    for k in range(K):
        sids = [it["subwindows"][k]["ids"] if len(it["subwindows"][k]["ids"]) else np.zeros(1, np.int64) for it in items]
        is_e = torch.tensor([it["subwindows"][k]["is_empty"] for it in items], dtype=torch.bool, device=device)
        with torch.no_grad():
            zk = model.target_latent(_pad(sids, device), is_e).cpu().numpy()
        for i, it in enumerate(items):
            n_k = it["subwindows"][k]["n"]
            out["fine"]["queries"].append(zf_query[i, k]); out["fine"]["targets"].append(zk[i])
            out["fine"]["index"].append(_idx(it, "fine", k, n_k))
            subB = budget_subsample_ids(it["subwindows"][k]["ids"], budget_B, int(rng.integers(1 << 30)))
            if subB is not None:
                with torch.no_grad():
                    zbk = model.target_latent(_pad([subB], device), torch.tensor([False], dtype=torch.bool, device=device)).cpu().numpy()[0]
                out["fine_B"]["queries"].append(zf_query[i, k]); out["fine_B"]["targets"].append(zbk)
                out["fine_B"]["index"].append(_idx(it, "fine_B", k, len(subB)))
    # K=1 harness null: fine at K==1 IS coarse (same query + full-W target).
    if K == 1:
        out["k1"] = {"queries": list(out["fine"]["queries"]), "targets": list(out["fine"]["targets"]),
                     "index": [{**r, "granularity": "k1"} for r in out["fine"]["index"]]}

    return {g: {"queries": np.asarray(v["queries"], dtype=np.float32), "targets": np.asarray(v["targets"], dtype=np.float32),
                "index": v["index"]} for g, v in out.items() if v["index"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export rung-0 coarse/fine latents for one (source, W) manifest")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--target-blocks", required=True, help="wall-clock T0 target-block manifest at horizon W")
    ap.add_argument("--window-days", type=float, required=True)
    ap.add_argument("--delta-days", type=float, required=True)
    ap.add_argument("--partition-mode", default="fixed_width_delta")
    ap.add_argument("--budget-B", type=int, default=2)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    import h5py
    import torch

    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset_cfg = load_yaml(args.dataset_config)
    time_channel = str(dataset_cfg.get("time_channel") or "cumulative_days")
    model = build_mean_token_jepa_from_checkpoint(torch.load(args.checkpoint, map_location=device)).to(device).eval()
    blocks = read_json(args.target_blocks).get("blocks", [])

    outdir = ensure_dir(args.output_dir)
    file_cache: dict[str, Any] = {}
    agg: dict[str, dict[str, Any]] = {}
    batch: list[dict[str, Any]] = []
    n_censored = 0

    def _flush():
        nonlocal batch
        if not batch:
            return
        part = export_blocks(model, batch, W=args.window_days, delta_days=args.delta_days,
                             mode=args.partition_mode, budget_B=args.budget_B, seed=args.seed, device=device)
        for g, v in part.items():
            a = agg.setdefault(g, {"queries": [], "targets": [], "index": []})
            a["queries"].append(v["queries"]); a["targets"].append(v["targets"]); a["index"].extend(v["index"])
        batch = []

    try:
        for b in blocks:
            path = str(b.get("sequence_file") or "")
            grp = str(b.get("sequence_group") or b.get("sequence_id"))
            if not path or is_censored(b):
                n_censored += 1 if is_censored(b) else 0
                continue
            if path not in file_cache:
                file_cache[path] = h5py.File(path, "r")
            g = file_cache[path].get(grp)
            if g is None or "token_ids" not in g or time_channel not in g:
                continue
            item = _read_block(b, g["token_ids"][:], g[time_channel][:],
                               delta_days=args.delta_days, mode=args.partition_mode, max_context=args.max_context_tokens)
            if item is None:
                n_censored += 1
                continue
            batch.append(item)
            if len(batch) >= args.batch_size:
                _flush()
        _flush()
    finally:
        for f in file_cache.values():
            f.close()

    report: dict[str, Any] = {"created_utc": now_utc(), "window_days": args.window_days,
                              "partition_mode": args.partition_mode, "delta_days": args.delta_days,
                              "budget_B": args.budget_B, "censored_excluded": int(n_censored),
                              "granularities": {}, "aggregate_only": True}
    for g, v in agg.items():
        Q = np.concatenate(v["queries"], 0) if v["queries"] else np.zeros((0, 0), np.float32)
        T = np.concatenate(v["targets"], 0) if v["targets"] else np.zeros((0, 0), np.float32)
        np.save(outdir / f"{g}_queries.npy", Q.astype(np.float16))
        np.save(outdir / f"{g}_targets.npy", T.astype(np.float16))
        with (outdir / f"{g}_index.jsonl").open("w") as f:
            for row in v["index"]:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        report["granularities"][g] = {"n": len(v["index"]), "dim": int(Q.shape[1]) if Q.size else 0}
    write_json(outdir / "export-report.json", report)
    print(json.dumps({"output": str(outdir), "window_days": args.window_days,
                      "granularities": report["granularities"], "censored_excluded": n_censored}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
