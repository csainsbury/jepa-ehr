from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from clinical_jepa.targets.block_spans import is_censored, is_empty_target
from clinical_jepa.utils import ensure_dir, now_utc, write_json


class RolloutExportError(RuntimeError):
    pass


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_metadata(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".jsonl":
        rows = list(_iter_jsonl(p))
    else:
        raw = json.loads(p.read_text())
        rows = raw.get("rows", raw if isinstance(raw, list) else [])
    return {str(row.get("block_id")): row for row in rows if row.get("block_id")}


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _sidecar_row(block: dict[str, Any], meta: dict[str, Any], *, row_idx: int, context_len: int, target_len: int, sequence_len: int) -> dict[str, Any]:
    """Return a local sidecar row without identifiers or raw tokens."""
    return {
        "row": int(row_idx),
        "split": block.get("split"),
        "target_type": block.get("target_type"),
        "horizon_descriptor": block.get("horizon_descriptor"),
        "source_dataset": block.get("source_dataset"),
        "context_len": int(meta.get("context_len", context_len)),
        "target_len": int(meta.get("target_len", target_len)),
        "sequence_len": int(meta.get("sequence_len", sequence_len)),
        "context_med_count": _safe_int(meta, "context_med_count", 0),
        "context_lab_count": _safe_int(meta, "context_lab_count", 0),
        "context_state_count": _safe_int(meta, "context_state_count", 0),
        "target_med_count": _safe_int(meta, "target_med_count", 0),
        "target_lab_count": _safe_int(meta, "target_lab_count", 0),
        "target_state_count": _safe_int(meta, "target_state_count", 0),
    }


def _select_blocks(
    manifest: dict[str, Any],
    meta_by_block: dict[str, dict[str, Any]],
    *,
    splits: set[str],
    target_types: set[str],
    max_blocks: int,
    seed: int,
    require_metadata: bool,
    scenario_query_field: str | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for block in manifest.get("blocks", []):
        if splits and str(block.get("split")) not in splits:
            continue
        if target_types and str(block.get("target_type")) not in target_types:
            continue
        if not block.get("sequence_file") or not block.get("sequence_group"):
            continue
        meta = meta_by_block.get(str(block.get("block_id")), {})
        if require_metadata and not meta:
            continue
        if scenario_query_field and not _safe_int(meta, scenario_query_field, 0):
            continue
        selected.append((block, meta))
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_blocks > 0:
        selected = selected[:max_blocks]
    return selected


def _pad(batch: list[np.ndarray], device: Any):
    import torch

    max_len = max(len(x) for x in batch)
    out = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    for i, arr in enumerate(batch):
        out[i, : len(arr)] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return out


def export_mean_token_rollouts(
    *,
    checkpoint_path: str | Path,
    target_blocks_path: str | Path,
    output_dir: str | Path,
    metadata_index_path: str | None = None,
    splits: tuple[str, ...] = ("dev", "test"),
    target_types: tuple[str, ...] = ("T0",),
    max_blocks: int = 5000,
    batch_size: int = 256,
    max_context_tokens: int = 128,
    target_window_events: int = 32,
    horizon_count: int = 3,
    horizon_stride_events: int = 32,
    seed: int = 20260523,
    scenario_query_field: str | None = None,
    require_metadata: bool = False,
    cpu: bool = True,
) -> dict[str, Any]:
    """Export local-only mean-token latent rollout sidecars.

    The sidecar arrays and JSONL index are local governed-data artifacts. They
    are intentionally not aggregate-only and must not be committed or copied to
    Cog/wiki/public reports. The JSON manifest emitted by this function is safe
    and aggregate-only.
    """
    if horizon_count <= 0:
        raise ValueError("horizon_count must be positive")
    if target_window_events <= 0:
        raise ValueError("target_window_events must be positive")
    if horizon_stride_events <= 0:
        raise ValueError("horizon_stride_events must be positive")

    import h5py
    import torch

    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint, checkpoint_autoregression_config

    checkpoint_path = Path(checkpoint_path)
    target_blocks_path = Path(target_blocks_path)
    outdir = ensure_dir(output_dir)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dim = int(ckpt["embedding_dim"])
    autoreg_config = checkpoint_autoregression_config(ckpt)
    model = build_mean_token_jepa_from_checkpoint(ckpt)
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()

    manifest = json.loads(target_blocks_path.read_text())
    meta_by_block = _load_metadata(metadata_index_path)
    selected = _select_blocks(
        manifest,
        meta_by_block,
        splits=set(splits),
        target_types=set(target_types),
        max_blocks=max_blocks,
        seed=seed,
        require_metadata=require_metadata,
        scenario_query_field=scenario_query_field,
    )

    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    teacher_forced: list[np.ndarray] = []
    # Teacher forcing is only meaningful for the RECURSIVE arm (a horizon-conditioned head takes the
    # context latent, not the previous step). Without it the exposure gap is ABSENT, not zero.
    teacher_forcing_available = str(autoreg_config["autoregression_mode"]) == "recursive"
    rows: list[dict[str, Any]] = []
    skipped_short_future = 0
    skipped_read_error = 0
    cache: dict[str, Any] = {}

    try:
        with torch.no_grad():
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                ctxs: list[np.ndarray] = []
                targets_by_h: list[list[np.ndarray]] = [[] for _ in range(horizon_count)]
                kept: list[tuple[dict[str, Any], dict[str, Any], int, int, int]] = []
                for block, meta in batch:
                    try:
                        path = str(block["sequence_file"])
                        if path not in cache:
                            cache[path] = h5py.File(path, "r")
                        h5 = cache[path]
                        group = str(block.get("sequence_group") or block.get("sequence_id"))
                        token_ids = h5[group]["token_ids"][:]
                        seq_len = len(token_ids)
                        # Empty/censored wall-clock targets have no event-index
                        # rollout (target_start_ref = -1) — skip, never clamp to arr[0:].
                        if is_empty_target(block) or is_censored(block):
                            skipped_short_future += 1
                            continue
                        c0 = max(0, int(block.get("context_start_ref", 0)))
                        c1 = min(seq_len - 1, int(block["context_end_ref"]))
                        t0 = int(block["target_start_ref"])  # >= 0 (empties skipped)
                        if c1 < c0 or t0 >= seq_len:
                            skipped_short_future += 1
                            continue
                        horizon_targets: list[np.ndarray] = []
                        ok = True
                        for h in range(horizon_count):
                            start_idx = t0 + h * horizon_stride_events
                            end_idx = start_idx + target_window_events
                            if end_idx > seq_len:
                                ok = False
                                break
                            horizon_targets.append(np.asarray(token_ids[start_idx:end_idx], dtype=np.int64))
                        if not ok:
                            skipped_short_future += 1
                            continue
                        ctx = np.asarray(token_ids[c0 : c1 + 1][-max_context_tokens:], dtype=np.int64)
                        if len(ctx) == 0:
                            skipped_short_future += 1
                            continue
                        ctxs.append(ctx)
                        for h, arr in enumerate(horizon_targets):
                            targets_by_h[h].append(arr)
                        kept.append((block, meta, len(ctx), target_window_events, seq_len))
                    except Exception:
                        skipped_read_error += 1
                        continue
                if not kept:
                    continue
                ctx_tensor = _pad(ctxs, device)
                pred_t = model.predict_rollout_from_context_ids(ctx_tensor, horizon_count)   # FREE-RUNNING
                pred_batch = pred_t.detach().cpu().numpy().astype(np.float16)
                obs_steps = []
                for h in range(horizon_count):
                    obs_steps.append(model.mean_embed(_pad(targets_by_h[h], device)))
                obs_t = torch.stack(obs_steps, dim=1)                                        # TRUE latents
                obs_batch = obs_t.detach().cpu().numpy().astype(np.float16)

                # TEACHER-FORCED rollout (Rung-2 sub-gate 1): step h is predicted from the TRUE latent at
                # h-1 rather than from the model's own h-1 output, so the exposure gap
                # g_h = d_self_free(h) - d_self_tf(h) isolates compounding self-conditioning error from
                # one-step predictor error. Step 0 is predicted from the true CONTEXT latent in both arms,
                # so g_0 is identically 0 by construction — asserted below as a harness invariant.
                # Only defined for the recursive arm: a horizon-conditioned head does not consume a
                # previous-step latent, so there is nothing to teacher-force.
                if teacher_forcing_available:
                    tf_steps = [pred_t[:, 0, :]]                                  # h=0 == free-running
                    for h in range(1, horizon_count):
                        tf_steps.append(model.predict_rollout_from_latent(obs_t[:, h - 1, :], 1)[:, 0, :])
                    tf_t = torch.stack(tf_steps, dim=1)
                    if not torch.allclose(tf_t[:, 0, :], pred_t[:, 0, :]):
                        raise RuntimeError("teacher-forced step 0 must equal free-running step 0 "
                                           "(both start from the true context latent)")
                    teacher_forced.append(tf_t.detach().cpu().numpy().astype(np.float16))
                row_start = len(rows)
                predicted.append(pred_batch)
                observed.append(obs_batch)
                for j, (block, meta, context_len, target_len, seq_len) in enumerate(kept):
                    rows.append(_sidecar_row(block, meta, row_idx=row_start + j, context_len=context_len, target_len=target_len, sequence_len=seq_len))
    finally:
        for handle in cache.values():
            handle.close()

    if predicted:
        pred_arr = np.concatenate(predicted, axis=0)
        obs_arr = np.concatenate(observed, axis=0)
    else:
        pred_arr = np.zeros((0, horizon_count, dim), dtype=np.float16)
        obs_arr = np.zeros((0, horizon_count, dim), dtype=np.float16)
    tf_arr = (np.concatenate(teacher_forced, axis=0) if teacher_forced
              else np.zeros((0, horizon_count, dim), dtype=np.float16))

    pred_path = outdir / "predicted-rollout.fp16.npy"
    obs_path = outdir / "observed-rollout.fp16.npy"
    index_path = outdir / "rollout-index.local.jsonl"
    np.save(pred_path, pred_arr)
    np.save(obs_path, obs_arr)
    tf_path = outdir / "teacher-forced-rollout.fp16.npy"
    if teacher_forcing_available:
        np.save(tf_path, tf_arr)
    index_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))

    splits_out: dict[str, int] = {}
    target_types_out: dict[str, int] = {}
    for row in rows:
        splits_out[str(row.get("split"))] = splits_out.get(str(row.get("split")), 0) + 1
        target_types_out[str(row.get("target_type"))] = target_types_out.get(str(row.get("target_type")), 0) + 1
    report = {
        "created_utc": now_utc(),
        "aggregate_only": True,
        "local_sidecar": True,
        "checkpoint_name": checkpoint_path.name,
        "target_blocks_name": target_blocks_path.name,
        "rows_exported": int(len(rows)),
        "candidate_blocks_selected": int(len(selected)),
        "skipped_short_future": int(skipped_short_future),
        "skipped_read_error": int(skipped_read_error),
        "splits": splits_out,
        "target_types": target_types_out,
        "embedding_dim": int(dim),
        # Downstream MUST be able to tell "gap is zero" from "gap was never produced".
        "teacher_forcing_available": bool(teacher_forcing_available),
        "teacher_forced_rows": int(tf_arr.shape[0]),
        "teacher_forcing_note": ("free-running vs teacher-forced enables the exposure gap; step 0 is "
                                 "identical in both arms by construction"
                                 if teacher_forcing_available else
                                 "NOT produced: teacher forcing is undefined for a non-recursive arm, so "
                                 "the exposure gap is ABSENT and the recursive signature is NOT_EVALUABLE"),
        "autoregression_mode": autoreg_config["autoregression_mode"],
        "horizon_conditioning": autoreg_config["horizon_conditioning"],
        "horizon_count_trained": int(autoreg_config["horizon_count_trained"]),
        "max_horizons": int(autoreg_config["max_horizons"]),
        "horizon_count": int(horizon_count),
        "target_window_events": int(target_window_events),
        "horizon_stride_events": int(horizon_stride_events),
        "predicted_rollout_path": str(pred_path),
        "observed_rollout_path": str(obs_path),
        "index_path": str(index_path),
        "notes": "Local governed sidecar for autoregression readiness. Do not publish arrays or JSONL index. Index omits block IDs, patient hashes, sequence IDs, raw tokens, and examples.",
    }
    write_json(outdir / "rollout-export-manifest.json", report)
    if len(rows) == 0:
        raise RolloutExportError("No rollout rows exported; check target horizon/window settings and input target blocks")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export local-only mean-token v0B latent autoregression rollout arrays")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata-index")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--target-types", nargs="+", default=["T0"])
    ap.add_argument("--max-blocks", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--target-window-events", type=int, default=32)
    ap.add_argument("--horizon-count", type=int, default=3)
    ap.add_argument("--horizon-stride-events", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--scenario-query-field", help="Optional metadata field that must be truthy for a block to be exported")
    ap.add_argument("--require-metadata", action="store_true")
    ap.add_argument("--cpu", action="store_true", default=True)
    args = ap.parse_args(argv)
    report = export_mean_token_rollouts(
        checkpoint_path=args.checkpoint,
        target_blocks_path=args.target_blocks,
        output_dir=args.output_dir,
        metadata_index_path=args.metadata_index,
        splits=tuple(args.splits),
        target_types=tuple(args.target_types),
        max_blocks=args.max_blocks,
        batch_size=args.batch_size,
        max_context_tokens=args.max_context_tokens,
        target_window_events=args.target_window_events,
        horizon_count=args.horizon_count,
        horizon_stride_events=args.horizon_stride_events,
        seed=args.seed,
        scenario_query_field=args.scenario_query_field,
        require_metadata=args.require_metadata,
        cpu=args.cpu,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
