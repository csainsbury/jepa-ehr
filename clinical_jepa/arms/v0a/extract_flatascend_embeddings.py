from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json
from clinical_jepa.validation import validate_artifact


def _dry_run(args: argparse.Namespace, arm: dict[str, Any], targets: dict[str, Any], outdir: Path) -> int:
    manifest = {
        "schema_version": "clinical-jepa-v0a-embedding-cache-v0",
        "created_utc": now_utc(),
        "dry_run": True,
        "flatascend_checkpoint_alias": "placeholder-local-alias",
        "tokenizer_version": "placeholder-version",
        "prefix_only": True,
        "target_layers": arm.get("target_layers", ["shallow", "mid", "final"]),
        "pooling": arm.get("pooling", ["mean", "attention"]),
        "counts": {"contexts": len(targets.get("blocks", [])), "targets": len(targets.get("blocks", []))},
        "leakage_audit_status": "pass",
        "notes": "dry-run manifest only; no FlatASCEND checkpoint loaded",
    }
    validate_artifact("v0a-embedding-cache", manifest)
    write_json(outdir / "embedding-cache-manifest.json", manifest)
    print(json.dumps({"embedding_manifest": str(outdir / "embedding-cache-manifest.json")}, indent=2))
    return 0


def _load_blocks(targets: dict[str, Any], max_blocks: int) -> list[dict[str, Any]]:
    blocks = [b for b in targets.get("blocks", []) if b.get("sequence_file") and b.get("sequence_group")]
    # Prefer balanced dev/test visibility after train, but keep deterministic manifest order.
    if max_blocks and len(blocks) > max_blocks:
        by_split = {"train": [], "dev": [], "test": []}
        for b in blocks:
            by_split.setdefault(str(b.get("split", "train")), []).append(b)
        train_cap = int(max_blocks * 0.70)
        dev_cap = int(max_blocks * 0.15)
        test_cap = max_blocks - train_cap - dev_cap
        blocks = by_split.get("train", [])[:train_cap] + by_split.get("dev", [])[:dev_cap] + by_split.get("test", [])[:test_cap]
    return blocks


def _load_id_to_token_from_dataset_config(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    cfg = load_yaml(path)
    vocab_path = cfg.get("vocabulary", {}).get("vocab_json_path")
    if not vocab_path:
        return {}
    p = Path(vocab_path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    if "id_to_token" in raw:
        return {int(k): str(v) for k, v in raw["id_to_token"].items()}
    if "token_to_id" in raw:
        return {int(v): str(k) for k, v in raw["token_to_id"].items()}
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def _token_family_counts(ids: np.ndarray, id_to_token: dict[int, str]) -> dict[str, int]:
    counts = {"med": 0, "lab": 0, "state": 0}
    for tid in ids:
        tok = id_to_token.get(int(tid), "")
        if tok.startswith("MED:"):
            counts["med"] += 1
        elif tok.startswith("LAB:"):
            counts["lab"] += 1
        elif tok.startswith("STATE:"):
            counts["state"] += 1
    return counts


def _pad(ids: list[np.ndarray], dts: list[np.ndarray], device: Any):
    import torch

    max_len = max(len(x) for x in ids)
    tok = torch.zeros((len(ids), max_len), dtype=torch.long, device=device)
    tim = torch.zeros((len(ids), max_len), dtype=torch.float32, device=device)
    mask = torch.zeros((len(ids), max_len), dtype=torch.float32, device=device)
    for i, (arr, dt) in enumerate(zip(ids, dts)):
        n = len(arr)
        tok[i, :n] = torch.as_tensor(arr, dtype=torch.long, device=device)
        tim[i, :n] = torch.as_tensor(dt, dtype=torch.float32, device=device)
        mask[i, :n] = 1.0
    return tok, tim, mask


def _pool_hidden(model: Any, ids: list[np.ndarray], dts: list[np.ndarray], device: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    tok, tim, mask = _pad(ids, dts, device)
    out = model(tok, tim, attention_mask=mask)
    h = out.hidden_states
    denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
    last_idx = mask.sum(dim=1).long().clamp_min(1) - 1
    final = h[torch.arange(h.size(0), device=device), last_idx]
    return pooled.detach().cpu().numpy().astype(np.float16), final.detach().cpu().numpy().astype(np.float16)


def _real_run(args: argparse.Namespace, arm: dict[str, Any], targets: dict[str, Any], outdir: Path) -> int:
    import h5py
    import torch

    source_root = Path(arm.get("source_root") or "/workspace/ascend-flat-src")
    sys.path.insert(0, str(source_root))
    from src.model.flat_ascendgpt import FlatASCENDgpt, FlatASCENDgptConfig  # type: ignore

    checkpoint_path = Path(arm.get("checkpoint_path", ""))
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing FlatASCEND checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    model = FlatASCENDgpt(FlatASCENDgptConfig.from_dict(cfg))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()

    blocks = _load_blocks(targets, args.max_blocks)
    if not blocks:
        raise SystemExit("No eligible target blocks with sequence_file/sequence_group")
    hidden = int(cfg.get("hidden_size", 768))
    context_mean = np.zeros((len(blocks), hidden), dtype=np.float16)
    context_final = np.zeros((len(blocks), hidden), dtype=np.float16)
    target_mean = np.zeros((len(blocks), hidden), dtype=np.float16) if args.include_target_embeddings else None
    target_final = np.zeros((len(blocks), hidden), dtype=np.float16) if args.include_target_embeddings else None
    index_path = outdir / "embedding-index.jsonl"
    file_cache: dict[str, Any] = {}
    id_to_token = _load_id_to_token_from_dataset_config(args.dataset_config)

    def read_span(block: dict[str, Any], start_ref: str, end_ref: str, max_tokens: int, from_end: bool) -> tuple[np.ndarray, np.ndarray, int]:
        path = str(block["sequence_file"])
        if path not in file_cache:
            file_cache[path] = h5py.File(path, "r")
        h5 = file_cache[path]
        group = str(block["sequence_group"])
        arr = h5[group]["token_ids"][:]
        if "time_deltas" in h5[group]:
            dt = h5[group]["time_deltas"][:]
        else:
            dt = np.zeros_like(arr, dtype=np.float32)
        s = max(0, int(block[start_ref]))
        e = min(len(arr) - 1, int(block[end_ref]))
        ids = np.asarray(arr[s : e + 1], dtype=np.int64)
        dts = np.asarray(dt[s : e + 1], dtype=np.float32)
        if max_tokens and len(ids) > max_tokens:
            if from_end:
                ids = ids[-max_tokens:]
                dts = dts[-max_tokens:]
            else:
                ids = ids[:max_tokens]
                dts = dts[:max_tokens]
        return ids, dts, int(len(arr))

    try:
        with torch.no_grad(), index_path.open("w") as idx_f:
            for start in range(0, len(blocks), args.batch_size):
                batch_blocks = blocks[start : start + args.batch_size]
                c_spans = [read_span(b, "context_start_ref", "context_end_ref", args.max_context_tokens, True) for b in batch_blocks]
                c_ids = [x[0] for x in c_spans]
                c_dts = [x[1] for x in c_spans]
                seq_lens = [x[2] for x in c_spans]
                pooled, final = _pool_hidden(model, list(c_ids), list(c_dts), device)
                context_mean[start : start + len(batch_blocks)] = pooled
                context_final[start : start + len(batch_blocks)] = final

                if args.include_target_embeddings and target_mean is not None and target_final is not None:
                    t_spans = [read_span(b, "target_start_ref", "target_end_ref", args.max_target_tokens, False) for b in batch_blocks]
                    t_ids = [x[0] for x in t_spans]
                    t_dts = [x[1] for x in t_spans]
                    t_pooled, t_final = _pool_hidden(model, list(t_ids), list(t_dts), device)
                    target_mean[start : start + len(batch_blocks)] = t_pooled
                    target_final[start : start + len(batch_blocks)] = t_final

                for j, b in enumerate(batch_blocks):
                    context_counts = _token_family_counts(c_ids[j], id_to_token)
                    target_ids = t_ids[j] if args.include_target_embeddings else np.asarray([], dtype=np.int64)
                    target_counts = _token_family_counts(target_ids, id_to_token)
                    rec = {
                        "row": start + j,
                        "block_id": b.get("block_id"),
                        "split": b.get("split"),
                        "target_type": b.get("target_type"),
                        "horizon_descriptor": b.get("horizon_descriptor"),
                        "gap_events": b.get("gap_events"),
                        "context_len": int(len(c_ids[j])),
                        "target_len": int(len(target_ids)) if args.include_target_embeddings else int(int(b.get("target_end_ref", 0)) - int(b.get("target_start_ref", 0)) + 1),
                        "sequence_len": int(seq_lens[j]),
                        "context_med_count": int(context_counts["med"]),
                        "context_lab_count": int(context_counts["lab"]),
                        "context_state_count": int(context_counts["state"]),
                        "target_med_count": int(target_counts["med"]),
                        "target_lab_count": int(target_counts["lab"]),
                        "target_state_count": int(target_counts["state"]),
                        "source_dataset": b.get("source_dataset"),
                    }
                    idx_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    finally:
        for f in file_cache.values():
            f.close()

    context_mean_path = outdir / "context-final-layer-mean.fp16.npy"
    context_final_path = outdir / "context-final-token.fp16.npy"
    np.save(context_mean_path, context_mean)
    np.save(context_final_path, context_final)
    embedding_files: dict[str, str] = {
        "final_mean_fp16": str(context_mean_path),
        "final_token_fp16": str(context_final_path),
        "index_jsonl": str(index_path),
    }
    if args.include_target_embeddings and target_mean is not None and target_final is not None:
        target_mean_path = outdir / "target-final-layer-mean.fp16.npy"
        target_final_path = outdir / "target-final-token.fp16.npy"
        np.save(target_mean_path, target_mean)
        np.save(target_final_path, target_final)
        embedding_files.update({"target_final_mean_fp16": str(target_mean_path), "target_final_token_fp16": str(target_final_path)})

    counts: dict[str, int] = {}
    for b in blocks:
        key = f"{b.get('split')}:{b.get('target_type')}"
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "schema_version": "clinical-jepa-v0a-embedding-cache-v0",
        "created_utc": now_utc(),
        "dry_run": False,
        "flatascend_checkpoint_alias": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_version": "flatascend_outcomes_b1a_v1",
        "prefix_only": True,
        "target_layers": ["final"],
        "pooling": ["mean", "final_token"],
        "counts": {"contexts": len(blocks), "targets": len(blocks) if args.include_target_embeddings else 0},
        "count_breakdown": counts,
        "embedding_files": embedding_files,
        "embedding_shape": [len(blocks), hidden],
        "target_embedding_shape": [len(blocks), hidden] if args.include_target_embeddings else None,
        "device": str(device),
        "max_context_tokens": args.max_context_tokens,
        "max_target_tokens": args.max_target_tokens,
        "leakage_audit_status": "pass",
        "notes": "Prefix-only FlatASCEND context embeddings plus optional target-block embeddings; no tokens, source ids, or patient examples written.",
    }
    validate_artifact("v0a-embedding-cache", manifest)
    write_json(outdir / "embedding-cache-manifest.json", manifest)
    print(json.dumps({"embedding_manifest": str(outdir / "embedding-cache-manifest.json"), "contexts": len(blocks), "targets": manifest["counts"]["targets"], "shape": [len(blocks), hidden]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0A FlatASCEND embedding scaffold / extractor")
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--dataset-config")
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-context-tokens", type=int, default=256)
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--include-target-embeddings", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    require_pass_leakage(args.leakage_report)
    cfg = load_yaml(args.arms_config)
    arm = cfg.get("v0A_flatascend_scaffold", {})
    targets = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    if args.dry_run:
        return _dry_run(args, arm, targets, outdir)
    if not arm.get("enabled"):
        raise SystemExit("v0A is disabled/gated in arms config")
    return _real_run(args, arm, targets, outdir)


if __name__ == "__main__":
    sys.exit(main())
