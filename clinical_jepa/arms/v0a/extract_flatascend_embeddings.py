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
    mean_emb = np.zeros((len(blocks), hidden), dtype=np.float16)
    final_emb = np.zeros((len(blocks), hidden), dtype=np.float16)
    index_path = outdir / "embedding-index.jsonl"
    file_cache: dict[str, Any] = {}

    def read_context(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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
        c0 = max(0, int(block.get("context_start_ref", 0)))
        c1 = min(len(arr) - 1, int(block["context_end_ref"]))
        ids = np.asarray(arr[c0 : c1 + 1][-args.max_context_tokens:], dtype=np.int64)
        dts = np.asarray(dt[c0 : c1 + 1][-args.max_context_tokens:], dtype=np.float32)
        return ids, dts

    try:
        rows = []
        with torch.no_grad(), index_path.open("w") as idx_f:
            for start in range(0, len(blocks), args.batch_size):
                batch_blocks = blocks[start : start + args.batch_size]
                ids, dts = zip(*(read_context(b) for b in batch_blocks))
                tok, tim, mask = _pad(list(ids), list(dts), device)
                out = model(tok, tim, attention_mask=mask)
                h = out.hidden_states
                denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
                pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
                last_idx = mask.sum(dim=1).long().clamp_min(1) - 1
                final = h[torch.arange(h.size(0), device=device), last_idx]
                mean_emb[start : start + len(batch_blocks)] = pooled.detach().cpu().numpy().astype(np.float16)
                final_emb[start : start + len(batch_blocks)] = final.detach().cpu().numpy().astype(np.float16)
                for j, b in enumerate(batch_blocks):
                    rec = {"row": start + j, "block_id": b.get("block_id"), "split": b.get("split"), "target_type": b.get("target_type"), "source_dataset": b.get("source_dataset")}
                    idx_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    finally:
        for f in file_cache.values():
            f.close()

    mean_path = outdir / "context-final-layer-mean.fp16.npy"
    final_path = outdir / "context-final-token.fp16.npy"
    np.save(mean_path, mean_emb)
    np.save(final_path, final_emb)
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
        "counts": {"contexts": len(blocks), "targets": 0},
        "count_breakdown": counts,
        "embedding_files": {"final_mean_fp16": str(mean_path), "final_token_fp16": str(final_path), "index_jsonl": str(index_path)},
        "embedding_shape": [len(blocks), hidden],
        "device": str(device),
        "max_context_tokens": args.max_context_tokens,
        "leakage_audit_status": "pass",
        "notes": "Prefix-only FlatASCEND context embeddings from re-keyed bundle; no tokens, source ids, or patient examples written.",
    }
    validate_artifact("v0a-embedding-cache", manifest)
    write_json(outdir / "embedding-cache-manifest.json", manifest)
    print(json.dumps({"embedding_manifest": str(outdir / "embedding-cache-manifest.json"), "contexts": len(blocks), "shape": [len(blocks), hidden]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0A FlatASCEND embedding scaffold / extractor")
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-context-tokens", type=int, default=256)
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
