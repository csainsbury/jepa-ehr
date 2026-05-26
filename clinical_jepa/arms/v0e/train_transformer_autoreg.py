from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.arms.v0b.train_minimal_jepa import effective_rank
from clinical_jepa.arms.v0e.transformer_autoreg import TransformerAutoregConfig, TransformerHorizonAutoregressor, checkpoint_metadata
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json


def _read_examples(blocks: list[dict[str, Any]], max_blocks: int, max_context: int, target_window: int, horizon_count: int, horizon_stride: int, seed: int) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    import h5py

    rng = random.Random(seed)
    rows = [b for b in blocks if b.get("sequence_file") and b.get("sequence_group") and b.get("target_start_ref") is not None]
    rng.shuffle(rows)
    if max_blocks > 0:
        rows = rows[:max_blocks]
    cache: dict[str, Any] = {}
    examples: list[tuple[str, np.ndarray, list[np.ndarray]]] = []
    try:
        for b in rows:
            path = str(b["sequence_file"])
            if path not in cache:
                cache[path] = h5py.File(path, "r")
            h5 = cache[path]
            group = str(b.get("sequence_group") or b.get("sequence_id"))
            arr = h5[group]["token_ids"][:]
            c0 = max(0, int(b.get("context_start_ref", 0)))
            c1 = min(len(arr) - 1, int(b["context_end_ref"]))
            t0 = max(0, int(b["target_start_ref"]))
            if c1 < c0 or t0 >= len(arr):
                continue
            context = np.asarray(arr[c0 : c1 + 1][-max_context:], dtype=np.int64)
            targets = []
            ok = len(context) > 0
            for h in range(horizon_count):
                start = t0 + h * horizon_stride
                end = start + target_window
                if end > len(arr):
                    ok = False
                    break
                targets.append(np.asarray(arr[start:end], dtype=np.int64))
            if ok:
                examples.append((str(b.get("split", "train")), context, targets))
    finally:
        for f in cache.values():
            f.close()
    return examples


def _pad(batch: list[np.ndarray], device: Any):
    import torch

    max_len = max(len(x) for x in batch)
    out = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    for i, arr in enumerate(batch):
        out[i, : len(arr)] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return out


def _dry_run(args: argparse.Namespace, outdir: Path, vocab_size: int) -> int:
    config = TransformerAutoregConfig(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        max_horizons=args.max_horizons or args.horizon_count,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        max_len=max(args.max_context_tokens, args.target_window_events),
        dropout=args.dropout,
    )
    report = {
        "schema_version": "clinical-jepa-transformer-autoreg-train-v0",
        "created_utc": now_utc(),
        "dry_run": True,
        "aggregate_only": True,
        "architecture": config.to_dict(),
        "horizon_count_trained": int(args.horizon_count),
        "target_window_events": int(args.target_window_events),
        "horizon_stride_events": int(args.horizon_stride_events),
        "checkpoint_written": False,
        "notes": "Dry run only; no governed data read and no checkpoint written.",
    }
    write_json(outdir / "train-manifest.json", report)
    print(json.dumps({"train_manifest": str(outdir / "train-manifest.json"), "dry_run": True}, indent=2))
    return 0


def _eval_split(model: Any, rows: list[tuple[np.ndarray, list[np.ndarray]]], *, batch_size: int, horizon_count: int, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not rows:
        return {"n": 0, "cosine": 0.0, "effective_rank": 0.0}
    preds = []
    cosines = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            ctx = _pad([b[0] for b in batch], device)
            target_ids = [_pad([b[1][h] for b in batch], device) for h in range(horizon_count)]
            pred = model.predict_rollout_from_context_ids(ctx, horizon_count)
            target = model.encode_target_rollout_from_ids(target_ids)
            cos = F.cosine_similarity(pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1)
            cosines.extend(float(x) for x in cos.cpu())
            preds.append(pred.reshape(-1, pred.shape[-1]).detach().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, model.embedding_dim), dtype=np.float32)
    return {"n": len(rows), "cosine": float(np.mean(cosines)), "effective_rank": effective_rank(pred_arr)}


def _real_run(args: argparse.Namespace, dataset: dict[str, Any], targets: dict[str, Any], outdir: Path) -> int:
    import torch

    seed = int(dataset.get("run", {}).get("seed", 20260523))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab_size = int(dataset.get("vocabulary", {}).get("vocab_size") or args.vocab_size)
    examples = _read_examples(
        targets.get("blocks", []),
        args.max_blocks,
        args.max_context_tokens,
        args.target_window_events,
        args.horizon_count,
        args.horizon_stride_events,
        seed,
    )
    train = [(c, t) for split, c, t in examples if split == "train"]
    dev = [(c, t) for split, c, t in examples if split == "dev"]
    if not train:
        raise SystemExit("No train examples available for transformer autoreg run")
    config = TransformerAutoregConfig(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        max_horizons=args.max_horizons or args.horizon_count,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        max_len=max(args.max_context_tokens, args.target_window_events),
        dropout=args.dropout,
        predictor_hidden_mult=args.predictor_hidden_mult,
    )
    model = TransformerHorizonAutoregressor(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(seed)
    metrics_last: dict[str, float] = {}
    model.train()
    for _step in range(1, args.real_steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        ctx = _pad([b[0] for b in batch], device)
        target_ids = [_pad([b[1][h] for b in batch], device) for h in range(args.horizon_count)]
        loss, metrics_last = model.training_loss(
            ctx,
            target_ids,
            variance_weight=args.variance_weight,
            wrong_horizon_weight=args.wrong_horizon_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_eval = _eval_split(model, train[: min(len(train), args.eval_examples)], batch_size=args.batch_size, horizon_count=args.horizon_count, device=device)
    dev_eval = _eval_split(model, dev[: min(len(dev), args.eval_examples)], batch_size=args.batch_size, horizon_count=args.horizon_count, device=device)
    ckpt_path = outdir / "transformer-autoreg-v0e.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        **checkpoint_metadata(config, horizon_count_trained=args.horizon_count),
        "created_utc": now_utc(),
        "target_window_events": int(args.target_window_events),
        "horizon_stride_events": int(args.horizon_stride_events),
    }, ckpt_path)
    report = {
        "schema_version": "clinical-jepa-transformer-autoreg-train-v0",
        "created_utc": now_utc(),
        "dry_run": False,
        "aggregate_only": True,
        "architecture": config.to_dict(),
        "horizon_count_trained": int(args.horizon_count),
        "target_window_events": int(args.target_window_events),
        "horizon_stride_events": int(args.horizon_stride_events),
        "n_examples_loaded": len(examples),
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "trained_steps": int(args.real_steps),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "last_batch_metrics": metrics_last,
        "train_eval": train_eval,
        "dev_eval": dev_eval,
        "checkpoint_written": True,
        "checkpoint_name": ckpt_path.name,
        "notes": "Aggregate-only training manifest; checkpoint remains local governed artifact when run on governed data.",
    }
    write_json(outdir / "train-manifest.json", report)
    print(json.dumps({"train_manifest": str(outdir / "train-manifest.json"), "dev_cosine": dev_eval["cosine"], "checkpoint": str(ckpt_path)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train small transformer autoregressive latent predictor scaffold")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--variance-weight", type=float, default=0.01)
    ap.add_argument("--wrong-horizon-weight", type=float, default=0.02)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--encoder-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--predictor-hidden-mult", type=int, default=2)
    ap.add_argument("--horizon-count", type=int, default=3)
    ap.add_argument("--max-horizons", type=int, default=0)
    ap.add_argument("--target-window-events", type=int, default=32)
    ap.add_argument("--horizon-stride-events", type=int, default=128)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=5000)
    ap.add_argument("--eval-examples", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    require_pass_leakage(args.leakage_report)
    dataset = load_yaml(args.dataset_config)
    targets = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    if args.horizon_count <= 0:
        raise SystemExit("--horizon-count must be positive")
    if args.dry_run:
        vocab_size = int(dataset.get("vocabulary", {}).get("vocab_size") or args.vocab_size)
        return _dry_run(args, outdir, vocab_size)
    return _real_run(args, dataset, targets, outdir)


if __name__ == "__main__":
    sys.exit(main())
