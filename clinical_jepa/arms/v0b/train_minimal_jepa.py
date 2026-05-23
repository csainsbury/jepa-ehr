from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json
from clinical_jepa.validation import validate_artifact


def effective_rank(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    _, s, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _dry_run(args: argparse.Namespace, arms: dict[str, Any], dataset: dict[str, Any], targets: dict[str, Any], outdir: Path) -> int:
    rng = np.random.default_rng(20260523)
    n = max(8, len(targets.get("blocks", [])))
    dim = int(arms.get("v0B_minimal_jepa", {}).get("context_encoder", {}).get("hidden_dim", 256))
    dim = min(dim, 64)
    emb = rng.normal(size=(n, dim)).astype("float32")
    diag = {
        "schema_version": "clinical-jepa-v0b-collapse-diagnostics-v0",
        "created_utc": now_utc(),
        "dry_run": args.dry_run,
        "effective_rank": effective_rank(emb),
        "per_dimension_variance_mean": float(emb.var(axis=0).mean()),
        "average_target_predictor_competitive": False,
        "utilisation_correlation_placeholder": 0.0,
    }
    train_manifest = {
        "schema_version": "clinical-jepa-v0b-train-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": args.dry_run,
        "architecture": arms.get("v0B_minimal_jepa", {}),
        "n_synthetic_examples": n,
        "trained_steps": 0,
        "leakage_audit_status": "pass",
    }
    validate_artifact("v0b-train-manifest", train_manifest)
    write_json(outdir / "train-manifest.json", train_manifest)
    write_json(outdir / "collapse-diagnostics.json", diag)
    write_json(outdir / "embedding-manifest.json", {"created_utc": now_utc(), "dry_run": True, "embedding_shape": [n, dim], "aggregate_only": True})
    write_json(outdir / "checkpoint-manifest.json", {"created_utc": now_utc(), "checkpoint_written": False, "reason": "dry_run"})
    print(json.dumps({"train_manifest": str(outdir / "train-manifest.json"), "effective_rank": diag["effective_rank"]}, indent=2))
    return 0


def _read_examples(blocks: list[dict[str, Any]], max_blocks: int, max_context: int, max_target: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    import h5py

    rng = random.Random(seed)
    blocks = [b for b in blocks if b.get("sequence_file") and b.get("sequence_group") and b.get("target_start_ref") is not None]
    rng.shuffle(blocks)
    if max_blocks > 0:
        blocks = blocks[:max_blocks]
    cache: dict[str, Any] = {}
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        for b in blocks:
            path = str(b["sequence_file"])
            if path not in cache:
                cache[path] = h5py.File(path, "r")
            h5 = cache[path]
            group = str(b["sequence_group"])
            arr = h5[group]["token_ids"][:]
            c0 = max(0, int(b.get("context_start_ref", 0)))
            c1 = min(len(arr) - 1, int(b["context_end_ref"]))
            t0 = max(0, int(b["target_start_ref"]))
            t1 = min(len(arr) - 1, int(b["target_end_ref"]))
            if c1 < c0 or t1 < t0:
                continue
            context = np.asarray(arr[c0 : c1 + 1][-max_context:], dtype=np.int64)
            target = np.asarray(arr[t0 : t1 + 1][:max_target], dtype=np.int64)
            if len(context) == 0 or len(target) == 0:
                continue
            examples.append((str(b.get("split", "train")), context, target))
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


def _real_run(args: argparse.Namespace, arms: dict[str, Any], dataset: dict[str, Any], targets: dict[str, Any], outdir: Path) -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed = int(dataset.get("run", {}).get("seed", 20260523))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab_size = int(dataset.get("vocabulary", {}).get("vocab_size") or 438)
    dim = int(args.embedding_dim)

    examples = _read_examples(targets.get("blocks", []), args.max_blocks, args.max_context_tokens, args.max_target_tokens, seed)
    train = [(c, t) for split, c, t in examples if split == "train"]
    dev = [(c, t) for split, c, t in examples if split == "dev"]
    if not train:
        raise SystemExit("No train examples available for real v0B run")

    class MeanJEPA(nn.Module):
        def __init__(self, vocab: int, d: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab, d, padding_idx=0)
            self.predictor = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

        def mean_embed(self, ids):
            mask = (ids != 0).float().unsqueeze(-1)
            emb = self.embedding(ids) * mask
            return emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        def forward(self, ctx, tgt):
            ctx_z = self.mean_embed(ctx)
            with torch.no_grad():
                tgt_z = self.mean_embed(tgt)
            pred = self.predictor(ctx_z)
            return pred, tgt_z

    model = MeanJEPA(vocab_size, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    losses: list[float] = []
    model.train()
    for step in range(1, args.real_steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        ctx = _pad([b[0] for b in batch], device)
        tgt = _pad([b[1] for b in batch], device)
        pred, tgt_z = model(ctx, tgt)
        cos_loss = 1.0 - F.cosine_similarity(pred, tgt_z, dim=-1).mean()
        var = pred.var(dim=0).mean()
        loss = cos_loss + 0.01 * torch.relu(torch.tensor(0.05, device=device) - var)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    def eval_split(rows: list[tuple[np.ndarray, np.ndarray]], n_eval: int = 2048) -> dict[str, float]:
        if not rows:
            return {"n": 0, "cosine": 0.0, "loss": 0.0, "effective_rank": 0.0}
        model.eval()
        sample = rows[:]
        rng.shuffle(sample)
        sample = sample[: min(n_eval, len(sample))]
        preds = []
        losses_eval = []
        cosines = []
        with torch.no_grad():
            for i in range(0, len(sample), args.batch_size):
                batch = sample[i : i + args.batch_size]
                ctx = _pad([b[0] for b in batch], device)
                tgt = _pad([b[1] for b in batch], device)
                pred, tgt_z = model(ctx, tgt)
                cos = F.cosine_similarity(pred, tgt_z, dim=-1)
                cosines.extend([float(x) for x in cos.cpu()])
                losses_eval.extend([float(x) for x in (1.0 - cos).cpu()])
                preds.append(pred.detach().cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, dim), dtype="float32")
        return {"n": len(sample), "cosine": float(np.mean(cosines)), "loss": float(np.mean(losses_eval)), "effective_rank": effective_rank(pred_arr)}

    train_eval = eval_split(train)
    dev_eval = eval_split(dev)
    ckpt_path = outdir / "minimal-jepa-v0b.pt"
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "embedding_dim": dim, "created_utc": now_utc()}, ckpt_path)

    train_manifest = {
        "schema_version": "clinical-jepa-v0b-train-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": False,
        "architecture": {"name": "mean-token-jepa-v0", "embedding_dim": dim, "predictor": "2-layer-mlp", "target_encoder": "shared_embedding_stop_gradient"},
        "n_synthetic_examples": 0,
        "n_real_examples_loaded": len(examples),
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "trained_steps": int(args.real_steps),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "leakage_audit_status": "pass",
        "final_train_loss": float(losses[-1]) if losses else None,
        "checkpoint_path": str(ckpt_path),
        "aggregate_only": True,
    }
    validate_artifact("v0b-train-manifest", train_manifest)
    write_json(outdir / "train-manifest.json", train_manifest)
    diag = {"schema_version": "clinical-jepa-v0b-collapse-diagnostics-v0", "created_utc": now_utc(), "dry_run": False, "train_eval": train_eval, "dev_eval": dev_eval, "per_dimension_variance_mean": None, "average_target_predictor_competitive": None, "aggregate_only": True}
    write_json(outdir / "collapse-diagnostics.json", diag)
    write_json(outdir / "checkpoint-manifest.json", {"created_utc": now_utc(), "checkpoint_written": True, "checkpoint_path": str(ckpt_path), "size_bytes": ckpt_path.stat().st_size, "aggregate_only": True})
    (outdir / "summary.md").write_text(f"# v0B minimal JEPA real run\n\nSteps: {args.real_steps}\n\nTrain examples: {len(train)}\n\nDev cosine: {dev_eval['cosine']:.4f}\n\nDev loss: {dev_eval['loss']:.4f}\n\nEffective rank(dev preds): {dev_eval['effective_rank']:.2f}\n")
    print(json.dumps({"train_manifest": str(outdir / "train-manifest.json"), "dev_cosine": dev_eval["cosine"], "checkpoint": str(ckpt_path)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0B minimal JEPA scaffold / real baseline")
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=50000)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--max-target-tokens", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    require_pass_leakage(args.leakage_report)
    arms = load_yaml(args.arms_config)
    dataset = load_yaml(args.dataset_config)
    targets = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    if args.dry_run:
        return _dry_run(args, arms, dataset, targets, outdir)
    return _real_run(args, arms, dataset, targets, outdir)


if __name__ == "__main__":
    sys.exit(main())
