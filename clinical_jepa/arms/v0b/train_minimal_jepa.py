from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.rung2_contract import (
    TRANSITION_META_KEY, is_fixed_width_transition_training,
)
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json
from clinical_jepa.validation import validate_artifact

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_SIZE = 438


def resolve_vocab_size(dataset: dict[str, Any]) -> int:
    """Resolve the JEPA embedding vocab size from the dataset config.

    The joint substrate has vocab_size=1050. If the config omits
    ``vocabulary.vocab_size`` the old default (438) is used but a warning is
    emitted — silently defaulting to 438 causes an embedding out-of-bounds crash
    on the joint substrate (token ids up to 1049).
    """
    vocab = dataset.get("vocabulary", {}) or {}
    vs = vocab.get("vocab_size")
    if vs:
        return int(vs)
    logger.warning(
        "dataset config has no vocabulary.vocab_size; falling back to %d. "
        "Set vocabulary.vocab_size (e.g. 1050 for the joint MIMIC+SCI-D substrate) "
        "to avoid an embedding out-of-bounds crash.",
        DEFAULT_VOCAB_SIZE,
    )
    return DEFAULT_VOCAB_SIZE


def max_token_id_in_examples(examples: list[tuple[str, np.ndarray, list[np.ndarray]]]) -> int:
    """Largest token id across all context/target windows (or -1 if empty)."""
    max_id = -1
    for _split, ctx, targets in examples:
        if len(ctx):
            max_id = max(max_id, int(np.max(ctx)))
        for t in targets:
            if len(t):
                max_id = max(max_id, int(np.max(t)))
    return max_id


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
        "architecture": {
            **arms.get("v0B_minimal_jepa", {}),
            "autoregression_mode": args.autoregression_mode,
            "horizon_count_trained": int(args.horizon_count),
            "horizon_stride_tokens": int(args.horizon_stride_tokens or args.max_target_tokens),
            "max_horizons": int(args.max_horizons or args.horizon_count),
            "max_target_tokens": int(args.max_target_tokens),
            TRANSITION_META_KEY: _transition_flag(args, encode_empty=False),
        },
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


def _read_examples(
    blocks: list[dict[str, Any]],
    max_blocks: int,
    max_context: int,
    max_target: int,
    seed: int,
    *,
    horizon_count: int = 1,
    horizon_stride_tokens: int = 0,
) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    import h5py

    from clinical_jepa.targets.block_spans import is_censored, is_empty_target

    rng = random.Random(seed)
    # Event-index rollout path: empty/censored wall-clock targets (target_start_ref
    # = -1) have no event-index rollout and must not be clamped to arr[0:] — they
    # are handled by the encode-empty path (_encode_empty_run), excluded here.
    blocks = [
        b for b in blocks
        if b.get("sequence_file") and b.get("sequence_group")
        and b.get("target_start_ref") is not None
        and not is_empty_target(b) and not is_censored(b)
    ]
    rng.shuffle(blocks)
    if max_blocks > 0:
        blocks = blocks[:max_blocks]
    cache: dict[str, Any] = {}
    examples: list[tuple[str, np.ndarray, list[np.ndarray]]] = []
    horizon_count = max(1, int(horizon_count))
    stride = int(horizon_stride_tokens) if int(horizon_stride_tokens) > 0 else int(max_target)
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
            t0 = int(b["target_start_ref"])  # guaranteed >= 0 (empties excluded above)
            if c1 < c0 or t0 >= len(arr):
                continue
            context = np.asarray(arr[c0 : c1 + 1][-max_context:], dtype=np.int64)
            targets_by_horizon: list[np.ndarray] = []
            ok = len(context) > 0
            for h in range(horizon_count):
                start_idx = t0 + h * stride
                end_idx = start_idx + max_target
                if end_idx > len(arr):
                    ok = False
                    break
                target = np.asarray(arr[start_idx:end_idx], dtype=np.int64)
                if len(target) == 0:
                    ok = False
                    break
                targets_by_horizon.append(target)
            if not ok:
                continue
            examples.append((str(b.get("split", "train")), context, targets_by_horizon))
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
    import torch.nn.functional as F

    from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA

    seed = int(dataset.get("run", {}).get("seed", 20260523))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab_size = resolve_vocab_size(dataset)
    dim = int(args.embedding_dim)

    examples = _read_examples(
        targets.get("blocks", []),
        args.max_blocks,
        args.max_context_tokens,
        args.max_target_tokens,
        seed,
        horizon_count=args.horizon_count,
        horizon_stride_tokens=args.horizon_stride_tokens,
    )
    max_id = max_token_id_in_examples(examples)
    if max_id >= vocab_size:
        raise SystemExit(
            f"Token id {max_id} exceeds vocab_size {vocab_size}; set the dataset "
            "config vocabulary.vocab_size to cover the substrate (joint = 1050)."
        )
    train = [(c, t) for split, c, t in examples if split == "train"]
    dev = [(c, t) for split, c, t in examples if split == "dev"]
    if not train:
        raise SystemExit("No train examples available for real v0B run")

    max_horizons = int(args.max_horizons) if int(args.max_horizons) > 0 else int(args.horizon_count)
    model = MeanTokenJEPA(
        vocab_size,
        dim,
        autoregression_mode=args.autoregression_mode,
        max_horizons=max_horizons,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    losses: list[float] = []
    model.train()
    for step in range(1, args.real_steps + 1):
        batch = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        ctx = _pad([b[0] for b in batch], device)
        target_steps = []
        with torch.no_grad():
            for h in range(args.horizon_count):
                tgt_h = _pad([b[1][h] for b in batch], device)
                target_steps.append(model.mean_embed(tgt_h))
        tgt_z = torch.stack(target_steps, dim=1)
        pred = model.predict_rollout_from_context_ids(ctx, args.horizon_count)
        cos_loss = 1.0 - F.cosine_similarity(pred.reshape(-1, dim), tgt_z.reshape(-1, dim), dim=-1).mean()
        var = pred.reshape(-1, dim).var(dim=0).mean()
        loss = cos_loss + 0.01 * torch.relu(torch.tensor(0.05, device=device) - var)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    def eval_split(rows: list[tuple[np.ndarray, list[np.ndarray]]], n_eval: int = 2048) -> dict[str, Any]:
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
                target_steps = []
                for h in range(args.horizon_count):
                    tgt_h = _pad([b[1][h] for b in batch], device)
                    target_steps.append(model.mean_embed(tgt_h))
                tgt_z = torch.stack(target_steps, dim=1)
                pred = model.predict_rollout_from_context_ids(ctx, args.horizon_count)
                cos = F.cosine_similarity(pred.reshape(-1, dim), tgt_z.reshape(-1, dim), dim=-1)
                cosines.extend([float(x) for x in cos.cpu()])
                losses_eval.extend([float(x) for x in (1.0 - cos).cpu()])
                preds.append(pred.reshape(-1, dim).detach().cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, dim), dtype="float32")
        return {"n": len(sample), "cosine": float(np.mean(cosines)), "loss": float(np.mean(losses_eval)), "effective_rank": effective_rank(pred_arr)}

    train_eval = eval_split(train)
    dev_eval = eval_split(dev)
    ckpt_path = outdir / "minimal-jepa-v0b.pt"
    architecture = model.architecture_metadata()
    architecture["horizon_count_trained"] = int(args.horizon_count)
    architecture["horizon_stride_tokens"] = int(args.horizon_stride_tokens or args.max_target_tokens)
    architecture["max_target_tokens"] = int(args.max_target_tokens)
    architecture[TRANSITION_META_KEY] = _transition_flag(args, encode_empty=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": dim,
        "created_utc": now_utc(),
        "architecture": architecture,
        "autoregression_mode": args.autoregression_mode,
        "horizon_count_trained": int(args.horizon_count),
        "horizon_stride_tokens": int(args.horizon_stride_tokens or args.max_target_tokens),
        "max_horizons": int(max_horizons),
        "max_target_tokens": int(args.max_target_tokens),
        TRANSITION_META_KEY: _transition_flag(args, encode_empty=False),
    }, ckpt_path)

    train_manifest = {
        "schema_version": "clinical-jepa-v0b-train-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": False,
        "architecture": architecture,
        "n_synthetic_examples": 0,
        "n_real_examples_loaded": len(examples),
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "trained_steps": int(args.real_steps),
        "batch_size": int(args.batch_size),
        "autoregression_mode": args.autoregression_mode,
        "horizon_count_trained": int(args.horizon_count),
        "horizon_stride_tokens": int(args.horizon_stride_tokens or args.max_target_tokens),
        "max_horizons": int(max_horizons),
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


def _transition_flag(args, *, encode_empty: bool) -> bool:
    """DERIVED fixed-width-non-overlapping-transition flag (Rung-2 sub-gate 1 recursive path).

    Without this on the checkpoint, `recursive_path_evaluable` returns False and the recursive-transition
    diagnostics are NOT_EVALUABLE — so a run that IS in the right regime would still be unusable. Derived,
    never asserted."""
    return is_fixed_width_transition_training(
        autoregression_mode=args.autoregression_mode,
        horizon_count=args.horizon_count,
        horizon_stride_tokens=int(args.horizon_stride_tokens or args.max_target_tokens),
        max_target_tokens=int(args.max_target_tokens),
        encode_empty=encode_empty)


def _read_encode_empty_examples(blocks, max_blocks, max_context, max_target, seed, *, source=None):
    """Read (split, ctx_ids, tgt_ids, is_empty, count) per wall-clock T0 block, via
    the shared -1-safe helper. Censored blocks are excluded (ineligible); empty
    targets carry an empty tgt + is_empty=True (routed to z_empty at train time);
    populated-but-unreadable rows are skipped. ``source`` restricts to one
    source_dataset (per-source rung-0 checkpoints)."""
    import h5py

    from clinical_jepa.targets.block_spans import (
        is_censored,
        is_empty_target,
        read_target_span,
        target_occupancy,
    )

    rng = random.Random(seed)
    rows = [b for b in blocks if b.get("sequence_file") and b.get("sequence_group") and not is_censored(b)
            and (source is None or str(b.get("source_dataset")) == str(source))]
    rng.shuffle(rows)
    if max_blocks > 0:
        rows = rows[:max_blocks]
    cache: dict[str, Any] = {}
    examples: list[tuple[str, np.ndarray, np.ndarray, bool, int]] = []
    try:
        for b in rows:
            path = str(b["sequence_file"])
            if path not in cache:
                cache[path] = h5py.File(path, "r")
            grp = cache[path].get(str(b["sequence_group"]))
            if grp is None or "token_ids" not in grp:
                continue
            arr = grp["token_ids"][:]
            c0 = max(0, int(b.get("context_start_ref", 0)))
            c1 = min(len(arr) - 1, int(b["context_end_ref"]))
            if c1 < c0:
                continue
            ctx = np.asarray(arr[c0 : c1 + 1][-max_context:], dtype=np.int64)
            if len(ctx) == 0:
                continue
            if is_empty_target(b):
                examples.append((str(b.get("split", "train")), ctx, np.asarray([], dtype=np.int64), True, 0))
            else:
                tgt_ids, _ = read_target_span(b, arr)
                tgt = np.asarray(tgt_ids[:max_target], dtype=np.int64)
                if len(tgt) == 0:
                    continue  # populated but unreadable
                _, count = target_occupancy(b)
                examples.append((str(b.get("split", "train")), ctx, tgt, False, int(count)))
    finally:
        for f in cache.values():
            f.close()
    return examples


def _encode_empty_run(args, arms, dataset, targets, outdir):
    """Encode-empty (Option A) v0B run: hybrid silence objective (Pi R4)."""
    import torch

    from clinical_jepa.arms.v0b.encode_empty_train import (
        calibration_report,
        collapse_diagnostics,
        encode_empty_loss,
        maybe_reseed_prototype,
    )
    from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA

    seed = int(dataset.get("run", {}).get("seed", 20260523))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab_size = resolve_vocab_size(dataset)
    dim = int(args.embedding_dim)

    examples = _read_encode_empty_examples(
        targets.get("blocks", []), args.max_blocks, args.max_context_tokens, args.max_target_tokens, seed,
        source=getattr(args, "source", None),
    )
    if not examples:
        raise SystemExit("No encode-empty examples (wall-clock T0 blocks) available")
    max_id = max((int(c.max()) if len(c) else -1) for _s, c, _t, _e, _n in examples)
    max_id = max(max_id, max((int(t.max()) if len(t) else -1) for _s, _c, t, _e, _n in examples))
    if max_id >= vocab_size:
        raise SystemExit(f"Token id {max_id} exceeds vocab_size {vocab_size}; set vocabulary.vocab_size")

    train = [e for e in examples if e[0] == "train"]
    dev = [e for e in examples if e[0] == "dev"]
    if not train:
        raise SystemExit("No train encode-empty examples")

    model = MeanTokenJEPA(
        vocab_size, dim, autoregression_mode=args.autoregression_mode, max_horizons=1,
        encode_empty=True, empty_prototype_seed=(args.empty_prototype_seed or None),
    ).to(device)

    # Init separation check (Pi Q2): reseed z_empty away from a non-empty sample.
    nonempty_ctx = [e[1] for e in train if not e[3]]
    sep_report: dict[str, Any] = {"skipped": True, "reason": "no non-empty train examples"}
    if nonempty_ctx:
        sample = _pad(nonempty_ctx[: min(256, len(nonempty_ctx))], device)
        sep_report = maybe_reseed_prototype(model, sample, threshold=0.15)
    proto_before = model.empty_prototype.clone()

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    emp_idx = [i for i, e in enumerate(train) if e[3]]
    pop_idx = [i for i, e in enumerate(train) if not e[3]]
    cap = float(args.empty_fraction_cap)
    B = int(args.batch_size)
    losses: list[float] = []
    model.train()
    for _step in range(1, args.real_steps + 1):
        n_emp = min(int(cap * B), len(emp_idx)) if emp_idx else 0
        n_pop = B - n_emp
        idx = ([rng.choice(emp_idx) for _ in range(n_emp)] if emp_idx else []) + (
            [rng.choice(pop_idx) for _ in range(n_pop)] if pop_idx else []
        )
        if not idx:
            break
        batch = [train[i] for i in idx]
        ctx = _pad([b[1] for b in batch], device)
        tgt = _pad([b[2] if len(b[2]) else np.zeros(1, dtype=np.int64) for b in batch], device)
        is_empty = torch.tensor([b[3] for b in batch], dtype=torch.bool, device=device)
        count = torch.tensor([b[4] for b in batch], dtype=torch.long, device=device)
        loss, parts = encode_empty_loss(model, ctx, tgt, is_empty, count)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(parts["total"])

    # Eval + calibration on dev at NATURAL prevalence (no cap) — Pi Q5.
    model.eval()
    diag_metrics: dict[str, Any] = {}
    calib: dict[str, Any] = {}
    empty_fraction_natural = None
    if dev:
        sample = dev[: min(4096, len(dev))]
        dctx = _pad([r[1] for r in sample], device)
        die = torch.tensor([r[3] for r in sample], dtype=torch.bool, device=device)
        dcn = torch.tensor([r[4] for r in sample], dtype=torch.long, device=device)
        diag_metrics = collapse_diagnostics(model, dctx, die, dcn)
        with torch.no_grad():
            occ_logit, _ = model.predict_occupancy_from_latent(model.mean_embed(dctx), 1)
            occ_prob = torch.sigmoid(occ_logit[:, 0, 0])
            y_occ = (~die.to(torch.bool)).float()
        calib = calibration_report(occ_prob, y_occ)
        empty_fraction_natural = float(die.float().mean())

    prototype_unmoved = bool(torch.equal(model.empty_prototype, proto_before))
    architecture = model.architecture_metadata()
    ckpt_path = outdir / "minimal-jepa-v0b-encode-empty.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "vocab_size": vocab_size, "embedding_dim": dim,
        "created_utc": now_utc(), "architecture": architecture, "autoregression_mode": args.autoregression_mode,
        "encode_empty": True, "max_horizons": 1,
        # encode-empty pins horizon_count to 1, so it can never be a fixed-width TRANSITION regime.
        # Stamped explicitly so the recursive path refuses on substance, not on a missing field.
        "horizon_count_trained": 1, "max_target_tokens": int(args.max_target_tokens),
        TRANSITION_META_KEY: _transition_flag(args, encode_empty=True),
    }, ckpt_path)

    n_train_empty = len(emp_idx)
    train_manifest = {
        "schema_version": "clinical-jepa-v0b-train-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": False,
        "architecture": {**architecture, "encode_empty": True, "empty_fraction_cap": cap,
                         "horizon_count_trained": 1, "max_target_tokens": int(args.max_target_tokens),
                         TRANSITION_META_KEY: _transition_flag(args, encode_empty=True)},
        "n_synthetic_examples": 0,
        "n_real_examples_loaded": len(examples),
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "n_train_empty": n_train_empty,
        "trained_steps": int(len(losses)),
        "batch_size": B,
        "autoregression_mode": args.autoregression_mode,
        "horizon_count_trained": 1,
        "horizon_stride_tokens": 0,
        "max_horizons": 1,
        "device": str(device),
        "leakage_audit_status": "pass",
        "final_train_loss": float(losses[-1]) if losses else None,
        "checkpoint_path": str(ckpt_path),
        "aggregate_only": True,
    }
    validate_artifact("v0b-train-manifest", train_manifest)
    write_json(outdir / "train-manifest.json", train_manifest)
    write_json(outdir / "collapse-diagnostics.json", {
        "schema_version": "clinical-jepa-v0b-collapse-diagnostics-v0",
        "created_utc": now_utc(), "dry_run": False, "mode": "encode_empty",
        "separation_check": sep_report,
        "prototype_unmoved_during_training": prototype_unmoved,
        "empty_fraction_train_capped": cap,
        "empty_fraction_dev_natural": empty_fraction_natural,
        "collapse_diagnostics": diag_metrics,
        "calibration_natural_prevalence": calib,
        "aggregate_only": True,
    })
    write_json(outdir / "checkpoint-manifest.json", {
        "created_utc": now_utc(), "checkpoint_written": True, "checkpoint_path": str(ckpt_path),
        "size_bytes": ckpt_path.stat().st_size, "aggregate_only": True,
    })
    print(json.dumps({
        "train_manifest": str(outdir / "train-manifest.json"),
        "occupancy_auc": diag_metrics.get("occupancy_auc"),
        "empty_vs_populated_margin": diag_metrics.get("empty_vs_populated_margin"),
        "prototype_unmoved": prototype_unmoved,
        "checkpoint": str(ckpt_path),
    }, indent=2))
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
    ap.add_argument("--autoregression-mode", choices=("recursive", "horizon_conditioned"), default="recursive", help="recursive keeps v0B backward compatibility; horizon_conditioned uses explicit horizon-specific heads")
    ap.add_argument("--horizon-count", type=int, default=1, help="Number of future target windows to train against")
    ap.add_argument("--horizon-stride-tokens", type=int, default=0, help="Stride between target windows; defaults to max-target-tokens")
    ap.add_argument("--max-horizons", type=int, default=0, help="Capacity for horizon-conditioned heads; defaults to horizon-count")
    ap.add_argument("--max-blocks", type=int, default=50000)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--max-target-tokens", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--encode-empty", action="store_true", help="Encode-empty (Option A): silence as a first-class latent via a frozen z_empty prototype + occupancy/count heads (Pi R4). For wall-clock T0 blocks; horizon_count pinned to 1.")
    ap.add_argument("--source", default=None, help="Restrict encode-empty training to one source_dataset (per-source rung-0 checkpoints)")
    ap.add_argument("--empty-fraction-cap", type=float, default=0.5, help="Max empty fraction per training batch (balance; the natural prior is restored via calibration on uncapped dev)")
    ap.add_argument("--empty-prototype-seed", type=int, default=0, help="Seed for the frozen z_empty prototype (0 = model default; the init separation check may reseed)")
    args = ap.parse_args(argv)
    if args.horizon_count <= 0:
        raise SystemExit("--horizon-count must be positive")
    if args.autoregression_mode == "horizon_conditioned" and args.max_horizons and args.horizon_count > args.max_horizons:
        raise SystemExit("--horizon-count cannot exceed --max-horizons for horizon_conditioned mode")
    require_pass_leakage(args.leakage_report)
    arms = load_yaml(args.arms_config)
    dataset = load_yaml(args.dataset_config)
    targets = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    if args.dry_run:
        return _dry_run(args, arms, dataset, targets, outdir)
    if args.encode_empty:
        return _encode_empty_run(args, arms, dataset, targets, outdir)
    return _real_run(args, arms, dataset, targets, outdir)


if __name__ == "__main__":
    sys.exit(main())
