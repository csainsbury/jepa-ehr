#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from clinical_jepa.arms.v0a.train_predictor import _evaluate_task, _load_id_to_token, _read_target_labels
from clinical_jepa.eval.retrieval import compute_retrieval_metrics


class MeanJEPA(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d, padding_idx=0)
        self.predictor = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def mean_embed(self, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        emb = self.embedding(ids) * mask
        return emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, ctx):
        return self.predictor(self.mean_embed(ctx))


def pad(batch: list[np.ndarray], device: torch.device) -> torch.Tensor:
    max_len = max(len(x) for x in batch)
    out = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    for i, arr in enumerate(batch):
        out[i, : len(arr)] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return out


def token_family_counts(ids: np.ndarray, id_to_token: dict[int, str]) -> dict[str, int]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate v0B learned prediction embeddings with aggregate probes/retrieval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-blocks", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--max-target-tokens", type=int, default=32)
    ap.add_argument("--retrieval-max-candidates", type=int, default=4096)
    ap.add_argument("--retrieval-policy", default="same_split_target_type", choices=["same_split", "same_split_target_type", "same_split_target_type_len_bin", "same_split_target_type_len_seq_util_bin"])
    args = ap.parse_args()

    # YAML is only needed for the vocabulary path; avoid logging config contents.
    import yaml

    cfg = yaml.safe_load(Path(args.dataset_config).read_text())
    vocab_json = cfg.get("vocabulary", {}).get("vocab_json_path")
    id_to_token = _load_id_to_token(vocab_json)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = MeanJEPA(int(ckpt["vocab_size"]), int(ckpt["embedding_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    target_manifest = json.loads(Path(args.target_blocks).read_text())
    blocks = [b for b in target_manifest.get("blocks", []) if b.get("target_type") == "T0" and b.get("sequence_file")]
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]

    rows: list[dict] = []
    dim = int(ckpt["embedding_dim"])
    context_pred = np.zeros((len(blocks), dim), dtype=np.float16)
    target_mean = np.zeros((len(blocks), dim), dtype=np.float16)
    cache: dict[str, h5py.File] = {}

    def read_pair(block: dict) -> tuple[np.ndarray, np.ndarray, int]:
        path = str(block["sequence_file"])
        if path not in cache:
            cache[path] = h5py.File(path, "r")
        h5 = cache[path]
        group = str(block.get("sequence_group") or block.get("sequence_id"))
        arr = h5[group]["token_ids"][:]
        c0 = max(0, int(block.get("context_start_ref", 0)))
        c1 = min(len(arr) - 1, int(block["context_end_ref"]))
        t0 = max(0, int(block["target_start_ref"]))
        t1 = min(len(arr) - 1, int(block["target_end_ref"]))
        context = np.asarray(arr[c0 : c1 + 1][-args.max_context_tokens :], dtype=np.int64)
        target = np.asarray(arr[t0 : t1 + 1][: args.max_target_tokens], dtype=np.int64)
        return context, target, int(len(arr))

    try:
        with torch.no_grad():
            for start in range(0, len(blocks), args.batch_size):
                batch_blocks = blocks[start : start + args.batch_size]
                pairs = [read_pair(b) for b in batch_blocks]
                ctx_ids = [p[0] for p in pairs]
                tgt_ids = [p[1] for p in pairs]
                seq_lens = [p[2] for p in pairs]
                context_pred[start : start + len(batch_blocks)] = model(pad(ctx_ids, device)).detach().cpu().numpy().astype(np.float16)
                target_mean[start : start + len(batch_blocks)] = model.mean_embed(pad(tgt_ids, device)).detach().cpu().numpy().astype(np.float16)
                for j, block in enumerate(batch_blocks):
                    context_counts = token_family_counts(ctx_ids[j], id_to_token)
                    target_counts = token_family_counts(tgt_ids[j], id_to_token)
                    rows.append(
                        {
                            "row": start + j,
                            "block_id": block.get("block_id"),
                            "split": block.get("split"),
                            "target_type": block.get("target_type"),
                            "horizon_descriptor": block.get("horizon_descriptor"),
                            "gap_events": block.get("gap_events"),
                            "context_len": int(len(ctx_ids[j])),
                            "target_len": int(len(tgt_ids[j])),
                            "sequence_len": int(seq_lens[j]),
                            "context_med_count": int(context_counts["med"]),
                            "context_lab_count": int(context_counts["lab"]),
                            "context_state_count": int(context_counts["state"]),
                            "target_med_count": int(target_counts["med"]),
                            "target_lab_count": int(target_counts["lab"]),
                            "target_state_count": int(target_counts["state"]),
                            "source_dataset": block.get("source_dataset"),
                        }
                    )
    finally:
        for f in cache.values():
            f.close()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "v0b-context-pred.fp16.npy", context_pred)
    np.save(out / "v0b-target-mean.fp16.npy", target_mean)
    (out / "embedding-index.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")

    labels = _read_target_labels({str(r["block_id"]) for r in rows}, target_manifest, id_to_token)
    metrics = []
    x = context_pred.astype(np.float32)
    metrics.extend(_evaluate_task(x, rows, labels, "next_med_family", "med", "v0b_pred"))
    metrics.extend(_evaluate_task(x, rows, labels, "next_lab_family", "lab", "v0b_pred"))
    metrics.extend(_evaluate_task(x, rows, labels, "next_state_family", "state", "v0b_pred"))
    retrieval = compute_retrieval_metrics(
        context_pred,
        rows,
        target_mean,
        rows,
        distractor_policy=args.retrieval_policy,
        max_candidates_per_group=args.retrieval_max_candidates,
    )

    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "embedding_shape": list(context_pred.shape),
        "target_embedding_shape": list(target_mean.shape),
        "n_labeled_blocks": len(labels),
        "metrics": metrics,
        "retrieval": retrieval,
        "aggregate_only": True,
    }
    (out / "v0b-probe-results.json").write_text(json.dumps(report, indent=2))
    (out / "retrieval-summary.md").write_text(
        "\n".join(
            [
                "# v0B target retrieval",
                "",
                f"Distractor policy: {retrieval['distractor_policy']}",
                f"Queries: {retrieval['overall']['n']}",
                f"Recall@1: {retrieval['overall']['recall_at_1']:.4f}",
                f"Recall@5: {retrieval['overall']['recall_at_5']:.4f}",
                f"Recall@10: {retrieval['overall']['recall_at_10']:.4f}",
                f"MRR: {retrieval['overall']['mrr']:.4f}",
                f"Median rank: {retrieval['overall']['median_rank']}",
            ]
        )
        + "\n"
    )
    lines = ["# v0B learned-prediction embedding probes", "", f"Embedding rows: {len(rows)}", f"Labeled T0 blocks: {len(labels)}", ""]
    for metric in metrics:
        if metric.get("baseline") == "ridge_linear_probe":
            lines.append(
                f"- {metric['task']} / {metric['split']}: top1={metric['top1_accuracy']:.3f}, n={metric['n_evaluated']}, classes={metric['n_classes_train']}"
            )
    lines.extend(["", "## Retrieval", "", f"- Policy: {retrieval['distractor_policy']}", f"- Recall@10: {retrieval['overall']['recall_at_10']:.4f}", f"- MRR: {retrieval['overall']['mrr']:.4f}"])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out / "v0b-probe-results.json"), "n_metrics": len(metrics), "retrieval_queries": retrieval["overall"]["n"]}, indent=2))


if __name__ == "__main__":
    main()
