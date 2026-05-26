from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, write_json

DEFAULT_EVENT_TYPES = ("MED", "LAB", "STATE")


@dataclass(frozen=True)
class FutureSummarySpec:
    """Safe-public coded-event future-summary specification.

    This is a readout/speaker scaffold: it summarizes observed coded-event
    windows into aggregate family targets. It is not an event generator,
    renderer, treatment-effect estimator, or clinical decision model.
    """

    spec_id: str = "coded_event_future_summary_v0"
    event_types: tuple[str, ...] = DEFAULT_EVENT_TYPES
    label_depth: int = 2
    top_k: int = 5
    min_label_count: int = 1

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FutureSummarySpec":
        spec_id = str(row.get("spec_id", "coded_event_future_summary_v0")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec_id):
            raise ValueError(f"Invalid spec_id {spec_id!r}")
        event_types = tuple(str(x).upper() for x in row.get("event_types", DEFAULT_EVENT_TYPES))
        if not event_types:
            raise ValueError("event_types cannot be empty")
        label_depth = int(row.get("label_depth", 2))
        if label_depth < 1:
            raise ValueError("label_depth must be >= 1")
        top_k = int(row.get("top_k", 5))
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        min_label_count = int(row.get("min_label_count", 1))
        if min_label_count < 1:
            raise ValueError("min_label_count must be >= 1")
        return cls(spec_id=spec_id, event_types=event_types, label_depth=label_depth, top_k=top_k, min_label_count=min_label_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "event_types": list(self.event_types),
            "label_depth": int(self.label_depth),
            "top_k": int(self.top_k),
            "min_label_count": int(self.min_label_count),
        }


def load_future_summary_spec(path: str | Path | None = None) -> FutureSummarySpec:
    if path is None:
        return FutureSummarySpec()
    p = Path(path)
    data = load_yaml(p) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
    row = data.get("future_summary", data)
    if not isinstance(row, dict):
        raise ValueError("Future summary config must be an object or contain a future_summary object")
    return FutureSummarySpec.from_dict(row)


def event_type(token: str) -> str:
    text = str(token).strip()
    if not text:
        return "UNKNOWN"
    return text.split(":", 1)[0].upper()


def event_label(token: str, *, label_depth: int = 2) -> str:
    parts = [p.strip().upper() for p in str(token).split(":") if p.strip()]
    if not parts:
        return "UNKNOWN"
    return ":".join(parts[: max(1, int(label_depth))])


def summarize_events(events: Iterable[str], spec: FutureSummarySpec) -> dict[str, Any]:
    counts = {family: 0 for family in spec.event_types}
    label_counts: dict[str, int] = {}
    total = 0
    for token in events:
        typ = event_type(str(token))
        if typ not in counts:
            continue
        total += 1
        counts[typ] += 1
        label = event_label(str(token), label_depth=spec.label_depth)
        label_counts[label] = label_counts.get(label, 0) + 1
    denom = float(total) if total else 1.0
    distribution = {family: counts[family] / denom for family in spec.event_types}
    return {
        "event_count": int(total),
        "type_counts": counts,
        "type_distribution": distribution,
        "label_counts": dict(sorted(label_counts.items())),
        "label_presence": sorted(label_counts),
    }


def label_vocabulary(target_summaries: list[dict[str, Any]], *, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for summary in target_summaries:
        for label in summary.get("label_presence", []):
            counts[str(label)] = counts.get(str(label), 0) + 1
    return sorted(label for label, count in counts.items() if count >= min_count)


def multilabel_matrix(summaries: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    y = np.zeros((len(summaries), len(labels)), dtype=np.float32)
    for r, summary in enumerate(summaries):
        for label in summary.get("label_presence", []):
            if label in index:
                y[r, index[label]] = 1.0
    return y


def distribution_matrix(summaries: list[dict[str, Any]], event_types: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((len(summaries), len(event_types)), dtype=np.float32)
    for r, summary in enumerate(summaries):
        dist = summary.get("type_distribution", {})
        for c, family in enumerate(event_types):
            out[r, c] = float(dist.get(family, 0.0))
    return out


def empirical_prior_scores(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    if y_train.size == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    prior = y_train.mean(axis=0, keepdims=True)
    return np.repeat(prior, n_rows, axis=0).astype(np.float32)


def context_presence_scores(context_summaries: list[dict[str, Any]], labels: list[str], *, fallback_prior: np.ndarray | None = None) -> np.ndarray:
    scores = multilabel_matrix(context_summaries, labels)
    if fallback_prior is not None and fallback_prior.size:
        prior = np.asarray(fallback_prior, dtype=np.float32).reshape(1, -1)
        scores = np.maximum(scores, 0.25 * prior)
    return scores.astype(np.float32)


def context_distribution_scores(context_summaries: list[dict[str, Any]], event_types: tuple[str, ...]) -> np.ndarray:
    return distribution_matrix(context_summaries, event_types)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float32)
    s = np.asarray(y_score, dtype=np.float32)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    denom = np.arange(1, len(y_sorted) + 1, dtype=np.float32)
    precision = tp / denom
    return float((precision * y_sorted).sum() / positives)


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(present)) if present else None


def evaluate_multilabel(y_true: np.ndarray, y_score: np.ndarray, labels: list[str], *, top_k: int = 5) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float32)
    scores = np.asarray(y_score, dtype=np.float32)
    if y.shape != scores.shape:
        raise ValueError(f"y_true and y_score shapes differ: {y.shape} vs {scores.shape}")
    if y.shape[1] != len(labels):
        raise ValueError("label count does not match matrix width")
    ap_values: list[float | None] = []
    for i, _label in enumerate(labels):
        ap_values.append(_average_precision(y[:, i], scores[:, i]))
    micro_ap = _average_precision(y.reshape(-1), scores.reshape(-1)) if y.size else None
    k = min(max(1, int(top_k)), y.shape[1]) if y.shape[1] else 0
    recalls: list[float] = []
    precisions: list[float] = []
    hit_rows = 0
    if k:
        order = np.argsort(-scores, axis=1, kind="mergesort")[:, :k]
        for r in range(y.shape[0]):
            positives = float(y[r].sum())
            hits = float(y[r, order[r]].sum())
            if positives > 0:
                recalls.append(hits / positives)
                hit_rows += int(hits > 0)
            precisions.append(hits / k)
    return {
        "n_rows": int(y.shape[0]),
        "n_labels": int(y.shape[1]),
        "top_k": int(k),
        "macro_average_precision": _mean_present(ap_values),
        "micro_average_precision": micro_ap,
        "label_average_precision_summary": {
            "min": float(np.min([v for v in ap_values if v is not None])) if any(v is not None for v in ap_values) else None,
            "median": float(np.median([v for v in ap_values if v is not None])) if any(v is not None for v in ap_values) else None,
            "max": float(np.max([v for v in ap_values if v is not None])) if any(v is not None for v in ap_values) else None,
        },
        "top_k_recall_mean": float(np.mean(recalls)) if recalls else None,
        "top_k_precision_mean": float(np.mean(precisions)) if precisions else None,
        "top_k_any_hit_rate": float(hit_rows / len(recalls)) if recalls else None,
    }


def evaluate_distribution(target_distribution: np.ndarray, predicted_distribution: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target_distribution, dtype=np.float32)
    pred = np.asarray(predicted_distribution, dtype=np.float32)
    if target.shape != pred.shape:
        raise ValueError(f"distribution shapes differ: {target.shape} vs {pred.shape}")
    l1 = np.abs(target - pred).sum(axis=1)
    denom = np.linalg.norm(target, axis=1) * np.linalg.norm(pred, axis=1)
    denom = np.where(denom < 1e-8, 1.0, denom)
    cosine = (target * pred).sum(axis=1) / denom
    return {
        "n_rows": int(target.shape[0]),
        "l1_mean": float(np.mean(l1)) if len(l1) else 0.0,
        "l1_median": float(np.median(l1)) if len(l1) else 0.0,
        "cosine_mean": float(np.mean(cosine)) if len(cosine) else 0.0,
        "cosine_median": float(np.median(cosine)) if len(cosine) else 0.0,
    }


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    rows = data.get("rows", data if isinstance(data, list) else None)
    if not isinstance(rows, list):
        raise ValueError("Input must be a list or an object with a rows list")
    return rows


def _events(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of synthetic/pre-extracted coded event strings")
    return [str(x) for x in value]


def build_future_summary_report(rows: list[dict[str, Any]], spec: FutureSummarySpec, *, scenario_id: str = "coded_event_future_summary") -> dict[str, Any]:
    context = [summarize_events(_events(row, "context_events"), spec) for row in rows]
    target = [summarize_events(_events(row, "target_events"), spec) for row in rows]
    labels = label_vocabulary(target, min_count=spec.min_label_count)
    y = multilabel_matrix(target, labels)
    prior_scores = empirical_prior_scores(y, len(rows))
    prior = y.mean(axis=0) if y.size else np.zeros((0,), dtype=np.float32)
    context_scores = context_presence_scores(context, labels, fallback_prior=prior)
    target_dist = distribution_matrix(target, spec.event_types)
    prior_dist = np.repeat(target_dist.mean(axis=0, keepdims=True), len(rows), axis=0) if len(rows) else np.zeros((0, len(spec.event_types)), dtype=np.float32)
    context_dist = context_distribution_scores(context, spec.event_types)
    report = {
        "schema_version": "clinical-jepa-future-summary-readout-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "spec": spec.to_dict(),
        "n_rows": int(len(rows)),
        "n_labels": int(len(labels)),
        "label_names_suppressed": True,
        "target_event_count_mean": float(np.mean([s["event_count"] for s in target])) if target else 0.0,
        "context_event_count_mean": float(np.mean([s["event_count"] for s in context])) if context else 0.0,
        "baselines": {
            "empirical_prior_presence": evaluate_multilabel(y, prior_scores, labels, top_k=spec.top_k),
            "context_presence_copy": evaluate_multilabel(y, context_scores, labels, top_k=spec.top_k),
            "empirical_prior_distribution": evaluate_distribution(target_dist, prior_dist),
            "context_distribution_copy": evaluate_distribution(target_dist, context_dist),
        },
        "bridge_contract": bridge_contract(spec),
        "notes": "Aggregate-only coded-event future-summary/readout scaffold. Inputs may be synthetic or reviewed local pre-extracted coded-event summaries; outputs contain no row IDs, patient IDs, raw tokens, examples, generated sequences, or clinical/treatment claims.",
    }
    return report


def bridge_contract(spec: FutureSummarySpec) -> dict[str, Any]:
    return {
        "contract_id": "flatascend_compatible_coded_summary_v0",
        "not_generation": True,
        "input": "prefix/context latent or coded-event summary vector",
        "output": "aggregate/interpretable future summary scores, not event sequences",
        "event_types": list(spec.event_types),
        "label_depth": int(spec.label_depth),
        "supported_metrics": ["type_distribution_l1", "type_distribution_cosine", "multi_label_average_precision", "top_k_recall"],
        "local_extraction_required": "reviewed governed local pre-extraction for real data; no HDF5/checkpoint/sidecar paths in public artifacts",
    }


def command_plan(spec: FutureSummarySpec, *, output_root_placeholder: str = "<LOCAL_OUTPUT_ROOT>") -> str:
    return "\n".join([
        "# Placeholder-only BP007 local command plan",
        "# Replace placeholders only in reviewed local governed context; do not commit local paths or row-level outputs.",
        "python -m clinical_jepa.speaker.future_summary \\",
        "  --spec-config configs/v0/future_summary.example.yaml \\",
        "  --input-json <LOCAL_PREEXTRACTED_CODED_EVENT_SUMMARY_ROWS.json> \\",
        f"  --output-dir {output_root_placeholder}/future-summary-readout \\",
        "  --scenario-id bp007_local_coded_event_summary",
        "",
        "# Expected local input rows are pre-extracted coded-event summaries with context_events and target_events lists.",
        "# This CLI emits aggregate metrics only and does not render/generate event sequences.",
    ])


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA coded-event future-summary readout",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Rows: {report['n_rows']}",
        f"Labels: {report['n_labels']}",
        "",
        "This is a coded-event summary/readout scaffold, not event generation, clinical utility, or treatment-effect estimation.",
        "",
        "## Baselines",
        "",
        "| Baseline | Macro AP | Micro AP | Top-k recall | Top-k precision | Distribution L1 | Distribution cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baselines = report["baselines"]
    for label, presence_key, distribution_key in (
        ("empirical_prior", "empirical_prior_presence", "empirical_prior_distribution"),
        ("context_presence_copy", "context_presence_copy", "context_distribution_copy"),
    ):
        presence = baselines[presence_key]
        distribution = baselines[distribution_key]
        lines.append(
            f"| `{label}` | {_fmt(presence.get('macro_average_precision'))} | {_fmt(presence.get('micro_average_precision'))} | {_fmt(presence.get('top_k_recall_mean'))} | {_fmt(presence.get('top_k_precision_mean'))} | {_fmt(distribution.get('l1_mean'))} | {_fmt(distribution.get('cosine_mean'))} |"
        )
    lines.extend([
        "",
        "## Bridge boundary",
        "",
        "- Outputs are future-summary scores only, not generated event sequences.",
        "- Real-data use requires reviewed local pre-extraction and aggregate-only sync-back.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe-public coded-event future-summary readout scaffold")
    ap.add_argument("--spec-config", help="YAML/JSON future_summary config; defaults to coded_event_future_summary_v0")
    ap.add_argument("--input-json", help="Synthetic or reviewed local pre-extracted rows with context_events and target_events")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="coded_event_future_summary")
    ap.add_argument("--emit-command-plan", action="store_true")
    args = ap.parse_args(argv)

    spec = load_future_summary_spec(args.spec_config)
    outdir = ensure_dir(args.output_dir)
    if args.input_json:
        report = build_future_summary_report(_load_rows(args.input_json), spec, scenario_id=args.scenario_id)
    else:
        report = {
            "schema_version": "clinical-jepa-future-summary-readout-v0",
            "created_utc": now_utc(),
            "scenario_id": args.scenario_id,
            "aggregate_only": True,
            "spec": spec.to_dict(),
            "n_rows": 0,
            "n_labels": 0,
            "status": "planned_not_evaluated",
            "bridge_contract": bridge_contract(spec),
            "notes": "Command-plan/scaffold only; no input rows evaluated.",
        }
    write_json(outdir / "future-summary-readout.json", report)
    if "baselines" in report:
        (outdir / "summary.md").write_text(_summary_md(report))
    if args.emit_command_plan or not args.input_json:
        (outdir / "command-plan.md").write_text(command_plan(spec))
    print(json.dumps({"output": str(outdir / "future-summary-readout.json"), "n_rows": report.get("n_rows", 0), "n_labels": report.get("n_labels", 0)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
