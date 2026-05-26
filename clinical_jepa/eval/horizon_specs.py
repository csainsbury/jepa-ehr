from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, write_json


DEFAULT_EVENT_HORIZON_SPECS: tuple[dict[str, Any], ...] = (
    {
        "spec_id": "event32_stride32",
        "kind": "event_count",
        "target_window_events": 32,
        "horizon_stride_events": 32,
        "horizon_count": 9,
        "rationale": "BP004 baseline; adjacent windows were highly similar.",
    },
    {
        "spec_id": "event32_stride64",
        "kind": "event_count",
        "target_window_events": 32,
        "horizon_stride_events": 64,
        "horizon_count": 5,
        "rationale": "Same target resolution with larger separation between horizons.",
    },
    {
        "spec_id": "event32_stride128",
        "kind": "event_count",
        "target_window_events": 32,
        "horizon_stride_events": 128,
        "horizon_count": 3,
        "rationale": "Stronger temporal separation while preserving a short target window.",
    },
    {
        "spec_id": "event64_stride64",
        "kind": "event_count",
        "target_window_events": 64,
        "horizon_stride_events": 64,
        "horizon_count": 5,
        "rationale": "Smoother target state with moderate separation.",
    },
    {
        "spec_id": "event64_stride128",
        "kind": "event_count",
        "target_window_events": 64,
        "horizon_stride_events": 128,
        "horizon_count": 3,
        "rationale": "Smoother target state with stronger separation.",
    },
)


@dataclass(frozen=True)
class HorizonSpec:
    spec_id: str
    kind: str
    target_window_events: int | None = None
    horizon_stride_events: int | None = None
    horizon_count: int = 1
    target_window_hours: float | None = None
    horizon_stride_hours: float | None = None
    rationale: str = ""
    diagnostic_distances: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "HorizonSpec":
        spec_id = str(row.get("spec_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec_id):
            raise ValueError(f"Invalid spec_id {spec_id!r}; use only letters, numbers, '.', '_' or '-'")
        kind = str(row.get("kind", "event_count")).strip()
        if kind not in {"event_count", "time_hours"}:
            raise ValueError(f"Unsupported horizon spec kind {kind!r}; expected 'event_count' or 'time_hours'")
        horizon_count = int(row.get("horizon_count", 1))
        if horizon_count < 2:
            raise ValueError(f"{spec_id}: horizon_count must be at least 2 for horizon diagnostics")
        target_window_events = _optional_positive_int(row.get("target_window_events"), f"{spec_id}.target_window_events")
        horizon_stride_events = _optional_positive_int(row.get("horizon_stride_events"), f"{spec_id}.horizon_stride_events")
        target_window_hours = _optional_positive_float(row.get("target_window_hours"), f"{spec_id}.target_window_hours")
        horizon_stride_hours = _optional_positive_float(row.get("horizon_stride_hours"), f"{spec_id}.horizon_stride_hours")
        if kind == "event_count" and (target_window_events is None or horizon_stride_events is None):
            raise ValueError(f"{spec_id}: event_count specs require target_window_events and horizon_stride_events")
        if kind == "time_hours" and (target_window_hours is None or horizon_stride_hours is None):
            raise ValueError(f"{spec_id}: time_hours specs require target_window_hours and horizon_stride_hours")
        distances = tuple(sorted({int(x) for x in row.get("diagnostic_distances", []) if int(x) > 0}))
        if not distances:
            distances = default_diagnostic_distances(horizon_count)
        distances = tuple(d for d in distances if d < horizon_count)
        if not distances:
            raise ValueError(f"{spec_id}: no diagnostic_distances are < horizon_count")
        return cls(
            spec_id=spec_id,
            kind=kind,
            target_window_events=target_window_events,
            horizon_stride_events=horizon_stride_events,
            horizon_count=horizon_count,
            target_window_hours=target_window_hours,
            horizon_stride_hours=horizon_stride_hours,
            rationale=str(row.get("rationale", "")),
            diagnostic_distances=distances,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "spec_id": self.spec_id,
            "kind": self.kind,
            "horizon_count": int(self.horizon_count),
            "diagnostic_distances": [int(x) for x in self.diagnostic_distances],
            "rationale": self.rationale,
        }
        if self.target_window_events is not None:
            out["target_window_events"] = int(self.target_window_events)
        if self.horizon_stride_events is not None:
            out["horizon_stride_events"] = int(self.horizon_stride_events)
        if self.target_window_hours is not None:
            out["target_window_hours"] = float(self.target_window_hours)
        if self.horizon_stride_hours is not None:
            out["horizon_stride_hours"] = float(self.horizon_stride_hours)
        return out

    def export_rollout_command(self, *, output_root_placeholder: str = "<LOCAL_OUTPUT_ROOT>") -> str | None:
        if self.kind != "event_count":
            return None
        return " ".join([
            "python -m clinical_jepa.eval.export_mean_token_rollouts",
            "--checkpoint <LOCAL_CHECKPOINT>",
            "--target-blocks <LOCAL_TARGET_BLOCK_MANIFEST>",
            f"--output-dir {output_root_placeholder}/{self.spec_id}",
            "--splits dev test",
            "--target-types T0",
            "--max-blocks <MAX_BLOCKS>",
            f"--target-window-events {self.target_window_events}",
            f"--horizon-stride-events {self.horizon_stride_events}",
            f"--horizon-count {self.horizon_count}",
            "--batch-size <BATCH_SIZE>",
        ])

    def readiness_command(self, *, output_root_placeholder: str = "<LOCAL_OUTPUT_ROOT>") -> str:
        distances = " ".join(str(x) for x in self.diagnostic_distances)
        spec_root = f"{output_root_placeholder}/{self.spec_id}"
        return " ".join([
            "python -m clinical_jepa.eval.autoregression_readiness",
            f"--predicted-rollout {spec_root}/predicted-rollout.fp16.npy",
            f"--target-rollout {spec_root}/observed-rollout.fp16.npy",
            f"--index {spec_root}/rollout-index.local.jsonl",
            f"--output-dir {spec_root}/readiness-forward-distances",
            "--distractor-policy same_split_target_type_len_seq_util_bin",
            "--control-mode all",
            "--time-shift-mode noncyclic_forward",
            f"--time-shift-distances {distances}",
        ])


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def default_diagnostic_distances(horizon_count: int) -> tuple[int, ...]:
    distances = [1]
    value = 2
    while value < horizon_count:
        distances.append(value)
        value *= 2
    return tuple(d for d in distances if d < horizon_count)


def default_event_horizon_specs() -> list[HorizonSpec]:
    return [HorizonSpec.from_dict(dict(row)) for row in DEFAULT_EVENT_HORIZON_SPECS]


def load_horizon_specs(path: str | Path | None = None) -> list[HorizonSpec]:
    if path is None:
        return default_event_horizon_specs()
    p = Path(path)
    data = load_yaml(p) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
    rows = data.get("horizon_specs", data if isinstance(data, list) else None)
    if not isinstance(rows, list):
        raise ValueError("Horizon spec config must be a list or an object with a 'horizon_specs' list")
    specs = [HorizonSpec.from_dict(row) for row in rows]
    ids = [spec.spec_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate horizon spec_id values are not allowed")
    return specs


def _as_rollout(x: np.ndarray, *, spec_id: str) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"{spec_id}: target rollout must have shape (n, horizon, dim); got {x.shape}")
    if x.shape[0] == 0 or x.shape[1] < 2 or x.shape[2] == 0:
        raise ValueError(f"{spec_id}: target rollout must have n>0, horizon>=2, dim>0; got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{spec_id}: target rollout contains non-finite values")
    return x.astype(np.float32, copy=False)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return np.sum(a * b, axis=1) / denom


def target_horizon_similarity(target_rollout: np.ndarray, *, distances: Iterable[int]) -> dict[str, Any]:
    target = target_rollout.astype(np.float32, copy=False)
    horizons = int(target.shape[1])
    per_distance: list[dict[str, Any]] = []
    for distance in sorted({int(x) for x in distances if int(x) > 0}):
        if distance >= horizons:
            per_distance.append({
                "distance": int(distance),
                "n_horizon_pairs": 0,
                "cosine_mean_over_pairs": None,
                "cosine_median_over_pairs": None,
                "l2_mean_over_pairs": None,
                "mae_mean_over_pairs": None,
                "per_pair": [],
            })
            continue
        per_pair: list[dict[str, Any]] = []
        for h in range(horizons - distance):
            a = target[:, h, :]
            b = target[:, h + distance, :]
            cos = _safe_cosine(a, b)
            l2 = np.linalg.norm(a - b, axis=1)
            mae = np.mean(np.abs(a - b), axis=1)
            per_pair.append({
                "from_horizon_index": int(h),
                "to_horizon_index": int(h + distance),
                "cosine_mean": float(np.mean(cos)),
                "cosine_median": float(np.median(cos)),
                "l2_mean": float(np.mean(l2)),
                "mae_mean": float(np.mean(mae)),
            })
        cosine_means = [row["cosine_mean"] for row in per_pair]
        l2_means = [row["l2_mean"] for row in per_pair]
        mae_means = [row["mae_mean"] for row in per_pair]
        per_distance.append({
            "distance": int(distance),
            "n_horizon_pairs": len(per_pair),
            "cosine_mean_over_pairs": float(np.mean(cosine_means)) if cosine_means else None,
            "cosine_median_over_pairs": float(np.median(cosine_means)) if cosine_means else None,
            "l2_mean_over_pairs": float(np.mean(l2_means)) if l2_means else None,
            "mae_mean_over_pairs": float(np.mean(mae_means)) if mae_means else None,
            "per_pair": per_pair,
        })
    return {
        "mode": "observed_target_forward_horizon_similarity",
        "distances": [row["distance"] for row in per_distance],
        "per_distance": per_distance,
    }


def summarize_candidate_spec(
    spec: HorizonSpec,
    target_rollout: np.ndarray | None = None,
    *,
    min_target_cosine_drop: float = 0.05,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "spec": spec.to_dict(),
        "kind": spec.kind,
        "event_export_supported": spec.kind == "event_count",
        "time_based_status": "not_time_based" if spec.kind == "event_count" else "requires_separate_reviewed_time_based_extractor",
    }
    export_cmd = spec.export_rollout_command()
    if export_cmd:
        out["rollout_export_command_template"] = export_cmd
    out["readiness_command_template"] = spec.readiness_command()
    if target_rollout is None:
        out["status"] = "planned_not_evaluated"
        return out
    target = _as_rollout(target_rollout, spec_id=spec.spec_id)
    if target.shape[1] != spec.horizon_count:
        raise ValueError(f"{spec.spec_id}: target rollout horizon count {target.shape[1]} does not match spec horizon_count {spec.horizon_count}")
    similarity = target_horizon_similarity(target, distances=spec.diagnostic_distances)
    per_distance = [row for row in similarity["per_distance"] if row["cosine_mean_over_pairs"] is not None]
    first = next((row for row in per_distance if row["distance"] == 1), per_distance[0] if per_distance else None)
    last = per_distance[-1] if per_distance else None
    cosine_drop = None
    if first and last:
        cosine_drop = float(first["cosine_mean_over_pairs"] - last["cosine_mean_over_pairs"])
    out.update({
        "status": "evaluated_synthetic_or_local_aggregate",
        "n_sequences": int(target.shape[0]),
        "n_horizons": int(target.shape[1]),
        "embedding_dim": int(target.shape[2]),
        "target_horizon_similarity": similarity,
        "target_cosine_drop_from_distance1_to_max": cosine_drop,
        "meets_min_target_cosine_drop": bool(cosine_drop is not None and cosine_drop >= min_target_cosine_drop),
    })
    return out


def build_horizon_spec_report(
    specs: list[HorizonSpec],
    *,
    target_rollouts: dict[str, np.ndarray] | None = None,
    scenario_id: str = "horizon_spec_candidate_grid",
    min_target_cosine_drop: float = 0.05,
) -> dict[str, Any]:
    target_rollouts = target_rollouts or {}
    candidate_results = [
        summarize_candidate_spec(spec, target_rollouts.get(spec.spec_id), min_target_cosine_drop=min_target_cosine_drop)
        for spec in specs
    ]
    evaluated = [row for row in candidate_results if row.get("status") == "evaluated_synthetic_or_local_aggregate"]
    recommended = [
        row["spec"]["spec_id"]
        for row in evaluated
        if row.get("meets_min_target_cosine_drop")
    ]
    return {
        "schema_version": "clinical-jepa-horizon-spec-diagnostic-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "n_specs": len(specs),
        "n_specs_evaluated": len(evaluated),
        "min_target_cosine_drop": float(min_target_cosine_drop),
        "candidate_results": candidate_results,
        "recommended_specs_by_target_separability": recommended,
        "notes": "Safe-public horizon specification scaffold. Command templates use placeholders and are not approval to run governed data, remote compute, renderer generation, or clinical/treatment-effect analyses.",
    }


def _parse_rollout_args(items: list[str] | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--target-rollout entries must have the form spec_id=/path/to/target-rollout.npy")
        spec_id, path = item.split("=", 1)
        spec_id = spec_id.strip()
        if not spec_id:
            raise ValueError("--target-rollout spec_id cannot be empty")
        out[spec_id] = np.load(path)
    return out


def _command_plan_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA horizon-spec command plan",
        "",
        "These commands are templates for reviewed local execution only. Replace placeholders locally; do not commit local paths, arrays, checkpoints, HDF5s, sidecars, or row-level outputs.",
        "",
    ]
    for row in report["candidate_results"]:
        spec_id = row["spec"]["spec_id"]
        lines.append(f"## `{spec_id}`")
        lines.append("")
        if row.get("rollout_export_command_template"):
            lines.append("Rollout export template:")
            lines.append("")
            lines.append("```bash")
            lines.append(row["rollout_export_command_template"])
            lines.append("```")
            lines.append("")
        else:
            lines.append("No event-count rollout export template is emitted for this spec; time-based specs require separate reviewed extraction support.")
            lines.append("")
        lines.append("Readiness diagnostic template:")
        lines.append("")
        lines.append("```bash")
        lines.append(row["readiness_command_template"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA horizon-spec diagnostic scaffold",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Specs: {report['n_specs']}",
        f"Evaluated target rollouts: {report['n_specs_evaluated']}",
        "",
        "This is a target-window/horizon specification diagnostic, not a renderer/generation, clinical-utility, or treatment-effect claim.",
        "",
        "| Spec | Kind | Window | Stride | Horizons | Distances | Status | Target cosine drop | Meets drop gate |",
        "|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for row in report["candidate_results"]:
        spec = row["spec"]
        window = spec.get("target_window_events", spec.get("target_window_hours", ""))
        stride = spec.get("horizon_stride_events", spec.get("horizon_stride_hours", ""))
        drop = row.get("target_cosine_drop_from_distance1_to_max")
        drop_text = f"{drop:.4f}" if isinstance(drop, float) else ""
        lines.append(
            f"| `{spec['spec_id']}` | {spec['kind']} | {window} | {stride} | {spec['horizon_count']} | {spec['diagnostic_distances']} | {row['status']} | {drop_text} | {row.get('meets_min_target_cosine_drop', '')} |"
        )
    if report.get("recommended_specs_by_target_separability"):
        lines.extend([
            "",
            "## Specs meeting target-separability drop gate",
            "",
            ", ".join(f"`{x}`" for x in report["recommended_specs_by_target_separability"]),
        ])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe-public horizon specification diagnostics for Clinical-JEPA")
    ap.add_argument("--spec-config", help="YAML/JSON config with horizon_specs list; defaults to BP005 event-count candidate grid")
    ap.add_argument("--target-rollout", action="append", help="Optional synthetic/local aggregate target rollout as spec_id=/path/to/target-rollout.npy")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="horizon_spec_candidate_grid")
    ap.add_argument("--min-target-cosine-drop", type=float, default=0.05)
    args = ap.parse_args(argv)

    specs = load_horizon_specs(args.spec_config)
    target_rollouts = _parse_rollout_args(args.target_rollout)
    known = {spec.spec_id for spec in specs}
    extra = sorted(set(target_rollouts) - known)
    if extra:
        raise SystemExit(f"Target rollout provided for unknown spec_id(s): {', '.join(extra)}")
    report = build_horizon_spec_report(
        specs,
        target_rollouts=target_rollouts,
        scenario_id=args.scenario_id,
        min_target_cosine_drop=args.min_target_cosine_drop,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "horizon-spec-diagnostic.json", report)
    (outdir / "summary.md").write_text(_summary_md(report))
    (outdir / "command-plan.md").write_text(_command_plan_md(report))
    print(json.dumps({"output": str(outdir / "horizon-spec-diagnostic.json"), "n_specs": report["n_specs"], "n_specs_evaluated": report["n_specs_evaluated"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
