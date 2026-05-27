from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from clinical_jepa.speaker.future_summary import _fmt, evaluate_multilabel
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, write_json

VALID_ROLE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_CONTEXT_KEYS = ("context_events",)
DEFAULT_TARGET_KEY = "target_events"
DEFAULT_CONTROL_EVENT_KEYS = ("matched_random_events", "time_shift_target_events", "negative_control_events")


@dataclass(frozen=True)
class TargetFamilySpec:
    """Configurable aggregate coded-summary target family.

    Matching is prefix/predicate based over pre-extracted coded-event strings.
    Public configs and tests should use synthetic/generic names only. Aggregate
    reports suppress family names and predicate strings by default.
    """

    name: str
    include_prefixes: tuple[str, ...]
    exclude_prefixes: tuple[str, ...] = ()
    role: str = "candidate"
    summary_modes: tuple[str, ...] = ("presence", "start")
    negative_control: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TargetFamilySpec":
        name = str(row.get("name", "")).strip()
        if not name or not VALID_ROLE_RE.fullmatch(name):
            raise ValueError(f"Invalid target-family name {name!r}")
        include_prefixes = tuple(_normalise_prefix(x) for x in row.get("include_prefixes", []))
        if not include_prefixes:
            raise ValueError(f"Target family {name!r} must define include_prefixes")
        exclude_prefixes = tuple(_normalise_prefix(x) for x in row.get("exclude_prefixes", []))
        role = str(row.get("role", "candidate")).strip() or "candidate"
        if not VALID_ROLE_RE.fullmatch(role):
            raise ValueError(f"Invalid target-family role {role!r}")
        modes = tuple(str(x).strip().lower() for x in row.get("summary_modes", ("presence", "start")))
        if not modes:
            raise ValueError(f"Target family {name!r} must define at least one summary mode")
        allowed_modes = {"presence", "start", "continuation", "absence"}
        unknown = [mode for mode in modes if mode not in allowed_modes]
        if unknown:
            raise ValueError(f"Unknown summary_modes for {name!r}: {unknown}")
        return cls(
            name=name,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            role=role,
            summary_modes=modes,
            negative_control=bool(row.get("negative_control", role == "negative_control")),
        )

    def to_safe_dict(self, index: int) -> dict[str, Any]:
        return {
            "target_index": int(index),
            "role": self.role,
            "negative_control": bool(self.negative_control),
            "summary_modes": list(self.summary_modes),
            "predicate_names_suppressed": True,
        }


@dataclass(frozen=True)
class ScenarioOntologySpec:
    """BP008 scenario-specific aggregate target ontology specification."""

    spec_id: str = "scenario_coded_summary_v0"
    target_families: tuple[TargetFamilySpec, ...] = field(default_factory=tuple)
    context_event_keys: tuple[str, ...] = DEFAULT_CONTEXT_KEYS
    target_event_key: str = DEFAULT_TARGET_KEY
    control_event_keys: tuple[str, ...] = DEFAULT_CONTROL_EVENT_KEYS
    top_k: int = 5
    min_positive_count: int = 2
    max_prior_micro_ap: float = 0.85
    max_target_prevalence: float = 0.80
    utilization_bins: tuple[int, ...] = (0, 4, 8, 16, 32)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScenarioOntologySpec":
        spec_id = str(row.get("spec_id", "scenario_coded_summary_v0")).strip()
        if not VALID_ROLE_RE.fullmatch(spec_id):
            raise ValueError(f"Invalid spec_id {spec_id!r}")
        families = tuple(TargetFamilySpec.from_dict(x) for x in row.get("target_families", []))
        if not families:
            raise ValueError("scenario ontology requires at least one target_family")
        context_event_keys = tuple(str(x).strip() for x in row.get("context_event_keys", DEFAULT_CONTEXT_KEYS))
        target_event_key = str(row.get("target_event_key", DEFAULT_TARGET_KEY)).strip()
        control_event_keys = tuple(str(x).strip() for x in row.get("control_event_keys", DEFAULT_CONTROL_EVENT_KEYS))
        top_k = int(row.get("top_k", 5))
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        min_positive_count = int(row.get("min_positive_count", 2))
        if min_positive_count < 1:
            raise ValueError("min_positive_count must be >= 1")
        max_prior_micro_ap = float(row.get("max_prior_micro_ap", 0.85))
        if not (0.0 <= max_prior_micro_ap <= 1.0):
            raise ValueError("max_prior_micro_ap must be between 0 and 1")
        max_target_prevalence = float(row.get("max_target_prevalence", 0.80))
        if not (0.0 < max_target_prevalence <= 1.0):
            raise ValueError("max_target_prevalence must be in (0, 1]")
        bins = tuple(sorted({int(x) for x in row.get("utilization_bins", (0, 4, 8, 16, 32))}))
        if not bins:
            raise ValueError("utilization_bins cannot be empty")
        return cls(
            spec_id=spec_id,
            target_families=families,
            context_event_keys=context_event_keys,
            target_event_key=target_event_key,
            control_event_keys=control_event_keys,
            top_k=top_k,
            min_positive_count=min_positive_count,
            max_prior_micro_ap=max_prior_micro_ap,
            max_target_prevalence=max_target_prevalence,
            utilization_bins=bins,
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "n_target_families": len(self.target_families),
            "target_families": [family.to_safe_dict(i) for i, family in enumerate(self.target_families)],
            "context_event_keys": list(self.context_event_keys),
            "target_event_key": self.target_event_key,
            "control_event_keys": list(self.control_event_keys),
            "top_k": int(self.top_k),
            "min_positive_count": int(self.min_positive_count),
            "max_prior_micro_ap": float(self.max_prior_micro_ap),
            "max_target_prevalence": float(self.max_target_prevalence),
            "utilization_bins": list(self.utilization_bins),
            "target_names_suppressed": True,
            "predicate_names_suppressed": True,
        }


def load_scenario_ontology_spec(path: str | Path | None = None) -> ScenarioOntologySpec:
    if path is None:
        return default_diuretic_synthetic_spec()
    p = Path(path)
    data = load_yaml(p) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
    row = data.get("scenario_ontology", data)
    if not isinstance(row, dict):
        raise ValueError("Scenario ontology config must be an object or contain a scenario_ontology object")
    return ScenarioOntologySpec.from_dict(row)


def default_diuretic_synthetic_spec() -> ScenarioOntologySpec:
    return ScenarioOntologySpec(
        target_families=(
            TargetFamilySpec(
                name="synthetic_diuretic_med_family",
                include_prefixes=("MED:SYN_DIURETIC",),
                role="medication_subclass",
                summary_modes=("presence", "start", "continuation"),
            ),
            TargetFamilySpec(
                name="synthetic_renal_lab_family",
                include_prefixes=("LAB:SYN_RENAL", "STATE:SYN_RENAL"),
                role="renal_lab_state",
                summary_modes=("presence", "start"),
            ),
            TargetFamilySpec(
                name="synthetic_electrolyte_lab_family",
                include_prefixes=("LAB:SYN_ELECTROLYTE", "STATE:SYN_ELECTROLYTE"),
                role="renal_lab_state",
                summary_modes=("presence", "start"),
            ),
            TargetFamilySpec(
                name="synthetic_glucose_negative_control",
                include_prefixes=("LAB:SYN_NEG_GLUCOSE",),
                role="negative_control",
                summary_modes=("presence",),
                negative_control=True,
            ),
        )
    )


def _normalise_prefix(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Empty prefix is not allowed")
    return text


def _events(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of synthetic/pre-extracted coded event strings")
    return [str(x) for x in value]


def collect_events(row: dict[str, Any], keys: Iterable[str]) -> list[str]:
    events: list[str] = []
    for key in keys:
        events.extend(_events(row, key))
    return events


def matches_family(token: str, family: TargetFamilySpec) -> bool:
    text = str(token).strip().upper()
    if not text:
        return False
    if family.exclude_prefixes and any(text.startswith(prefix) for prefix in family.exclude_prefixes):
        return False
    return any(text.startswith(prefix) for prefix in family.include_prefixes)


def count_family(events: Iterable[str], family: TargetFamilySpec) -> int:
    return int(sum(1 for event in events if matches_family(event, family)))


def family_features(context_events: list[str], target_events: list[str], family: TargetFamilySpec) -> dict[str, int]:
    context_count = count_family(context_events, family)
    target_count = count_family(target_events, family)
    context_present = int(context_count > 0)
    target_present = int(target_count > 0)
    return {
        "context_count": context_count,
        "target_count": target_count,
        "presence": target_present,
        "start": int(target_present == 1 and context_present == 0),
        "continuation": int(target_present == 1 and context_present == 1),
        "absence": int(target_present == 0),
    }


def target_columns(spec: ScenarioOntologySpec) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, family in enumerate(spec.target_families):
        for mode in family.summary_modes:
            out.append((i, mode))
    return out


def scenario_target_matrix(rows: list[dict[str, Any]], spec: ScenarioOntologySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    columns = target_columns(spec)
    y = np.zeros((len(rows), len(columns)), dtype=np.float32)
    context_scores = np.zeros_like(y)
    target_counts = np.zeros_like(y)
    for r, row in enumerate(rows):
        context_events = collect_events(row, spec.context_event_keys)
        target_events = _events(row, spec.target_event_key)
        for c, (family_idx, mode) in enumerate(columns):
            family = spec.target_families[family_idx]
            features = family_features(context_events, target_events, family)
            y[r, c] = float(features[mode])
            target_counts[r, c] = float(features["target_count"])
            if mode == "presence":
                context_scores[r, c] = float(features["context_count"] > 0)
            elif mode == "start":
                context_scores[r, c] = float(features["context_count"] == 0)
            elif mode == "continuation":
                context_scores[r, c] = float(features["context_count"] > 0)
            elif mode == "absence":
                context_scores[r, c] = float(features["context_count"] == 0)
    return y, context_scores, target_counts, columns


def empirical_prior_scores(y: np.ndarray, n_rows: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    prior = y.mean(axis=0, keepdims=True)
    return np.repeat(prior, n_rows, axis=0).astype(np.float32)


def utilization_scores(rows: list[dict[str, Any]], y: np.ndarray, spec: ScenarioOntologySpec) -> np.ndarray:
    if y.size == 0:
        return np.zeros_like(y)
    buckets = np.asarray([utilization_bucket(len(collect_events(row, spec.context_event_keys)), spec.utilization_bins) for row in rows], dtype=np.int32)
    global_prior = y.mean(axis=0)
    scores = np.zeros_like(y)
    for bucket in sorted(set(int(x) for x in buckets.tolist())):
        mask = buckets == bucket
        if int(mask.sum()) < 2:
            scores[mask] = global_prior
        else:
            scores[mask] = y[mask].mean(axis=0)
    return scores.astype(np.float32)


def utilization_bucket(count: int, bins: tuple[int, ...]) -> int:
    value = int(count)
    bucket = 0
    for threshold in bins:
        if value >= threshold:
            bucket = int(threshold)
    return bucket


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


def per_target_diagnostics(
    y: np.ndarray,
    prior_scores: np.ndarray,
    context_scores: np.ndarray,
    utilization_control_scores: np.ndarray,
    columns: list[tuple[int, str]],
    spec: ScenarioOntologySpec,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for c, (family_idx, mode) in enumerate(columns):
        prevalence = float(y[:, c].mean()) if len(y) else 0.0
        positives = int(y[:, c].sum()) if len(y) else 0
        prior_ap = _average_precision(y[:, c], prior_scores[:, c]) if len(y) else None
        context_ap = _average_precision(y[:, c], context_scores[:, c]) if len(y) else None
        utilization_ap = _average_precision(y[:, c], utilization_control_scores[:, c]) if len(y) else None
        family = spec.target_families[family_idx]
        high_prevalence = prevalence > spec.max_target_prevalence
        low_support = positives < spec.min_positive_count
        context_delta = _safe_delta(context_ap, prior_ap)
        utilization_delta = _safe_delta(context_ap, utilization_ap)
        prior_dominant = bool(
            high_prevalence
            or (prior_ap is not None and prior_ap >= spec.max_prior_micro_ap)
            or (context_delta is not None and context_delta <= 0.0)
        )
        diagnostics.append(
            {
                "target_index": int(c),
                "family_index": int(family_idx),
                "mode": mode,
                "role": family.role,
                "negative_control": bool(family.negative_control),
                "positives": positives,
                "prevalence": prevalence,
                "prior_average_precision": prior_ap,
                "context_average_precision": context_ap,
                "utilization_average_precision": utilization_ap,
                "context_minus_prior_ap": context_delta,
                "context_minus_utilization_ap": utilization_delta,
                "high_prevalence": bool(high_prevalence),
                "low_support": bool(low_support),
                "prior_dominant": bool(prior_dominant),
                "target_name_suppressed": True,
            }
        )
    return diagnostics


def base_rate_domination_summary(diagnostics: list[dict[str, Any]], spec: ScenarioOntologySpec, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    n = len(diagnostics)
    dominated = [d for d in diagnostics if d["prior_dominant"]]
    high_prev = [d for d in diagnostics if d["high_prevalence"]]
    low_support = [d for d in diagnostics if d["low_support"]]
    candidates = [d for d in diagnostics if not d["negative_control"]]
    viable = [
        d
        for d in candidates
        if not d["prior_dominant"]
        and not d["low_support"]
        and (d["context_minus_prior_ap"] is not None and d["context_minus_prior_ap"] > 0.0)
        and (d["context_minus_utilization_ap"] is None or d["context_minus_utilization_ap"] >= -0.02)
    ]
    prior_micro = baseline_metrics["empirical_prior"]["micro_average_precision"]
    context_micro = baseline_metrics["context_summary"]["micro_average_precision"]
    utilization_micro = baseline_metrics["utilization_control"]["micro_average_precision"]
    return {
        "n_targets": int(n),
        "n_prior_dominant": int(len(dominated)),
        "n_high_prevalence": int(len(high_prev)),
        "n_low_support": int(len(low_support)),
        "n_candidate_targets": int(len(candidates)),
        "n_viable_candidate_targets": int(len(viable)),
        "prior_micro_ap": prior_micro,
        "context_micro_ap": context_micro,
        "utilization_micro_ap": utilization_micro,
        "context_minus_prior_micro_ap": _safe_delta(context_micro, prior_micro),
        "context_minus_utilization_micro_ap": _safe_delta(context_micro, utilization_micro),
        "prior_micro_ap_threshold": float(spec.max_prior_micro_ap),
        "base_rate_domination_flag": bool(len(dominated) == n or (prior_micro is not None and prior_micro >= spec.max_prior_micro_ap and _safe_delta(context_micro, prior_micro) is not None and _safe_delta(context_micro, prior_micro) <= 0.0)),
        "recommendation": "promising_for_local_feasibility_scan" if viable else "refine_or_park_if_local_scan_repeats_base_rate_domination",
    }


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):
        return None
    return float(left) - float(right)


def control_event_set_summary(rows: list[dict[str, Any]], spec: ScenarioOntologySpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in spec.control_event_keys:
        present_rows = [row for row in rows if key in row]
        if not present_rows:
            continue
        counts = []
        any_target_family = []
        for row in present_rows:
            events = _events(row, key)
            counts.append(len(events))
            any_target_family.append(int(any(count_family(events, family) > 0 for family in spec.target_families)))
        out[key] = {
            "n_rows_with_control_events": int(len(present_rows)),
            "event_count_mean": float(np.mean(counts)) if counts else 0.0,
            "any_target_family_presence_rate": float(np.mean(any_target_family)) if any_target_family else 0.0,
            "label_names_suppressed": True,
        }
    return out


def build_scenario_ontology_report(rows: list[dict[str, Any]], spec: ScenarioOntologySpec, *, scenario_id: str = "scenario_coded_summary") -> dict[str, Any]:
    y, context_scores, target_counts, columns = scenario_target_matrix(rows, spec)
    prior_scores = empirical_prior_scores(y, len(rows))
    utilization_control = utilization_scores(rows, y, spec)
    labels = [f"target_{i:03d}" for i in range(y.shape[1])]
    baseline_metrics = {
        "empirical_prior": evaluate_multilabel(y, prior_scores, labels, top_k=spec.top_k),
        "context_summary": evaluate_multilabel(y, context_scores, labels, top_k=spec.top_k),
        "utilization_control": evaluate_multilabel(y, utilization_control, labels, top_k=spec.top_k),
    }
    diagnostics = per_target_diagnostics(y, prior_scores, context_scores, utilization_control, columns, spec)
    row_context_counts = [len(collect_events(row, spec.context_event_keys)) for row in rows]
    row_target_counts = [len(_events(row, spec.target_event_key)) for row in rows]
    report = {
        "schema_version": "clinical-jepa-scenario-coded-summary-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "not_generation": True,
        "not_clinical_or_treatment_effect": True,
        "spec": spec.to_safe_dict(),
        "n_rows": int(len(rows)),
        "n_targets": int(y.shape[1]),
        "target_names_suppressed": True,
        "predicate_names_suppressed": True,
        "row_event_count_summary": {
            "context_mean": float(np.mean(row_context_counts)) if row_context_counts else 0.0,
            "target_mean": float(np.mean(row_target_counts)) if row_target_counts else 0.0,
        },
        "baseline_metrics": baseline_metrics,
        "target_diagnostics": diagnostics,
        "base_rate_domination": base_rate_domination_summary(diagnostics, spec, baseline_metrics),
        "negative_control_hooks": {
            "configured_negative_control_targets": int(sum(1 for family in spec.target_families if family.negative_control)),
            "configured_control_event_keys": list(spec.control_event_keys),
            "observed_control_event_sets": control_event_set_summary(rows, spec),
        },
        "bridge_contract": bridge_contract(spec),
        "notes": "Aggregate-only scenario-specific coded-summary ontology scaffold. Inputs may be synthetic or reviewed local pre-extracted summaries; outputs suppress target names/predicates and contain no row IDs, patient IDs, raw tokens, examples, generated sequences, clinical claims, or treatment-effect claims.",
    }
    return report


def bridge_contract(spec: ScenarioOntologySpec) -> dict[str, Any]:
    return {
        "contract_id": "scenario_specific_coded_summary_v0",
        "not_generation": True,
        "input": "prefix/context coded-summary vector or latent readout scores from reviewed local extraction",
        "output": "aggregate scenario-specific future-summary metrics, not event sequences",
        "target_family_count": int(len(spec.target_families)),
        "supported_baselines": ["empirical_prior", "context_summary", "utilization_control"],
        "negative_control_hooks": list(spec.control_event_keys),
        "local_extraction_required": "reviewed governed local pre-extraction for real data; no HDF5/checkpoint/sidecar paths or raw token examples in public artifacts",
    }


def command_plan(spec: ScenarioOntologySpec, *, output_root_placeholder: str = "<LOCAL_OUTPUT_ROOT>") -> str:
    return "\n".join(
        [
            "# Placeholder-only BP008 local command plan",
            "# Replace placeholders only in reviewed local governed context; do not commit local paths or row-level outputs.",
            "python -m clinical_jepa.speaker.scenario_ontology \\",
            "  --spec-config configs/v0/scenario_ontology.example.yaml \\",
            "  --input-json <LOCAL_PREEXTRACTED_SCENARIO_CODED_SUMMARY_ROWS.json> \\",
            f"  --output-dir {output_root_placeholder}/scenario-coded-summary-readout \\",
            "  --scenario-id bp008_local_scenario_coded_summary",
            "",
            "# Expected local input rows contain reviewed pre-extracted context_events and target_events lists,",
            "# optionally matched_random_events/time_shift_target_events/negative_control_events for controls.",
            "# The CLI emits aggregate metrics only and does not render/generate event sequences,",
            "# estimate treatment effects, or make clinical recommendations.",
            f"# Configured target families: {len(spec.target_families)} (names/predicates suppressed in aggregate reports).",
        ]
    )


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    rows = data.get("rows", data if isinstance(data, list) else None)
    if not isinstance(rows, list):
        raise ValueError("Input must be a list or an object with a rows list")
    return rows


def _summary_md(report: dict[str, Any]) -> str:
    baseline = report["baseline_metrics"]
    domination = report["base_rate_domination"]
    lines = [
        "# Clinical-JEPA scenario-specific coded-summary readout",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Rows: {report['n_rows']}",
        f"Targets: {report['n_targets']}",
        "",
        "This is an aggregate coded-summary/readout scaffold, not event generation, clinical utility, treatment-effect estimation, or treatment recommendation.",
        "",
        "## Baselines",
        "",
        "| Baseline | Macro AP | Micro AP | Top-k recall | Top-k precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("empirical_prior", baseline["empirical_prior"]),
        ("context_summary", baseline["context_summary"]),
        ("utilization_control", baseline["utilization_control"]),
    ):
        lines.append(
            f"| `{name}` | {_fmt(metrics.get('macro_average_precision'))} | {_fmt(metrics.get('micro_average_precision'))} | {_fmt(metrics.get('top_k_recall_mean'))} | {_fmt(metrics.get('top_k_precision_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Base-rate domination diagnostic",
            "",
            f"- Prior-dominant targets: {domination['n_prior_dominant']} / {domination['n_targets']}",
            f"- Viable candidate targets: {domination['n_viable_candidate_targets']} / {domination['n_candidate_targets']}",
            f"- Flag: `{domination['base_rate_domination_flag']}`",
            f"- Recommendation: `{domination['recommendation']}`",
            "",
            "## Boundary",
            "",
            "- Target family names and predicates are suppressed in aggregate reports by default.",
            "- Local real-data use requires reviewed pre-extraction and aggregate-only sync-back.",
            "- Outputs are scenario summary scores only, not generated event sequences.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe-public BP008 scenario-specific coded-summary ontology scaffold")
    ap.add_argument("--spec-config", help="YAML/JSON scenario_ontology config; defaults to a synthetic diuretic-style scaffold")
    ap.add_argument("--input-json", help="Synthetic or reviewed local pre-extracted rows with context_events and target_events")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="scenario_coded_summary")
    ap.add_argument("--emit-command-plan", action="store_true")
    args = ap.parse_args(argv)

    spec = load_scenario_ontology_spec(args.spec_config)
    outdir = ensure_dir(args.output_dir)
    if args.input_json:
        report = build_scenario_ontology_report(_load_rows(args.input_json), spec, scenario_id=args.scenario_id)
    else:
        report = {
            "schema_version": "clinical-jepa-scenario-coded-summary-v0",
            "created_utc": now_utc(),
            "scenario_id": args.scenario_id,
            "aggregate_only": True,
            "not_generation": True,
            "not_clinical_or_treatment_effect": True,
            "status": "planned_not_evaluated",
            "spec": spec.to_safe_dict(),
            "n_rows": 0,
            "n_targets": 0,
            "bridge_contract": bridge_contract(spec),
            "notes": "Command-plan/scaffold only; no input rows evaluated.",
        }
    write_json(outdir / "scenario-coded-summary-readout.json", report)
    if "baseline_metrics" in report:
        (outdir / "summary.md").write_text(_summary_md(report))
    if args.emit_command_plan or not args.input_json:
        (outdir / "command-plan.md").write_text(command_plan(spec))
    print(json.dumps({"output": str(outdir / "scenario-coded-summary-readout.json"), "n_rows": report.get("n_rows", 0), "n_targets": report.get("n_targets", 0)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
