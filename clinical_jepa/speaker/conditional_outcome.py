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

VALID_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_CONTEXT_KEYS = ("context_events",)
DEFAULT_TARGET_KEY = "target_events"
DEFAULT_PRIOR_CONTEXT_KEYS = ("prior_context_events",)
DEFAULT_CONTROL_EVENT_KEYS = ("matched_random_events", "time_shift_target_events", "negative_control_events")
DEFAULT_STRATA_FIELDS = ("utilization_bucket", "context_presence", "related_context_presence", "prior_context_presence")


@dataclass(frozen=True)
class ConditionalTargetFamilySpec:
    """Future-only target family for BP010 conditional-outcome diagnostics.

    The target label is always computed from target_events only. Context and
    prior-context predicates are used only to define strata or context/readout
    scores. Public configs/tests should use synthetic names; aggregate reports
    suppress target-family names and predicate strings by default.
    """

    name: str
    include_prefixes: tuple[str, ...]
    exclude_prefixes: tuple[str, ...] = ()
    related_prefixes: tuple[str, ...] = ()
    role: str = "candidate"
    negative_control: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ConditionalTargetFamilySpec":
        name = str(row.get("name", "")).strip()
        if not name or not VALID_ID_RE.fullmatch(name):
            raise ValueError(f"Invalid target-family name {name!r}")
        include_prefixes = tuple(_normalise_prefix(x) for x in row.get("include_prefixes", []))
        if not include_prefixes:
            raise ValueError(f"Target family {name!r} must define include_prefixes")
        exclude_prefixes = tuple(_normalise_prefix(x) for x in row.get("exclude_prefixes", []))
        related_prefixes = tuple(_normalise_prefix(x) for x in row.get("related_prefixes", []))
        role = str(row.get("role", "candidate")).strip() or "candidate"
        if not VALID_ID_RE.fullmatch(role):
            raise ValueError(f"Invalid target-family role {role!r}")
        return cls(
            name=name,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            related_prefixes=related_prefixes,
            role=role,
            negative_control=bool(row.get("negative_control", role == "negative_control")),
        )

    def to_safe_dict(self, index: int) -> dict[str, Any]:
        return {
            "target_index": int(index),
            "role": self.role,
            "negative_control": bool(self.negative_control),
            "future_only_target": True,
            "has_related_context_predicates": bool(self.related_prefixes),
            "target_name_suppressed": True,
            "predicate_names_suppressed": True,
        }


@dataclass(frozen=True)
class ConditionalOutcomeSpec:
    """BP010 fixed-stratum conditional future-event outcome specification."""

    spec_id: str = "conditional_future_outcome_v0"
    target_families: tuple[ConditionalTargetFamilySpec, ...] = field(default_factory=tuple)
    context_event_keys: tuple[str, ...] = DEFAULT_CONTEXT_KEYS
    target_event_key: str = DEFAULT_TARGET_KEY
    prior_context_event_keys: tuple[str, ...] = DEFAULT_PRIOR_CONTEXT_KEYS
    control_event_keys: tuple[str, ...] = DEFAULT_CONTROL_EVENT_KEYS
    readout_score_key: str = "readout_scores"
    strata_fields: tuple[str, ...] = DEFAULT_STRATA_FIELDS
    utilization_bins: tuple[int, ...] = (0, 4, 8, 16, 32)
    top_k: int = 5
    min_positive_count: int = 2
    min_stratum_rows: int = 2

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ConditionalOutcomeSpec":
        spec_id = str(row.get("spec_id", "conditional_future_outcome_v0")).strip()
        if not VALID_ID_RE.fullmatch(spec_id):
            raise ValueError(f"Invalid spec_id {spec_id!r}")
        families = tuple(ConditionalTargetFamilySpec.from_dict(x) for x in row.get("target_families", []))
        if not families:
            raise ValueError("conditional outcome spec requires at least one target_family")
        strata_fields = tuple(str(x).strip() for x in row.get("strata_fields", DEFAULT_STRATA_FIELDS))
        allowed = {"utilization_bucket", "context_presence", "related_context_presence", "prior_context_presence"}
        unknown = [x for x in strata_fields if x not in allowed]
        if unknown:
            raise ValueError(f"Unknown strata_fields: {unknown}")
        bins = tuple(sorted({int(x) for x in row.get("utilization_bins", (0, 4, 8, 16, 32))}))
        if not bins:
            raise ValueError("utilization_bins cannot be empty")
        top_k = int(row.get("top_k", 5))
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        min_positive_count = int(row.get("min_positive_count", 2))
        if min_positive_count < 1:
            raise ValueError("min_positive_count must be >= 1")
        min_stratum_rows = int(row.get("min_stratum_rows", 2))
        if min_stratum_rows < 1:
            raise ValueError("min_stratum_rows must be >= 1")
        return cls(
            spec_id=spec_id,
            target_families=families,
            context_event_keys=tuple(str(x).strip() for x in row.get("context_event_keys", DEFAULT_CONTEXT_KEYS)),
            target_event_key=str(row.get("target_event_key", DEFAULT_TARGET_KEY)).strip(),
            prior_context_event_keys=tuple(str(x).strip() for x in row.get("prior_context_event_keys", DEFAULT_PRIOR_CONTEXT_KEYS)),
            control_event_keys=tuple(str(x).strip() for x in row.get("control_event_keys", DEFAULT_CONTROL_EVENT_KEYS)),
            readout_score_key=str(row.get("readout_score_key", "readout_scores")).strip(),
            strata_fields=strata_fields,
            utilization_bins=bins,
            top_k=top_k,
            min_positive_count=min_positive_count,
            min_stratum_rows=min_stratum_rows,
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "n_target_families": int(len(self.target_families)),
            "target_families": [family.to_safe_dict(i) for i, family in enumerate(self.target_families)],
            "context_event_keys": list(self.context_event_keys),
            "target_event_key": self.target_event_key,
            "prior_context_event_keys": list(self.prior_context_event_keys),
            "control_event_keys": list(self.control_event_keys),
            "readout_score_key": self.readout_score_key,
            "strata_fields": list(self.strata_fields),
            "utilization_bins": list(self.utilization_bins),
            "top_k": int(self.top_k),
            "min_positive_count": int(self.min_positive_count),
            "min_stratum_rows": int(self.min_stratum_rows),
            "future_only_targets": True,
            "context_used_for_strata_not_target_definition": True,
            "target_names_suppressed": True,
            "predicate_names_suppressed": True,
        }


def load_conditional_outcome_spec(path: str | Path | None = None) -> ConditionalOutcomeSpec:
    if path is None:
        return default_conditional_synthetic_spec()
    p = Path(path)
    data = load_yaml(p) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
    row = data.get("conditional_outcome", data)
    if not isinstance(row, dict):
        raise ValueError("Conditional outcome config must be an object or contain a conditional_outcome object")
    return ConditionalOutcomeSpec.from_dict(row)


def default_conditional_synthetic_spec() -> ConditionalOutcomeSpec:
    return ConditionalOutcomeSpec(
        target_families=(
            ConditionalTargetFamilySpec(
                name="synthetic_diuretic_future_presence",
                include_prefixes=("MED:SYN_DIURETIC",),
                related_prefixes=("MED:SYN_COMPARATOR",),
                role="medication_future_presence",
            ),
            ConditionalTargetFamilySpec(
                name="synthetic_renal_future_bucket",
                include_prefixes=("LAB:SYN_RENAL", "STATE:SYN_RENAL"),
                related_prefixes=("LAB:SYN_RENAL_BUCKET", "STATE:SYN_RENAL_BUCKET"),
                role="renal_future_bucket",
            ),
            ConditionalTargetFamilySpec(
                name="synthetic_electrolyte_future_bucket",
                include_prefixes=("LAB:SYN_ELECTROLYTE", "STATE:SYN_ELECTROLYTE"),
                related_prefixes=("LAB:SYN_ELECTROLYTE_BUCKET", "STATE:SYN_ELECTROLYTE_BUCKET"),
                role="electrolyte_future_bucket",
            ),
            ConditionalTargetFamilySpec(
                name="synthetic_glucose_negative_control",
                include_prefixes=("LAB:SYN_NEG_GLUCOSE",),
                role="negative_control",
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


def matches_family(token: str, family: ConditionalTargetFamilySpec) -> bool:
    text = str(token).strip().upper()
    if not text:
        return False
    if family.exclude_prefixes and any(text.startswith(prefix) for prefix in family.exclude_prefixes):
        return False
    return any(text.startswith(prefix) for prefix in family.include_prefixes)


def count_family(events: Iterable[str], family: ConditionalTargetFamilySpec) -> int:
    return int(sum(1 for event in events if matches_family(event, family)))


def count_prefixes(events: Iterable[str], prefixes: tuple[str, ...]) -> int:
    if not prefixes:
        return 0
    normalised = tuple(_normalise_prefix(prefix) for prefix in prefixes)
    return int(sum(1 for event in events if any(str(event).strip().upper().startswith(prefix) for prefix in normalised)))


def utilization_bucket(count: int, bins: tuple[int, ...]) -> int:
    value = int(count)
    bucket = 0
    for threshold in bins:
        if value >= threshold:
            bucket = int(threshold)
    return bucket


def row_family_context_features(row: dict[str, Any], family: ConditionalTargetFamilySpec, spec: ConditionalOutcomeSpec) -> dict[str, int]:
    context_events = collect_events(row, spec.context_event_keys)
    prior_context_events = collect_events(row, spec.prior_context_event_keys)
    context_count = count_family(context_events, family)
    related_context_count = count_prefixes(context_events, family.related_prefixes)
    prior_context_count = count_family(prior_context_events, family)
    return {
        "context_count": int(context_count),
        "related_context_count": int(related_context_count),
        "prior_context_count": int(prior_context_count),
        "context_presence": int(context_count > 0),
        "related_context_presence": int(related_context_count > 0),
        "prior_context_presence": int(prior_context_count > 0),
        "utilization_bucket": utilization_bucket(len(context_events), spec.utilization_bins),
    }


def target_matrix(rows: list[dict[str, Any]], spec: ConditionalOutcomeSpec) -> np.ndarray:
    y = np.zeros((len(rows), len(spec.target_families)), dtype=np.float32)
    for r, row in enumerate(rows):
        target_events = _events(row, spec.target_event_key)
        for c, family in enumerate(spec.target_families):
            y[r, c] = float(count_family(target_events, family) > 0)
    return y


def context_score_matrix(rows: list[dict[str, Any]], spec: ConditionalOutcomeSpec) -> np.ndarray:
    scores = np.zeros((len(rows), len(spec.target_families)), dtype=np.float32)
    for r, row in enumerate(rows):
        explicit_scores = row.get(spec.readout_score_key)
        if isinstance(explicit_scores, list) and len(explicit_scores) == len(spec.target_families):
            scores[r, :] = np.asarray([float(x) for x in explicit_scores], dtype=np.float32)
            continue
        for c, family in enumerate(spec.target_families):
            features = row_family_context_features(row, family, spec)
            score = 0.0
            score += 0.60 * features["context_presence"]
            score += 0.30 * features["related_context_presence"]
            score += 0.10 * features["prior_context_presence"]
            scores[r, c] = min(1.0, float(score))
    return scores


def empirical_prior_scores(y: np.ndarray, n_rows: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    prior = y.mean(axis=0, keepdims=True)
    return np.repeat(prior, n_rows, axis=0).astype(np.float32)


def utilization_control_scores(rows: list[dict[str, Any]], y: np.ndarray, spec: ConditionalOutcomeSpec) -> np.ndarray:
    if y.size == 0:
        return np.zeros_like(y)
    buckets = np.asarray([utilization_bucket(len(collect_events(row, spec.context_event_keys)), spec.utilization_bins) for row in rows], dtype=np.int32)
    global_prior = y.mean(axis=0)
    scores = np.zeros_like(y)
    for bucket in sorted(set(int(x) for x in buckets.tolist())):
        mask = buckets == bucket
        scores[mask] = y[mask].mean(axis=0) if int(mask.sum()) >= spec.min_stratum_rows else global_prior
    return scores.astype(np.float32)


def stratum_key(row: dict[str, Any], family: ConditionalTargetFamilySpec, spec: ConditionalOutcomeSpec) -> tuple[Any, ...]:
    features = row_family_context_features(row, family, spec)
    return tuple(features[field] for field in spec.strata_fields)


def stratum_prior_scores(rows: list[dict[str, Any]], y: np.ndarray, spec: ConditionalOutcomeSpec) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if y.size == 0:
        return np.zeros_like(y), []
    scores = np.zeros_like(y)
    diagnostics: list[dict[str, Any]] = []
    global_prior = y.mean(axis=0)
    for c, family in enumerate(spec.target_families):
        keys = [stratum_key(row, family, spec) for row in rows]
        groups: dict[tuple[Any, ...], list[int]] = {}
        for r, key in enumerate(keys):
            groups.setdefault(key, []).append(r)
        stratum_sizes = [len(v) for v in groups.values()]
        positive_strata = 0
        supported_strata = 0
        for key, idxs in groups.items():
            if len(idxs) < spec.min_stratum_rows:
                scores[idxs, c] = global_prior[c]
                continue
            supported_strata += 1
            value = float(y[idxs, c].mean())
            positive_strata += int(float(y[idxs, c].sum()) > 0.0)
            scores[idxs, c] = value
        diagnostics.append(
            {
                "target_index": int(c),
                "role": family.role,
                "negative_control": bool(family.negative_control),
                "n_strata": int(len(groups)),
                "n_supported_strata": int(supported_strata),
                "n_positive_supported_strata": int(positive_strata),
                "min_stratum_rows": int(min(stratum_sizes) if stratum_sizes else 0),
                "median_stratum_rows": float(np.median(stratum_sizes)) if stratum_sizes else 0.0,
                "max_stratum_rows": int(max(stratum_sizes) if stratum_sizes else 0),
                "target_name_suppressed": True,
            }
        )
    return scores.astype(np.float32), diagnostics


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


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):
        return None
    return float(left) - float(right)


def per_target_diagnostics(
    y: np.ndarray,
    prior_scores: np.ndarray,
    context_scores: np.ndarray,
    utilization_scores_: np.ndarray,
    stratum_scores: np.ndarray,
    stratum_diagnostics: list[dict[str, Any]],
    spec: ConditionalOutcomeSpec,
) -> list[dict[str, Any]]:
    strata_by_target = {int(row["target_index"]): row for row in stratum_diagnostics}
    out: list[dict[str, Any]] = []
    for c, family in enumerate(spec.target_families):
        positives = int(y[:, c].sum()) if len(y) else 0
        prevalence = float(y[:, c].mean()) if len(y) else 0.0
        prior_ap = _average_precision(y[:, c], prior_scores[:, c]) if len(y) else None
        context_ap = _average_precision(y[:, c], context_scores[:, c]) if len(y) else None
        utilization_ap = _average_precision(y[:, c], utilization_scores_[:, c]) if len(y) else None
        stratum_ap = _average_precision(y[:, c], stratum_scores[:, c]) if len(y) else None
        stratum_row = strata_by_target.get(c, {})
        adequate_support = positives >= spec.min_positive_count and int(stratum_row.get("n_positive_supported_strata", 0)) > 0
        out.append(
            {
                "target_index": int(c),
                "role": family.role,
                "negative_control": bool(family.negative_control),
                "future_only_target": True,
                "context_used_for_strata_not_target_definition": True,
                "positives": positives,
                "prevalence": prevalence,
                "adequate_support": bool(adequate_support),
                "prior_average_precision": prior_ap,
                "context_average_precision": context_ap,
                "utilization_average_precision": utilization_ap,
                "stratum_prior_average_precision": stratum_ap,
                "context_minus_prior_ap": _safe_delta(context_ap, prior_ap),
                "context_minus_utilization_ap": _safe_delta(context_ap, utilization_ap),
                "context_minus_stratum_prior_ap": _safe_delta(context_ap, stratum_ap),
                "n_strata": int(stratum_row.get("n_strata", 0)),
                "n_supported_strata": int(stratum_row.get("n_supported_strata", 0)),
                "n_positive_supported_strata": int(stratum_row.get("n_positive_supported_strata", 0)),
                "target_name_suppressed": True,
            }
        )
    return out


def control_event_set_diagnostics(rows: list[dict[str, Any]], context_scores: np.ndarray, spec: ConditionalOutcomeSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    labels = [f"target_{i:03d}" for i in range(len(spec.target_families))]
    for key in spec.control_event_keys:
        present_rows = [row for row in rows if key in row]
        if not present_rows:
            continue
        y_control = np.zeros((len(present_rows), len(spec.target_families)), dtype=np.float32)
        row_indices = [i for i, row in enumerate(rows) if key in row]
        for r, row in enumerate(present_rows):
            events = _events(row, key)
            for c, family in enumerate(spec.target_families):
                y_control[r, c] = float(count_family(events, family) > 0)
        scores = context_scores[row_indices, :] if row_indices else np.zeros_like(y_control)
        out[key] = {
            "n_rows_with_control_events": int(len(present_rows)),
            "any_target_family_presence_rate": float((y_control.sum(axis=1) > 0).mean()) if len(y_control) else 0.0,
            "context_vs_control_targets": evaluate_multilabel(y_control, scores, labels, top_k=spec.top_k),
            "target_names_suppressed": True,
        }
    return out


def conditional_outcome_summary(diagnostics: list[dict[str, Any]], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    candidates = [d for d in diagnostics if not d["negative_control"]]
    negative = [d for d in diagnostics if d["negative_control"]]
    viable = [
        d
        for d in candidates
        if d["adequate_support"]
        and d["context_minus_stratum_prior_ap"] is not None
        and d["context_minus_stratum_prior_ap"] > 0.0
        and d["context_minus_utilization_ap"] is not None
        and d["context_minus_utilization_ap"] > 0.0
    ]
    neg_deltas = [d["context_minus_stratum_prior_ap"] for d in negative if d["context_minus_stratum_prior_ap"] is not None]
    context_micro = baseline_metrics["context_summary"].get("micro_average_precision")
    stratum_micro = baseline_metrics["stratum_prior"].get("micro_average_precision")
    utilization_micro = baseline_metrics["utilization_control"].get("micro_average_precision")
    return {
        "n_targets": int(len(diagnostics)),
        "n_candidate_targets": int(len(candidates)),
        "n_negative_control_targets": int(len(negative)),
        "n_adequate_support_targets": int(sum(1 for d in diagnostics if d["adequate_support"])),
        "n_viable_candidate_targets": int(len(viable)),
        "context_minus_stratum_prior_micro_ap": _safe_delta(context_micro, stratum_micro),
        "context_minus_utilization_micro_ap": _safe_delta(context_micro, utilization_micro),
        "negative_control_context_minus_stratum_prior_ap_max": float(max(neg_deltas)) if neg_deltas else None,
        "future_only_targets": True,
        "context_used_for_strata_not_target_definition": True,
        "recommendation": "promising_for_local_conditional_scan" if viable else "refine_or_park_if_local_scan_fails_stratum_controls",
    }


def build_conditional_outcome_report(rows: list[dict[str, Any]], spec: ConditionalOutcomeSpec, *, scenario_id: str = "conditional_future_outcome") -> dict[str, Any]:
    y = target_matrix(rows, spec)
    context_scores = context_score_matrix(rows, spec)
    prior_scores = empirical_prior_scores(y, len(rows))
    util_scores = utilization_control_scores(rows, y, spec)
    strata_scores, strata_diagnostics = stratum_prior_scores(rows, y, spec)
    labels = [f"target_{i:03d}" for i in range(y.shape[1])]
    baseline_metrics = {
        "empirical_prior": evaluate_multilabel(y, prior_scores, labels, top_k=spec.top_k),
        "context_summary": evaluate_multilabel(y, context_scores, labels, top_k=spec.top_k),
        "utilization_control": evaluate_multilabel(y, util_scores, labels, top_k=spec.top_k),
        "stratum_prior": evaluate_multilabel(y, strata_scores, labels, top_k=spec.top_k),
    }
    diagnostics = per_target_diagnostics(y, prior_scores, context_scores, util_scores, strata_scores, strata_diagnostics, spec)
    row_context_counts = [len(collect_events(row, spec.context_event_keys)) for row in rows]
    row_target_counts = [len(_events(row, spec.target_event_key)) for row in rows]
    report = {
        "schema_version": "clinical-jepa-conditional-future-outcome-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "not_generation": True,
        "not_clinical_or_treatment_effect": True,
        "future_only_targets": True,
        "context_used_for_strata_not_target_definition": True,
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
        "stratum_diagnostics": strata_diagnostics,
        "control_event_diagnostics": control_event_set_diagnostics(rows, context_scores, spec),
        "conditional_outcome_summary": conditional_outcome_summary(diagnostics, baseline_metrics),
        "bridge_contract": bridge_contract(spec),
        "notes": "Aggregate-only conditional future-event outcome scaffold. Targets are computed from future/target events only; context is used for strata and readout scores, not target definitions. Outputs suppress target names/predicates and contain no row IDs, patient IDs, raw tokens, examples, generated sequences, clinical claims, or treatment-effect claims.",
    }
    return report


def bridge_contract(spec: ConditionalOutcomeSpec) -> dict[str, Any]:
    return {
        "contract_id": "conditional_future_event_outcome_v0",
        "not_generation": True,
        "input": "reviewed pre-extracted coded-event summaries or latent readout scores evaluated within fixed strata",
        "target_definition": "future-event presence only; context predicates are strata/readout covariates, not target-label conditions",
        "output": "aggregate conditional future-outcome diagnostics, not event sequences",
        "target_family_count": int(len(spec.target_families)),
        "supported_baselines": ["empirical_prior", "context_summary", "utilization_control", "stratum_prior"],
        "control_hooks": list(spec.control_event_keys),
        "local_extraction_required": "reviewed governed local pre-extraction for real data; no HDF5/checkpoint/sidecar paths or raw token examples in public artifacts",
    }


def command_plan(spec: ConditionalOutcomeSpec, *, output_root_placeholder: str = "<LOCAL_OUTPUT_ROOT>") -> str:
    return "\n".join(
        [
            "# Placeholder-only BP010 local command plan",
            "# Replace placeholders only in reviewed local governed context; do not commit local paths or row-level outputs.",
            "python -m clinical_jepa.speaker.conditional_outcome \\",
            "  --spec-config configs/v0/conditional_outcome.example.yaml \\",
            "  --input-json <LOCAL_PREEXTRACTED_CONDITIONAL_OUTCOME_ROWS.json> \\",
            f"  --output-dir {output_root_placeholder}/conditional-future-outcome-readout \\",
            "  --scenario-id bp010_local_conditional_future_outcome",
            "",
            "# Expected local input rows contain reviewed pre-extracted context_events and target_events lists,",
            "# optionally prior_context_events plus matched_random_events/time_shift_target_events/negative_control_events for controls.",
            "# The CLI emits aggregate metrics only and does not render/generate event sequences,",
            "# estimate treatment effects, or make clinical recommendations.",
            "# BP010 target labels are future-only; context/utilisation variables define strata and controls only.",
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
    summary = report["conditional_outcome_summary"]
    lines = [
        "# Clinical-JEPA conditional future-event outcome readout",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Rows: {report['n_rows']}",
        f"Targets: {report['n_targets']}",
        "",
        "This is an aggregate conditional future-event outcome scaffold, not event generation, clinical utility, treatment-effect estimation, or treatment recommendation.",
        "",
        "Targets are future-only; context and utilisation variables define strata/readout scores, not target labels.",
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
        ("stratum_prior", baseline["stratum_prior"]),
    ):
        lines.append(
            f"| `{name}` | {_fmt(metrics.get('macro_average_precision'))} | {_fmt(metrics.get('micro_average_precision'))} | {_fmt(metrics.get('top_k_recall_mean'))} | {_fmt(metrics.get('top_k_precision_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Conditional-outcome diagnostic",
            "",
            f"- Viable candidate targets: {summary['n_viable_candidate_targets']} / {summary['n_candidate_targets']}",
            f"- Adequate-support targets: {summary['n_adequate_support_targets']} / {summary['n_targets']}",
            f"- Context minus stratum-prior micro AP: {_fmt(summary.get('context_minus_stratum_prior_micro_ap'))}",
            f"- Context minus utilisation micro AP: {_fmt(summary.get('context_minus_utilization_micro_ap'))}",
            f"- Recommendation: `{summary['recommendation']}`",
            "",
            "## Boundary",
            "",
            "- Target family names and predicates are suppressed in aggregate reports by default.",
            "- Local real-data use requires reviewed pre-extraction and aggregate-only sync-back.",
            "- Outputs are conditional future-outcome scores only, not generated event sequences.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe-public BP010 conditional future-event outcome scaffold")
    ap.add_argument("--spec-config", help="YAML/JSON conditional_outcome config; defaults to a synthetic diuretic-style scaffold")
    ap.add_argument("--input-json", help="Synthetic or reviewed local pre-extracted rows with context_events and target_events")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="conditional_future_outcome")
    ap.add_argument("--emit-command-plan", action="store_true")
    args = ap.parse_args(argv)

    spec = load_conditional_outcome_spec(args.spec_config)
    outdir = ensure_dir(args.output_dir)
    if args.input_json:
        report = build_conditional_outcome_report(_load_rows(args.input_json), spec, scenario_id=args.scenario_id)
    else:
        report = {
            "schema_version": "clinical-jepa-conditional-future-outcome-v0",
            "created_utc": now_utc(),
            "scenario_id": args.scenario_id,
            "aggregate_only": True,
            "not_generation": True,
            "not_clinical_or_treatment_effect": True,
            "future_only_targets": True,
            "context_used_for_strata_not_target_definition": True,
            "status": "planned_not_evaluated",
            "spec": spec.to_safe_dict(),
            "n_rows": 0,
            "n_targets": 0,
            "bridge_contract": bridge_contract(spec),
            "notes": "Command-plan/scaffold only; no input rows evaluated.",
        }
    write_json(outdir / "conditional-future-outcome-readout.json", report)
    if "baseline_metrics" in report:
        (outdir / "summary.md").write_text(_summary_md(report))
    if args.emit_command_plan or not args.input_json:
        (outdir / "command-plan.md").write_text(command_plan(spec))
    print(json.dumps({"output": str(outdir / "conditional-future-outcome-readout.json"), "n_rows": report.get("n_rows", 0), "n_targets": report.get("n_targets", 0)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
