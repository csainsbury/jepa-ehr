"""Synthetic h5 + config fixtures for joint-substrate tests (no governed data).

Builds tiny h5 files that mimic the joint corrected substrate schema:
per group, datasets token_ids / time_deltas / cumulative_days / is_outcome_label,
with the DATASET:* source token at index 0 and [BOS] at index 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCID_TOKEN = 1048
MIMIC_TOKEN = 1049
BOS = 1
MED_TOKEN = 951  # "MED:TEST" in the tiny vocab below
BENIGN_TOKEN = 60  # a diagnosis-range token, never an anchor


def make_sequence(
    source_token_id: int,
    length: int,
    *,
    outcome_at: Iterable[int] = (),
    med_at: Iterable[int] = (),
    med_token: int = MED_TOKEN,
    cumulative_days: np.ndarray | list[float] | None = None,
    segment_ids: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray]:
    length = int(length)
    token_ids = np.full(length, BENIGN_TOKEN, dtype=np.int64)
    token_ids[0] = source_token_id
    if length > 1:
        token_ids[1] = BOS
    for i in med_at:
        if 0 <= int(i) < length:
            token_ids[int(i)] = med_token
    time_deltas = np.ones(length, dtype=np.float32)
    time_deltas[0] = 0.0
    if cumulative_days is None:
        cdays = np.arange(length, dtype=np.float32)
    else:
        cdays = np.asarray(cumulative_days, dtype=np.float32)
    is_outcome_label = np.zeros(length, dtype=np.int8)
    for i in outcome_at:
        if 0 <= int(i) < length:
            is_outcome_label[int(i)] = 1
    out: dict[str, np.ndarray] = {
        "token_ids": token_ids,
        "time_deltas": time_deltas,
        "cumulative_days": cdays,
        "is_outcome_label": is_outcome_label,
    }
    if segment_ids is not None:
        out["segment_ids"] = np.asarray(segment_ids, dtype=np.int64)
    return out


def write_h5(path: str | Path, groups: dict[str, dict[str, np.ndarray]]) -> str:
    import h5py

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for name, arrays in groups.items():
            grp = h5.create_group(name)
            for key, arr in arrays.items():
                grp.create_dataset(key, data=arr)
    return path


def write_tiny_vocab(path: str | Path) -> str:
    """id-string -> token vocab (FlatASCEND format) with a MED anchor."""
    vocab: dict[str, str] = {
        "0": "[PAD]", "1": "[BOS]", "2": "[EOS]", "3": "[DEATH]",
        str(MED_TOKEN): "MED:TEST",
        str(SCID_TOKEN): "DATASET:SCID", str(MIMIC_TOKEN): "DATASET:MIMIC",
        str(BENIGN_TOKEN): "DX:TEST",
    }
    path = str(path)
    Path(path).write_text(json.dumps(vocab))
    return path


def build_source_h5s(
    root: str | Path,
    *,
    scid_lens: list[int],
    mimic_lens: list[int],
    splits: tuple[str, ...] = ("train", "dev", "test"),
    scid_outcome_at: dict[int, list[int]] | None = None,
    med_at: list[int] | None = None,
) -> dict[str, Any]:
    """Write per-source h5s for each split and return path metadata.

    ``scid_outcome_at`` maps a SCID sequence index -> outcome positions (applied
    to every split). ``med_at`` plants MED anchors in every sequence.
    """
    root = Path(root)
    scid_outcome_at = scid_outcome_at or {}
    med_at = med_at or []
    scid_paths: dict[str, str] = {}
    mimic_paths: dict[str, str] = {}
    for split in splits:
        scid_groups = {
            f"scid_{split}_{i}": make_sequence(
                SCID_TOKEN, n, outcome_at=scid_outcome_at.get(i, []), med_at=med_at,
            )
            for i, n in enumerate(scid_lens)
        }
        mimic_groups = {
            f"mimic_{split}_{i}": make_sequence(MIMIC_TOKEN, n, med_at=med_at)
            for i, n in enumerate(mimic_lens)
        }
        scid_paths[split] = write_h5(root / f"scid_{split}.h5", scid_groups)
        mimic_paths[split] = write_h5(root / f"mimic_{split}.h5", mimic_groups)
    return {"scid": scid_paths, "mimic": mimic_paths}


def joint_dataset_config(
    root: str | Path,
    source_paths: dict[str, Any],
    *,
    vocab_path: str,
    index_dir: str | Path,
    source_prefix_len: int = 2,
    require_source_mask: bool = True,
    min_valid_windows: int = 500,
    min_matched_candidates: int = 2,
    candidate_windows: list[int] | None = None,
) -> dict[str, Any]:
    index_dir = Path(index_dir)
    return {
        "schema_version": "clinical-jepa-dataset-config-v0",
        "vocabulary": {
            "vocab_name": "joint-test",
            "vocab_size": 1050,
            "vocab_json_path": vocab_path,
            "family_ranges": {"medication": [951, 1032]},
        },
        "time_channel": "cumulative_days",
        "leakage": {"outcome_label_dataset": "is_outcome_label"},
        "mask": {"source_prefix_len": source_prefix_len, "require_source_mask": require_source_mask},
        "source_token_ids": {"SCID": SCID_TOKEN, "MIMIC": MIMIC_TOKEN},
        "verify_source_token": True,
        "sources": {
            "primary": {
                "name": "joint-test",
                "role": "train_dev_test",
                "source_datasets": [
                    {"name": "SCID", "kind": "per_source", "h5_paths": source_paths["scid"]},
                    {"name": "MIMIC", "kind": "per_source", "h5_paths": source_paths["mimic"]},
                ],
                "split_index_paths": {
                    "train": str(index_dir / "train.index.jsonl"),
                    "dev": str(index_dir / "dev.index.jsonl"),
                    "test": str(index_dir / "test.index.jsonl"),
                },
            },
            "locked_stress": {"name": "none-staged"},
        },
        "readiness": {
            "min_valid_windows_per_source_per_split": min_valid_windows,
            "min_matched_candidates": min_matched_candidates,
            "t0_gap_events": 0,
            "candidate_windows": candidate_windows or [8, 16, 32],
        },
        "split": {"policy": "inherited-source-splits-no-resplitting"},
    }


def joint_arms_config() -> dict[str, Any]:
    return {
        "schema_version": "clinical-jepa-arms-config-v0",
        "common": {
            "target_blocks": {
                "T0": {
                    "enabled": True,
                    "event_count_window": 32,
                    "min_context": 8,
                    # Wall-clock mode: source-specific windows for yield; common
                    # horizons for the cross-source hierarchy comparison (rung 0).
                    "wall_clock": {
                        "window_days": 90.0,
                        "gap_days": 0.0,
                        "common_horizons_days": [30.0, 90.0],
                    },
                    "per_source": {
                        "SCID": {
                            "event_count_window": 32,
                            "min_context": 8,
                            "wall_clock": {"window_days": 90.0, "min_context": 8},
                        },
                        "MIMIC": {
                            "event_count_window": 8,
                            "min_context": 4,
                            # Short per-admission MIMIC: narrower window + gap.
                            "wall_clock": {"window_days": 5.0, "gap_days": 0.0, "min_context": 4},
                        },
                    },
                },
                "T1": {
                    "enabled": True,
                    "event_count_window": 32,
                    "min_context": 8,
                    "per_source": {
                        "SCID": {"event_count_window": 32, "min_context": 8},
                        "MIMIC": {"event_count_window": 8, "min_context": 4},
                    },
                },
            },
        },
    }


def write_yaml(path: str | Path, data: dict[str, Any]) -> str:
    import yaml

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data))
    return path
