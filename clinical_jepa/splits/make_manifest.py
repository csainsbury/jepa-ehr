from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, split_ids, stable_hmac, synthetic_patient_ids, write_json
from clinical_jepa.validation import validate_artifact


def _count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as f:
        return sum(1 for _ in f)


def _split_paths(source_cfg: dict[str, Any], key: str, *, optional: bool = False) -> dict[str, str]:
    paths = source_cfg.get(key, {}) or {}
    if optional and not paths:
        return {}
    required = ["train", "dev", "test"]
    missing = [s for s in required if not paths.get(s)]
    if missing:
        raise ValueError(f"source {source_cfg.get('name')} missing {key}: {missing}")
    return {s: str(paths[s]) for s in required}


def build_manifest(dataset_cfg: dict, splits: dict[str, list[str]], seed: int, salt_exported: bool) -> dict:
    name = dataset_cfg.get("sources", {}).get("primary", {}).get("name", "synthetic-placeholder")
    return {
        "schema_version": "clinical-jepa-split-manifest-v0",
        "created_utc": now_utc(),
        "dataset": name,
        "split_policy": "patient-level-before-windowing",
        "seed": seed,
        "hash_method": "salted-hmac-sha256-local-salt-not-exported",
        "salt_exported": salt_exported,
        "counts": {
            "patients_train": len(splits["train"]),
            "patients_dev": len(splits["dev"]),
            "patients_test": len(splits["test"]),
        },
        "locked_stress_tests": [dataset_cfg.get("sources", {}).get("locked_stress", {}).get("name", "eICU-CRD-placeholder")],
        "notes": "synthetic/dry-run manifest unless generated from approved local metadata",
    }


def build_real_manifest(dataset_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    sources = dataset_cfg.get("sources", {})
    primary = sources.get("primary", {})
    if not primary:
        raise ValueError("dataset config missing sources.primary")
    index_paths = _split_paths(primary, "split_index_paths")
    # Joint / multi-source substrate: per-source h5 files live in the combined
    # index rows (source_h5_path), so a single split-level h5 path is optional.
    multi_source = bool(primary.get("source_datasets"))
    h5_paths = _split_paths(primary, "h5_paths", optional=multi_source)
    seq_counts = {split: _count_jsonl(path) for split, path in index_paths.items()}

    split_cfg = dataset_cfg.get("split", {}) or {}
    bundle = dataset_cfg.get("bundle", {}) or {}
    locked_name = sources.get("locked_stress", {}).get("name", "none-staged")
    manifest: dict[str, Any] = {
        "schema_version": "clinical-jepa-split-manifest-v0",
        "created_utc": now_utc(),
        "dataset": str(primary.get("name", bundle.get("name", "flatascend-b1a-rekeyed"))),
        "split_policy": str(split_cfg.get("policy", "inherited-source-splits-no-resplitting")),
        "seed": int(seed),
        "hash_method": str(split_cfg.get("hash_method", "none-exported-rekeyed-sequence-ids-only")),
        "salt_exported": False,
        # The v0 schema still calls these patient counts. For this re-keyed bundle,
        # the exported unit is a sequence/admission id with no source-id mapping.
        "counts": {
            "patients_train": int(seq_counts["train"]),
            "patients_dev": int(seq_counts["dev"]),
            "patients_test": int(seq_counts["test"]),
        },
        "locked_stress_tests": [str(locked_name)],
        "dry_run": False,
        "unit": "rekeyed_sequence_or_admission",
        "sequence_counts": seq_counts,
        "source_name": primary.get("name"),
        "source_role": primary.get("role"),
        "source_kind": primary.get("source_kind"),
        "source_index_paths": index_paths,
        "source_h5_paths": h5_paths,
        "source_datasets": [str(s.get("name", s.get("kind", "unknown"))) for s in (primary.get("source_datasets") or [])],
        "bundle": {
            "name": bundle.get("name"),
            "root": bundle.get("root"),
            "manifest_path": bundle.get("manifest_path"),
            "original_source_ids_exported": bool(bundle.get("original_source_ids_exported", False)),
            "source_id_mapping_exported": bool(bundle.get("source_id_mapping_exported", False)),
        },
        "notes": "Real metadata-only split manifest from approved re-keyed tokenised bundle; no source ids or source-id mapping exported.",
    }

    secondary = sources.get("secondary") or {}
    if secondary.get("split_index_paths"):
        sec_index_paths = _split_paths(secondary, "split_index_paths")
        sec_h5_paths = _split_paths(secondary, "h5_paths") if secondary.get("h5_paths") else {}
        manifest["external_validation"] = {
            "name": secondary.get("name"),
            "role": secondary.get("role"),
            "source_kind": secondary.get("source_kind"),
            "sequence_counts": {split: _count_jsonl(path) for split, path in sec_index_paths.items()},
            "source_index_paths": sec_index_paths,
            "source_h5_paths": sec_h5_paths,
        }
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create Clinical-JEPA split manifest")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-hashed-lists", action="store_true")
    ap.add_argument("--synthetic-patients", type=int, default=120)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.dataset_config)
    outdir = ensure_dir(args.output_dir)

    if not args.dry_run:
        manifest = build_real_manifest(cfg, args.seed)
        validate_artifact("split-manifest", manifest)
        write_json(outdir / "split-manifest.json", manifest)
        write_json(outdir / "split-real-report.json", {
            "created_utc": now_utc(),
            "dataset": manifest["dataset"],
            "unit": manifest.get("unit"),
            "counts": manifest["counts"],
            "external_validation_counts": manifest.get("external_validation", {}).get("sequence_counts"),
            "aggregate_only": True,
        })
        print(json.dumps({"manifest": str(outdir / "split-manifest.json"), "counts": manifest["counts"], "dry_run": False}, indent=2))
        return 0

    split_cfg = cfg.get("split", {})
    train = float(split_cfg.get("train_fraction") or 0.70)
    dev = float(split_cfg.get("dev_fraction") or 0.15)
    ids = synthetic_patient_ids(args.synthetic_patients)
    raw_splits = split_ids(ids, args.seed, train, dev)
    salt_env = split_cfg.get("salt_env_var", "CLINICAL_JEPA_SPLIT_SALT") or "CLINICAL_JEPA_SPLIT_SALT"
    salt = os.environ.get(salt_env, "synthetic-salt-not-for-real-use")
    hashed = {split: [stable_hmac(pid, salt) for pid in vals] for split, vals in raw_splits.items()}
    manifest = build_manifest(cfg, hashed, args.seed, salt_exported=False)
    manifest["dry_run"] = True
    manifest["synthetic_patients"] = args.synthetic_patients
    validate_artifact("split-manifest", manifest)
    write_json(outdir / "split-manifest.json", manifest)

    if args.write_hashed_lists:
        for split, vals in hashed.items():
            with (outdir / f"{split}.hashes.local.jsonl").open("w") as f:
                for h in vals:
                    f.write(json.dumps({"patient_hash": h, "split": split}) + "\n")
    write_json(outdir / "split-dry-run-report.json", {"created_utc": now_utc(), "counts": manifest["counts"], "dry_run": True})
    print(json.dumps({"manifest": str(outdir / "split-manifest.json"), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
