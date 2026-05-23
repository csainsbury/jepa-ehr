from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text())
    return data or {}


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def stable_hmac(value: str, salt: str) -> str:
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()


def synthetic_patient_ids(n: int) -> list[str]:
    return [f"SYN{idx:06d}" for idx in range(1, n + 1)]


def split_ids(ids: list[str], seed: int, train: float, dev: float) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train))
    n_dev = int(round(n * dev))
    n_train = min(max(n_train, 0), n)
    n_dev = min(max(n_dev, 0), n - n_train)
    return {
        "train": ids[:n_train],
        "dev": ids[n_train : n_train + n_dev],
        "test": ids[n_train + n_dev :],
    }


def is_placeholder_path(value: str | None) -> bool:
    if not value:
        return True
    return "PLACEHOLDER" in value or value.startswith("/PLACEHOLDER")


def require_pass_leakage(report_path: str | Path) -> dict[str, Any]:
    report = read_json(report_path)
    if report.get("overall_status") != "pass":
        raise SystemExit(f"Leakage audit is not passing: {report_path}")
    return report
