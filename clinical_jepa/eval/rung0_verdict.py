"""Rung-0 verdict CLI — load exported sidecars → per-source hierarchy decision.

Loads the coarse/fine latent sidecars written by ``export_coarse_fine_latents`` for
each source × frozen horizon, assembles the ``evaluate_source`` bundle, and emits the
aggregate-only per-source three-way verdict (Pi R5). The raw-count corroboration and
the drift-sufficiency co-gate are supplied as per-source flags (computed by the
count-route and ``drift_ablation`` steps) — a BUILD requires them true, so a run
without them can at best report NO-BUILD_INCONCLUSIVE (honest, never a false BUILD).

Layout: ``<sidecar_root>/<source>/W<W>/<granularity>_{queries,targets}.npy`` + ``_index.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import read_jsonl
from clinical_jepa.eval.rung0_horizon_decay import GRANULARITIES, build_manifest, evaluate_source
from clinical_jepa.utils import ensure_dir, write_json


def _wtag(W: float) -> str:
    return f"W{('%g' % float(W))}"


def load_bundle(sidecar_root: str | Path, source: str, horizons: list[float]) -> dict[float, dict[str, dict[str, Any]]]:
    root = Path(sidecar_root) / source
    per_W: dict[float, dict[str, dict[str, Any]]] = {}
    for W in horizons:
        d = root / _wtag(W)
        cells: dict[str, dict[str, Any]] = {}
        for g in GRANULARITIES:
            q, t, ix = d / f"{g}_queries.npy", d / f"{g}_targets.npy", d / f"{g}_index.jsonl"
            if q.exists() and t.exists() and ix.exists():
                cells[g] = {"queries": np.load(q).astype(np.float32),
                            "targets": np.load(t).astype(np.float32),
                            "index": read_jsonl(ix)}
        if cells:
            per_W[float(W)] = cells
    return per_W


def run_verdict(
    sidecar_root: str | Path,
    sources_cfg: dict[str, dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 20260523,
    max_candidates: int = 200,
    adequacy_floor: int = 500,
) -> dict[str, Any]:
    """sources_cfg = {source: {horizons: [...], level_horizons: [...], raw_count_ok: bool,
    sufficiency_ok: bool, veto: bool}}."""
    verdicts = []
    for source, cfg in sources_cfg.items():
        per_W = load_bundle(sidecar_root, source, cfg["horizons"])
        if not per_W:
            verdicts.append({"source": source, "decision": "NO-BUILD_INCONCLUSIVE",
                             "reason": "no sidecars found", "aggregate_only": True})
            continue
        verdicts.append(evaluate_source(
            source, per_W, level_horizons=[float(w) for w in cfg["level_horizons"]],
            n_boot=n_boot, seed=seed, max_candidates=max_candidates, adequacy_floor=adequacy_floor,
            raw_count_ok=bool(cfg.get("raw_count_ok", False)),
            sufficiency_ok=bool(cfg.get("sufficiency_ok", False)),
            veto=bool(cfg.get("veto", False)),
        ))
    return build_manifest(verdicts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-0 per-source hierarchy verdict from exported sidecars")
    ap.add_argument("--sidecar-root", required=True)
    ap.add_argument("--sources-config", required=True, help="JSON: {source: {horizons, level_horizons, raw_count_ok, sufficiency_ok, veto}}")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--adequacy-floor", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    sources_cfg = json.loads(Path(args.sources_config).read_text())
    manifest = run_verdict(args.sidecar_root, sources_cfg, n_boot=args.n_boot, seed=args.seed,
                           max_candidates=args.max_candidates, adequacy_floor=args.adequacy_floor)
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "rung0-verdict-manifest.json", manifest)
    print(json.dumps({"output": str(outdir / "rung0-verdict-manifest.json"), "decisions": manifest["decisions"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
