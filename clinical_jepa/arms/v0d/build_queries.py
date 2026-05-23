from __future__ import annotations

import argparse
import json
import sys
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, write_json
from clinical_jepa.validation import validate_artifact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build v0D synthetic query descriptors")
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    arms = load_yaml(args.arms_config)
    _targets = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    configured = arms.get("v0D_query_baseline", {}).get("query_descriptors", [])
    queries = []
    for q in configured:
        if q.get("enabled", True) is False:
            continue
        queries.append({
            "query_id": q["query_id"],
            "target_type": q.get("target_type", "T0"),
            "horizon": q.get("horizon", "predeclared-horizon-name"),
            "label_source": "target-block-derived-prefix-safe",
            "task_type": q.get("task_type", "binary"),
            "leakage_notes": "synthetic descriptor; aggregate only",
        })
    descriptor = {"schema_version": "clinical-jepa-query-descriptors-v0", "created_utc": now_utc(), "dry_run": args.dry_run, "queries": queries}
    validate_artifact("query-descriptors", descriptor)
    write_json(outdir / "query-descriptors.json", descriptor)
    print(json.dumps({"query_descriptors": str(outdir / "query-descriptors.json"), "n_queries": len(queries)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
