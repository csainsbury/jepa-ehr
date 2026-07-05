#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="${1:-/tmp/clinical_jepa_synthetic_check}"
rm -rf "$WORK"
mkdir -p "$WORK"/{splits,target-blocks,leakage,results,v0A,v0B,v0D}

# Bootstrap the (gitignored) run-workspace example configs from the tracked
# configs/v0 so the synthetic check runs on a fresh checkout.
WS_CFG="run-workspace/state/task-work/clinical-jepa-pilot/configs/v0"
mkdir -p "$WS_CFG"
cp configs/v0/*.yaml "$WS_CFG"/

DATASET="$WS_CFG/dataset.example.yaml"
ARMS="$WS_CFG/arms.example.yaml"
METRICS="$WS_CFG/metrics.example.yaml"

python3 -m compileall -q clinical_jepa
python3 -m unittest discover -s tests
python3 -m clinical_jepa.splits.make_manifest --dataset-config "$DATASET" --output-dir "$WORK/splits" --dry-run --write-hashed-lists --synthetic-patients 48
python3 -m clinical_jepa.targets.extract_blocks --dataset-config "$DATASET" --split-manifest "$WORK/splits/split-manifest.json" --arms-config "$ARMS" --output-dir "$WORK/target-blocks" --targets T0 T1 --dry-run
python3 -m clinical_jepa.audit.run_leakage_audit --dataset-config "$DATASET" --split-manifest "$WORK/splits/split-manifest.json" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --output "$WORK/leakage/leakage-audit-report.json"
python3 -m clinical_jepa.eval.run_metrics --metrics-config "$METRICS" --split-manifest "$WORK/splits/split-manifest.json" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/results" --dry-run
python3 -m clinical_jepa.arms.v0d.build_queries --arms-config "$ARMS" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --output-dir "$WORK/v0D" --dry-run
python3 -m clinical_jepa.arms.v0d.train_query_baseline --dataset-config "$DATASET" --query-descriptors "$WORK/v0D/query-descriptors.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0D" --dry-run
python3 -m clinical_jepa.arms.v0b.train_minimal_jepa --arms-config "$ARMS" --dataset-config "$DATASET" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0B" --dry-run
python3 -m clinical_jepa.arms.v0a.extract_flatascend_embeddings --arms-config "$ARMS" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0A" --dry-run
python3 -m clinical_jepa.arms.v0a.train_predictor --embedding-manifest "$WORK/v0A/embedding-cache-manifest.json" --variant linear,mlp --output-dir "$WORK/v0A" --dry-run

echo "Synthetic scaffold checks passed. Artifacts: $WORK"
find "$WORK" -maxdepth 3 -type f | sort
