#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="${1:-/tmp/clinical_jepa_schema_check}"
rm -rf "$WORK"
mkdir -p "$WORK"/{splits,target-blocks,leakage,results,scenario,pseudo,autoreg,v0A,v0B,v0D}
DATASET="configs/v0/dataset.example.yaml"
ARMS="configs/v0/arms.example.yaml"
METRICS="configs/v0/metrics.example.yaml"
SCENARIO="configs/v0/scenario.example.yaml"
METADATA_REQ="configs/v0/metadata_availability.example.yaml"

python3 -m compileall -q clinical_jepa
python3 -m unittest discover -s tests
python3 -m clinical_jepa.splits.make_manifest --dataset-config "$DATASET" --output-dir "$WORK/splits" --dry-run --write-hashed-lists --synthetic-patients 36
python3 -m clinical_jepa.validation --file "$WORK/splits/split-manifest.json"
python3 -m clinical_jepa.targets.extract_blocks --dataset-config "$DATASET" --split-manifest "$WORK/splits/split-manifest.json" --arms-config "$ARMS" --output-dir "$WORK/target-blocks" --targets T0 T1 --dry-run
python3 -m clinical_jepa.validation --file "$WORK/target-blocks/target-block-manifest.json"
python3 -m clinical_jepa.audit.run_leakage_audit --dataset-config "$DATASET" --split-manifest "$WORK/splits/split-manifest.json" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --output "$WORK/leakage/leakage-audit-report.json"
python3 -m clinical_jepa.validation --file "$WORK/leakage/leakage-audit-report.json"
python3 - <<'PY' "$WORK/target-blocks/target-block-manifest.json" "$WORK/scenario/metadata.json"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
rows = []
for block in manifest["blocks"]:
    is_t1 = block.get("target_type") == "T1"
    rows.append({
        "block_id": block["block_id"],
        "context_med_count": 1,
        "context_lab_count": 2,
        "context_state_count": 0,
        "target_med_count": int(is_t1),
        "target_lab_count": 1,
        "target_state_count": 0,
        "equivalent_contact_present": int(not is_t1),
        "negative_control_present": 0,
        "scenario_consistent": True,
        "sequence_len": 96,
        "contact_count": 4,
    })
out.write_text(json.dumps({"rows": rows}, indent=2))
PY
python3 -m clinical_jepa.tte.audit_metadata_availability --requirements "$METADATA_REQ" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --scenario-card "$SCENARIO" --metadata-index "$WORK/scenario/metadata.json" --output-dir "$WORK/scenario"
python3 -m clinical_jepa.validation --file "$WORK/scenario/metadata-availability.json"
python3 -m clinical_jepa.tte.scan_feasibility --target-blocks "$WORK/target-blocks/target-block-manifest.json" --scenario-card "$SCENARIO" --metadata-index "$WORK/scenario/metadata.json" --metadata-availability-report "$WORK/scenario/metadata-availability.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/scenario"
python3 -m clinical_jepa.validation --file "$WORK/scenario/scenario-feasibility.json"
python3 - <<'PY' "$WORK/pseudo"
import json
import sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
arr = np.eye(4, dtype=np.float32)
np.save(out / "query.npy", arr)
np.save(out / "target.npy", arr)
rows = []
for i in range(4):
    rows.append({
        "block_id": f"synthetic-block-{i}",
        "patient_hash": f"synthetic-patient-{i}",
        "split": "dev",
        "target_type": "T0",
        "context_len": 16,
        "target_len": 16,
        "sequence_len": 64,
        "context_med_count": 1,
        "context_lab_count": 2,
        "context_state_count": 0,
        "target_med_count": int(i % 2 == 0),
        "target_lab_count": 1,
        "target_state_count": 0,
        "scenario_consistent": i != 3,
        "negative_control_present": i == 3,
    })
for name in ["query-index.jsonl", "target-index.jsonl"]:
    (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
PY
python3 -m clinical_jepa.eval.pseudo_rendering --query-embeddings "$WORK/pseudo/query.npy" --query-index "$WORK/pseudo/query-index.jsonl" --target-embeddings "$WORK/pseudo/target.npy" --target-index "$WORK/pseudo/target-index.jsonl" --output-dir "$WORK/pseudo" --distractor-policy same_split_target_type --top-k 2
python3 -m clinical_jepa.validation --file "$WORK/pseudo/pseudo-rendering-readiness.json"
python3 - <<'PY' "$WORK/autoreg"
import json
import sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
arr = np.stack([np.eye(4, dtype=np.float32), np.roll(np.eye(4, dtype=np.float32), shift=1, axis=1)], axis=1)
np.save(out / "predicted-rollout.npy", arr)
np.save(out / "target-rollout.npy", arr)
rows = []
for i in range(4):
    rows.append({
        "block_id": f"synthetic-block-{i}",
        "patient_hash": f"synthetic-patient-{i}",
        "split": "dev",
        "target_type": "T1",
        "context_len": 16,
        "target_len": 8,
        "sequence_len": 64,
        "context_med_count": 1,
        "context_lab_count": 2,
        "context_state_count": 0,
    })
(out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
PY
python3 -m clinical_jepa.eval.autoregression_readiness --predicted-rollout "$WORK/autoreg/predicted-rollout.npy" --target-rollout "$WORK/autoreg/target-rollout.npy" --index "$WORK/autoreg/index.jsonl" --output-dir "$WORK/autoreg" --distractor-policy same_split_target_type --control-mode all
python3 -m clinical_jepa.validation --file "$WORK/autoreg/autoregression-readiness.json"
python3 -m clinical_jepa.eval.run_metrics --metrics-config "$METRICS" --split-manifest "$WORK/splits/split-manifest.json" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/results" --dry-run
python3 -m clinical_jepa.validation --kind metrics-bundle --file "$WORK/results/baseline-results.json"
python3 -m clinical_jepa.validation --kind metrics-bundle --file "$WORK/results/retrieval-results.json"
python3 -m clinical_jepa.arms.v0d.build_queries --arms-config "$ARMS" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --output-dir "$WORK/v0D" --dry-run
python3 -m clinical_jepa.validation --file "$WORK/v0D/query-descriptors.json"
python3 -m clinical_jepa.arms.v0d.train_query_baseline --dataset-config "$DATASET" --query-descriptors "$WORK/v0D/query-descriptors.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0D" --dry-run
python3 -m clinical_jepa.arms.v0b.train_minimal_jepa --arms-config "$ARMS" --dataset-config "$DATASET" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0B" --dry-run
python3 -m clinical_jepa.validation --file "$WORK/v0B/train-manifest.json"
python3 -m clinical_jepa.arms.v0a.extract_flatascend_embeddings --arms-config "$ARMS" --target-blocks "$WORK/target-blocks/target-block-manifest.json" --leakage-report "$WORK/leakage/leakage-audit-report.json" --output-dir "$WORK/v0A" --dry-run
python3 -m clinical_jepa.validation --file "$WORK/v0A/embedding-cache-manifest.json"
python3 -m clinical_jepa.arms.v0a.train_predictor --embedding-manifest "$WORK/v0A/embedding-cache-manifest.json" --variant linear,mlp --output-dir "$WORK/v0A" --dry-run

echo "Schema validation checks passed. Artifacts: $WORK"
find "$WORK" -maxdepth 3 -type f | sort
