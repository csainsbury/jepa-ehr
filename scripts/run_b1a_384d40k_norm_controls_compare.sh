#!/usr/bin/env bash
set -euo pipefail
cd /workspace/clinical-jepa-autonomous-run
RUN_ID="b1a-384d40k-normalized-controls-compare-$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="run-workspace/state/task-work/clinical-jepa-pilot/v0/real-b1a-pilot-70k"
EXT="$ROOT/external-validation/inspect"
RUNDIR="run-workspace/state/task-work/clinical-jepa-pilot/v0/autonomous-runs/${RUN_ID}"
mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"
exec > >(tee -a "$LOG") 2>&1
PY=".venv/bin/python"
MODEL="v0B-384d-40k-probes-util-matched"
POLICY="same_split_target_type_len_seq_util_bin"
MIN=128
MAX=512

echo "RUN_ID=$RUN_ID"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Model=$MODEL Policy=$POLICY min=$MIN max=$MAX"

run_case() {
  local label="$1" case_root="$2" targets="$3"
  local dir="$case_root/$MODEL"
  if [[ ! -f "$dir/v0b-context-pred.fp16.npy" ]]; then
    echo "Missing embeddings for $label at $dir" >&2
    return 1
  fi
  echo
  echo "## $label candidate-normalized retrieval"
  PYTHONPATH=. "$PY" -m clinical_jepa.eval.retrieval \
    --query-embeddings "$dir/v0b-context-pred.fp16.npy" \
    --query-index "$dir/embedding-index.jsonl" \
    --target-embeddings "$dir/v0b-target-mean.fp16.npy" \
    --target-index "$dir/embedding-index.jsonl" \
    --output-dir "$dir/candidate-normalized-min${MIN}-max${MAX}" \
    --distractor-policy "$POLICY" \
    --min-candidates-per-group "$MIN" \
    --max-candidates-per-group "$MAX" \
    --batch-size 512

  echo
  echo "## $label query/target/time-shift controls"
  mkdir -p "$case_root/controls/query-time"
  PYTHONPATH=. "$PY" scripts/run_retrieval_shuffle_control.py \
    --query-embeddings "$dir/v0b-context-pred.fp16.npy" \
    --query-index "$dir/embedding-index.jsonl" \
    --target-embeddings "$dir/v0b-target-mean.fp16.npy" \
    --target-index "$dir/embedding-index.jsonl" \
    --target-blocks "$targets" \
    --output-dir "$case_root/controls/query-time/$MODEL-candidate-normalized-min${MIN}-max${MAX}" \
    --distractor-policy "$POLICY" \
    --min-candidates-per-group "$MIN" \
    --max-candidates-per-group "$MAX" \
    --n-shuffles 3
}

run_case mimic_gap0 "$ROOT" "$ROOT/target-blocks/target-block-manifest.json"
run_case mimic_gap16 "$ROOT/horizon-gap/gap16" "$ROOT/horizon-gap/gap16/target-blocks/target-block-manifest.json"
run_case mimic_gap64 "$ROOT/horizon-gap/gap64" "$ROOT/horizon-gap/gap64/target-blocks/target-block-manifest.json"
run_case inspect_gap0 "$EXT/gap0" "$EXT/gap0/target-blocks/target-block-manifest.json"
run_case inspect_gap16 "$EXT/gap16" "$EXT/gap16/target-blocks/target-block-manifest.json"
run_case inspect_gap64 "$EXT/gap64" "$EXT/gap64/target-blocks/target-block-manifest.json"

echo

echo "## refresh comparison"
bash scripts/run_b1a_transformer_jepa_compare_all.sh

echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
