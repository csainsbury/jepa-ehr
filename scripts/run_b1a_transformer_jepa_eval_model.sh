#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <model-label> <checkpoint-path>" >&2
  exit 2
fi
MODEL_LABEL="$1"
CKPT="$2"
cd /workspace/clinical-jepa-autonomous-run
RUN_ID="b1a-transformer-jepa-eval-${MODEL_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="run-workspace/state/task-work/clinical-jepa-pilot/v0/real-b1a-pilot-70k"
EXT="$ROOT/external-validation/inspect"
RUNDIR="run-workspace/state/task-work/clinical-jepa-pilot/v0/autonomous-runs/${RUN_ID}"
mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"
exec > >(tee -a "$LOG") 2>&1
PY=".venv/bin/python"
DATASET="run-workspace/state/task-work/clinical-jepa-pilot/configs/v0/dataset.yaml"
POLICY="same_split_target_type_len_seq_util_bin"
MAX_MAIN=4096
MIN_NORM=128
MAX_NORM=512

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'MODEL_LABEL=%s\n' "$MODEL_LABEL"
printf 'CHECKPOINT=%s\n' "$CKPT"
printf 'START_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Policy=%s main_max=%s normalized_min=%s normalized_max=%s\n' "$POLICY" "$MAX_MAIN" "$MIN_NORM" "$MAX_NORM"

run_eval_case() {
  local case_label="$1" case_root="$2" targets="$3"
  local out_name="v0B-${MODEL_LABEL}-probes-util-matched"
  local out_dir="$case_root/$out_name"
  echo
  echo "## $MODEL_LABEL $case_label transformer+EMA retrieval"
  echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  PYTHONPATH=. "$PY" scripts/eval_transformer_jepa_probe.py \
    --checkpoint "$CKPT" \
    --dataset-config "$DATASET" \
    --target-blocks "$targets" \
    --output-dir "$out_dir" \
    --max-blocks 60000 \
    --batch-size 256 \
    --max-context-tokens 256 \
    --max-target-tokens 64 \
    --retrieval-policy "$POLICY" \
    --retrieval-max-candidates "$MAX_MAIN"

  echo
  echo "## $MODEL_LABEL $case_label candidate-normalized retrieval"
  PYTHONPATH=. "$PY" -m clinical_jepa.eval.retrieval \
    --query-embeddings "$out_dir/v0b-context-pred.fp16.npy" \
    --query-index "$out_dir/embedding-index.jsonl" \
    --target-embeddings "$out_dir/v0b-target-mean.fp16.npy" \
    --target-index "$out_dir/embedding-index.jsonl" \
    --output-dir "$out_dir/candidate-normalized-min${MIN_NORM}-max${MAX_NORM}" \
    --distractor-policy "$POLICY" \
    --min-candidates-per-group "$MIN_NORM" \
    --max-candidates-per-group "$MAX_NORM" \
    --batch-size 512

  echo
  echo "## $MODEL_LABEL $case_label query/target/time-shift controls"
  mkdir -p "$case_root/controls/query-time"
  PYTHONPATH=. "$PY" scripts/run_retrieval_shuffle_control.py \
    --query-embeddings "$out_dir/v0b-context-pred.fp16.npy" \
    --query-index "$out_dir/embedding-index.jsonl" \
    --target-embeddings "$out_dir/v0b-target-mean.fp16.npy" \
    --target-index "$out_dir/embedding-index.jsonl" \
    --target-blocks "$targets" \
    --output-dir "$case_root/controls/query-time/$out_name-candidate-normalized-min${MIN_NORM}-max${MAX_NORM}" \
    --distractor-policy "$POLICY" \
    --min-candidates-per-group "$MIN_NORM" \
    --max-candidates-per-group "$MAX_NORM" \
    --n-shuffles 3
}

run_eval_case mimic_gap0 "$ROOT" "$ROOT/target-blocks/target-block-manifest.json"
run_eval_case mimic_gap16 "$ROOT/horizon-gap/gap16" "$ROOT/horizon-gap/gap16/target-blocks/target-block-manifest.json"
run_eval_case mimic_gap64 "$ROOT/horizon-gap/gap64" "$ROOT/horizon-gap/gap64/target-blocks/target-block-manifest.json"
run_eval_case inspect_gap0 "$EXT/gap0" "$EXT/gap0/target-blocks/target-block-manifest.json"
run_eval_case inspect_gap16 "$EXT/gap16" "$EXT/gap16/target-blocks/target-block-manifest.json"
run_eval_case inspect_gap64 "$EXT/gap64" "$EXT/gap64/target-blocks/target-block-manifest.json"

echo
echo "## consolidate"
PYTHONPATH=. "$PY" - <<PY
from pathlib import Path
import json, time
ROOT=Path("$ROOT"); EXT=Path("$EXT"); RUNDIR=Path("$RUNDIR"); MODEL_LABEL="$MODEL_LABEL"
cases={
  "mimic_gap0": ROOT,
  "mimic_gap16": ROOT/"horizon-gap/gap16",
  "mimic_gap64": ROOT/"horizon-gap/gap64",
  "inspect_gap0": EXT/"gap0",
  "inspect_gap16": EXT/"gap16",
  "inspect_gap64": EXT/"gap64",
}
summary={"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id":"$RUN_ID", "model_label": MODEL_LABEL, "policy":"$POLICY", "main_max_candidates_per_group": int("$MAX_MAIN"), "normalized_min_candidates_per_group": int("$MIN_NORM"), "normalized_max_candidates_per_group": int("$MAX_NORM"), "aggregate_only": True, "results":{}}
for cname,croot in cases.items():
    out_name=f"v0B-{MODEL_LABEL}-probes-util-matched"
    base=croot/out_name
    item={}
    p=base/"v0b-probe-results.json"
    if p.exists():
        d=json.loads(p.read_text()); r=d.get("retrieval",{}).get("overall",{})
        item["main"]={"recall_at_10": r.get("recall_at_10"), "mrr": r.get("mrr"), "median_rank": r.get("median_rank"), "n": r.get("n"), "skipped_no_candidates": d.get("retrieval",{}).get("skipped_no_candidates")}
    p=base/"candidate-normalized-min$MIN_NORM-max$MAX_NORM/retrieval-metrics.json"
    if p.exists():
        d=json.loads(p.read_text()); r=d.get("overall",{})
        item["candidate_normalized"]={"recall_at_10": r.get("recall_at_10"), "mrr": r.get("mrr"), "median_rank": r.get("median_rank"), "n": r.get("n"), "skipped_no_candidates": d.get("skipped_no_candidates"), "candidate_groups": d.get("n_candidate_groups")}
    p=croot/"controls/query-time"/f"{out_name}-candidate-normalized-min$MIN_NORM-max$MAX_NORM"/"retrieval-shuffle-control.json"
    if p.exists():
        d=json.loads(p.read_text()); cs=d.get("control_summary",{})
        item["controls"]={
            "observed_recall_at_10": d.get("observed",{}).get("overall",{}).get("recall_at_10"),
            "target_shuffle_recall_at_10_mean": cs.get("target_shuffle",{}).get("recall_at_10_mean"),
            "query_shuffle_recall_at_10_mean": cs.get("query_shuffle",{}).get("recall_at_10_mean"),
            "time_shift_recall_at_10": cs.get("time_shift",{}).get("recall_at_10"),
            "observed_mrr": d.get("observed",{}).get("overall",{}).get("mrr"),
            "target_shuffle_mrr_mean": cs.get("target_shuffle",{}).get("mrr_mean"),
            "query_shuffle_mrr_mean": cs.get("query_shuffle",{}).get("mrr_mean"),
            "time_shift_mrr": cs.get("time_shift",{}).get("mrr"),
        }
    summary["results"][cname]=item
(RUNDIR/"transformer-jepa-eval-summary.json").write_text(json.dumps(summary, indent=2))
lines=["# Transformer+EMA JEPA evaluation summary", "", f"Run id: $RUN_ID", f"Model: {MODEL_LABEL}", "", f"Policy: $POLICY", f"Candidate-normalized: min $MIN_NORM / max $MAX_NORM", ""]
for key,item in sorted(summary["results"].items()):
    main=item.get("main",{}); norm=item.get("candidate_normalized",{}); ctrl=item.get("controls",{})
    lines.append("- {}: main R@10={}, MRR={}; norm R@10={}, MRR={}, n={}; controls target/query/time R@10={}/{}/{}".format(
        key,
        None if main.get("recall_at_10") is None else f"{main['recall_at_10']:.4f}",
        None if main.get("mrr") is None else f"{main['mrr']:.4f}",
        None if norm.get("recall_at_10") is None else f"{norm['recall_at_10']:.4f}",
        None if norm.get("mrr") is None else f"{norm['mrr']:.4f}",
        norm.get("n"),
        None if ctrl.get("target_shuffle_recall_at_10_mean") is None else f"{ctrl['target_shuffle_recall_at_10_mean']:.4f}",
        None if ctrl.get("query_shuffle_recall_at_10_mean") is None else f"{ctrl['query_shuffle_recall_at_10_mean']:.4f}",
        None if ctrl.get("time_shift_recall_at_10") is None else f"{ctrl['time_shift_recall_at_10']:.4f}",
    ))
(RUNDIR/"transformer-jepa-eval-summary.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"summary": str(RUNDIR/"transformer-jepa-eval-summary.md"), "results": len(summary["results"])}, indent=2))
PY

printf 'END_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
