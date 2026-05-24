#!/usr/bin/env bash
set -euo pipefail
cd /workspace/clinical-jepa-autonomous-run
RUN_ID="b1a-transformer-jepa-compare-all-$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="run-workspace/state/task-work/clinical-jepa-pilot/v0/real-b1a-pilot-70k"
RUNDIR="run-workspace/state/task-work/clinical-jepa-pilot/v0/autonomous-runs/${RUN_ID}"
mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"
exec > >(tee -a "$LOG") 2>&1
PY=".venv/bin/python"

echo "RUN_ID=$RUN_ID"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="$RUN_ID" RUNDIR="$RUNDIR" PYTHONPATH=. "$PY" - <<'PY'
from pathlib import Path
import json, os, time
ROOT=Path('run-workspace/state/task-work/clinical-jepa-pilot/v0/real-b1a-pilot-70k')
run_id=os.environ['RUN_ID']
outdir=Path(os.environ['RUNDIR'])

def case_from_path(p: Path) -> str:
    s=str(p)
    if '/external-validation/inspect/gap0/' in s: return 'inspect_gap0'
    if '/external-validation/inspect/gap16/' in s: return 'inspect_gap16'
    if '/external-validation/inspect/gap64/' in s: return 'inspect_gap64'
    if '/horizon-gap/gap16/' in s: return 'mimic_gap16'
    if '/horizon-gap/gap64/' in s: return 'mimic_gap64'
    return 'mimic_gap0'

def model_from_path(p: Path) -> str:
    for part in p.parts:
        if part.startswith('v0B-') and part.endswith('-probes-util-matched'):
            return part[len('v0B-'):-len('-probes-util-matched')]
    return 'unknown'

def family(model: str) -> str:
    if model.startswith('transformer-ema-'): return 'transformer_ema'
    if model.startswith('scaled-') or model in {'256d-50k','384d-40k'}: return 'mean_token_scaled'
    return 'other'

results={}
for p in ROOT.glob('**/candidate-normalized-min128-max512/retrieval-metrics.json'):
    model=model_from_path(p)
    if model == 'unknown':
        continue
    d=json.loads(p.read_text()); r=d.get('overall',{})
    case=case_from_path(p)
    item={
        'model': model,
        'family': family(model),
        'case': case,
        'recall_at_10': r.get('recall_at_10'),
        'mrr': r.get('mrr'),
        'median_rank': r.get('median_rank'),
        'n': r.get('n'),
        'skipped_no_candidates': d.get('skipped_no_candidates'),
        'path': str(p),
    }
    model_dir=p.parents[1]
    case_root=model_dir.parent
    ctrl=case_root/'controls/query-time'/f"v0B-{model}-probes-util-matched-candidate-normalized-min128-max512"/'retrieval-shuffle-control.json'
    if ctrl.exists():
        c=json.loads(ctrl.read_text()); cs=c.get('control_summary',{})
        item['target_shuffle_recall_at_10_mean']=cs.get('target_shuffle',{}).get('recall_at_10_mean')
        item['query_shuffle_recall_at_10_mean']=cs.get('query_shuffle',{}).get('recall_at_10_mean')
        item['time_shift_recall_at_10']=cs.get('time_shift',{}).get('recall_at_10')
    results[f'{model}|{case}']=item

cases=['mimic_gap0','mimic_gap16','mimic_gap64','inspect_gap0','inspect_gap16','inspect_gap64']
by_case={}
for case in cases:
    rows=[v for v in results.values() if v['case']==case and v.get('recall_at_10') is not None]
    rows.sort(key=lambda x: x['recall_at_10'], reverse=True)
    by_case[case]=rows

summary={
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'run_id': run_id,
    'aggregate_only': True,
    'results': results,
    'best_by_case': {case: rows[:5] for case, rows in by_case.items()},
}
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'transformer-jepa-comparison.json').write_text(json.dumps(summary, indent=2))
lines=['# Clinical-JEPA v0 candidate-normalized comparison', '', f'Run id: {run_id}', '', 'Aggregate-only comparison of available mean-token and transformer+EMA candidate-normalized retrieval results.', '']
for case in cases:
    lines += [f'## {case}', '', '| rank | model | family | R@10 | MRR | n | target-shuffle R@10 |', '|---:|---|---|---:|---:|---:|---:|']
    for i,row in enumerate(by_case.get(case, [])[:10],1):
        ts=row.get('target_shuffle_recall_at_10_mean')
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {} | {} |'.format(i,row['model'],row['family'],row['recall_at_10'],row['mrr'],row.get('n'), '' if ts is None else f'{ts:.4f}'))
    lines.append('')
lines += ['## Provisional interpretation', '', '- Use INSPECT gap16/gap64 and candidate-normalized retrieval as the transfer robustness gate.', '- If transformer+EMA variants remain below the scaled mean-token scaffold on INSPECT, keep transformer+EMA as an architecture diagnostic rather than the main v0B evidence line.', '- Controls should remain far below observed retrieval before any promotion claim.', '']
(outdir/'transformer-jepa-comparison.md').write_text('\n'.join(lines))
print(json.dumps({'summary': str(outdir/'transformer-jepa-comparison.md'), 'n_results': len(results)}, indent=2))
PY

echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
