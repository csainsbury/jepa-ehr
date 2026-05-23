#!/usr/bin/env python3
from __future__ import annotations

import argparse, collections, json, random, time
from pathlib import Path
from typing import Any

import h5py


def load_id_to_token(path: Path) -> dict[int, str]:
    raw=json.loads(path.read_text())
    if 'id_to_token' in raw: return {int(k):str(v) for k,v in raw['id_to_token'].items()}
    if 'token_to_id' in raw: return {int(v):str(k) for k,v in raw['token_to_id'].items()}
    return {int(k):str(v) for k,v in raw.items() if str(k).isdigit()}

def fam(tok: str, kind: str) -> str|None:
    pref={'med':'MED:','lab':'LAB:','state':'STATE:'}[kind]
    if not tok.startswith(pref): return None
    parts=tok.split(':')
    return ':'.join(parts[:2]) if len(parts)>=2 else tok

def first_label(toks, kind):
    for t in toks:
        x=fam(t,kind)
        if x: return x
    return None

def last_label(toks, kind):
    for t in reversed(toks):
        x=fam(t,kind)
        if x: return x
    return None

def mode_label(toks, kind):
    c=collections.Counter(x for t in toks if (x:=fam(t,kind)))
    return c.most_common(1)[0][0] if c else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--target-blocks', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-blocks', type=int, default=0)
    args=ap.parse_args()
    root=Path(args.dataset_root)
    id_to_token=load_id_to_token(root/'schema/vocab/vocabulary.json')
    blocks=[b for b in json.loads(Path(args.target_blocks).read_text()).get('blocks',[]) if b.get('target_type')=='T0']
    if args.max_blocks: blocks=blocks[:args.max_blocks]
    cache={}
    rows=[]
    try:
        for b in blocks:
            path=str(b['sequence_file'])
            if path not in cache: cache[path]=h5py.File(path,'r')
            h5=cache[path]
            group=str(b.get('sequence_group') or b.get('sequence_id'))
            arr=h5[group]['token_ids'][:]
            c0=max(0,int(b.get('context_start_ref',0))); c1=min(len(arr)-1,int(b['context_end_ref']))
            t0=max(0,int(b['target_start_ref'])); t1=min(len(arr)-1,int(b['target_end_ref']))
            ctx=[id_to_token.get(int(x),'') for x in arr[c0:c1+1]]
            tgt=[id_to_token.get(int(x),'') for x in arr[t0:t1+1]]
            rec={'split':b.get('split')}
            for k in ['med','lab','state']:
                rec[f'{k}_label']=first_label(tgt,k)
                rec[f'{k}_final_context']=fam(ctx[-1],k) if ctx else None
                rec[f'{k}_last_context']=last_label(ctx,k)
                rec[f'{k}_mode_context']=mode_label(ctx,k)
            rows.append(rec)
    finally:
        for f in cache.values(): f.close()
    metrics=[]
    rng=random.Random(20260523)
    for kind in ['med','lab','state']:
        train_labels=[r[f'{kind}_label'] for r in rows if r['split']=='train' and r[f'{kind}_label']]
        prior=collections.Counter(train_labels).most_common(1)[0][0] if train_labels else None
        for split in ['dev','test']:
            eval_rows=[r for r in rows if r['split']==split and r[f'{kind}_label']]
            labels=[r[f'{kind}_label'] for r in eval_rows]
            if not labels: continue
            shuffled=labels[:]; rng.shuffle(shuffled)
            for baseline,key in [('empirical_prior',None),('final_context_same_family',f'{kind}_final_context'),('last_context_same_family',f'{kind}_last_context'),('mode_context_same_family',f'{kind}_mode_context')]:
                preds=[]
                for r in eval_rows:
                    p=prior if key is None else (r.get(key) or prior)
                    preds.append(p)
                metrics.append({'task':f'next_{kind}_family','split':split,'baseline':baseline,'n':len(labels),'coverage':sum(1 for r in eval_rows if key and r.get(key)) if key else len(labels),'top1_accuracy':sum(p==y for p,y in zip(preds,labels))/len(labels)})
                if key:
                    metrics.append({'task':f'next_{kind}_family','split':split,'baseline':baseline+'_permuted_label_control','n':len(labels),'coverage':sum(1 for r in eval_rows if r.get(key)),'top1_accuracy':sum(p==y for p,y in zip(preds,shuffled))/len(labels)})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    report={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'n_blocks':len(blocks),'metrics':metrics,'aggregate_only':True,'notes':'Context-family controls only; no token sequences, source ids, or patient examples written.'}
    (out/'context-family-controls.json').write_text(json.dumps(report,indent=2))
    lines=['# B1a context-family controls','',f'T0 blocks scanned: {len(blocks)}','']
    for m in metrics:
        if 'permuted' not in m['baseline']:
            lines.append(f"- {m['task']} / {m['split']} / {m['baseline']}: top1={m['top1_accuracy']:.3f}, coverage={m['coverage']}/{m['n']}")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(out/'context-family-controls.json'),'metrics':len(metrics)},indent=2))

if __name__=='__main__': main()
