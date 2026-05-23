#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from clinical_jepa.arms.v0a.train_predictor import _load_id_to_token, _read_target_labels, _evaluate_task


class MeanJEPA(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d, padding_idx=0)
        self.predictor = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def mean_embed(self, ids):
        mask=(ids!=0).float().unsqueeze(-1)
        emb=self.embedding(ids)*mask
        return emb.sum(dim=1)/mask.sum(dim=1).clamp_min(1.0)
    def forward(self, ctx):
        return self.predictor(self.mean_embed(ctx))


def pad(batch, device):
    max_len=max(len(x) for x in batch)
    out=torch.zeros((len(batch),max_len),dtype=torch.long,device=device)
    for i,arr in enumerate(batch): out[i,:len(arr)]=torch.as_tensor(arr,dtype=torch.long,device=device)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset-config', required=True)
    ap.add_argument('--target-blocks', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-blocks', type=int, default=60000)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--max-context-tokens', type=int, default=128)
    args=ap.parse_args()
    # yaml is only needed for vocab path; avoid logging config contents.
    import yaml
    cfg=yaml.safe_load(Path(args.dataset_config).read_text())
    vocab_json=cfg.get('vocabulary',{}).get('vocab_json_path')
    id_to_token=_load_id_to_token(vocab_json)
    ck=torch.load(args.checkpoint,map_location='cpu')
    model=MeanJEPA(int(ck['vocab_size']), int(ck['embedding_dim']))
    model.load_state_dict(ck['model_state_dict'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    target_manifest=json.loads(Path(args.target_blocks).read_text())
    blocks=[b for b in target_manifest.get('blocks',[]) if b.get('target_type')=='T0' and b.get('sequence_file')]
    if args.max_blocks: blocks=blocks[:args.max_blocks]
    rows=[]; embs=np.zeros((len(blocks), int(ck['embedding_dim'])), dtype=np.float16)
    cache={}
    def read_ctx(b):
        path=str(b['sequence_file'])
        if path not in cache: cache[path]=h5py.File(path,'r')
        h5=cache[path]; group=str(b.get('sequence_group') or b.get('sequence_id'))
        arr=h5[group]['token_ids'][:]
        c0=max(0,int(b.get('context_start_ref',0))); c1=min(len(arr)-1,int(b['context_end_ref']))
        return np.asarray(arr[c0:c1+1][-args.max_context_tokens:], dtype=np.int64)
    try:
        with torch.no_grad():
            for start in range(0,len(blocks),args.batch_size):
                bb=blocks[start:start+args.batch_size]
                ids=[read_ctx(b) for b in bb]
                pred=model(pad(ids,device)).detach().cpu().numpy().astype(np.float16)
                embs[start:start+len(bb)] = pred
                for j,b in enumerate(bb):
                    rows.append({'row':start+j,'block_id':b.get('block_id'),'split':b.get('split'),'target_type':b.get('target_type')})
    finally:
        for f in cache.values(): f.close()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    np.save(out/'v0b-context-pred.fp16.npy', embs)
    (out/'embedding-index.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    labels=_read_target_labels({str(r['block_id']) for r in rows}, target_manifest, id_to_token)
    metrics=[]
    x=embs.astype(np.float32)
    metrics.extend(_evaluate_task(x, rows, labels, 'next_med_family', 'med', 'v0b_pred'))
    metrics.extend(_evaluate_task(x, rows, labels, 'next_lab_family', 'lab', 'v0b_pred'))
    metrics.extend(_evaluate_task(x, rows, labels, 'next_state_family', 'state', 'v0b_pred'))
    report={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoint':args.checkpoint,'embedding_shape':list(embs.shape),'n_labeled_blocks':len(labels),'metrics':metrics,'aggregate_only':True}
    (out/'v0b-probe-results.json').write_text(json.dumps(report,indent=2))
    lines=['# v0B learned-prediction embedding probes','',f'Embedding rows: {len(rows)}',f'Labeled T0 blocks: {len(labels)}','']
    for m in metrics:
        if m.get('baseline')=='ridge_linear_probe':
            lines.append(f"- {m['task']} / {m['split']}: top1={m['top1_accuracy']:.3f}, n={m['n_evaluated']}, classes={m['n_classes_train']}")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(out/'v0b-probe-results.json'),'n_metrics':len(metrics)},indent=2))

if __name__=='__main__': main()
