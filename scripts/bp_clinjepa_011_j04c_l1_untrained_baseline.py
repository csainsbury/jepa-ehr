#!/usr/bin/env python3
"""Fable-required eval-only untrained-encoder baseline for the fixed-generator L1 recipe beta."""
from __future__ import annotations
import json,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split,make_r0_encoder
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,evaluate_readouts,fit_condition_readouts,freeze_encoder
G=1102;MODELS=(2101,2102,2103);READOUT=2102

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);results={};values=[[],[],[]]
 for seed in MODELS:
  c=TrainedCondition('R0_INIT',make_r0_encoder(seed),None,None,{'attempted_steps':0,'successful_steps':0,'optimizer_steps':0,'ema_updates':0,'losses':None});freeze_encoder(c);probes,head,fit=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']]
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'readout_fit':fit}
 print(json.dumps({'schema':'BP011-J04C-L1-UNTRAINED-ENCODER-BASELINE-V1','contract':{'role':'eval-only baseline before opening L1 3x3 panel','interpretation_rule':'trained minus untrained mean BA must be >=0.05 per factor for useful learned signal / recipe-viability language; otherwise only signal-preserving collapse-floored pipeline','not_training':True},'generator_seed':G,'model_seeds':MODELS,'common_readout_seed':READOUT,'factor_order':['composition','order','time'],'untrained_factor_ba_by_factor':values,'untrained_factor_ba_means':[sum(v)/len(v) for v in values],'results':results,'claim_ceiling':'PUBLIC_SYNTHETIC_EVAL_ONLY_BASELINE'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
