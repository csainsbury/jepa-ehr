#!/usr/bin/env python3
"""Strengthen the collapse-aligned directional-variance penalty from 1 to 5."""
from __future__ import annotations
import json, torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
G=1102; SEEDS=(2101,2102,2103); READOUT=2102; WEIGHT=5.0; FLOOR=0.005

def main():
 torch.use_deterministic_algorithms(True); torch.set_num_threads(1); train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G); probe=generate_factor_split(G,PROBE_FIT,2048); cal=generate_factor_split(G,CAL_OOD,2048); transform=fit_stage0_time_transform(train); reference,digest=threshold_reference_report(); results={}; comp=[]
 for seed in SEEDS:
  c=train_l0_decoupled_seeds(train,transform,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR); freeze_encoder(c); probes,head,_=fit_condition_readouts(c,probe,transform,READOUT); ev,_=evaluate_readouts(c,probes,head,cal,transform); ba=[f['balanced_accuracy'] for f in ev['factors']]; rows,_=trained_collapse_diagnostics(c,cal,transform,reference); comp.append(ba[0]); results[str(seed)]={'factor_ba':ba,'collapse_all_pass':all(r['both_metrics_pass'] for r in rows),'collapse_rows':rows,'training_loss_summary':c.training['losses']}
 checks={'collapse_all_three':all(v['collapse_all_pass'] for v in results.values()),'composition_ba_at_least_0_80_all_three':min(comp)>=.80,'composition_ba_range_at_most_0_05':max(comp)-min(comp)<=.05}
 print(json.dumps({'schema':'BP011-J04C-L0-DIRECTIONAL-VARIANCE-WEIGHT5-V1','contract':{'axis':'raise collapse-aligned directional-variance weight from 1 to 5 at fixed floor 0.005','pass':'collapse 3/3, composition BA >=0.80 in 3/3, range <=0.05','not_tested':'direct superiority or target-family changes'},'generator_seed':G,'model_seeds':SEEDS,'common_readout_seed':READOUT,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'composition_ba':comp,'composition_ba_range':max(comp)-min(comp),'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'PUBLIC_SYNTHETIC_DIRECTIONAL_ANTICOLLAPSE_INTERVENTION'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
