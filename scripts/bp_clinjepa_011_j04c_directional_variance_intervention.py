#!/usr/bin/env python3
"""Align L0 anti-collapse training with the directional-variance collapse diagnostic."""
from __future__ import annotations
import json
import torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts, fit_condition_readouts, freeze_encoder, threshold_reference_report, trained_collapse_diagnostics
GENERATOR_SEED=1102; MODEL_SEEDS=(2101,2102,2103); COMMON_READOUT_SEED=2102; WEIGHT=1.0; FLOOR=0.005

def main():
 torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
 train=independent_train_nuisance(generate_factor_split(GENERATOR_SEED,TRAIN,8192),GENERATOR_SEED); probe=generate_factor_split(GENERATOR_SEED,PROBE_FIT,2048); cal=generate_factor_split(GENERATOR_SEED,CAL_OOD,2048); transform=fit_stage0_time_transform(train); reference,digest=threshold_reference_report(); results={}; composition=[]
 for seed in MODEL_SEEDS:
  condition=train_l0_decoupled_seeds(train,transform,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR); freeze_encoder(condition); probes,head,_=fit_condition_readouts(condition,probe,transform,COMMON_READOUT_SEED); evaluation,_=evaluate_readouts(condition,probes,head,cal,transform); ba=[f['balanced_accuracy'] for f in evaluation['factors']]; rows,_=trained_collapse_diagnostics(condition,cal,transform,reference); composition.append(ba[0]); results[str(seed)]={'factor_ba':ba,'collapse_all_pass':all(r['both_metrics_pass'] for r in rows),'collapse_rows':rows,'training_loss_summary':condition.training['losses']}
 checks={'collapse_all_three':all(v['collapse_all_pass'] for v in results.values()),'composition_ba_at_least_0_80_all_three':min(composition)>=0.80,'composition_ba_range_at_most_0_05':max(composition)-min(composition)<=0.05}
 print(json.dumps({'schema':'BP011-J04C-L0-DIRECTIONAL-VARIANCE-INTERVENTION-V1','contract':{'axis':'penalize predictor directional variance below 0.005 using the collapse diagnostic normalization','pass':'collapse 3/3, composition BA >=0.80 in 3/3, range <=0.05','not_tested':'direct superiority or target-family changes'},'generator_seed':GENERATOR_SEED,'model_seeds':MODEL_SEEDS,'common_readout_seed':COMMON_READOUT_SEED,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'composition_ba':composition,'composition_ba_range':max(composition)-min(composition),'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'PUBLIC_SYNTHETIC_DIRECTIONAL_ANTICOLLAPSE_INTERVENTION'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__': main()
