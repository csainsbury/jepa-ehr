#!/usr/bin/env python3
"""Restore generator variation around the passing identity-predictor directional-variance beta."""
from __future__ import annotations
import json, torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
GENERATORS=(1101,1102,1103); MODELS=(2101,2102,2103); READOUT=2102; WEIGHT=5.0; FLOOR=.005

def main():
 torch.use_deterministic_algorithms(True); torch.set_num_threads(1); reference,digest=threshold_reference_report(); results={}; matrix=[]; all_ba=[]
 for g in GENERATORS:
  train=independent_train_nuisance(generate_factor_split(g,TRAIN,8192),g); probe=generate_factor_split(g,PROBE_FIT,2048); cal=generate_factor_split(g,CAL_OOD,2048); transform=fit_stage0_time_transform(train); row=[]
  for seed in MODELS:
   c=train_l0_decoupled_seeds(train,transform,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR); freeze_encoder(c); probes,head,_=fit_condition_readouts(c,probe,transform,READOUT); ev,_=evaluate_readouts(c,probes,head,cal,transform); ba=[f['balanced_accuracy'] for f in ev['factors']]; collapse,_=trained_collapse_diagnostics(c,cal,transform,reference); key=f'g{g}_m{seed}'; results[key]={'factor_ba':ba,'collapse_all_pass':all(r['both_metrics_pass'] for r in collapse),'collapse_rows':collapse}; row.append(ba[0]); all_ba.append(ba[0])
  matrix.append(row)
 checks={'collapse_all_nine':all(v['collapse_all_pass'] for v in results.values()),'composition_ba_at_least_0_80_all_nine':min(all_ba)>=.80,'global_composition_range_at_most_0_06':max(all_ba)-min(all_ba)<=.06,'within_generator_model_range_at_most_0_05':all(max(row)-min(row)<=.05 for row in matrix)}
 print(json.dumps({'schema':'BP011-J04C-L0-DIRECTIONAL-BETA-3X3-V1','contract':{'axis':'restore three generator/data seeds around the passing three-model-seed beta','pass':'collapse 9/9, composition BA >=0.80 in 9/9, global range <=0.06, within-generator model range <=0.05','not_tested':'direct superiority or target-family changes'},'generator_seeds':GENERATORS,'model_seeds':MODELS,'common_readout_seed':READOUT,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'composition_ba_matrix_rows_generator_cols_model':matrix,'composition_ba_range':max(all_ba)-min(all_ba),'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'PUBLIC_SYNTHETIC_3X3_L0_BRIDGE_BETA'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
