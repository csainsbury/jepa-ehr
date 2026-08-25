#!/usr/bin/env python3
"""Transfer the frozen passing L0 bridge mechanism to the complete L1_AVG recipe contract."""
from __future__ import annotations
import json, torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
GENERATOR=1102; MODELS=(2101,2102,2103); READOUT=2102; RECIPE='L1_AVG'; WEIGHT=5.0; FLOOR=.005

def main():
 torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
 train=independent_train_nuisance(generate_factor_split(GENERATOR,TRAIN,8192),GENERATOR); probe=generate_factor_split(GENERATOR,PROBE_FIT,2048); cal=generate_factor_split(GENERATOR,CAL_OOD,2048); transform=fit_stage0_time_transform(train); reference,digest=threshold_reference_report(); results={}; factor_values=[[],[],[]]
 for seed in MODELS:
  c=train_recipe_decoupled_seeds(train,transform,RECIPE,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR); freeze_encoder(c); probes,head,_=fit_condition_readouts(c,probe,transform,READOUT); ev,_=evaluate_readouts(c,probes,head,cal,transform); ba=[f['balanced_accuracy'] for f in ev['factors']]; rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
  for factor,value in enumerate(ba): factor_values[factor].append(value)
  results[str(seed)]={'factor_ba':ba,'collapse_all_pass':all(r['both_metrics_pass'] for r in rows),'collapse_rows':rows,'position_specificity_descriptive':position,'training_loss_summary':c.training['losses']}
 ranges=[max(values)-min(values) for values in factor_values]
 checks={'all_four_identities_pass_all_three_seeds':all(v['collapse_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(value for values in factor_values for value in values)>=.80,'each_factor_seed_range_at_most_0_05':max(ranges)<=.05}
 print(json.dumps({'schema':'BP011-J04C-L1-RECIPE-TRANSFER-V1','contract':{'axis':'transfer the frozen L0 bridge mechanism to the complete L1_AVG recipe contract, including four target identities and position-coded predictor behavior','directional_penalty_semantics':'unchanged aggregate scalar hinge over the four identities; per-identity collapse gates remain unchanged','position_specificity':'measured with accepted statistic, expected >0 descriptively, not gated; a pass supports L1 recipe viability but not position-resolved restoration','pass':'all four identity collapse gates pass in 3/3 seeds; all factor BA >=0.80; each factor seed range <=0.05','not_tested':'position-resolved restoration, direct superiority, L2, or governed data'},'generator_seed':GENERATOR,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':RECIPE,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'factor_ba_by_factor':factor_values,'factor_ba_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'PUBLIC_SYNTHETIC_L1_RECIPE_VIABILITY'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
