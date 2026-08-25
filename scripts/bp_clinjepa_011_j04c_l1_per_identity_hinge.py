#!/usr/bin/env python3
"""Split the red L1 aggregate directional hinge into one unchanged hinge per target identity."""
from __future__ import annotations
import json, torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
G=1102; MODELS=(2101,2102,2103); READOUT=2102; RECIPE='L1_AVG'; WEIGHT=5.; FLOOR=.005

def main():
 torch.use_deterministic_algorithms(True); torch.set_num_threads(1); train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G); probe=generate_factor_split(G,PROBE_FIT,2048); cal=generate_factor_split(G,CAL_OOD,2048); transform=fit_stage0_time_transform(train); reference,digest=threshold_reference_report(); results={}; values=[[],[],[]]
 for seed in MODELS:
  c=train_recipe_decoupled_seeds(train,transform,RECIPE,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR,per_identity_directional_hinge=True); freeze_encoder(c); probes,head,_=fit_condition_readouts(c,probe,transform,READOUT); ev,_=evaluate_readouts(c,probes,head,cal,transform); ba=[f['balanced_accuracy'] for f in ev['factors']]; rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
  for i,v in enumerate(ba): values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'collapse_all_pass':all(r['both_metrics_pass'] for r in rows),'collapse_rows':rows,'position_specificity_descriptive':position,'training_loss_summary':c.training['losses']}
 ranges=[max(v)-min(v) for v in values]; checks={'all_four_identities_pass_all_three_seeds':all(v['collapse_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.80,'each_factor_seed_range_at_most_0_05':max(ranges)<=.05}
 print(json.dumps({'schema':'BP011-J04C-L1-PER-IDENTITY-HINGE-V1','contract':{'axis':'replace aggregate L1 directional hinge with one weight-5/floor-0.005 hinge per target identity after aggregate transfer failed one identity','pass':'all four identity collapse gates pass in 3/3 seeds; all factor BA >=0.80; each factor seed range <=0.05','position_specificity':'descriptive >0 expectation only; pass supports L1 recipe viability, not position-resolved restoration','not_tested':'direct superiority, L2, or governed data'},'generator_seed':G,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':RECIPE,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'per_identity_directional_hinge':True,'factor_ba_by_factor':values,'factor_ba_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'PUBLIC_SYNTHETIC_L1_RECIPE_VIABILITY'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
