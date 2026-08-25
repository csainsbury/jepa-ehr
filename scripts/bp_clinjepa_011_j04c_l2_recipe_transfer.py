#!/usr/bin/env python3
"""Reviewed fixed-generator transfer of the frozen L1 mechanism to complete L2_SEP."""
from __future__ import annotations
import json,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
G=1102;MODELS=(2101,2102,2103);READOUT=2102;RECIPE='L2_SEP';WEIGHT=5.;FLOOR=.01;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};values=[[],[],[]]
 for seed in MODELS:
  c=train_recipe_decoupled_seeds(train,transform,RECIPE,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR,per_identity_directional_hinge=True);freeze_encoder(c);probes,head,_=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
  if len(rows)!=16:raise RuntimeError('L2 requires exactly 16 ordered identity constraints')
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'constraint_transfer_rows':rows,'position_specificity_descriptive':position,'mean_last_100_train_batchwise_min_identity_variance':c.training['losses']['components']['v_direction_min']['last_100_mean'],'training_loss_summary':c.training['losses']}
 means=[sum(v)/len(v) for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'all_sixteen_identity_constraints_all_three_seeds':all(v['constraint_transfer_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'each_factor_seed_range_at_most_0_05':max(ranges)<=.05,'trained_minus_untrained_mean_at_least_0_05_each_factor':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-L2-RECIPE-TRANSFER-V1','contract':{'axis':'transfer the fixed complete L1 mechanism to complete L2_SEP at generator 1102','identity_order':'row-major position x selected layer [1,2,3,4], 16 identities','constraint_transfer':'TRAIN-side per-identity floor holding OOD variance above unchanged thresholds; not independent anti-collapse evidence','pass':'all 16 constraints pass in 3/3 seeds; all factor BA >=0.80; each factor range <=0.05; trained-minus-untrained mean BA >=0.05 per factor','position_specificity':'descriptive >0 expectation only','no_tuning_after_open':True},'generator_seed':G,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':RECIPE,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'per_identity_directional_hinge':True,'untrained_factor_ba_means_exact':UNTRAINED,'trained_factor_ba_by_factor':values,'trained_factor_ba_means':means,'trained_minus_untrained_mean_ba':deltas,'factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_L2_RECIPE_BETA'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
