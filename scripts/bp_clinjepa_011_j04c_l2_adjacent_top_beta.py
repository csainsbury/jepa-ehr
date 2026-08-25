#!/usr/bin/env python3
"""Simplify the red maximally separated L2 beta to adjacent top teacher layers 3 and 4."""
from __future__ import annotations
import json,numpy as np,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,_normalized_diagnostics,fit_stage0_time_transform,generate_factor_split,position_specificity
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _prefix_inputs,evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report
from clinical_jepa.targets.next_event_contract import construct_latent_targets
G=1102;MODELS=(2101,2102,2103);READOUT=2102;RECIPE='L2_SEP';LAYERS=(3,4);WEIGHT=10.;FLOOR=.01;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

def subset_diagnostics(c,split,transform,reference):
 ids,times=_prefix_inputs(split,transform,RECIPE);target_ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False))
 with torch.no_grad():
  _,_,pooled=c.encoder(ids,times,causal=False);prediction=c.predictor(pooled,RECIPE,layers=LAYERS);blocks,_,_=c.teacher(target_ids,target_times,causal=True);target,mask,selected=construct_latent_targets(blocks,torch.ones_like(target_ids,dtype=torch.bool),RECIPE);indices=[selected.index(x) for x in LAYERS];target=target[:,:,indices,:];mask=mask[:,:,indices]
 if prediction.shape!=(2048,4,2,16) or target.shape!=(2048,4,2,16) or mask.shape!=(2048,4,2):raise RuntimeError('two-layer L2 diagnostic shape mismatch')
 pred_flat=prediction.reshape(2048,8,16);target_flat=target.reshape(2048,8,16);full=reference['arms']['L2_SEP']['identities'];rows=[]
 for position in range(4):
  for subset_index,layer in enumerate(LAYERS):
   identity=position*2+subset_index;threshold=full[position*4+(layer-1)];metrics=_normalized_diagnostics(pred_flat[:,identity].cpu().numpy());teacher=_normalized_diagnostics(target_flat[:,identity].cpu().numpy());vp=metrics['normalized_variance']>threshold['normalized_variance']['threshold_midpoint'];rp=metrics['effective_rank']>threshold['effective_rank']['threshold_midpoint'];rows.append({'position_index':position,'layer':layer,'identity_index':identity,'full_threshold_identity_index':position*4+(layer-1),'normalized_variance':metrics['normalized_variance'],'variance_threshold':threshold['normalized_variance']['threshold_midpoint'],'variance_pass':bool(vp),'effective_rank':metrics['effective_rank'],'rank_threshold':threshold['effective_rank']['threshold_midpoint'],'rank_pass':bool(rp),'both_metrics_pass':bool(vp and rp),'teacher_target_effective_rank':teacher['effective_rank']})
 position=position_specificity(prediction,target,mask[:,:,0]);position={k:v for k,v in position.items() if k!='per_example'};return rows,position

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};values=[[],[],[]]
 for seed in MODELS:
  c=train_recipe_decoupled_seeds(train,transform,RECIPE,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR,per_identity_directional_hinge=True,l2_layers=LAYERS);freeze_encoder(c);probes,head,_=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=subset_diagnostics(c,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'constraint_transfer_rows':rows,'position_specificity_descriptive':position,'mean_last_100_train_batchwise_min_identity_variance':c.training['losses']['components']['v_direction_min']['last_100_mean']}
 means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'all_eight_identity_constraints_all_three_seeds':all(v['constraint_transfer_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'each_factor_seed_range_at_most_0_05':max(ranges)<=.05,'trained_minus_untrained_mean_at_least_0_05_each_factor':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-L2-ADJACENT-TOP-BETA-V1','contract':{'axis':'simplify red maximally separated two-layer L2 to the smallest adjacent top-layer separation [3,4]','weight_scaling':'weight 10 preserves L1 per-identity gradient strength for 8 versus 4 identities','pass':'all 8 selected identity constraints pass in 3/3; all BA >=0.80; factor ranges <=0.05; mean trained-minus-untrained >=0.05 per factor','no_tuning_after_open':True},'generator_seed':G,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':RECIPE,'selected_layers':LAYERS,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'trained_factor_ba_by_factor':values,'trained_factor_ba_means':means,'trained_minus_untrained_mean_ba':deltas,'factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_TWO_LAYER_L2_BETA'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
