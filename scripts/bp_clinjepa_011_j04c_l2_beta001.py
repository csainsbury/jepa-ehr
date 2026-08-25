#!/usr/bin/env python3
"""Test gradient-calibrated beta=0.01 against the frozen matched beta=0 control."""
from __future__ import annotations
import json,numpy as np,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split,make_initialized_teacher
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_l1_plus_l2_beta
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _fresh_predictor,_prefix_inputs,evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
from clinical_jepa.targets.next_event_contract import construct_latent_targets,latent_objective
G=1102;MODELS=(2101,2102,2103);READOUT=2102;BETA=.01;BETA0_FIXED=(0.563904881477356,0.5450581312179565,0.4805530607700348);UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

def fixed_encoder_l2_cosine(condition,split,transform,seed):
 _,teacher=make_initialized_teacher(seed);teacher.eval();predictor=_fresh_predictor(seed).eval();set_identity_gelu_predictor(predictor);ids,times=_prefix_inputs(split,transform,'L2_SEP');target_ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False))
 with torch.no_grad():
  _,_,pooled=condition.encoder(ids,times,causal=False);prediction=predictor(pooled,'L2_SEP',layers=[1,2,3,4]);blocks,_,_=teacher(target_ids,target_times,causal=True);target,valid,selected=construct_latent_targets(blocks,torch.ones_like(target_ids,dtype=torch.bool),'L2_SEP');_,parts=latent_objective(prediction,target,valid)
 if selected!=[1,2,3,4]:raise RuntimeError('fixed L2 identity mismatch')
 return float(parts['cosine'])

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();fixed=[];values=[[],[],[]];results={}
 for seed in MODELS:
  c=train_l1_plus_l2_beta(train,transform,seed,l2_beta=BETA,total_steps=2000,scale_l2_constraint_with_beta=False);freeze_encoder(c);fixed_loss=fixed_encoder_l2_cosine(c,cal,transform,seed);fixed.append(fixed_loss);probes,head,_=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'fixed_encoder_l2_cosine':fixed_loss,'l2_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'l2_constraint_transfer_rows':rows,'position_specificity_descriptive':position,'training_loss_summary':c.training['losses']}
 mean0=sum(BETA0_FIXED)/3;mean1=sum(fixed)/3;relative=(mean0-mean1)/mean0;means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'fixed_encoder_l2_cosine_lower_than_beta0_all_three':all(a>b for a,b in zip(BETA0_FIXED,fixed)),'mean_relative_fixed_encoder_l2_cosine_reduction_at_least_0_05':relative>=.05,'all_sixteen_constraints_all_three':all(v['l2_constraint_transfer_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'each_factor_range_at_most_0_05':max(ranges)<=.05,'trained_minus_untrained_mean_at_least_0_05_each':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-L2-BETA001-V1','contract':{'axis':'reduce L2 target beta from 0.1 to gradient-calibrated 0.01 after raw L2 target gradients measured 6.1-9.6x L1','effective_initial_scale':'approximately 6-10% of L1 gradient norm at the last-green L1 diagnostic state','matched_beta0':'frozen exact fixed-encoder losses from j04c-l2-beta0-stdout.json','target_specific_pass':'lower than beta0 all three seeds and >=5% mean relative reduction','ordinary_gates':'all 48 constraints, BA >=0.80, ranges <=0.05, trained-untrained means >=0.05'},'generator_seed':G,'model_seeds':MODELS,'l2_beta':BETA,'beta0_fixed_encoder_l2_cosine':BETA0_FIXED,'beta001_fixed_encoder_l2_cosine':fixed,'mean_relative_fixed_encoder_l2_cosine_reduction':relative,'trained_factor_ba_by_factor':values,'trained_factor_ba_means':means,'trained_minus_untrained_mean_ba':deltas,'factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_GRADIENT_CALIBRATED_L2_TARGET_BETA'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
