#!/usr/bin/env python3
"""Matched beta=0 falsifier for the L1-anchored beta=0.1 full-L2 target term."""
from __future__ import annotations
import json,numpy as np,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,TRAIN,fit_stage0_time_transform,generate_factor_split,make_initialized_teacher
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_l1_plus_l2_beta
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _fresh_predictor,_prefix_inputs,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
from clinical_jepa.targets.next_event_contract import construct_latent_targets,latent_objective
G=1102;MODELS=(2101,2102,2103);BETAS=(0.0,0.1)

def fixed_encoder_l2_cosine(condition,split,transform,seed):
 _,fixed_teacher=make_initialized_teacher(seed);fixed_teacher.eval();predictor=_fresh_predictor(seed).eval();set_identity_gelu_predictor(predictor);ids,times=_prefix_inputs(split,transform,'L2_SEP');target_ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False))
 with torch.no_grad():
  _,_,pooled=condition.encoder(ids,times,causal=False);prediction=predictor(pooled,'L2_SEP',layers=[1,2,3,4]);blocks,_,_=fixed_teacher(target_ids,target_times,causal=True);target,valid,selected=construct_latent_targets(blocks,torch.ones_like(target_ids,dtype=torch.bool),'L2_SEP');_,parts=latent_objective(prediction,target,valid)
 if selected!=[1,2,3,4] or prediction.shape!=(2048,4,4,16):raise RuntimeError('fixed L2 evaluation identity mismatch')
 return float(parts['cosine'])

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};losses={str(beta):[] for beta in BETAS}
 for seed in MODELS:
  results[str(seed)]={}
  for beta in BETAS:
   c=train_l1_plus_l2_beta(train,transform,seed,l2_beta=beta,total_steps=2000,scale_l2_constraint_with_beta=False);freeze_encoder(c);fixed_loss=fixed_encoder_l2_cosine(c,cal,transform,seed);rows,position=trained_collapse_diagnostics(c,cal,transform,reference);losses[str(beta)].append(fixed_loss);results[str(seed)][str(beta)]={'fixed_identity_predictor_fixed_initialized_teacher_l2_cosine':fixed_loss,'l2_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'l2_constraint_transfer_rows':rows,'position_specificity_descriptive':position,'training_loss_summary':c.training['losses']}
 mean0=sum(losses['0.0'])/3;mean1=sum(losses['0.1'])/3;relative=(mean0-mean1)/mean0;checks={'beta01_lower_fixed_encoder_l2_cosine_all_three':all(a>b for a,b in zip(losses['0.0'],losses['0.1'])),'mean_relative_fixed_encoder_l2_cosine_reduction_at_least_0_05':relative>=.05}
 print(json.dumps({'schema':'BP011-J04C-L2-BETA0-CONTROL-V1','contract':{'axis':'matched beta=0 versus beta=0.1 with identical unscaled L2 directional constraint','primary':'fixed identity predictor plus fixed initialized teacher isolates encoder effect on held-out CAL-OOD full-L2 cosine','pass':'beta=0.1 lower cosine in all 3 seeds and mean relative reduction >=5%','interpretation':'online loss decline alone is descriptive; constraint passage is metric-aligned'},'generator_seed':G,'model_seeds':MODELS,'betas':BETAS,'fixed_encoder_l2_cosine_by_beta':losses,'mean_beta0':mean0,'mean_beta01':mean1,'mean_relative_reduction':relative,'checks':checks,'target_specific_contrast_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_L2_TARGET_ENCODER_EFFECT'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
