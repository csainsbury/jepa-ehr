#!/usr/bin/env python3
"""Eval-only linear target-space redundancy diagnostic for L2 residuals."""
from __future__ import annotations
import json,numpy as np,torch
from torch.nn import functional as F
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from bp_clinjepa_011_j04c_factorized_residual_l2 import residual_stats,target_bundle
G=1102;MODELS=(2101,2102,2103);RIDGE=1e-3;COS_THRESHOLD=.15;R2_THRESHOLD=.80

def targets(teacher,split,transform):
 ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False));mean,valid,layer_target,layer_valid,residual,layers=target_bundle(teacher,ids,times)
 return mean,residual,layer_valid

def fit_and_evaluate(fit_mean,fit_residual,fit_valid,eval_mean,eval_residual,eval_valid):
 rows=[];all_cos=[];total_sse=0.;total_sst=0.;total_intercept_cos=[]
 for position in range(4):
  for layer in range(4):
   fm=fit_valid[:,position,layer];em=eval_valid[:,position,layer];x=fit_mean[fm,position].double();y=fit_residual[fm,position,layer].double();xe=eval_mean[em,position].double();ye=eval_residual[em,position,layer].double();xa=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],dim=1);xea=torch.cat([xe,torch.ones((xe.shape[0],1),dtype=torch.float64)],dim=1);reg=torch.eye(17,dtype=torch.float64)*RIDGE;reg[-1,-1]=0.;coef=torch.linalg.solve(xa.T@xa+reg,xa.T@y);pred=xea@coef;fit_mean_baseline=y.mean(dim=0,keepdim=True).expand_as(ye);cos=1.-F.cosine_similarity(pred,ye,dim=-1,eps=1e-8);intercept_cos=1.-F.cosine_similarity(fit_mean_baseline,ye,dim=-1,eps=1e-8);sse=float((pred-ye).square().sum());sst=float((ye-fit_mean_baseline).square().sum());r2=1.-sse/sst;identity_pass=float(cos.mean())<=COS_THRESHOLD and r2>=R2_THRESHOLD;rows.append({'identity_index':position*4+layer,'position_index':position,'layer_index':layer,'cosine_loss':float(cos.mean()),'intercept_only_cosine_loss':float(intercept_cos.mean()),'r2':r2,'cosine_margin_to_maximum':COS_THRESHOLD-float(cos.mean()),'r2_margin_to_minimum':r2-R2_THRESHOLD,'sse':sse,'sst':sst,'fit_rows':int(fm.sum()),'eval_rows':int(em.sum()),'operational_thresholds_pass':identity_pass});all_cos.extend(cos.tolist());total_intercept_cos.extend(intercept_cos.tolist());total_sse+=sse;total_sst+=sst
 aggregate={'cosine_loss':float(np.mean(all_cos)),'intercept_only_cosine_loss':float(np.mean(total_intercept_cos)),'pooled_r2':1.-total_sse/total_sst,'sse':total_sse,'sst':total_sst,'all_identities_pass':all(row['operational_thresholds_pass'] for row in rows),'minimum_cosine_margin':min(row['cosine_margin_to_maximum'] for row in rows),'minimum_r2_margin':min(row['r2_margin_to_minimum'] for row in rows)}
 return rows,aggregate

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);results={}
 for seed in MODELS:
  base=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);teacher=base.teacher.eval();pm,pr,pv=targets(teacher,probe,transform);cm,cr,cv=targets(teacher,cal,transform);rows,aggregate=fit_and_evaluate(pm,pr,pv,cm,cr,cv);results[str(seed)]={'identity_rows':rows,'aggregate':aggregate,'probe_fit_target_stats':residual_stats(pr,pv),'cal_ood_target_stats':residual_stats(cr,cv)}
 gate=all(value['aggregate']['all_identities_pass'] for value in results.values());print(json.dumps({'schema':'BP011-J04C-CONDITIONAL-TARGET-NOVELTY-V1','generator_seed':G,'model_seeds':MODELS,'assay':{'input':'oracle frozen-teacher L1 mean target','output':'raw zero-sum layer residual target','fit':'PROBE_FIT per-identity float64 ridge with unpenalized intercept','eval':'CAL_OOD','ridge_lambda':RIDGE,'operational_cosine_loss_maximum':COS_THRESHOLD,'operational_r2_minimum':R2_THRESHOLD,'family_gate':'both thresholds pass for every identity and seed'},'results':results,'linear_target_space_redundancy_gate_pass':gate,'interpretation':('STOP_CURRENT_L2_REPRESENTATION_TARGET_FAMILY_NO_EXTRA_SLOTS' if gate else 'TARGET_NOVELTY_UNRESOLVED_INSPECT_MATERIAL_FAILING_IDENTITIES'),'claim_ceiling':'FIXED_GENERATOR_ORACLE_L1_MEAN_TO_LAYER_RESIDUAL_LINEAR_TARGET_SPACE_ASSAY'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
