#!/usr/bin/env python3
"""Three-arm mechanical falsifier for exact L2 target redundancy."""
from __future__ import annotations
import json,numpy as np,torch
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import EMATeacher
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _fresh_encoder
from bp_clinjepa_011_j04c_factorized_residual_l2 import residual_stats,target_bundle
G=1102;MODELS=(2101,2102,2103);RIDGE=1e-3;EPS=1e-6

def targets(teacher,split,transform):
 ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False));mean,valid,layer_target,layer_valid,residual,layers=target_bundle(teacher,ids,times);return mean,residual,layer_valid

def epsilon_rank(matrix):
 centered=matrix.double()-matrix.double().mean(dim=0,keepdim=True);singular=torch.linalg.svdvals(centered);top=float(singular[0]) if singular.numel() else 0.;rank=int((singular>EPS*top).sum()) if top>0 else 0;return {'epsilon_relative':EPS,'rank':rank,'dimension':int(matrix.shape[1]),'top_singular_value':top,'singular_values':[float(x) for x in singular]}

def assay(fit_mean,fit_residual,fit_valid,eval_mean,eval_residual,eval_valid):
 rows=[];total_full=total_intercept=total_zero=0.;cos_full=[];cos_intercept=[]
 for position in range(4):
  for layer in range(4):
   fm=fit_valid[:,position,layer];em=eval_valid[:,position,layer];x=fit_mean[fm,position].double();y=fit_residual[fm,position,layer].double();xe=eval_mean[em,position].double();ye=eval_residual[em,position,layer].double();xa=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],dim=1);xea=torch.cat([xe,torch.ones((xe.shape[0],1),dtype=torch.float64)],dim=1);reg=torch.eye(17,dtype=torch.float64)*RIDGE;reg[-1,-1]=0.;coef=torch.linalg.solve(xa.T@xa+reg,xa.T@y);a=coef[:-1];b=coef[-1];pred=xea@coef;intercept=y.mean(dim=0,keepdim=True).expand_as(ye);zero=torch.zeros_like(ye);sse_full=float((pred-ye).square().sum());sse_intercept=float((intercept-ye).square().sum());sse_zero=float((zero-ye).square().sum());linear_eval=xe@a;linear_centered=linear_eval-linear_eval.mean(dim=0,keepdim=True);full_cos=1.-F.cosine_similarity(pred,ye,dim=-1,eps=1e-8);intercept_cos=1.-F.cosine_similarity(intercept,ye,dim=-1,eps=1e-8);joint=torch.cat([x,y],dim=1);rows.append({'identity_index':position*4+layer,'position_index':position,'layer_index':layer,'full_ridge_sse':sse_full,'intercept_only_sse':sse_intercept,'zero_sse':sse_zero,'full_vs_intercept_sse_ratio':sse_full/sse_intercept if sse_intercept>0 else None,'intercept_explained_energy_vs_zero':1.-sse_intercept/sse_zero,'full_explained_energy_vs_zero':1.-sse_full/sse_zero,'full_cosine_loss':float(full_cos.mean()),'intercept_only_cosine_loss':float(intercept_cos.mean()),'a_frobenius_norm':float(torch.linalg.vector_norm(a)),'intercept_norm':float(torch.linalg.vector_norm(b)),'linear_contribution_rms':float(torch.sqrt((linear_eval.square().sum(dim=1)).mean())),'centered_linear_contribution_rms':float(torch.sqrt((linear_centered.square().sum(dim=1)).mean())),'joint_m_residual_epsilon_rank':epsilon_rank(joint)});total_full+=sse_full;total_intercept+=sse_intercept;total_zero+=sse_zero;cos_full.extend(full_cos.tolist());cos_intercept.extend(intercept_cos.tolist())
 return {'identity_rows':rows,'aggregate':{'full_ridge_sse':total_full,'intercept_only_sse':total_intercept,'zero_sse':total_zero,'full_vs_intercept_sse_ratio':total_full/total_intercept,'intercept_explained_energy_vs_zero':1.-total_intercept/total_zero,'full_explained_energy_vs_zero':1.-total_full/total_zero,'full_cosine_loss':float(np.mean(cos_full)),'intercept_only_cosine_loss':float(np.mean(cos_intercept))},'mean_epsilon_rank_by_position':[epsilon_rank(fit_mean[:,position]) for position in range(4)]}

def unique_state_report(split,mean):
 joint=np.concatenate([split.target_type_ids,split.target_intervals],axis=1);return {'unique_S_states':int(np.unique(split.S,axis=0).shape[0]),'unique_target_configurations':int(np.unique(joint,axis=0).shape[0]),'unique_exact_mean_rows_by_position':[int(np.unique(mean[:,p].detach().cpu().numpy(),axis=0).shape[0]) for p in range(4)]}

def teacher_assay(teacher,probe,cal,transform):
 pm,pr,pv=targets(teacher,probe,transform);cm,cr,cv=targets(teacher,cal,transform);return {'assay':assay(pm,pr,pv,cm,cr,cv),'probe_fit_states':unique_state_report(probe,pm),'cal_ood_states':unique_state_report(cal,cm),'probe_fit_residual_stats':residual_stats(pr,pv),'cal_ood_residual_stats':residual_stats(cr,cv)}

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);results={}
 for seed in MODELS:
  random_teacher=EMATeacher(_fresh_encoder(seed),momentum=.996).eval();base=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);trained=teacher_assay(base.teacher.eval(),probe,cal,transform);random=teacher_assay(random_teacher,probe,cal,transform);results[str(seed)]={'trained_post_l1_teacher':trained,'paired_random_initial_teacher':random}
 print(json.dumps({'schema':'BP011-J04C-L2-REDUNDANCY-FALSIFIER-V1','generator_seed':G,'model_seeds':MODELS,'contract':{'arm_a':'full ridge versus intercept-only contribution','arm_b':'epsilon-rank 1e-6 and distinct-state audit','arm_c':'paired random initial teacher same architecture/init seed','ridge_lambda':RIDGE,'training':'none beyond reconstructing established post-L1 teachers; no result-contingent tuning'},'results':results,'claim_ceiling':'MECHANISTIC_CHARACTERIZATION_OF_L2_REDUNDANCY_ON_FIXED_SYNTHETIC_SUPPORT'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
