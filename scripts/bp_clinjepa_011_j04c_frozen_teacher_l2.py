#!/usr/bin/env python3
"""Matched frozen-post-L1-teacher test of a beta=0.03 full-L2 target term."""
from __future__ import annotations
import copy,hashlib,json,math,numpy as np,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_fresh_predictor,_loss_summary,_prefix_inputs,_tensor_rows,evaluate_readouts,fit_condition_readouts,freeze_encoder,optimizer_membership,pretraining_indices,threshold_reference_report,trained_collapse_diagnostics
from clinical_jepa.targets.next_event_contract import construct_latent_targets,latent_objective
G=1102;MODELS=(2101,2102,2103);READOUT=2102;BETA=.03;STEPS=2000;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

def digest_module(module):
 h=hashlib.sha256()
 for name,tensor in sorted(module.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def assert_cloned(left,right):
 ls=left.state_dict();rs=right.state_dict()
 if ls.keys()!=rs.keys():raise RuntimeError('clone state names differ')
 for name in ls:
  if not torch.equal(ls[name],rs[name]) or ls[name].data_ptr()==rs[name].data_ptr():raise RuntimeError(f'nonidentical or shared clone state storage: {name}')

def identity_penalty(prediction,weight):
 variance=torch.nn.functional.normalize(prediction,dim=-1,eps=1e-8).var(dim=0,correction=0).mean(dim=-1)
 return weight*torch.clamp(.01-variance,min=0.).mean(),variance.min()

def frozen_targets(teacher,target_ids,target_times):
 with torch.no_grad():blocks,_,_=teacher(target_ids,target_times,causal=True);mask=torch.ones_like(target_ids,dtype=torch.bool);t1,v1,_=construct_latent_targets(blocks,mask,'L1_AVG');t2,v2,layers=construct_latent_targets(blocks,mask,'L2_SEP')
 if layers!=[1,2,3,4] or t2.shape[1:]!=(4,4,16):raise RuntimeError('frozen L2 target identity mismatch')
 return t1.detach(),v1,t2.detach(),v2,layers

def arm_forward(encoder,predictor,prefix_ids,prefix_times,t1,v1,t2,v2,layers,beta):
 _,_,pooled=encoder(prefix_ids,prefix_times,causal=False);p1=predictor(pooled,'L1_AVG');p2=predictor(pooled,'L2_SEP',layers=layers)
 if p2.shape[1:]!=(4,4,16):raise RuntimeError('continuation L2 prediction identity mismatch')
 l1,part1=latent_objective(p1,t1,v1);l2,part2=latent_objective(p2,t2,v2);pen1,min1=identity_penalty(p1,5.);pen2,min2=identity_penalty(p2,20.);total=l1+pen1+pen2+beta*l2
 return total,{'l1_cosine':part1['cosine'],'l2_cosine':part2['cosine'],'l1_penalty':pen1,'l2_penalty':pen2,'l1_v_min':min1,'l2_v_min':min2}

def continue_pair(base,teacher,train,transform,seed):
 ce=copy.deepcopy(base.encoder).train();cp=copy.deepcopy(base.predictor).train();xe=copy.deepcopy(base.encoder).train();xp=copy.deepcopy(base.predictor).train();assert_cloned(ce,xe);assert_cloned(cp,xp)
 co=_adamw(list(ce.parameters())+list(cp.parameters()),lr=3e-4,weight_decay=1e-4);xo=_adamw(list(xe.parameters())+list(xp.parameters()),lr=3e-4,weight_decay=1e-4)
 control_config=[{k:v for k,v in group.items() if k!='params'} for group in co.param_groups];candidate_config=[{k:v for k,v in group.items() if k!='params'} for group in xo.param_groups]
 if co is xo or co.state or xo.state or control_config!=candidate_config or not optimizer_membership((ce,cp),co) or not optimizer_membership((xe,xp),xo):raise RuntimeError('paired optimizer invariant failed')
 teacher_ids={id(p) for p in teacher.parameters()};opt_ids={id(p) for opt in (co,xo) for group in opt.param_groups for p in group['params']}
 if teacher_ids&opt_ids or any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('frozen teacher entered gradient path or optimizer')
 control_schedule=[x.copy() for x in pretraining_indices(seed+10000)];candidate_schedule=[x.copy() for x in pretraining_indices(seed+10000)]
 if len(control_schedule)!=STEPS or len(candidate_schedule)!=STEPS or any(not np.array_equal(a,b) for a,b in zip(control_schedule,candidate_schedule)):raise RuntimeError('paired continuation schedule mismatch')
 target_times_all=transform.transform(train.target_intervals).astype('float32',copy=False);totals={'control':[],'candidate':[]};components={arm:{k:[] for k in ('l1_cosine','l2_cosine','l1_penalty','l2_penalty','l1_v_min','l2_v_min')} for arm in totals};pre_difference=None
 for step,(control_indices,candidate_indices) in enumerate(zip(control_schedule,candidate_schedule)):
  if not np.array_equal(control_indices,candidate_indices):raise RuntimeError('paired schedule diverged during iteration')
  indices=control_indices
  prefix_ids,prefix_times=_prefix_inputs(train,transform,'L2_SEP',indices);target_ids=_tensor_rows(train.target_type_ids,indices,dtype=torch.long);target_times=_tensor_rows(target_times_all,indices,dtype=torch.float32);t1,v1,t2,v2,layers=frozen_targets(teacher,target_ids,target_times)
  co.zero_grad(set_to_none=True);xo.zero_grad(set_to_none=True);closs,cparts=arm_forward(ce,cp,prefix_ids,prefix_times,t1,v1,t2,v2,layers,0.);xloss,xparts=arm_forward(xe,xp,prefix_ids,prefix_times,t1,v1,t2,v2,layers,BETA)
  if step==0:
   expected=BETA*latent_objective(xp(xe(prefix_ids,prefix_times,causal=False)[2],'L2_SEP',layers=layers),t2,v2)[0]
   actual=xloss-closs
   if not torch.allclose(actual,expected,rtol=1e-6,atol=1e-7):raise RuntimeError('pre-update paired loss difference mismatch')
   pre_difference={'actual':float(actual.detach()),'expected':float(expected.detach())}
  for loss,opt,encoder,predictor,parts,arm in ((closs,co,ce,cp,cparts,'control'),(xloss,xo,xe,xp,xparts,'candidate')):
   _finite_loss(loss);loss.backward()
   if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in list(encoder.parameters())+list(predictor.parameters())):raise FloatingPointError('continuation gradient invariant failed')
   opt.step();totals[arm].append(float(loss.detach()))
   for name,value in parts.items():components[arm][name].append(float(value.detach()))
 if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('frozen teacher accumulated gradients')
 audit={'pre_update_loss_difference':pre_difference,'continuation_schedule_seed':seed+10000,'optimizer_updates_per_arm':STEPS,'phase_two_ema_updates_per_arm':0,'control_training':_loss_summary(totals['control'],components['control']),'candidate_training':_loss_summary(totals['candidate'],components['candidate'])}
 return TrainedCondition('L2_SEP',ce,teacher,cp,audit['control_training']),TrainedCondition('L2_SEP',xe,teacher,xp,audit['candidate_training']),audit

def fixed_encoder_cosine(encoder,teacher,predictor,split,transform):
 ids,times=_prefix_inputs(split,transform,'L2_SEP');target_ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False));t1,v1,t2,v2,layers=frozen_targets(teacher,target_ids,target_times)
 with torch.no_grad():
  _,_,pooled=encoder(ids,times,causal=False);prediction=predictor(pooled,'L2_SEP',layers=layers);_,parts=latent_objective(prediction,t2,v2)
 return float(parts['cosine'])

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};control_losses=[];candidate_losses=[];values=[[],[],[]];teacher_digests=[]
 for seed in MODELS:
  base=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);teacher=copy.deepcopy(base.teacher).eval()
  for p in teacher.parameters():p.requires_grad_(False)
  before=digest_module(teacher);teacher_digests.append(before);control,candidate,audit=continue_pair(base,teacher,train,transform,seed);after=digest_module(teacher)
  if before!=after:raise RuntimeError('frozen teacher changed during continuation')
  shared_predictor=_fresh_predictor(seed).eval();set_identity_gelu_predictor(shared_predictor);base_loss=fixed_encoder_cosine(base.encoder,teacher,shared_predictor,cal,transform);control_loss=fixed_encoder_cosine(control.encoder,teacher,shared_predictor,cal,transform);candidate_loss=fixed_encoder_cosine(candidate.encoder,teacher,shared_predictor,cal,transform);control_losses.append(control_loss);candidate_losses.append(candidate_loss)
  freeze_encoder(control);freeze_encoder(candidate);probes,head,_=fit_condition_readouts(candidate,probe,transform,READOUT);ev,_=evaluate_readouts(candidate,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(candidate,cal,transform,reference);control_rows,control_position=trained_collapse_diagnostics(control,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'teacher_digest_before_after':[before,after],'pre_continuation_fixed_encoder_l2_cosine':base_loss,'control_fixed_encoder_l2_cosine':control_loss,'candidate_fixed_encoder_l2_cosine':candidate_loss,'candidate_factor_ba':ba,'candidate_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'candidate_constraint_rows':rows,'candidate_position_specificity_descriptive':position,'control_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in control_rows),'control_position_specificity_descriptive':control_position,'continuation_audit':audit}
 mean_control=sum(control_losses)/3;mean_candidate=sum(candidate_losses)/3;relative=(mean_control-mean_candidate)/mean_control;means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'candidate_lower_fixed_teacher_l2_cosine_all_three':all(a>b for a,b in zip(control_losses,candidate_losses)),'mean_relative_reduction_at_least_0_05':relative>=.05,'candidate_all_sixteen_constraints_all_three':all(v['candidate_constraint_transfer_all_pass'] for v in results.values()),'candidate_all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'candidate_each_factor_range_at_most_0_05':max(ranges)<=.05,'candidate_mean_gain_vs_original_untrained_at_least_0_05_each':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-FROZEN-POST-L1-TEACHER-L2-V1','contract':{'paired_seed_specific_teacher':'one post-L1 frozen EMA snapshot shared only by the two arms within each seed','only_arm_difference':'candidate adds beta 0.03 times full frozen-teacher L2 target loss','continuation_updates':STEPS,'claim':'relative L2-target encoder effect under frozen post-L1 teacher; not evidence that teacher movement caused earlier failures or that freezing beats matched moving EMA'},'generator_seed':G,'model_seeds':MODELS,'l2_beta':BETA,'teacher_digests':teacher_digests,'control_fixed_encoder_l2_cosine':control_losses,'candidate_fixed_encoder_l2_cosine':candidate_losses,'mean_relative_reduction':relative,'candidate_factor_ba_by_factor':values,'candidate_factor_ba_means':means,'candidate_minus_original_untrained_mean_ba':deltas,'candidate_factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_FROZEN_POST_L1_TEACHER_RELATIVE_L2_TARGET_EFFECT'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
