#!/usr/bin/env python3
"""Matched fixed target-aligned residual decoder mechanism test."""
from __future__ import annotations
import copy,json,math,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import sinusoidal_code
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_loss_summary,_prefix_inputs,_tensor_rows,evaluate_readouts,fit_condition_readouts,freeze_encoder,pretraining_indices,threshold_reference_report,trained_collapse_diagnostics
from clinical_jepa.targets.next_event_contract import latent_objective
from bp_clinjepa_011_j04c_factorized_residual_l2 import assert_cloned,digest_module,identity_penalty,residual_mse,residual_stats,ridge_residual_probe,split_features_targets,target_bundle
G=1102;MODELS=(2101,2102,2103);READOUT=2102;STEPS=2000;DECODER_RIDGE=1e-3;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262);FIT_COUNTS={}

class FixedResidualDecoder(nn.Module):
 def __init__(self,weights):
  super().__init__();self.heads=nn.ModuleList([nn.Linear(16,16,bias=False) for _ in range(4)])
  with torch.no_grad():
   for head,weight in zip(self.heads,weights):head.weight.copy_(weight)
  for parameter in self.parameters():parameter.requires_grad_(False)
 def forward(self,pooled):
  positions=sinusoidal_code(torch.arange(4,device=pooled.device),dtype=pooled.dtype);features=F.layer_norm(pooled[:,None,:]+positions[None,:,:],(16,),eps=1e-5);raw=torch.stack([head(features) for head in self.heads],dim=2);residual=raw-raw.mean(dim=2,keepdim=True)
  if residual.shape!=(pooled.shape[0],4,4,16) or not torch.allclose(residual.sum(dim=2),torch.zeros_like(residual[:,:,0]),atol=1e-6,rtol=0):raise RuntimeError('fixed decoder output geometry mismatch')
  return residual

class FixedFactorizedPredictor(nn.Module):
 def __init__(self,base,decoder):super().__init__();self.base=copy.deepcopy(base);self.decoder=decoder
 def factorized(self,pooled):
  shared=self.base(pooled,'L1_AVG');residual=self.decoder(pooled);absolute=shared.unsqueeze(2)+residual
  if not torch.equal(absolute,shared.unsqueeze(2)+residual):raise RuntimeError('fixed absolute L2 composition mismatch')
  return absolute,residual,shared
 def forward(self,pooled,recipe,*,k=4,layers=(1,2,3,4)):
  if recipe!='L2_SEP':return self.base(pooled,recipe,k=k,layers=layers)
  if k!=4 or list(layers)!=[1,2,3,4]:raise ValueError('fixed decoder requires ordered full L2')
  return self.factorized(pooled)[0]

def fit_fixed_decoder(base,teacher,train,transform,seed):
 FIT_COUNTS[seed]=FIT_COUNTS.get(seed,0)+1
 if FIT_COUNTS[seed]!=1 or len(train.target_type_ids)!=8192:raise RuntimeError('decoder must be fit exactly once on full TRAIN')
 ids,times=_prefix_inputs(train,transform,'L2_SEP');target_ids=torch.as_tensor(train.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(train.target_intervals).astype(np.float32,copy=False));mean,valid,lt,lv,residual,layers=target_bundle(teacher,target_ids,target_times)
 with torch.no_grad():_,_,pooled=base.encoder(ids,times,causal=False);positions=sinusoidal_code(torch.arange(4),dtype=pooled.dtype);features=F.layer_norm(pooled[:,None,:]+positions[None,:,:],(16,),eps=1e-5).reshape(-1,16).double()
 weights=[]
 for layer in range(4):
  target=residual[:,:,layer,:].reshape(-1,16).double();reg=torch.eye(16,dtype=torch.float64)*DECODER_RIDGE;coef=torch.linalg.solve(features.T@features+reg,features.T@target);weights.append(coef.T.float())
 decoder=FixedResidualDecoder(weights).eval();maps=[head.weight.detach() for head in decoder.heads]
 if any(not bool(torch.isfinite(w).all()) or not bool(torch.count_nonzero(w)) for w in maps) or all(torch.equal(maps[0],w) for w in maps[1:]):raise RuntimeError('fixed decoder map nonzero/diversity invariant failed')
 with torch.no_grad():pred=decoder(pooled);zero=torch.zeros_like(pred);aggregate_pred=residual_mse(pred,residual,lv);aggregate_zero=residual_mse(zero,residual,lv);identity=[]
 for position in range(4):
  for layer in range(4):
   mask=lv[:,position,layer];pm=(pred[mask,position,layer]-residual[mask,position,layer]).square().mean();zm=residual[mask,position,layer].square().mean();identity.append({'identity_index':position*4+layer,'position_index':position,'layer_index':layer,'decoder_mse':float(pm),'zero_decoder_mse':float(zm)})
 if not bool(torch.isfinite(aggregate_pred)) or not float(aggregate_pred)<float(aggregate_zero):raise RuntimeError('centered decoder does not beat zero decoder on TRAIN')
 return decoder,{'fit_calls':FIT_COUNTS[seed],'split':'TRAIN','rows':8192,'aggregate_decoder_mse':float(aggregate_pred),'aggregate_zero_decoder_mse':float(aggregate_zero),'identity_mse':identity}

def optimizer_ids(optimizer):return {id(p) for group in optimizer.param_groups for p in group['params']}

def arm_forward(encoder,predictor,prefix_ids,prefix_times,mean,valid,lt,lv,residual_target,beta):
 _,_,pooled=encoder(prefix_ids,prefix_times,causal=False);p1=predictor(pooled,'L1_AVG');absolute,residual,shared=predictor.factorized(pooled)
 if not torch.equal(p1,shared) or not torch.equal(absolute,shared.unsqueeze(2)+residual):raise RuntimeError('fixed factorized composition mismatch')
 l1,part1=latent_objective(p1,mean,valid);pen1,min1=identity_penalty(p1,5.);pen2,min2=identity_penalty(absolute,20.);r_loss=residual_mse(residual,residual_target,lv);total=l1+pen1+pen2+beta*r_loss
 return total,{'l1_cosine':part1['cosine'],'residual_mse':r_loss,'l1_penalty':pen1,'l2_penalty':pen2,'l1_v_min':min1,'l2_v_min':min2},absolute

def continue_pair(base,teacher,decoder,train,transform,seed):
 ce=copy.deepcopy(base.encoder).train();xe=copy.deepcopy(base.encoder).train();cp=FixedFactorizedPredictor(base.predictor,decoder).train();xp=FixedFactorizedPredictor(base.predictor,decoder).train();assert_cloned(ce,xe);assert_cloned(cp.base,xp.base)
 if cp.decoder is not xp.decoder or cp.decoder is not decoder:raise RuntimeError('paired arms do not share one fixed decoder')
 trainable_control=list(ce.parameters())+list(cp.base.parameters());trainable_candidate=list(xe.parameters())+list(xp.base.parameters());co=_adamw(trainable_control,lr=3e-4,weight_decay=1e-4);xo=_adamw(trainable_candidate,lr=3e-4,weight_decay=1e-4);cc=[{k:v for k,v in g.items() if k!='params'} for g in co.param_groups];xc=[{k:v for k,v in g.items() if k!='params'} for g in xo.param_groups]
 if co is xo or co.state or xo.state or cc!=xc or optimizer_ids(co)!={id(p) for p in trainable_control} or optimizer_ids(xo)!={id(p) for p in trainable_candidate}:raise RuntimeError('fixed decoder optimizer invariant failed')
 frozen_ids={id(p) for p in teacher.parameters()}|{id(p) for p in decoder.parameters()}
 if frozen_ids&(optimizer_ids(co)|optimizer_ids(xo)) or any(p.requires_grad for p in list(teacher.parameters())+list(decoder.parameters())):raise RuntimeError('fixed module gradient/optimizer leak')
 cs=[x.copy() for x in pretraining_indices(seed+10000)];xs=[x.copy() for x in pretraining_indices(seed+10000)]
 if len(cs)!=STEPS or len(xs)!=STEPS or any(not np.array_equal(a,b) for a,b in zip(cs,xs)):raise RuntimeError('paired fixed-decoder schedules differ')
 target_times_all=transform.transform(train.target_intervals).astype('float32',copy=False);totals={'control':[],'candidate':[]};names=('l1_cosine','residual_mse','l1_penalty','l2_penalty','l1_v_min','l2_v_min');components={arm:{k:[] for k in names} for arm in totals};first=None
 for step,(ci,xi) in enumerate(zip(cs,xs)):
  if not np.array_equal(ci,xi):raise RuntimeError('paired fixed-decoder schedule diverged')
  indices=ci;prefix_ids,prefix_times=_prefix_inputs(train,transform,'L2_SEP',indices);target_ids=_tensor_rows(train.target_type_ids,indices,dtype=torch.long);target_times=_tensor_rows(target_times_all,indices,dtype=torch.float32);mean,valid,lt,lv,rt,layers=target_bundle(teacher,target_ids,target_times);co.zero_grad(set_to_none=True);xo.zero_grad(set_to_none=True);closs,cparts,cabs=arm_forward(ce,cp,prefix_ids,prefix_times,mean,valid,lt,lv,rt,0.);xloss,xparts,xabs=arm_forward(xe,xp,prefix_ids,prefix_times,mean,valid,lt,lv,rt,1.)
  if step==0:
   common=('l1_cosine','residual_mse','l1_penalty','l2_penalty','l1_v_min','l2_v_min')
   if not torch.equal(cabs,xabs) or any(not torch.equal(cparts[k],xparts[k]) for k in common):raise RuntimeError('pre-update common predictions/losses differ')
   residual_grads=torch.autograd.grad(xparts['residual_mse'],list(xe.parameters()),retain_graph=True);grad_norm=math.sqrt(sum(float((g.detach()*g.detach()).sum()) for g in residual_grads));actual=xloss-closs
   if not math.isfinite(grad_norm) or not grad_norm>0 or not torch.allclose(actual,xparts['residual_mse'],rtol=1e-6,atol=1e-7):raise RuntimeError('residual-only encoder gradient/loss invariant failed')
   first={'actual_loss_difference':float(actual.detach()),'expected_residual_mse':float(xparts['residual_mse'].detach()),'residual_only_encoder_gradient_norm':grad_norm}
  for loss,opt,encoder,predictor,parts,arm in ((closs,co,ce,cp,cparts,'control'),(xloss,xo,xe,xp,xparts,'candidate')):
   _finite_loss(loss);loss.backward();parameters=list(encoder.parameters())+list(predictor.base.parameters())
   if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):raise FloatingPointError('fixed-decoder trainable gradient invariant failed')
   if any(p.grad is not None for p in decoder.parameters()):raise RuntimeError('fixed decoder accumulated gradient')
   opt.step();totals[arm].append(float(loss.detach()))
   for name,value in parts.items():components[arm][name].append(float(value.detach()))
 if any(p.grad is not None for p in list(teacher.parameters())+list(decoder.parameters())):raise RuntimeError('fixed modules accumulated gradients')
 audit={'first_update':first,'optimizer_updates_per_arm':STEPS,'phase_two_ema_updates_per_arm':0,'continuation_schedule_seed':seed+10000,'control_training':_loss_summary(totals['control'],components['control']),'candidate_training':_loss_summary(totals['candidate'],components['candidate'])}
 return TrainedCondition('L2_SEP',ce,teacher,cp,audit['control_training']),TrainedCondition('L2_SEP',xe,teacher,xp,audit['candidate_training']),audit

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};control_losses=[];candidate_losses=[];values=[[],[],[]]
 for seed in MODELS:
  base=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);teacher=copy.deepcopy(base.teacher).eval()
  for p in teacher.parameters():p.requires_grad_(False)
  decoder,fit_report=fit_fixed_decoder(base,teacher,train,transform,seed);teacher_before=digest_module(teacher);decoder_before=digest_module(decoder);control,candidate,audit=continue_pair(base,teacher,decoder,train,transform,seed);teacher_after=digest_module(teacher);decoder_after=digest_module(decoder)
  if teacher_before!=teacher_after or decoder_before!=decoder_after:raise RuntimeError('fixed teacher or decoder changed')
  cpf,ptr,pv=split_features_targets(control.encoder,teacher,probe,transform);ccf,ctr,cv=split_features_targets(control.encoder,teacher,cal,transform);xpf,xtr,xv=split_features_targets(candidate.encoder,teacher,probe,transform);xcf,xctr,xcv=split_features_targets(candidate.encoder,teacher,cal,transform)
  if not torch.equal(ptr,xtr) or not torch.equal(ctr,xctr) or not torch.equal(pv,xv) or not torch.equal(cv,xcv):raise RuntimeError('paired fixed-decoder probe targets differ')
  probe_stats=residual_stats(ptr,pv);cal_stats=residual_stats(ctr,cv);control_loss,control_rows=ridge_residual_probe(cpf,ccf,ptr,ctr,pv,cv);candidate_loss,candidate_rows=ridge_residual_probe(xpf,xcf,xtr,xctr,xv,xcv);control_losses.append(control_loss);candidate_losses.append(candidate_loss)
  freeze_encoder(control);freeze_encoder(candidate);probes,head,_=fit_condition_readouts(candidate,probe,transform,READOUT);ev,_=evaluate_readouts(candidate,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(candidate,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'teacher_digest_before_after':[teacher_before,teacher_after],'decoder_digest_before_after':[decoder_before,decoder_after],'decoder_fit_report':fit_report,'control_ridge_residual_cosine_loss':control_loss,'candidate_ridge_residual_cosine_loss':candidate_loss,'control_probe_rows':control_rows,'candidate_probe_rows':candidate_rows,'probe_fit_target_stats':probe_stats,'cal_ood_target_stats':cal_stats,'candidate_factor_ba':ba,'candidate_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'candidate_constraint_rows':rows,'candidate_position_specificity_descriptive':position,'continuation_audit':audit}
 mean_control=sum(control_losses)/3;mean_candidate=sum(candidate_losses)/3;relative=(mean_control-mean_candidate)/mean_control;means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'candidate_lower_probe_loss_all_three':all(a>b for a,b in zip(control_losses,candidate_losses)),'relative_reduction_between_mean_probe_losses_at_least_0_05':relative>=.05,'candidate_all_sixteen_constraints_all_three':all(v['candidate_constraint_transfer_all_pass'] for v in results.values()),'candidate_all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'candidate_each_factor_range_at_most_0_05':max(ranges)<=.05,'candidate_mean_gain_vs_untrained_at_least_0_05_each':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-FIXED-RESIDUAL-DECODER-V1','contract':{'decoder':'seed-specific TRAIN-fitted position-conditioned bias-free ridge maps lambda 1e-3, output-centered and fixed','only_arm_difference':'candidate adds unit-weight raw residual MSE through fixed decoder','primary_probe':'raw-pooled per-identity ridge lambda 1e-3, PROBE_FIT to CAL-OOD','primary_pass':'candidate lower all 3 and >=5% reduction between mean losses','claim':'conditional effect under prespecified fixed decoder and raw-pooled ridge assay'},'generator_seed':G,'model_seeds':MODELS,'decoder_ridge_lambda':DECODER_RIDGE,'control_residual_probe_cosine_loss':control_losses,'candidate_residual_probe_cosine_loss':candidate_losses,'relative_reduction_between_mean_probe_losses':relative,'candidate_factor_ba_by_factor':values,'candidate_factor_ba_means':means,'candidate_minus_untrained_mean_ba':deltas,'candidate_factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PRESPECIFIED_FIXED_DECODER_RAW_POOLED_RIDGE_EFFECT'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
