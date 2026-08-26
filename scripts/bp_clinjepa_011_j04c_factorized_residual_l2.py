#!/usr/bin/env python3
"""Matched factorized layer-residual L2 mechanism test."""
from __future__ import annotations
import copy,hashlib,json,math,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import sinusoidal_code
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_loss_summary,_prefix_inputs,_tensor_rows,evaluate_readouts,fit_condition_readouts,freeze_encoder,optimizer_membership,pretraining_indices,threshold_reference_report,trained_collapse_diagnostics
from clinical_jepa.targets.next_event_contract import construct_latent_targets,latent_objective
G=1102;MODELS=(2101,2102,2103);READOUT=2102;STEPS=2000;RIDGE=1e-3;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

class FactorizedPredictor(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=copy.deepcopy(base);self.residual_heads=nn.ModuleList([nn.Linear(16,16,bias=False) for _ in range(4)])
  with torch.no_grad():
   for head in self.residual_heads:head.weight.zero_()
 def factorized(self,pooled):
  shared=self.base(pooled,'L1_AVG');positions=sinusoidal_code(torch.arange(4,device=pooled.device),dtype=pooled.dtype);features=F.layer_norm(pooled[:,None,:]+positions[None,:,:],(16,),eps=1e-5);raw=torch.stack([head(features) for head in self.residual_heads],dim=2);residual=raw-raw.mean(dim=2,keepdim=True);absolute=shared.unsqueeze(2)+residual
  if absolute.shape!=(pooled.shape[0],4,4,16) or not torch.allclose(residual.sum(dim=2),torch.zeros_like(residual[:,:,0]),atol=1e-6,rtol=0):raise RuntimeError('factorized prediction geometry mismatch')
  return absolute,residual,shared
 def forward(self,pooled,recipe,*,k=4,layers=(1,2,3,4)):
  if recipe!='L2_SEP':return self.base(pooled,recipe,k=k,layers=layers)
  if k!=4 or list(layers)!=[1,2,3,4]:raise ValueError('factorized predictor requires ordered full L2 identities')
  return self.factorized(pooled)[0]

def digest_module(module):
 h=hashlib.sha256()
 for name,tensor in sorted(module.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def assert_cloned(left,right):
 ls=left.state_dict();rs=right.state_dict()
 if ls.keys()!=rs.keys():raise RuntimeError('clone state names differ')
 for name in ls:
  if not torch.equal(ls[name],rs[name]) or ls[name].data_ptr()==rs[name].data_ptr():raise RuntimeError(f'nonidentical or shared clone state: {name}')

def assert_zero_heads(predictor):
 weights=[head.weight for head in predictor.residual_heads]
 if len(weights)!=4 or any(bool(torch.count_nonzero(w)) for w in weights):raise RuntimeError('residual heads must initialize exactly zero')

def identity_penalty(prediction,weight):
 variance=F.normalize(prediction,dim=-1,eps=1e-8).var(dim=0,correction=0).mean(dim=-1);return weight*torch.clamp(.01-variance,min=0.).mean(),variance.min()

def target_bundle(teacher,target_ids,target_times):
 with torch.no_grad():blocks,_,_=teacher(target_ids,target_times,causal=True);mask=torch.ones_like(target_ids,dtype=torch.bool);mean,valid,_=construct_latent_targets(blocks,mask,'L1_AVG');layers_target,layer_valid,layers=construct_latent_targets(blocks,mask,'L2_SEP')
 if layers!=[1,2,3,4] or mean.shape!=(target_ids.shape[0],4,16) or layers_target.shape!=(target_ids.shape[0],4,4,16) or layer_valid.shape!=(target_ids.shape[0],4,4):raise RuntimeError('factorized target shape/order mismatch')
 arithmetic=layers_target.mean(dim=2)
 if not torch.allclose(mean,arithmetic,atol=1e-6,rtol=1e-6):raise RuntimeError('L1 target is not L2 arithmetic layer mean')
 residual=layers_target-mean.unsqueeze(2)
 if not torch.allclose(residual.sum(dim=2),torch.zeros_like(mean),atol=1e-6,rtol=0):raise RuntimeError('teacher residuals are not zero-sum')
 flat=residual.reshape(target_ids.shape[0],16,16)
 for position in range(4):
  for layer in range(4):
   if not torch.equal(flat[:,position*4+layer],residual[:,position,layer]):raise RuntimeError('identity flattening is not row-major position x layer')
 return mean.detach(),valid,layers_target.detach(),layer_valid,residual.detach(),layers

def residual_mse(prediction,target,valid):
 per=(prediction-target).square().mean(dim=-1);flat=per.reshape(per.shape[0],-1);mask=valid.reshape(valid.shape[0],-1);count=mask.sum(dim=1)
 if bool((count==0).any()):raise RuntimeError('empty residual target example')
 return ((flat*mask).sum(dim=1)/count).mean()

def arm_forward(encoder,predictor,prefix_ids,prefix_times,mean,valid,layer_target,layer_valid,residual_target,beta):
 _,_,pooled=encoder(prefix_ids,prefix_times,causal=False);p1=predictor(pooled,'L1_AVG');absolute,residual,shared=predictor.factorized(pooled)
 if not torch.equal(p1,shared) or not torch.equal(absolute,shared.unsqueeze(2)+residual):raise RuntimeError('L1 broadcast or factorized composition mismatch')
 l1,part1=latent_objective(p1,mean,valid);pen1,min1=identity_penalty(p1,5.);pen2,min2=identity_penalty(absolute,20.);r_loss=residual_mse(residual,residual_target,layer_valid);total=l1+pen1+pen2+beta*r_loss
 return total,{'l1_cosine':part1['cosine'],'residual_mse':r_loss,'l1_penalty':pen1,'l2_penalty':pen2,'l1_v_min':min1,'l2_v_min':min2}

def continue_pair(base,teacher,train,transform,seed):
 ce=copy.deepcopy(base.encoder).train();xe=copy.deepcopy(base.encoder).train();cp=FactorizedPredictor(base.predictor).train();xp=FactorizedPredictor(base.predictor).train();assert_cloned(ce,xe);assert_cloned(cp,xp);assert_zero_heads(cp);assert_zero_heads(xp)
 co=_adamw(list(ce.parameters())+list(cp.parameters()),lr=3e-4,weight_decay=1e-4);xo=_adamw(list(xe.parameters())+list(xp.parameters()),lr=3e-4,weight_decay=1e-4);cc=[{k:v for k,v in g.items() if k!='params'} for g in co.param_groups];xc=[{k:v for k,v in g.items() if k!='params'} for g in xo.param_groups]
 if co is xo or co.state or xo.state or cc!=xc or not optimizer_membership((ce,cp),co) or not optimizer_membership((xe,xp),xo):raise RuntimeError('paired residual optimizer invariant failed')
 teacher_ids={id(p) for p in teacher.parameters()};opt_ids={id(p) for opt in (co,xo) for g in opt.param_groups for p in g['params']}
 if teacher_ids&opt_ids or any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('frozen teacher gradient/optimizer leak')
 cs=[x.copy() for x in pretraining_indices(seed+10000)];xs=[x.copy() for x in pretraining_indices(seed+10000)]
 if len(cs)!=STEPS or len(xs)!=STEPS or any(not np.array_equal(a,b) for a,b in zip(cs,xs)):raise RuntimeError('paired residual schedules differ')
 target_times_all=transform.transform(train.target_intervals).astype('float32',copy=False);totals={'control':[],'candidate':[]};names=('l1_cosine','residual_mse','l1_penalty','l2_penalty','l1_v_min','l2_v_min');components={arm:{k:[] for k in names} for arm in totals};first=None
 for step,(ci,xi) in enumerate(zip(cs,xs)):
  if not np.array_equal(ci,xi):raise RuntimeError('paired schedule diverged')
  indices=ci
  prefix_ids,prefix_times=_prefix_inputs(train,transform,'L2_SEP',indices);target_ids=_tensor_rows(train.target_type_ids,indices,dtype=torch.long);target_times=_tensor_rows(target_times_all,indices,dtype=torch.float32);mean,valid,lt,lv,rt,layers=target_bundle(teacher,target_ids,target_times);co.zero_grad(set_to_none=True);xo.zero_grad(set_to_none=True);closs,cparts=arm_forward(ce,cp,prefix_ids,prefix_times,mean,valid,lt,lv,rt,0.);xloss,xparts=arm_forward(xe,xp,prefix_ids,prefix_times,mean,valid,lt,lv,rt,1.)
  if step==0:
   actual=xloss-closs;expected=xparts['residual_mse'];head_grads=torch.autograd.grad(expected,list(xp.residual_heads.parameters()),retain_graph=True);grad_norm=math.sqrt(sum(float((g.detach()*g.detach()).sum()) for g in head_grads))
   if not bool(torch.isfinite(expected)) or not float(expected.detach())>0 or not math.isfinite(grad_norm) or not grad_norm>0 or not torch.allclose(actual,expected,rtol=1e-6,atol=1e-7):raise RuntimeError('first residual objective/gradient invariant failed')
   first={'actual_loss_difference':float(actual.detach()),'expected_residual_mse':float(expected.detach()),'residual_head_gradient_norm':grad_norm}
  for loss,opt,encoder,predictor,parts,arm in ((closs,co,ce,cp,cparts,'control'),(xloss,xo,xe,xp,xparts,'candidate')):
   _finite_loss(loss);loss.backward()
   parameters=list(encoder.parameters())+list(predictor.parameters())
   if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise FloatingPointError('factorized continuation gradient invariant failed')
   if arm=='candidate' and any(p.grad is None for p in predictor.residual_heads.parameters()):raise FloatingPointError('candidate residual head lost target gradient')
   opt.step();totals[arm].append(float(loss.detach()))
   for name,value in parts.items():components[arm][name].append(float(value.detach()))
 if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('frozen teacher accumulated gradients')
 audit={'first_update':first,'optimizer_updates_per_arm':STEPS,'phase_two_ema_updates_per_arm':0,'continuation_schedule_seed':seed+10000,'control_training':_loss_summary(totals['control'],components['control']),'candidate_training':_loss_summary(totals['candidate'],components['candidate'])}
 return TrainedCondition('L2_SEP',ce,teacher,cp,audit['control_training']),TrainedCondition('L2_SEP',xe,teacher,xp,audit['candidate_training']),audit

def split_features_targets(encoder,teacher,split,transform):
 ids,times=_prefix_inputs(split,transform,'L2_SEP');target_ids=torch.as_tensor(split.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32,copy=False));mean,valid,lt,lv,residual,layers=target_bundle(teacher,target_ids,target_times)
 with torch.no_grad():_,_,pooled=encoder(ids,times,causal=False)
 return pooled.detach(),residual,lv

def residual_stats(residual,valid):
 rows=[]
 for position in range(4):
  for layer in range(4):
   mask=valid[:,position,layer];r=residual[mask,position,layer].double();norms=torch.linalg.vector_norm(r,dim=1);centered=r-r.mean(dim=0);eigen=torch.linalg.eigvalsh(centered.T@centered/r.shape[0]).clamp_min(0);total=float(eigen.sum());p=eigen[eigen>0]/eigen.sum();erank=float(torch.exp(-(p*torch.log(p)).sum())) if total>0 else 0.;rows.append({'identity_index':position*4+layer,'position_index':position,'layer_index':layer,'mean_norm':float(norms.mean()),'minimum_norm':float(norms.min()),'entropy_effective_rank':erank})
 if any(row['minimum_norm']<=1e-6 or row['entropy_effective_rank']<=1 for row in rows):raise RuntimeError('teacher residual target degeneracy')
 return rows

def ridge_residual_probe(train_features,eval_features,train_target,eval_target,train_valid,eval_valid):
 losses=[];rows=[];identity=0
 for position in range(4):
  for layer in range(4):
   tm=train_valid[:,position,layer];em=eval_valid[:,position,layer];x=train_features[tm].double();y=train_target[tm,position,layer].double();xe=eval_features[em].double();ye=eval_target[em,position,layer].double();xa=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],dim=1);xea=torch.cat([xe,torch.ones((xe.shape[0],1),dtype=torch.float64)],dim=1);reg=torch.eye(17,dtype=torch.float64)*RIDGE;reg[-1,-1]=0.;coef=torch.linalg.solve(xa.T@xa+reg,xa.T@y);pred=xea@coef;loss=1.-F.cosine_similarity(pred,ye,dim=-1,eps=1e-8);losses.extend(loss.tolist());rows.append({'identity_index':identity,'position_index':position,'layer_index':layer,'cosine_loss':float(loss.mean()),'fit_rows':int(tm.sum()),'eval_rows':int(em.sum())});identity+=1
 return float(np.mean(losses)),rows

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};control_losses=[];candidate_losses=[];values=[[],[],[]]
 for seed in MODELS:
  base=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);teacher=copy.deepcopy(base.teacher).eval()
  for p in teacher.parameters():p.requires_grad_(False)
  before=digest_module(teacher);control,candidate,audit=continue_pair(base,teacher,train,transform,seed);after=digest_module(teacher)
  if before!=after:raise RuntimeError('factorized frozen teacher changed')
  cpf,ptr,pv=split_features_targets(control.encoder,teacher,probe,transform);ccf,ctr,cv=split_features_targets(control.encoder,teacher,cal,transform);xpf,xtr,xv=split_features_targets(candidate.encoder,teacher,probe,transform);xcf,xctr,xcv=split_features_targets(candidate.encoder,teacher,cal,transform)
  if not torch.equal(ptr,xtr) or not torch.equal(ctr,xctr) or not torch.equal(pv,xv) or not torch.equal(cv,xcv):raise RuntimeError('paired residual probe targets or masks differ')
  probe_stats=residual_stats(ptr,pv);cal_stats=residual_stats(ctr,cv);control_loss,control_rows=ridge_residual_probe(cpf,ccf,ptr,ctr,pv,cv);candidate_loss,candidate_rows=ridge_residual_probe(xpf,xcf,xtr,xctr,xv,xcv);control_losses.append(control_loss);candidate_losses.append(candidate_loss)
  freeze_encoder(control);freeze_encoder(candidate);probes,head,_=fit_condition_readouts(candidate,probe,transform,READOUT);ev,_=evaluate_readouts(candidate,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(candidate,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'teacher_digest_before_after':[before,after],'control_ridge_residual_cosine_loss':control_loss,'candidate_ridge_residual_cosine_loss':candidate_loss,'control_probe_rows':control_rows,'candidate_probe_rows':candidate_rows,'probe_fit_target_stats':probe_stats,'cal_ood_target_stats':cal_stats,'candidate_factor_ba':ba,'candidate_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'candidate_constraint_rows':rows,'candidate_position_specificity_descriptive':position,'continuation_audit':audit}
 mean_control=sum(control_losses)/3;mean_candidate=sum(candidate_losses)/3;relative=(mean_control-mean_candidate)/mean_control;means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'candidate_lower_residual_probe_loss_all_three':all(a>b for a,b in zip(control_losses,candidate_losses)),'relative_reduction_between_mean_probe_losses_at_least_0_05':relative>=.05,'candidate_all_sixteen_constraints_all_three':all(v['candidate_constraint_transfer_all_pass'] for v in results.values()),'candidate_all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'candidate_each_factor_range_at_most_0_05':max(ranges)<=.05,'candidate_mean_gain_vs_untrained_at_least_0_05_each':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-FACTORIZED-RESIDUAL-L2-V1','contract':{'only_arm_difference':'candidate adds unit-weight raw zero-sum layer-residual MSE','residual_predictor':'four zero-initialized bias-free layer heads, centered across layers','primary_probe':'per-identity float64 ridge lambda 1e-3 with unpenalized intercept, PROBE_FIT to CAL-OOD','primary_pass':'candidate lower all 3 seeds and >=5% reduction between across-seed mean losses','claim':'fixed-generator factorized residual encoder effect under frozen post-L1 teacher'},'generator_seed':G,'model_seeds':MODELS,'ridge_lambda':RIDGE,'control_residual_probe_cosine_loss':control_losses,'candidate_residual_probe_cosine_loss':candidate_losses,'relative_reduction_between_mean_probe_losses':relative,'candidate_factor_ba_by_factor':values,'candidate_factor_ba_means':means,'candidate_minus_untrained_mean_ba':deltas,'candidate_factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_FACTORIZED_LAYER_RESIDUAL_ENCODER_EFFECT'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
