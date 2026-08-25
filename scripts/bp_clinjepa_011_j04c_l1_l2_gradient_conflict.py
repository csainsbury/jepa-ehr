#!/usr/bin/env python3
"""Measure L1/L2 encoder-gradient alignment under the last-green L1 state."""
from __future__ import annotations
import json,math,torch
from clinical_jepa.eval.j04c_falsifier import PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _prefix_inputs
from clinical_jepa.targets.next_event_contract import construct_latent_targets,latent_objective
G=1102;MODELS=(2101,2102,2103)

def group(name):
 if name.startswith('blocks.'):
  return '.'.join(name.split('.')[:2])
 return 'input' if name.startswith(('type_embedding','time_projection','input_norm')) else 'final_norm'

def alignment(named,g1,g2):
 grouped={}
 for (name,_),a,b in zip(named,g1,g2):
  key=group(name);entry=grouped.setdefault(key,[0.,0.,0.]);entry[0]+=float((a*b).sum());entry[1]+=float((a*a).sum());entry[2]+=float((b*b).sum())
 def finish(v):
  dot,n1,n2=v;return {'dot':dot,'l1_norm':math.sqrt(n1),'l2_norm':math.sqrt(n2),'cosine':dot/math.sqrt(n1*n2),'l2_to_l1_norm_ratio':math.sqrt(n2/n1)}
 total=[sum(v[i] for v in grouped.values()) for i in range(3)]
 return {'global':finish(total),'groups':{k:finish(v) for k,v in grouped.items()}}

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);held=independent_train_nuisance(generate_factor_split(G,PROBE_FIT,512),G);transform=fit_stage0_time_transform(train);results={};cosines=[]
 for seed in MODELS:
  c=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);named=list(c.encoder.named_parameters());params=[p for _,p in named];ids,times=_prefix_inputs(held,transform,'L2_SEP');target_ids=torch.as_tensor(held.target_type_ids,dtype=torch.long);target_times=torch.as_tensor(transform.transform(held.target_intervals),dtype=torch.float32);_,_,pooled=c.encoder(ids,times,causal=False);blocks,_,_=c.teacher(target_ids,target_times,causal=True);mask=torch.ones_like(target_ids,dtype=torch.bool);t1,v1,_=construct_latent_targets(blocks,mask,'L1_AVG');t2,v2,layers=construct_latent_targets(blocks,mask,'L2_SEP');p1=c.predictor(pooled,'L1_AVG');p2=c.predictor(pooled,'L2_SEP',layers=layers);loss1,parts1=latent_objective(p1,t1.detach(),v1);loss2,parts2=latent_objective(p2,t2.detach(),v2);g1_cos=torch.autograd.grad(parts1['cosine'],params,retain_graph=True);g2_cos=torch.autograd.grad(parts2['cosine'],params,retain_graph=True);g1_full=torch.autograd.grad(loss1,params,retain_graph=True);g2_full=torch.autograd.grad(loss2,params);cosine_alignment=alignment(named,g1_cos,g2_cos);full_alignment=alignment(named,g1_full,g2_full);cosines.append(cosine_alignment['global']['cosine']);results[str(seed)]={'heldout_l1_cosine_loss':float(parts1['cosine'].detach()),'heldout_l2_cosine_loss':float(parts2['cosine'].detach()),'cosine_target_gradient_alignment':cosine_alignment,'complete_target_gradient_alignment':full_alignment}
 checks={'negative_mean_global_cosine_target_gradient':sum(cosines)/3<0,'at_least_two_of_three_negative':sum(x<0 for x in cosines)>=2};print(json.dumps({'schema':'BP011-J04C-L1-L2-GRADIENT-CONFLICT-V1','contract':{'state':'last-green L1 recipe, generator 1102, model seeds 2101-2103','batch':'fixed independently nuisance-severed PROBE_FIT split used as held-out pretraining-format diagnostic','primary':'global encoder-gradient cosine between L1 and full-L2 cosine target losses','conflict_support':'negative mean and at least 2/3 negative seed cosines','constraints_excluded':'directional penalties excluded from primary target-gradient diagnostic'},'global_cosine_target_gradient_by_seed':cosines,'mean_global_cosine_target_gradient':sum(cosines)/3,'checks':checks,'gradient_conflict_supported':all(checks.values()),'results':results,'claim_ceiling':'FIXED_GENERATOR_DIAGNOSTIC_ONLY'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
