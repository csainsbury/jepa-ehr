#!/usr/bin/env python3
"""Prospective within-support generator-family generalization rung for frozen L1_AVG."""
from __future__ import annotations
import json,numpy as np,torch
from clinical_jepa.arms.v0f.own_latent import EMATeacher,SharedLatentPredictor
from clinical_jepa.eval.j04c_falsifier import ANCHOR,CAL_OOD,NUIS_COMP_0,NUIS_COMP_1,NUIS_ORDER_0,NUIS_ORDER_1,PROBE_FIT,SIG_COMP_0,SIG_COMP_1,SIG_ORDER_0,SIG_ORDER_1,TIME_NUIS,TIME_SIGNAL,TRAIN,TYPE_A,TYPE_B,TYPE_C,SyntheticFactorSplit,_validate_split,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_fresh_encoder,evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
TRAIN_FAMILY=((3100,(.50,.50,.50),(.15,.15,.15)),(3101,(.35,.50,.65),(.10,.15,.20)),(3102,(.65,.50,.35),(.20,.10,.15)),(3103,(.40,.60,.50),(.15,.25,.10)),(3104,(.60,.40,.50),(.25,.15,.10)),(3105,(.50,.35,.65),(.10,.20,.25)),(3106,(.50,.65,.35),(.20,.10,.25)),(3107,(.40,.40,.60),(.25,.20,.10)))
HELDOUT=((4100,(.35,.55,.65),(.10,.22,.18)),(4101,(.65,.45,.35),(.22,.10,.18)),(4102,(.55,.65,.45),(.18,.22,.10)))
MODELS=(2101,2102,2103);READOUT=2102;GAIN_THRESHOLDS=(.033,.034,.052);RANGE_MAX=.0538

def parameterized_split(seed,split_code,n,s_prob,x_flip_prob):
 rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,split_code])));sp=np.asarray(s_prob);xp=np.asarray(x_flip_prob);s=(rng.random((n,3))<sp).astype(np.uint8);x=np.bitwise_xor(s,(rng.random((n,3))<xp).astype(np.uint8));correlated=split_code==TRAIN;n_draw=(rng.random((n,3))<(0.05 if correlated else 0.5)).astype(np.uint8);nuis=np.bitwise_xor(s,n_draw) if correlated else n_draw;pt=np.empty((n,7),dtype=np.int64);pt[:,0]=ANCHOR;pt[:,1]=np.where(x[:,0]==0,SIG_COMP_0,SIG_COMP_1);pt[:,2]=np.where(x[:,1]==0,SIG_ORDER_0,SIG_ORDER_1);pt[:,3]=TIME_SIGNAL;pt[:,4]=np.where(nuis[:,0]==0,NUIS_COMP_0,NUIS_COMP_1);pt[:,5]=np.where(nuis[:,1]==0,NUIS_ORDER_0,NUIS_ORDER_1);pt[:,6]=TIME_NUIS;pi=np.ones((n,7),dtype=np.float64);pi[:,0]=0.;pi[:,3]=np.where(x[:,2]==0,1.,4.);pi[:,6]=np.where(nuis[:,2]==0,1.,4.);tt=np.empty((n,4),dtype=np.int64);ti=np.empty((n,4),dtype=np.float64)
 for row in range(n):
  types=[TYPE_A,TYPE_A,TYPE_B,TYPE_C] if s[row,0]==0 else [TYPE_A,TYPE_B,TYPE_B,TYPE_C]
  if s[row,1]==1:types.reverse()
  tt[row]=types;ti[row]=[1.,1.,4.,4.] if s[row,2]==0 else [4.,4.,1.,1.]
 result=SyntheticFactorSplit(pt,pi,tt,ti,s,x,nuis,s.copy());_validate_split(result);return result

def concatenate(splits):
 result=SyntheticFactorSplit(*(np.concatenate([getattr(x,name) for x in splits],axis=0) for name in ('prefix_type_ids','prefix_intervals','target_type_ids','target_intervals','S','X','N','L_after')));_validate_split(result);return result

def make_family_train():
 members=[]
 for seed,sp,xp in TRAIN_FAMILY:members.append(independent_train_nuisance(parameterized_split(seed,TRAIN,1024,sp,xp),seed))
 result=concatenate(members)
 if len(result.S)!=8192 or any(not np.array_equal(result.S[i*1024:(i+1)*1024],members[i].S) for i in range(8)):raise RuntimeError('family concatenation/order invariant failed')
 return result

def untrained_condition(seed):
 encoder=_fresh_encoder(seed).eval();teacher=EMATeacher(encoder,momentum=.996).eval();torch.manual_seed(seed);predictor=SharedLatentPredictor().eval();set_identity_gelu_predictor(predictor);return TrainedCondition('L1_AVG',encoder,teacher,predictor,{'untrained_seed':seed})

def ba_for(condition,probe,cal,transform):
 freeze_encoder(condition);probes,head,report=fit_condition_readouts(condition,probe,transform,READOUT);metrics,_=evaluate_readouts(condition,probes,head,cal,transform);return [row['balanced_accuracy'] for row in metrics['factors']],report

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);base_train=independent_train_nuisance(generate_factor_split(1102,TRAIN,8192),1102);transform=fit_stage0_time_transform(base_train);frozen_transform_state=transform.state_bytes().hex();family_train=make_family_train();reference,threshold_digest=threshold_reference_report();conditions={}
 for seed in MODELS:
  conditions[str(seed)]={'family':train_recipe_decoupled_seeds(family_train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True),'single1102':train_recipe_decoupled_seeds(base_train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True),'untrained':untrained_condition(seed)}
 if transform.state_bytes().hex()!=frozen_transform_state:raise RuntimeError('frozen 1102 transform changed')
 cells=[]
 for hseed,sp,xp in HELDOUT:
  probe=parameterized_split(hseed,PROBE_FIT,2048,sp,xp);cal=parameterized_split(hseed,CAL_OOD,2048,sp,xp)
  for seed in MODELS:
   arm_ba={};readout={}
   for arm in ('family','single1102','untrained'):arm_ba[arm],readout[arm]=ba_for(conditions[str(seed)][arm],probe,cal,transform)
   gains=[arm_ba['family'][f]-arm_ba['untrained'][f] for f in range(3)];single_gains=[arm_ba['single1102'][f]-arm_ba['untrained'][f] for f in range(3)];margins=[arm_ba['family'][f]-arm_ba['single1102'][f] for f in range(3)];rows,position=trained_collapse_diagnostics(conditions[str(seed)]['family'],cal,transform,reference);cells.append({'heldout_seed':hseed,'s_probabilities':sp,'signal_flip_probabilities':xp,'model_seed':seed,'factor_ba':arm_ba,'family_minus_untrained':gains,'single1102_minus_untrained':single_gains,'family_minus_single1102':margins,'family_constraints_all_pass':all(row['both_metrics_pass'] for row in rows),'family_constraint_rows':rows,'family_position_specificity_descriptive':position,'readout_seed':READOUT,'separate_readout_reports':readout})
 family_gains=[[],[],[]];margins=[[],[],[]];family_bas=[[],[],[]]
 for cell in cells:
  for f in range(3):family_gains[f].append(cell['family_minus_untrained'][f]);margins[f].append(cell['family_minus_single1102'][f]);family_bas[f].append(cell['factor_ba']['family'][f])
 per_generator_gain={};per_generator_margin={};per_generator_ranges={}
 for hseed,_,_ in HELDOUT:
  group=[c for c in cells if c['heldout_seed']==hseed];per_generator_gain[str(hseed)]=[float(np.mean([c['family_minus_untrained'][f] for c in group])) for f in range(3)];per_generator_margin[str(hseed)]=[float(np.mean([c['family_minus_single1102'][f] for c in group])) for f in range(3)];per_generator_ranges[str(hseed)]=[max(c['factor_ba']['family'][f] for c in group)-min(c['factor_ba']['family'][f] for c in group) for f in range(3)]
 mean_gains=[float(np.mean(x)) for x in family_gains];mean_margins=[float(np.mean(x)) for x in margins];checks={'aggregate_gain_thresholds':[mean_gains[f]>=GAIN_THRESHOLDS[f] for f in range(3)],'positive_gain_each_generator_factor':all(v>0 for row in per_generator_gain.values() for v in row),'all_family_factor_ba_at_least_0_80':min(x for row in family_bas for x in row)>=.8,'seed_ranges_each_generator_factor_at_most_0_0538':all(v<=RANGE_MAX for row in per_generator_ranges.values() for v in row),'constraints_pass_at_least_8_of_9':sum(c['family_constraints_all_pass'] for c in cells)>=8,'positive_overall_margin_every_factor':all(v>0 for v in mean_margins),'positive_margin_each_generator_factor':all(v>0 for row in per_generator_margin.values() for v in row)};primary=all(checks['aggregate_gain_thresholds']) and checks['positive_gain_each_generator_factor'] and checks['all_family_factor_ba_at_least_0_80'] and checks['seed_ranges_each_generator_factor_at_most_0_0538'] and checks['constraints_pass_at_least_8_of_9'];margin=checks['positive_overall_margin_every_factor'] and checks['positive_margin_each_generator_factor'];print(json.dumps({'schema':'BP011-J04C-L1-GENERATOR-FAMILY-V1','train_family':TRAIN_FAMILY,'heldout_family':HELDOUT,'model_seeds':MODELS,'readout_seed':READOUT,'frozen_1102_time_transform_state_hex':frozen_transform_state,'gain_thresholds':GAIN_THRESHOLDS,'range_maximum':RANGE_MAX,'cells':cells,'mean_family_minus_untrained':mean_gains,'per_generator_family_minus_untrained':per_generator_gain,'mean_family_minus_single1102':mean_margins,'per_generator_family_minus_single1102':per_generator_margin,'per_generator_seed_ranges':per_generator_ranges,'checks':checks,'primary_gate_pass':primary,'transfer_margin_gate_pass':margin,'full_contract_pass':primary and margin,'fallback_authorized_only_if_all_three_aggregate_gain_thresholds_red':not any(checks['aggregate_gain_thresholds']),'threshold_digest':threshold_digest,'claim_ceiling':'WITHIN_SUPPORT_PINNED_PREVALENCE_CORRUPTION_GENERALIZATION_WITH_PARAMETER_SPECIFIC_READOUTS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
