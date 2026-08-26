#!/usr/bin/env python3
"""Post-result class-balanced readout localization for the red O2 family rung."""
from __future__ import annotations
import json,numpy as np,torch
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import EMATeacher,SharedLatentPredictor
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_fresh_encoder,balanced_accuracy,freeze_encoder,frozen_representations,make_readout_initializations,probe_indices,threshold_reference_report,trained_collapse_diagnostics
from bp_clinjepa_011_j04c_l1_generator_family import GAIN_THRESHOLDS,HELDOUT,MODELS,RANGE_MAX,READOUT,make_family_train,parameterized_split

def untrained_condition(seed):
 encoder=_fresh_encoder(seed).eval();teacher=EMATeacher(encoder,momentum=.996).eval();torch.manual_seed(seed);predictor=SharedLatentPredictor().eval();set_identity_gelu_predictor(predictor);return TrainedCondition('L1_AVG',encoder,teacher,predictor,{'untrained_seed':seed})

def fit_probe(z,labels,factor,weighted):
 probes,_=make_readout_initializations(READOUT);probe=probes[factor];probe.train();optimizer=_adamw(list(probe.parameters()),lr=1e-2,weight_decay=1e-3);target=torch.as_tensor(labels,dtype=torch.float32);counts=np.bincount(labels.astype(np.int64),minlength=2)
 if np.any(counts==0):raise RuntimeError('class-balanced probe requires both classes')
 weights=torch.as_tensor(np.where(labels==0,len(labels)/(2*counts[0]),len(labels)/(2*counts[1])),dtype=torch.float32)
 for indices in probe_indices(READOUT,factor):
  idx=torch.as_tensor(indices,dtype=torch.long);optimizer.zero_grad(set_to_none=True);logits=probe(z[idx]).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,target[idx],weight=(weights[idx] if weighted else None));_finite_loss(loss);loss.backward()
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):raise FloatingPointError('probe gradient invariant failed')
  optimizer.step()
 probe.eval();return probe,{'fit_class_counts':counts.tolist(),'sample_weights':([float(len(labels)/(2*counts[0])),float(len(labels)/(2*counts[1]))] if weighted else [1.,1.]),'updates':250}

def evaluate_probe(probe,z,labels):
 with torch.no_grad():prediction=probe(z).squeeze(-1).ge(0).to(torch.uint8).cpu().numpy()
 labels=labels.astype(np.uint8);tp=int(np.sum((prediction==1)&(labels==1)));tn=int(np.sum((prediction==0)&(labels==0)));fp=int(np.sum((prediction==1)&(labels==0)));fn=int(np.sum((prediction==0)&(labels==1)));return {'balanced_accuracy':balanced_accuracy(prediction,labels),'predicted_positive_fraction':float(prediction.mean()),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'predicts_both_classes':bool(np.any(prediction==0) and np.any(prediction==1))}

def arm_readouts(condition,probe_split,cal_split,transform):
 freeze_encoder(condition);fit_reps=frozen_representations(condition,probe_split,transform);eval_reps=frozen_representations(condition,cal_split,transform);result={'original':[],'class_balanced':[]}
 for factor in range(3):
  for weighted,key in ((False,'original'),(True,'class_balanced')):
   probe,report=fit_probe(fit_reps.z,probe_split.S[:,factor],factor,weighted);metrics=evaluate_probe(probe,eval_reps.z,cal_split.S[:,factor]);result[key].append({**metrics,'fit_report':report})
 return result

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);base_train=independent_train_nuisance(generate_factor_split(1102,TRAIN,8192),1102);transform=fit_stage0_time_transform(base_train);transform_state=transform.state_bytes().hex();family_train=make_family_train();reference,threshold_digest=threshold_reference_report();conditions={}
 for seed in MODELS:conditions[str(seed)]={'family':train_recipe_decoupled_seeds(family_train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True),'single1102':train_recipe_decoupled_seeds(base_train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True),'untrained':untrained_condition(seed)}
 if transform.state_bytes().hex()!=transform_state:raise RuntimeError('frozen transform changed')
 cells=[]
 for hseed,sp,xp in HELDOUT:
  probe_split=parameterized_split(hseed,PROBE_FIT,2048,sp,xp);cal_split=parameterized_split(hseed,CAL_OOD,2048,sp,xp)
  for seed in MODELS:
   arms={arm:arm_readouts(conditions[str(seed)][arm],probe_split,cal_split,transform) for arm in ('family','single1102','untrained')};rows,position=trained_collapse_diagnostics(conditions[str(seed)]['family'],cal_split,transform,reference);cells.append({'heldout_seed':hseed,'model_seed':seed,'arms':arms,'family_constraints_all_pass':all(row['both_metrics_pass'] for row in rows),'family_constraint_rows':rows,'family_position_specificity_descriptive':position})
 gains=[[],[],[]];margins=[[],[],[]];bas=[[],[],[]];per_generator_gain={};per_generator_margin={};per_generator_ranges={}
 for cell in cells:
  for f in range(3):
   family=cell['arms']['family']['class_balanced'][f]['balanced_accuracy'];single=cell['arms']['single1102']['class_balanced'][f]['balanced_accuracy'];untrained=cell['arms']['untrained']['class_balanced'][f]['balanced_accuracy'];gains[f].append(family-untrained);margins[f].append(family-single);bas[f].append(family)
 for hseed,_,_ in HELDOUT:
  group=[c for c in cells if c['heldout_seed']==hseed];per_generator_gain[str(hseed)]=[float(np.mean([c['arms']['family']['class_balanced'][f]['balanced_accuracy']-c['arms']['untrained']['class_balanced'][f]['balanced_accuracy'] for c in group])) for f in range(3)];per_generator_margin[str(hseed)]=[float(np.mean([c['arms']['family']['class_balanced'][f]['balanced_accuracy']-c['arms']['single1102']['class_balanced'][f]['balanced_accuracy'] for c in group])) for f in range(3)];per_generator_ranges[str(hseed)]=[max(c['arms']['family']['class_balanced'][f]['balanced_accuracy'] for c in group)-min(c['arms']['family']['class_balanced'][f]['balanced_accuracy'] for c in group) for f in range(3)]
 mean_gains=[float(np.mean(x)) for x in gains];mean_margins=[float(np.mean(x)) for x in margins];affected=[c for c in cells if c['heldout_seed'] in (4100,4101)];readout_contribution=all(c['arms']['family']['class_balanced'][0]['balanced_accuracy']>=.60 and c['arms']['family']['class_balanced'][0]['predicts_both_classes'] for c in affected);primary_checks={'aggregate_gain_thresholds':[mean_gains[f]>=GAIN_THRESHOLDS[f] for f in range(3)],'positive_gain_each_generator_factor':all(v>0 for row in per_generator_gain.values() for v in row),'all_family_factor_ba_at_least_0_80':min(x for row in bas for x in row)>=.8,'seed_ranges_each_generator_factor_at_most_0_0538':all(v<=RANGE_MAX for row in per_generator_ranges.values() for v in row),'constraints_pass_at_least_8_of_9':sum(c['family_constraints_all_pass'] for c in cells)>=8};margin_checks={'positive_overall_margin_every_factor':all(v>0 for v in mean_margins),'positive_margin_each_generator_factor':all(v>0 for row in per_generator_margin.values() for v in row)};print(json.dumps({'schema':'BP011-J04C-L1-CLASS-BALANCED-READOUT-V1','contract':{'status':'post-result prospective localization; original O2 remains red','only_change':'inverse-frequency BCE sample weighting on complete PROBE_FIT','readout_contribution_gate':'affected family composition BA >=0.60 and both predicted classes','family_applicability_gate':'all original primary conditions recomputed class-balanced','mixed_family_advantage_gate':'all original paired margins recomputed class-balanced'},'model_seeds':MODELS,'heldout_family':HELDOUT,'readout_seed':READOUT,'frozen_transform_state_hex':transform_state,'cells':cells,'readout_contribution_gate_pass':readout_contribution,'mean_class_balanced_family_minus_untrained':mean_gains,'per_generator_class_balanced_family_minus_untrained':per_generator_gain,'per_generator_class_balanced_seed_ranges':per_generator_ranges,'class_balanced_primary_checks':primary_checks,'class_balanced_family_applicability_gate_pass':all(primary_checks['aggregate_gain_thresholds']) and all(v for k,v in primary_checks.items() if k!='aggregate_gain_thresholds'),'mean_class_balanced_family_minus_single1102':mean_margins,'per_generator_class_balanced_family_minus_single1102':per_generator_margin,'class_balanced_margin_checks':margin_checks,'class_balanced_mixed_family_advantage_gate_pass':all(margin_checks.values()),'threshold_digest':threshold_digest,'claim_ceiling':'POST_RESULT_CLASS_BALANCED_READOUT_LOCALIZATION_ON_PINNED_SAME_SUPPORT_SHIFTS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
