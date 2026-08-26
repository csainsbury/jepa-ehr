#!/usr/bin/env python3
"""Corruption-only within-support beta for the frozen generator-1102 L1 recipe."""
from __future__ import annotations
import hashlib,json,sys,numpy as np,torch
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import EMATeacher,SharedLatentPredictor
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_fresh_encoder,balanced_accuracy,freeze_encoder,frozen_representations,make_readout_initializations,pretraining_indices,probe_indices,threshold_reference_report,trained_collapse_diagnostics
from bp_clinjepa_011_j04c_l1_generator_family import GAIN_THRESHOLDS,RANGE_MAX,READOUT,parameterized_split
MODELS=(2101,2102,2103);HELDOUT=((5100,(.5,.5,.5),(.10,.22,.18)),(5101,(.5,.5,.5),(.22,.10,.18)),(5102,(.5,.5,.5),(.18,.22,.10)))
EXPECTED_RUNTIME=('3.10.12','2.9.1+cu128','2.2.6');EXPECTED_TRANSFORM='000000000000f03f16bd6a41e226ec3f9ef90546da31df3f';EXPECTED_TRAIN='5029185a98e0b0b4ce0f8506ad814755c1e365b9f93686ebfa806db379a11f3e';EXPECTED_SCHEDULE={2101:'d9f54fe1d72a6b8e17c88d53be7814f53f6e268633a564857ea12702aa8ee2bc',2102:'a3acc86a09fb9657b85e626456cf07dbee0377965fb3b1fcf237e7e41f4be778',2103:'d4249f8bc9483955e437f6050015dbc50aee4fe801c671b4a15191616d7cb14f'}
EXPECTED_SOURCES={'clinical_jepa/eval/j04c_initialization_bridge.py':'099237fc24382f1015df64d53a1017918d3a10d0bb741d6eed98522f4f2b23d9','clinical_jepa/eval/j04c_nuisance_bridge.py':'521d64d60cd91f041cf61c1bd94a2a45fb401772963d5ac4667781aa861eea2e','clinical_jepa/eval/j04c_falsifier.py':'206bf1d59d36a180168b6bb0954d68db50f46b3a0828c6c4c9bbc2e19c843a0a','clinical_jepa/eval/j04c_stage1.py':'167300f6a075b07a2cfcc53fe15c8b507120a9eeb0a553da37841a69b2511bb2','clinical_jepa/targets/next_event_contract.py':'7903f587996a2fd82fde5b13316f853cd06b2c1c8c13eca642dedcad7f8c755d','clinical_jepa/arms/v0f/own_latent.py':'f3cd838225f8099c79d604036961cc64170c98a87b925fd960550846b7c50dfb','scripts/bp_clinjepa_011_j04c_l1_generator_family.py':'f5c472768ecd2453f32455ac3c959b707bf8c9f025336e9ad5d92eae6fcb3950'}

def sha_file(path):
 h=hashlib.sha256();h.update(open(path,'rb').read());return h.hexdigest()
def array_bundle_digest(split):
 h=hashlib.sha256()
 for name in ('prefix_type_ids','prefix_intervals','target_type_ids','target_intervals','S','X','N','L_after'):
  a=getattr(split,name);h.update(name.encode());h.update(a.dtype.str.encode());h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes())
 return h.hexdigest()
def schedule_digest(seed):
 h=hashlib.sha256()
 for a in pretraining_indices(seed):h.update(np.asarray(a,dtype='<i8').tobytes())
 return h.hexdigest()
def state_digest(module):
 h=hashlib.sha256()
 for name,tensor in sorted(module.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def split_digest(split):return array_bundle_digest(split)
def untrained_condition(seed):
 encoder=_fresh_encoder(seed).eval();teacher=EMATeacher(encoder,momentum=.996).eval();torch.manual_seed(seed);predictor=SharedLatentPredictor().eval();set_identity_gelu_predictor(predictor);return TrainedCondition('L1_AVG',encoder,teacher,predictor,{'untrained_seed':seed})
def fit_balanced_probe(z,labels,factor):
 probes,_=make_readout_initializations(READOUT);probe=probes[factor];initial=state_digest(probe);target=torch.as_tensor(labels,dtype=torch.float32);counts=np.bincount(labels.astype(np.int64),minlength=2)
 if np.any(counts==0):raise RuntimeError('probe fit has empty class')
 weights=torch.as_tensor(np.where(labels==0,len(labels)/(2*counts[0]),len(labels)/(2*counts[1])),dtype=torch.float32);schedule=list(probe_indices(READOUT,factor));sh=hashlib.sha256()
 for indices in schedule:sh.update(np.asarray(indices,dtype='<i8').tobytes())
 probe.train();optimizer=_adamw(list(probe.parameters()),lr=1e-2,weight_decay=1e-3)
 for indices in schedule:
  idx=torch.as_tensor(indices,dtype=torch.long);optimizer.zero_grad(set_to_none=True);logits=probe(z[idx]).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,target[idx],weight=weights[idx]);_finite_loss(loss);loss.backward()
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):raise FloatingPointError('probe gradient invariant failed')
  optimizer.step()
 return probe.eval(),{'initial_state_digest':initial,'schedule_digest':sh.hexdigest(),'fit_class_counts':counts.tolist(),'sample_weights':[float(len(labels)/(2*counts[0])),float(len(labels)/(2*counts[1]))],'updates':len(schedule)}
def evaluate(probe,z,labels):
 with torch.no_grad():logits=probe(z).squeeze(-1)
 if not bool(torch.isfinite(logits).all()):raise FloatingPointError('nonfinite probe logits')
 pred=logits.ge(0).to(torch.uint8).cpu().numpy();labels=labels.astype(np.uint8);tp=int(np.sum((pred==1)&(labels==1)));tn=int(np.sum((pred==0)&(labels==0)));fp=int(np.sum((pred==1)&(labels==0)));fn=int(np.sum((pred==0)&(labels==1)))
 if tp+tn+fp+fn!=2048:raise RuntimeError('confusion total mismatch')
 return {'balanced_accuracy':balanced_accuracy(pred,labels),'predicted_positive_fraction':float(pred.mean()),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'predicts_both_classes':bool(np.any(pred==0) and np.any(pred==1))}
def arm_readouts(condition,probe_split,cal_split,transform):
 freeze_encoder(condition);fit=frozen_representations(condition,probe_split,transform);ev=frozen_representations(condition,cal_split,transform);out=[]
 for factor in range(3):
  probe,report=fit_balanced_probe(fit.z,probe_split.S[:,factor],factor);metrics=evaluate(probe,ev.z,cal_split.S[:,factor]);out.append({**metrics,'fit_report':report})
 return out

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);runtime=(sys.version.split()[0],torch.__version__,np.__version__)
 if runtime!=EXPECTED_RUNTIME:raise RuntimeError(f'runtime mismatch: {runtime}')
 actual_sources={path:sha_file(path) for path in EXPECTED_SOURCES}
 if actual_sources!=EXPECTED_SOURCES:raise RuntimeError('approved source digest mismatch')
 train=independent_train_nuisance(generate_factor_split(1102,TRAIN,8192),1102);train_digest=array_bundle_digest(train);transform=fit_stage0_time_transform(train)
 if train_digest!=EXPECTED_TRAIN or transform.state_bytes().hex()!=EXPECTED_TRANSFORM:raise RuntimeError('accepted TRAIN/transform mismatch')
 schedule_digests={seed:schedule_digest(seed) for seed in MODELS}
 if schedule_digests!=EXPECTED_SCHEDULE:raise RuntimeError('accepted training schedule mismatch')
 reference,threshold_digest=threshold_reference_report();conditions={}
 for seed in MODELS:
  trained=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);meta=trained.training
  required={'encoder_seed':seed,'predictor_seed':seed,'schedule_seed':seed,'successful_steps':2000,'optimizer_steps':2000,'ema_updates':2000,'identity_predictor_initialization':True,'directional_variance_weight':5.0,'directional_variance_floor':.01,'per_identity_directional_hinge':True,'recipe':'L1_AVG'}
  if any(meta.get(k)!=v for k,v in required.items()):raise RuntimeError(f'training contract mismatch seed {seed}')
  conditions[str(seed)]={'trained':trained,'untrained':untrained_condition(seed)}
 cells=[]
 for hseed,sp,xp in HELDOUT:
  probe_split=parameterized_split(hseed,PROBE_FIT,2048,sp,xp);cal_split=parameterized_split(hseed,CAL_OOD,2048,sp,xp);pd=split_digest(probe_split);cd=split_digest(cal_split)
  for seed in MODELS:
   arms={arm:arm_readouts(conditions[str(seed)][arm],probe_split,cal_split,transform) for arm in ('trained','untrained')}
   for factor in range(3):
    if arms['trained'][factor]['fit_report']['initial_state_digest']!=arms['untrained'][factor]['fit_report']['initial_state_digest'] or arms['trained'][factor]['fit_report']['schedule_digest']!=arms['untrained'][factor]['fit_report']['schedule_digest']:raise RuntimeError('probe initialization/schedule mismatch across arms')
   rows,position=trained_collapse_diagnostics(conditions[str(seed)]['trained'],cal_split,transform,reference);cells.append({'heldout_seed':hseed,'s_probabilities':sp,'signal_flip_probabilities':xp,'model_seed':seed,'probe_fit_digest':pd,'cal_ood_digest':cd,'arms':arms,'paired_gains':[arms['trained'][f]['balanced_accuracy']-arms['untrained'][f]['balanced_accuracy'] for f in range(3)],'constraints_all_pass':all(r['both_metrics_pass'] for r in rows),'constraint_rows':rows,'position_specificity_descriptive':position})
 if len(cells)!=9:raise RuntimeError('evaluation cell count mismatch')
 gains=[[],[],[]];bas=[[],[],[]];per_gen_gain={};per_gen_range={}
 for c in cells:
  for f in range(3):gains[f].append(c['paired_gains'][f]);bas[f].append(c['arms']['trained'][f]['balanced_accuracy'])
 for hseed,_,_ in HELDOUT:
  group=[c for c in cells if c['heldout_seed']==hseed];per_gen_gain[str(hseed)]=[float(np.mean([c['paired_gains'][f] for c in group])) for f in range(3)];per_gen_range[str(hseed)]=[max(c['arms']['trained'][f]['balanced_accuracy'] for c in group)-min(c['arms']['trained'][f]['balanced_accuracy'] for c in group) for f in range(3)]
 mean_gains=[float(np.mean(x)) for x in gains];checks={'aggregate_gain_thresholds':[mean_gains[f]>=GAIN_THRESHOLDS[f] for f in range(3)],'positive_gain_each_generator_factor':all(v>0 for row in per_gen_gain.values() for v in row),'all_trained_ba_at_least_0_80':min(x for row in bas for x in row)>=.8,'seed_ranges_at_most_0_0538':all(v<=RANGE_MAX for row in per_gen_range.values() for v in row),'constraints_pass_at_least_8_of_9':sum(c['constraints_all_pass'] for c in cells)>=8,'all_trained_and_untrained_probes_nondegenerate':all(c['arms'][arm][f]['predicts_both_classes'] and sum(c['arms'][arm][f][k] for k in ('tp','tn','fp','fn'))==2048 and all(n>0 for n in c['arms'][arm][f]['fit_report']['fit_class_counts']) for c in cells for arm in ('trained','untrained') for f in range(3))};passed=all(checks['aggregate_gain_thresholds']) and all(v for k,v in checks.items() if k!='aggregate_gain_thresholds');print(json.dumps({'schema':'BP011-J04C-L1-CORRUPTION-ONLY-BETA-V1','contract':{'axis':'signal corruption only at generating p(S)=0.5','models':'accepted generator-1102 L1_AVG versus seed-matched untrained','readout':'cell-specific inverse-frequency BCE rule, seed 2102','fallback':'none'},'runtime':runtime,'source_digests':actual_sources,'train_digest':train_digest,'schedule_digests':schedule_digests,'frozen_transform_state_hex':transform.state_bytes().hex(),'heldout':HELDOUT,'cells':cells,'mean_trained_minus_untrained':mean_gains,'per_generator_trained_minus_untrained':per_gen_gain,'per_generator_seed_ranges':per_gen_range,'checks':checks,'contract_pass':passed,'threshold_digest':threshold_digest,'claim_ceiling':'PINNED_TRAIN_UNSEEN_WITHIN_SUPPORT_CORRUPTION_ROBUSTNESS_AT_BALANCED_PREVALENCE_WITH_PARAMETER_SPECIFIC_READOUTS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
