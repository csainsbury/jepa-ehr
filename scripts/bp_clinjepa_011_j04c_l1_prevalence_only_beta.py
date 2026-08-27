#!/usr/bin/env python3
"""Prevalence-only within-support beta for the frozen generator-1102 L1 recipe."""
from __future__ import annotations
import hashlib,json,sys,numpy as np,torch
from torch.nn import functional as F
from clinical_jepa.arms.v0f.own_latent import EMATeacher,SharedLatentPredictor
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor,train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import TrainedCondition,_adamw,_finite_loss,_fresh_encoder,balanced_accuracy,freeze_encoder,frozen_representations,make_readout_initializations,pretraining_indices,probe_indices,threshold_reference_report,trained_collapse_diagnostics
from bp_clinjepa_011_j04c_l1_generator_family import GAIN_THRESHOLDS,RANGE_MAX,READOUT,parameterized_split
MODELS=(2101,2102,2103);HELDOUT=((8200,(.35,.55,.65),(.15,.15,.15)),(8201,(.65,.45,.35),(.15,.15,.15)),(8202,(.55,.65,.45),(.15,.15,.15)));Q_GLOBAL=.8887;Q_LOCAL=.8017
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
def finite_tree(value):
 if isinstance(value,(float,np.floating)):return bool(np.isfinite(value))
 if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
 if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
 return True
def seed_path_audit():
 row={(seed,split) for seed,_,_ in HELDOUT for split in (PROBE_FIT,CAL_OOD)};other={(1102,TRAIN),(1102,TRAIN,7101)}
 other|={(seed,epoch,6101) for seed in MODELS for epoch in range(32)};other|={(READOUT,factor,epoch,6201) for factor in range(3) for epoch in range(32)};other|={(seed,split) for seed in (1101,1102,1103,4100,4101,4102,5100,5101,5102,7200,7201,7202) for split in (PROBE_FIT,CAL_OOD)}
 if row&other or len(row)!=6:raise RuntimeError('row seed-sequence path collision')
 return {'row_paths':[list(x) for x in sorted(row)],'prior_row_paths':[list(x) for x in sorted(x for x in other if len(x)==2 and x[1] in (PROBE_FIT,CAL_OOD))],'training_nuisance_readout_path_count':sum(1 for x in other if not (len(x)==2 and x[1] in (PROBE_FIT,CAL_OOD))),'exact_path_intersection':[]}
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
 expected_heldout=((8200,(.35,.55,.65),(.15,.15,.15)),(8201,(.65,.45,.35),(.15,.15,.15)),(8202,(.55,.65,.45),(.15,.15,.15)))
 if runtime!=EXPECTED_RUNTIME or MODELS!=(2101,2102,2103) or READOUT!=2102 or HELDOUT!=expected_heldout or (PROBE_FIT,CAL_OOD)!=(6,3):raise RuntimeError('runtime or exact prevalence contract mismatch')
 actual_sources={path:sha_file(path) for path in EXPECTED_SOURCES}
 if actual_sources!=EXPECTED_SOURCES:raise RuntimeError('approved source digest mismatch')
 seed_audit=seed_path_audit()
 train=independent_train_nuisance(generate_factor_split(1102,TRAIN,8192),1102);train_digest=array_bundle_digest(train);transform=fit_stage0_time_transform(train)
 if train_digest!=EXPECTED_TRAIN or transform.state_bytes().hex()!=EXPECTED_TRANSFORM:raise RuntimeError('accepted TRAIN/transform mismatch')
 schedule_digests={seed:schedule_digest(seed) for seed in MODELS}
 if schedule_digests!=EXPECTED_SCHEDULE:raise RuntimeError('accepted training schedule mismatch')
 reference,threshold_digest=threshold_reference_report();conditions={};model_state_digests={}
 for seed in MODELS:
  trained=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);meta=trained.training
  required={'encoder_seed':seed,'predictor_seed':seed,'schedule_seed':seed,'successful_steps':2000,'optimizer_steps':2000,'ema_updates':2000,'identity_predictor_initialization':True,'directional_variance_weight':5.0,'directional_variance_floor':.01,'per_identity_directional_hinge':True,'recipe':'L1_AVG'}
  if any(meta.get(k)!=v for k,v in required.items()):raise RuntimeError(f'training contract mismatch seed {seed}')
  conditions[str(seed)]={'trained':trained,'untrained':untrained_condition(seed)};model_state_digests[str(seed)]={arm:{'encoder':state_digest(condition.encoder),'teacher':state_digest(condition.teacher),'predictor':state_digest(condition.predictor)} for arm,condition in conditions[str(seed)].items()}
 cells=[]
 for hseed,sp,xp in HELDOUT:
  probe_split=parameterized_split(hseed,PROBE_FIT,2048,sp,xp);cal_split=parameterized_split(hseed,CAL_OOD,2048,sp,xp);pd=split_digest(probe_split);cd=split_digest(cal_split)
  if len(probe_split.S)!=2048 or len(cal_split.S)!=2048 or xp!=(.15,.15,.15):raise RuntimeError('split row count or corruption mismatch')
  probe_counts=[np.bincount(probe_split.S[:,f].astype(np.int64),minlength=2).tolist() for f in range(3)];cal_counts=[np.bincount(cal_split.S[:,f].astype(np.int64),minlength=2).tolist() for f in range(3)]
  if any(min(row)<=0 or sum(row)!=2048 for row in probe_counts+cal_counts):raise RuntimeError('empty or malformed true class')
  for seed in MODELS:
   arms={arm:arm_readouts(conditions[str(seed)][arm],probe_split,cal_split,transform) for arm in ('trained','untrained')}
   for factor in range(3):
    tr=arms['trained'][factor]['fit_report'];ur=arms['untrained'][factor]['fit_report']
    if tr['initial_state_digest']!=ur['initial_state_digest'] or tr['schedule_digest']!=ur['schedule_digest'] or tr['fit_class_counts']!=probe_counts[factor] or ur['fit_class_counts']!=probe_counts[factor] or tr['sample_weights']!=ur['sample_weights'] or any(not np.isfinite(w) or w<=0 for w in tr['sample_weights']):raise RuntimeError('probe initialization/schedule/count/weight mismatch across arms')
   trained_q=[(arms['trained'][f]['balanced_accuracy']-.5)/(.5-xp[f]) for f in range(3)]
   if any(not np.isfinite(q) for q in trained_q):raise FloatingPointError('nonfinite Bayes-excess retention')
   rows,position=trained_collapse_diagnostics(conditions[str(seed)]['trained'],cal_split,transform,reference);gains=[arms['trained'][f]['balanced_accuracy']-arms['untrained'][f]['balanced_accuracy'] for f in range(3)];constraint_pass=all(r['both_metrics_pass'] and r['variance_pass'] and r['rank_pass'] for r in rows)
   if len(rows)!=4 or not finite_tree({'arms':arms,'q':trained_q,'gains':gains,'rows':rows,'position':position}):raise FloatingPointError('malformed/nonfinite prevalence evaluation')
   cells.append({'heldout_seed':hseed,'s_probabilities':sp,'signal_flip_probabilities':xp,'model_seed':seed,'probe_fit_digest':pd,'cal_ood_digest':cd,'probe_fit_true_class_counts':probe_counts,'cal_ood_true_class_counts':cal_counts,'arms':arms,'trained_q_unclipped':trained_q,'paired_gains':gains,'constraints_all_pass':constraint_pass,'constraint_rows':rows,'position_specificity_descriptive':position})
 if len(cells)!=9 or len({(c['heldout_seed'],c['model_seed']) for c in cells})!=9 or any(len(c['arms'][arm])!=3 for c in cells for arm in ('trained','untrained')):raise RuntimeError('evaluation cell uniqueness/shape mismatch')
 gains=[[],[],[]];q_values=[[],[],[]];bas=[[],[],[]];per_gen_gain={};per_gen_range={};per_gen_q={}
 for c in cells:
  for f in range(3):gains[f].append(c['paired_gains'][f]);q_values[f].append(c['trained_q_unclipped'][f]);bas[f].append(c['arms']['trained'][f]['balanced_accuracy'])
 for hseed,_,_ in HELDOUT:
  group=[c for c in cells if c['heldout_seed']==hseed];per_gen_gain[str(hseed)]=[float(np.mean([c['paired_gains'][f] for c in group])) for f in range(3)];per_gen_range[str(hseed)]=[max(c['arms']['trained'][f]['balanced_accuracy'] for c in group)-min(c['arms']['trained'][f]['balanced_accuracy'] for c in group) for f in range(3)];per_gen_q[str(hseed)]=[float(np.mean([c['trained_q_unclipped'][f] for c in group])) for f in range(3)]
 mean_gains=[float(np.mean(x)) for x in gains];mean_q=[float(np.mean(x)) for x in q_values];checks={'g1_factor_mean_q_at_least_0_8887':[q>=Q_GLOBAL for q in mean_q],'g2_generator_factor_mean_q_at_least_0_8017':all(q>=Q_LOCAL for row in per_gen_q.values() for q in row),'aggregate_gain_thresholds':[mean_gains[f]>=GAIN_THRESHOLDS[f] for f in range(3)],'positive_gain_each_generator_factor':all(v>0 for row in per_gen_gain.values() for v in row),'seed_ranges_at_most_0_0538':all(v<=RANGE_MAX for row in per_gen_range.values() for v in row),'constraints_pass_at_least_8_of_9':sum(c['constraints_all_pass'] for c in cells)>=8,'all_trained_and_untrained_probes_nondegenerate':all(c['arms'][arm][f]['predicts_both_classes'] and sum(c['arms'][arm][f][k] for k in ('tp','tn','fp','fn'))==2048 and all(n>0 for n in c['arms'][arm][f]['fit_report']['fit_class_counts']) for c in cells for arm in ('trained','untrained') for f in range(3))};passed=all(checks['g1_factor_mean_q_at_least_0_8887']) and all(checks['aggregate_gain_thresholds']) and all(v for k,v in checks.items() if k not in ('g1_factor_mean_q_at_least_0_8887','aggregate_gain_thresholds'));print(json.dumps({'schema':'BP011-J04C-L1-PREVALENCE-ONLY-BETA-V1','contract':{'axis':'generating factor prevalence only at uniform corruption p=0.15','models':'accepted generator-1102 L1_AVG versus seed-matched untrained','metric':'unclipped Q=(BA-0.5)/0.35','g1_factor_mean_q_minimum':Q_GLOBAL,'g2_generator_factor_mean_q_minimum':Q_LOCAL,'readout':'cell-specific inverse-frequency BCE rule, seed 2102','fallback':'none'},'runtime':runtime,'source_digests':actual_sources,'train_digest':train_digest,'schedule_digests':schedule_digests,'model_state_digests':model_state_digests,'row_seed_path_audit':seed_audit,'frozen_transform_state_hex':transform.state_bytes().hex(),'heldout':HELDOUT,'cells':cells,'mean_trained_q_unclipped':mean_q,'per_generator_trained_q_unclipped':per_gen_q,'mean_trained_minus_untrained':mean_gains,'per_generator_trained_minus_untrained':per_gen_gain,'per_generator_seed_ranges':per_gen_range,'checks':checks,'contract_pass':passed,'threshold_digest':threshold_digest,'claim_ceiling':'PINNED_TRAIN_UNSEEN_WITHIN_SUPPORT_PREVALENCE_ROBUSTNESS_AT_CORRUPTION_0_15_WITH_PARAMETER_SPECIFIC_CLASS_BALANCED_READOUTS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
