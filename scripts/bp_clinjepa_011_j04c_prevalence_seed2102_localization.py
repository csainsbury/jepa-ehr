#!/usr/bin/env python3
"""Post-result fixed-ridge localization of prevalence composition instability."""
from __future__ import annotations
import hashlib,json,sys,numpy as np,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import balanced_accuracy,freeze_encoder,frozen_representations
from bp_clinjepa_011_j04c_l1_generator_family import parameterized_split
from bp_clinjepa_011_j04c_l1_prevalence_only_beta import EXPECTED_RUNTIME,EXPECTED_SOURCES,EXPECTED_TRAIN,EXPECTED_TRANSFORM,EXPECTED_SCHEDULE,HELDOUT,MODELS,array_bundle_digest,schedule_digest,state_digest,untrained_condition
LAMBDA=1e-3;RANGE_REF=.0538;Q_ADEQUACY=.8017
PREVALENCE_RUNNER_SHA='03130cdbf4463a6b4f8401aeb4d033689302e7d1cf1ec94b49d3e59a65c72af5'
EXPECTED_ROWS={8200:('862d74835f1711aa48ba673eaa506b27bb9db7f448f9f537c1816f10be282a20','e1d1a253869398dd5fb275f00657f9b838d6aaaeec21a885c4e40dcfcf493916'),8201:('a3afb802b4810057e477e54784378e237d60a80f465a16abc8b70662983e5941','c30268cfb12c670be4cf595ff4bc537af36cdad36bfbd37a15c04ae7e6d30b3e'),8202:('6db9681f4a4e8761fc8258bef5f6db6e73439258dc7a254b020eeabb9191a16e','50431669fcb5ed1b37eee43f49315cc4325115149153d059ea9f83677c170b7a')}
EXPECTED_ENCODERS={2101:{'trained':'e1b63bdb05606e7a577347e182e35e9b9ab0825412d22cfc2978e87c19e3db2b','untrained':'090c5d8f4e3aa0ef85751d214d2280b2c58aefb2a79f7d893ce564cc692e92ce'},2102:{'trained':'8c7f76447fe1a53ac3bad29bf99fa9e6b9be09289ac25a99483a903c748157b5','untrained':'a768cf64a159ab39cd2e03f446e3965248518d26e6d9a1bda9de6a53492e4dc5'},2103:{'trained':'03dabe2efa2ad96215498b0a32fdd6498366ec491e5c12361f1723817f7dc956','untrained':'787c90db10ef911cec626e793a9a11be8cde4c84ae8a2d2a708c2f7450ef3d3c'}}

def sha_file(path):return hashlib.sha256(open(path,'rb').read()).hexdigest()
def digest_arrays(*arrays):
 h=hashlib.sha256()
 for a in arrays:
  x=np.asarray(a);h.update(x.dtype.str.encode());h.update(np.asarray(x.shape,dtype='<i8').tobytes());h.update(x.tobytes())
 return h.hexdigest()
def fit_ridge(z,labels):
 x=z.detach().cpu().double();y=torch.as_tensor(labels,dtype=torch.float64);counts=np.bincount(labels.astype(np.int64),minlength=2)
 if min(counts)<=0 or len(labels)!=2048:raise RuntimeError('ridge fit class/count invariant failed')
 weights_np=np.where(labels==0,len(labels)/(2*counts[0]),len(labels)/(2*counts[1])).astype(np.float64);weights=torch.as_tensor(weights_np,dtype=torch.float64);xa=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],dim=1);matrix=xa.T@(weights[:,None]*xa)/len(labels);regularizer=torch.eye(17,dtype=torch.float64)*LAMBDA;regularizer[-1,-1]=0.;matrix=matrix+regularizer;rhs=xa.T@(weights*y)/len(labels);coef=torch.linalg.solve(matrix,rhs);residual=torch.linalg.vector_norm(matrix@coef-rhs)/torch.linalg.vector_norm(rhs).clamp_min(1e-30);condition=torch.linalg.cond(matrix,p=2)
 if not bool(torch.isfinite(coef).all()) or not bool(torch.isfinite(condition)) or not float(residual)<=1e-10:raise FloatingPointError('ridge solve invariant failed')
 return {'coef':coef,'coefficient_digest':digest_arrays(coef.numpy()),'weight_target_digest':digest_arrays(labels,weights_np),'fit_class_counts':counts.tolist(),'sample_weights':[float(len(labels)/(2*counts[0])),float(len(labels)/(2*counts[1]))],'weight_norm':float(torch.linalg.vector_norm(coef[:-1])),'intercept':float(coef[-1]),'condition_number_2':float(condition),'relative_solve_residual':float(residual)}
def evaluate_ridge(fit,z,labels):
 x=z.detach().cpu().double();xa=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],dim=1);prediction_value=xa@fit['coef'];pred=prediction_value.ge(.5).to(torch.uint8).numpy();labels=labels.astype(np.uint8);counts=np.bincount(labels.astype(np.int64),minlength=2);tp=int(np.sum((pred==1)&(labels==1)));tn=int(np.sum((pred==0)&(labels==0)));fp=int(np.sum((pred==1)&(labels==0)));fn=int(np.sum((pred==0)&(labels==1)))
 if min(counts)<=0 or tp+tn+fp+fn!=2048 or not bool(torch.isfinite(prediction_value).all()):raise RuntimeError('ridge evaluation invariant failed')
 return {'balanced_accuracy':balanced_accuracy(pred,labels),'q_unclipped':(balanced_accuracy(pred,labels)-.5)/.35,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'predicts_both_classes':bool(np.any(pred==0) and np.any(pred==1)),'predicted_positive_fraction':float(pred.mean()),'prediction_min':float(prediction_value.min()),'prediction_max':float(prediction_value.max()),'prediction_digest':digest_arrays(prediction_value.numpy()),'cal_true_class_counts':counts.tolist()}
def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);runtime=(sys.version.split()[0],torch.__version__,np.__version__)
 if runtime!=EXPECTED_RUNTIME or sha_file('scripts/bp_clinjepa_011_j04c_l1_prevalence_only_beta.py')!=PREVALENCE_RUNNER_SHA:raise RuntimeError('runtime/prevalence source mismatch')
 if {p:sha_file(p) for p in EXPECTED_SOURCES}!=EXPECTED_SOURCES:raise RuntimeError('approved source digest mismatch')
 train=independent_train_nuisance(generate_factor_split(1102,TRAIN,8192),1102);transform=fit_stage0_time_transform(train)
 if array_bundle_digest(train)!=EXPECTED_TRAIN or transform.state_bytes().hex()!=EXPECTED_TRANSFORM or {s:schedule_digest(s) for s in MODELS}!=EXPECTED_SCHEDULE:raise RuntimeError('accepted reconstruction authority mismatch')
 conditions={};actual_encoders={}
 for seed in MODELS:
  trained=train_recipe_decoupled_seeds(train,transform,'L1_AVG',seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True);conditions[str(seed)]={'trained':trained,'untrained':untrained_condition(seed)};actual_encoders[seed]={arm:state_digest(c.encoder) for arm,c in conditions[str(seed)].items()}
 if actual_encoders!=EXPECTED_ENCODERS:raise RuntimeError('encoder state digest mismatch from prevalence run')
 groups=[]
 for hseed,sp,xp in HELDOUT:
  probe=parameterized_split(hseed,PROBE_FIT,2048,sp,xp)
  if array_bundle_digest(probe)!=EXPECTED_ROWS[hseed][0]:raise RuntimeError('PROBE_FIT row digest mismatch')
  fits={};target_digests=[]
  for seed in MODELS:
   for arm in ('trained','untrained'):
    condition=conditions[str(seed)][arm];freeze_encoder(condition);reps=frozen_representations(condition,probe,transform);factor_fits=[]
    for factor in range(3):factor_fits.append(fit_ridge(reps.z,probe.S[:,factor]))
    fits[(seed,arm)]=factor_fits;target_digests.extend(f['weight_target_digest'] for f in factor_fits)
  for factor in range(3):
   if len({fits[(seed,arm)][factor]['weight_target_digest'] for seed in MODELS for arm in ('trained','untrained')})!=1:raise RuntimeError('target/weight digest differs across arms or seeds')
  cal=parameterized_split(hseed,CAL_OOD,2048,sp,xp)
  if array_bundle_digest(cal)!=EXPECTED_ROWS[hseed][1]:raise RuntimeError('CAL_OOD row digest mismatch')
  for seed in MODELS:
   for arm in ('trained','untrained'):
    reps=frozen_representations(conditions[str(seed)][arm],cal,transform);results=[]
    for factor in range(3):
     fit=fits[(seed,arm)][factor];metrics=evaluate_ridge(fit,reps.z,cal.S[:,factor]);results.append({**{k:v for k,v in fit.items() if k!='coef'},**metrics})
    groups.append({'heldout_seed':hseed,'model_seed':seed,'arm':arm,'factors':results,'fit_completed_before_cal_access':True})
 if len(groups)!=18 or len({(g['heldout_seed'],g['model_seed'],g['arm']) for g in groups})!=18 or any(len(g['factors'])!=3 for g in groups):raise RuntimeError('ridge group shape/uniqueness mismatch')
 trained={(g['heldout_seed'],g['model_seed']):g for g in groups if g['arm']=='trained'};untrained={(g['heldout_seed'],g['model_seed']):g for g in groups if g['arm']=='untrained'};decisions={};adequacy=True;all_ranges=True;all_unique_low=True;gains=[]
 for hseed,_,_ in HELDOUT:
  bas={seed:trained[(hseed,seed)]['factors'][0]['balanced_accuracy'] for seed in MODELS};qmean=float(np.mean([trained[(hseed,seed)]['factors'][0]['q_unclipped'] for seed in MODELS]));range_value=max(bas.values())-min(bas.values());rh=range_value<=RANGE_REF;uh=bas[2102]<min(bas[2101],bas[2103]);adequacy=adequacy and qmean>=Q_ADEQUACY;all_ranges=all_ranges and rh;all_unique_low=all_unique_low and uh;ordering=sorted(MODELS,key=lambda s:(bas[s],s));decisions[str(hseed)]={'composition_ba_by_seed':bas,'composition_q_mean':qmean,'adequacy_pass':qmean>=Q_ADEQUACY,'range':range_value,'R_h':rh,'U_h':uh,'seed_order_low_to_high':ordering};gains.extend(trained[(hseed,seed)]['factors'][0]['balanced_accuracy']-untrained[(hseed,seed)]['factors'][0]['balanced_accuracy'] for seed in MODELS)
 if not adequacy:classification='MIXED_UNRESOLVED_INADEQUATE_RIDGE_ASSAY'
 elif all_ranges and not all_unique_low:classification='ORIGINAL_READOUT_PROTOCOL_SPECIFICITY_SUPPORTED'
 elif not all_ranges and all_unique_low:classification='WEAKNESS_PERSISTS_UNDER_RIDGE'
 else:classification='MIXED_UNRESOLVED'
 print(json.dumps({'schema':'BP011-J04C-PREVALENCE-SEED2102-RIDGE-LOCALIZATION-V1','contract':{'status':'post-result localization; prevalence beta remains red','objective':'mean weighted SSE plus lambda 1e-3 weight penalty; intercept unpenalized','ridge_composition_adequacy_q_minimum':Q_ADEQUACY,'range_reference':RANGE_REF,'fallback':'none'},'runtime':runtime,'encoder_state_digests':actual_encoders,'row_digests':EXPECTED_ROWS,'groups':groups,'heldout_composition_decisions':decisions,'A_all_ranges_pass':all_ranges,'U_seed2102_unique_low_all_heldouts':all_unique_low,'ridge_composition_assay_adequate_all_heldouts':adequacy,'localization':classification,'ridge_composition_trained_minus_untrained':gains,'claim_ceiling':'POST_RESULT_COMPARISON_OF_ORIGINAL_AND_FIXED_RIDGE_READOUT_PROTOCOLS_ON_OPENED_PREVALENCE_ROWS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
