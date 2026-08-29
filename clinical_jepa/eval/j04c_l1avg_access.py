"""Matched E0-versus-L1_AVG representation-accessibility beta for BP011."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import copy,hashlib,json,math
from typing import Callable,Iterable,Mapping,Sequence
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_l1avg_delta_c0cal as delta
from clinical_jepa.eval import j04c_stage1 as stage1
from clinical_jepa.eval import j04c_initialization_bridge as bridge

PrototypeInvariantError=full.PrototypeInvariantError
NAMESPACE="BP011-J04C-L1AVG-ACCESS-1P1M-K0"
CONTRACT_SHA256="4223618efec329d1b8150974eb91da863458474bf2042b710d1358998d42c7fe"
TARGET_COMMIT="3a259746cd6bab13b54bd3e4888d893d5c288316"
PAIR_S=(0.425,0.525,0.575);PAIR_FLIP=(0.125,0.185,0.165)
IMPLEMENTATION_PATHS=("clinical_jepa/eval/j04c_l1avg_access.py","scripts/bp_clinjepa_011_j04c_l1avg_access_beta.py","tests/test_bp_clinjepa_011_j04c_l1avg_access.py")
SOURCE_DIGESTS={
"clinical_jepa/eval/j04c_initialization_bridge.py":"099237fc24382f1015df64d53a1017918d3a10d0bb741d6eed98522f4f2b23d9",
"clinical_jepa/eval/j04c_stage1.py":"167300f6a075b07a2cfcc53fe15c8b507120a9eeb0a553da37841a69b2511bb2",
"clinical_jepa/eval/j04c_falsifier.py":"206bf1d59d36a180168b6bb0954d68db50f46b3a0828c6c4c9bbc2e19c843a0a",
"clinical_jepa/eval/j04c_nuisance_bridge.py":"521d64d60cd91f041cf61c1bd94a2a45fb401772963d5ac4667781aa861eea2e",
"clinical_jepa/arms/v0f/own_latent.py":"f3cd838225f8099c79d604036961cc64170c98a87b925fd960550846b7c50dfb",
"clinical_jepa/targets/next_event_contract.py":"7903f587996a2fd82fde5b13316f853cd06b2c1c8c13eca642dedcad7f8c755d",
"scripts/bp_clinjepa_011_j04c_l1_generator_family.py":"f5c472768ecd2453f32455ac3c959b707bf8c9f025336e9ad5d92eae6fcb3950",
"clinical_jepa/eval/j04c_l1avg_delta_c0cal.py":"fc8f50da46d23537d1eafeee93dcd3ed3fe052fecd8d4f8b118d828da6794667"}
CLOSED_DELTA_SOURCE_KEY="closed-full-delta-c0cal/production-generated-seed-audit.json"
CLOSED_DELTA_AUDIT_SHA256="7fb46c498c3b441154d74cb91dd0ce02cec989a9120008496e265f34efa5f04d"
CLOSED_DELTA_PREFIX="CLOSED_FULL_DELTA_C0CAL_K0__"
PURPOSE_COUNTS={"TRAIN_GENERATOR_SPLIT":1,"TRAIN_NUISANCE":1,"HELDOUT_PROBE":1,"HELDOUT_CAL":1,"E0_INIT":1,"L1_PREDICTOR_INIT":1,"TRAIN_SCHEDULE":32,"READOUT_INIT":1,"READOUT_SCHEDULE":32,"BOOTSTRAP":1}

@dataclass(frozen=True)
class SeedManifest:
 schema:str;train_generator_seed:int;heldout_generator_seed:int;model_seed:int;readout_seed:int;bootstrap_root:int
@dataclass(frozen=True)
class ApprovedSeedEnvelope:
 schema:str;manifest_sha256:str;historical_inventory_sha256:str;expected_generated_audit_sha256:str;production_path_count:int
@dataclass(frozen=True)
class BuildProvenance:
 schema:str;target_commit:str;implementation_commit:str;clean_tree:bool;source_digests:dict[str,str];implementation_digests:dict[str,str];python_version:str;numpy_version:str;torch_version:str;platform_machine:str;platform_system:str;blas_fingerprint:str

def _digest(x:object)->bool:return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)
def _int(x:object)->bool:return isinstance(x,int) and not isinstance(x,bool)
def _finite(x:object)->bool:return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def canonical(x:object)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha256_hex(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def _exact(v:object,keys:set[str])->bool:return isinstance(v,dict) and set(v)==keys

def seed_manifest_from_dict(v:object)->SeedManifest:
 keys={"schema","train_generator_seed","heldout_generator_seed","model_seed","readout_seed","bootstrap_root"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-SEEDS-V1" or any(not _int(v[k]) for k in keys-{"schema"}):raise PrototypeInvariantError("INPUT_SCHEMA")
 m=SeedManifest(**v);validate_seed_manifest(m);return m

def validate_seed_manifest(m:SeedManifest)->None:
 roots=(m.train_generator_seed,m.heldout_generator_seed,m.model_seed,m.readout_seed,m.bootstrap_root)
 if m.schema!="BP011-J04C-L1AVG-ACCESS-SEEDS-V1" or len(set(roots))!=5 or any(x<2**31 or x>=2**32 for x in roots):raise PrototypeInvariantError("SEED_COLLISION")

def approved_envelope_from_dict(v:object)->ApprovedSeedEnvelope:
 keys={"schema","manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256","production_path_count"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-SEED-APPROVAL-V1" or any(not _digest(v[k]) for k in ("manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256")) or not _int(v.get("production_path_count")) or v["production_path_count"]!=72:raise PrototypeInvariantError("INPUT_SCHEMA")
 return ApprovedSeedEnvelope(**v)

def build_provenance_from_dict(v:object)->BuildProvenance:
 keys={"schema","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-BUILD-PROVENANCE-V1" or v.get("target_commit")!=TARGET_COMMIT or not isinstance(v.get("implementation_commit"),str) or len(v["implementation_commit"])!=40 or any(c not in "0123456789abcdef" for c in v["implementation_commit"]) or v.get("clean_tree") is not True or v.get("source_digests")!=SOURCE_DIGESTS or not _exact(v.get("implementation_digests"),set(IMPLEMENTATION_PATHS)) or any(not _digest(x) for x in v["implementation_digests"].values()) or any(not isinstance(v.get(k),str) or not v[k] for k in ("python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint")):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 return BuildProvenance(**v)

def _r(p:str,*x:int)->dict[str,object]:return {"purpose":p,"path":list(x)}
def generated_seed_audit(m:SeedManifest)->list[dict[str,object]]:
 validate_seed_manifest(m);a=[_r("TRAIN_GENERATOR_SPLIT",m.train_generator_seed,1),_r("TRAIN_NUISANCE",m.train_generator_seed,1,7101),_r("HELDOUT_PROBE",m.heldout_generator_seed,6),_r("HELDOUT_CAL",m.heldout_generator_seed,3),_r("E0_INIT",m.model_seed,1),_r("L1_PREDICTOR_INIT",m.model_seed,2),_r("READOUT_INIT",m.readout_seed,10),_r("BOOTSTRAP",m.bootstrap_root,7601)]
 a += [_r("TRAIN_SCHEDULE",m.model_seed,e,6101) for e in range(32)]
 a += [_r("READOUT_SCHEDULE",m.readout_seed,0,e,6201) for e in range(32)]
 a.sort(key=lambda x:(x["purpose"],x["path"]));return a

def validate_closed_inventory(inv:Mapping[str,object])->None:
 delta.validate_closed_inventory(inv)
 src=inv.get("source_artifact_digests")
 if not isinstance(src,dict) or src.get(CLOSED_DELTA_SOURCE_KEY)!=CLOSED_DELTA_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 rows=[{"purpose":str(x["purpose"])[len(CLOSED_DELTA_PREFIX):],"path":x["path"]} for x in inv["records"] if isinstance(x,dict) and isinstance(x.get("purpose"),str) and x["purpose"].startswith(CLOSED_DELTA_PREFIX)]
 rows.sort(key=lambda x:(x["purpose"],x["path"]))
 if len(rows)!=39 or sha256_hex(canonical(rows))!=CLOSED_DELTA_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")

def validate_seed_audit(m:SeedManifest,mraw:bytes,e:ApprovedSeedEnvelope,inv:Mapping[str,object],iraw:bytes)->tuple[list[dict[str,object]],dict[str,int]]:
 validate_closed_inventory(inv);a=generated_seed_audit(m)
 if sha256_hex(mraw)!=e.manifest_sha256 or sha256_hex(iraw)!=e.historical_inventory_sha256 or sha256_hex(canonical(a))!=e.expected_generated_audit_sha256 or e.production_path_count!=72 or len(a)!=72 or Counter(x["purpose"] for x in a)!=PURPOSE_COUNTS:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 p={tuple(x["path"]) for x in a};r={x["path"][0] for x in a};hp={tuple(x["path"]) for x in inv["records"]};hr=set(inv["roots"])
 if len(p)!=72 or len(r)!=5 or p&hp or r&hr:raise PrototypeInvariantError("SEED_COLLISION")
 return a,{"manifest_sha256":e.manifest_sha256,"historical_inventory_sha256":e.historical_inventory_sha256,"generated_audit_sha256":e.expected_generated_audit_sha256,"production_path_count":72,"historical_path_count":len(hp),"path_intersection_count":0,"root_intersection_count":0}

def state_hash(module:nn.Module)->str:return full.canonical_state_sha256(module)
def pooled(condition:stage1.TrainedCondition,split:object,transform:object)->torch.Tensor:return stage1.frozen_representations(condition,split,transform).z

def train_balanced_probe(z:torch.Tensor,labels:np.ndarray,readout_seed:int,*,schedule:Iterable[np.ndarray]|None=None)->tuple[nn.Linear,dict[str,object]]:
 if z.shape!=(2048,16) or labels.shape!=(2048,):raise PrototypeInvariantError("READOUT_INVALID")
 probes,_=stage1.make_readout_initializations(readout_seed);probe=probes[0];initial=state_hash(probe);y=np.asarray(labels,dtype=np.uint8);counts=np.bincount(y.astype(np.int64),minlength=2)
 if np.any(counts<=0):raise PrototypeInvariantError("READOUT_INVALID")
 weights=np.where(y==0,2048/(2*counts[0]),2048/(2*counts[1])).astype(np.float32);target=torch.as_tensor(y,dtype=torch.float32);wt=torch.as_tensor(weights);optimizer=stage1._adamw(list(probe.parameters()),lr=1e-2,weight_decay=1e-3);rows=list(stage1.probe_indices(readout_seed,0) if schedule is None else schedule)
 if len(rows)!=250:raise PrototypeInvariantError("READOUT_INVALID")
 losses=[];sh=hashlib.sha256()
 for idx0 in rows:
  idx=np.asarray(idx0,dtype=np.int64);sh.update(np.ascontiguousarray(idx,dtype="<i8").tobytes());ti=torch.as_tensor(idx,dtype=torch.long);optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(probe(z[ti]).squeeze(-1),target[ti],weight=wt[ti]);stage1._finite_loss(loss);loss.backward()
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  optimizer.step()
  if any(not bool(torch.isfinite(p).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  losses.append(float(loss.detach()))
 probe.eval();report={"initial_state_hash":initial,"final_state_hash":state_hash(probe),"schedule_hash":sh.hexdigest(),"target_hash":full.array_sha256("probe.target",y),"weight_hash":full.array_sha256("probe.weight",weights),"class_counts":counts.tolist(),"class_weights":[float(2048/(2*counts[0])),float(2048/(2*counts[1]))],"attempted_steps":250,"successful_steps":250,"optimizer_steps":250,"first_100_mean_loss":float(np.mean(losses[:100])),"last_100_mean_loss":float(np.mean(losses[-100:]))}
 return probe,report

def predict(probe:nn.Module,z:torch.Tensor)->np.ndarray:
 with torch.no_grad():logits=probe(z).squeeze(-1)
 if logits.shape!=(2048,) or not bool(torch.isfinite(logits).all()):raise PrototypeInvariantError("READOUT_INVALID")
 return logits.ge(0).to(torch.uint8).cpu().numpy()

def assay(pred:np.ndarray,y:np.ndarray)->dict[str,object]:
 p=np.asarray(pred,dtype=np.uint8);t=np.asarray(y,dtype=np.uint8);counts=np.bincount(t.astype(np.int64),minlength=2)
 if p.shape!=(2048,) or t.shape!=(2048,) or np.any(counts<=0) or len(np.unique(p))!=2:raise PrototypeInvariantError("READOUT_INVALID")
 tp=int(np.sum((p==1)&(t==1)));tn=int(np.sum((p==0)&(t==0)));fp=int(np.sum((p==1)&(t==0)));fn=int(np.sum((p==0)&(t==1)));ba=.5*(tp/(tp+fn)+tn/(tn+fp))
 score=2048*.5*(p==t)/counts[t]
 return {"balanced_accuracy":float(ba),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"prediction_hash":full.array_sha256("cal.prediction",p),"target_hash":full.array_sha256("cal.target",t),"row_score_hash":full.array_sha256("cal.ba_score",np.asarray(score,dtype="<f8")),"row_scores":np.asarray(score,dtype="<f8")}

def bootstrap_access(rows:np.ndarray,root:int,*,replicates:int=10000,supplied_indices:np.ndarray|None=None)->tuple[float,dict[str,object]]:
 x=np.ascontiguousarray(np.asarray(rows,dtype="<f8"));n=x.size
 if x.shape!=(n,) or n==0 or not np.isfinite(x).all():raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 obs=float(x.mean())
 if supplied_indices is None:
  if n!=2048 or replicates!=10000:raise PrototypeInvariantError("BOOTSTRAP_INVALID")
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,7601])));idx=rng.integers(0,n,size=(replicates,n),dtype=np.int64,endpoint=False)
 else:
  idx=np.asarray(supplied_indices,dtype=np.int64)
  if idx.shape!=(replicates,n) or np.any(idx<0) or np.any(idx>=n):raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 centered=np.empty(replicates)
 for s in range(0,replicates,64):centered[s:s+len(idx[s:s+64])]=x[idx[s:s+64]].mean(axis=1)-obs
 crit=float(np.quantile(centered,.95,method="linear"));return crit,{"name":"d_ACCESS","observed":obs,"lcb95":float(obs-crit)}

def terminal(l1_ba:float,constraints:bool,contrast:Mapping[str,object])->tuple[str,bool,dict[str,bool]]:
 gates={"l1_ba_at_least_0_80":bool(l1_ba>=.80),"access_gain_lcb_positive":bool(float(contrast["lcb95"])>0),"l1_constraints_all_pass":bool(constraints)};ok=all(gates.values());return ("SUPPORTED" if ok else "NOT_SUPPORTED"),ok,gates

def fixed()->dict[str,object]:return {"train_n":8192,"probe_n":2048,"cal_n":2048,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP),"target_factor":0,"model_count":1,"l1_updates":2000,"probe_updates":250,"probe_class_balanced":True,"probe_threshold":0.0,"useful_ba_floor":0.80,"bootstrap_replicates":10000,"claim":"MATCHED_REPRESENTATION_ACCESSIBILITY_NOT_INCREMENTAL_R0"}

def _validate_probe(v:object)->None:
 keys={"initial_state_hash","final_state_hash","schedule_hash","target_hash","weight_hash","class_counts","class_weights","attempted_steps","successful_steps","optimizer_steps","first_100_mean_loss","last_100_mean_loss"}
 if not _exact(v,keys) or any(not _digest(v[k]) for k in ("initial_state_hash","final_state_hash","schedule_hash","target_hash","weight_hash")) or v["class_counts"] is None or not isinstance(v["class_counts"],list) or len(v["class_counts"])!=2 or any(not _int(x) or x<=0 for x in v["class_counts"]) or sum(v["class_counts"])!=2048 or not isinstance(v["class_weights"],list) or len(v["class_weights"])!=2 or any(not _finite(x) or x<=0 for x in v["class_weights"]) or any(not _int(v[k]) for k in ("attempted_steps","successful_steps","optimizer_steps")) or tuple(v[k] for k in ("attempted_steps","successful_steps","optimizer_steps"))!=(250,250,250) or not _finite(v["first_100_mean_loss"]) or not _finite(v["last_100_mean_loss"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")

def _validate_assay(v:object)->None:
 keys={"balanced_accuracy","tp","tn","fp","fn","prediction_hash","target_hash","row_score_hash"}
 if not _exact(v,keys) or not _finite(v["balanced_accuracy"]) or not 0<=v["balanced_accuracy"]<=1 or any(not _int(v[k]) or v[k]<0 for k in ("tp","tn","fp","fn")) or sum(v[k] for k in ("tp","tn","fp","fn"))!=2048 or any(not _digest(v[k]) for k in ("prediction_hash","target_hash","row_score_hash")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 tp,tn,fp,fn=(v[k] for k in ("tp","tn","fp","fn"))
 if tp+fn<=0 or tn+fp<=0 or tp+fp<=0 or tn+fn<=0 or not math.isclose(float(v["balanced_accuracy"]),.5*(tp/(tp+fn)+tn/(tn+fp)),rel_tol=0,abs_tol=1e-15):raise PrototypeInvariantError("SERIALIZATION_INVALID")

def _validate_l1_training(v:object)->None:
 keys={"attempted_steps","successful_steps","optimizer_steps","ema_updates","losses","identity_predictor_initialization","student_variance_weight","student_variance_floor","directional_variance_weight","directional_variance_floor","recipe","per_identity_directional_hinge","l2_layers"}
 if not _exact(v,keys) or any(not _int(v[k]) for k in ("attempted_steps","successful_steps","optimizer_steps","ema_updates")) or tuple(v[k] for k in ("attempted_steps","successful_steps","optimizer_steps","ema_updates"))!=(2000,2000,2000,2000) or v["identity_predictor_initialization"] is not True or v["student_variance_weight"]!=0.0 or v["student_variance_floor"]!=0.05 or v["directional_variance_weight"]!=5.0 or v["directional_variance_floor"]!=0.01 or v["recipe"]!="L1_AVG" or v["per_identity_directional_hinge"] is not True or v["l2_layers"] is not None:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 losses=v["losses"]
 if not _exact(losses,{"first_100_mean_total","last_100_mean_total","components"}) or not _finite(losses["first_100_mean_total"]) or not _finite(losses["last_100_mean_total"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 names={"cosine","variance","v_pred","directional_variance_penalty","v_direction","v_direction_min"}
 if not _exact(losses["components"],names):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for item in losses["components"].values():
  if not _exact(item,{"first_100_mean","last_100_mean"}) or not _finite(item["first_100_mean"]) or not _finite(item["last_100_mean"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")

def _validate_constraint_rows(v:object)->bool:
 keys={"arm_name","identity_index","normalized_variance","variance_threshold","variance_pass","effective_rank","rank_threshold","rank_pass","both_metrics_pass","teacher_target_effective_rank"}
 if not isinstance(v,list) or len(v)!=4:return False
 for i,row in enumerate(v):
  if not _exact(row,keys) or row["arm_name"]!="L1_AVG" or not _int(row["identity_index"]) or row["identity_index"]!=i or any(not _finite(row[k]) for k in ("normalized_variance","variance_threshold","effective_rank","rank_threshold","teacher_target_effective_rank")) or any(not isinstance(row[k],bool) for k in ("variance_pass","rank_pass","both_metrics_pass")):return False
  vp=float(row["normalized_variance"])>float(row["variance_threshold"]);rp=float(row["effective_rank"])>float(row["rank_threshold"])
  if row["variance_pass"] is not vp or row["rank_pass"] is not rp or row["both_metrics_pass"] is not (vp and rp):return False
 return True

def validate_success_schema(v:object)->None:
 roots={"schema","namespace","contract_sha256","claim_ceiling","provenance","seed_audit","fixed","checks","model","bootstrap","valid","supported","scientific_gates","terminal_outcome"}
 if not _exact(v,roots) or v["schema"]!="BP011-J04C-L1AVG-ACCESS-RESULT-V1" or v["namespace"]!=NAMESPACE or v["contract_sha256"]!=CONTRACT_SHA256 or v["claim_ceiling"]!="ONE_PAIR_ONE_MODEL_MATCHED_L1_AVG_ACCESSIBILITY_ONLY" or v["fixed"]!=fixed() or v["valid"] is not True or not isinstance(v["supported"],bool) or v["terminal_outcome"] not in {"SUPPORTED","NOT_SUPPORTED"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 prov=v["provenance"];pk={"build_provenance_sha256","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not _exact(prov,pk) or not _digest(prov["build_provenance_sha256"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 build_provenance_from_dict({"schema":"BP011-J04C-L1AVG-ACCESS-BUILD-PROVENANCE-V1",**{k:prov[k] for k in pk-{"build_provenance_sha256"}}})
 seed=v["seed_audit"];sk={"approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256","production_path_count","historical_path_count","path_intersection_count","root_intersection_count"}
 if not _exact(seed,sk) or any(not _digest(seed[k]) for k in ("approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256")) or any(not _int(seed[k]) or seed[k]<0 for k in ("production_path_count","historical_path_count","path_intersection_count","root_intersection_count")) or (seed["production_path_count"],seed["path_intersection_count"],seed["root_intersection_count"])!=(72,0,0):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 checks=v["checks"];ck={"source_pins","seed_audit","matched_initialization","e0_immutable","train_only_representation_learning","l1_training_exact","identical_probe_protocol","probe_only_fit","cal_only_evaluation","paired_bootstrap","output_allowlist","pair_exact"}
 if not _exact(checks,ck) or not all(x is True for x in checks.values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 model=v["model"]
 if not _exact(model,{"state_hashes","l1_training","probe_fits","cal","constraints"}) or not _exact(model["state_hashes"],{"e0","l1_encoder","l1_teacher","l1_predictor"}) or any(not _digest(x) for x in model["state_hashes"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 tr=model["l1_training"];_validate_l1_training(tr)
 if not _exact(model["probe_fits"],{"E0","L1"}):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for x in model["probe_fits"].values():_validate_probe(x)
 p0,p1=model["probe_fits"]["E0"],model["probe_fits"]["L1"]
 for key in ("initial_state_hash","schedule_hash","target_hash","weight_hash","class_counts","class_weights"):
  if p0[key]!=p1[key]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not _exact(model["cal"],{"E0","L1"}):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for x in model["cal"].values():_validate_assay(x)
 a0,a1=model["cal"]["E0"],model["cal"]["L1"]
 if a0["target_hash"]!=a1["target_hash"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 cons=model["constraints"]
 if not _exact(cons,{"threshold_digest","all_pass","rows"}) or not _digest(cons["threshold_digest"]) or not isinstance(cons["all_pass"],bool) or not _validate_constraint_rows(cons["rows"]) or cons["all_pass"] is not all(row["both_metrics_pass"] for row in cons["rows"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 boot=v["bootstrap"]
 if not _exact(boot,{"replicates","family_size","quantile_method","critical_value","contrast"}) or not _int(boot["replicates"]) or boot["replicates"]!=10000 or not _int(boot["family_size"]) or boot["family_size"]!=1 or boot["quantile_method"]!="linear" or not _finite(boot["critical_value"]) or not _exact(boot["contrast"],{"name","observed","lcb95"}) or boot["contrast"]["name"]!="d_ACCESS" or not _finite(boot["contrast"]["observed"]) or not _finite(boot["contrast"]["lcb95"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not math.isclose(float(boot["contrast"]["observed"])-float(boot["critical_value"]),float(boot["contrast"]["lcb95"]),rel_tol=0,abs_tol=1e-15) or not math.isclose(float(a1["balanced_accuracy"])-float(a0["balanced_accuracy"]),float(boot["contrast"]["observed"]),rel_tol=0,abs_tol=1e-15):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 gates=v["scientific_gates"]
 if not _exact(gates,{"l1_ba_at_least_0_80","access_gain_lcb_positive","l1_constraints_all_pass"}) or any(not isinstance(x,bool) for x in gates.values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 o,s,g=terminal(model["cal"]["L1"]["balanced_accuracy"],cons["all_pass"],boot["contrast"])
 if v["terminal_outcome"]!=o or v["supported"] is not s or gates!=g:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 full.validate_recursive_output(v);delta.prior._validate_digest_fields(v)

def failure_artifact(phase:str,code:str)->dict[str,object]:
 return {"schema":"BP011-J04C-L1AVG-ACCESS-INVALID-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"NO_SCIENTIFIC_INTERPRETATION","valid":False,"supported":False,"terminal_outcome":"INVALID","failure":{"phase":phase,"code":code}}
def validate_failure_schema(v:object)->None:
 if not _exact(v,{"schema","namespace","contract_sha256","claim_ceiling","valid","supported","terminal_outcome","failure"}) or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-INVALID-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="NO_SCIENTIFIC_INTERPRETATION" or v.get("valid") is not False or v.get("supported") is not False or v.get("terminal_outcome")!="INVALID" or not _exact(v.get("failure"),{"phase","code"}) or v["failure"]["phase"] not in {"PROVENANCE","SEED_AUDIT","GENERATION","TRAINING","READOUT","BOOTSTRAP","SERIALIZATION"} or not isinstance(v["failure"]["code"],str) or not v["failure"]["code"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 full.validate_recursive_output(v)

def run_beta(m:SeedManifest,p:BuildProvenance,seed:Mapping[str,object],*,build_provenance_sha256:str,approved_envelope_sha256:str,phase_callback:Callable[[str],None]|None=None)->dict[str,object]:
 from clinical_jepa.eval.j04c_falsifier import TRAIN,PROBE_FIT,CAL_OOD,fit_stage0_time_transform,generate_factor_split
 from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
 from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split
 notify=phase_callback or (lambda _:None);notify("GENERATION");train=independent_train_nuisance(generate_factor_split(m.train_generator_seed,TRAIN,8192),m.train_generator_seed);probe=parameterized_split(m.heldout_generator_seed,PROBE_FIT,2048,PAIR_S,PAIR_FLIP);cal=parameterized_split(m.heldout_generator_seed,CAL_OOD,2048,PAIR_S,PAIR_FLIP);transform=fit_stage0_time_transform(train)
 notify("TRAINING");e0=stage1._fresh_encoder(m.model_seed).eval();witness=stage1._fresh_encoder(m.model_seed).eval();e0_before=full._state_dict_bytes(e0)
 if full._state_dict_bytes(witness)!=e0_before:raise PrototypeInvariantError("TRAINING_INVARIANT")
 l1=bridge.train_recipe_decoupled_seeds(train,transform,"L1_AVG",m.model_seed,m.model_seed,schedule_seed=m.model_seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True)
 required={"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"ema_updates":2000,"encoder_seed":m.model_seed,"predictor_seed":m.model_seed,"schedule_seed":m.model_seed,"identity_predictor_initialization":True,"directional_variance_weight":5.0,"directional_variance_floor":.01,"per_identity_directional_hinge":True,"recipe":"L1_AVG"}
 if any(l1.training.get(k)!=x for k,x in required.items()) or full._state_dict_bytes(e0)!=e0_before:raise PrototypeInvariantError("TRAINING_INVARIANT")
 e0c=stage1.TrainedCondition("L1_AVG",e0,None,None,{"untrained_seed":m.model_seed});stage1.freeze_encoder(e0c);stage1.freeze_encoder(l1)
 notify("READOUT");z0p=pooled(e0c,probe,transform);z1p=pooled(l1,probe,transform);yprobe=probe.S[:,0].astype(np.uint8);schedule=list(stage1.probe_indices(m.readout_seed,0));p0,r0=train_balanced_probe(z0p,yprobe,m.readout_seed,schedule=schedule);p1,r1=train_balanced_probe(z1p,yprobe,m.readout_seed,schedule=schedule)
 if r0["initial_state_hash"]!=r1["initial_state_hash"] or r0["schedule_hash"]!=r1["schedule_hash"] or r0["target_hash"]!=r1["target_hash"] or r0["weight_hash"]!=r1["weight_hash"] or full._state_dict_bytes(e0)!=e0_before:raise PrototypeInvariantError("READOUT_INVALID")
 z0c=pooled(e0c,cal,transform);z1c=pooled(l1,cal,transform);ycal=cal.S[:,0].astype(np.uint8);a0=assay(predict(p0,z0c),ycal);a1=assay(predict(p1,z1c),ycal)
 reference,td=stage1.threshold_reference_report();rows,_=stage1.trained_collapse_diagnostics(l1,cal,transform,reference);cp=bool(len(rows)==4 and all(x["both_metrics_pass"] for x in rows))
 notify("BOOTSTRAP");crit,contrast=bootstrap_access(a1.pop("row_scores")-a0.pop("row_scores"),m.bootstrap_root);out,supported,gates=terminal(a1["balanced_accuracy"],cp,contrast)
 notify("SERIALIZATION");result={"schema":"BP011-J04C-L1AVG-ACCESS-RESULT-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"ONE_PAIR_ONE_MODEL_MATCHED_L1_AVG_ACCESSIBILITY_ONLY","provenance":{"build_provenance_sha256":build_provenance_sha256,"target_commit":p.target_commit,"implementation_commit":p.implementation_commit,"clean_tree":p.clean_tree,"source_digests":p.source_digests,"implementation_digests":p.implementation_digests,"python_version":p.python_version,"numpy_version":p.numpy_version,"torch_version":p.torch_version,"platform_machine":p.platform_machine,"platform_system":p.platform_system,"blas_fingerprint":p.blas_fingerprint},"seed_audit":{"approved_envelope_sha256":approved_envelope_sha256,**dict(seed)},"fixed":fixed(),"checks":{k:True for k in ("source_pins","seed_audit","matched_initialization","e0_immutable","train_only_representation_learning","l1_training_exact","identical_probe_protocol","probe_only_fit","cal_only_evaluation","paired_bootstrap","output_allowlist","pair_exact")},"model":{"state_hashes":{"e0":state_hash(e0),"l1_encoder":state_hash(l1.encoder),"l1_teacher":state_hash(l1.teacher),"l1_predictor":state_hash(l1.predictor)},"l1_training":{k:v for k,v in l1.training.items() if k not in {"encoder_seed","predictor_seed","schedule_seed"}},"probe_fits":{"E0":r0,"L1":r1},"cal":{"E0":a0,"L1":a1},"constraints":{"threshold_digest":td,"all_pass":cp,"rows":rows}},"bootstrap":{"replicates":10000,"family_size":1,"quantile_method":"linear","critical_value":crit,"contrast":contrast},"valid":True,"supported":supported,"scientific_gates":gates,"terminal_outcome":out};validate_success_schema(result);return result
