"""Finite-three-model matched L1_AVG accessibility mean for BP011."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib,json,math
from typing import Callable,Mapping
import numpy as np
from clinical_jepa.eval import j04c_l1avg_access as one
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_stage1 as stage1
from clinical_jepa.eval import j04c_initialization_bridge as bridge
PrototypeInvariantError=full.PrototypeInvariantError
NAMESPACE="BP011-J04C-L1AVG-ACCESS-3M-K0";CONTRACT_SHA256="891834e6be1505f897440da2a6b7df37c24990f372c1def97cfb1a397ab41139";TARGET_COMMIT="b26acb4f67930311918b7e171c8c7efb465a5ee1"
PAIR_S=one.PAIR_S;PAIR_FLIP=one.PAIR_FLIP
IMPLEMENTATION_PATHS=("clinical_jepa/eval/j04c_l1avg_access_3m.py","scripts/bp_clinjepa_011_j04c_l1avg_access_3m_beta.py","tests/test_bp_clinjepa_011_j04c_l1avg_access_3m.py")
SOURCE_DIGESTS={"clinical_jepa/eval/j04c_l1avg_access.py":"de47256dd993488710d5636116a53a451bba33ecc7da593e138b7210f0f54b9c",**one.SOURCE_DIGESTS}
CLOSED_SOURCE_KEY="closed-l1avg-access-k0/production-generated-seed-audit.json";CLOSED_AUDIT_SHA256="abb49bb7df5dbd274e4cdb194b07b4efbc2510da853f4ca80ff2c6e27da45e75";CLOSED_PREFIX="CLOSED_L1AVG_ACCESS_K0__"
PURPOSE_COUNTS={"TRAIN_GENERATOR_SPLIT":1,"TRAIN_NUISANCE":1,"HELDOUT_PROBE":1,"HELDOUT_CAL":1,"E0_INIT":3,"L1_PREDICTOR_INIT":3,"TRAIN_SCHEDULE":96,"READOUT_INIT":1,"READOUT_SCHEDULE":32,"BOOTSTRAP":1}
@dataclass(frozen=True)
class SeedManifest:schema:str;train_generator_seed:int;heldout_generator_seed:int;model_seeds:tuple[int,int,int];readout_seed:int;bootstrap_root:int
@dataclass(frozen=True)
class Envelope:schema:str;manifest_sha256:str;historical_inventory_sha256:str;expected_generated_audit_sha256:str;production_path_count:int
BuildProvenance=one.BuildProvenance

def canonical(x):return one.canonical(x)
def sha256_hex(x):return one.sha256_hex(x)
def _r(p,*x):return {"purpose":p,"path":list(x)}
def manifest_from_dict(v):
 keys={"schema","train_generator_seed","heldout_generator_seed","model_seeds","readout_seed","bootstrap_root"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-3M-SEEDS-V1" or not isinstance(v.get("model_seeds"),list) or len(v["model_seeds"])!=3 or any(not one._int(v[k]) for k in ("train_generator_seed","heldout_generator_seed","readout_seed","bootstrap_root")) or any(not one._int(x) for x in v["model_seeds"]):raise PrototypeInvariantError("INPUT_SCHEMA")
 m=SeedManifest(v["schema"],v["train_generator_seed"],v["heldout_generator_seed"],tuple(v["model_seeds"]),v["readout_seed"],v["bootstrap_root"]);validate_manifest(m);return m
def validate_manifest(m):
 roots=(m.train_generator_seed,m.heldout_generator_seed,*m.model_seeds,m.readout_seed,m.bootstrap_root)
 if len(set(roots))!=7 or any(x<2**31 or x>=2**32 for x in roots):raise PrototypeInvariantError("SEED_COLLISION")
def envelope_from_dict(v):
 keys={"schema","manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256","production_path_count"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-3M-SEED-APPROVAL-V1" or any(not one._digest(v[k]) for k in ("manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256")) or not one._int(v.get("production_path_count")) or v["production_path_count"]!=140:raise PrototypeInvariantError("INPUT_SCHEMA")
 return Envelope(**v)
def provenance_from_dict(v):
 keys={"schema","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-3M-BUILD-PROVENANCE-V1" or v.get("target_commit")!=TARGET_COMMIT or not isinstance(v.get("implementation_commit"),str) or len(v["implementation_commit"])!=40 or any(c not in "0123456789abcdef" for c in v["implementation_commit"]) or v.get("clean_tree") is not True or v.get("source_digests")!=SOURCE_DIGESTS or not isinstance(v.get("implementation_digests"),dict) or set(v["implementation_digests"])!=set(IMPLEMENTATION_PATHS) or any(not one._digest(z) for z in v["implementation_digests"].values()) or any(not isinstance(v.get(k),str) or not v[k] for k in ("python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint")):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 return BuildProvenance(**v)

def audit(m):
 validate_manifest(m);a=[_r("TRAIN_GENERATOR_SPLIT",m.train_generator_seed,1),_r("TRAIN_NUISANCE",m.train_generator_seed,1,7101),_r("HELDOUT_PROBE",m.heldout_generator_seed,6),_r("HELDOUT_CAL",m.heldout_generator_seed,3),_r("READOUT_INIT",m.readout_seed,10),_r("BOOTSTRAP",m.bootstrap_root,7601)]+[_r("READOUT_SCHEDULE",m.readout_seed,0,e,6201) for e in range(32)]
 for root in m.model_seeds:a += [_r("E0_INIT",root,1),_r("L1_PREDICTOR_INIT",root,2)]+[_r("TRAIN_SCHEDULE",root,e,6101) for e in range(32)]
 a.sort(key=lambda x:(x["purpose"],x["path"]));return a
def validate_closed(inv):
 one.validate_closed_inventory(inv);src=inv.get("source_artifact_digests")
 if not isinstance(src,dict) or src.get(CLOSED_SOURCE_KEY)!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 rows=[{"purpose":x["purpose"][len(CLOSED_PREFIX):],"path":x["path"]} for x in inv["records"] if x["purpose"].startswith(CLOSED_PREFIX)];rows.sort(key=lambda x:(x["purpose"],x["path"]))
 if len(rows)!=72 or sha256_hex(canonical(rows))!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
def validate_seed(m,mraw,e,inv,iraw):
 validate_closed(inv);a=audit(m)
 if sha256_hex(mraw)!=e.manifest_sha256 or sha256_hex(iraw)!=e.historical_inventory_sha256 or sha256_hex(canonical(a))!=e.expected_generated_audit_sha256 or len(a)!=140 or Counter(x["purpose"] for x in a)!=PURPOSE_COUNTS:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 p={tuple(x["path"]) for x in a};r={x["path"][0] for x in a};hp={tuple(x["path"]) for x in inv["records"]};hr=set(inv["roots"])
 if len(p)!=140 or len(r)!=7 or p&hp or r&hr:raise PrototypeInvariantError("SEED_COLLISION")
 return {"manifest_sha256":e.manifest_sha256,"historical_inventory_sha256":e.historical_inventory_sha256,"generated_audit_sha256":e.expected_generated_audit_sha256,"production_path_count":140,"historical_path_count":len(hp),"path_intersection_count":0,"root_intersection_count":0}
def bootstrap(matrix,root,replicates=10000,supplied_indices=None):
 x=np.asarray(matrix,dtype="<f8")
 if x.shape!=(3,2048) or not np.isfinite(x).all():raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 obs=float(x.mean())
 if supplied_indices is None:
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,7601])));idx=rng.integers(0,2048,size=(replicates,2048),dtype=np.int64)
 else:idx=np.asarray(supplied_indices,dtype=np.int64)
 if idx.shape!=(replicates,2048) or np.any(idx<0) or np.any(idx>=2048):raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 centered=np.empty(replicates)
 for s in range(0,replicates,64):centered[s:s+len(idx[s:s+64])]=x[:,idx[s:s+64]].mean(axis=(0,2))-obs
 crit=float(np.quantile(centered,.95,method="linear"));return crit,{"name":"d_ACCESS_3M","observed":obs,"lcb95":obs-crit}
def terminal(mean_ba,constraints,c):
 g={"mean_l1_ba_at_least_0_80":mean_ba>=.8,"access_3m_lcb_positive":float(c["lcb95"])>0,"all_12_constraints_pass":constraints};ok=all(g.values());return ("SUPPORTED" if ok else "NOT_SUPPORTED"),ok,g
def failure_artifact(p,c):
 return {"schema":"BP011-J04C-L1AVG-ACCESS-3M-INVALID-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"NO_SCIENTIFIC_INTERPRETATION","valid":False,"supported":False,"terminal_outcome":"INVALID","failure":{"phase":p,"code":c}}
def validate_failure_schema(v):
 keys={"schema","namespace","contract_sha256","claim_ceiling","valid","supported","terminal_outcome","failure"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-3M-INVALID-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="NO_SCIENTIFIC_INTERPRETATION" or v.get("valid") is not False or v.get("supported") is not False or v.get("terminal_outcome")!="INVALID" or not isinstance(v.get("failure"),dict) or set(v["failure"])!={"phase","code"} or v["failure"]["phase"] not in {"PROVENANCE","SEED_AUDIT","GENERATION","TRAINING","READOUT","BOOTSTRAP","SERIALIZATION"} or v["failure"]["code"] not in {"PROVENANCE_CONTENT","SEED_AUDIT_DIGEST","SEED_COLLISION","GENERATION_INVARIANT","TRAINING_INVARIANT","READOUT_INVALID","BOOTSTRAP_INVALID","SERIALIZATION_INVALID","INPUT_SCHEMA"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
def validate_success(v):
 keys={"schema","namespace","contract_sha256","claim_ceiling","provenance","seed_audit","fixed","checks","models","bootstrap","valid","supported","scientific_gates","terminal_outcome"};fixed={"model_count":3,"train_n":8192,"probe_n":2048,"cal_n":2048,"target_factor":0,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP),"bootstrap_replicates":10000,"bootstrap_resamples_models":False};check_names={"source_pins","seed_audit","matched_initializations","e0_immutable","train_only","l1_exact","probe_protocol_identical","cal_only","shared_row_bootstrap","output_allowlist"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-ACCESS-3M-RESULT-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="EXACT_FINITE_THREE_MODEL_MEAN_ACCESSIBILITY_ONLY" or v.get("fixed")!=fixed or v.get("valid") is not True or not isinstance(v.get("supported"),bool) or not isinstance(v.get("checks"),dict) or set(v["checks"])!=check_names or not all(x is True for x in v["checks"].values()) or not isinstance(v.get("models"),list) or len(v["models"])!=3:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 prov=v["provenance"];pk={"build_provenance_sha256","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not isinstance(prov,dict) or set(prov)!=pk or not one._digest(prov["build_provenance_sha256"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 provenance_from_dict({"schema":"BP011-J04C-L1AVG-ACCESS-3M-BUILD-PROVENANCE-V1",**{k:prov[k] for k in pk-{"build_provenance_sha256"}}})
 seed=v["seed_audit"];sk={"approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256","production_path_count","historical_path_count","path_intersection_count","root_intersection_count"}
 if not isinstance(seed,dict) or set(seed)!=sk or any(not one._digest(seed[k]) for k in ("approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256")) or any(not one._int(seed[k]) for k in ("production_path_count","historical_path_count","path_intersection_count","root_intersection_count")) or (seed["production_path_count"],seed["path_intersection_count"],seed["root_intersection_count"])!=(140,0,0):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 protocol=None;cal_target=None;bas=[];gains=[];all_constraints=True
 for m in v["models"]:
  if not isinstance(m,dict) or set(m)!={"state_hashes","l1_training","probe_fits","cal","constraints"} or not isinstance(m["state_hashes"],dict) or set(m["state_hashes"])!={"e0","l1_encoder","l1_teacher","l1_predictor"} or any(not one._digest(x) for x in m["state_hashes"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
  one._validate_l1_training(m["l1_training"])
  if not isinstance(m["probe_fits"],dict) or set(m["probe_fits"])!={"E0","L1"} or not isinstance(m["cal"],dict) or set(m["cal"])!={"E0","L1"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
  for x in m["probe_fits"].values():one._validate_probe(x)
  sig=tuple((m["probe_fits"]["E0"][k],m["probe_fits"]["L1"][k]) for k in ("initial_state_hash","schedule_hash","target_hash","weight_hash","class_counts","class_weights"))
  if any(x!=y for x,y in sig) or (protocol is not None and sig!=protocol):raise PrototypeInvariantError("SERIALIZATION_INVALID")
  protocol=sig
  for x in m["cal"].values():one._validate_assay(x)
  target=m["cal"]["E0"]["target_hash"]
  if target!=m["cal"]["L1"]["target_hash"] or (cal_target is not None and target!=cal_target):raise PrototypeInvariantError("SERIALIZATION_INVALID")
  cal_target=target
  c=m["constraints"]
  if not isinstance(c,dict) or set(c)!={"threshold_digest","all_pass","rows"} or not one._digest(c["threshold_digest"]) or not isinstance(c["all_pass"],bool) or not one._validate_constraint_rows(c["rows"]) or c["all_pass"] is not all(r["both_metrics_pass"] for r in c["rows"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
  all_constraints=all_constraints and c["all_pass"];bas.append(m["cal"]["L1"]["balanced_accuracy"]);gains.append(m["cal"]["L1"]["balanced_accuracy"]-m["cal"]["E0"]["balanced_accuracy"])
 b=v["bootstrap"]
 if not isinstance(b,dict) or set(b)!={"replicates","family_size","quantile_method","critical_value","contrast"} or not one._int(b["replicates"]) or b["replicates"]!=10000 or not one._int(b["family_size"]) or b["family_size"]!=1 or b["quantile_method"]!="linear" or not one._finite(b["critical_value"]) or not isinstance(b["contrast"],dict) or set(b["contrast"])!={"name","observed","lcb95"} or b["contrast"]["name"]!="d_ACCESS_3M" or not one._finite(b["contrast"]["observed"]) or not one._finite(b["contrast"]["lcb95"]) or not math.isclose(b["contrast"]["observed"]-b["critical_value"],b["contrast"]["lcb95"],rel_tol=0,abs_tol=1e-15) or not math.isclose(b["contrast"]["observed"],float(np.mean(gains)),rel_tol=0,abs_tol=1e-15):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 o,s,g=terminal(float(np.mean(bas)),all_constraints,b["contrast"])
 if v.get("terminal_outcome")!=o or v["supported"] is not s or v.get("scientific_gates")!=g:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 full.validate_recursive_output(v);one.delta.prior._validate_digest_fields(v)

def run_beta(m,p,seed,*,build_provenance_sha256,approved_envelope_sha256,phase_callback:Callable[[str],None]|None=None):
 from clinical_jepa.eval.j04c_falsifier import TRAIN,PROBE_FIT,CAL_OOD,fit_stage0_time_transform,generate_factor_split
 from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
 from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split
 n=phase_callback or (lambda _:None);n("GENERATION");train=independent_train_nuisance(generate_factor_split(m.train_generator_seed,TRAIN,8192),m.train_generator_seed);probe=parameterized_split(m.heldout_generator_seed,PROBE_FIT,2048,PAIR_S,PAIR_FLIP);cal=parameterized_split(m.heldout_generator_seed,CAL_OOD,2048,PAIR_S,PAIR_FLIP);t=fit_stage0_time_transform(train);reference,td=stage1.threshold_reference_report();schedule=list(stage1.probe_indices(m.readout_seed,0));models=[];matrix=[];protocol_ref=None
 for root in m.model_seeds:
  n("TRAINING");e0=stage1._fresh_encoder(root).eval();w=stage1._fresh_encoder(root).eval();before=full._state_dict_bytes(e0)
  if full._state_dict_bytes(w)!=before:raise PrototypeInvariantError("TRAINING_INVARIANT")
  l1=bridge.train_recipe_decoupled_seeds(train,t,"L1_AVG",root,root,schedule_seed=root,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True)
  required={"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"ema_updates":2000,"encoder_seed":root,"predictor_seed":root,"schedule_seed":root,"identity_predictor_initialization":True,"student_variance_weight":0.0,"student_variance_floor":0.05,"directional_variance_weight":5.0,"directional_variance_floor":0.01,"recipe":"L1_AVG","per_identity_directional_hinge":True,"l2_layers":None}
  if any(l1.training.get(k)!=v for k,v in required.items()) or full._state_dict_bytes(e0)!=before:raise PrototypeInvariantError("TRAINING_INVARIANT")
  e=stage1.TrainedCondition("L1_AVG",e0,None,None,{});stage1.freeze_encoder(e);stage1.freeze_encoder(l1);n("READOUT");p0,r0=one.train_balanced_probe(one.pooled(e,probe,t),probe.S[:,0].astype(np.uint8),m.readout_seed,schedule=schedule);p1,r1=one.train_balanced_probe(one.pooled(l1,probe,t),probe.S[:,0].astype(np.uint8),m.readout_seed,schedule=schedule)
  signature=tuple((r0[k],r1[k]) for k in ("initial_state_hash","schedule_hash","target_hash","weight_hash","class_counts","class_weights"))
  if any(x!=y for x,y in signature) or (protocol_ref is not None and signature!=protocol_ref):raise PrototypeInvariantError("READOUT_INVALID")
  protocol_ref=signature
  a0=one.assay(one.predict(p0,one.pooled(e,cal,t)),cal.S[:,0].astype(np.uint8));a1=one.assay(one.predict(p1,one.pooled(l1,cal,t)),cal.S[:,0].astype(np.uint8));matrix.append(a1.pop("row_scores")-a0.pop("row_scores"));rows,_=stage1.trained_collapse_diagnostics(l1,cal,t,reference)
  if a0["target_hash"]!=a1["target_hash"] or full._state_dict_bytes(e0)!=before:raise PrototypeInvariantError("READOUT_INVALID")
  models.append({"state_hashes":{"e0":one.state_hash(e0),"l1_encoder":one.state_hash(l1.encoder),"l1_teacher":one.state_hash(l1.teacher),"l1_predictor":one.state_hash(l1.predictor)},"l1_training":{k:v for k,v in l1.training.items() if k not in {"encoder_seed","predictor_seed","schedule_seed"}},"probe_fits":{"E0":r0,"L1":r1},"cal":{"E0":a0,"L1":a1},"constraints":{"threshold_digest":td,"all_pass":all(x["both_metrics_pass"] for x in rows),"rows":rows}})
 n("BOOTSTRAP");crit,c=bootstrap(np.asarray(matrix),m.bootstrap_root);mean=float(np.mean([x["cal"]["L1"]["balanced_accuracy"] for x in models]));cp=all(x["constraints"]["all_pass"] for x in models);o,s,g=terminal(mean,cp,c);n("SERIALIZATION");v={"schema":"BP011-J04C-L1AVG-ACCESS-3M-RESULT-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"EXACT_FINITE_THREE_MODEL_MEAN_ACCESSIBILITY_ONLY","provenance":{"build_provenance_sha256":build_provenance_sha256,"target_commit":p.target_commit,"implementation_commit":p.implementation_commit,"clean_tree":p.clean_tree,"source_digests":p.source_digests,"implementation_digests":p.implementation_digests,"python_version":p.python_version,"numpy_version":p.numpy_version,"torch_version":p.torch_version,"platform_machine":p.platform_machine,"platform_system":p.platform_system,"blas_fingerprint":p.blas_fingerprint},"seed_audit":{"approved_envelope_sha256":approved_envelope_sha256,**dict(seed)},"fixed":{"model_count":3,"train_n":8192,"probe_n":2048,"cal_n":2048,"target_factor":0,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP),"bootstrap_replicates":10000,"bootstrap_resamples_models":False},"checks":{k:True for k in ("source_pins","seed_audit","matched_initializations","e0_immutable","train_only","l1_exact","probe_protocol_identical","cal_only","shared_row_bootstrap","output_allowlist")},"models":models,"bootstrap":{"replicates":10000,"family_size":1,"quantile_method":"linear","critical_value":crit,"contrast":c},"valid":True,"supported":s,"scientific_gates":g,"terminal_outcome":o};validate_success(v);return v
