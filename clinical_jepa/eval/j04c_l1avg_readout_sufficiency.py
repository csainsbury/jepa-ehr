"""Prospective BP011 matched readout-sufficiency diagnostic."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import copy,hashlib,io,json,math
from pathlib import Path
from typing import Callable,Mapping,Sequence
import numpy as np
import torch
import torch.nn.functional as F
from clinical_jepa.eval import j04c_l1avg_access as one
from clinical_jepa.eval import j04c_l1avg_access_3m as three
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_stage1 as stage1
from clinical_jepa.eval import j04c_initialization_bridge as bridge

PrototypeInvariantError=full.PrototypeInvariantError
NAMESPACE="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-1P1M-K0"
CONTRACT_SHA256="3c3da48c07465ba63a2cb2ae089f60110ad099f176b7bd97e30784d845f79ed3"
TARGET_COMMIT="a8bd75c060a00faf1f8c06227871cd6b0844a77c"
PAIR_S=one.PAIR_S;PAIR_FLIP=one.PAIR_FLIP
IMPLEMENTATION_PATHS=("clinical_jepa/eval/j04c_l1avg_readout_sufficiency.py","scripts/bp_clinjepa_011_j04c_l1avg_readout_sufficiency_beta.py","tests/test_bp_clinjepa_011_j04c_l1avg_readout_sufficiency.py")
SOURCE_DIGESTS={"clinical_jepa/eval/j04c_l1avg_access.py":"de47256dd993488710d5636116a53a451bba33ecc7da593e138b7210f0f54b9c","clinical_jepa/eval/j04c_l1avg_access_3m.py":"1406408aa900648f191db987eee0bd4b430737ef217aa1df5811c41514cd582c",**one.SOURCE_DIGESTS}
CLOSED_SOURCE_KEY="closed-l1avg-access-3m-k0/production-generated-seed-audit.json";CLOSED_AUDIT_SHA256="c23a1349e59dc315438745e2799d6b147e5e952eee26e4bcf619e0ad8d8c5d02";CLOSED_PREFIX="CLOSED_L1AVG_ACCESS_3M_K0__"
PURPOSE_COUNTS={"TRAIN_GENERATOR_SPLIT":1,"TRAIN_NUISANCE":1,"HELDOUT_PROBE":1,"HELDOUT_CAL":1,"E0_INIT":1,"L1_PREDICTOR_INIT":1,"TRAIN_SCHEDULE":32,"READOUT_INIT":1,"READOUT_SHORT_SCHEDULE":32,"READOUT_EXTENSION_SCHEDULE":219,"BOOTSTRAP":1}
ARRAY_FILES={"d_short":"d_short.f64le","d_long":"d_long.f64le","d_recovery":"d_recovery.f64le","M":"bootstrap_M.f64le"}
@dataclass(frozen=True)
class SeedManifest:schema:str;train_generator_seed:int;heldout_generator_seed:int;model_seed:int;readout_seed:int;bootstrap_root:int
@dataclass(frozen=True)
class Envelope:schema:str;manifest_sha256:str;historical_inventory_sha256:str;expected_generated_audit_sha256:str;production_path_count:int
BuildProvenance=one.BuildProvenance

def canonical(x):return one.canonical(x)
def sha256_hex(x):return one.sha256_hex(x)
def _r(p,*x):return {"purpose":p,"path":list(x)}
def manifest_from_dict(v):
 keys={"schema","train_generator_seed","heldout_generator_seed","model_seed","readout_seed","bootstrap_root"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-SEEDS-V1" or any(not one._int(v[k]) for k in keys-{"schema"}):raise PrototypeInvariantError("INPUT_SCHEMA")
 m=SeedManifest(**v);validate_manifest(m);return m
def validate_manifest(m):
 roots=(m.train_generator_seed,m.heldout_generator_seed,m.model_seed,m.readout_seed,m.bootstrap_root)
 if len(set(roots))!=5 or any(x<2**31 or x>=2**32 for x in roots):raise PrototypeInvariantError("SEED_COLLISION")
def envelope_from_dict(v):
 keys={"schema","manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256","production_path_count"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-SEED-APPROVAL-V1" or any(not one._digest(v[k]) for k in ("manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256")) or v.get("production_path_count")!=291:raise PrototypeInvariantError("INPUT_SCHEMA")
 return Envelope(**v)
def provenance_from_dict(v):
 keys={"schema","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-BUILD-PROVENANCE-V1" or v.get("target_commit")!=TARGET_COMMIT or not isinstance(v.get("implementation_commit"),str) or len(v["implementation_commit"])!=40 or any(c not in "0123456789abcdef" for c in v["implementation_commit"]) or v.get("clean_tree") is not True or v.get("source_digests")!=SOURCE_DIGESTS or not isinstance(v.get("implementation_digests"),dict) or set(v["implementation_digests"])!=set(IMPLEMENTATION_PATHS) or any(not one._digest(x) for x in v["implementation_digests"].values()) or any(not isinstance(v.get(k),str) or not v[k] for k in ("python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint")):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 return BuildProvenance(**v)
def audit(m):
 validate_manifest(m);a=[_r("TRAIN_GENERATOR_SPLIT",m.train_generator_seed,1),_r("TRAIN_NUISANCE",m.train_generator_seed,1,7101),_r("HELDOUT_PROBE",m.heldout_generator_seed,6),_r("HELDOUT_CAL",m.heldout_generator_seed,3),_r("E0_INIT",m.model_seed,1),_r("L1_PREDICTOR_INIT",m.model_seed,2),_r("READOUT_INIT",m.readout_seed,10),_r("BOOTSTRAP",m.bootstrap_root,7601)]
 a += [_r("TRAIN_SCHEDULE",m.model_seed,e,6101) for e in range(32)]+[_r("READOUT_SHORT_SCHEDULE",m.readout_seed,0,e,6201) for e in range(32)]+[_r("READOUT_EXTENSION_SCHEDULE",m.readout_seed,0,e,6202) for e in range(219)];a.sort(key=lambda x:(x["purpose"],x["path"]));return a
def validate_closed(inv):
 three.validate_closed(inv);src=inv.get("source_artifact_digests")
 if not isinstance(src,dict) or src.get(CLOSED_SOURCE_KEY)!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 rows=[{"purpose":x["purpose"][len(CLOSED_PREFIX):],"path":x["path"]} for x in inv["records"] if x["purpose"].startswith(CLOSED_PREFIX)];rows.sort(key=lambda x:(x["purpose"],x["path"]))
 if len(rows)!=140 or sha256_hex(canonical(rows))!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
def validate_seed(m,mraw,e,inv,iraw):
 validate_closed(inv);a=audit(m)
 if sha256_hex(mraw)!=e.manifest_sha256 or sha256_hex(iraw)!=e.historical_inventory_sha256 or sha256_hex(canonical(a))!=e.expected_generated_audit_sha256 or len(a)!=291 or Counter(x["purpose"] for x in a)!=PURPOSE_COUNTS:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 p={tuple(x["path"]) for x in a};r={x["path"][0] for x in a};hp={tuple(x["path"]) for x in inv["records"]};hr=set(inv["roots"])
 if len(p)!=291 or len(r)!=5 or p&hp or r&hr:raise PrototypeInvariantError("SEED_COLLISION")
 return {"manifest_sha256":e.manifest_sha256,"historical_inventory_sha256":e.historical_inventory_sha256,"generated_audit_sha256":e.expected_generated_audit_sha256,"production_path_count":291,"historical_path_count":len(hp),"path_intersection_count":0,"root_intersection_count":0}

def extension_indices(root,n=2048,batch_size=256):
 if (n,batch_size)!=(2048,256):raise PrototypeInvariantError("READOUT_INVALID")
 for epoch in range(219):
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,0,epoch,6202])));p=rng.permutation(n);count=8 if epoch<218 else 6
  for j in range(count):yield p[j*batch_size:(j+1)*batch_size]
def _object_hash(v):
 h=hashlib.sha256()
 def walk(x):
  if torch.is_tensor(x):
   a=np.ascontiguousarray(x.detach().cpu().numpy());h.update(b"T"+str(a.dtype).encode()+canonical(list(a.shape))+a.tobytes())
  elif isinstance(x,dict):
   h.update(b"D");[(h.update(str(k).encode()),walk(x[k])) for k in sorted(x,key=str)]
  elif isinstance(x,(list,tuple)):h.update(b"L");[walk(y) for y in x]
  else:h.update(b"J"+canonical(x))
 walk(v);return h.hexdigest()
def _trajectory(z,labels,root,short,extension):
 n=int(z.shape[0]);y=np.asarray(labels,dtype=np.uint8);counts=np.bincount(y.astype(np.int64),minlength=2)
 if z.shape!=(n,16) or y.shape!=(n,) or np.any(counts<=0) or len(short)!=250 or len(extension)!=1750:raise PrototypeInvariantError("READOUT_INVALID")
 probes,_=stage1.make_readout_initializations(root);probe=probes[0];initial=one.state_hash(probe);weights=np.where(y==0,n/(2*counts[0]),n/(2*counts[1])).astype(np.float32);target=torch.as_tensor(y,dtype=torch.float32);wt=torch.as_tensor(weights);opt=stage1._adamw(list(probe.parameters()),lr=1e-2,weight_decay=1e-3);losses=[];sh=hashlib.sha256();snapshots={}
 for step,idx0 in enumerate([*short,*extension],1):
  idx=np.asarray(idx0,dtype=np.int64)
  if idx.ndim!=1 or len(idx)==0 or np.any(idx<0) or np.any(idx>=n):raise PrototypeInvariantError("READOUT_INVALID")
  sh.update(np.ascontiguousarray(idx,dtype="<i8").tobytes());ti=torch.as_tensor(idx,dtype=torch.long);opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(probe(z[ti]).squeeze(-1),target[ti],weight=wt[ti]);stage1._finite_loss(loss);loss.backward()
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  opt.step()
  if any(not bool(torch.isfinite(p).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  losses.append(float(loss.detach()))
  if step in (250,2000):
   state=full._state_dict_bytes(probe);snapshots[step]={"probe":copy.deepcopy(probe),"probe_state_bytes":state,"probe_state_hash":hashlib.sha256(state).hexdigest(),"optimizer_state_hash":_object_hash(opt.state_dict()),"complete_schedule_hash":sh.hexdigest(),"attempted_steps":step,"successful_steps":step,"optimizer_steps":step,"first_100_mean_loss":float(np.mean(losses[:100])),"last_100_mean_loss":float(np.mean(losses[-100:]))}
 common={"initial_state_hash":initial,"target_hash":full.array_sha256("probe.target",y),"weight_hash":full.array_sha256("probe.weight",weights),"class_counts":counts.tolist(),"class_weights":[float(n/(2*counts[0])),float(n/(2*counts[1]))]};return snapshots,common
def train_probe_trajectory(z,labels,root):return _trajectory(z,labels,root,list(stage1.probe_indices(root,0)),list(extension_indices(root)))
def predict(probe,z):return one.predict(probe,z)
def assay(pred,y):return one.assay(pred,y)
def _assert_frozen_encoder_bytes(e0,l1,frozen):
 checks={"e0_immutable":full._state_dict_bytes(e0)==frozen["E0"],"l1_immutable":full._state_dict_bytes(l1.encoder)==frozen["L1"]}
 if not all(checks.values()):raise PrototypeInvariantError("READOUT_INVALID")
 return checks

def bootstrap(rows,root,replicates=10000,supplied_indices=None):
 x=np.ascontiguousarray(np.asarray(rows,dtype="<f8"))
 if x.shape!=(3,2048) or not np.isfinite(x).all():raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 obs=x.mean(axis=1)
 if supplied_indices is None:rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,7601])));idx=rng.integers(0,2048,size=(replicates,2048),dtype=np.int64)
 else:idx=np.asarray(supplied_indices,dtype=np.int64)
 if idx.shape!=(replicates,2048) or np.any(idx<0) or np.any(idx>=2048):raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 M=np.empty(replicates,dtype="<f8")
 for s in range(0,replicates,64):
  e=x[:,idx[s:s+64]].mean(axis=2).T-obs;M[s:s+len(e)]=np.maximum.reduce((-e[:,0],e[:,1],-e[:,1],e[:,2]))
 q=float(np.quantile(M,.95,method="linear"));bounds={"UCB95_short":float(obs[0]+q),"LCB95_long":float(obs[1]-q),"UCB95_long":float(obs[1]+q),"LCB95_recovery":float(obs[2]-q)};return M,q,obs,bounds
def terminal(l1_long_ba,constraints,b):
 read=constraints and l1_long_ba>=.8 and b["UCB95_short"]<=0 and b["LCB95_long"]>0 and b["LCB95_recovery"]>0
 rep=(not read) and constraints and b["UCB95_short"]<=0 and b["UCB95_long"]<=0
 return "READOUT_LIMITED" if read else "REPRESENTATION_LIMITED" if rep else "UNINFORMATIVE"
def array_artifacts(rows,M):
 arrays={"d_short":np.asarray(rows[0],dtype="<f8"),"d_long":np.asarray(rows[1],dtype="<f8"),"d_recovery":np.asarray(rows[2],dtype="<f8"),"M":np.asarray(M,dtype="<f8")};raw={k:np.ascontiguousarray(v).tobytes() for k,v in arrays.items()};meta={k:{"file":ARRAY_FILES[k],"dtype":"<f8","shape":list(arrays[k].shape),"sha256":sha256_hex(raw[k]),"byte_count":len(raw[k])} for k in arrays};return raw,meta
def validate_retained_arrays(result,raw):
 if set(raw)!=set(ARRAY_FILES):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 arrays={k:np.frombuffer(raw[k],dtype="<f8") for k in raw};meta=result["retained_arrays"]
 for k,a in arrays.items():
  if meta[k]!={"file":ARRAY_FILES[k],"dtype":"<f8","shape":[2048 if k!="M" else 10000],"sha256":sha256_hex(raw[k]),"byte_count":len(raw[k])}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not np.array_equal(arrays["d_recovery"],arrays["d_long"]-arrays["d_short"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 obs=np.array([arrays[k].mean() for k in ("d_short","d_long","d_recovery")]);q=float(np.quantile(arrays["M"],.95,method="linear"));b={"UCB95_short":float(obs[0]+q),"LCB95_long":float(obs[1]-q),"UCB95_long":float(obs[1]+q),"LCB95_recovery":float(obs[2]-q)}
 if not np.array_equal(obs,np.asarray(result["bootstrap"]["observed"])) or q!=result["bootstrap"]["critical_value"] or b!=result["bootstrap"]["bounds"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
def failure_artifact(p,c):return {"schema":"BP011-J04C-L1AVG-READOUT-SUFFICIENCY-INVALID-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"NO_SCIENTIFIC_INTERPRETATION","valid":False,"terminal_outcome":"INVALID","failure":{"phase":p,"code":c}}
def validate_failure_schema(v):
 if not isinstance(v,dict) or set(v)!={"schema","namespace","contract_sha256","claim_ceiling","valid","terminal_outcome","failure"} or v.get("schema")!="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-INVALID-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="NO_SCIENTIFIC_INTERPRETATION" or v.get("valid") is not False or v.get("terminal_outcome")!="INVALID" or not isinstance(v.get("failure"),dict) or set(v["failure"])!={"phase","code"} or v["failure"]["phase"] not in {"PROVENANCE","SEED_AUDIT","GENERATION","TRAINING","READOUT","BOOTSTRAP","SERIALIZATION"} or v["failure"]["code"] not in {"PROVENANCE_CONTENT","SEED_AUDIT_DIGEST","SEED_COLLISION","GENERATION_INVARIANT","TRAINING_INVARIANT","READOUT_INVALID","BOOTSTRAP_INVALID","SERIALIZATION_INVALID","INPUT_SCHEMA"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
def validate_success(v):
 keys={"schema","namespace","contract_sha256","claim_ceiling","provenance","seed_audit","fixed","checks","model","bootstrap","retained_arrays","valid","terminal_outcome"}
 if not isinstance(v,dict) or set(v)!=keys or v.get("schema")!="BP011-J04C-L1AVG-READOUT-SUFFICIENCY-RESULT-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="EXACT_ONE_PAIR_ONE_MODEL_LINEAR_READOUT_SUFFICIENCY_ONLY" or v.get("valid") is not True or v.get("terminal_outcome") not in {"READOUT_LIMITED","REPRESENTATION_LIMITED","UNINFORMATIVE"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if v.get("fixed")!={"train_n":8192,"probe_n":2048,"cal_n":2048,"model_count":1,"target_factor":0,"l1_updates":2000,"short_probe_updates":250,"long_probe_updates":2000,"bootstrap_replicates":10000,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP)}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not isinstance(v.get("checks"),dict) or set(v["checks"])!={"source_pins","seed_audit","matched_initialization","e0_immutable","l1_immutable","l1_exact","uninterrupted_trajectories","byte_exact_short_snapshots","identical_probe_protocol","cal_only","two_predicted_classes","simultaneous_bootstrap","retained_arrays","output_allowlist"} or not all(x is True for x in v["checks"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 provenance_from_dict({"schema":"BP011-J04C-L1AVG-READOUT-SUFFICIENCY-BUILD-PROVENANCE-V1",**{k:v["provenance"][k] for k in v["provenance"] if k!="build_provenance_sha256"}})
 if not one._digest(v["provenance"].get("build_provenance_sha256")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not isinstance(v.get("retained_arrays"),dict) or set(v["retained_arrays"])!=set(ARRAY_FILES):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for k,m in v["retained_arrays"].items():
  if m!={"file":ARRAY_FILES[k],"dtype":"<f8","shape":[2048 if k!="M" else 10000],"sha256":m.get("sha256"),"byte_count":16384 if k!="M" else 80000} or not one._digest(m.get("sha256")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 boot=v.get("bootstrap");
 if not isinstance(boot,dict) or set(boot)!={"replicates","family_size","quantile_method","critical_value","observed","bounds"} or boot["replicates"]!=10000 or boot["family_size"]!=4 or boot["quantile_method"]!="linear" or not one._finite(boot["critical_value"]) or not isinstance(boot["observed"],list) or len(boot["observed"])!=3 or any(not one._finite(x) for x in boot["observed"]) or not isinstance(boot["bounds"],dict) or set(boot["bounds"])!={"UCB95_short","LCB95_long","UCB95_long","LCB95_recovery"} or any(not one._finite(x) for x in boot["bounds"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 seed=v.get("seed_audit");sk={"approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256","production_path_count","historical_path_count","path_intersection_count","root_intersection_count"}
 if not isinstance(seed,dict) or set(seed)!=sk or any(not one._digest(seed[k]) for k in ("approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256")) or any(not one._int(seed[k]) or seed[k]<0 for k in ("production_path_count","historical_path_count","path_intersection_count","root_intersection_count")) or (seed["production_path_count"],seed["path_intersection_count"],seed["root_intersection_count"])!=(291,0,0):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 model=v.get("model")
 if not isinstance(model,dict) or set(model)!={"state_hashes","l1_training","probe_fits","cal","constraints"} or not isinstance(model["state_hashes"],dict) or set(model["state_hashes"])!={"e0","l1_encoder","l1_teacher","l1_predictor"} or any(not one._digest(x) for x in model["state_hashes"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 one._validate_l1_training(model["l1_training"]);c=model["constraints"]
 if not isinstance(c,dict) or set(c)!={"threshold_digest","all_pass","rows"} or not one._digest(c["threshold_digest"]) or not isinstance(c["all_pass"],bool) or not one._validate_constraint_rows(c["rows"]) or c["all_pass"] is not all(x["both_metrics_pass"] for x in c["rows"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 commons=[]
 for arm in ("E0","L1"):
  pf=model["probe_fits"].get(arm)
  if not isinstance(pf,dict) or set(pf)!={"common","short","long"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
  common=pf["common"]
  if not isinstance(common,dict) or set(common)!={"initial_state_hash","target_hash","weight_hash","class_counts","class_weights"} or any(not one._digest(common[k]) for k in ("initial_state_hash","target_hash","weight_hash")) or not isinstance(common["class_counts"],list) or len(common["class_counts"])!=2 or any(not one._int(x) or x<=0 for x in common["class_counts"]) or sum(common["class_counts"])!=2048 or not isinstance(common["class_weights"],list) or len(common["class_weights"])!=2 or any(not one._finite(x) or x<=0 for x in common["class_weights"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
  commons.append(common)
  for name,steps in (("short",250),("long",2000)):
   s=pf[name]
   if not isinstance(s,dict) or set(s)!={"probe_state_hash","optimizer_state_hash","complete_schedule_hash","attempted_steps","successful_steps","optimizer_steps","first_100_mean_loss","last_100_mean_loss"} or any(not one._digest(s[k]) for k in ("probe_state_hash","optimizer_state_hash","complete_schedule_hash")) or tuple(s[k] for k in ("attempted_steps","successful_steps","optimizer_steps"))!=(steps,steps,steps) or not one._finite(s["first_100_mean_loss"]) or not one._finite(s["last_100_mean_loss"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if commons[0]!=commons[1] or any(model["probe_fits"]["E0"][name]["complete_schedule_hash"]!=model["probe_fits"]["L1"][name]["complete_schedule_hash"] for name in ("short","long")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not isinstance(model["cal"],dict) or set(model["cal"])!={"short","long"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 target=None
 for t in ("short","long"):
  if not isinstance(model["cal"][t],dict) or set(model["cal"][t])!={"E0","L1"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
  for arm in ("E0","L1"):
   one._validate_assay(model["cal"][t][arm]);current=model["cal"][t][arm]["target_hash"]
   if target is not None and current!=target:raise PrototypeInvariantError("SERIALIZATION_INVALID")
   target=current
 if terminal(model["cal"]["long"]["L1"]["balanced_accuracy"],c["all_pass"],boot["bounds"])!=v["terminal_outcome"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 full.validate_recursive_output(v);one.delta.prior._validate_digest_fields(v)

def run_beta(m,p,seed,*,build_provenance_sha256,approved_envelope_sha256,phase_callback:Callable[[str],None]|None=None):
 from clinical_jepa.eval.j04c_falsifier import TRAIN,PROBE_FIT,CAL_OOD,fit_stage0_time_transform,generate_factor_split
 from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
 from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split
 n=phase_callback or (lambda _:None);n("GENERATION");train=independent_train_nuisance(generate_factor_split(m.train_generator_seed,TRAIN,8192),m.train_generator_seed);probe=parameterized_split(m.heldout_generator_seed,PROBE_FIT,2048,PAIR_S,PAIR_FLIP);cal=parameterized_split(m.heldout_generator_seed,CAL_OOD,2048,PAIR_S,PAIR_FLIP);t=fit_stage0_time_transform(train)
 n("TRAINING");e0=stage1._fresh_encoder(m.model_seed).eval();w=stage1._fresh_encoder(m.model_seed).eval();before=full._state_dict_bytes(e0)
 if full._state_dict_bytes(w)!=before:raise PrototypeInvariantError("TRAINING_INVARIANT")
 l1=bridge.train_recipe_decoupled_seeds(train,t,"L1_AVG",m.model_seed,m.model_seed,schedule_seed=m.model_seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True)
 required={"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"ema_updates":2000,"encoder_seed":m.model_seed,"predictor_seed":m.model_seed,"schedule_seed":m.model_seed,"identity_predictor_initialization":True,"student_variance_weight":0.0,"student_variance_floor":0.05,"directional_variance_weight":5.0,"directional_variance_floor":0.01,"recipe":"L1_AVG","per_identity_directional_hinge":True,"l2_layers":None}
 if any(l1.training.get(k)!=x for k,x in required.items()) or full._state_dict_bytes(e0)!=before:raise PrototypeInvariantError("TRAINING_INVARIANT")
 e=stage1.TrainedCondition("L1_AVG",e0,None,None,{});stage1.freeze_encoder(e);stage1.freeze_encoder(l1);frozen={"E0":full._state_dict_bytes(e0),"L1":full._state_dict_bytes(l1.encoder)};immutable_assertions=[];n("READOUT");y=probe.S[:,0].astype(np.uint8);zp={"E0":one.pooled(e,probe,t),"L1":one.pooled(l1,probe,t)};immutable_assertions.append(_assert_frozen_encoder_bytes(e0,l1,frozen));tr0,c0=train_probe_trajectory(zp["E0"],y,m.readout_seed);tr1,c1=train_probe_trajectory(zp["L1"],y,m.readout_seed)
 if c0!=c1 or tr0[250]["probe_state_bytes"]==tr0[2000]["probe_state_bytes"] or tr1[250]["probe_state_bytes"]==tr1[2000]["probe_state_bytes"]:raise PrototypeInvariantError("READOUT_INVALID")
 ycal=cal.S[:,0].astype(np.uint8);zc={"E0":one.pooled(e,cal,t),"L1":one.pooled(l1,cal,t)};immutable_assertions.append(_assert_frozen_encoder_bytes(e0,l1,frozen));calout={};scores={}
 for label,tr in (("E0",tr0),("L1",tr1)):
  calout[label]={}
  for name,step in (("short",250),("long",2000)):
   a=assay(predict(tr[step]["probe"],zc[label]),ycal);scores[label,name]=a.pop("row_scores");calout[label][name]=a
 rows=np.vstack((scores["L1","short"]-scores["E0","short"],scores["L1","long"]-scores["E0","long"]));rows=np.vstack((rows,rows[1]-rows[0]));reference,td=stage1.threshold_reference_report();constraint_rows,_=stage1.trained_collapse_diagnostics(l1,cal,t,reference);immutable_assertions.append(_assert_frozen_encoder_bytes(e0,l1,frozen));immutable_checks={k:all(x[k] for x in immutable_assertions) for k in ("e0_immutable","l1_immutable")};cp=all(x["both_metrics_pass"] for x in constraint_rows)
 n("BOOTSTRAP");M,q,obs,bounds=bootstrap(rows,m.bootstrap_root);raw,meta=array_artifacts(rows,M);out=terminal(calout["L1"]["long"]["balanced_accuracy"],cp,bounds);n("SERIALIZATION")
 def report(tr):
  common={**c0};return {"common":common,"short":{k:v for k,v in tr[250].items() if k not in {"probe","probe_state_bytes"}},"long":{k:v for k,v in tr[2000].items() if k not in {"probe","probe_state_bytes"}}}
 result={"schema":"BP011-J04C-L1AVG-READOUT-SUFFICIENCY-RESULT-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"EXACT_ONE_PAIR_ONE_MODEL_LINEAR_READOUT_SUFFICIENCY_ONLY","provenance":{"build_provenance_sha256":build_provenance_sha256,"target_commit":p.target_commit,"implementation_commit":p.implementation_commit,"clean_tree":p.clean_tree,"source_digests":p.source_digests,"implementation_digests":p.implementation_digests,"python_version":p.python_version,"numpy_version":p.numpy_version,"torch_version":p.torch_version,"platform_machine":p.platform_machine,"platform_system":p.platform_system,"blas_fingerprint":p.blas_fingerprint},"seed_audit":{"approved_envelope_sha256":approved_envelope_sha256,**dict(seed)},"fixed":{"train_n":8192,"probe_n":2048,"cal_n":2048,"model_count":1,"target_factor":0,"l1_updates":2000,"short_probe_updates":250,"long_probe_updates":2000,"bootstrap_replicates":10000,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP)},"checks":{**{k:True for k in ("source_pins","seed_audit","matched_initialization","l1_exact","uninterrupted_trajectories","byte_exact_short_snapshots","identical_probe_protocol","cal_only","two_predicted_classes","simultaneous_bootstrap","retained_arrays","output_allowlist")},**immutable_checks},"model":{"state_hashes":{"e0":one.state_hash(e0),"l1_encoder":one.state_hash(l1.encoder),"l1_teacher":one.state_hash(l1.teacher),"l1_predictor":one.state_hash(l1.predictor)},"l1_training":{k:v for k,v in l1.training.items() if k not in {"encoder_seed","predictor_seed","schedule_seed"}},"probe_fits":{"E0":report(tr0),"L1":report(tr1)},"cal":{"short":{"E0":calout["E0"]["short"],"L1":calout["L1"]["short"]},"long":{"E0":calout["E0"]["long"],"L1":calout["L1"]["long"]}},"constraints":{"threshold_digest":td,"all_pass":cp,"rows":constraint_rows}},"bootstrap":{"replicates":10000,"family_size":4,"quantile_method":"linear","critical_value":q,"observed":obs.tolist(),"bounds":bounds},"retained_arrays":meta,"valid":True,"terminal_outcome":out};validate_success(result);validate_retained_arrays(result,raw);return result,raw
