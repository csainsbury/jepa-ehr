"""Prospective BP011 matched 256-label sample-efficiency assay."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import contextlib,hashlib,io,json,math,platform,subprocess
from pathlib import Path
from typing import Callable,Mapping
import numpy as np
import torch
import torch.nn.functional as F
from clinical_jepa.eval import j04c_l1avg_readout_sufficiency as accepted
from clinical_jepa.eval import j04c_l1avg_access as access
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_stage1 as stage1
from clinical_jepa.eval import j04c_initialization_bridge as bridge

PrototypeInvariantError=full.PrototypeInvariantError
NAMESPACE="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-1P1M-K0"
CONTRACT_SHA256="2b770a05696d3d883baf25bac2dc7e61ece0b4ae16b66c581b5ad5f51e3900ba"
TARGET_COMMIT="8ed46be700cd9f9f89c4ba8b8b59b1a14a3806b1"
PAIR_S=access.PAIR_S;PAIR_FLIP=access.PAIR_FLIP
IMPLEMENTATION_PATHS=("clinical_jepa/eval/j04c_l1avg_sample_efficiency.py","scripts/bp_clinjepa_011_j04c_l1avg_sample_efficiency_beta.py","tests/test_bp_clinjepa_011_j04c_l1avg_sample_efficiency.py")
SOURCE_DIGESTS={"clinical_jepa/eval/j04c_l1avg_readout_sufficiency.py":"709b501bdf141102d9ae90724eb9158e628feba1a1987eccc806131e61574e41",**accepted.SOURCE_DIGESTS}
CLOSED_SOURCE_KEY="closed-readout-sufficiency-k0/production-generated-seed-audit.json"
CLOSED_AUDIT_SHA256="369fc196d8c5827d2dd25dc43804dfaa89737a2b9ef2e97c08a9dba501b89902"
CLOSED_PREFIX="CLOSED_READOUT_SUFFICIENCY_K0__"
PURPOSE_COUNTS={"TRAIN_GENERATOR_SPLIT":1,"TRAIN_NUISANCE":1,"HELDOUT_PROBE":1,"HELDOUT_CAL":1,"E0_INIT":1,"L1_PREDICTOR_INIT":1,"TRAIN_SCHEDULE":32,"READOUT_INIT":1,"READOUT_SCHEDULE":2000,"BOOTSTRAP":1}
ARRAY_FILES={"selected_indices":"selected_indices.i64le","d_gain":"d_gain.f64le","bootstrap_centered":"bootstrap_centered.f64le"}

@dataclass(frozen=True)
class SeedManifest:
 schema:str;train_generator_seed:int;heldout_generator_seed:int;model_seed:int;readout_seed:int;bootstrap_root:int
@dataclass(frozen=True)
class Envelope:
 schema:str;manifest_sha256:str;historical_inventory_sha256:str;expected_generated_audit_sha256:str;production_path_count:int
BuildProvenance=access.BuildProvenance

@dataclass(frozen=True)
class ExecutionEnvironmentVerification:
 implementation_commit:str;source_digests:Mapping[str,str];implementation_digests:Mapping[str,str];runtime_fingerprint:Mapping[str,str];source_pins:bool

def canonical(v):return access.canonical(v)
def sha256_hex(v):return access.sha256_hex(v)
def _command(*args):return subprocess.check_output(args,text=True).strip()
def _file_sha256(path):return sha256_hex(Path(path).read_bytes())
def runtime_fingerprint():
 blas=io.StringIO()
 with contextlib.redirect_stdout(blas):np.__config__.show()
 return {"python_version":platform.python_version(),"numpy_version":np.__version__,"torch_version":torch.__version__,"platform_machine":platform.machine(),"platform_system":platform.system(),"blas_fingerprint":"sha256:"+sha256_hex(blas.getvalue().encode())}
def verify_execution_environment(p):
 """Verify the live checkout and runtime against supplied provenance immediately before execution."""
 if _command("git","rev-parse","HEAD")!=p.implementation_commit or _command("git","status","--porcelain"):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 try:subprocess.check_call(["git","merge-base","--is-ancestor",p.target_commit,p.implementation_commit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 except subprocess.CalledProcessError as exc:raise PrototypeInvariantError("PROVENANCE_CONTENT") from exc
 for path,digest in {**p.source_digests,**p.implementation_digests}.items():
  if _file_sha256(path)!=digest:raise PrototypeInvariantError("PROVENANCE_CONTENT")
 runtime=runtime_fingerprint()
 if any(getattr(p,key)!=value for key,value in runtime.items()):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 return ExecutionEnvironmentVerification(p.implementation_commit,dict(p.source_digests),dict(p.implementation_digests),runtime,True)
def _r(p,*x):return {"purpose":p,"path":list(x)}
def _exact(v,keys):return isinstance(v,dict) and set(v)==set(keys)
def manifest_from_dict(v):
 keys={"schema","train_generator_seed","heldout_generator_seed","model_seed","readout_seed","bootstrap_root"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-SEEDS-V1" or any(not access._int(v[k]) for k in keys-{"schema"}):raise PrototypeInvariantError("INPUT_SCHEMA")
 m=SeedManifest(**v);validate_manifest(m);return m
def validate_manifest(m):
 roots=(m.train_generator_seed,m.heldout_generator_seed,m.model_seed,m.readout_seed,m.bootstrap_root)
 if len(set(roots))!=5 or any(x<2**31 or x>=2**32 for x in roots):raise PrototypeInvariantError("SEED_COLLISION")
def envelope_from_dict(v):
 keys={"schema","manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256","production_path_count"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-SEED-APPROVAL-V1" or any(not access._digest(v[k]) for k in ("manifest_sha256","historical_inventory_sha256","expected_generated_audit_sha256")) or v.get("production_path_count")!=2040:raise PrototypeInvariantError("INPUT_SCHEMA")
 return Envelope(**v)
def provenance_from_dict(v):
 keys={"schema","target_commit","implementation_commit","clean_tree","source_digests","implementation_digests","python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-BUILD-PROVENANCE-V1" or v.get("target_commit")!=TARGET_COMMIT or not isinstance(v.get("implementation_commit"),str) or len(v["implementation_commit"])!=40 or any(c not in "0123456789abcdef" for c in v["implementation_commit"]) or v.get("clean_tree") is not True or v.get("source_digests")!=SOURCE_DIGESTS or not _exact(v.get("implementation_digests"),IMPLEMENTATION_PATHS) or any(not access._digest(x) for x in v["implementation_digests"].values()) or any(not isinstance(v.get(k),str) or not v[k] for k in ("python_version","numpy_version","torch_version","platform_machine","platform_system","blas_fingerprint")):raise PrototypeInvariantError("PROVENANCE_CONTENT")
 return BuildProvenance(**v)

def audit(m):
 validate_manifest(m);rows=[_r("TRAIN_GENERATOR_SPLIT",m.train_generator_seed,1),_r("TRAIN_NUISANCE",m.train_generator_seed,1,7101),_r("HELDOUT_PROBE",m.heldout_generator_seed,6),_r("HELDOUT_CAL",m.heldout_generator_seed,3),_r("E0_INIT",m.model_seed,1),_r("L1_PREDICTOR_INIT",m.model_seed,2),_r("READOUT_INIT",m.readout_seed,10),_r("BOOTSTRAP",m.bootstrap_root,7601)]
 rows += [_r("TRAIN_SCHEDULE",m.model_seed,e,6101) for e in range(32)]
 rows += [_r("READOUT_SCHEDULE",m.readout_seed,0,u,6201) for u in range(2000)]
 rows.sort(key=lambda x:(x["purpose"],x["path"]));return rows
def validate_closed(inv):
 accepted.validate_closed(inv);src=inv.get("source_artifact_digests")
 if not isinstance(src,dict) or src.get(CLOSED_SOURCE_KEY)!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 rows=[{"purpose":x["purpose"][len(CLOSED_PREFIX):],"path":x["path"]} for x in inv["records"] if x["purpose"].startswith(CLOSED_PREFIX)];rows.sort(key=lambda x:(x["purpose"],x["path"]))
 if len(rows)!=291 or sha256_hex(canonical(rows))!=CLOSED_AUDIT_SHA256:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
def validate_seed(m,mraw,e,inv,iraw):
 validate_closed(inv);rows=audit(m)
 if sha256_hex(mraw)!=e.manifest_sha256 or sha256_hex(iraw)!=e.historical_inventory_sha256 or sha256_hex(canonical(rows))!=e.expected_generated_audit_sha256 or len(rows)!=2040 or Counter(x["purpose"] for x in rows)!=PURPOSE_COUNTS:raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
 paths={tuple(x["path"]) for x in rows};roots={x["path"][0] for x in rows};historical_paths={tuple(x["path"]) for x in inv["records"]};historical_roots=set(inv["roots"])
 if len(paths)!=2040 or len(roots)!=5 or paths&historical_paths or roots&historical_roots:raise PrototypeInvariantError("SEED_COLLISION")
 return {"manifest_sha256":e.manifest_sha256,"historical_inventory_sha256":e.historical_inventory_sha256,"generated_audit_sha256":e.expected_generated_audit_sha256,"production_path_count":2040,"historical_path_count":len(historical_paths),"path_intersection_count":0,"root_intersection_count":0}

def select_probe_indices(labels):
 y=np.asarray(labels)
 if y.shape!=(2048,) or y.dtype.kind not in "biu" or np.any((y!=0)&(y!=1)):raise PrototypeInvariantError("SELECTION_INVALID")
 zero=np.flatnonzero(y==0)[:128];one=np.flatnonzero(y==1)[:128]
 if len(zero)!=128 or len(one)!=128:raise PrototypeInvariantError("SELECTION_INVALID")
 idx=np.ascontiguousarray(np.sort(np.concatenate((zero,one))),dtype="<i8")
 if idx.shape!=(256,) or len(np.unique(idx))!=256 or np.any(idx<0) or np.any(idx>=2048) or np.bincount(y[idx].astype(np.int64),minlength=2).tolist()!=[128,128]:raise PrototypeInvariantError("SELECTION_INVALID")
 return idx
def readout_indices(root):
 for update in range(2000):
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,0,update,6201])));yield rng.permutation(256)
def train_probe(z,labels,root,schedule=None):
 if tuple(z.shape)!=(256,16):raise PrototypeInvariantError("READOUT_INVALID")
 y=np.asarray(labels,dtype=np.uint8);counts=np.bincount(y.astype(np.int64),minlength=2)
 if y.shape!=(256,) or counts.tolist()!=[128,128]:raise PrototypeInvariantError("READOUT_INVALID")
 batches=list(readout_indices(root) if schedule is None else schedule)
 if len(batches)!=2000:raise PrototypeInvariantError("READOUT_INVALID")
 probes,_=stage1.make_readout_initializations(root);probe=probes[0];initial=access.state_hash(probe);weights=np.where(y==0,256/(2*counts[0]),256/(2*counts[1])).astype(np.float32);target=torch.as_tensor(y,dtype=torch.float32);weight=torch.as_tensor(weights);opt=stage1._adamw(list(probe.parameters()),lr=1e-2,weight_decay=1e-3);losses=[];sh=hashlib.sha256()
 for idx0 in batches:
  idx=np.ascontiguousarray(np.asarray(idx0),dtype="<i8")
  if idx.shape!=(256,) or not np.array_equal(np.sort(idx),np.arange(256)):raise PrototypeInvariantError("READOUT_INVALID")
  sh.update(idx.tobytes());ti=torch.as_tensor(idx.astype(np.int64),dtype=torch.long);opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(probe(z[ti]).squeeze(-1),target[ti],weight=weight[ti]);stage1._finite_loss(loss);loss.backward()
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  opt.step()
  if any(not bool(torch.isfinite(p).all()) for p in probe.parameters()):raise PrototypeInvariantError("READOUT_INVALID")
  losses.append(float(loss.detach()))
 probe.eval();report={"initial_state_hash":initial,"final_state_hash":access.state_hash(probe),"complete_schedule_hash":sh.hexdigest(),"target_hash":full.array_sha256("selected.target",y),"weight_hash":full.array_sha256("selected.weight",weights),"class_counts":counts.tolist(),"class_weights":[1.0,1.0],"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"first_100_mean_loss":float(np.mean(losses[:100])),"last_100_mean_loss":float(np.mean(losses[-100:]))};return probe,report
def _assert_frozen(e0,l1,frozen):
 if full._state_dict_bytes(e0)!=frozen["E0"] or full._state_dict_bytes(l1.encoder)!=frozen["L1"]:raise PrototypeInvariantError("FROZEN_STATE_INVALID")

def bootstrap(d_gain,root,replicates=10000,supplied_indices=None):
 d=np.ascontiguousarray(np.asarray(d_gain,dtype="<f8"))
 if d.shape!=(2048,) or not np.isfinite(d).all():raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 gain=float(d.mean())
 if supplied_indices is None:
  if replicates!=10000:raise PrototypeInvariantError("BOOTSTRAP_INVALID")
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,7601])));idx=rng.integers(0,2048,size=(replicates,2048),dtype=np.int64)
 else:idx=np.asarray(supplied_indices,dtype=np.int64)
 if idx.shape!=(replicates,2048) or np.any(idx<0) or np.any(idx>=2048):raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 e=np.empty(replicates,dtype="<f8")
 for start in range(0,replicates,64):e[start:start+len(idx[start:start+64])]=d[idx[start:start+64]].mean(axis=1)-gain
 if not np.isfinite(e).all():raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 q05=float(np.quantile(e,.05,method="linear"));q95=float(np.quantile(e,.95,method="linear"));bounds={"LCB95_gain":float(gain-q95),"UCB95_gain":float(gain-q05)}
 if bounds["LCB95_gain"]>bounds["UCB95_gain"]:raise PrototypeInvariantError("BOOTSTRAP_INVALID")
 return e,{"gain":gain,"q05":q05,"q95":q95,"bounds":bounds}
def terminal(l1_ba,constraints,bounds):
 useful=bool(constraints and l1_ba>=.80)
 if useful and bounds["LCB95_gain"]>0:return "SAMPLE_EFFICIENT",useful
 if constraints and bounds["UCB95_gain"]<=0:return "NO_SAMPLE_EFFICIENCY",useful
 return "UNINFORMATIVE",useful
def array_artifacts(selected,d,e):
 arrays={"selected_indices":np.ascontiguousarray(selected,dtype="<i8"),"d_gain":np.ascontiguousarray(d,dtype="<f8"),"bootstrap_centered":np.ascontiguousarray(e,dtype="<f8")};expected={"selected_indices":(256,"<i8"),"d_gain":(2048,"<f8"),"bootstrap_centered":(10000,"<f8")}
 for k,(n,_) in expected.items():
  if arrays[k].shape!=(n,):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 raw={k:v.tobytes() for k,v in arrays.items()};meta={k:{"file":ARRAY_FILES[k],"dtype":expected[k][1],"shape":[expected[k][0]],"sha256":sha256_hex(raw[k]),"byte_count":len(raw[k])} for k in arrays};return raw,meta
def validate_retained_arrays(result,raw):
 if set(raw)!=set(ARRAY_FILES):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 specs={"selected_indices":("<i8",256),"d_gain":("<f8",2048),"bootstrap_centered":("<f8",10000)};arrays={k:np.frombuffer(raw[k],dtype=specs[k][0]) for k in raw}
 for k,a in arrays.items():
  expected={"file":ARRAY_FILES[k],"dtype":specs[k][0],"shape":[specs[k][1]],"sha256":sha256_hex(raw[k]),"byte_count":a.nbytes}
  if a.shape!=(specs[k][1],) or result["retained_arrays"].get(k)!=expected:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 idx=arrays["selected_indices"];d=arrays["d_gain"];e=arrays["bootstrap_centered"]
 if len(np.unique(idx))!=256 or np.any(idx<0) or np.any(idx>=2048) or np.any(idx[1:]<=idx[:-1]) or not np.isfinite(d).all() or not np.isfinite(e).all():raise PrototypeInvariantError("SERIALIZATION_INVALID")
 gain=float(d.mean());q05=float(np.quantile(e,.05,method="linear"));q95=float(np.quantile(e,.95,method="linear"));b={"LCB95_gain":float(gain-q95),"UCB95_gain":float(gain-q05)};boot=result["bootstrap"]
 if (gain,q05,q95)!=(boot["gain"],boot["q05"],boot["q95"]) or b!=boot["bounds"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
def failure_artifact(phase,code):return {"schema":"BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-INVALID-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"NO_SCIENTIFIC_INTERPRETATION","valid":False,"terminal_outcome":"INVALID","failure":{"phase":phase,"code":code}}
def validate_failure_schema(v):
 phases={"PROVENANCE","SEED_AUDIT","GENERATION","TRAINING","SELECTION","READOUT","CAL","BOOTSTRAP","SERIALIZATION"};codes={"PROVENANCE_CONTENT","SEED_AUDIT_DIGEST","SEED_COLLISION","GENERATION_INVARIANT","TRAINING_INVARIANT","SELECTION_INVALID","READOUT_INVALID","FROZEN_STATE_INVALID","CAL_INVALID","BOOTSTRAP_INVALID","SERIALIZATION_INVALID","INPUT_SCHEMA"}
 if not _exact(v,{"schema","namespace","contract_sha256","claim_ceiling","valid","terminal_outcome","failure"}) or v.get("schema")!="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-INVALID-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="NO_SCIENTIFIC_INTERPRETATION" or v.get("valid") is not False or v.get("terminal_outcome")!="INVALID" or not _exact(v.get("failure"),{"phase","code"}) or v["failure"]["phase"] not in phases or v["failure"]["code"] not in codes:raise PrototypeInvariantError("SERIALIZATION_INVALID")
def fixed():return {"train_n":8192,"probe_pool_n":2048,"probe_fit_n":256,"probe_class_counts":[128,128],"cal_n":2048,"model_count":1,"target_factor":0,"l1_updates":2000,"readout_updates":2000,"readout_batch_n":256,"bootstrap_replicates":10000,"threshold":0.0,"useful_ba_floor":0.80,"s_probabilities":list(PAIR_S),"signal_flip_probabilities":list(PAIR_FLIP)}
def validate_success(v):
 keys={"schema","namespace","contract_sha256","claim_ceiling","provenance","seed_audit","fixed","checks","selection","model","bootstrap","retained_arrays","valid","terminal_outcome"}
 if not _exact(v,keys) or v.get("schema")!="BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-RESULT-V1" or v.get("namespace")!=NAMESPACE or v.get("contract_sha256")!=CONTRACT_SHA256 or v.get("claim_ceiling")!="EXACT_ONE_PAIR_ONE_MODEL_256_LABEL_MATCHED_ACCESSIBILITY_ONLY" or v.get("fixed")!=fixed() or v.get("valid") is not True or v.get("terminal_outcome") not in {"SAMPLE_EFFICIENT","NO_SAMPLE_EFFICIENCY","UNINFORMATIVE"}:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 checks=v.get("checks");required={"source_pins","seed_audit","selection_exact","matched_initialization","e0_immutable","l1_immutable","l1_training_exact","identical_readout_protocol","full_batch_schedule","cal_only_evaluation","paired_bootstrap","retained_arrays","output_allowlist"}
 if not _exact(checks,required) or not all(x is True for x in checks.values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 sel=v.get("selection")
 if not _exact(sel,{"rule","selected_index_hash","selected_target_hash","class_counts","shape"}) or sel["rule"]!="FIRST_128_PER_CLASS_UNION_CANONICAL_ORDER" or not access._digest(sel["selected_index_hash"]) or not access._digest(sel["selected_target_hash"]) or sel["class_counts"]!=[128,128] or sel["shape"]!=[256]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 provenance_from_dict({"schema":"BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-BUILD-PROVENANCE-V1",**{k:v["provenance"][k] for k in v["provenance"] if k!="build_provenance_sha256"}})
 if not access._digest(v["provenance"].get("build_provenance_sha256")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 seed=v.get("seed_audit")
 if not _exact(seed,{"approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256","production_path_count","historical_path_count","path_intersection_count","root_intersection_count"}) or any(not access._digest(seed[k]) for k in ("approved_envelope_sha256","manifest_sha256","historical_inventory_sha256","generated_audit_sha256")) or (seed["production_path_count"],seed["path_intersection_count"],seed["root_intersection_count"])!=(2040,0,0):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 model=v.get("model")
 if not _exact(model,{"state_hashes","l1_training","probe_fits","cal","constraints"}) or not _exact(model["state_hashes"],{"e0","l1_encoder","l1_teacher","l1_predictor"}) or any(not access._digest(x) for x in model["state_hashes"].values()):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 access._validate_l1_training(model["l1_training"])
 fits=model["probe_fits"]
 if not _exact(fits,{"E0","L1"}):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 fitkeys={"initial_state_hash","final_state_hash","complete_schedule_hash","target_hash","weight_hash","selected_index_hash","class_counts","class_weights","attempted_steps","successful_steps","optimizer_steps","first_100_mean_loss","last_100_mean_loss"}
 for fit in fits.values():
  if not _exact(fit,fitkeys) or any(not access._digest(fit[k]) for k in ("initial_state_hash","final_state_hash","complete_schedule_hash","target_hash","weight_hash","selected_index_hash")) or fit["class_counts"]!=[128,128] or fit["class_weights"]!=[1.0,1.0] or tuple(fit[k] for k in ("attempted_steps","successful_steps","optimizer_steps"))!=(2000,2000,2000) or not access._finite(fit["first_100_mean_loss"]) or not access._finite(fit["last_100_mean_loss"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for k in ("initial_state_hash","complete_schedule_hash","target_hash","weight_hash","selected_index_hash","class_counts","class_weights"):
  if fits["E0"][k]!=fits["L1"][k]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if fits["E0"]["selected_index_hash"]!=sel["selected_index_hash"] or fits["E0"]["target_hash"]!=sel["selected_target_hash"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 cal=model.get("cal")
 if not _exact(cal,{"E0","L1"}):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 for x in cal.values():access._validate_assay(x)
 if cal["E0"]["target_hash"]!=cal["L1"]["target_hash"]:raise PrototypeInvariantError("SERIALIZATION_INVALID")
 cons=model.get("constraints")
 if not _exact(cons,{"threshold_digest","all_pass","rows"}) or not access._digest(cons["threshold_digest"]) or not isinstance(cons["all_pass"],bool) or not access._validate_constraint_rows(cons["rows"]) or cons["all_pass"] is not all(x["both_metrics_pass"] for x in cons["rows"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 boot=v.get("bootstrap")
 if not _exact(boot,{"replicates","quantile_method","gain","q05","q95","bounds","useful"}) or boot["replicates"]!=10000 or boot["quantile_method"]!="linear" or any(not access._finite(boot[k]) for k in ("gain","q05","q95")) or not _exact(boot["bounds"],{"LCB95_gain","UCB95_gain"}) or any(not access._finite(x) for x in boot["bounds"].values()) or boot["bounds"]["LCB95_gain"]>boot["bounds"]["UCB95_gain"] or not isinstance(boot["useful"],bool):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not math.isclose(cal["L1"]["balanced_accuracy"]-cal["E0"]["balanced_accuracy"],boot["gain"],rel_tol=0,abs_tol=1e-15):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 outcome,useful=terminal(cal["L1"]["balanced_accuracy"],cons["all_pass"],boot["bounds"])
 if (outcome,useful)!=(v["terminal_outcome"],boot["useful"]):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 if not _exact(v.get("retained_arrays"),ARRAY_FILES):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 expected={"selected_indices":("<i8",256,2048),"d_gain":("<f8",2048,16384),"bootstrap_centered":("<f8",10000,80000)}
 for k,(dtype,n,bytes_) in expected.items():
  m=v["retained_arrays"][k]
  if not _exact(m,{"file","dtype","shape","sha256","byte_count"}) or m!={"file":ARRAY_FILES[k],"dtype":dtype,"shape":[n],"sha256":m.get("sha256"),"byte_count":bytes_} or not access._digest(m.get("sha256")):raise PrototypeInvariantError("SERIALIZATION_INVALID")
 full.validate_recursive_output(v);access.delta.prior._validate_digest_fields(v)

def run_beta(m,p,seed,*,execution_verification,build_provenance_sha256,approved_envelope_sha256,phase_callback:Callable[[str],None]|None=None):
 if not isinstance(execution_verification,ExecutionEnvironmentVerification) or execution_verification.source_pins is not True or execution_verification.implementation_commit!=p.implementation_commit or dict(execution_verification.source_digests)!=dict(p.source_digests) or dict(execution_verification.implementation_digests)!=dict(p.implementation_digests) or dict(execution_verification.runtime_fingerprint)!=runtime_fingerprint():raise PrototypeInvariantError("PROVENANCE_CONTENT")
 source_pins=execution_verification.source_pins
 from clinical_jepa.eval.j04c_falsifier import TRAIN,PROBE_FIT,CAL_OOD,fit_stage0_time_transform,generate_factor_split
 from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
 from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split
 notify=phase_callback or (lambda _:None);notify("GENERATION");train=independent_train_nuisance(generate_factor_split(m.train_generator_seed,TRAIN,8192),m.train_generator_seed);probe=parameterized_split(m.heldout_generator_seed,PROBE_FIT,2048,PAIR_S,PAIR_FLIP);cal=parameterized_split(m.heldout_generator_seed,CAL_OOD,2048,PAIR_S,PAIR_FLIP);transform=fit_stage0_time_transform(train)
 notify("TRAINING");e0=stage1._fresh_encoder(m.model_seed).eval();witness=stage1._fresh_encoder(m.model_seed).eval();e0_initial=full._state_dict_bytes(e0)
 if full._state_dict_bytes(witness)!=e0_initial:raise PrototypeInvariantError("TRAINING_INVARIANT")
 l1=bridge.train_recipe_decoupled_seeds(train,transform,"L1_AVG",m.model_seed,m.model_seed,schedule_seed=m.model_seed,identity_predictor=True,total_steps=2000,directional_variance_weight=5.,directional_variance_floor=.01,per_identity_directional_hinge=True)
 required={"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"ema_updates":2000,"encoder_seed":m.model_seed,"predictor_seed":m.model_seed,"schedule_seed":m.model_seed,"identity_predictor_initialization":True,"student_variance_weight":0.0,"student_variance_floor":0.05,"directional_variance_weight":5.0,"directional_variance_floor":0.01,"recipe":"L1_AVG","per_identity_directional_hinge":True,"l2_layers":None}
 if any(l1.training.get(k)!=x for k,x in required.items()) or full._state_dict_bytes(e0)!=e0_initial:raise PrototypeInvariantError("TRAINING_INVARIANT")
 e0c=stage1.TrainedCondition("E0",e0,None,None,{});stage1.freeze_encoder(e0c);stage1.freeze_encoder(l1);frozen={"E0":full._state_dict_bytes(e0),"L1":full._state_dict_bytes(l1.encoder)};_assert_frozen(e0,l1,frozen)
 notify("SELECTION");y_pool=probe.S[:,0].astype(np.uint8);selected=select_probe_indices(y_pool);ys=y_pool[selected];selected_hash=sha256_hex(selected.tobytes());target_hash=full.array_sha256("selected.target",ys);_assert_frozen(e0,l1,frozen)
 notify("READOUT");z0=access.pooled(e0c,probe,transform)[selected];z1=access.pooled(l1,probe,transform)[selected];_assert_frozen(e0,l1,frozen);schedule=list(readout_indices(m.readout_seed));p0,r0=train_probe(z0,ys,m.readout_seed,schedule);_assert_frozen(e0,l1,frozen);p1,r1=train_probe(z1,ys,m.readout_seed,schedule);_assert_frozen(e0,l1,frozen)
 for r in (r0,r1):r["selected_index_hash"]=selected_hash
 if any(r0[k]!=r1[k] for k in ("initial_state_hash","complete_schedule_hash","target_hash","weight_hash","selected_index_hash","class_counts","class_weights")):raise PrototypeInvariantError("READOUT_INVALID")
 notify("CAL");z0c=access.pooled(e0c,cal,transform);z1c=access.pooled(l1,cal,transform);_assert_frozen(e0,l1,frozen);ycal=cal.S[:,0].astype(np.uint8);a0=access.assay(access.predict(p0,z0c),ycal);a1=access.assay(access.predict(p1,z1c),ycal);d=np.ascontiguousarray(a1.pop("row_scores")-a0.pop("row_scores"),dtype="<f8");_assert_frozen(e0,l1,frozen);reference,td=stage1.threshold_reference_report();rows,_=stage1.trained_collapse_diagnostics(l1,cal,transform,reference);constraints=bool(len(rows)==4 and all(x["both_metrics_pass"] for x in rows));_assert_frozen(e0,l1,frozen)
 notify("BOOTSTRAP");centered,stats=bootstrap(d,m.bootstrap_root);_assert_frozen(e0,l1,frozen);out,useful=terminal(a1["balanced_accuracy"],constraints,stats["bounds"]);raw,meta=array_artifacts(selected,d,centered)
 notify("SERIALIZATION");prov={"build_provenance_sha256":build_provenance_sha256,"target_commit":p.target_commit,"implementation_commit":p.implementation_commit,"clean_tree":p.clean_tree,"source_digests":p.source_digests,"implementation_digests":p.implementation_digests,"python_version":p.python_version,"numpy_version":p.numpy_version,"torch_version":p.torch_version,"platform_machine":p.platform_machine,"platform_system":p.platform_system,"blas_fingerprint":p.blas_fingerprint};result={"schema":"BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-RESULT-V1","namespace":NAMESPACE,"contract_sha256":CONTRACT_SHA256,"claim_ceiling":"EXACT_ONE_PAIR_ONE_MODEL_256_LABEL_MATCHED_ACCESSIBILITY_ONLY","provenance":prov,"seed_audit":{"approved_envelope_sha256":approved_envelope_sha256,**dict(seed)},"fixed":fixed(),"checks":{"source_pins":source_pins,**{k:True for k in ("seed_audit","selection_exact","matched_initialization","e0_immutable","l1_immutable","l1_training_exact","identical_readout_protocol","full_batch_schedule","cal_only_evaluation","paired_bootstrap","retained_arrays","output_allowlist")}},"selection":{"rule":"FIRST_128_PER_CLASS_UNION_CANONICAL_ORDER","selected_index_hash":selected_hash,"selected_target_hash":target_hash,"class_counts":[128,128],"shape":[256]},"model":{"state_hashes":{"e0":access.state_hash(e0),"l1_encoder":access.state_hash(l1.encoder),"l1_teacher":access.state_hash(l1.teacher),"l1_predictor":access.state_hash(l1.predictor)},"l1_training":{k:v for k,v in l1.training.items() if k not in {"encoder_seed","predictor_seed","schedule_seed"}},"probe_fits":{"E0":r0,"L1":r1},"cal":{"E0":a0,"L1":a1},"constraints":{"threshold_digest":td,"all_pass":constraints,"rows":rows}},"bootstrap":{"replicates":10000,"quantile_method":"linear",**stats,"useful":useful},"retained_arrays":meta,"valid":True,"terminal_outcome":out};_assert_frozen(e0,l1,frozen);validate_success(result);validate_retained_arrays(result,raw);return result,raw
