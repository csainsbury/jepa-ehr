from __future__ import annotations
import ast,hashlib,importlib.util,json
from pathlib import Path
import numpy as np,pytest,torch
from clinical_jepa.eval import j04c_l1avg_access as a
from clinical_jepa.eval import j04c_l1avg_delta_c0cal as delta
D="0"*64;H=2**31
def man():return a.SeedManifest("BP011-J04C-L1AVG-ACCESS-SEEDS-V1",H+1,H+2,H+3,H+4,H+5)
def test_five_roots_72_paths():
 m=man();a.validate_seed_manifest(m);r=a.generated_seed_audit(m);assert len(r)==72 and len({tuple(x['path']) for x in r})==72 and len({x['path'][0] for x in r})==5
 assert {k:sum(x['purpose']==k for x in r) for k in a.PURPOSE_COUNTS}==a.PURPOSE_COUNTS
 with pytest.raises(a.PrototypeInvariantError):a.validate_seed_manifest(a.SeedManifest(m.schema,H+1,H+1,H+2,H+3,H+4))
 with pytest.raises(a.PrototypeInvariantError):a.seed_manifest_from_dict({"schema":m.schema,"train_generator_seed":float(H+1),"heldout_generator_seed":H+2,"model_seed":H+3,"readout_seed":H+4,"bootstrap_root":H+5})
def test_bootstrap_and_terminal_goldens():
 x=np.array([-.2,.1,.3,.4],dtype=float);idx=np.tile(np.arange(4),(20,1));c,q=a.bootstrap_access(x,99,replicates=20,supplied_indices=idx);assert c==pytest.approx(0,abs=1e-15) and q['observed']==pytest.approx(.15) and q['lcb95']==pytest.approx(.15)
 assert a.terminal(.8,True,q)[0]=='SUPPORTED';q['lcb95']=0;assert a.terminal(.9,True,q)[0]=='NOT_SUPPORTED'
def test_balanced_row_score_mean_is_ba():
 y=np.array([0,0,0,1,1,1,1,1],dtype=np.uint8);p=np.array([0,1,0,1,0,1,1,1],dtype=np.uint8);yy=np.resize(y,2048);pp=np.resize(p,2048);v=a.assay(pp,yy);assert v['row_scores'].mean()==pytest.approx(v['balanced_accuracy'])
def test_probe_protocol_identical_and_finite():
 rng=np.random.default_rng(4);z=torch.tensor(rng.normal(size=(2048,16)),dtype=torch.float32);y=np.tile(np.array([0,1],dtype=np.uint8),1024);schedule=[np.arange(256,dtype=np.int64) for _ in range(250)]
 p0,r0=a.train_balanced_probe(z,y,H+9,schedule=schedule);p1,r1=a.train_balanced_probe(z,y,H+9,schedule=schedule);assert r0==r1 and a.state_hash(p0)==a.state_hash(p1) and r0['successful_steps']==250
def test_closed_delta_inventory_extension(monkeypatch):
 monkeypatch.setattr(delta,"validate_closed_inventory",lambda x:None);rows=[{"purpose":a.CLOSED_DELTA_PREFIX+x["purpose"],"path":x["path"]} for x in delta.generated_seed_audit(delta.SeedManifest("BP011-J04C-L1AVG-DELTA-C0CAL-SEEDS-V1",H+10,H+11,H+12,H+13))]
 raw=[{"purpose":x['purpose'][len(a.CLOSED_DELTA_PREFIX):],"path":x['path']} for x in rows];monkeypatch.setattr(a,"CLOSED_DELTA_AUDIT_SHA256",a.sha256_hex(a.canonical(sorted(raw,key=lambda x:(x['purpose'],x['path'])))))
 inv={"records":rows,"roots":[],"source_artifact_digests":{a.CLOSED_DELTA_SOURCE_KEY:a.CLOSED_DELTA_AUDIT_SHA256}};a.validate_closed_inventory(inv);inv['records'].pop()
 with pytest.raises(a.PrototypeInvariantError):a.validate_closed_inventory(inv)
def probe():return {"initial_state_hash":D,"final_state_hash":D,"schedule_hash":D,"target_hash":D,"weight_hash":D,"class_counts":[1000,1048],"class_weights":[1.024,.977],"attempted_steps":250,"successful_steps":250,"optimizer_steps":250,"first_100_mean_loss":.5,"last_100_mean_loss":.4}
def assay(tp,tn,fp,fn):return {"balanced_accuracy":.5*(tp/(tp+fn)+tn/(tn+fp)),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"prediction_hash":D,"target_hash":D,"row_score_hash":D}
def training():
 component={"first_100_mean":.5,"last_100_mean":.4}
 return {"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,"ema_updates":2000,"losses":{"first_100_mean_total":.5,"last_100_mean_total":.4,"components":{k:dict(component) for k in ("cosine","variance","v_pred","directional_variance_penalty","v_direction","v_direction_min")}},"identity_predictor_initialization":True,"student_variance_weight":0.0,"student_variance_floor":0.05,"directional_variance_weight":5.0,"directional_variance_floor":0.01,"recipe":"L1_AVG","per_identity_directional_hinge":True,"l2_layers":None}
def constraints():return [{"arm_name":"L1_AVG","identity_index":i,"normalized_variance":.2,"variance_threshold":.1,"variance_pass":True,"effective_rank":3.,"rank_threshold":2.,"rank_pass":True,"both_metrics_pass":True,"teacher_target_effective_rank":4.} for i in range(4)]
def success():
 prov={"build_provenance_sha256":D,"target_commit":a.TARGET_COMMIT,"implementation_commit":"a"*40,"clean_tree":True,"source_digests":a.SOURCE_DIGESTS,"implementation_digests":{p:D for p in a.IMPLEMENTATION_PATHS},"python_version":"3","numpy_version":"2","torch_version":"2","platform_machine":"x","platform_system":"Linux","blas_fingerprint":"x"}
 return {"schema":"BP011-J04C-L1AVG-ACCESS-RESULT-V1","namespace":a.NAMESPACE,"contract_sha256":a.CONTRACT_SHA256,"claim_ceiling":"ONE_PAIR_ONE_MODEL_MATCHED_L1_AVG_ACCESSIBILITY_ONLY","provenance":prov,"seed_audit":{"approved_envelope_sha256":D,"manifest_sha256":D,"historical_inventory_sha256":D,"generated_audit_sha256":D,"production_path_count":72,"historical_path_count":1,"path_intersection_count":0,"root_intersection_count":0},"fixed":a.fixed(),"checks":{k:True for k in ("source_pins","seed_audit","matched_initialization","e0_immutable","train_only_representation_learning","l1_training_exact","identical_probe_protocol","probe_only_fit","cal_only_evaluation","paired_bootstrap","output_allowlist","pair_exact")},"model":{"state_hashes":{"e0":D,"l1_encoder":D,"l1_teacher":D,"l1_predictor":D},"l1_training":training(),"probe_fits":{"E0":probe(),"L1":probe()},"cal":{"E0":assay(768,768,256,256),"L1":assay(870,870,154,154)},"constraints":{"threshold_digest":D,"all_pass":True,"rows":constraints()}},"bootstrap":{"replicates":10000,"family_size":1,"quantile_method":"linear","critical_value":.01,"contrast":{"name":"d_ACCESS","observed":.099609375,"lcb95":.089609375}},"valid":True,"supported":True,"scientific_gates":{"l1_ba_at_least_0_80":True,"access_gain_lcb_positive":True,"l1_constraints_all_pass":True},"terminal_outcome":"SUPPORTED"}
def test_success_schema_fail_closed():
 v=success();a.validate_success_schema(v)
 mutations=[]
 for path,val in [(('supported',),1),(('bootstrap','replicates'),10000.0),(('bootstrap','contrast','extra'),1),(('model','probe_fits','E0','optimizer_steps'),250.0),(('seed_audit','manifest_sha256'),'bad'),(('model','l1_training','recipe'),'bad'),(('model','constraints','rows',0,'variance_pass'),False),(('model','constraints','all_pass'),False),(('model','cal','L1','balanced_accuracy'),.9),(('model','probe_fits','L1','schedule_hash'),'f'*64),(('bootstrap','contrast','lcb95'),.08)]:
  x=json.loads(json.dumps(v));q=x
  for k in path[:-1]:q=q[k]
  q[path[-1]]=val;mutations.append(x)
 for x in mutations:
  with pytest.raises(a.PrototypeInvariantError,match='SERIALIZATION_INVALID'):a.validate_success_schema(x)
def test_runner_and_static_guards(capfd):
 path=Path('scripts/bp_clinjepa_011_j04c_l1avg_access_beta.py');spec=importlib.util.spec_from_file_location('runner',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);assert r.main([])==2;v=json.loads(capfd.readouterr().out);a.validate_failure_schema(v);assert v['schema']=='BP011-J04C-L1AVG-ACCESS-INVALID-V1' and v['namespace']==a.NAMESPACE and v['contract_sha256']==a.CONTRACT_SHA256
 module=Path('clinical_jepa/eval/j04c_l1avg_access.py');tree=ast.parse(module.read_text());assert 'MATCHED_REPRESENTATION_ACCESSIBILITY_NOT_INCREMENTAL_R0' in module.read_text()
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in {'pathlib','subprocess','socket','requests','urllib'} for x in n.names)
 assert hashlib.sha256(Path('clinical_jepa/eval/j04c_l1avg_delta_c0cal.py').read_bytes()).hexdigest()==a.SOURCE_DIGESTS['clinical_jepa/eval/j04c_l1avg_delta_c0cal.py']
