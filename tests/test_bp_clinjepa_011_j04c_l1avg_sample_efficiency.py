import importlib.util,json
from pathlib import Path
import numpy as np,pytest,torch
from clinical_jepa.eval import j04c_l1avg_sample_efficiency as assay
from clinical_jepa.eval import j04c_stage1 as stage1
H=2**31

def manifest():return assay.SeedManifest("BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-SEEDS-V1",H+1,H+2,H+3,H+4,H+5)
def test_exact_five_root_2040_path_complete_grammar():
 rows=assay.audit(manifest());assert len(rows)==len({tuple(x["path"]) for x in rows})==2040;assert len({x["path"][0] for x in rows})==5;assert dict(sorted({k:sum(x["purpose"]==k for x in rows) for k in assay.PURPOSE_COUNTS}.items()))==dict(sorted(assay.PURPOSE_COUNTS.items()))
 assert [x["path"][2] for x in rows if x["purpose"]=="READOUT_SCHEDULE"]==list(range(2000));assert all(x["path"][1]==0 and x["path"][3]==6201 for x in rows if x["purpose"]=="READOUT_SCHEDULE")
def test_manifest_envelope_failure_and_provenance_schemas_are_strict():
 value={"schema":"BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-SEEDS-V1","train_generator_seed":H+1,"heldout_generator_seed":H+2,"model_seed":H+3,"readout_seed":H+4,"bootstrap_root":H+5};assert assay.manifest_from_dict(value)==manifest();value["extra"]=1
 with pytest.raises(assay.PrototypeInvariantError):assay.manifest_from_dict(value)
 failure=assay.failure_artifact("SELECTION","SELECTION_INVALID");assay.validate_failure_schema(failure);failure["failure"]["code"]="OTHER"
 with pytest.raises(assay.PrototypeInvariantError):assay.validate_failure_schema(failure)
 digest="0"*64;provenance={"schema":"BP011-J04C-L1AVG-SAMPLE-EFFICIENCY-N256-BUILD-PROVENANCE-V1","target_commit":assay.TARGET_COMMIT,"implementation_commit":"a"*40,"clean_tree":True,"source_digests":assay.SOURCE_DIGESTS,"implementation_digests":{p:digest for p in assay.IMPLEMENTATION_PATHS},"python_version":"3","numpy_version":"2","torch_version":"2","platform_machine":"x","platform_system":"Linux","blas_fingerprint":"x"};assay.provenance_from_dict(provenance);provenance["clean_tree"]=False
 with pytest.raises(assay.PrototypeInvariantError):assay.provenance_from_dict(provenance)
def test_canonical_first_128_per_class_selection_and_hash_golden():
 y=np.resize(np.array([1,0,1,1,0],dtype=np.uint8),2048);idx=assay.select_probe_indices(y);expected=np.sort(np.concatenate((np.flatnonzero(y==0)[:128],np.flatnonzero(y==1)[:128]))).astype("<i8");assert np.array_equal(idx,expected);assert np.bincount(y[idx],minlength=2).tolist()==[128,128];assert assay.sha256_hex(idx.tobytes())=="790f052aacabdd94e0c5026624ab5d2d7606967440c35dd5a2e41aa0c9a3170a"
 for bad in (np.zeros(2048,dtype=np.uint8),np.resize(np.array([0,1],dtype=np.uint8),2047),np.full(2048,2,dtype=np.uint8)):
  with pytest.raises(assay.PrototypeInvariantError,match="SELECTION_INVALID"):assay.select_probe_indices(bad)
def test_readout_schedule_and_real_full_batch_2000_update_seam():
 schedule=list(assay.readout_indices(H+4));assert len(schedule)==2000 and all(np.array_equal(np.sort(x),np.arange(256)) for x in schedule);assert np.array_equal(schedule[0],list(assay.readout_indices(H+4))[0])
 torch.set_num_threads(1);z=torch.arange(4096,dtype=torch.float32).reshape(256,16)/4096;y=np.resize(np.array([0,1],dtype=np.uint8),256);probe,report=assay.train_probe(z,y,H+4,schedule);assert report["class_counts"]==[128,128] and report["class_weights"]==[1.,1.] and (report["attempted_steps"],report["successful_steps"],report["optimizer_steps"])==(2000,2000,2000);assert report["initial_state_hash"]!=report["final_state_hash"]
 bad=list(schedule);bad[3]=bad[3][:-1]
 with pytest.raises(assay.PrototypeInvariantError,match="READOUT_INVALID"):assay.train_probe(z,y,H+4,bad)
@pytest.mark.parametrize("arm",["E0","L1"])
def test_frozen_e0_l1_bytes_reject_perturbation_at_boundary(arm):
 e0=stage1._fresh_encoder(H+3).eval();l1=stage1.TrainedCondition("L1_AVG",stage1._fresh_encoder(H+3).eval(),None,None,{});stage1.freeze_encoder(stage1.TrainedCondition("E0",e0,None,None,{}));stage1.freeze_encoder(l1);frozen={"E0":assay.full._state_dict_bytes(e0),"L1":assay.full._state_dict_bytes(l1.encoder)};assay._assert_frozen(e0,l1,frozen);module=e0 if arm=="E0" else l1.encoder
 with torch.no_grad():next(module.parameters()).view(-1)[0].add_(1.)
 with pytest.raises(assay.PrototypeInvariantError,match="FROZEN_STATE_INVALID"):assay._assert_frozen(e0,l1,frozen)
def test_paired_bootstrap_golden_retained_arrays_and_offline_reproduction():
 d=np.resize(np.array([-1.,0.,1.,2.]),2048).astype("<f8");idx=np.tile(np.arange(2048),(9,1));centered,stats=assay.bootstrap(d,H+5,replicates=9,supplied_indices=idx);assert np.array_equal(centered,np.zeros(9));assert stats=={"gain":.5,"q05":0.,"q95":0.,"bounds":{"LCB95_gain":.5,"UCB95_gain":.5}}
 selected=np.arange(256,dtype="<i8");raw,meta=assay.array_artifacts(selected,d,np.zeros(10000,dtype="<f8"));result={"retained_arrays":meta,"bootstrap":{"gain":.5,"q05":0.,"q95":0.,"bounds":{"LCB95_gain":.5,"UCB95_gain":.5}}};assay.validate_retained_arrays(result,raw);bad=dict(raw);bad["d_gain"]=bad["d_gain"][:-8]
 with pytest.raises(assay.PrototypeInvariantError):assay.validate_retained_arrays(result,bad)
def test_terminal_order_and_exact_arithmetic_boundaries():
 assert assay.terminal(.8,True,{"LCB95_gain":1e-12,"UCB95_gain":.1})==("SAMPLE_EFFICIENT",True);assert assay.terminal(.9,True,{"LCB95_gain":0.,"UCB95_gain":0.})==("NO_SAMPLE_EFFICIENCY",True);assert assay.terminal(.79,True,{"LCB95_gain":.1,"UCB95_gain":.2})==("UNINFORMATIVE",False);assert assay.terminal(.9,False,{"LCB95_gain":-.2,"UCB95_gain":-.1})==("UNINFORMATIVE",False)
def test_guarded_runner_requires_canonical_inputs_and_empty_retained_dir(tmp_path,capfd):
 path=Path("scripts/bp_clinjepa_011_j04c_l1avg_sample_efficiency_beta.py");spec=importlib.util.spec_from_file_location("sample_runner",path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);directory=tmp_path/"arrays";directory.mkdir();raw={"selected_indices":np.arange(256,dtype="<i8").tobytes(),"d_gain":np.zeros(2048,dtype="<f8").tobytes(),"bootstrap_centered":np.zeros(10000,dtype="<f8").tobytes()};made=runner.write_arrays(directory,raw);assert {x.name for x in made}==set(assay.ARRAY_FILES.values());assert all((x.stat().st_mode&0o777)==0o400 for x in made)
 with pytest.raises(assay.PrototypeInvariantError):runner.write_arrays(directory,raw)
 assert runner.main([])==2;value=json.loads(capfd.readouterr().out);assay.validate_failure_schema(value)
