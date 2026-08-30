import copy,importlib.util,json
from pathlib import Path
import numpy as np,pytest,torch
from clinical_jepa.eval import j04c_l1avg_readout_sufficiency as a
from clinical_jepa.eval import j04c_l1avg_access as accepted
from clinical_jepa.eval import j04c_stage1 as stage1
H=2**31
def manifest():return a.SeedManifest('BP011-J04C-L1AVG-READOUT-SUFFICIENCY-SEEDS-V1',H+1,H+2,H+3,H+4,H+5)
def test_exact_five_root_291_path_complete_grammar():
 rows=a.audit(manifest());assert len(rows)==len({tuple(x['path']) for x in rows})==291;assert len({x['path'][0] for x in rows})==5;assert {k:sum(x['purpose']==k for x in rows) for k in a.PURPOSE_COUNTS}==a.PURPOSE_COUNTS
 assert [x['path'][2] for x in rows if x['purpose']=='READOUT_EXTENSION_SCHEDULE']==list(range(219))
def test_manifest_envelope_and_failure_schemas_reject_malformed():
 v={'schema':'BP011-J04C-L1AVG-READOUT-SUFFICIENCY-SEEDS-V1','train_generator_seed':H+1,'heldout_generator_seed':H+2,'model_seed':H+3,'readout_seed':H+4,'bootstrap_root':H+5};assert a.manifest_from_dict(v)==manifest();v['extra']=1
 with pytest.raises(a.PrototypeInvariantError):a.manifest_from_dict(v)
 f=a.failure_artifact('READOUT','READOUT_INVALID');a.validate_failure_schema(f);f['failure']['code']='OTHER'
 with pytest.raises(a.PrototypeInvariantError):a.validate_failure_schema(f)
def test_extension_schedule_exact_1750_and_fresh_domain():
 rows=list(a.extension_indices(H+4));assert len(rows)==1750 and all(x.shape==(256,) for x in rows);short=list(stage1.probe_indices(H+4,0));assert not np.array_equal(rows[0],short[0]);assert np.array_equal(rows[0],list(a.extension_indices(H+4))[0])
def test_uninterrupted_real_probe_short_snapshot_is_byte_exact_accepted():
 torch.set_num_threads(1);rng=np.random.default_rng(7);z=torch.as_tensor(rng.normal(size=(2048,16)).astype(np.float32));y=np.tile(np.array([0,1],dtype=np.uint8),1024);short=list(stage1.probe_indices(H+4,0));ext=list(a.extension_indices(H+4));tr,common=a._trajectory(z,y,H+4,short,ext);probe,report=accepted.train_balanced_probe(z,y,H+4,schedule=short)
 assert tr[250]['probe_state_bytes']==a.full._state_dict_bytes(probe);assert tr[250]['probe_state_hash']==__import__('hashlib').sha256(tr[250]['probe_state_bytes']).hexdigest();assert common['initial_state_hash']==report['initial_state_hash'];assert tr[250]['complete_schedule_hash']==report['schedule_hash'];assert tr[250]['optimizer_state_hash']!=tr[2000]['optimizer_state_hash'];assert tr[250]['probe_state_bytes']!=tr[2000]['probe_state_bytes'];assert (tr[2000]['attempted_steps'],tr[2000]['successful_steps'],tr[2000]['optimizer_steps'])==(2000,2000,2000)
def test_tiny_real_encoder_and_probe_seams():
 e0=stage1._fresh_encoder(H+3).eval();e1=stage1._fresh_encoder(H+3).eval();assert a.full._state_dict_bytes(e0)==a.full._state_dict_bytes(e1)
 z=torch.arange(128,dtype=torch.float32).reshape(8,16)/128;y=np.array([0,1]*4,dtype=np.uint8);batch=[np.array([0,1,2,3])]*250;ext=[np.array([4,5,6,7])]*1750;tr,c=a._trajectory(z,y,H+4,batch,ext);assert c['class_counts']==[4,4] and tr[250]['probe_state_bytes']!=tr[2000]['probe_state_bytes']
@pytest.mark.parametrize('arm',['E0','L1'])
def test_frozen_encoder_byte_assertion_rejects_perturbation(arm):
 e0=stage1._fresh_encoder(H+3).eval();l1=stage1.TrainedCondition('L1_AVG',stage1._fresh_encoder(H+3).eval(),None,None,{});stage1.freeze_encoder(stage1.TrainedCondition('E0',e0,None,None,{}));stage1.freeze_encoder(l1);frozen={'E0':a.full._state_dict_bytes(e0),'L1':a.full._state_dict_bytes(l1.encoder)}
 assert a._assert_frozen_encoder_bytes(e0,l1,frozen)=={'e0_immutable':True,'l1_immutable':True};module=e0 if arm=='E0' else l1.encoder
 with torch.no_grad():next(module.parameters()).view(-1)[0].add_(1.)
 with pytest.raises(a.PrototypeInvariantError,match='READOUT_INVALID'):a._assert_frozen_encoder_bytes(e0,l1,frozen)
def test_simultaneous_bootstrap_golden_and_retained_offline_reproduction():
 base=np.resize(np.array([-1.,0.,1.,2.]),2048);rows=np.vstack((base,base+.25,np.full(2048,.25)));idx=np.tile(np.arange(2048),(9,1));M,q,obs,bounds=a.bootstrap(rows,H+5,replicates=9,supplied_indices=idx);assert np.array_equal(M,np.zeros(9));assert q==0 and np.array_equal(obs,np.array([.5,.75,.25]));assert bounds=={'UCB95_short':.5,'LCB95_long':.75,'UCB95_long':.75,'LCB95_recovery':.25}
 raw,meta=a.array_artifacts(rows,np.zeros(10000));result={'retained_arrays':meta,'bootstrap':{'observed':obs.tolist(),'critical_value':0.,'bounds':bounds}};a.validate_retained_arrays(result,raw);bad=dict(raw);bad['M']=bad['M'][:-8]
 with pytest.raises(a.PrototypeInvariantError):a.validate_retained_arrays(result,bad)
def test_terminal_order_and_boundaries():
 b={'UCB95_short':0.,'LCB95_long':.01,'UCB95_long':.02,'LCB95_recovery':.01};assert a.terminal(.8,True,b)=='READOUT_LIMITED';b['LCB95_long']=0.;b['UCB95_long']=0.;assert a.terminal(.1,True,b)=='REPRESENTATION_LIMITED';b['UCB95_short']=1e-12;assert a.terminal(.9,True,b)=='UNINFORMATIVE'
def test_runner_guards_empty_retained_directory(tmp_path,capfd):
 p=Path('scripts/bp_clinjepa_011_j04c_l1avg_readout_sufficiency_beta.py');s=importlib.util.spec_from_file_location('rs_runner',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);d=tmp_path/'arrays';d.mkdir();raw={k:np.arange(2,dtype='<f8').tobytes() for k in a.ARRAY_FILES};made=r.write_arrays(d,raw);assert {x.name for x in made}==set(a.ARRAY_FILES.values());assert all((x.stat().st_mode&0o777)==0o400 for x in made)
 with pytest.raises(a.PrototypeInvariantError):r.write_arrays(d,raw)
 assert r.main([])==2;v=json.loads(capfd.readouterr().out);a.validate_failure_schema(v)
def test_provenance_and_validator_have_strict_guards():
 D='0'*64;v={'schema':'BP011-J04C-L1AVG-READOUT-SUFFICIENCY-BUILD-PROVENANCE-V1','target_commit':a.TARGET_COMMIT,'implementation_commit':'a'*40,'clean_tree':True,'source_digests':a.SOURCE_DIGESTS,'implementation_digests':{p:D for p in a.IMPLEMENTATION_PATHS},'python_version':'3','numpy_version':'2','torch_version':'2','platform_machine':'x','platform_system':'Linux','blas_fingerprint':'x'};a.provenance_from_dict(v);v['clean_tree']=False
 with pytest.raises(a.PrototypeInvariantError):a.provenance_from_dict(v)
 text=Path(a.IMPLEMENTATION_PATHS[0]).read_text();assert 'validate_retained_arrays' in text and 'np.maximum.reduce((-e[:,0],e[:,1],-e[:,1],e[:,2]))' in text
