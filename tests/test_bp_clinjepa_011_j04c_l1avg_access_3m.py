import importlib.util,json
from pathlib import Path
import numpy as np,pytest
from clinical_jepa.eval import j04c_l1avg_access_3m as a
H=2**31
def m():return a.SeedManifest('BP011-J04C-L1AVG-ACCESS-3M-SEEDS-V1',H+1,H+2,(H+3,H+4,H+5),H+6,H+7)
def test_exact_7_roots_140_paths():
 x=m();a.validate_manifest(x);r=a.audit(x);assert len(r)==140 and len({tuple(v['path']) for v in r})==140 and len({v['path'][0] for v in r})==7;assert {k:sum(v['purpose']==k for v in r) for k in a.PURPOSE_COUNTS}==a.PURPOSE_COUNTS
def test_manifest_strict():
 v={'schema':'BP011-J04C-L1AVG-ACCESS-3M-SEEDS-V1','train_generator_seed':H+1,'heldout_generator_seed':H+2,'model_seeds':[H+3,H+4,H+5],'readout_seed':H+6,'bootstrap_root':H+7};assert a.manifest_from_dict(v).model_seeds==(H+3,H+4,H+5);v['model_seeds'][2]=H+3
 with pytest.raises(a.PrototypeInvariantError):a.manifest_from_dict(v)
def test_shared_row_bootstrap_preserves_all_models():
 x=np.vstack((np.ones(2048),np.ones(2048)*2,np.ones(2048)*3));idx=np.tile(np.arange(2048),(20,1));c,q=a.bootstrap(x,9,replicates=20,supplied_indices=idx);assert c==pytest.approx(0,abs=1e-15) and q['observed']==pytest.approx(2) and q['lcb95']==pytest.approx(2)
def test_terminal():
 c={'lcb95':.01};assert a.terminal(.8,True,c)[0]=='SUPPORTED';c['lcb95']=0;assert a.terminal(.9,True,c)[0]=='NOT_SUPPORTED'
def test_runner_failure_is_3m(capfd):
 p=Path('scripts/bp_clinjepa_011_j04c_l1avg_access_3m_beta.py');s=importlib.util.spec_from_file_location('r3',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);assert r.main([])==2;v=json.loads(capfd.readouterr().out);assert v['schema']=='BP011-J04C-L1AVG-ACCESS-3M-INVALID-V1' and v['contract_sha256']==a.CONTRACT_SHA256;a.validate_failure_schema(v);v['failure']['code']='ARBITRARY'
 with pytest.raises(a.PrototypeInvariantError):a.validate_failure_schema(v)
def test_provenance_strict():
 D='0'*64;v={'schema':'BP011-J04C-L1AVG-ACCESS-3M-BUILD-PROVENANCE-V1','target_commit':a.TARGET_COMMIT,'implementation_commit':'a'*40,'clean_tree':True,'source_digests':a.SOURCE_DIGESTS,'implementation_digests':{p:D for p in a.IMPLEMENTATION_PATHS},'python_version':'3','numpy_version':'2','torch_version':'2','platform_machine':'x','platform_system':'Linux','blas_fingerprint':'x'};a.provenance_from_dict(v);v['implementation_commit']='bad'
 with pytest.raises(a.PrototypeInvariantError):a.provenance_from_dict(v)
def test_validator_contains_cross_model_and_strict_arithmetic_guards():
 text=Path('clinical_jepa/eval/j04c_l1avg_access_3m.py').read_text();assert 'cal_target=None' in text and text.count('rel_tol=0')>=2
