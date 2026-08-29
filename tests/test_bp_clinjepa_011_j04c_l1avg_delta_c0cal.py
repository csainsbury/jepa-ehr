from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from clinical_jepa.eval.j04c_falsifier import TRAIN, PROBE_FIT, CAL_OOD, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import tiny_pretraining_indices
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_v3_r0resid_c0cal as prior
from clinical_jepa.eval import j04c_l1avg_delta_c0cal as delta
from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split

HIGH = 2**31
DIGEST = "0" * 64


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def manifest(): return delta.SeedManifest("BP011-J04C-L1AVG-DELTA-C0CAL-SEEDS-V1", HIGH+11, HIGH+22, HIGH+33, HIGH+44)


def test_manifest_exact_four_roots_and_39_paths():
    m=manifest(); delta.validate_seed_manifest(m); records=delta.generated_seed_audit(m)
    assert len(records)==39 and records==sorted(records,key=lambda x:(x["purpose"],x["path"]))
    counts={p:sum(x["purpose"]==p for x in records) for p in {x["purpose"] for x in records}}
    assert counts=={"TRAIN_GENERATOR_SPLIT":1,"TRAIN_NUISANCE":1,"HELDOUT_PROBE":1,"HELDOUT_CAL":1,
                    "E0_INIT":1,"C0_HEAD_INIT":1,"TRAIN_SCHEDULE":32,"BOOTSTRAP":1}
    assert len({x["path"][0] for x in records})==4
    with pytest.raises(delta.PrototypeInvariantError,match="SEED_COLLISION"):
        delta.validate_seed_manifest(delta.SeedManifest(m.schema,HIGH+1,HIGH+1,HIGH+2,HIGH+3))
    with pytest.raises(delta.PrototypeInvariantError,match="INPUT_SCHEMA"):
        delta.approved_envelope_from_dict({"schema":"BP011-J04C-L1AVG-DELTA-C0CAL-SEED-APPROVAL-V1",
            "manifest_sha256":DIGEST,"historical_inventory_sha256":DIGEST,
            "expected_generated_audit_sha256":DIGEST,"production_path_count":39.0})


def test_exact_first_ordinal_parameterized_pair():
    split=parameterized_split(123,PROBE_FIT,2048,delta.PAIR_S,delta.PAIR_FLIP)
    assert split.S.shape==(2048,3) and split.prefix_type_ids.shape==(2048,7)
    assert delta._fixed()["s_probabilities"]==list(delta.PAIR_S)
    assert delta._fixed()["signal_flip_probabilities"]==list(delta.PAIR_FLIP)


def test_head_nonzero_bias_free_and_deterministic():
    a=delta.make_head(202); b=delta.make_head(202)
    assert a.bias is None and torch.equal(a.weight,b.weight)
    assert torch.isfinite(a.weight).all() and torch.count_nonzero(a.weight)>0


def test_tiny_full_encoder_clone_delta_training_and_nested_assay():
    gt,gh,m=101,303,202
    train=independent_train_nuisance(generate_factor_split(gt,TRAIN,64),gt)
    probe=parameterized_split(gh,PROBE_FIT,64,delta.PAIR_S,delta.PAIR_FLIP)
    cal=parameterized_split(gh,CAL_OOD,64,delta.PAIR_S,delta.PAIR_FLIP)
    transform=fit_stage0_time_transform(train)
    e0=delta.stage1._fresh_encoder(m).eval(); before=full._state_dict_bytes(e0)
    z0_train=delta.pooled_numpy(e0,train,transform)
    train_summary,train_logits,_=full._fit_base_bundle(z0_train,train.S[:,0].astype(float),{},"tiny.train_r0")
    offsets=train_logits["probe"].copy()
    product=delta.train_conditioned_c0(train,transform,m,e0,train_logits["probe"],
        tiny_pretraining_indices(m,n=64,batch_size=32,epochs=1),expected_steps=2)
    assert full._state_dict_bytes(e0)==before and np.array_equal(train_logits["probe"],offsets)
    assert product.training["successful_steps"]==2 and product.training["ema_updates"]==0
    z0p=delta.pooled_numpy(e0,probe,transform); z0c=delta.pooled_numpy(e0,cal,transform)
    zcp=delta.pooled_numpy(product.encoder,probe,transform); zcc=delta.pooled_numpy(product.encoder,cal,transform)
    yp=probe.S[:,0].astype(float); yc=cal.S[:,0].astype(float)
    rsum,rlog,_=full._fit_base_bundle(z0p,yp,{"cal":z0c},"tiny.r0")
    csum,clog,_=full._fit_additive_bundle(zcp-z0p,yp,rlog["probe"],{"cal":zcc-z0c},{"cal":rlog["cal"]},"tiny.c0")
    assert train_summary["converged"] and rsum["converged"] and csum["converged"]
    rows=full.binary_row_nll(rlog["cal"],yc)-full.binary_row_nll(clog["cal"],yc)
    indices=np.tile(np.arange(64),(20,1)); critical,contrasts,_=prior.bootstrap_one_lcb(rows,99,replicates=20,supplied_indices=indices)
    assert critical==pytest.approx(0,abs=1e-15) and contrasts[0]["name"]=="d_C0"


def test_closed_c0cal_inventory_extension_gate(monkeypatch):
    monkeypatch.setattr(prior,"validate_closed_lineages_inventory",lambda inventory:None)
    m0=prior.C0CalSeedManifest("BP011-J04C-V3-R0RESID-C0CAL-SEEDS-V1",HIGH+101,HIGH+102,HIGH+103)
    audit=prior.generated_seed_audit(m0); sha=full.sha256_hex(canonical(audit))
    monkeypatch.setattr(delta,"CLOSED_C0CAL_AUDIT_SHA256",sha)
    records=[dict(x,purpose=delta.CLOSED_C0CAL_PREFIX+x["purpose"]) for x in audit]
    inventory={"schema":"BP011-HISTORICAL-SEED-PATH-INVENTORY-V1","through_rows":"test",
               "roots":sorted({x["path"][0] for x in audit}),"records":records,
               "source_artifact_digests":{delta.CLOSED_C0CAL_SOURCE_KEY:sha}}
    delta.validate_closed_inventory(inventory)
    bad=json.loads(json.dumps(inventory));bad["records"].pop()
    with pytest.raises(delta.PrototypeInvariantError,match="SEED_AUDIT_DIGEST"):delta.validate_closed_inventory(bad)


def readout(): return {"preprocess_hash":DIGEST,"coefficient_hash":DIGEST,"constant_coordinate_mask_hash":DIGEST,"iterations":2,"converged":True,"zero_short_circuit":False}
def assay(): return {"nll_mean":.4,"balanced_accuracy":.8,"row_nll_hash":DIGEST,"logit_hash":DIGEST}


def success(eligible=True):
    contrast={"name":"d_C0","observed":.1,"lcb95":.01 if eligible else 0.0}
    return {"schema":"BP011-J04C-L1AVG-DELTA-C0CAL-RESULT-V1","namespace":delta.NAMESPACE,
      "contract_sha256":delta.CONTRACT_SHA256,"claim_ceiling":"SAFE_PUBLIC_FULL_ENCODER_DELTA_C0_ASSAY_CALIBRATION_ONLY",
      "provenance":{"build_provenance_sha256":DIGEST,"target_commit":delta.TARGET_COMMIT,"implementation_commit":"a"*40,
        "clean_tree":True,"source_digests":delta.SOURCE_DIGESTS,
        "implementation_digests":{p:DIGEST for p in delta.IMPLEMENTATION_PATHS},"python_version":"3","numpy_version":"2",
        "torch_version":"2","platform_machine":"x","platform_system":"Linux","blas_fingerprint":"sha256:x"},
      "seed_audit":{"approved_envelope_sha256":DIGEST,"manifest_sha256":DIGEST,"historical_inventory_sha256":DIGEST,
        "generated_audit_sha256":DIGEST,"production_path_count":39,"historical_path_count":1,
        "path_intersection_count":0,"root_intersection_count":0},"fixed":delta._fixed(),
      "checks":{x:True for x in ("source_pins","seed_audit","e0_clone_exact","e0_immutable","train_r0_only",
        "train_r0_frozen","initial_delta_zero","head_bias_free","first_encoder_gradient_nonzero","first_head_gradient_zero",
        "successful_steps","readout_deterministic","baseline_frozen","family_exact","output_allowlist","pair_exact")},
      "model":{"state_hashes":{x:DIGEST for x in ("e0","c0_encoder","c0_training_head","train_r0_logit_hash")},
        "training":{"C0_FULL_ENCODER_CONDITIONED":{"attempted_steps":2000,"successful_steps":2000,"optimizer_steps":2000,
          "ema_updates":0,"first_100_mean_total":.5,"last_100_mean_total":.4}},
        "readouts":{x:readout() for x in ("TRAIN_R0_OPTIMIZATION_ONLY","PROBE_R0_BASE","C0_DELTA_ADDITIVE")},
        "cal":{x:assay() for x in ("R0_BASE","C0_DELTA_ADDITIVE")}},
      "bootstrap":{"replicates":10000,"family_size":1,"quantile_method":"linear","critical_value":.09,"contrasts":[contrast]},
      "valid":True,"eligible":eligible,"scientific_gates":{"d_c0":eligible},"terminal_outcome":"ELIGIBLE" if eligible else "INELIGIBLE"}


def test_complete_success_failure_schemas():
    delta.validate_success_schema(success());delta.validate_success_schema(success(False))
    mutations=[]
    v=success();v["extra"]=1;mutations.append(v)
    v=success();v["provenance"]["extra"]=1;mutations.append(v)
    v=success();del v["seed_audit"]["manifest_sha256"];mutations.append(v)
    v=success();v["seed_audit"]["generated_audit_sha256"]="bad";mutations.append(v)
    v=success();v["provenance"]["implementation_digests"][delta.IMPLEMENTATION_PATHS[0]]="bad";mutations.append(v)
    v=success();v["model"]["training"]["C0_FULL_ENCODER_CONDITIONED"]["attempted_steps"]=2000.0;mutations.append(v)
    v=success();v["bootstrap"]["replicates"]=10000.0;mutations.append(v)
    v=success();v["bootstrap"]["family_size"]=1.0;mutations.append(v)
    v=success();v["eligible"]=1;mutations.append(v)
    v=success();v["scientific_gates"]={"d_c0":1};mutations.append(v)
    v=success();v["bootstrap"]["contrasts"]=[["bad"]];mutations.append(v)
    v=success();v["bootstrap"]["contrasts"][0]["extra"]=1;mutations.append(v)
    v=success();v["model"]["readouts"]["PROBE_R0_BASE"]["extra"]=1;mutations.append(v)
    v=success();v["model"]["readouts"]["PROBE_R0_BASE"]["iterations"]=2.0;mutations.append(v)
    v=success();v["model"]["readouts"]["PROBE_R0_BASE"]["zero_short_circuit"]=0;mutations.append(v)
    v=success();v["model"]["cal"]["R0_BASE"]["extra"]=1;mutations.append(v)
    for v in mutations:
        with pytest.raises(delta.PrototypeInvariantError,match="SERIALIZATION_INVALID"):
            delta.validate_success_schema(v)
    f=delta.failure_artifact("READOUT","READOUT_INVALID");delta.validate_failure_schema(f)


def test_runner_fail_closed_and_phase_defaults(capfd):
    path=Path("scripts/bp_clinjepa_011_j04c_l1avg_delta_c0cal_beta.py")
    spec=importlib.util.spec_from_file_location("delta_runner",path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.main([])==2;captured=capfd.readouterr();assert captured.err=="";delta.validate_failure_schema(json.loads(captured.out))
    defaults={"PROVENANCE":"PROVENANCE_CONTENT","SEED_AUDIT":"SEED_AUDIT_DIGEST","GENERATION":"GENERATION_INVARIANT",
              "TRAINING":"TRAINING_INVARIANT","READOUT":"READOUT_INVALID","BOOTSTRAP":"BOOTSTRAP_INVALID","SERIALIZATION":"SERIALIZATION_INVALID"}
    for phase,code in defaults.items():assert runner._normalize_failure(RuntimeError("generic"),phase)==(phase,code)


def test_build_provenance_and_static_guards():
    v={"schema":"BP011-J04C-L1AVG-DELTA-C0CAL-BUILD-PROVENANCE-V1","target_commit":delta.TARGET_COMMIT,
       "implementation_commit":"b"*40,"clean_tree":True,"source_digests":delta.SOURCE_DIGESTS,
       "implementation_digests":{p:DIGEST for p in delta.IMPLEMENTATION_PATHS},"python_version":"3","numpy_version":"2",
       "torch_version":"2","platform_machine":"x","platform_system":"Linux","blas_fingerprint":"sha256:x"}
    assert delta.build_provenance_from_dict(v).target_commit==delta.TARGET_COMMIT
    assert hashlib.sha256(Path("clinical_jepa/eval/j04c_v3_r0resid_c0cal.py").read_bytes()).hexdigest()==delta.SOURCE_DIGESTS["clinical_jepa/eval/j04c_v3_r0resid_c0cal.py"]
    module=Path("clinical_jepa/eval/j04c_l1avg_delta_c0cal.py");runner=Path("scripts/bp_clinjepa_011_j04c_l1avg_delta_c0cal_beta.py")
    mt=ast.parse(module.read_text());rt=ast.parse(runner.read_text())
    forbidden={"pathlib","subprocess","socket","requests","urllib"}
    for node in ast.walk(mt):
        if isinstance(node,ast.Import):assert all(a.name.split(".")[0] not in forbidden for a in node.names)
        if isinstance(node,ast.ImportFrom):assert (node.module or "").split(".")[0] not in forbidden
        if isinstance(node,ast.Attribute):assert node.attr not in {"cuda","save","load"}
    assert len([n for n in ast.walk(rt) if isinstance(n,ast.Attribute) and n.attr=="read_bytes"])==4
    text=module.read_text();assert "train_resid_candidate(" not in text and "POOLED_ONLINE_ENCODER_DELTA_Z_C0_MINUS_Z0" in text
