#!/usr/bin/env python3
import argparse,json,os,sys,tempfile
from pathlib import Path
from clinical_jepa.eval import j04c_l1avg_sample_efficiency as assay
P=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
for name in ("build-provenance","build-provenance-sha256","seed-manifest","approved-seed-envelope","approved-seed-envelope-sha256","historical-path-inventory","retained-array-dir"):P.add_argument("--"+name,required=True)
def load(path):
 raw=Path(path).read_bytes();value=json.loads(raw)
 if assay.canonical(value)!=raw:raise assay.PrototypeInvariantError("INPUT_SCHEMA")
 return value,raw
def write_arrays(directory,raw):
 target_dir=Path(directory)
 if not target_dir.is_dir() or target_dir.is_symlink() or any(target_dir.iterdir()):raise assay.PrototypeInvariantError("SERIALIZATION_INVALID")
 created=[]
 try:
  for key,name in assay.ARRAY_FILES.items():
   fd,tmp=tempfile.mkstemp(prefix="."+name+".",dir=target_dir);os.fchmod(fd,0o400)
   with os.fdopen(fd,"wb") as handle:handle.write(raw[key]);handle.flush();os.fsync(handle.fileno())
   target=target_dir/name;os.replace(tmp,target);created.append(target)
  return created
 except BaseException:
  for path in created:path.unlink(missing_ok=True)
  for path in target_dir.glob(".*.*le.*"):path.unlink(missing_ok=True)
  raise
def main(argv=None):
 state={"phase":"PROVENANCE"}
 try:
  args=P.parse_args(argv);pv,pr=load(args.build_provenance);mv,mr=load(args.seed_manifest);ev,er=load(args.approved_seed_envelope);iv,ir=load(args.historical_path_inventory)
  if assay.sha256_hex(pr)!=args.build_provenance_sha256:raise assay.PrototypeInvariantError("PROVENANCE_CONTENT")
  provenance=assay.provenance_from_dict(pv);manifest=assay.manifest_from_dict(mv);envelope=assay.envelope_from_dict(ev);state["phase"]="SEED_AUDIT"
  if assay.sha256_hex(er)!=args.approved_seed_envelope_sha256:raise assay.PrototypeInvariantError("SEED_AUDIT_DIGEST")
  seed=assay.validate_seed(manifest,mr,envelope,iv,ir);state["phase"]="PROVENANCE";execution_verification=assay.verify_execution_environment(provenance);result,raw=assay.run_beta(manifest,provenance,seed,execution_verification=execution_verification,build_provenance_sha256=args.build_provenance_sha256,approved_envelope_sha256=args.approved_seed_envelope_sha256,phase_callback=lambda phase:state.__setitem__("phase",phase));write_arrays(args.retained_array_dir,raw);sys.stdout.buffer.write(assay.canonical(result)+b"\n");return 0
 except BaseException as exc:
  if isinstance(exc,SystemExit) and exc.code==0:raise
  text=str(exc);codes=("PROVENANCE_CONTENT","SEED_AUDIT_DIGEST","SEED_COLLISION","GENERATION_INVARIANT","TRAINING_INVARIANT","SELECTION_INVALID","READOUT_INVALID","FROZEN_STATE_INVALID","CAL_INVALID","BOOTSTRAP_INVALID","SERIALIZATION_INVALID","INPUT_SCHEMA");defaults={"PROVENANCE":"PROVENANCE_CONTENT","SEED_AUDIT":"SEED_AUDIT_DIGEST","GENERATION":"GENERATION_INVARIANT","TRAINING":"TRAINING_INVARIANT","SELECTION":"SELECTION_INVALID","READOUT":"READOUT_INVALID","CAL":"CAL_INVALID","BOOTSTRAP":"BOOTSTRAP_INVALID","SERIALIZATION":"SERIALIZATION_INVALID"};code=next((x for x in codes if x in text),defaults.get(state["phase"],"SERIALIZATION_INVALID"));value=assay.failure_artifact(state["phase"],code);assay.validate_failure_schema(value);sys.stdout.buffer.write(assay.canonical(value)+b"\n");return 2
if __name__=="__main__":raise SystemExit(main())
