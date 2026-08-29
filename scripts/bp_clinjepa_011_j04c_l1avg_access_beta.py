#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from clinical_jepa.eval import j04c_l1avg_access as access
PARSER=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
for x in ("build-provenance","build-provenance-sha256","seed-manifest","approved-seed-envelope","approved-seed-envelope-sha256","historical-path-inventory"):PARSER.add_argument(f"--{x}",required=True)
CODES={"PROVENANCE":"PROVENANCE_CONTENT","SEED_AUDIT":"SEED_AUDIT_DIGEST","GENERATION":"GENERATION_INVARIANT","TRAINING":"TRAINING_INVARIANT","READOUT":"READOUT_INVALID","BOOTSTRAP":"BOOTSTRAP_INVALID","SERIALIZATION":"SERIALIZATION_INVALID"}
def _load(path:str)->tuple[object,bytes]:
 raw=Path(path).read_bytes();v=json.loads(raw)
 if access.canonical(v)!=raw:raise access.PrototypeInvariantError("INPUT_SCHEMA")
 return v,raw
def _normalize(exc:BaseException,phase:str)->tuple[str,str]:
 text=str(exc)
 for code in ("PROVENANCE_CONTENT","SEED_AUDIT_DIGEST","SEED_COLLISION","GENERATION_INVARIANT","TRAINING_INVARIANT","READOUT_INVALID","BOOTSTRAP_INVALID","SERIALIZATION_INVALID","INPUT_SCHEMA"):
  if code in text:return phase,code
 return phase,CODES.get(phase,"SERIALIZATION_INVALID")
def main(argv:list[str]|None=None)->int:
 state={"phase":"PROVENANCE"}
 try:
  args=PARSER.parse_args(argv);pv,praw=_load(args.build_provenance);mv,mraw=_load(args.seed_manifest);ev,eraw=_load(args.approved_seed_envelope);iv,iraw=_load(args.historical_path_inventory)
  if access.sha256_hex(praw)!=args.build_provenance_sha256:raise access.PrototypeInvariantError("PROVENANCE_CONTENT")
  p=access.build_provenance_from_dict(pv);m=access.seed_manifest_from_dict(mv);e=access.approved_envelope_from_dict(ev);state["phase"]="SEED_AUDIT"
  if access.sha256_hex(eraw)!=args.approved_seed_envelope_sha256:raise access.PrototypeInvariantError("SEED_AUDIT_DIGEST")
  _,seed=access.validate_seed_audit(m,mraw,e,iv,iraw)
  result=access.run_beta(m,p,seed,build_provenance_sha256=args.build_provenance_sha256,approved_envelope_sha256=args.approved_seed_envelope_sha256,phase_callback=lambda x:state.__setitem__("phase",x))
  sys.stdout.buffer.write(access.canonical(result)+b"\n");return 0
 except BaseException as exc:
  if isinstance(exc,SystemExit) and exc.code==0:raise
  ph,code=_normalize(exc,state["phase"]);artifact=access.failure_artifact(ph,code);access.validate_failure_schema(artifact);sys.stdout.buffer.write(access.canonical(artifact)+b"\n");return 2
if __name__=="__main__":raise SystemExit(main())
