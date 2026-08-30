#!/usr/bin/env python3
import argparse,json,os,sys,tempfile
from pathlib import Path
from clinical_jepa.eval import j04c_l1avg_readout_sufficiency as a
P=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
for x in ("build-provenance","build-provenance-sha256","seed-manifest","approved-seed-envelope","approved-seed-envelope-sha256","historical-path-inventory","retained-array-dir"):P.add_argument('--'+x,required=True)
def load(p):
 r=Path(p).read_bytes();v=json.loads(r)
 if a.canonical(v)!=r:raise a.PrototypeInvariantError('INPUT_SCHEMA')
 return v,r
def write_arrays(directory,raw):
 d=Path(directory)
 if not d.is_dir() or d.is_symlink() or any(d.iterdir()):raise a.PrototypeInvariantError('SERIALIZATION_INVALID')
 created=[]
 try:
  for key,name in a.ARRAY_FILES.items():
   fd,tmp=tempfile.mkstemp(prefix='.'+name+'.',dir=d);os.fchmod(fd,0o400)
   with os.fdopen(fd,'wb') as f:f.write(raw[key]);f.flush();os.fsync(f.fileno())
   target=d/name;os.replace(tmp,target);created.append(target)
  return created
 except BaseException:
  for p in created:p.unlink(missing_ok=True)
  for p in d.glob('.*.f64le.*'):p.unlink(missing_ok=True)
  raise
def main(argv=None):
 st={'p':'PROVENANCE'}
 try:
  x=P.parse_args(argv);pv,pr=load(x.build_provenance);mv,mr=load(x.seed_manifest);ev,er=load(x.approved_seed_envelope);iv,ir=load(x.historical_path_inventory)
  if a.sha256_hex(pr)!=x.build_provenance_sha256:raise a.PrototypeInvariantError('PROVENANCE_CONTENT')
  p=a.provenance_from_dict(pv);m=a.manifest_from_dict(mv);e=a.envelope_from_dict(ev);st['p']='SEED_AUDIT'
  if a.sha256_hex(er)!=x.approved_seed_envelope_sha256:raise a.PrototypeInvariantError('SEED_AUDIT_DIGEST')
  seed=a.validate_seed(m,mr,e,iv,ir);v,raw=a.run_beta(m,p,seed,build_provenance_sha256=x.build_provenance_sha256,approved_envelope_sha256=x.approved_seed_envelope_sha256,phase_callback=lambda z:st.__setitem__('p',z));write_arrays(x.retained_array_dir,raw);sys.stdout.buffer.write(a.canonical(v)+b'\n');return 0
 except BaseException as exc:
  if isinstance(exc,SystemExit) and exc.code==0:raise
  text=str(exc);code=next((z for z in ('PROVENANCE_CONTENT','SEED_AUDIT_DIGEST','SEED_COLLISION','GENERATION_INVARIANT','TRAINING_INVARIANT','READOUT_INVALID','BOOTSTRAP_INVALID','SERIALIZATION_INVALID','INPUT_SCHEMA') if z in text),{'PROVENANCE':'PROVENANCE_CONTENT','SEED_AUDIT':'SEED_AUDIT_DIGEST','GENERATION':'GENERATION_INVARIANT','TRAINING':'TRAINING_INVARIANT','READOUT':'READOUT_INVALID','BOOTSTRAP':'BOOTSTRAP_INVALID','SERIALIZATION':'SERIALIZATION_INVALID'}.get(st['p'],'SERIALIZATION_INVALID'));v=a.failure_artifact(st['p'],code);a.validate_failure_schema(v);sys.stdout.buffer.write(a.canonical(v)+b'\n');return 2
if __name__=='__main__':raise SystemExit(main())
