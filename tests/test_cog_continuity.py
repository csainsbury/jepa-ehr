import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
import yaml

SOURCE=Path(__file__).resolve().parents[1]/".cog/emit_checkpoint.py"
spec=importlib.util.spec_from_file_location("emit",SOURCE);emit=importlib.util.module_from_spec(spec);spec.loader.exec_module(emit)
SCHEMA_SOURCE=Path(__file__).resolve().parents[1]/".cog/schemas"

def run(root,*args):return subprocess.run(["git","-C",str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
def manifest(path="README.md"):
 return {"schema_version":"cog-project-manifest-v1","project":{"slug":"jepa","repository":"csainsbury/jepa-ehr"},"transport":{"repository":"csainsbury/cog-continuity-inbox"},"allowed_refs":["refs/heads/cog/jepa-continuity-pilot"],"documents":[{"path":path,"label":"Overview","role":"resume","classification":"safe_distilled"}],"limits":{"max_file_bytes":262144,"max_total_bytes":524288}}

class EmitterTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);run(self.root,"init","-b","cog/jepa-continuity-pilot");run(self.root,"config","user.email","test@example.invalid");run(self.root,"config","user.name","Test")
  (self.root/".cog/schemas").mkdir(parents=True)
  for p in SCHEMA_SOURCE.glob("*.json"):(self.root/".cog/schemas"/p.name).write_bytes(p.read_bytes())
  self.write_manifest(manifest());(self.root/"README.md").write_text("# Synthetic\n");run(self.root,"add",".");run(self.root,"commit","-m","synthetic")
  self.old_root,self.old_schemas=emit.ROOT,emit.SCHEMAS;emit.ROOT=self.root;emit.SCHEMAS=self.root/".cog/schemas"
 def tearDown(self):emit.ROOT,emit.SCHEMAS=self.old_root,self.old_schemas;self.t.cleanup()
 def write_manifest(self,m):
  (self.root/".cog").mkdir(exist_ok=True);(self.root/".cog/project.yaml").write_text(yaml.safe_dump(m,sort_keys=False))
 def build(self,ref="refs/heads/cog/jepa-continuity-pilot"):
  sha=run(self.root,"rev-parse","HEAD");return emit.build_event(sha,ref,"2026-07-10T10:00:00Z")
 def test_metadata_only_valid_and_idempotent(self):
  e=self.build();self.assertNotIn("body",json.dumps(e));self.assertEqual("Committed evidence checkpoint captured",e["milestone"]["headline"]);out=self.root/"out";p=emit.write_event(e,out);self.assertEqual(p,emit.write_event(e,out));self.assertEqual("unreviewed",e["provenance"]["review_status"])
 def test_canonical_and_path_vectors(self):
  vectors=json.loads((Path(__file__).resolve().parents[1]/".cog/canonical-vectors.json").read_text());a,b=vectors["equivalent"];self.assertEqual(emit.canonical_bytes(a),emit.canonical_bytes(b))
  with self.assertRaisesRegex(ValueError,"collision"):emit.canonical_bytes(vectors["collision"])
  for path in vectors["paths"]["valid"]:self.assertTrue(emit.safe_path(path),path)
  for path in vectors["paths"]["invalid"]:self.assertFalse(emit.safe_path(path),path)
 def test_ref_rejected(self):
  with self.assertRaisesRegex(ValueError,"allowlisted"):self.build("refs/heads/main")
 def test_traversal_protected_and_symlink_rejected(self):
  for path in ("../x.md","raw/x.md"):
   self.write_manifest(manifest(path));run(self.root,"add",".cog/project.yaml");run(self.root,"commit","-m","bad")
   with self.assertRaises((ValueError,Exception)):self.build()
  (self.root/"link.md").symlink_to("README.md");self.write_manifest(manifest("link.md"));run(self.root,"add",".");run(self.root,"commit","-m","link")
  with self.assertRaisesRegex(ValueError,"regular committed blob"):self.build()
 def test_oversize_rejected(self):
  (self.root/"big.md").write_bytes(b"x"*262145);self.write_manifest(manifest("big.md"));run(self.root,"add",".");run(self.root,"commit","-m","big")
  with self.assertRaisesRegex(ValueError,"size limit"):self.build()
 def test_idempotency_conflict(self):
  e=self.build();out=self.root/"out";p=emit.write_event(e,out);p.write_text("{}\n")
  with self.assertRaisesRegex(ValueError,"idempotency conflict"):emit.write_event(e,out)
 def test_schema_rejects_body_and_trust_escalation(self):
  e=self.build();e["documents"][0]["body"]="x"
  with self.assertRaises(Exception):emit.schema_validate(e,emit.SCHEMAS/"cog-continuity-checkpoint-v1.schema.json")
  e=self.build();e["provenance"]["review_status"]="confirmed"
  with self.assertRaises(Exception):emit.schema_validate(e,emit.SCHEMAS/"cog-continuity-checkpoint-v1.schema.json")
if __name__=="__main__":unittest.main()
