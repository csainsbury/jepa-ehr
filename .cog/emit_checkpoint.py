#!/usr/bin/env python3
"""Emit a deterministic metadata-only checkpoint from committed Git objects."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, re, subprocess, sys, unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import jsonschema, yaml

ROOT=Path(__file__).resolve().parents[1]
SCHEMAS=ROOT/".cog/schemas"
PROTECTED=re.compile(r"(^|/)(?:\.git|\.env|raw|data|datasets?|patients?|embeddings?|checkpoints?|transcripts?|emails?|provider-logs?|browser-profiles?|secrets?)(?:/|$)",re.I)

def git(*args:str, text=False):
    result=subprocess.run(["git","-C",str(ROOT),*args],check=True,capture_output=True,text=text)
    return result.stdout

def normalize_json(value):
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            nkey = unicodedata.normalize("NFC", key)
            if nkey in normalized:
                raise ValueError("NFC normalization key collision")
            normalized[nkey] = normalize_json(item)
        return normalized
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(normalize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

def core(value):
    x=copy.deepcopy(value)
    for k in ("event_id","idempotency_key","payload_hash"):x.pop(k,None)
    return x
def identity(value):
    d=hashlib.sha256(canonical_bytes(core(value))).hexdigest();when=value["recorded_at"].replace("-","").replace(":","")
    return f"pe_jepa_{when}_{value['repository']['commit_sha'][:12]}_{d[:12]}",f"sha256:{d}",f"sha256:{d}"
def schema_validate(value,path):jsonschema.Draft202012Validator(json.loads(path.read_text()),format_checker=jsonschema.FormatChecker()).validate(value)
def path_issue(path: str) -> str | None:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        return "unsafe-path"
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path):
        return "control-character"
    if any(char in path for char in "*?[]{}!"):
        return "glob-character"
    p = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in p.parts) or p.as_posix() != path:
        return "unsafe-path"
    if PROTECTED.search(path):
        return "protected-path"
    if p.suffix.lower() not in (".md", ".markdown"):
        return "not-markdown"
    return None

def safe_path(path): return path_issue(path) is None

def load_manifest(commit):
    raw=git("show",f"{commit}:.cog/project.yaml")
    value=yaml.safe_load(raw.decode("utf-8"));schema_validate(value,SCHEMAS/"cog-project-manifest-v1.schema.json");return value

def blob_meta(commit,path):
    if not safe_path(path):raise ValueError(f"unsafe document path: {path}")
    line=git("ls-tree",commit,"--",path,text=True).strip()
    parts=line.split(None,3)
    if len(parts)!=4 or parts[1]!="blob" or parts[0] not in ("100644","100755"):raise ValueError(f"not a regular committed blob: {path}")
    sha=parts[2];size=int(git("cat-file","-s",sha,text=True).strip());return sha,size

def build_event(commit,ref,recorded_at):
    commit=git("rev-parse",f"{commit}^{{commit}}",text=True).strip();tree=git("show","-s","--format=%T",commit,text=True).strip()
    manifest=load_manifest(commit)
    if ref not in manifest["allowed_refs"]:raise ValueError("ref is not allowlisted")
    if manifest["project"]!={"slug":"jepa","repository":"csainsbury/jepa-ehr"}:raise ValueError("wrong project")
    docs=[];total=0;seen=set()
    for item in manifest["documents"]:
        path=item["path"]
        if path in seen:raise ValueError("duplicate document path")
        seen.add(path);sha,size=blob_meta(commit,path);total+=size
        if size>manifest["limits"]["max_file_bytes"] or total>manifest["limits"]["max_total_bytes"]:raise ValueError("document size limit exceeded")
        docs.append({**item,"blob_sha":sha,"github_url":f"https://github.com/csainsbury/jepa-ehr/blob/{commit}/{quote(path,safe='/')}","media_type":"text/markdown","size_bytes":size})
    event={"schema_version":"cog-continuity-checkpoint-v1","project":manifest["project"],"repository":{"ref":ref,"commit_sha":commit,"tree_sha":tree,"commit_url":f"https://github.com/csainsbury/jepa-ehr/commit/{commit}"},"trigger":{"kind":"private_inbox_collector","source":"github_public_branch"},"occurred_at":recorded_at,"recorded_at":recorded_at,"milestone":{"kind":"allowlisted_document_push","headline":"Committed evidence checkpoint captured","state_effect":"no_state_change"},"documents":docs,"provenance":{"capture":"deterministic_import","review_status":"unreviewed","sensitivity":"safe_distilled"}}
    event["event_id"],event["idempotency_key"],event["payload_hash"]=identity(event);schema_validate(event,SCHEMAS/"cog-continuity-checkpoint-v1.schema.json");return event

def write_event(event,output_root):
    month=event["recorded_at"][:7].replace("-","/");path=output_root/"events/jepa"/month/f"{event['event_id']}.json";data=canonical_bytes(event);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError("idempotency conflict")
        return path
    path.write_bytes(data);return path

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--commit",required=True);p.add_argument("--ref",required=True);p.add_argument("--recorded-at",required=True);p.add_argument("--output-root",type=Path,required=True);a=p.parse_args(argv)
    event=build_event(a.commit,a.ref,a.recorded_at);path=write_event(event,a.output_root);print(json.dumps({"event_id":event["event_id"],"path":path.as_posix(),"payload_hash":event["payload_hash"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
