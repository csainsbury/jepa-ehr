#!/usr/bin/env python3
"""Create the first Clinical-JEPA v0 protocol workspace skeleton."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

FILES = {
"state/task-work/clinical-jepa-pilot/v0/protocol.md": """# Clinical-JEPA v0 protocol\n\n## Purpose\n\nRun a substrate bake-off for Clinical-JEPA: frozen FlatASCEND scaffold, flat-token JEPA from scratch, raw/MEDS-lite feasibility, and direct query baseline.\n\n## Frozen scope\n\n- v0A: frozen FlatASCEND hidden-state target.\n- v0B: flat-token JEPA from scratch.\n- v0C: raw/MEDS/OMOP-lite JEPA feasibility arm.\n- v0D: direct query-conditioned baseline.\n\n## Required target blocks\n\n- T0: next fixed event/window target.\n- T1: medication-change interval target.\n- T2: outcome-proximal target, optional first-pass stress block.\n\n## Required gates\n\nSee `../next-actions.md` and `arms.md`. No model training should begin until split, leakage, metric, and baseline rules are complete.\n""",
"state/task-work/clinical-jepa-pilot/v0/arms.md": """# v0 arms\n\n## v0A — frozen FlatASCEND scaffold\n\nStatus: verify / bootstrap only.\n\nRequired: target-layer ablation, pooling ablation, comparison to frozen FlatASCEND context embedding, utilisation controls.\n\n## v0B — flat-token JEPA from scratch\n\nStatus: promote for bake-off.\n\nRequired: EMA target encoder, anti-collapse regularisation, same target blocks as v0A.\n\n## v0C — raw/MEDS-lite JEPA feasibility\n\nStatus: verify as bounded spike.\n\nRequired: safe extraction checklist before any implementation; only minimal schema.\n\n## v0D — direct query-conditioned baseline\n\nStatus: mandatory baseline.\n\nRequired: 3-5 query descriptors and shared split/metric harness.\n""",
"state/task-work/clinical-jepa-pilot/v0/metrics.md": """# v0 metrics\n\n## Representation metrics\n\n- target retrieval Recall@1/5/10;\n- MRR;\n- true target rank among matched distractors;\n- CKA/Procrustes as diagnostic only.\n\n## Clinical probes\n\n- prefix-only mortality/shock probe;\n- next lab category/direction;\n- next medication family;\n- medication/lab preservation in retrieved targets.\n\n## Collapse and care-process diagnostics\n\n- effective rank;\n- embedding variance/covariance spectrum;\n- nearest-neighbour diversity;\n- correlation with sequence length, visit count, lab count, medication count;\n- horizon sensitivity;\n- patient/time/action shuffles.\n\n## Baselines\n\n- empirical prior;\n- bag/count/time;\n- utilisation-intensity;\n- frozen FlatASCEND context embedding;\n- query-conditioned baseline.\n""",
"state/task-work/clinical-jepa-pilot/splits/split-spec.md": """# Split specification\n\n## Rules\n\n- Patient-level split only.\n- No patient overlap across train/dev/test.\n- Temporal holdout where feasible.\n- eICU is locked stress test, not tuning data, unless explicitly changed.\n- Store only hashed/safe identifiers in manifests.\n\n## Manifest schema\n\n```json\n{\n  \"dataset\": \"mimic-iv-tokenised\",\n  \"created_utc\": \"...\",\n  \"split_policy\": \"patient-level\",\n  \"counts\": {\"train\": 0, \"dev\": 0, \"test\": 0},\n  \"hash_method\": \"...\"\n}\n```\n""",
"state/task-work/clinical-jepa-pilot/leakage-rules/leakage-audit-plan.md": """# Leakage audit plan\n\n## Context rules\n\n- Context must end before target block start.\n- Endpoint confirmation tokens after time zero are forbidden in prefix-only probes.\n- Target-window tokens must not leak into context through cached embeddings.\n\n## Audits\n\n- forbidden-token audit for each endpoint;\n- horizon-boundary audit;\n- patient-overlap audit;\n- duplicated-window audit;\n- measurement-opportunity/utilisation audit;\n- full-sequence vs prefix-only check.\n\n## Stop conditions\n\nStop training/evaluation if leakage audit fails or if endpoint labels require future-confirmation tokens in the context.\n""",
"state/task-work/clinical-jepa-pilot/data-access-checklist.md": """# Data-access checklist\n\n## Safe distilled planning\n\nAllowed in prompts: this package, blueprint, safe wiki/source summaries, aggregate non-sensitive metrics.\n\n## Tokenised implementation\n\nBefore using tokenised data, confirm local path, DUA/governance status, no patient identifiers in logs, and output policy.\n\n## Raw/MEDS/OMOP-lite implementation\n\nRequires explicit approval before extraction. Confirm source system, allowed fields, local-only handling, de-identification, schema manifest, and no upload of patient-level data.\n\n## Backblaze/Vast\n\nUpload only package artifacts, code, configs, aggregate metrics, and checkpoints if approved. Do not upload raw clinical data unless separately authorised and encrypted.\n""",
"state/task-work/clinical-jepa-pilot/reviews/self-review.md": """# Self-review\n\n## Current status\n\nInitial workspace skeleton created. Needs human/agent completion against `blueprints/atomic-blueprint.md`.\n\n## Known risks to check\n\n- teacher circularity;\n- care-process/utilisation confounding;\n- raw-lite sprawl;\n- leakage;\n- inadequate baselines;\n- causal overclaim.\n""",
"state/task-work/clinical-jepa-pilot/next-actions.md": """# Next actions\n\n1. Complete v0 protocol details from the atomic blueprint.\n2. Run `scripts/local_outcome_check.py`.\n3. Review missing criteria.\n4. Only after protocol passes, plan v0A/v0B implementation in the relevant code repo.\n""",
}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="run-workspace")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(text)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "files": sorted(FILES),
        "source_blueprint": "blueprints/atomic-blueprint.md",
    }
    mpath = root / "state/task-work/clinical-jepa-pilot/workspace-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"created workspace: {root}")
    print(f"manifest: {mpath}")

if __name__ == "__main__":
    main()
