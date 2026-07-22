"""Step-4 launch approval POLICY DATA (Pi §1) — the static trust root for reviewed jobs.

This is DATA, not logic. It is deliberately EXCLUDED from the runner code closure (so populating an approval
does not change the code-closure identity — the same logic/data separation used by the aggregate-read policy).
It ships EMPTY / fail-closed: with no entries, NO step-4 job may launch. An approval is populated ONLY after Pi
reviews the exact revised manifest hashes and issues a gate event; the runner rejects any (run_id, job_kind,
reviewed_commit, manifest_hash) not present here.

Each entry is keyed by the approved run_id and pins the exact immutable job identity:

    APPROVED_STEP4_JOBS = {
        "m3a-step4-power-v1-run1": {
            "job_kind": "m3a-step4-power-v1",
            "reviewed_commit": "<exact full 40-char commit>",
            "manifest_hash": "<exact manifest_hash that reproduces at that commit>",
            "gate_event": "<agent-room gate event id>",
        },
        ...
    }
"""
from __future__ import annotations

# EMPTY = fail-closed. No launch is authorized until Pi populates exact approvals here.
APPROVED_STEP4_JOBS: dict = {}
