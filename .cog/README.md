# Cog continuity pilot

This directory declares the six reviewed, committed Markdown documents that may participate in the Clinical-JEPA continuity pilot.

The public source repository contains no secret-consuming continuity workflow and no cross-repository publisher. A scheduled/dispatch workflow in private `csainsbury/cog-continuity-inbox` polls `refs/heads/docs/rung-minus1-readiness`, reads this committed manifest and regular-blob metadata through the GitHub API, and writes an immutable metadata-only event to its own repository using its repository-scoped `GITHUB_TOKEN`.

`.cog/emit_checkpoint.py` is retained only as a deterministic local reference/test implementation. It reads manifest/document metadata from committed Git objects and never reads document bodies into an event, working-tree diffs, Zellij state or transcripts.
