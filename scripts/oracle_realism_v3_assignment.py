#!/usr/bin/env python3
"""Oracle realism v3 — COUNTER-ADDRESSABLE permutation-assignment law (Pi rev-22 ruling 2).

The rev-22 engine drew every permutation mask from ONE sequential `default_rng(seed)` stream, so replicate `j`
could not be regenerated without replaying `1..j-1`. That makes the whole-block split / checkpoint / resume plan
(§9.1) impossible to implement, which is why Pi ruled that the assignment RNG must become counter-addressable.

The benchmark's first proposal — `sha256(namespace | group | experiment | replicate_index)` — was rejected for
two reasons, both fixed here:

  * it OMITTED the issued assignment root seed, so issuance would not actually select the assignment stream;
  * it hashed an ambiguous string concatenation rather than structured, domain-separated data.

The law implemented here derives each mask from a CANONICAL STRUCTURED payload:

    {law_version, assignment_root_seed_identity, registry_variant_identity, group_id, experiment_id,
     replicate_index}

hashed with the repository's canonical hash and consumed at its FULL digest width (no truncation). Consequences:

  * replicate `j` is addressable directly — arbitrary block order, split execution and resume are bit-for-bit
    identical to a monolithic run;
  * `j = 0` is the DETERMINISTIC OBSERVED SPLIT and consumes no random draw at all;
  * for each `(experiment, j)` exactly ONE mask is generated and reused by every cell of that experiment, while
    experiments are addressed independently under the shared MC index `j`;
  * IID-with-replacement semantics are unchanged, so DUPLICATE assignments remain valid and are never refused.

**Assignment provenance is NOT fixture provenance.** The draft RNG manifest covers fixture/coupling generation;
this law needs its own manifest section, built separately (`oracle_realism_v3_manifest.build_draft_assignment_
manifest`). The issued assignment ROOT SEED stays `RESERVED_ASSIGNMENT_ROOT_NOT_ISSUED` — seed issuance is
blocked — so every registered derivation refuses, and development work uses an explicitly dev-labelled root.

Development-only. Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/oracle_realism_v3_assignment.py
"""
from __future__ import annotations

import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from scripts.oracle_realism_v3_randomization import RefusalError, _canonical_mask, _perm_mask, _exp_total

ASSIGNMENT_LAW_VERSION = "v3-assignment-law-1"

# The issued registered root seed is NOT drawn (seed issuance is blocked); registered derivation refuses on it.
RESERVED_ASSIGNMENT_ROOT = "RESERVED_ASSIGNMENT_ROOT_NOT_ISSUED"
DEV_ASSIGNMENT_ROOT = "DEV-ASSIGNMENT-ROOT-v3-not-an-issued-seed"

# `j = 0` is the observed split, not a random replicate.
OBSERVED_INDEX = 0


def _require_nonempty_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise RefusalError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_index(j, b_total):
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
        raise RefusalError(f"replicate_index must be a non-bool integer, got {j!r}")
    j = int(j)
    if j < OBSERVED_INDEX or j > b_total:
        raise RefusalError(f"replicate_index {j} out of range [{OBSERVED_INDEX}, {b_total}]")
    return j


def assignment_seed_payload(*, root_seed_identity, registry_variant_identity, group_id, experiment_id,
                            replicate_index, b_total):
    """The exact structured, domain-separated payload the assignment seed is derived from (Pi rev-22 #2)."""
    _require_nonempty_str(root_seed_identity, "assignment_root_seed_identity")
    _require_nonempty_str(registry_variant_identity, "registry_variant_identity")
    _require_nonempty_str(group_id, "group_id")
    _require_nonempty_str(experiment_id, "experiment_id")
    j = _require_index(replicate_index, b_total)
    if j == OBSERVED_INDEX:
        raise RefusalError("replicate_index 0 is the deterministic observed split and consumes no random draw")
    return {
        "law_version": ASSIGNMENT_LAW_VERSION,
        "assignment_root_seed_identity": root_seed_identity,
        "registry_variant_identity": registry_variant_identity,
        "group_id": group_id,
        "experiment_id": experiment_id,
        "replicate_index": j,
    }


def assignment_seed(**kw):
    """Full-width seed: the canonical hash of the structured payload, consumed at its FULL digest width."""
    if kw.get("root_seed_identity") == RESERVED_ASSIGNMENT_ROOT:
        raise RefusalError("registered assignment root seed is RESERVED / not issued — derivation refused")
    return int(canonical_hash(assignment_seed_payload(**kw)), 16)


def assignment_mask(strata, *, root_seed_identity, registry_variant_identity, group_id, experiment_id,
                    replicate_index, b_total):
    """The mask for one `(experiment, replicate_index)`. `j = 0` is the canonical observed split (no draw)."""
    j = _require_index(replicate_index, b_total)
    if j == OBSERVED_INDEX:
        return _canonical_mask(strata)
    seed = assignment_seed(root_seed_identity=root_seed_identity,
                           registry_variant_identity=registry_variant_identity,
                           group_id=group_id, experiment_id=experiment_id,
                           replicate_index=j, b_total=b_total)
    return _perm_mask(np.random.default_rng(seed), strata)


def assignment_law_identity(*, root_seed_identity, registry_variant_identity):
    """Seed-INDEPENDENT law identity: the derivation contract, not any particular replicate."""
    return canonical_hash({
        "law_version": ASSIGNMENT_LAW_VERSION,
        "assignment_root_seed_identity": root_seed_identity,
        "registry_variant_identity": registry_variant_identity,
        "payload_fields": ["law_version", "assignment_root_seed_identity", "registry_variant_identity",
                           "group_id", "experiment_id", "replicate_index"],
        "digest_width": "full canonical_hash digest, no truncation",
        "observed_index_rule": "j=0 is the deterministic canonical split and consumes no random draw",
        "per_experiment_rule": "one mask per (experiment, j), reused by every cell of that experiment; "
                               "experiments are addressed independently under the shared MC index j",
        "replacement_rule": "IID with replacement; duplicate assignments are VALID and never refused",
        "stratum_rule": "within-stratum label permutation preserving each stratum's (n_candidate, n_reference)",
    })


def block_masks(strata, *, first, last, **kw):
    """Masks for an inclusive replicate range — the checkpoint-block unit. Directly addressable, so a shard can
    compute `[first..last]` without touching any other block."""
    if last < first:
        raise RefusalError(f"empty replicate range [{first}, {last}]")
    return {j: assignment_mask(strata, replicate_index=j, **kw) for j in range(first, last + 1)}


# ---------------------------------------------------------------------------------------------------
# self-tests — Pi rev-22 #2's required list
# ---------------------------------------------------------------------------------------------------
def selftest():
    errs = []
    strata = [(6, 6), (4, 4)]
    B = 40
    base = dict(root_seed_identity=DEV_ASSIGNMENT_ROOT, registry_variant_identity="REGVAR-with_exemption",
                group_id="G_full_burst_timing", experiment_id="null_scid", b_total=B)

    def mask(j, **over):
        return assignment_mask(strata, **{**base, **over}, replicate_index=j)

    def key(m):
        return tuple(int(i) for i in np.where(m)[0])

    # 1. deterministic replay
    if key(mask(7)) != key(mask(7)):
        errs.append("assignment is not deterministic for the same (root, variant, group, exp, j)")

    # 2. monolithic == arbitrary block order == split, bit-for-bit
    monolithic = {j: key(mask(j)) for j in range(1, B + 1)}
    shuffled_order = list(range(1, B + 1))
    np.random.default_rng(5).shuffle(shuffled_order)
    out_of_order = {j: key(mask(j)) for j in shuffled_order}
    if monolithic != out_of_order:
        errs.append("arbitrary block order does not reproduce monolithic assignments")
    split = {}
    for lo, hi in ((1, 10), (11, 25), (26, B)):
        split.update({j: key(m) for j, m in block_masks(strata, first=lo, last=hi, **base).items()})
    if monolithic != split:
        errs.append("split execution does not reproduce monolithic assignments bit-for-bit")
    # resume: recompute only a middle block
    resumed = dict(monolithic)
    resumed.update({j: key(m) for j, m in block_masks(strata, first=11, last=25, **base).items()})
    if resumed != monolithic:
        errs.append("resume of a middle block is not bit-for-bit identical")

    # 3. exact index coverage across a block partition
    covered = sorted(split)
    if covered != list(range(1, B + 1)):
        errs.append(f"block partition covers {covered[:3]}..{covered[-3:]}, expected exactly 1..{B}")

    # 4. domain separation: root / registry variant / group / experiment / index each change the mask
    variations = {
        "root": mask(7, root_seed_identity=DEV_ASSIGNMENT_ROOT + "-other"),
        "registry_variant": mask(7, registry_variant_identity="REGVAR-without_exemption"),
        "group": mask(7, group_id="G_full_phase_seam"),
        "experiment": mask(7, experiment_id="null_mimic"),
        "index": mask(8),
    }
    for name, m in variations.items():
        if key(m) == key(mask(7)):
            errs.append(f"no domain separation on {name}: the mask is unchanged")

    # 5. quota preservation within every stratum, for every replicate
    for j in range(0, B + 1):
        m = mask(j)
        off = 0
        for (nA, nB) in strata:
            seg = m[off:off + nA + nB]
            if int(seg.sum()) != nA or seg.shape[0] != nA + nB:
                errs.append(f"replicate {j} violates stratum quota {(nA, nB)}")
                break
            off += nA + nB
        if m.shape[0] != _exp_total(strata):
            errs.append(f"replicate {j} mask length {m.shape[0]} != {_exp_total(strata)}")

    # 6. j=0 is the canonical observed split and consumes NO draw
    if key(mask(0)) != key(_canonical_mask(strata)):
        errs.append("replicate 0 is not the canonical observed split")
    try:
        assignment_seed(**base, replicate_index=0)
        errs.append("assignment_seed(j=0) returned a seed; j=0 must consume no random draw")
    except RefusalError:
        pass

    # 7. duplicates are VALID (IID with replacement) — never refused, and they do occur on a small support
    small = [(2, 2)]
    seen = [key(assignment_mask(small, **{**base, "b_total": 200}, replicate_index=j)) for j in range(1, 201)]
    if len(set(seen)) == len(seen):
        errs.append("no duplicate assignment observed on a 6-assignment support — replacement semantics suspect")

    # 8. malformed / out-of-range indices refuse
    def refused(fn, label):
        try:
            fn()
        except RefusalError:
            return
        errs.append(f"accepted but must refuse: {label}")

    refused(lambda: mask(-1), "negative replicate index")
    refused(lambda: mask(B + 1), "replicate index beyond B")
    refused(lambda: mask(True), "bool replicate index")
    refused(lambda: mask(3.0), "float replicate index")
    refused(lambda: mask(7, group_id=""), "empty group id")
    refused(lambda: mask(7, experiment_id="   "), "blank experiment id")
    refused(lambda: mask(7, root_seed_identity=""), "empty assignment root")
    refused(lambda: assignment_seed(**{**base, "root_seed_identity": RESERVED_ASSIGNMENT_ROOT},
                                    replicate_index=7), "RESERVED (unissued) registered root seed")
    refused(lambda: block_masks(strata, first=10, last=9, **base), "inverted replicate range")

    # law identity is seed-independent but root/variant sensitive
    a = assignment_law_identity(root_seed_identity=DEV_ASSIGNMENT_ROOT, registry_variant_identity="V1")
    if a != assignment_law_identity(root_seed_identity=DEV_ASSIGNMENT_ROOT, registry_variant_identity="V1"):
        errs.append("assignment law identity is not deterministic")
    if a == assignment_law_identity(root_seed_identity=DEV_ASSIGNMENT_ROOT, registry_variant_identity="V2"):
        errs.append("assignment law identity ignores the registry variant")
    if a == assignment_law_identity(root_seed_identity=RESERVED_ASSIGNMENT_ROOT, registry_variant_identity="V1"):
        errs.append("assignment law identity ignores the assignment root")
    return errs


def main():
    errs = selftest()
    out = {
        "module": "oracle_realism_v3_assignment",
        "law_version": ASSIGNMENT_LAW_VERSION,
        "purpose": ("counter-addressable permutation-assignment law (Pi rev-22 ruling 2): replicate j is derived "
                    "directly from a canonical STRUCTURED, domain-separated payload including the ISSUED "
                    "assignment root seed identity, so split / arbitrary-order / resume execution is bit-for-bit "
                    "identical to a monolithic run and the whole-block checkpoint plan becomes implementable."),
        "payload_fields": ["law_version", "assignment_root_seed_identity", "registry_variant_identity",
                           "group_id", "experiment_id", "replicate_index"],
        "digest": "full canonical_hash digest width, no truncation",
        "observed_index": OBSERVED_INDEX,
        "registered_root_seed": RESERVED_ASSIGNMENT_ROOT,
        "dev_law_identity": assignment_law_identity(root_seed_identity=DEV_ASSIGNMENT_ROOT,
                                                    registry_variant_identity="dev"),
        "provenance_note": ("assignment provenance is NOT fixture provenance — the draft RNG manifest covers "
                            "fixture/coupling generation; this law carries its own manifest section."),
        "authorization": ("DEV-ONLY. The registered assignment root seed is RESERVED and every derivation "
                          "against it REFUSES; no seed issuance, no registered execution."),
        "selftests_pass": not errs,
        "selftest_errors": errs,
    }
    print(json.dumps(out, indent=2, default=str))
    print("\nselftests_pass:", json.dumps(not errs))
    for e in errs:
        print("  SELFTEST ERROR:", e)
    return out


if __name__ == "__main__":
    main()
