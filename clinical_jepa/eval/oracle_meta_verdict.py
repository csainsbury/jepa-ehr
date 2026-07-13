"""Registry-integrated candidate verdict (Pi keystone #4/#7).

The verdict path is driven THROUGH the registry: register the recipe spec, take a registry-owned split
assignment + seed ledger, fit ONCE on the train assignment, compute the typed UnlockEvaluation, and
record the outcome through legal state transitions. The candidate verdict is a pure function of the
UnlockEvaluation. Authorization to governed T4 stays FALSE — that needs the committed oracle policy and
a separate gate; this only records SYNTHETIC-RECOVERY candidacy. Safe-public / synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from clinical_jepa.eval import oracle_registry as REG
from clinical_jepa.eval.oracle_contracts import SplitAssignment
from clinical_jepa.eval.oracle_meta_gen import HELDOUT_FAMILIES, TRAIN_FAMILIES, invariant_hash
from clinical_jepa.eval.oracle_unlock import (
    CERTIFIED_CANDIDATE, REFUTED, CandidateVerdict, certify_from_unlock, compute_unlock,
)


@dataclass(frozen=True)
class VerdictRecord:
    verdict: CandidateVerdict
    recipe_hash: str
    artifact_hash: str
    registry_outcome: str            # CERTIFIED | REFUTED (registry state)
    authorization_ready: bool        # synthetic-recovery readiness ONLY (never governed T4)


def _split_assignment(seed_tag: str) -> SplitAssignment:
    """Registry-OWNED assignment: train/held-out family membership, disjoint, with unique seed IDs."""
    seeds = tuple(f"seed::{seed_tag}::{i}" for i in range(4))
    return SplitAssignment(train=TRAIN_FAMILIES, dev=("dev::" + TRAIN_FAMILIES[0],),
                           sealed_cert=HELDOUT_FAMILIES, family_ids=(*TRAIN_FAMILIES, *HELDOUT_FAMILIES),
                           seed_ids=seeds)


def certify_recipe(recipe_factory: Callable, *, seed: int = 0,
                   registry: REG.OracleRegistry | None = None,
                   presented_sampler_fingerprint: str | None = None) -> VerdictRecord:
    """Full registry-integrated candidate certification of a fit-once recipe. If a sampler fingerprint
    is presented, it MUST match the recipe's registered sampler (a mismatch is refused)."""
    from clinical_jepa.eval.oracle_meta_recipe import sampler_fingerprint
    reg = registry or REG.OracleRegistry()
    recipe = recipe_factory()
    registered_fp = sampler_fingerprint(recipe)
    if presented_sampler_fingerprint is not None and presented_sampler_fingerprint != registered_fp:
        raise RuntimeError("sampler fingerprint mismatch — refusing to score with a non-registered sampler")
    rh = reg.register(recipe.spec())                              # identity recomputed by the registry
    reg.assign(rh, _split_assignment(str(seed)))                 # registry-owned split + seed ledger
    recipe.fit_on_train(seed=seed)                               # fit ONCE on the train assignment
    # provenance guard: the fit touched only TRAIN families (never held-out).
    if set(recipe.fit_provenance["families"]) & set(HELDOUT_FAMILIES):
        raise RuntimeError("recipe fit provenance touched a held-out family")
    artifact = recipe.artifact()
    identities = {
        "recipe_hash": rh, "artifact_hash": artifact.artifact_hash,
        "mechanism_hash": invariant_hash(), "evaluator_identity": recipe.spec().evaluator_identity,
        "sampler_fingerprint": registered_fp,
        "bit_accounting": recipe.spec().bit_accounting,
        "split_assignment_hash": _split_assignment(str(seed)).assignment_hash(),
        "seed_ids": list(_split_assignment(str(seed)).seed_ids),
    }
    unlock = compute_unlock(recipe, seed=seed, identities=identities)
    verdict = certify_from_unlock(unlock)
    outcome = REG.OUTCOME_CERTIFIED if verdict.outcome == CERTIFIED_CANDIDATE else REG.OUTCOME_REFUTED
    reg.record_outcome(rh, outcome, artifact, evaluator_identity=identities["evaluator_identity"],
                       mechanism_hash=identities["mechanism_hash"], calibration_hash="synthetic_no_calibration")
    return VerdictRecord(verdict, rh, artifact.artifact_hash, outcome, reg.authorization_ready(rh))
