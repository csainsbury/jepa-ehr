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


from clinical_jepa.eval.oracle_meta_gen import KAPPA_TRAIN_GRID

NO_CALIBRATION = ""            # empty => registry authorization_ready() is False (Pi #9: no real calibration)


@dataclass(frozen=True)
class VerdictRecord:
    verdict: CandidateVerdict
    recipe_hash: str
    artifact_hash: str
    registry_outcome: str            # CERTIFIED | REFUTED (registry state)
    synthetic_registry_complete: bool  # outcome CERTIFIED + seeds retired + identities (NOT gov auth)
    authorization_ready: bool        # requires a REAL approved calibration hash -> False in this stage


def _split_assignment(seed_tag: str) -> SplitAssignment:
    """Registry-OWNED assignment: train/held-out family membership, disjoint, with UNIQUE seed IDs."""
    seeds = tuple(f"seed::{seed_tag}::{i}" for i in range(4))
    return SplitAssignment(train=TRAIN_FAMILIES, dev=("dev::" + TRAIN_FAMILIES[0],),
                           sealed_cert=HELDOUT_FAMILIES, family_ids=(*TRAIN_FAMILIES, *HELDOUT_FAMILIES),
                           seed_ids=seeds)


def _validate_assignment(a: SplitAssignment) -> None:
    if len(set(a.seed_ids)) != len(a.seed_ids):
        raise RuntimeError("duplicate seed IDs within the assignment")
    if set(a.train) & set(a.sealed_cert):
        raise RuntimeError("train / held-out (sealed_cert) families are not disjoint")
    if not (set(a.train) <= set(a.family_ids) and set(a.sealed_cert) <= set(a.family_ids)):
        raise RuntimeError("assignment train/held-out not covered by family_ids")


def certify_recipe(recipe_factory: Callable, *, seed: int = 0,
                   registry: REG.OracleRegistry | None = None,
                   presented_sampler_fingerprint: str | None = None) -> VerdictRecord:
    """Full registry-integrated candidate certification. The recipe fits ONLY from a registry-owned
    data loader; contamination is refused from that loader's EXTERNAL access trace (not recipe-reported
    metadata). A capability violation (label read) or sampler mismatch is refused."""
    from clinical_jepa.eval.oracle_contracts import CapabilityError
    from clinical_jepa.eval.oracle_meta_gen import L_ITEMS
    from clinical_jepa.eval.oracle_meta_recipe import RegistryDataLoader, sampler_fingerprint, validate_bit_budget
    reg = registry or REG.OracleRegistry()
    recipe = recipe_factory()
    validate_bit_budget(recipe, L_ITEMS)                        # fail-closed bit accounting (Pi #7)
    registered_fp = sampler_fingerprint(recipe)
    if presented_sampler_fingerprint is not None and presented_sampler_fingerprint != registered_fp:
        raise RuntimeError("sampler fingerprint mismatch — refusing to score with a non-registered sampler")
    assignment = _split_assignment(str(seed))
    _validate_assignment(assignment)                             # dup seeds / disjointness (Pi #9)
    rh = reg.register(recipe.spec())                             # identity recomputed by the registry
    reg.assign(rh, assignment)                                  # registry-owned split + seed ledger
    loader = RegistryDataLoader(assignment, seed=seed)          # the ONLY training data source (Pi #1)
    try:
        recipe.fit(loader)                                     # fit ONCE via the loader's capability views
    except CapabilityError:
        raise RuntimeError("capability violation during fit — recipe reached for a non-context channel")
    # EXTERNAL contamination guard: the loader's access trace must be train-only (not recipe-reported).
    tr = loader.access_trace()
    if set(tr["families"]) & set(HELDOUT_FAMILIES) or not set(tr["kappas"]) <= set(KAPPA_TRAIN_GRID):
        raise RuntimeError("loader access trace touched a held-out family / non-train κ")
    artifact = recipe.artifact()
    identities = {
        "recipe_hash": rh, "artifact_hash": artifact.artifact_hash,
        "mechanism_hash": invariant_hash(), "evaluator_identity": recipe.spec().evaluator_identity,
        "sampler_fingerprint": registered_fp, "bit_accounting": recipe.spec().bit_accounting,
        "split_assignment_hash": assignment.assignment_hash(), "seed_ids": list(assignment.seed_ids),
        "access_trace": tr,
    }
    try:
        unlock = compute_unlock(recipe, seed=seed, identities=identities)
    except CapabilityError:
        raise RuntimeError("capability violation during scoring — recipe reached for a non-context channel")
    verdict = certify_from_unlock(unlock)
    outcome = REG.OUTCOME_CERTIFIED if verdict.outcome == CERTIFIED_CANDIDATE else REG.OUTCOME_REFUTED
    reg.record_outcome(rh, outcome, artifact, evaluator_identity=identities["evaluator_identity"],
                       mechanism_hash=identities["mechanism_hash"], calibration_hash=NO_CALIBRATION)
    complete = outcome == REG.OUTCOME_CERTIFIED and all(reg.seed_state(s) == REG.SEED_RETIRED
                                                        for s in assignment.seed_ids)
    return VerdictRecord(verdict, rh, artifact.artifact_hash, outcome, complete, reg.authorization_ready(rh))
