"""Recipe / seed REGISTRY state machine (Pi 2nd-pass Phase 1).

Two ORTHOGONAL state variables so retiring seeds never loses the scientific outcome (Pi 2nd-pass #5):
  * recipe lifecycle:  REGISTERED -> ASSIGNED -> EVALUATED   (+ immutable outcome CERTIFIED | REFUTED)
  * seed lifecycle:    UNASSIGNED -> ASSIGNED -> RETIRED

Governed-T4 authorization requires ``outcome == CERTIFIED``, seeds RETIRED, and the exact artifact /
recipe / evaluator / mechanism / calibration identities pinned. A REFUTED recipe ALSO retires its
seeds (a sealed seed is spent once, pass or fail — reuse is refused). ``recipe_hash`` is recomputed
from the spec; a recipe-supplied hash is never trusted.

This is an in-memory, safe-public ledger. It opens NO real sealed split and populates NO policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical_jepa.eval.oracle_contracts import (
    FittedRecipeArtifact, RecipeSpec, SplitAssignment,
)

# recipe lifecycle
RECIPE_REGISTERED = "REGISTERED"
RECIPE_ASSIGNED = "ASSIGNED"
RECIPE_EVALUATED = "EVALUATED"
_RECIPE_ORDER = (RECIPE_REGISTERED, RECIPE_ASSIGNED, RECIPE_EVALUATED)
# immutable outcomes (set only at EVALUATED)
OUTCOME_CERTIFIED = "CERTIFIED"
OUTCOME_REFUTED = "REFUTED"
# seed lifecycle
SEED_UNASSIGNED = "UNASSIGNED"
SEED_ASSIGNED = "ASSIGNED"
SEED_RETIRED = "RETIRED"


class RegistryError(RuntimeError):
    """Raised on an illegal transition, an unknown recipe, or a sealed-seed reuse attempt."""


@dataclass
class _RecipeRecord:
    recipe_hash: str
    spec: RecipeSpec
    state: str = RECIPE_REGISTERED
    outcome: str | None = None
    assignment: SplitAssignment | None = None
    artifact: FittedRecipeArtifact | None = None
    evaluator_identity: str | None = None
    mechanism_hash: str | None = None
    calibration_hash: str | None = None


class OracleRegistry:
    def __init__(self) -> None:
        self._recipes: dict[str, _RecipeRecord] = {}
        self._seed_state: dict[str, str] = {}

    # ---- recipe registration (identity recomputed, not trusted) ----
    def register(self, spec: RecipeSpec, *, claimed_hash: str | None = None) -> str:
        rh = spec.recipe_hash()                     # recompute; never trust a supplied string
        if claimed_hash is not None and claimed_hash != rh:
            raise RegistryError("claimed recipe_hash does not match the recomputed spec hash")
        if rh in self._recipes:
            raise RegistryError(f"recipe {rh[:12]} already registered")
        self._recipes[rh] = _RecipeRecord(recipe_hash=rh, spec=spec)
        return rh

    def recipe_state(self, recipe_hash: str) -> str:
        return self._require(recipe_hash).state

    def recipe_outcome(self, recipe_hash: str) -> str | None:
        return self._require(recipe_hash).outcome

    def seed_state(self, seed_id: str) -> str:
        return self._seed_state.get(seed_id, SEED_UNASSIGNED)

    # ---- assignment: recipe REGISTERED->ASSIGNED, seeds UNASSIGNED->ASSIGNED (reuse refused) ----
    def assign(self, recipe_hash: str, assignment: SplitAssignment) -> None:
        rec = self._require(recipe_hash)
        if rec.state != RECIPE_REGISTERED:
            raise RegistryError(f"cannot assign recipe in state {rec.state}")
        reused = [s for s in assignment.seed_ids if self.seed_state(s) != SEED_UNASSIGNED]
        if reused:
            raise RegistryError(f"sealed-seed reuse refused: {reused}")   # spent seeds never reused
        if not assignment.seed_ids:
            raise RegistryError("assignment must bind at least one seed")
        for s in assignment.seed_ids:
            self._seed_state[s] = SEED_ASSIGNED
        rec.assignment = assignment
        rec.state = RECIPE_ASSIGNED

    # ---- evaluation: recipe ASSIGNED->EVALUATED, immutable outcome, seeds ASSIGNED->RETIRED ----
    def record_outcome(self, recipe_hash: str, outcome: str, artifact: FittedRecipeArtifact, *,
                       evaluator_identity: str, mechanism_hash: str, calibration_hash: str) -> None:
        rec = self._require(recipe_hash)
        if rec.state != RECIPE_ASSIGNED:
            raise RegistryError(f"cannot evaluate recipe in state {rec.state}")
        if outcome not in (OUTCOME_CERTIFIED, OUTCOME_REFUTED):
            raise RegistryError(f"illegal outcome {outcome!r}")
        if artifact.originating_recipe_hash != recipe_hash:
            raise RegistryError("artifact does not originate from this recipe")
        rec.outcome = outcome                      # immutable from here (state advances to EVALUATED)
        rec.artifact = artifact
        rec.evaluator_identity = evaluator_identity
        rec.mechanism_hash = mechanism_hash
        rec.calibration_hash = calibration_hash
        rec.state = RECIPE_EVALUATED
        for s in (rec.assignment.seed_ids if rec.assignment else ()):   # spend seeds pass OR fail
            self._seed_state[s] = SEED_RETIRED

    # ---- authorization readiness (does NOT itself unlock governed T4) ----
    def authorization_ready(self, recipe_hash: str) -> bool:
        rec = self._recipes.get(recipe_hash)
        if rec is None or rec.state != RECIPE_EVALUATED or rec.outcome != OUTCOME_CERTIFIED:
            return False
        if rec.assignment is None or not all(self.seed_state(s) == SEED_RETIRED
                                             for s in rec.assignment.seed_ids):
            return False
        return all(bool(x) for x in (rec.artifact, rec.evaluator_identity, rec.mechanism_hash,
                                     rec.calibration_hash))

    def _require(self, recipe_hash: str) -> _RecipeRecord:
        rec = self._recipes.get(recipe_hash)
        if rec is None:
            raise RegistryError(f"unknown recipe {recipe_hash[:12]}")
        return rec
