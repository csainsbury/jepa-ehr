"""Rung-1 frozen-decode-ceiling CONTRACT — the single source of truth (Pi R7/R8).

Everything the governed run must FREEZE before touching dev lives here so it can be
content-hashed and fail-hard tested. No torch import (pure-python + numpy) so it is cheap
to import and test everywhere.

Frozen by Pi R8 (do not change without re-gating):
  * absolute gates: exact-order >=0.70, exact-count >=0.80, temporal-slot >=0.70;
    KS-D upper-95%-CI <=0.05; normalized-CRPS-skill lower-95%-CI >=0.05;
  * count/order require the readout's OWN paired excess lower-CI > 0.10 on top of the
    absolute gate (Pi R7/R8 Q3 — a weak M1 can never veto a nonlinear M2);
  * d_time = 8 (primary; NO dev-side 4/8 search); temporal-slot M=4 primary, M=8 sensitivity;
  * cluster-based adequacy: order/count >=500 clusters, timing >=500 clusters + >=1000
    inter-event intervals; below floor => NOT_EVALUABLE, adequate-but-imprecise => INCONCLUSIVE;
  * precision sim: nominal bootstrap-CI coverage >=0.95 AND power >=0.80 to certify
    KS-upper-CI<=0.05 at the calibrated alternative D*=0.025, else NOT_EVALUABLE / raise floor;
  * information scope is part of the verdict (Pi R8 #1): arm-A order is ALWAYS
    STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY, arm-A timing at best CONTENT_PROXY_DECODABLE,
    TAP timing DIRECT_TIMING_CEILING_DECODABLE, temporal-slot at best COARSE_SLOT_DECODABLE;
    oracle-assisted companions NEVER drive nomination;
  * matched-head parameter budget (fixed formula, not a bottleneck choice);
  * deterministic wrong-instance swap = seeded derangement, patient-disjoint, same cell, no self;
  * TEST is inaccessible in this run (no confirm-test path anywhere).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

CONTRACT_VERSION = "rung1-contract-v1-pi-r8"

# ---- absolute gates -------------------------------------------------------
EXACT_ORDER_GATE = 0.70          # unconditional exact ordered-sequence reconstruction
EXACT_COUNT_GATE = 0.80          # exact-count incl. count-0 owned by the empty falsifier
SLOT_GATE = 0.70                 # temporal-slot: exact ALL-slot multiset reconstruction
EXCESS_MARGIN = 0.10             # count/order readout's own paired excess lower-CI must exceed this
KS_D_GATE = 0.05                 # gate on the UPPER-95%-CI of KS-D (skeptic-favouring)
CRPS_SKILL_GATE = 0.05           # normalized CRPS skill 1 - CRPS_cond/CRPS_marg, lower-95%-CI

# ---- panel geometry (frozen; no dev-side search) --------------------------
D_TIME = 8                       # TAP time-feature width (primary; Pi R8 #4: no 4/8 search)
M_PRIMARY = 4                    # temporal-slot primary granularity
M_SENSITIVITY = 8                # temporal-slot sensitivity granularity (cannot rescue M=4)

# ---- cluster-based adequacy floors (Pi R8 #6 / Q4) ------------------------
ORDER_CLUSTER_FLOOR = 500        # >=500 distinct patient/sequence clusters with N>=2
COUNT_CLUSTER_FLOOR = 500        # >=500 non-empty clusters
TIMING_CLUSTER_FLOOR = 500       # >=500 clusters with N>=2 ...
TIMING_INTERVAL_FLOOR = 1000     # ... AND >=1000 inter-event intervals (200 is too small)

# ---- KS precision simulation (Pi R8 #3, falsifiable) ----------------------
PSIM_COVERAGE = 0.95             # nominal bootstrap-CI coverage floor
PSIM_POWER = 0.80                # prob. of certifying KS-upper-CI<=0.05 at the alternative
PSIM_D_STAR = 0.025              # predeclared calibrated alternative population KS distance
PSIM_REPS = 400                  # fixed simulation repetitions
PSIM_SEED = 20260710

# ---- bootstrap / seeds ----------------------------------------------------
N_BOOT = 2000
SEED = 20260523
SWAP_SEED = 20260711

# ---- matched-head parameter budget (Pi R8 #4 — a FORMULA, not a bottleneck)
# Every arm's readout is a 2-layer MLP input_dim -> h_arm -> output_dim. To equalise the
# trained-head parameter (and, to first order, FLOP) budget across arms of different
# input_dim, hold P_BUDGET_FACTOR * D^2 params fixed and solve the hidden width per arm:
#   h_arm = round(P_BUDGET_FACTOR * D^2 / (input_dim + output_dim)),  floored at H_MIN.
# A larger input_dim (temporal-slot M*D) therefore gets a proportionally NARROWER hidden
# layer, so it cannot win the contrast merely by having more decoder capacity.
P_BUDGET_FACTOR = 4.0
H_MIN = 16


def matched_head_hidden(input_dim: int, output_dim: int, embedding_dim: int) -> int:
    """Frozen matched-parameter hidden width for a readout head (Pi R8 #4)."""
    budget = P_BUDGET_FACTOR * float(embedding_dim) ** 2
    return max(H_MIN, int(round(budget / max(1, input_dim + output_dim))))


# ---- information scope (Pi R8 #1) -----------------------------------------
SCOPE_DIRECT = "direct"
SCOPE_CONTENT_PROXY = "content_proxy"
SCOPE_COARSE_SLOT = "coarse_slot"
SCOPE_ORACLE_ASSISTED = "oracle_assisted"
INFORMATION_SCOPES = (SCOPE_DIRECT, SCOPE_CONTENT_PROXY, SCOPE_COARSE_SLOT, SCOPE_ORACLE_ASSISTED)

# Only DIRECT and COARSE_SLOT scopes may ever drive a Rung-2 nomination. Content-proxy and
# oracle-assisted findings are measurable but never nominate (Pi R8 #1).
NOMINATING_SCOPES = frozenset({SCOPE_DIRECT, SCOPE_COARSE_SLOT})


def can_nominate(scope: str) -> bool:
    return scope in NOMINATING_SCOPES


# ---- readout classification labels (Pi R7 #1 four-way) --------------------
DECODABLE_SIMPLE = "DECODABLE_SIMPLE"          # M1 absolute gate + M1 own-excess pass
DECODABLE_NONLINEAR = "DECODABLE_NONLINEAR"    # M2 gate + M2's OWN excess pass (regardless of M1)
PRIOR_MASKED = "PRIOR_MASKED"                  # M2 raw gate pass but M2 excess/corruption fails
NOT_DECODABLE = "NOT_DECODABLE"                # neither absolute gate clears, precision adequate
INCONCLUSIVE = "INCONCLUSIVE"                  # adequate cell but CI too wide to decide
NOT_EVALUABLE = "NOT_EVALUABLE"                # below the cluster/interval floor

# ---- scoped arm/property verdict labels (Pi R8 #1) ------------------------
STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY = "STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY"
CONTENT_PROXY_DECODABLE = "CONTENT_PROXY_DECODABLE"
DIRECT_TIMING_CEILING_DECODABLE = "DIRECT_TIMING_CEILING_DECODABLE"
COARSE_SLOT_DECODABLE = "COARSE_SLOT_DECODABLE"

# ---- arm registry ---------------------------------------------------------
# extra_dim adds to the embedding_dim D. "rung" 1a = incumbent (never nominates);
# 1b = target panel (nominate-only). Order/timing scope is fixed per arm so a
# permutation-invariant / timing-free arm can never claim a DIRECT order/timing ceiling.
ARMS: dict[str, dict[str, Any]] = {
    "mean_embed": {"rung": "1a", "extra_dim": 0, "slots": 1,
                   "scope": {"marginals": SCOPE_DIRECT, "count": SCOPE_DIRECT,
                             "order": SCOPE_CONTENT_PROXY, "timing": SCOPE_CONTENT_PROXY}},
    "tap_concat": {"rung": "1b", "extra_dim": D_TIME, "slots": 1,
                   "scope": {"marginals": SCOPE_DIRECT, "count": SCOPE_DIRECT,
                             "order": SCOPE_CONTENT_PROXY, "timing": SCOPE_DIRECT}},
    "count_concat": {"rung": "1b", "extra_dim": 1, "slots": 1,
                     "scope": {"marginals": SCOPE_DIRECT, "count": SCOPE_DIRECT,
                               "order": SCOPE_CONTENT_PROXY, "timing": SCOPE_CONTENT_PROXY}},
    "temporal_slot": {"rung": "1b", "extra_dim": 0, "slots": M_PRIMARY,
                      "scope": {"marginals": SCOPE_DIRECT, "count": SCOPE_DIRECT,
                                "order": SCOPE_COARSE_SLOT, "timing": SCOPE_COARSE_SLOT}},
}
PROPERTIES = ("marginals", "count", "order", "timing")
PRIMARY_ARM = "mean_embed"

# Frozen hierarchical arm-comparison order for the 1b nomination multiplicity control
# (Pi R7 #6 / R8 #4). Arms are considered in this fixed order; no best-of-panel selection.
ARM_COMPARISON_ORDER = ("tap_concat", "count_concat", "temporal_slot")


def input_dim(arm: str, embedding_dim: int) -> int:
    a = ARMS[arm]
    return int(a["slots"]) * int(embedding_dim) + int(a["extra_dim"])


def arm_scope(arm: str, prop: str) -> str:
    return ARMS[arm]["scope"][prop]


def direct_order_forbidden(arm: str) -> bool:
    """Arms that are permutation-invariant beyond a (multiset/slot) may NEVER produce a
    direct-order-decodable verdict, whatever the raw exact-order score (Pi R8 #1/#5)."""
    return arm_scope(arm, "order") != SCOPE_DIRECT


# ---- MIMIC 2 d is a sensitivity horizon, never a primary gate (Pi R6/R7) --
SENSITIVITY_HORIZONS = {"MIMIC": (2.0,)}


def is_primary_cell(source: str, window_days: float) -> bool:
    return float(window_days) not in {float(w) for w in SENSITIVITY_HORIZONS.get(source, ())}


# ---- config hashing (Pi R8 #4: a hashed pre-run config) -------------------
def frozen_contract() -> dict[str, Any]:
    """The frozen scalar contract (excludes per-source horizons, which live in the run
    config JSON and are hashed together with this)."""
    return {
        "contract_version": CONTRACT_VERSION,
        "gates": {"exact_order": EXACT_ORDER_GATE, "exact_count": EXACT_COUNT_GATE,
                  "slot": SLOT_GATE, "excess_margin": EXCESS_MARGIN, "ks_d": KS_D_GATE,
                  "crps_skill": CRPS_SKILL_GATE},
        "panel": {"d_time": D_TIME, "m_primary": M_PRIMARY, "m_sensitivity": M_SENSITIVITY,
                  "arms": {k: {kk: v[kk] for kk in ("rung", "extra_dim", "slots")} for k, v in ARMS.items()},
                  "comparison_order": list(ARM_COMPARISON_ORDER)},
        "adequacy": {"order_clusters": ORDER_CLUSTER_FLOOR, "count_clusters": COUNT_CLUSTER_FLOOR,
                     "timing_clusters": TIMING_CLUSTER_FLOOR, "timing_intervals": TIMING_INTERVAL_FLOOR},
        "precision_sim": {"coverage": PSIM_COVERAGE, "power": PSIM_POWER, "d_star": PSIM_D_STAR,
                          "reps": PSIM_REPS, "seed": PSIM_SEED},
        "head_budget": {"factor": P_BUDGET_FACTOR, "h_min": H_MIN},
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "swap_seed": SWAP_SEED},
        "test_access": False,
    }


def config_hash(run_config: dict[str, Any] | None = None) -> str:
    """Content hash of the frozen contract + the per-run config (horizons/floors/arms).
    Recorded in every manifest so a governed run's pre-registration is auditable."""
    payload = {"contract": frozen_contract(), "run_config": run_config or {}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---- deterministic wrong-instance swap (Pi R8 #4) -------------------------
def deterministic_derangement(patients: list[str], *, seed: int = SWAP_SEED) -> np.ndarray:
    """A fixed-seed swap partner index for each row such that partner != self AND
    partner's patient != self's patient (patient-disjoint, no self-pair). Callers restrict
    `patients` to a single matching cell so the swap stays on-manifold. Returns -1 where no
    patient-disjoint partner exists (row excluded from the swap-floor)."""
    n = len(patients)
    pats = np.asarray(patients, dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    partner = np.full(n, -1, dtype=np.int64)
    # greedy on a fixed permutation: pair i with the next row of a different patient
    for a in range(n):
        i = int(order[a])
        # candidate partners = a rotation of the fixed order, first patient-disjoint hit
        for step in range(1, n):
            j = int(order[(a + step) % n])
            if pats[j] != pats[i]:
                partner[i] = j
                break
    return partner


_DECODABLE = frozenset({DECODABLE_SIMPLE, DECODABLE_NONLINEAR})


def scoped_verdict(arm: str, prop: str, base_class: str, *, rung: str | None = None) -> dict[str, Any]:
    """Map a base readout classification to the SCOPED arm/property verdict + information
    scope + nomination eligibility (Pi R8 #1). The scope is fixed per (arm, prop) so a
    permutation-invariant / timing-free arm can never claim a DIRECT order/timing ceiling,
    whatever its raw score. Only rung-1b arms with a DIRECT or COARSE_SLOT scope may nominate;
    content-proxy and oracle-assisted findings never nominate."""
    if arm not in ARMS or prop not in PROPERTIES:
        raise ValueError(f"unknown arm/prop {arm}/{prop}")
    rung = rung or ARMS[arm]["rung"]
    scope = arm_scope(arm, prop)
    decodable = base_class in _DECODABLE
    passthrough = base_class in (NOT_EVALUABLE, INCONCLUSIVE)  # keep evaluability status verbatim

    if prop == "order":
        if scope == SCOPE_COARSE_SLOT:      # temporal-slot: gated on slot-wise structure
            label = COARSE_SLOT_DECODABLE if decodable else base_class
        else:                                # order-blind arms: forced content-prior-only
            label = base_class if passthrough else STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY
    elif prop == "timing":
        if scope == SCOPE_DIRECT:
            label = DIRECT_TIMING_CEILING_DECODABLE if decodable else base_class
        elif scope == SCOPE_COARSE_SLOT:
            label = COARSE_SLOT_DECODABLE if decodable else base_class
        else:
            label = CONTENT_PROXY_DECODABLE if (decodable and not passthrough) else base_class
    else:                                    # count / marginals: direct property
        label = base_class

    nominate = bool(decodable and rung == "1b" and can_nominate(scope))
    if prop == "order" and direct_order_forbidden(arm) and scope == SCOPE_DIRECT:
        raise AssertionError(f"contract breach: {arm} order scope must not be direct")
    if nominate and not can_nominate(scope):  # belt-and-braces
        raise AssertionError(f"contract breach: nomination from non-nominating scope {scope}")
    return {"verdict": label, "information_scope": scope, "can_nominate": nominate,
            "base_class": base_class, "arm": arm, "property": prop, "rung": rung}


def classify_readout(
    *, m1_gate_ok: bool, m1_excess_lo: float,
    m2_gate_ok: bool, m2_excess_lo: float, m2_copy_ok: bool,
    evaluable: bool, precise: bool, margin: float = EXCESS_MARGIN,
) -> str:
    """Per-readout four-way classification (Pi R7 #1). M2 is judged by M2's OWN excess and
    copy-floor, never vetoed by a weak M1. Order of precedence: evaluability -> a decodable
    verdict -> prior-masked -> not-decodable / inconclusive."""
    if not evaluable:
        return NOT_EVALUABLE
    m1_ok = bool(m1_gate_ok and m1_excess_lo > margin)
    m2_ok = bool(m2_gate_ok and m2_excess_lo > margin and m2_copy_ok)
    if m2_ok:
        return DECODABLE_NONLINEAR          # M2 earns it on its own excess, regardless of M1
    if m1_ok:
        return DECODABLE_SIMPLE
    if m2_gate_ok and not (m2_excess_lo > margin and m2_copy_ok):
        return PRIOR_MASKED                 # raw gate met but the decoder read its prior/copied
    if not precise:
        return INCONCLUSIVE
    return NOT_DECODABLE
