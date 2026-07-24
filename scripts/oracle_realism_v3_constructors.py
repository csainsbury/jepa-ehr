#!/usr/bin/env python3
"""Oracle realism v3 — CANONICAL STRUCTURED CONTROL CONSTRUCTORS (boundary-short; Pi rev-8 #8).

The registry declares three exchangeability strata for the boundary-short experiment
(`len_0 / len_1 / len_2`, allocation `CONTROL_ALLOC = (2667, 2667, 2666)`, "permute WITHIN each length
stratum"), and the RNG manifest binds its constructor route as `bounded_length_control`. Nothing implemented
that route. Every draw site routed boundary-short through the GENERIC single-profile fixture path
(`sample_fixture("MIMIC", PROFILES["boundary_short"], N, seed)`), which yields ONE pooled homogeneous sample.

Two consequences, both reproduced by this module's self-tests:

  * the declared `len_i` strata were FICTITIOUS — a pooled `uniform_int[1,7]` draw carries no length
    stratification, so labelling its first 2667 records `len_0` binds a design identity the data does not have
    (contrast `structural_zero`, whose `_multiscale` constructor genuinely creates three length strata);
  * `G_bounded_support` was NOT RUNNABLE. The engine's `_assemble_arms` correctly REFUSES the generic path's
    single-stratum arms against the canonical three — fail-closed, but it means the bounded group has been
    wired (rev-19) without any constructor able to feed it.

This module supplies the missing executable route: a genuinely length-stratified bounded control whose strata
are disjoint length bands partitioning the canonical support `L in [1,7]`, drawn with the SAME `uniform_int`
family and varying the per-stratum bounds — exactly the shape of `_multiscale`, which keeps the length family
and varies the per-stratum location.

**The band partition forces a design choice that is NOT ours to make**, so BOTH coherent variants are
implemented, tested and reported, and the caller must name one (there is no default):

  * `registered_alloc` — keeps the registered `CONTROL_ALLOC (2667, 2667, 2666)`. Because the bands have widths
    2 / 2 / 3, equal-ish allocation over unequal widths CHANGES the pooled marginal length law from uniform on
    {1..7} to a band mixture (P(L=1) = 0.1667 vs 0.1429 uniform).
  * `width_proportional` — allocates in proportion to band width, which restores the uniform {1..7} marginal up
    to integer-allocation rounding (EXACT iff `n_total` is a multiple of 7; at the registered N=8000 the residual
    is 2.4e-05, i.e. 1300× smaller than `registered_alloc`'s), but changes the registered stratum allocation to
    (2286, 2286, 3428).

Either way the STRUCTURAL bound `L <= 7` is preserved, so the boundary-short guarantee that no 8-item block can
form (S9 seam checks NE by construction) still holds — that is asserted, not assumed.

DEVELOPMENT ONLY. This implements and identity-binds the constructor route; it does NOT perform the registered
bounded-control draw, which stays RESERVED along with the map-set and RNG manifest. No calibration/evaluation
seed, no policy, no persisted artifact.

Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/oracle_realism_v3_constructors.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture, FIXTURE_IMPL
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_battery import CONTROL_ALLOC, REGISTERED_N

import scripts.oracle_realism_v3_engine as ENG
from scripts.oracle_realism_v3_engine import RefusalError

DEV_NS = "v3-constructor-dev"          # development namespace; NOT a calibration/evaluation/map-design namespace

BOUNDED_PROFILE = "boundary_short"
BOUNDED_SKELETON = "MIMIC"             # must agree with ENG._PROFILE_SKELETON

# The canonical partition of the bounded support L in [1,7] into three ORDERED, DISJOINT, COVERING length bands.
# Widths 2 / 2 / 3 — the coarsest partition into three contiguous non-empty bands that keeps the short band
# (L in {1,2}, where the length-conditioned checks are most degenerate) separate from the long band.
BOUNDED_BANDS = ((1, 2), (3, 4), (5, 7))
BOUNDED_STRATUM_IDS = ("len_0", "len_1", "len_2")

ALLOC_VARIANTS = ("registered_alloc", "width_proportional")

# Canonical constructor route ids — MUST agree with the RNG manifest's `_constructor_route` (self-tested).
CANONICAL_ROUTE = {
    "structural_zero_control": "structural_multiscale",
    "boundary_short": "bounded_length_control",
    "scid_scale_control": "source_profile_fixture",
    "mimic_scale_control": "source_profile_fixture",
}


def _derive_seed(*parts):
    """Same derivation law as the v2 battery's `_derive_seed` (mirrored, not re-invented)."""
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def dseed(*parts):
    return _derive_seed(DEV_NS, *parts)


# ---------------------------------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------------------------------
def band_widths(bands=BOUNDED_BANDS):
    return tuple(hi - lo + 1 for lo, hi in bands)


def bounded_allocation(variant, n_total=REGISTERED_N, bands=BOUNDED_BANDS):
    """Per-stratum counts summing EXACTLY to n_total. Fail-closed on an unknown variant — there is no default."""
    if variant not in ALLOC_VARIANTS:
        raise RefusalError(f"unknown bounded allocation variant {variant!r}; expected one of {ALLOC_VARIANTS}")
    if isinstance(n_total, bool) or not isinstance(n_total, int) or n_total <= 0:
        raise RefusalError(f"n_total {n_total!r} must be a positive non-bool int")
    if variant == "registered_alloc":
        if n_total == REGISTERED_N:
            alloc = tuple(int(a) for a in CONTROL_ALLOC)
        else:                                            # dev-scaled, same equal-ish shape, exact sum
            base, rem = divmod(n_total, len(bands))
            alloc = tuple(base + (1 if i < rem else 0) for i in range(len(bands)))
    else:
        widths = band_widths(bands); total_w = sum(widths)
        exact = [n_total * w / total_w for w in widths]
        floors = [int(np.floor(x)) for x in exact]
        # deterministic largest-remainder assignment of the shortfall (ties broken by band order)
        short = n_total - sum(floors)
        order = sorted(range(len(bands)), key=lambda i: (-(exact[i] - floors[i]), i))
        for i in order[:short]:
            floors[i] += 1
        alloc = tuple(floors)
    if sum(alloc) != n_total or any(a <= 0 for a in alloc):
        raise RefusalError(f"allocation {alloc} does not partition {n_total} into positive strata")
    return alloc


def pooled_length_pmf(variant, n_total=REGISTERED_N, bands=BOUNDED_BANDS):
    """The pooled marginal P(L = l) the constructor induces — the quantity the variant choice trades off."""
    alloc = bounded_allocation(variant, n_total, bands)
    pmf = {}
    for (lo, hi), a in zip(bands, alloc):
        for l in range(lo, hi + 1):
            pmf[l] = pmf.get(l, 0.0) + (a / n_total) / (hi - lo + 1)
    return {l: pmf[l] for l in sorted(pmf)}


def uniform_reference_pmf(bands=BOUNDED_BANDS):
    lo, hi = bands[0][0], bands[-1][1]
    return {l: 1.0 / (hi - lo + 1) for l in range(lo, hi + 1)}


# ---------------------------------------------------------------------------------------------------
# the constructor
# ---------------------------------------------------------------------------------------------------
def _band_profile(lo, hi):
    """The canonical bounded profile restricted to one length band — SAME family, per-stratum bounds only."""
    base = PROFILES[BOUNDED_PROFILE]
    if base["length"]["family"] != "uniform_int":
        raise RefusalError(f"{BOUNDED_PROFILE} length family {base['length']['family']!r} is not uniform_int")
    return {**base, "length": {**base["length"], "min": int(lo), "max": int(hi)}}


def build_bounded_length_control_strata(variant, tag, seed, *, n_total=REGISTERED_N, bands=BOUNDED_BANDS):
    """The CANONICAL `bounded_length_control` route: one `uniform_int` draw per disjoint length band.

    Returns {stratum_id: [records]}. The per-stratum seed derivation mirrors `_multiscale`'s
    (`_derive_seed(tag, seed, i)`), so the strata are independent and the whole draw is reproducible from
    (tag, seed)."""
    alloc = bounded_allocation(variant, n_total, bands)
    if len(bands) != len(BOUNDED_STRATUM_IDS):
        raise RefusalError(f"{len(bands)} bands but {len(BOUNDED_STRATUM_IDS)} canonical stratum ids")
    out = {}
    for i, ((lo, hi), a) in enumerate(zip(bands, alloc)):
        out[BOUNDED_STRATUM_IDS[i]] = sample_fixture(
            BOUNDED_SKELETON, _band_profile(lo, hi), int(a), seed=_derive_seed(tag, seed, i))
    return out


def bounded_control_arms(variant, exp_id, trial, *, n_per_arm=REGISTERED_N, bands=BOUNDED_BANDS):
    """{stratum_id: {candidate, reference}} for the boundary-short experiment, conformant with the canonical
    group's stratum ids and order. `exp_id` enters every seed (Pi rev-6 #1b: no cross-experiment reuse)."""
    cand = build_bounded_length_control_strata(
        variant, f"bsc|{exp_id}", int(dseed("bs_cand", variant, exp_id, trial)), n_total=n_per_arm, bands=bands)
    ref = build_bounded_length_control_strata(
        variant, f"bsr|{exp_id}", int(dseed("bs_ref", variant, exp_id, trial)), n_total=n_per_arm, bands=bands)
    return {sid: {"candidate": list(cand[sid]), "reference": list(ref[sid])} for sid in BOUNDED_STRATUM_IDS}


# ---------------------------------------------------------------------------------------------------
# validation (fail-closed; the constructor's guarantees are asserted, never assumed)
# ---------------------------------------------------------------------------------------------------
def validate_bounded_strata(strata, variant, *, n_total=REGISTERED_N, bands=BOUNDED_BANDS):
    """REFUSE unless: canonical stratum ids in order; exact per-stratum counts; every record's length inside ITS
    OWN band; the global structural bound L <= max(band) preserved; bands disjoint, ordered and covering."""
    if list(strata) != list(BOUNDED_STRATUM_IDS):
        raise RefusalError(f"stratum ids/order {list(strata)} != canonical {list(BOUNDED_STRATUM_IDS)}")
    prev_hi = None
    for lo, hi in bands:
        if lo > hi or (prev_hi is not None and lo != prev_hi + 1):
            raise RefusalError(f"bands {bands} are not ordered, disjoint and contiguous")
        prev_hi = hi
    alloc = bounded_allocation(variant, n_total, bands)
    hard_max = bands[-1][1]
    for i, sid in enumerate(BOUNDED_STRATUM_IDS):
        recs = strata[sid]; lo, hi = bands[i]
        if len(recs) != alloc[i]:
            raise RefusalError(f"stratum {sid} has {len(recs)} records, registered quota {alloc[i]}")
        if not recs:
            raise RefusalError(f"stratum {sid} is empty")
        L = np.array([int(r.L_total) for r in recs])
        if L.min() < lo or L.max() > hi:
            raise RefusalError(f"stratum {sid} lengths [{L.min()},{L.max()}] escape its band [{lo},{hi}]")
        if L.max() > hard_max:
            raise RefusalError(f"stratum {sid} violates the structural bound L<={hard_max}")
    return True


def constructor_route_identity(variant, *, n_total=REGISTERED_N, bands=BOUNDED_BANDS):
    """Deterministic identity for the bounded route: route id + bands + allocation + the exact per-band profile
    payloads + the fixture implementation identity. Seed-independent (it identifies the ROUTE, not a draw)."""
    return canonical_hash({
        "route": CANONICAL_ROUTE[BOUNDED_PROFILE],
        "profile": BOUNDED_PROFILE, "skeleton": BOUNDED_SKELETON,
        "bands": [list(b) for b in bands], "stratum_ids": list(BOUNDED_STRATUM_IDS),
        "allocation_variant": variant, "allocation": list(bounded_allocation(variant, n_total, bands)),
        "n_total": n_total,
        "band_profiles": [_band_profile(lo, hi) for lo, hi in bands],
        "seed_law": "per-stratum _derive_seed(tag, seed, band_index); tag carries the experiment id",
        "fixture_impl": FIXTURE_IMPL,
        "structural_bound": f"L<={bands[-1][1]} preserved => no 8-item block => S9 seam checks NE by construction",
    })


# The registered bounded-control DRAW stays RESERVED, exactly like the map-set and the RNG manifest.
RESERVED_REGISTERED_BOUNDED_DRAW = "RESERVED_BOUNDED_CONTROL_NOT_DRAWN"


# ---------------------------------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------------------------------
def selftest():
    errs = []
    n_dev = 600                                     # dev-scaled; the registered draw stays reserved

    # 0. the route ids must agree with what the RNG manifest binds
    try:
        import scripts.oracle_realism_v3_manifest as MAN
        for src, route in CANONICAL_ROUTE.items():
            if MAN._constructor_route(src) != route:
                errs.append(f"route disagreement for {src}: manifest {MAN._constructor_route(src)!r} != {route!r}")
    except ImportError as ex:                        # pragma: no cover
        errs.append(f"cannot import manifest to cross-check routes: {ex}")
    if ENG._PROFILE_SKELETON[BOUNDED_PROFILE] != BOUNDED_SKELETON:
        errs.append(f"skeleton disagreement: engine {ENG._PROFILE_SKELETON[BOUNDED_PROFILE]!r} != {BOUNDED_SKELETON!r}")

    # 1. allocations partition exactly; the registered variant reproduces CONTROL_ALLOC at the registered N
    if bounded_allocation("registered_alloc", REGISTERED_N) != tuple(int(a) for a in CONTROL_ALLOC):
        errs.append("registered_alloc does not reproduce CONTROL_ALLOC at the registered N")
    for v in ALLOC_VARIANTS:
        for n in (REGISTERED_N, n_dev, 601):
            if sum(bounded_allocation(v, n)) != n:
                errs.append(f"{v} allocation does not sum to {n}")

    # 2. the marginal-law claim, stated honestly: width_proportional restores uniformity only up to INTEGER
    #    allocation rounding — exactly iff n_total is a multiple of the support size (7).
    ref = uniform_reference_pmf()
    support = len(ref)
    wp = pooled_length_pmf("width_proportional", REGISTERED_N)
    ra = pooled_length_pmf("registered_alloc", REGISTERED_N)
    wp_dev = max(abs(wp[l] - ref[l]) for l in ref)
    ra_dev = max(abs(ra[l] - ref[l]) for l in ref)
    if wp_dev > 1.0 / REGISTERED_N:                        # rounding bound: at most one record per band
        errs.append(f"width_proportional deviates from uniform by {wp_dev} > 1/N rounding bound")
    exact_n = support * (REGISTERED_N // support)          # 7994: a multiple of the support size
    wp_exact = pooled_length_pmf("width_proportional", exact_n)
    if max(abs(wp_exact[l] - ref[l]) for l in ref) > 1e-12:
        errs.append(f"width_proportional is not EXACT at n_total={exact_n} (a multiple of {support})")
    if ra_dev <= wp_dev:
        errs.append(f"the variant trade-off claim is wrong: registered_alloc deviation {ra_dev} "
                    f"<= width_proportional {wp_dev}")

    # 3. both variants: strata validate, bands hold, structural bound holds, S9 degeneracy holds
    for v in ALLOC_VARIANTS:
        arms = bounded_control_arms(v, "boundary_short", 0, n_per_arm=n_dev)
        for role in ("candidate", "reference"):
            strata = {sid: arms[sid][role] for sid in BOUNDED_STRATUM_IDS}
            try:
                validate_bounded_strata(strata, v, n_total=n_dev)
            except RefusalError as ex:
                errs.append(f"{v}/{role} failed validation: {ex}")
            allL = np.concatenate([[int(r.L_total) for r in strata[sid]] for sid in BOUNDED_STRATUM_IDS])
            if allL.max() > BOUNDED_BANDS[-1][1]:
                errs.append(f"{v}/{role} escapes the structural bound (max L={allL.max()})")
            if allL.max() >= 8:
                errs.append(f"{v}/{role} can form an 8-item block — the S9 NE guarantee is broken")

    # 4. the arms now ASSEMBLE through the engine for the canonical bounded group (they did not before)
    canon = ENG.CANONICAL_GROUPS["G_bounded_support"]
    for v in ALLOC_VARIANTS:
        arms = bounded_control_arms(v, "boundary_short", 1, n_per_arm=n_dev)
        try:
            exps = ENG._assemble_arms({"boundary_short": arms}, canon, exact_counts=False)
            got = [s["stratum_id"] if isinstance(s, dict) else s
                   for s in canon["experiments"]["boundary_short"]["stratum_ids"]]
            if list(arms) != list(got):
                errs.append(f"{v}: assembled stratum order {list(arms)} != canonical {got}")
            n_asm = sum(a + b for a, b in exps["boundary_short"]["strata"])
            if n_asm != 2 * n_dev:
                errs.append(f"{v}: assembled pooled size {n_asm} != {2 * n_dev}")
        except RefusalError as ex:
            errs.append(f"{v}: canonical bounded arms REFUSED by the engine: {ex}")

    # 5. the OLD generic single-profile path must still be refused (the defect this module fixes)
    generic = {"len_0": {"candidate": list(sample_fixture(BOUNDED_SKELETON, PROFILES[BOUNDED_PROFILE], n_dev,
                                                          seed=dseed("generic", "c"))),
                         "reference": list(sample_fixture(BOUNDED_SKELETON, PROFILES[BOUNDED_PROFILE], n_dev,
                                                          seed=dseed("generic", "r")))}}
    try:
        ENG._assemble_arms({"boundary_short": generic}, canon, exact_counts=False)
        errs.append("the generic single-stratum bounded path was ACCEPTED — it must be refused")
    except RefusalError:
        pass

    # 6. adversarial refusals
    def refused(fn, label):
        try:
            fn()
        except RefusalError:
            return
        errs.append(f"adversarial case ACCEPTED but must refuse: {label}")

    refused(lambda: bounded_allocation("no_such_variant", n_dev), "unknown allocation variant")
    refused(lambda: bounded_allocation("registered_alloc", 0), "non-positive n_total")
    refused(lambda: bounded_allocation("registered_alloc", True), "bool n_total")
    good = bounded_control_arms("registered_alloc", "boundary_short", 2, n_per_arm=n_dev)
    good_strata = {sid: good[sid]["candidate"] for sid in BOUNDED_STRATUM_IDS}
    refused(lambda: validate_bounded_strata({k: v for k, v in list(good_strata.items())[:2]},
                                            "registered_alloc", n_total=n_dev), "missing stratum")
    refused(lambda: validate_bounded_strata({**good_strata, "len_0": good_strata["len_0"][:-1]},
                                            "registered_alloc", n_total=n_dev), "short stratum count")
    swapped = {**good_strata, "len_0": good_strata["len_2"], "len_2": good_strata["len_0"]}
    refused(lambda: validate_bounded_strata(swapped, "registered_alloc", n_total=n_dev),
            "records outside their own band (swapped strata)")
    refused(lambda: validate_bounded_strata(good_strata, "registered_alloc", n_total=n_dev,
                                            bands=((1, 2), (4, 5), (6, 7))), "non-contiguous bands")

    # 7. reproducibility: same (variant, exp, trial) => identical draw; different exp => different draw
    a1 = bounded_control_arms("registered_alloc", "boundary_short", 3, n_per_arm=200)
    a2 = bounded_control_arms("registered_alloc", "boundary_short", 3, n_per_arm=200)
    def h(arms):
        return canonical_hash({sid: {ro: [[int(r.L_total), r.class_ids.tolist(),
                                           np.asarray(r.timestamps).tolist()] for r in arms[sid][ro]]
                                     for ro in ("candidate", "reference")} for sid in BOUNDED_STRATUM_IDS})
    if h(a1) != h(a2):
        errs.append("constructor is not reproducible for the same (variant, exp, trial)")
    if h(a1) == h(bounded_control_arms("registered_alloc", "other_exp", 3, n_per_arm=200)):
        errs.append("exp_id does not enter the seed derivation (cross-experiment reuse)")
    if h(a1) == h(bounded_control_arms("width_proportional", "boundary_short", 3, n_per_arm=200)):
        errs.append("allocation variant does not change the draw")

    return errs


def main():
    errs = selftest()
    ref = uniform_reference_pmf()
    out = {
        "module": "oracle_realism_v3_constructors",
        "route": CANONICAL_ROUTE[BOUNDED_PROFILE],
        "bands": [list(b) for b in BOUNDED_BANDS],
        "stratum_ids": list(BOUNDED_STRATUM_IDS),
        "defect_addressed": (
            "the registry declared three length strata (len_0/len_1/len_2) and the RNG manifest bound the route "
            "`bounded_length_control`, but every draw site used the GENERIC single-profile fixture path: the "
            "strata were fictitious and the engine correctly REFUSED the resulting single-stratum arms, so "
            "G_bounded_support was not runnable."),
        "variants": {
            v: {
                "allocation_registered_N": list(bounded_allocation(v, REGISTERED_N)),
                "route_identity": constructor_route_identity(v),
                "pooled_length_pmf": {str(k): round(x, 6) for k, x in pooled_length_pmf(v, REGISTERED_N).items()},
                "max_abs_deviation_from_uniform": round(
                    max(abs(pooled_length_pmf(v, REGISTERED_N)[l] - ref[l]) for l in ref), 8),
                "uniform_up_to_integer_rounding": max(
                    abs(pooled_length_pmf(v, REGISTERED_N)[l] - ref[l]) for l in ref) <= 1.0 / REGISTERED_N,
                "preserves_registered_allocation": list(bounded_allocation(v, REGISTERED_N)) == list(CONTROL_ALLOC),
            } for v in ALLOC_VARIANTS
        },
        "decision_required": (
            "band widths are 2/2/3, so a three-stratum bounded control CANNOT simultaneously keep the registered "
            "allocation (2667,2667,2666) and the uniform L~U{1..7} pooled marginal. `registered_alloc` keeps the "
            "allocation and PERTURBS the marginal: P(L)=0.16669 on the short/mid bands and 0.11108 on the long "
            "band vs uniform 0.142857 — max deviation 0.031774, i.e. the long lengths L in {5,6,7} are "
            "under-represented by 22%. `width_proportional` restores uniformity to within integer-allocation "
            "rounding (2.4e-05 at N=8000; exact iff n_total is a multiple of 7) but moves the allocation to "
            "(2286,2286,3428), which would re-mint the registry stratum quotas and the manifest's "
            "stratum_allocation. NEITHER is adopted as the default — the caller must name a variant, and the "
            "choice is a reviewer decision."),
        "registered_draw": RESERVED_REGISTERED_BOUNDED_DRAW,
        "structural_bound": f"L<={BOUNDED_BANDS[-1][1]} preserved by both variants (asserted, not assumed); "
                            f"no 8-item block can form, so the S9 seam checks stay NE by construction",
        "authorization": ("DEV-ONLY constructor implementation + route identity binding. The registered "
                          "bounded-control draw stays RESERVED; no calibration/evaluation seed, no map-set draw, "
                          "no manifest population or freeze, no policy, no persisted artifact."),
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
