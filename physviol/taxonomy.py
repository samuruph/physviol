"""The dataset taxonomy as data — docs/PLAN.md Part 2.

Four levels: DOMAIN -> FAMILY -> SCENARIO -> INSTANCE.

This module is imported by BOTH environments (Python 3.9 inside the Kubric
container and Python 3.11 on the host), so it must stay pure stdlib with no
third-party imports and no 3.10+ syntax.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------
# Level 1 -- domains: which physical law is at stake
# --------------------------------------------------------------------------
DOMAINS: Dict[str, str] = {
    "identity": "does the object persist and stay itself?",
    "kinematics": "is unsupported motion consistent with g?",
    "contact": "do bodies interact legally when they touch?",
    "dynamics": "do forces and masses behave?",
    "equilibrium": "do resting and supported bodies behave?",
    "optical": "is light consistent with geometry?",
    "appearance": "does the object look like itself from frame to frame?",
    "global": "are the scene's constants physical?",
}


class Family(NamedTuple):
    """Level 2 -- a specific way a law breaks."""

    domain: str
    injection: str
    magnitude_unit: str
    kind: str  # instant | sustained | repeated
    law: str  # which residual in residuals/laws.py scores it
    intphys2: Optional[str]  # cross-reference; None is a novelty claim
    likephys: Optional[str]


FAMILIES: Dict[str, Family] = {
    # -- identity ----------------------------------------------------------
    "permanence": Family(
        "identity", "remove a body, ideally while occluded",
        "mass_ratio", "sustained", "mass_continuity", "permanence", "rigid_body"),
    "immutability": Family(
        "identity", "the body grows or shrinks, ideally behind an occluder",
        "volume_ratio", "sustained", "shape_continuity", "immutability", "rigid_body"),
    "fission": Family(
        "identity", "one body becomes two, which fly apart",
        "count_ratio", "sustained", "object_count", "permanence", "rigid_body"),
    "fusion": Family(
        "identity", "bodies converge and come out as one, same size",
        "count_ratio", "sustained", "object_count", "permanence", "rigid_body"),
    # -- kinematics --------------------------------------------------------
    "continuity": Family(
        "kinematics", "discontinuous position set",
        "m_jump_distance", "instant", "position_continuity",
        "spatio_temporal_continuity", "rigid_body"),
    "non_parabolic": Family(
        "kinematics", "replace the free-flight arc with a non-g curve",
        "m_rms_from_parabola", "sustained", "trajectory_shape", None, None),
    "antigravity": Family(
        "kinematics", "per-body gravity scale g -> alpha*g",
        "gravity_scale_deviation", "sustained", "free_fall", None, "rigid_body"),
    "newton1_inertia": Family(
        "kinematics", "a moving body stops dead and stays stopped",
        "dv_over_g_dt", "sustained", "linear_momentum", None, None),
    # -- contact -----------------------------------------------------------
    "solidity": Family(
        "contact", "disable a collision pair for N frames",
        "m_penetration_depth", "sustained", "penetration", "solidity", "rigid_body"),
    "superelastic": Family(
        "contact", "restitution e > 1",
        "energy_gain_ratio", "repeated", "energy_at_contact", None, "rigid_body"),
    "newton3_reaction": Family(
        "contact", "in a collision only one body responds",
        "momentum_imbalance", "instant", "linear_momentum", None, None),
    # -- dynamics ----------------------------------------------------------
    "phantom_impulse": Family(
        "dynamics", "impulse applied with no contact",
        "impulse_over_m_vtyp", "instant", "linear_momentum", None, "rigid_body"),
    "newton2_mass": Family(
        "dynamics", "identical-looking bodies respond differently to equal impulse",
        "effective_mass_ratio", "instant", "linear_momentum", None, None),
    "angular_momentum": Family(
        "dynamics", "spin reverses, or torque appears with no contact",
        "angular_momentum_defect", "instant", "angular_momentum", None, None),
    # -- equilibrium -------------------------------------------------------
    "support": Family(
        "equilibrium", "an unsupported body hovers, or an unstable stack holds",
        "m_support_clearance", "sustained", "support", None, None),
    "friction": Family(
        "equilibrium", "a body accelerates against its motion, or slides forever",
        "effective_mu_ratio", "sustained", "friction", None, None),
    # -- optical -----------------------------------------------------------
    "shadow": Family(
        "optical", "the shadow detaches from its caster and slides away",
        "shadow_caster_offset_radii", "sustained", "shadow_consistency", None, "optical"),
    "shadow_shape": Family(
        "optical", "the shadow stays put but stops matching the caster's shape",
        "shadow_aspect_ratio", "sustained", "shape_anisotropy", None, "optical"),
    # -- appearance --------------------------------------------------------
    "deformation": Family(
        "appearance", "a rigid body squashes or stretches out of proportion",
        "aspect_ratio_change", "sustained", "shape_anisotropy", None, None),
    # -- global ------------------------------------------------------------
    "global_gravity": Family(
        "global", "the whole scene runs at alpha*g, internally consistent",
        "gravity_scale_deviation", "sustained", "free_fall", None, None),
}


class Scenario(NamedTuple):
    """Level 3 -- the staged scene."""

    description: str
    event_structure: str
    has_occluder: bool
    physics_medium: str  # rigid | granular  (never "fluid" at schema v0)
    grounded_in: Optional[str]


SCENARIOS: Dict[str, Scenario] = {
    "ball_drop": Scenario(
        "sphere or cube falls to a floor and bounces",
        "free fall -> contact -> rebound", False, "rigid", "LikePhys Ball Drop"),
    "ball_collision": Scenario(
        "two spheres roll toward each other",
        "approach -> collision -> separation", False, "rigid", "LikePhys Ball Collision"),
    "ramp_slide": Scenario(
        "block slides down an incline",
        "sustained contact + friction", False, "rigid", "LikePhys Block Slide"),
    "projectile_toss": Scenario(
        "body thrown on a ballistic arc",
        "pure free flight, no contact", False, "rigid", None),
    "spin_toss": Scenario(
        "cube tumbling through the air, thrown with heavy spin",
        "free flight with visible rotation", False, "rigid", None),
    "occluder_pass": Scenario(
        "body travels behind a screen and re-emerges",
        "occlusion interval of known length", True, "rigid", "IntPhys 2 occlusion"),
    "barrier_pass": Scenario(
        "ball rolls into a solid wall and bounces back",
        "approach -> rebound off a static barrier", False, "rigid", None),
    "stack_topple": Scenario(
        "stacked bodies, marginally stable",
        "static equilibrium -> topple", False, "rigid", "IntPhys 2 / LikePhys"),
    "pyramid_impact": Scenario(
        "cube dropped onto a sphere pyramid",
        "multi-body contact chain", False, "rigid", "LikePhys Pyramid Impact"),
    "pendulum_swing": Scenario(
        "bob on a rigid rod, swung from a pivot (scripted, not solved)",
        "constrained periodic arc", False, "rigid", "LikePhys Pendulum"),
    "resting_table": Scenario(
        "several bodies at rest on a surface",
        "sustained static equilibrium", False, "rigid", "IntPhys 2 permanence"),
    "rolling_ramp": Scenario(
        "cube tumbles down a raised ramp and off its lip",
        "rolling contact, then a short free flight", False, "rigid", None),
    "shadow_track": Scenario(
        "object translates under a fixed light",
        "a clean, trackable cast shadow", False, "rigid", "LikePhys Moving Shadow"),
    "clutter_toss": Scenario(
        "MOVi-style multi-object toss",
        "dense collisions, heavy occlusion", True, "rigid", "MOVi baseline"),
    "granular_pour": Scenario(
        "a loose column of grains falls into an open box (40 at Tier D, 96 above)",
        "streaming flow, accumulation, break-up", False, "granular",
        "LikePhys Faucet Flow (as granular, not fluid)"),
}

BUILD, DEFER = "build", "defer"

# Level 3 x Level 2 -- which families are meaningful in which scenarios.
# "build" == a v0 deliverable; "defer" == valid but not scheduled.
COMPATIBILITY: Dict[str, Dict[str, str]] = {
    "permanence":       {"ball_drop": BUILD, "occluder_pass": BUILD,
                         "resting_table": BUILD, "stack_topple": BUILD,
                         "granular_pour": BUILD, "shadow_track": BUILD,
                         "barrier_pass": BUILD,
                         "ball_collision": DEFER, "projectile_toss": DEFER,
                         "pendulum_swing": BUILD, "clutter_toss": DEFER},
    "immutability":     {"ball_drop": BUILD, "occluder_pass": BUILD,
                         "projectile_toss": BUILD, "resting_table": BUILD,
                         "shadow_track": BUILD, "barrier_pass": BUILD,
                         "pendulum_swing": BUILD,
                         "ball_collision": DEFER, "clutter_toss": DEFER,
                         "spin_toss": DEFER},
    "fusion":           {"ball_collision": BUILD, "granular_pour": BUILD,
                         "pyramid_impact": DEFER, "stack_topple": DEFER,
                         "clutter_toss": DEFER},
    "fission":          {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "occluder_pass": BUILD, "barrier_pass": BUILD,
                         "ball_collision": DEFER, "spin_toss": DEFER,
                         "clutter_toss": DEFER},
    "continuity":       {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "occluder_pass": BUILD, "ramp_slide": BUILD,
                         "stack_topple": BUILD, "shadow_track": BUILD,
                         "ball_collision": DEFER, "pendulum_swing": DEFER,
                         "resting_table": DEFER, "rolling_ramp": DEFER,
                         "barrier_pass": DEFER, "clutter_toss": DEFER,
                         "granular_pour": DEFER},
    "non_parabolic":    {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "spin_toss": BUILD, "rolling_ramp": BUILD,
                         "occluder_pass": DEFER},
    "antigravity":      {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "granular_pour": BUILD, "ramp_slide": BUILD,
                         "spin_toss": BUILD,
                         "occluder_pass": DEFER, "resting_table": DEFER,
                         "clutter_toss": DEFER},
    # Not `resting_table`: nothing there is moving, so there is nothing to
    # halt. A body there *starting* to move is `phantom_impulse`, which is
    # built on that scenario -- the two used to overlap and now do not.
    "newton1_inertia":  {"ramp_slide": BUILD, "rolling_ramp": BUILD,
                         "resting_table": DEFER, "ball_collision": DEFER},
    # Not on `occluder_pass`: the only surface there is the floor, so the
    # violation sank the ball into the ground while it was hidden behind the
    # screen and the clip read as a vanish -- which is `permanence`, also built
    # on that scenario. `barrier_pass` exists to give it a wall to go through.
    "solidity":         {"ball_drop": BUILD, "ball_collision": BUILD,
                         "barrier_pass": BUILD, "pyramid_impact": BUILD,
                         "ramp_slide": BUILD, "stack_topple": BUILD,
                         "granular_pour": BUILD,
                         "occluder_pass": DEFER, "rolling_ramp": DEFER,
                         "clutter_toss": DEFER},
    "superelastic":     {"ball_drop": BUILD, "ball_collision": BUILD,
                         "pyramid_impact": BUILD, "barrier_pass": BUILD,
                         "granular_pour": BUILD, "clutter_toss": DEFER},
    # Not on `pyramid_impact`: a cube and a sphere are plainly different
    # objects, so "the cube is heavy" is a lawful reading of the clip and there
    # is no violation left for a viewer to see. Both Newton families need two
    # bodies that look the same.
    # Not on `pyramid_impact`, and it took two attempts to be sure. The cube
    # cannot be one of the pair -- next to a sphere, "the cube is heavier" is a
    # lawful reading -- and the spheres, which *are* identical to each other,
    # never strike one another hard enough: they touch from frame 0 and settle
    # rather than collide, so the reaction there is to suppress amounts to
    # nothing and the clips scored 0.01. Measured, not assumed.
    "newton3_reaction": {"ball_collision": BUILD, "pyramid_impact": DEFER,
                         "clutter_toss": DEFER},
    "phantom_impulse":  {"ball_collision": BUILD, "projectile_toss": BUILD,
                         "resting_table": BUILD, "spin_toss": BUILD,
                         "ramp_slide": BUILD, "stack_topple": BUILD,
                         "granular_pour": BUILD, "ball_drop": BUILD,
                         "occluder_pass": DEFER, "pendulum_swing": DEFER,
                         "rolling_ramp": DEFER, "clutter_toss": DEFER},
    "newton2_mass":     {"ball_collision": BUILD, "pyramid_impact": DEFER,
                         "ball_drop": DEFER, "ramp_slide": DEFER},
    "angular_momentum": {"spin_toss": BUILD, "pendulum_swing": BUILD,
                         "rolling_ramp": BUILD, "ball_collision": DEFER,
                         "ramp_slide": DEFER, "projectile_toss": DEFER},
    # On a ramp this is "the block floats just above the slab and stops
    # sliding" -- unsupported and motionless, which is exactly the claim.
    "support":          {"stack_topple": BUILD, "resting_table": BUILD,
                         "ramp_slide": BUILD, "rolling_ramp": BUILD,
                         "pyramid_impact": DEFER},
    "friction":         {"ramp_slide": BUILD, "rolling_ramp": BUILD,
                         "resting_table": DEFER, "granular_pour": DEFER},
    # `shadow` moves the shadow; `shadow_shape` distorts it where it stands.
    # Kept apart on purpose: a benchmark that wants to know whether a model
    # tracks shadow *position* should not be scored on clips where the shadow
    # is also the wrong shape.
    "shadow":           {"shadow_track": BUILD, "ball_drop": DEFER,
                         "projectile_toss": DEFER, "resting_table": DEFER},
    "shadow_shape":     {"shadow_track": BUILD},
    "deformation":      {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "resting_table": BUILD, "occluder_pass": BUILD,
                         "spin_toss": BUILD, "barrier_pass": BUILD,
                         "ramp_slide": BUILD, "stack_topple": BUILD,
                         "pendulum_swing": BUILD, "granular_pour": BUILD,
                         "ball_collision": DEFER, "clutter_toss": DEFER},
    # Only where at least two bodies move. With one object on screen, scaling
    # gravity for the scene and scaling it for that object render identically,
    # so a single-body `global_gravity` cell is an `antigravity` clip with a
    # different label. `plan()` enforces the same rule and returns None below
    # two, so these are not merely unscheduled -- they are unbuildable until
    # the population axis puts more movers in the frame.
    "global_gravity":   {"ball_collision": BUILD, "pyramid_impact": BUILD,
                         "stack_topple": BUILD, "resting_table": BUILD,
                         "granular_pour": BUILD,
                         "ball_drop": DEFER, "projectile_toss": DEFER,
                         "ramp_slide": DEFER, "occluder_pass": DEFER,
                         "pendulum_swing": DEFER, "rolling_ramp": DEFER,
                         "spin_toss": DEFER, "clutter_toss": DEFER},
}

# --------------------------------------------------------------------------
# Orthogonality -- which family is allowed to move which law
# --------------------------------------------------------------------------
# The dataset only supports the claim "this model cannot detect X" if clips
# labelled X do not also contain Y. That is not automatic: an injector that
# re-integrates a body will happily send it through a wall unless something
# stops it, and then a `global_gravity` clip is also a `solidity` clip while
# being scored as neither.
#
# These six laws are the ones with a clean zero baseline on a lawful clip -- a
# body either passed through something or it did not -- which makes them usable
# as tripwires. For each, the families that are *entitled* to move it. Any other
# family moving it is contamination, and `tests/test_orthogonality.py` fails.
#
# The continuous dynamical laws (`free_fall`, `linear_momentum`,
# `energy_at_contact`, `friction`, `support`, `trajectory_shape`,
# `angular_momentum`) are deliberately absent. They co-move by physics -- change
# a body's gravity and its momentum residual moves too -- so demanding
# orthogonality there would be demanding that physics be separable, which it is
# not. Orthogonality is a claim about *staging*, not about mechanics.
EXCLUSIVE_LAWS: Dict[str, Tuple[str, ...]] = {
    "penetration":         ("solidity",),
    "position_continuity": ("continuity",),
    # Not `fusion`: the body that ceases to exist there is the absorbed one,
    # and these tripwires are read on `causal_body_ids[0]`, which for a merge is
    # the survivor. A merge does remove a body -- that is unavoidable, you
    # cannot merge without one -- and what distinguishes it from `permanence` is
    # that the survivor swells to hold both, which `object_count` catches.
    "mass_continuity":     ("permanence",),
    "object_count":        ("fission", "fusion", "permanence"),
    # Neither `fission` nor `fusion`: both keep every surviving body at its
    # own size precisely so the clip is about object count and nothing else.
    # What distinguishes a merge from one body vanishing is the *approach* --
    # the two visibly converge -- not a size change on the survivor.
    "shape_continuity":    ("immutability",),
    "shape_anisotropy":    ("deformation", "shadow_shape"),
}

#: How far an unrelated law may move before it counts as contamination, in body
#: radii. A sub-radius step is not something a viewer can see and not something
#: a detector would fire on; a whole-radius one is a different violation.
ORTHOGONALITY_TOLERANCE = 1.0


SEVERITY_BINS: Tuple[str, ...] = ("weak", "medium", "strong")


def domain_of(family: str) -> str:
    return FAMILIES[family].domain


def build_cells() -> List[Tuple[str, str]]:
    """Every (scenario, family) pair scheduled for v0. The generator's job list."""
    return sorted(
        (scenario, family)
        for family, scen in COMPATIBILITY.items()
        for scenario, status in scen.items()
        if status == BUILD
    )


def is_compatible(scenario: str, family: str, require_build: bool = False) -> bool:
    status = COMPATIBILITY.get(family, {}).get(scenario)
    return status == BUILD if require_build else status in (BUILD, DEFER)


def validate_taxonomy() -> None:
    """Internal consistency. Called by tests and by `physviol validate`."""
    for name, fam in FAMILIES.items():
        assert fam.domain in DOMAINS, "%s: unknown domain %s" % (name, fam.domain)
        assert fam.kind in ("instant", "sustained", "repeated"), name
    for family, scen in COMPATIBILITY.items():
        assert family in FAMILIES, "compatibility for unknown family %s" % family
        for scenario, status in scen.items():
            assert scenario in SCENARIOS, "%s: unknown scenario %s" % (family, scenario)
            assert status in (BUILD, DEFER), "%s/%s: %s" % (family, scenario, status)
    missing = set(FAMILIES) - set(COMPATIBILITY)
    assert not missing, "families with no scenarios: %s" % sorted(missing)
