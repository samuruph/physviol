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
        "identity", "remove / duplicate a body, ideally while occluded",
        "mass_ratio", "instant", "mass_continuity", "permanence", "rigid_body"),
    "immutability": Family(
        "identity", "the body grows or shrinks, ideally behind an occluder",
        "volume_ratio", "sustained", "shape_continuity", "immutability", "rigid_body"),
    "fission": Family(
        "identity", "one body becomes two, each half the volume",
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
        "kinematics", "a resting body accelerates, or a sliding body stops unforced",
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
        "optical", "the shadow detaches, inverts, vanishes or mismatches its caster",
        "shadow_caster_offset_radii", "sustained", "shadow_consistency", None, "optical"),
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
                         "spin_toss": BUILD, "occluder_pass": DEFER},
    "antigravity":      {"ball_drop": BUILD, "projectile_toss": BUILD,
                         "granular_pour": BUILD, "ramp_slide": BUILD,
                         "spin_toss": BUILD,
                         "occluder_pass": DEFER, "resting_table": DEFER,
                         "clutter_toss": DEFER},
    "newton1_inertia":  {"ramp_slide": BUILD, "resting_table": BUILD, "rolling_ramp": BUILD,
                         "ball_collision": DEFER},
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
    "newton3_reaction": {"ball_collision": BUILD,
                         "pyramid_impact": DEFER, "clutter_toss": DEFER},
    "phantom_impulse":  {"ball_collision": BUILD, "projectile_toss": BUILD,
                         "resting_table": BUILD, "spin_toss": BUILD,
                         "ramp_slide": BUILD, "stack_topple": BUILD,
                         "granular_pour": BUILD, "ball_drop": BUILD,
                         "occluder_pass": DEFER, "pendulum_swing": DEFER,
                         "rolling_ramp": DEFER, "clutter_toss": DEFER},
    "newton2_mass":     {"ball_collision": BUILD,
                         "pyramid_impact": DEFER, "ball_drop": DEFER,
                         "ramp_slide": DEFER},
    "angular_momentum": {"spin_toss": BUILD, "pendulum_swing": BUILD,
                         "rolling_ramp": BUILD, "ball_collision": DEFER,
                         "ramp_slide": DEFER, "projectile_toss": DEFER},
    "support":          {"stack_topple": BUILD, "resting_table": BUILD,
                         "pyramid_impact": DEFER},
    "friction":         {"ramp_slide": BUILD, "rolling_ramp": BUILD,
                         "resting_table": DEFER, "granular_pour": DEFER},
    "shadow":           {"shadow_track": BUILD, "ball_drop": DEFER,
                         "projectile_toss": DEFER, "resting_table": DEFER},
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
