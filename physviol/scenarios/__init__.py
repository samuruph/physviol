"""Scenario registry. Importing this module registers every scenario."""
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, DEFAULT_TIER,  # noqa: F401
                   BodySpec, Complexity, LightSpec, Scenario, SceneSpec, Tier,
                   TIERS, available, get, implemented_complexities, register)
from . import (ball_collision, ball_drop, barrier_pass,  # noqa: F401
               granular_pour,
               occluder_pass, pendulum_swing, projectile_toss, pyramid_impact,
               ramp_slide, resting_table, rolling_ramp, shadow_track,
               spin_toss, stack_topple)
