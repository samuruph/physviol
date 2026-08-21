"""Scenario registry. Importing this module registers every scenario."""
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, DEFAULT_TIER,  # noqa: F401
                   BodySpec, Complexity, LightSpec, Scenario, SceneSpec, Tier,
                   TIERS, available, get, implemented_complexities, register)
from . import ball_drop, occluder_pass, projectile_toss  # noqa: F401
