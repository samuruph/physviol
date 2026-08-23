"""`pyramid_impact` -- a cube dropped onto a pyramid of spheres.

A multi-body contact chain: the impact propagates through four bodies, so
`newton3_reaction` and `newton2_mass` have a collision whose *lawful* outcome is
visibly rich. Grounded in LikePhys *Pyramid Impact*.
"""
from __future__ import annotations

import math

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class PyramidImpact(Scenario):
    name = "pyramid_impact"
    SEG_FLOOR, SEG_CUBE = 1, 2
    SEG_BALLS = (4, 5, 6, 7)

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        r = float(rng.uniform(0.26, 0.32))
        hue = float(rng.uniform(0, 1))
        # Three spheres on an equilateral base, one nested in the dimple above.
        s = r * 1.02
        base_xy = [(-s, -s / math.sqrt(3.0)), (s, -s / math.sqrt(3.0)),
                   (0.0, 2.0 * s / math.sqrt(3.0))]
        apex_z = r + math.sqrt(max((2 * r) ** 2 - (2 * s / math.sqrt(3.0)) ** 2,
                                   1e-4))

        balls = [BodySpec(name="ball_%d" % i, kind="sphere",
                          position=(x, y, r), scale=(r,) * 3, mass=0.8,
                          friction=0.5, restitution=0.35,
                          color=C.hue_rgb((hue + 0.13 * i) % 1.0),
                          segmentation_id=self.SEG_BALLS[i], role="prop")
                 for i, (x, y) in enumerate(base_xy)]
        balls.append(BodySpec(name="ball_apex", kind="sphere",
                              position=(0.0, 0.0, apex_z), scale=(r,) * 3,
                              mass=0.8, friction=0.5, restitution=0.35,
                              color=C.hue_rgb((hue + 0.39) % 1.0),
                              segmentation_id=self.SEG_BALLS[3], role="prop"))

        half = float(rng.uniform(0.24, 0.30))
        drop = apex_z + r + half + float(rng.uniform(0.9, 1.4))
        cube = BodySpec(
            name="cube", kind="cube", position=(0.0, 0.0, drop),
            scale=(half,) * 3, mass=2.2, friction=0.5, restitution=0.2,
            color=C.hue_rgb((hue + 0.5) % 1.0),
            segmentation_id=self.SEG_CUBE, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR)] + balls + [cube],
            lights=C.lights(cx, look_at=(0, 0, 0.5)),
            camera_position=(3.0, -4.6, 2.0), camera_look_at=(0.0, 0.0, 0.6),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"radius": r, "apex_z": apex_z, "drop_height": drop,
                   "pyramid_ids": list(self.SEG_BALLS),
                   # The falling cube is the actor, but it is not the
                   # interesting body to break. What a video generator gets
                   # wrong here is the *scatter*: a struck sphere driven
                   # through the ground or through its neighbour. The apex goes
                   # first because it takes the impact directly.
                   "family_targets": {
                       "solidity": [self.SEG_BALLS[3], self.SEG_BALLS[0],
                                    self.SEG_BALLS[1], self.SEG_BALLS[2]],
                       # The Newton families need two bodies a viewer cannot
                       # tell apart, so they get the spheres and never the
                       # cube -- otherwise "the cube is heavier" is a perfectly
                       # lawful reading of the clip.
                       "newton2_mass": list(self.SEG_BALLS),
                       "newton3_reaction": list(self.SEG_BALLS)}})


register(PyramidImpact())
