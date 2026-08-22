"""`rolling_ramp` -- a cube tumbles down a raised ramp and off its lip.

Two regimes in one clip, deliberately: sustained contact on the slab, then a
short free flight after the lip. `friction` and `newton1_inertia` fire in the
first; `angular_momentum` needs the second, because a body in contact with the
ground can change its spin lawfully and only a change made in mid-air is
unexplained.

A cube rather than a ball for the same reason `spin_toss` uses one: a uniformly
coloured sphere renders identically however it is spinning.
"""
from __future__ import annotations

import math

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class RollingRamp(Scenario):
    name = "rolling_ramp"
    SEG_FLOOR, SEG_BLOCK, SEG_RAMP = 1, 2, 3

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        tilt = float(rng.uniform(0.44, 0.56))
        half_len, thick = 1.5, 0.08
        centre = (0.0, 0.0, 1.5)
        half = float(rng.uniform(0.17, 0.22))
        mu = float(rng.uniform(0.25, 0.38))
        v0 = float(rng.uniform(1.9, 2.4))
        d, _ = C.ramp_axes(tilt)
        # Start close to the lip so the body leaves the slab around the middle
        # of the clip; starting at the top spends the whole of Tier D sliding
        # and never reaches the airborne stretch angular_momentum needs.
        start_along = half_len - float(rng.uniform(1.1, 1.35))

        block = BodySpec(
            name="block", kind="cube",
            position=C.on_ramp(centre, tilt, start_along, half + thick),
            scale=(half,) * 3,
            quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
            velocity=tuple(v0 * x for x in d),
            mass=1.0, friction=mu, restitution=0.2,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BLOCK, role="actor")

        lip = tuple(centre[i] + half_len * d[i] for i in range(3))
        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    C.ramp(self.SEG_RAMP, tilt, centre, half_len, 0.8, thick, mu),
                    block],
            lights=C.lights(cx, look_at=(0, 0, 1.2)),
            camera_position=(0.2, -7.0, 2.6), camera_look_at=(0.3, 0.0, 1.2),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"tilt_rad": tilt, "mu": mu, "lip": list(lip),
                   "ramp_id": self.SEG_RAMP})


register(RollingRamp())
