"""`barrier_pass` -- a ball rolls into a solid wall and bounces back.

Exists so `solidity` has a scenario where passing through is the whole point.
It was previously staged on `occluder_pass`, where the only surface available
was the floor -- so the violation sank the ball into the ground *while it was
hidden behind the screen*, and the clip read as the ball vanishing. That is
`permanence`'s job, and it is on `occluder_pass` too, so the two families were
producing the same picture under different labels.

Splitting them rather than reshaping `occluder_pass` keeps the thing that makes
that scenario worth having: its screen is the only source of observability lag
in the dataset, and a screen the ball can hit is a screen the ball never gets
behind.

The valid clip is a lawful rebound, which matters more here than usual. "A ball
passes through a wall" is only legible as a violation if the same ball, in the
same scene, is shown bouncing off it.
"""
from __future__ import annotations

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class BarrierPass(Scenario):
    name = "barrier_pass"
    SEG_FLOOR, SEG_BALL, SEG_WALL, SEG_SPLIT = 1, 2, 3, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        radius = float(rng.uniform(0.24, 0.32))
        thickness = float(rng.uniform(0.07, 0.11))
        wall_x = float(rng.uniform(1.05, 1.35))
        wall_h = float(rng.uniform(0.62, 0.85))
        flight = tier.num_frames / float(tier.fps)

        # Solve the approach so the impact lands just under halfway through the
        # clip: early enough that the rebound (or the pass-through) has room to
        # play out, late enough that the lawful approach is established first.
        speed = float(rng.uniform(2.6, 3.2))
        gap = speed * 0.45 * flight
        x0 = wall_x - thickness - radius - gap

        ball = BodySpec(
            name="ball", kind="sphere", position=(x0, 0.0, radius),
            scale=(radius,) * 3, velocity=(speed, 0.0, 0.0),
            # Rolling rather than sliding, and elastic enough that the lawful
            # rebound is unmistakable -- a ball that hits a wall and stops dead
            # makes a pass-through look like the more sensible of the two.
            angular_velocity=(0.0, speed / radius, 0.0),
            mass=1.0, friction=0.05, restitution=0.78,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BALL, role="actor")
        wall = BodySpec(
            name="wall", kind="cube",
            position=(wall_x, 0.0, wall_h), scale=(thickness, 1.15, wall_h),
            mass=0.0, static=True, friction=0.4, restitution=0.75,
            color=(0.30, 0.31, 0.37), segmentation_id=self.SEG_WALL,
            role="occluder")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), wall, ball,
                    C.understudy(ball, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.5)),
            camera_position=(0.1, -6.4, 1.7), camera_look_at=(0.25, 0.0, 0.55),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"radius": radius, "speed": speed, "wall_x": wall_x,
                   "wall_id": self.SEG_WALL,
                   "family_targets": {"solidity": [self.SEG_BALL]}})


register(BarrierPass())
