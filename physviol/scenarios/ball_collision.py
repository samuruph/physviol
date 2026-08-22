"""`ball_collision` -- two spheres roll toward each other and collide.

The only scenario in v0 with **two bodies that both ought to respond**, which is
what `newton3_reaction` and `newton2_mass` need: a violation where only one body
reacts is invisible unless a lawful reaction is the obvious alternative.
Grounded in LikePhys *Ball Collision*.
"""
from __future__ import annotations

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class BallCollision(Scenario):
    name = "ball_collision"
    SEG_FLOOR, SEG_A, SEG_B = 1, 2, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        # The two balls are deliberately IDENTICAL -- same radius, same colour.
        # `newton2_mass` stages a collision whose outcome would be lawful for
        # masses in some ratio k:1, and the only thing that makes it a violation
        # is that nothing in the image justifies that ratio. Give the balls
        # different sizes and "the big one is heavier" becomes a perfectly good
        # reading, and the family stops testing anything.
        radius = float(rng.uniform(0.28, 0.36))
        r_a = r_b = radius
        hue = float(rng.uniform(0, 1))
        speed = float(rng.uniform(2.0, 2.6))
        flight = tier.num_frames / float(tier.fps)
        # Meet at ~45% of the clip: late enough that the lawful approach is
        # established, early enough that the aftermath is on screen.
        half_gap = speed * 0.45 * flight

        def roller(name, sign, radius, seg):
            vx = -sign * speed
            return BodySpec(
                name=name, kind="sphere",
                position=(sign * (half_gap + radius), 0.0, radius),
                scale=(radius,) * 3, velocity=(vx, 0.0, 0.0),
                # Rolling without slipping, so the approach looks like rolling
                # rather than sliding: omega_y = vx / r.
                angular_velocity=(0.0, vx / radius, 0.0),
                mass=1.0, friction=0.05, restitution=0.75,
                color=C.hue_rgb(hue),
                segmentation_id=seg, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR),
                    roller("ball_a", -1.0, r_a, self.SEG_A),
                    roller("ball_b", +1.0, r_b, self.SEG_B)],
            lights=C.lights(cx, look_at=(0, 0, 0.4)),
            camera_position=(0.6, -6.4, 1.8), camera_look_at=(0.0, 0.0, 0.4),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"radius_a": r_a, "radius_b": r_b, "speed": speed,
                   "identical_actors": True})


register(BallCollision())
