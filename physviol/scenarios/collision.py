"""`collision` -- a rolling sphere strikes an identical one at rest.

The only scenario in v0 with **two bodies that both ought to respond**, which is
what `newton3_reaction` and `newton2_mass` need: a violation where only one body
reacts is invisible unless a lawful reaction is the obvious alternative.
Grounded in LikePhys *Ball Collision*.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


#: Hand-composed, and now also the reference the ball speed is derived from.
CAMERA = (0.3, -6.0, 1.7)
LOOK_AT = (0.1, 0.0, 0.35)


class Collision(Scenario):
    name = "collision"
    SEG_FLOOR, SEG_A, SEG_B, SEG_SPLIT = 1, 2, 4, 6

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
        flight = tier.num_frames / float(tier.fps)
        # Derived from the frame, not picked: the balls are all but frictionless
        # (0.05, so they roll rather than scrub) and therefore travel `v*T`
        # whatever we choose. A fixed 2.2-2.8 m/s framed the 13-frame clips it
        # was tuned against and rolled the struck ball out of shot at 25 and 49.
        speed = float(cam.traverse_speed(CAMERA, LOOK_AT, flight,
                                         fraction=float(rng.uniform(0.60, 0.72))))

        # A striker into a body at REST, not two balls closing head-on. Both
        # Newton families hinge on how the struck ball responds, and a target
        # that is already moving makes that unreadable twice over: "it did not
        # react" is indistinguishable from "it stopped dead", which is a
        # different violation, and a target that keeps coming at the striker
        # has to end up sharing space with it. Against a resting target,
        # `newton3` is simply "it never moved" -- no overlap, nothing to
        # misread.
        target_x = float(rng.uniform(0.15, 0.45))
        gap = speed * 0.45 * flight
        striker_x = target_x - r_a - r_b - gap

        striker = BodySpec(
            name="ball_a", kind="sphere",
            position=(striker_x, 0.0, r_a), scale=(r_a,) * 3,
            velocity=(speed, 0.0, 0.0),
            # Rolling without slipping, so the approach looks like rolling
            # rather than sliding: omega_y = vx / r.
            angular_velocity=(0.0, speed / r_a, 0.0),
            mass=1.0, friction=0.05, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_A, role="actor")
        target = BodySpec(
            name="ball_b", kind="sphere",
            position=(target_x, 0.0, r_b), scale=(r_b,) * 3,
            mass=1.0, friction=0.05, restitution=0.75,
            color=C.hue_rgb(hue), segmentation_id=self.SEG_B, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), striker, target,
                    C.understudy(striker, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.4)),
            camera_position=CAMERA, camera_look_at=LOOK_AT,
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(C.appearance_rng(seed)) if cx.background == "hdri" else None,
            notes={"radius_a": r_a, "radius_b": r_b, "speed": speed,
                   "identical_actors": True, "target_at_rest": True,
                   "striker_id": self.SEG_A, "target_id": self.SEG_B})


register(Collision())
