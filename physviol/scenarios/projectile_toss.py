"""`projectile_toss` -- a body thrown on a ballistic arc.

Pure free flight, no contact for the whole clip. This is the **clean control**
of docs/PLAN.md Part 2: nothing occludes the actor, so `t_event` and
`t_observable` coincide, and every occluded scenario's observability lag is
measured against it.
"""
from __future__ import annotations

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class ProjectileToss(Scenario):
    name = "projectile_toss"
    SEG_FLOOR, SEG_BALL = 1, 2

    def sample(self, seed: int, tier: Tier,
               complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        radius = float(rng.uniform(0.30, 0.45))
        # Aim the arc so the apex lands mid-clip and the body is still airborne
        # at the last frame -- free flight must cover the whole window.
        flight = tier.num_frames / float(tier.fps)
        vz = float(9.81 * flight * rng.uniform(0.55, 0.70))
        vx = float(rng.uniform(1.6, 2.8) * rng.choice([-1.0, 1.0]))
        z0 = float(rng.uniform(0.6, 1.1))

        hdri_id = pick_hdri(rng) if cx.background == "hdri" else None
        ball = BodySpec(
            name="ball", kind="sphere", position=(-vx * flight * 0.45, 0.0, z0),
            scale=(radius,) * 3, velocity=(vx, 0.0, vz), mass=1.0,
            friction=0.4, restitution=0.6, color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BALL, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), ball],
            lights=C.lights(cx, look_at=(0, 0, 1.2)),
            camera_position=(0.4, -9.0, 2.6), camera_look_at=(0.0, 0.0, 1.8),
            floor_level=0.0, complexity=complexity, hdri_id=hdri_id,
            notes={"radius": radius, "v0": [vx, 0.0, vz],
                   "flight_seconds": flight})


register(ProjectileToss())
