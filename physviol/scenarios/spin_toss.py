"""`spin_toss` -- a cube tumbling through the air, thrown with heavy spin.

Exists because a *sphere* cannot show rotation. Uniformly coloured primitives
are the v0 asset set, so a spinning ball is pixel-identical to a still one and
an `angular_momentum` violation would be perfectly annotated and completely
invisible. A cube's silhouette changes as it turns, which makes the violation
observable in the render and gives `observable_windows` something to find.

Pure free flight for the whole clip: no contact, so the angular-momentum law's
"nothing touched it" gate is satisfied on every frame.
"""
from __future__ import annotations

from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class SpinToss(Scenario):
    name = "spin_toss"
    SEG_FLOOR, SEG_CUBE, SEG_SPLIT = 1, 2, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        half = float(rng.uniform(0.26, 0.34))
        flight = tier.num_frames / float(tier.fps)
        # Apex mid-clip and still airborne at the last frame -- if it lands, the
        # contact would give the spin a lawful way to change and the residual's
        # contact gate would blank exactly the frames we care about.
        vz = float(9.81 * flight * rng.uniform(0.58, 0.70))
        vx = float(rng.uniform(1.4, 2.4) * rng.choice([-1.0, 1.0]))
        spin = (float(rng.uniform(-1.5, 1.5)), float(rng.uniform(3.5, 5.5)),
                float(rng.uniform(-1.5, 1.5)))

        cube = BodySpec(
            name="cube", kind="cube",
            position=(-vx * flight * 0.45, 0.0, float(rng.uniform(0.7, 1.1))),
            scale=(half,) * 3, velocity=(vx, 0.0, vz), angular_velocity=spin,
            mass=1.0, friction=0.4, restitution=0.4,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_CUBE, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), cube,
                    C.understudy(cube, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 1.4)),
            camera_position=(0.3, -7.6, 2.0), camera_look_at=(0.0, 0.0, 1.7),
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"half_extent": half, "spin": list(spin),
                   "v0": [vx, 0.0, vz]})


register(SpinToss())
