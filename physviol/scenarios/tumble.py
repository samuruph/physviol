"""`tumble` -- a cube tumbling through the air, thrown with heavy spin.

Exists because a *sphere* cannot show rotation. Uniformly coloured primitives
are the v0 asset set, so a spinning ball is pixel-identical to a still one and
an `angular_momentum` violation would be perfectly annotated and completely
invisible. A cube's silhouette changes as it turns, which makes the violation
observable in the render and gives `observable_windows` something to find.

Pure free flight for the whole clip: no contact, so the angular-momentum law's
"nothing touched it" gate is satisfied on every frame.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class Tumble(Scenario):
    name = "tumble"
    SEG_FLOOR, SEG_CUBE, SEG_SPLIT = 1, 2, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        # Airborne from the first frame to the last -- if it lands, the contact
        # gives the spin a lawful way to change and the angular-momentum
        # residual's contact gate blanks exactly the frames we care about. The
        # arc's duration is therefore the clip's, and the shot is derived from
        # it rather than hand-tuned; see `camera.frame_flight`.
        flight = tier.num_frames / float(tier.fps)
        f = cam.frame_flight(flight,
                             angular_radius=float(rng.uniform(0.086, 0.100)))
        camera_position, camera_look_at = cam.flight_camera(f)

        half = f.radius
        vx = float(f.x_travel / flight * rng.choice([-1.0, 1.0]))
        x0 = -vx * flight * 0.5
        # Spin rate is a rate, not a length, so it does NOT scale with the
        # scene: a cube turning three times over the flight looks the same
        # whether the flight is two metres or twenty.
        spin = (float(rng.uniform(-1.5, 1.5)), float(rng.uniform(3.5, 5.5)),
                float(rng.uniform(-1.5, 1.5)))

        cube = BodySpec(
            name="cube", kind="cube", position=(x0, 0.0, f.launch_z),
            scale=(half,) * 3, velocity=(vx, 0.0, f.vz), angular_velocity=spin,
            mass=1.0, friction=0.4, restitution=0.4,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_CUBE, role="actor")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR,
                             size=max(6.0, 1.6 * f.half_extent)),
                    cube, C.understudy(cube, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0.0, 0.0, f.look_z * 0.4),
                            scale=f.scene_scale),
            camera_position=camera_position, camera_look_at=camera_look_at,
            floor_level=0.0, complexity=complexity,
            hdri_id=pick_hdri(rng) if cx.background == "hdri" else None,
            notes={"half_extent": half, "spin": list(spin),
                   "v0": [vx, 0.0, f.vz], "flight_seconds": flight,
                   "apex": f.apex, "scene_scale": f.scene_scale,
                   "frame_half_extent": f.half_extent})


register(Tumble())
