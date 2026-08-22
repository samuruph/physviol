"""Shared scenario building blocks.

Scenario files stay short because everything generic lives here: the ground
plane (which becomes an HDRI dome from complexity L1 up), standard lighting and
a colour helper. A new scenario is then mostly a description of *what is staged*,
not renderer plumbing.
"""
from __future__ import annotations

import colorsys
import math
from typing import List, Tuple

from .base import BodySpec, Complexity, LightSpec


def ground(cx: Complexity, seg_id: int, size: float = 6.0) -> BodySpec:
    """At L0 a plain cube; from L1 up KuBasic's `dome`, which doubles as the
    HDRI backdrop -- the same trick MOVi uses."""
    if cx.background == "hdri":
        return BodySpec(name="floor", kind="dome", position=(0.0, 0.0, 0.0),
                        mass=0.0, static=True, friction=0.6, restitution=0.4,
                        segmentation_id=seg_id, role="floor")
    return BodySpec(name="floor", kind="cube", position=(0.0, 0.0, -0.1),
                    scale=(size, size, 0.1), mass=0.0, static=True,
                    friction=0.6, restitution=0.4, color=(0.32, 0.33, 0.36),
                    segmentation_id=seg_id, role="floor")


def lights(cx: Complexity, look_at=(0.0, 0.0, 0.6)) -> List[LightSpec]:
    """An HDRI environment lights the scene on its own; only L0 needs a sun."""
    if cx.background == "hdri":
        return []
    return [LightSpec("sun", position=(-2.2, -1.6, 4.5), look_at=look_at,
                      intensity=2.6)]


def hue_rgb(h: float, s: float = 0.62, v: float = 0.88) -> Tuple[float, float, float]:
    return tuple(float(c) for c in colorsys.hsv_to_rgb(h, s, v))


# --------------------------------------------------------------------------
# Inclines. Two scenarios stage one, and getting a body to *sit* on a tilted
# slab is fiddly enough that doing it twice by hand invites a body embedded in
# its own ramp -- which reads as a solidity violation nobody injected.
# --------------------------------------------------------------------------


def ramp_axes(tilt: float) -> Tuple[Tuple[float, float, float],
                                    Tuple[float, float, float]]:
    """(down-slope, surface-normal) unit vectors for a slab tilted about +Y.

    A rotation of `tilt` about +Y takes +X to (cos, 0, -sin), so that is the
    down-slope direction, and +Z to (sin, 0, cos), the outward normal.
    """
    c, s = math.cos(tilt), math.sin(tilt)
    return (c, 0.0, -s), (s, 0.0, c)


def ramp(seg_id: int, tilt: float, center, half_len: float, half_wid: float,
         thick: float, friction: float = 0.3, name: str = "ramp") -> BodySpec:
    return BodySpec(
        name=name, kind="cube", position=tuple(float(x) for x in center),
        scale=(half_len, half_wid, thick),
        quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
        mass=0.0, static=True, friction=friction, restitution=0.15,
        color=(0.38, 0.36, 0.42), segmentation_id=seg_id, role="prop")


def on_ramp(center, tilt: float, along: float, half_size: float,
            clearance: float = 0.02) -> Tuple[float, float, float]:
    """World position of a body of half-extent `half_size` resting on a ramp.

    `along` is signed distance from the slab's centre down the slope. The
    `clearance` lifts it a hair off the surface so the simulator settles it into
    contact rather than starting it interpenetrating.
    """
    d, n = ramp_axes(tilt)
    lift = half_size + clearance
    return tuple(float(center[i] + along * d[i] + lift * n[i]) for i in range(3))


def understudy(actor: BodySpec, seg_id: int) -> BodySpec:
    """A dormant duplicate of `actor`, parked out of the render until summoned.

    `fission` needs a second body to spawn, and a body cannot be added to a
    Kubric scene part-way through a render without perturbing the very render
    path prefix identity depends on. So the understudy is declared from frame 0,
    marked `dormant`, and contributes no pixels until an intervention switches
    it on. It costs one scene asset and nothing at all in the valid twin.
    """
    # Parked far below the scene, not at the actor's own position: the
    # understudy is `static` to the simulator but still a collidable body, and
    # sharing the actor's start pose made PyBullet report a contact between the
    # two on frame 0 -- which then became the "first contact" that `solidity`
    # and `superelastic` hung their violation on.
    return BodySpec(
        name=actor.name + "_split", kind=actor.kind, position=(0.0, 0.0, -1000.0),
        scale=actor.scale, quaternion=actor.quaternion, mass=actor.mass,
        friction=actor.friction, restitution=actor.restitution,
        color=actor.color, segmentation_id=seg_id, role="actor",
        scripted=True, dormant=True)
