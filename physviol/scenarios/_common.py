"""Shared scenario building blocks.

Scenario files stay short because everything generic lives here: the ground
plane (which becomes an HDRI dome from complexity L1 up), standard lighting and
a colour helper. A new scenario is then mostly a description of *what is staged*,
not renderer plumbing.
"""
from __future__ import annotations

import colorsys
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
