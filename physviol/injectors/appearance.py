"""Appearance-domain injectors: deformation.

A domain of its own because none of the seven physical ones covers it. A body
that squashes out of proportion has not moved unlawfully, gained energy or
broken a contact -- everything about its *trajectory* is fine. What is wrong is
that it stopped looking like itself, and that is among the most common things
real video generators get wrong.

(`shadow_shape` uses the same mechanism and the same law but lives in `optical`,
because the thing losing its shape there is the shadow rather than the object.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class _Squash(Injector):
    """Shared machinery: ease a body into a non-uniform scale and hold it.

    Volume is preserved -- one axis grows by `k`, the other two shrink by
    `1/sqrt(k)` -- so `shape_continuity` stays silent and only the aspect ratio
    moves. That separation is the whole reason the two laws exist: a family that
    changed size *and* proportions at once would fire both, and neither
    measurement would mean anything on its own.
    """

    persistent = True
    ASPECT_BY_BIN = {"weak": 1.35, "medium": 1.9, "strong": 2.8}
    RAMP_FRAMES = 5

    def strong_residual_reference(self, spec) -> float:
        k = self.ASPECT_BY_BIN["strong"]
        return abs(k * np.sqrt(k) - 1.0)

    def _targets(self, spec):
        return self._group(spec)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._targets(spec)
        if not targets:
            return None
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None

        k = float(self.ASPECT_BY_BIN[severity_bin])
        # Which axis stretches is a property of the scene, not of the bin, or
        # the three strengths would be three different distortions.
        axis = int(self._instance_rng(spec).randint(3))
        ramp = max(2, min(self.RAMP_FRAMES, T - t0))
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "aspect_scale", "aspect": k, "axis": axis,
                    "ramp_frames": int(ramp)},
            magnitude=float(k), magnitude_unit=self.magnitude_unit,
            severity_bin=severity_bin,
            notes={"radius": float(targets[0].bounding_radius),
                   "surface_top": _geom.surface_top(spec, targets[0]),
                   "aspect": k, "axis": axis, "ramp_frames": int(ramp),
                   "r_strong": self.strong_residual_reference(spec)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0 = plan.t_event
        k = float(plan.notes["aspect"])
        axis = int(plan.notes["axis"])
        ramp = int(plan.notes["ramp_frames"])
        n = traj.num_frames - t0

        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        ease = u * u * (3.0 - 2.0 * u)
        grow = 1.0 + (k - 1.0) * ease
        shrink = 1.0 / np.sqrt(grow)          # volume-preserving

        for bid in plan.causal_body_ids:
            bi = traj.index_of(int(bid))
            factor = np.tile(shrink[:, None], (1, 3))
            factor[:, axis] = grow
            out.scale_mul[t0:, bi, :] = factor.astype(np.float32)
            if axis == 2:
                # Growing downward would push the body into whatever it rests
                # on, which is a solidity violation this family never claimed.
                top = float(plan.notes["surface_top"])
                r_t = float(traj.radius[bi]) * grow
                out.pos[t0:, bi, 2] = np.maximum(out.pos[t0:, bi, 2], top + r_t)
                self._sync_velocity(traj, out, bi, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Deformation(_Squash):
    """A rigid body squashes or stretches out of proportion.

    Its path stays perfectly lawful -- the violation is that a rigid object is
    not supposed to change shape at all.
    """

    family = "deformation"
    magnitude_unit = "aspect_ratio_change"


class ShadowShape(_Squash):
    """The shadow stays where it belongs but stops matching its caster.

    Deliberately separate from `shadow`, which moves the shadow and leaves its
    shape alone. A benchmark asking whether a model tracks shadow *position*
    should not be scored on clips where the shadow is also the wrong shape, and
    vice versa -- so the two never appear in the same clip.
    """

    family = "shadow_shape"
    magnitude_unit = "shadow_aspect_ratio"

    def _targets(self, spec):
        shade = [b for b in spec.bodies if b.role == "shadow"]
        return shade[:1]


register(Deformation())
register(ShadowShape())
