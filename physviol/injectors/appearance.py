"""Appearance-domain injectors: deformation, colour_shift.

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
from ..residuals.laws import _srgb_to_lab
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
        #
        # A body RESTING on a surface may only stretch horizontally. Stretching
        # it vertically makes it taller, which either drives it into the floor
        # or lifts its centre of mass -- and lifting a centre of mass is real
        # work done from nowhere, so the clip would depict a shape violation
        # *and* an energy violation while claiming one. That is the same trap
        # the volume-preserving scaling avoids on the mass axis: this family's
        # whole job is to move the aspect ratio and nothing else.
        rng = self._instance_rng(spec)
        top = _geom.surface_top(spec, targets[0])
        radius = float(targets[0].bounding_radius)
        resting = bool(traj.pos[t0, traj.index_of(
            int(targets[0].segmentation_id)), 2] - radius <= top + 1e-2)
        axis = int(rng.randint(2)) if resting else int(rng.randint(3))
        ramp = max(2, min(self.RAMP_FRAMES, T - t0))
        union, applied, after = self._split_windows(t0, T, ramp)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "aspect_scale", "aspect": k, "axis": axis,
                    "ramp_frames": int(ramp)},
            magnitude=float(k), magnitude_unit=self.magnitude_unit,
            severity_bin=severity_bin,
            notes={"radius": radius, "surface_top": top,
                   "aspect": k, "axis": axis, "ramp_frames": int(ramp),
                   "resting": resting,
                   "r_strong": self.strong_residual_reference(spec)})

    def _apply(self, spec, traj, plan) -> Trajectory:
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


class ColourShift(Injector):
    """A body changes colour, with nothing in the scene to explain it.

    The most common perceptual failure in generated video, and the one furthest
    from anything mechanical: the trajectory is untouched and every physical law
    in the taxonomy stays satisfied. What is wrong is that the object stopped
    looking like itself.

    Material colour is animated the same way pose and scale are -- one fcurve
    per channel on the Principled BSDF's Base Colour, verified rendering
    against the pinned image. Eased rather than switched, for the reason
    `immutability` is: a colour that changes between two frames is a cut,
    one that visibly shifts is a violation.

    Severity is perceptual distance rather than RGB distance, and the bins are
    *solved for* in CIE-Lab rather than expressed as hue rotations -- see
    `DISTANCE_BY_BIN` for why the obvious version is not monotone.
    """

    family = "colour_shift"
    persistent = True
    #: Target perceptual separation, in CIE-Lab distance over 100. Specified
    #: here and *solved for* per body, rather than specified as a hue rotation.
    #:
    #: Setting the bins in hue was the first attempt and it is not monotone in
    #: what it claims to measure: from red, a 0.30 turn lands on green and a
    #: 0.50 turn lands on cyan, and green is the further of the two in Lab. So
    #: `medium` came out more different than `strong` on some colours and less
    #: on others, and the severity ordering the whole bin system rests on held
    #: only by luck. Solving for the distance makes the bins mean the same thing
    #: whatever colour a body started as.
    DISTANCE_BY_BIN = {"weak": 0.25, "medium": 0.55, "strong": 0.95}
    RAMP_FRACTION = 0.22
    RAMP_MIN = 3

    def strong_residual_reference(self, spec) -> float:
        return float(self.DISTANCE_BY_BIN["strong"])

    @staticmethod
    def _shift_to_distance(rgb, target: float, sign: float):
        """The colour a hue rotation reaches at `target` Lab distance from `rgb`.

        Bisection on the rotation, because Lab distance is monotone in hue turn
        over half the wheel but has no closed form. Twenty steps is well past
        the precision anything downstream can use.
        """
        import colorsys
        h, sat, val = colorsys.rgb_to_hsv(*rgb)
        sat = min(1.0, sat + 0.12)
        base = _srgb_to_lab(np.asarray(rgb, np.float64))

        def at(turn):
            out = np.asarray(colorsys.hsv_to_rgb((h + sign * turn) % 1.0,
                                                 sat, val), np.float64)
            return out, float(np.linalg.norm(_srgb_to_lab(out) - base) / 100.0)

        lo, hi = 0.0, 0.5
        best = at(hi)
        if best[1] <= target:
            return best                    # even the antipode is not far enough
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            cand = at(mid)
            if cand[1] < target:
                lo = mid
            else:
                hi, best = mid, cand
        return best

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = [b for b in self._group(spec) if b.kind != "dome"]
        if not targets:
            return None
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None

        target = float(self.DISTANCE_BY_BIN[severity_bin])
        # Direction per scene, not per bin, so the three strengths are the same
        # shift at three sizes rather than three different colours.
        sign = 1.0 if self._instance_rng(spec).rand() < 0.5 else -1.0
        ramp = max(self.RAMP_MIN, int(round(self.RAMP_FRACTION * T)))

        finals, reached = {}, []
        for body in targets:
            rgb, dist = self._shift_to_distance(body.color, target, sign)
            finals[int(body.segmentation_id)] = [float(x) for x in rgb]
            reached.append(dist)
        union, applied, after = self._split_windows(t0, T, ramp)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "colour_ramp", "target_distance": target,
                    "ramp_frames": int(min(ramp, T - t0))},
            magnitude=float(np.mean(reached)),
            magnitude_unit="lab_distance", severity_bin=severity_bin,
            notes={"radius": float(targets[0].bounding_radius),
                   "surface_top": _geom.surface_top(spec, targets[0]),
                   "final_colour": finals,
                   "ramp_frames": int(min(ramp, T - t0)),
                   "r_strong": self.strong_residual_reference(spec)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0 = plan.t_event
        ramp = int(plan.notes["ramp_frames"])
        n = traj.num_frames - t0
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / max(ramp, 1), 0.0, 1.0)
        ease = (u * u * (3.0 - 2.0 * u))[:, None]

        for bid, final in plan.notes["final_colour"].items():
            bi = traj.index_of(int(bid))
            start = traj.colour[t0 - 1, bi].astype(np.float64)
            end = np.asarray(final, np.float64)
            out.colour[t0:, bi, :] = (
                start[None, :] + (end - start)[None, :] * ease).astype(np.float32)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(ColourShift())
register(Deformation())
register(ShadowShape())
