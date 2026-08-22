"""Equilibrium-domain injectors: support, friction.

Both act on bodies that are *in contact with something* rather than in flight,
which makes them the two families most at risk of accidentally producing a
second violation. A hovering body must not also pass through the table it left;
a block that stops on a slope must stop *on the slope* and not beside it. The
implementations below are shaped almost entirely by those two constraints.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class Support(Injector):
    """A body rises off its support and hangs there.

    Frozen rather than drifting, because the violation is about *equilibrium*:
    a body at rest with nothing under it. Letting it drift would add a
    kinematics anomaly on top and blur which law the clip is evidence for.

    The clearance is measured against whatever is directly beneath the body --
    including another moving body. Measuring against the floor instead would
    report the top block of a stack as hovering two blocks' worth of height on a
    perfectly lawful clip, and the noise floor calibrated on that valid arm
    would swallow the real violation whole.
    """

    family = "support"
    persistent = True
    CLEARANCE_RADII = {"weak": 0.8, "medium": 2.0, "strong": 3.6}

    def strong_residual_reference(self, spec) -> float:
        return float(self.CLEARANCE_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        t0 = max(1, T // 3)
        if t0 >= T - 1:
            return None
        clearance_r = self.CLEARANCE_RADII[severity_bin]
        radius = float(actor.bounding_radius)
        top = _geom.support_under_any(spec, actor, traj, 0)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "hover", "clearance_radii": clearance_r},
            magnitude=float(clearance_r * radius),
            magnitude_unit="m_support_clearance", severity_bin=severity_bin,
            notes={"radius": radius, "surface_top": float(top),
                   "clearance_radii": clearance_r})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])
        lift = float(plan.notes["clearance_radii"]) * radius

        held = traj.pos[t0 - 1, bi].astype(np.float64).copy()
        held[2] = top + radius + lift
        out.pos[t0:, bi, :] = held.astype(np.float32)[None, :]
        out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
        out.lin_vel[t0:, bi, :] = 0.0
        out.ang_vel[t0:, bi, :] = 0.0

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Friction(Injector):
    """A sliding body decelerates, halts, or reverses with nothing to do it.

    Implemented by **replaying the body's own lawful path at a different rate**
    rather than by re-integrating it. That guarantees the body stays exactly on
    the surface it was sliding along: a ramp is not a horizontal plane, so an
    injector that re-integrates under gravity puts the block in mid-air next to
    its ramp -- a support violation smuggled into a friction clip and annotated
    as neither. Following the recorded path costs nothing and cannot leave it.

    `strong` reverses outright, which is the version a viewer cannot miss: a
    block sliding *up* a slope with nothing pushing it.
    """

    family = "friction"
    persistent = True
    RATE_BY_BIN = {"weak": 0.35, "medium": 0.0, "strong": -0.6}

    def strong_residual_reference(self, spec) -> float:
        return 2.0

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        bi = traj.index_of(int(actor.segmentation_id))
        speed = np.linalg.norm(traj.lin_vel[:, bi, :].astype(np.float64), axis=1)
        moving = np.flatnonzero(speed > 0.35)
        if moving.size < 3:
            return None
        t0 = int(max(moving[0] + 1, min(moving[-1] - 1, T // 3)))
        if not (1 <= t0 < T - 1):
            return None

        rate = self.RATE_BY_BIN[severity_bin]
        strong = self._retimed(traj, bi, t0, self.RATE_BY_BIN["strong"])
        r_strong = self._measure(strong, int(actor.segmentation_id), "friction", {})
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "path_rate_scale", "end_rate": rate},
            magnitude=float(1.0 - rate), magnitude_unit="effective_mu_ratio",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "end_rate": rate, "r_strong": float(r_strong)})

    # ------------------------------------------------------------------ #
    def _retimed(self, traj, bi: int, t0: int, end_rate: float) -> Trajectory:
        out = self._clone(traj)
        T = traj.num_frames
        n = T - t0
        # Rate eases from 1 (lawful) to `end_rate` over three frames and then
        # holds, so the body slows visibly instead of stopping between frames.
        ramp = np.linspace(1.0, end_rate, min(3, n) + 1)[1:]
        rate = np.concatenate([ramp, np.full((n - ramp.size,), end_rate)])
        u = float(t0 - 1) + np.cumsum(rate)
        pos = _geom.path_sample(traj.pos[:, bi, :], u)
        out.pos[t0:, bi, :] = pos.astype(np.float32)
        vel = np.zeros_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / traj.dt
        vel[0] = (pos[0] - traj.pos[t0 - 1, bi]) / traj.dt
        out.lin_vel[t0:, bi, :] = vel.astype(np.float32)
        out.ang_vel[t0:, bi, :] = (traj.ang_vel[t0:, bi, :]
                                   * rate[:, None].astype(np.float32))
        return out

    def apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        out = self._retimed(traj, bi, plan.t_event, float(plan.notes["end_rate"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Support())
register(Friction())
