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

    Two shapes, chosen by what the body was doing. One that was at rest hovers
    where it stood. One that was sliding keeps sliding, along exactly the path
    it would have taken, lifted clear of the surface -- freezing that one would
    add a Newton-1 violation on top, and "the box stopped" is a different claim
    from "the box is not touching the ramp".

    The clearance is measured against whatever is directly beneath the body --
    including another moving body. Measuring against the floor instead would
    report the top block of a stack as hovering two blocks' worth of height on a
    perfectly lawful clip, and the noise floor calibrated on that valid arm
    would swallow the real violation whole.
    """

    family = "support"
    persistent = True
    RISE_FRAMES = 3
    CLEARANCE_RADII = {"weak": 0.8, "medium": 2.0, "strong": 3.6}

    def strong_residual_reference(self, spec) -> float:
        return float(self.CLEARANCE_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        # WHOLE MEDIUM where the scene is made of interchangeable bodies. One
        # grain of forty hovering is perfectly annotated and impossible to see,
        # and it also breaks the residual: a lifted grain still has grains
        # beneath it, so `support`'s clearance is taken against them and the law
        # reads it as supported. Lifting the whole pour leaves nothing under any
        # of them, which is both visible and measurable.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        t0 = max(1, T // 3)
        if t0 >= T - 1:
            return None
        clearance_r = self.CLEARANCE_RADII[severity_bin]
        radius = float(actor.bounding_radius)
        _, top = _geom.support_under_any(spec, actor, traj, 0)
        bi = traj.index_of(int(actor.segmentation_id))
        speed = float(np.linalg.norm(traj.lin_vel[t0, bi]))
        # Decided by the body's state, not the scenario's name: a resting body
        # hovers where it is, a moving one keeps moving with nothing under it.
        mode = "hover_still" if speed < 0.3 else "hover_moving"
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets] or
                            [int(actor.segmentation_id)],
            params={"type": "hover", "clearance_radii": clearance_r,
                    "mode": mode},
            magnitude=float(clearance_r * radius),
            magnitude_unit="m_support_clearance", severity_bin=severity_bin,
            notes={"radius": radius,
                   # With the WHOLE medium lifted there is nothing left beneath
                   # any of it, so the clearance datum is the floor. Measured
                   # against the pile -- which is what `support_under_any`
                   # returns for a grain in a column -- a grain hovering at
                   # z = 0.33 sat *below* its recorded support and the law read
                   # clearance 0.000 on an obviously airborne pour.
                   "surface_top": (float(_geom.surface_top(spec, actor))
                                   if len(targets) > 1 else float(top)),
                   "clearance_radii": clearance_r, "mode": mode})

    #: Weightlessness is a force statement, so PyBullet can say it: cancel
    #: gravity on the body and let it keep whatever motion it had.
    simulated = True

    def stage(self, spec, simulator, objs, plan):
        """Lift the body clear and cancel gravity on it.

        The two modes fall out of the physics rather than needing separate code.
        A body at rest hovers where it stood, because nothing is pushing it. A
        body that was sliding keeps sliding in a straight line, because nothing
        is pulling it down -- which is exactly the "keeps moving, lifted clear"
        shape the family wants, and it now happens because it must rather than
        because it was written in.

        It also keeps colliding with everything else while it hovers, which the
        prescribed version could not promise.
        """
        import pybullet as pb
        from ..render import stepper

        clearance = float(plan.notes["clearance_radii"]) * float(
            plan.notes["radius"])
        g = np.asarray(spec.gravity, np.float64)
        targets = []
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is None:
                continue
            pos, quat = pb.getBasePositionAndOrientation(idx)
            lifted = (pos[0], pos[1], pos[2] + clearance)
            v, w = pb.getBaseVelocity(idx)
            pb.resetBasePositionAndOrientation(idx, lifted, quat)
            pb.resetBaseVelocity(idx, list(v), list(w))
            targets.append((idx, float(getattr(body, "mass", 1.0))))
        if not targets:
            return ()

        def weightless(_client, _step, _frame):
            for idx, mass in targets:
                pos, _ = pb.getBasePositionAndOrientation(idx)
                pb.applyExternalForce(idx, -1, (-g * mass).tolist(), list(pos),
                                      pb.WORLD_FRAME)

        return (weightless,)

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])
        lift = float(plan.notes["clearance_radii"]) * radius
        n = traj.num_frames - t0

        # Eased up over a few frames, not snapped. A body that jumps three radii
        # between two frames is a *teleport*, so snapping it made every
        # `support` clip trip the continuity detector as well.
        ramp = max(2, min(self.RISE_FRAMES, n))
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        ease = u * u * (3.0 - 2.0 * u)
        start = traj.pos[t0 - 1, bi].astype(np.float64)

        if plan.notes["mode"] == "hover_still":
            # It was at rest: it stays where it was, only higher.
            held = np.tile(start, (n, 1))
            held[:, 2] = start[2] + (top + radius + lift - start[2]) * ease
        else:
            # It was moving: it keeps travelling exactly the path it would have
            # taken, lifted clear of the surface. Freezing a sliding body would
            # be a *second* violation -- Newton 1 -- on top of this one, and it
            # is the version the eye reads as "that box is not touching the
            # ramp" rather than "that box stopped".
            held = traj.pos[t0:, bi, :].astype(np.float64).copy()
            held[:, 2] = held[:, 2] + lift * ease

        out.pos[t0:, bi, :] = held.astype(np.float32)
        out.quat[t0:, bi, :] = traj.quat[t0:, bi, :]
        out.ang_vel[t0:, bi, :] = 0.0
        self._sync_velocity(traj, out, bi, t0)

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
        # WHOLE MEDIUM where the scene is made of interchangeable bodies. One
        # grain of forty hovering is perfectly annotated and impossible to see,
        # and it also breaks the residual: a lifted grain still has grains
        # beneath it, so `support`'s clearance is taken against them and the law
        # reads it as supported. Lifting the whole pour leaves nothing under any
        # of them, which is both visible and measurable.
        targets = self._group(spec)
        actor = targets[0] if targets else self._primary(spec)
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

    #: NOT staged, though it easily could be. `changeDynamics(lateralFriction=)`
    #: works and the code below is correct, but it is a regression in practice:
    #: on `barrier_pass` the staged version scores 0.02 / 0.04 / 0.00 where the
    #: trajectory path scores 1.00, because this family's residual measures the
    #: *effective* coefficient recovered from the path, and a coefficient change
    #: applied to a body that is already nearly at rest barely moves it.
    #:
    #: Fixing that means giving the scenarios more of their clip in motion --
    #: docs/decisions_pending.md section 3 -- not switching paths. Left here
    #: rather than deleted so the next attempt starts from a working stage().
    simulated = False

    def stage(self, spec, simulator, objs, plan):
        """Scale the actor's friction coefficient and let it slide.

        You reported a phantom bounce mixed into this family: the trajectory was
        rewritten along a rescaled path and re-integrated, so a contact the
        resolver invented ended up inside a clip that claims to be about
        friction alone. Staged as a coefficient, the only thing that differs
        from the valid twin is how much grip the surface has.
        """
        import pybullet as pb
        from ..render import stepper

        rate = float(plan.params["end_rate"])
        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                pb.changeDynamics(
                    idx, -1,
                    lateralFriction=float(getattr(body, "friction", 0.5)) * rate)
        return ()

    def unstage(self, spec, simulator, objs, plan) -> None:
        import pybullet as pb
        from ..render import stepper

        for bid in plan.causal_body_ids:
            body = next((b for b in spec.bodies
                         if int(b.segmentation_id) == int(bid)), None)
            if body is None or body.static:
                continue
            idx = stepper.pybullet_index(simulator, objs, spec, int(bid))
            if idx is not None:
                pb.changeDynamics(idx, -1,
                                  lateralFriction=float(getattr(body, "friction", 0.5)))

    def _apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        out = self._retimed(traj, bi, plan.t_event, float(plan.notes["end_rate"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Support())
register(Friction())
