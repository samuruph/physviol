"""Kinematics-domain injectors: antigravity, continuity.

Both are pure trajectory edits, so they work on *any* scenario whose actor is
airborne -- ball_drop, projectile_toss, occluder_pass -- with no per-scenario
code. That composition is the point of the seam.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from .base import Injector, InterventionPlan, register


class AntiGravity(Injector):
    """Bend gravity on the actor for a bounded stretch of the clip.

    Severity is the peak deviation `|1 - alpha|`: 0 is normal gravity, 1 is
    weightless, 2 is gravity fully reversed.

    Two properties worth being explicit about, because the obvious
    implementation gets both wrong:

    * **The floor stays solid.** Changing gravity must not also drop the actor
      through the ground -- that is a *solidity* violation, and a clip carrying
      two violations while annotating one is worse than useless.
    * **The intervention has a shape.** Alpha ramps from 1 up to its peak and
      back to 1 over the window, after which the actor obeys real physics
      again. So the residual, and with it `severity_map`, traces a curve over
      time instead of sitting at a single value for the rest of the clip.
    """

    family = "antigravity"
    # Chosen so the bins are *visually* distinct on a falling body:
    #   weak   -- falls noticeably slowly
    #   medium -- hangs in the air, then drifts back up
    #   strong -- reverses outright and climbs
    # Capped so the actor stays inside the camera frustum: a body that exits
    # frame leaves an empty mask for the rest of the clip, which is a worse
    # annotation than a slightly less spectacular one.
    ALPHA_BY_BIN = {"weak": 0.30, "medium": -1.2, "strong": -2.8}
    WINDOW_FRACTION = 0.55        # of the frames remaining after t_event

    def strong_residual_reference(self, spec) -> float:
        return abs(1.0 - self.ALPHA_BY_BIN["strong"])          # == 3.8

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = _actor(spec)
        if actor is None:
            return None
        bi = traj.index_of(actor.segmentation_id)
        t0 = _airborne_start(traj, bi, spec, actor)
        if t0 is None or t0 >= traj.num_frames - 2:
            return None

        alpha = self.ALPHA_BY_BIN[severity_bin]
        n_left = traj.num_frames - t0
        n_win = self._window_len(max(2, int(round(self.WINDOW_FRACTION * n_left))),
                                 t0, traj.num_frames)
        t1 = min(traj.num_frames - 1, t0 + n_win - 1)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, t1)],
            causal_body_ids=[actor.segmentation_id],
            params={"type": "per_body_gravity_scale", "alpha_peak": alpha,
                    "profile": "raised_cosine",
                    "frames": int(t1 - t0 + 1)},
            magnitude=abs(1.0 - alpha),
            magnitude_unit="gravity_scale_deviation",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "alpha_peak": alpha,
                   "surface_top": _surface_top(spec)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = _actor(spec)
        bi = traj.index_of(actor.segmentation_id)
        t0, t1 = plan.windows[0]
        n_win = t1 - t0 + 1
        n_after = traj.num_frames - (t1 + 1)

        g = traj.gravity.astype(np.float64)
        alpha = self._pulse(n_win, float(plan.params["alpha_peak"]))
        g_seq = np.concatenate([alpha[:, None] * g[None, :],
                                np.tile(g[None, :], (max(n_after, 0), 1))])

        pos, vel = self._integrate_profile(
            traj.pos[t0 - 1, bi], traj.lin_vel[t0 - 1, bi], g_seq, traj.dt,
            float(plan.notes["surface_top"]), float(plan.notes["radius"]),
            float(actor.restitution))
        out.pos[t0:, bi, :] = pos
        out.lin_vel[t0:, bi, :] = vel

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["alpha_profile"] = [float(x) for x in alpha]
        out.meta["label"] = "invalid"
        return out


class Continuity(Injector):
    """Teleport the actor by a fixed distance at a single frame.

    Instant, and the only family whose mask is genuinely two-lobed: the actor is
    both where it should be and where it wrongly is, which is precisely what the
    union rule in PLAN 3.3 exists to capture.
    """

    family = "continuity"
    JUMP_RADII = {"weak": 1.5, "medium": 4.0, "strong": 9.0}

    def strong_residual_reference(self, spec) -> float:
        return float(self.JUMP_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = _actor(spec)
        if actor is None:
            return None
        bi = traj.index_of(actor.segmentation_id)
        t0 = _airborne_start(traj, bi, spec, actor)
        if t0 is None or t0 >= traj.num_frames - 2:
            return None
        radius = float(actor.bounding_radius)
        jump_r = self.JUMP_RADII[severity_bin]
        direction = np.array([1.0, 0.0, 0.0]) * float(rng.choice([-1.0, 1.0]))
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0, windows=[(t0, t0)],
            causal_body_ids=[actor.segmentation_id],
            params={"type": "position_set",
                    "delta_m": (direction * jump_r * radius).tolist()},
            magnitude=float(jump_r * radius), magnitude_unit="m_jump_distance",
            severity_bin=severity_bin,
            notes={"radius": radius, "jump_radii": jump_r,
                   "surface_top": _surface_top(spec)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = _actor(spec)
        bi = traj.index_of(actor.segmentation_id)
        t0 = plan.t_event
        delta = np.asarray(plan.params["delta_m"], np.float32)
        # Horizontal only, so the teleport cannot smuggle in a floor violation.
        out.pos[t0:, bi, :] = traj.pos[t0:, bi, :] + delta[None, :]
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


# ---------------------------------------------------------------------- #
def _actor(spec):
    return next((b for b in spec.bodies if b.role == "actor"), None)


def _surface_top(spec) -> float:
    surface = next((b for b in spec.bodies
                    if b.role in ("floor", "prop") and b.static), None)
    if surface is None:
        return float(spec.floor_level)
    if surface.kind == "cube":
        return float(surface.position[2] + surface.scale[2])
    return float(spec.floor_level)


def _airborne_start(traj: Trajectory, bi: int, spec, actor) -> Optional[int]:
    """First frame at which it is sensible to bend free flight.

    Prefer a frame while the actor is hidden -- that is what creates a non-zero
    observability lag -- otherwise a third of the way in, clear of the initial
    state and with room left to show consequences.
    """
    occ = spec.notes.get("occluded_frames") or []
    if occ:
        mid = occ[len(occ) // 2]
        if 1 <= mid < traj.num_frames - 1:
            return int(mid)
    t0 = max(1, traj.num_frames // 3)
    return int(t0) if t0 < traj.num_frames - 1 else None


register(AntiGravity())
register(Continuity())
