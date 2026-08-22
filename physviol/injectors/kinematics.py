"""Kinematics-domain injectors, plus the one global-domain family.

    antigravity      per-body gravity scale, bounded in time
    global_gravity   the same knob turned on the whole scene at once
    continuity       a discontinuous position set
    non_parabolic    free flight that no parabola fits
    newton1_inertia  a body that starts or stops with nothing acting on it

All five are pure trajectory edits, so they work on *any* scenario whose actor
is in the right state -- airborne, resting, sliding -- with no per-scenario code.
That composition is the point of the seam: thirteen scenarios times seventeen
families is 48 build cells and 21 files, not 221.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


# ---------------------------------------------------------------------- #
def _event_frame(spec, traj, body, num_frames: int) -> Optional[int]:
    """When to fire on a body that ought to be in free flight.

    Hidden if the scenario offers an occlusion interval -- that is what makes
    the observability lag non-zero. Otherwise inside the longest airborne run,
    no earlier than a quarter of the way in so there is a lawful prefix to
    contradict. Falling back to a fixed fraction of the clip is what puts
    `ball_drop` mid-bounce, and a parabola fitted across a bounce is not a
    parabola.
    """
    occ = _geom.occluded_midpoint(spec)
    if occ is not None and 1 <= occ < num_frames - 1:
        return int(occ)
    bi = traj.index_of(int(body.segmentation_id))
    run = _geom.longest_airborne_run(traj, bi, _geom.surface_top(spec, body))
    lo = max(1, num_frames // 4)
    if run is not None and run[1] > run[0]:
        t = min(max(run[0] + 1, lo), run[1])
        if 1 <= t < num_frames - 1:
            return int(t)
    t = max(1, num_frames // 3)
    return int(t) if t < num_frames - 1 else None


class _GravityScale(Injector):
    """Shared machinery for `antigravity` and `global_gravity`.

    They differ in *extent*, not in mechanism: one bends gravity for the culprit,
    the other for everything that moves. Keeping one implementation means the
    trapezoid profile, the solid floor and the bounded window are identical
    across both, so the only thing the label distinguishes is what it claims to.
    """

    ALPHA_BY_BIN = {"weak": 0.30, "medium": -1.2, "strong": -2.8}
    WINDOW_FRACTION = 0.55
    spatial_extent = "local"

    def strong_residual_reference(self, spec) -> float:
        return abs(1.0 - self.ALPHA_BY_BIN["strong"])

    def _targets(self, spec):
        return self._all_actors(spec)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._targets(spec)
        if not targets:
            return None
        t0 = _event_frame(spec, traj, targets[0], traj.num_frames)
        if t0 is None:
            return None

        alpha = self.ALPHA_BY_BIN[severity_bin]
        n_left = traj.num_frames - t0
        n_win = self._window_len(max(2, int(round(self.WINDOW_FRACTION * n_left))),
                                 t0, traj.num_frames)
        t1 = min(traj.num_frames - 1, t0 + n_win - 1)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "gravity_scale", "alpha_peak": alpha,
                    "profile": "trapezoid", "frames": int(t1 - t0 + 1),
                    "extent": self.spatial_extent},
            magnitude=abs(1.0 - alpha),
            magnitude_unit="gravity_scale_deviation",
            severity_bin=severity_bin, spatial_extent=self.spatial_extent,
            notes={"surface_top": _geom.surface_top(spec, targets[0]),
                   "radius": float(targets[0].bounding_radius),
                   "alpha_peak": alpha, "n_targets": len(targets)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0, t1 = plan.windows[0]
        n_win, n_after = t1 - t0 + 1, traj.num_frames - (t1 + 1)

        g = traj.gravity.astype(np.float64)
        alpha = self._pulse(n_win, float(plan.params["alpha_peak"]))
        g_seq = np.concatenate([alpha[:, None] * g[None, :],
                                np.tile(g[None, :], (max(n_after, 0), 1))])
        for body in self._targets(spec):
            self._rewrite_from(spec, traj, out, body, t0, g_per_frame=g_seq)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["alpha_profile"] = [float(x) for x in alpha]
        out.meta["label"] = "invalid"
        return out


class AntiGravity(_GravityScale):
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

    Bin values are chosen so the three are *visually* distinct on a falling
    body: weak falls noticeably slowly, medium hangs and drifts back up, strong
    reverses outright and climbs. Capped so the actor stays inside the frustum;
    a body that leaves frame has an empty mask for the rest of the clip, which
    is a worse annotation than a less spectacular one.
    """

    family = "antigravity"


class GlobalGravity(_GravityScale):
    """The same bend, applied to every moving body at once.

    The label a consumer sees is `spatial_extent: "global"`, and that is the
    whole difference: nothing in the frame contradicts anything else, so the
    only evidence is the viewer's prior about how fast things fall on Earth.
    That makes it the hardest family in the taxonomy and the one where a
    per-pixel mask is least informative -- the mask is every moving body.

    Weak and medium stay inside the "this is a different planet" regime.
    Strong deliberately leaves it and reverses, because `strong` is the debug
    default and a strongest bin nobody can see is not worth generating.
    """

    family = "global_gravity"
    ALPHA_BY_BIN = {"weak": 0.50, "medium": 0.15, "strong": -0.9}
    spatial_extent = "global"

    def _targets(self, spec):
        return [b for b in spec.bodies if not b.static and not b.dormant]


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
        actor = self._primary(spec)
        if actor is None:
            return None
        t0 = _event_frame(spec, traj, actor, traj.num_frames)
        if t0 is None:
            return None
        radius = float(actor.bounding_radius)
        jump_r = self.JUMP_RADII[severity_bin]
        direction = np.array([1.0, 0.0, 0.0]) * float(rng.choice([-1.0, 1.0]))
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0, windows=[(t0, t0)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "position_set",
                    "delta_m": (direction * jump_r * radius).tolist()},
            magnitude=float(jump_r * radius), magnitude_unit="m_jump_distance",
            severity_bin=severity_bin,
            notes={"radius": radius, "jump_radii": jump_r,
                   "surface_top": _geom.surface_top(spec, actor)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        delta = np.asarray(plan.params["delta_m"], np.float32)
        # Horizontal only, so the teleport cannot smuggle in a floor violation.
        out.pos[t0:, bi, :] = traj.pos[t0:, bi, :] + delta[None, :]
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class NonParabolic(Injector):
    """Free flight that no parabola fits.

    Distinct from `antigravity` in a way that matters: the *mean* acceleration
    over the window is still g, so a model that only checks the average fall
    rate sees nothing wrong. What is wrong is the shape -- the arc wobbles and
    returns to where the parabola would have put it, so the intervention leaves
    no lasting displacement and the body is lawful again afterwards.

    The residual is fitted against the *airborne* frames only, which the plan
    records. Fitting across a bounce would compare the arc to a parabola through
    two arcs and report a large residual on a perfectly lawful clip.
    """

    family = "non_parabolic"
    AMPLITUDE_RADII = {"weak": 0.7, "medium": 1.8, "strong": 3.4}
    DEFAULT_FRAMES = 5

    def strong_residual_reference(self, spec) -> float:
        # The least-squares parabola absorbs part of a short wobble, so the
        # peak measured deviation lands a little under the amplitude asked for.
        return float(self.AMPLITUDE_RADII["strong"]) * 0.9

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        bi = traj.index_of(int(actor.segmentation_id))
        top = _geom.surface_top(spec, actor)
        run = _geom.longest_airborne_run(traj, bi, top)
        if run is None or run[1] - run[0] < 3:
            return None
        t0 = max(run[0] + 1, min(run[1] - 2, traj.num_frames // 4))
        n_win = self._window_len(self.DEFAULT_FRAMES, t0, traj.num_frames)
        t1 = min(run[1], t0 + n_win - 1)
        if t1 <= t0:
            return None

        radius = float(actor.bounding_radius)
        amp = self.AMPLITUDE_RADII[severity_bin] * radius
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "path_warp", "profile": "one_cycle_sine",
                    "amplitude_m": amp, "frames": int(t1 - t0 + 1)},
            magnitude=float(amp), magnitude_unit="m_rms_from_parabola",
            severity_bin=severity_bin,
            notes={"radius": radius, "surface_top": top,
                   "flight_frames": list(range(int(run[0]), int(run[1]) + 1))})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0, t1 = plan.windows[0]
        n = t1 - t0 + 1
        amp = float(plan.params["amplitude_m"])
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])

        # One full sine cycle across the window: the offset and its slope are
        # both zero at each end, so the body rejoins its lawful arc smoothly and
        # the frames outside the window stay untouched.
        u = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
        wob = amp * np.sin(2.0 * np.pi * u)
        lateral = 0.55 * amp * np.sin(np.pi * u)

        pos = traj.pos[t0:t1 + 1, bi, :].astype(np.float64).copy()
        pos[:, 2] = np.maximum(pos[:, 2] + wob, top + radius)
        pos[:, 0] = pos[:, 0] + lateral
        out.pos[t0:t1 + 1, bi, :] = pos.astype(np.float32)
        vel = out.lin_vel[t0:t1 + 1, bi, :].astype(np.float64)
        vel[1:] = (pos[1:] - pos[:-1]) / traj.dt
        out.lin_vel[t0:t1 + 1, bi, :] = vel.astype(np.float32)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Newton1Inertia(Injector):
    """A body that starts moving, or stops moving, with nothing acting on it.

    Which of the two happens is decided by the body's *state*, not by the
    scenario's name -- a resting mug is made to slide, a sliding block is made
    to stop dead. That is the rule that keeps one injector working across
    `resting_table`, `ramp_slide` and `rolling_ramp`.

    The two branches also get different windows, and the difference is
    physical. A body that stops dead is breaking Newton 1 for as long as it sits
    there, so the window runs to the end of the clip. A body given a shove is
    breaking it only at the shove; everything after is lawful ballistic motion,
    which is why the window is short and the consequences are not annotated as
    the violation.
    """

    family = "newton1_inertia"
    KICK_BY_BIN = {"weak": 0.9, "medium": 2.0, "strong": 3.6}   # m/s
    AT_REST = 0.25                                              # m/s
    STRONG_REFERENCE = 3.0                                      # dv / (g*dt)

    def strong_residual_reference(self, spec) -> float:
        return self.STRONG_REFERENCE

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = max(1, T // 3)
        if t0 >= T - 1:
            return None
        speed = float(np.linalg.norm(traj.lin_vel[t0, bi]))
        mode = "halt" if speed > self.AT_REST else "start"

        if mode == "halt":
            windows = [(t0, T - 1)]
            dv = speed
            params = {"type": "velocity_zero", "frames": int(T - t0)}
        else:
            n_win = self._window_len(2, t0, T)
            windows = [(t0, min(T - 1, t0 + n_win - 1))]
            dv = self.KICK_BY_BIN[severity_bin]
            heading = float(rng.uniform(0.0, 2.0 * np.pi))
            params = {"type": "velocity_set",
                      "delta_v": [dv * float(np.cos(heading)),
                                  dv * float(np.sin(heading)), 0.0]}

        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        return InterventionPlan(
            family=self.family, kind="sustained" if mode == "halt" else "instant",
            t_event=t0, windows=windows,
            causal_body_ids=[int(actor.segmentation_id)],
            params=dict(params, mode=mode),
            magnitude=float(dv / max(g_dt, 1e-9)),
            magnitude_unit="dv_over_g_dt", severity_bin=severity_bin,
            notes={"mode": mode, "radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "speed_at_event": speed})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event

        if plan.notes["mode"] == "halt":
            # Frozen where it was: position held, velocity zero, spin zero.
            out.pos[t0:, bi, :] = traj.pos[t0 - 1, bi][None, :]
            out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
            out.lin_vel[t0:, bi, :] = 0.0
            out.ang_vel[t0:, bi, :] = 0.0
        else:
            v0 = (traj.lin_vel[t0 - 1, bi].astype(np.float64)
                  + np.asarray(plan.params["delta_v"], np.float64))
            self._rewrite_from(spec, traj, out, actor, t0, v0=v0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(AntiGravity())
register(GlobalGravity())
register(Continuity())
register(NonParabolic())
register(Newton1Inertia())
