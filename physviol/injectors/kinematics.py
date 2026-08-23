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

    ALPHA_BY_BIN = {"weak": -0.6, "medium": -1.6, "strong": -2.8}
    WINDOW_FRACTION = 0.55
    MIN_BODIES = 1
    spatial_extent = "local"

    def strong_residual_reference(self, spec) -> float:
        return abs(1.0 - self.ALPHA_BY_BIN["strong"])

    def _targets(self, spec):
        return self._all_actors(spec)

    def _choose(self, spec, traj):
        """Which bodies to bend, given the rollout. Overridden by `antigravity`."""
        return self._targets(spec)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._choose(spec, traj)
        if len(targets) < self.MIN_BODIES:
            return None
        t0 = _event_frame(spec, traj, targets[0], traj.num_frames)
        if t0 is None:
            return None

        n_left = traj.num_frames - t0
        n_want = self._window_len(max(2, int(round(self.WINDOW_FRACTION * n_left))),
                                  t0, traj.num_frames)
        alpha = self.ALPHA_BY_BIN[severity_bin]
        strongest = self.ALPHA_BY_BIN["strong"]

        # Keep the bodies on screen by **shortening the window**, not by
        # weakening the intervention. The bin is a qualitative claim -- gravity
        # reverses and the body climbs -- and scaling alpha back until the body
        # fits turns that into "gravity is slightly reduced", which is a
        # different violation wearing this one's label. Fitted on the strongest
        # bin so all three share a window and stay comparable.
        n_win = self._fit_window_to_frame(
            spec, traj, targets, t0, n_want,
            lambda n: self._rollout(spec, traj, t0,
                                    min(traj.num_frames - 1, t0 + n - 1),
                                    strongest, targets))
        t1 = min(traj.num_frames - 1, t0 + n_win - 1)

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "gravity_scale", "alpha_peak": alpha,
                    "profile": "trapezoid", "frames": int(t1 - t0 + 1),
                    "frames_requested": int(n_want),
                    "extent": self.spatial_extent},
            magnitude=abs(1.0 - alpha),
            magnitude_unit="gravity_scale_deviation",
            severity_bin=severity_bin, spatial_extent=self.spatial_extent,
            notes={"surface_top": _geom.surface_top(spec, targets[0]),
                   "radius": float(targets[0].bounding_radius),
                   "alpha_peak": alpha, "n_targets": len(targets)})

    def _rollout(self, spec, traj, t0: int, t1: int, alpha_peak: float,
                 targets) -> Trajectory:
        """Integrate the targets under a trapezoidal gravity pulse over [t0,t1].

        Shared by `plan` -- which runs it a few times to pick a peak that keeps
        the bodies in shot -- and by `apply`, so the clip that ships is exactly
        the one the plan was chosen from.
        """
        out = self._clone(traj)
        n_win, n_after = t1 - t0 + 1, traj.num_frames - (t1 + 1)
        g = traj.gravity.astype(np.float64)
        alpha = self._pulse(n_win, float(alpha_peak))
        g_seq = np.concatenate([alpha[:, None] * g[None, :],
                                np.tile(g[None, :], (max(n_after, 0), 1))])
        # Stepped together. Under `global_gravity` every moving body in the
        # scene is being re-integrated at once, so doing them one at a time
        # against each other's *lawful* paths made each dodge where its
        # neighbours would have been rather than where they now are -- which is
        # how a pyramid's spheres ended up passing through the cube that struck
        # them, a solidity failure inside a gravity clip.
        self._rewrite_group(spec, traj, out, targets, t0,
                            g_by_body={int(b.segmentation_id): g_seq
                                       for b in targets})
        out.meta = dict(traj.meta)
        out.meta["alpha_profile"] = [float(x) for x in alpha]
        return out

    def apply(self, spec, traj, plan) -> Trajectory:
        t0, t1 = plan.windows[0]
        # The bodies the PLAN named, never a fresh choice. `_choose` picks the
        # most airborne actor while `_targets` picks a random one, so
        # re-deriving here bent a different grain than the plan had annotated
        # -- and since the annotated grain was then untouched, the clip came out
        # byte-identical to its valid twin with a full set of labels attached.
        by_id = {int(b.segmentation_id): b for b in spec.bodies}
        targets = [by_id[int(i)] for i in plan.causal_body_ids if int(i) in by_id]
        out = self._rollout(spec, traj, t0, t1,
                            float(plan.params["alpha_peak"]), targets)
        out.meta["intervention"] = plan.to_dict()
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

    **Every bin reverses the actor's direction**; they differ in how hard. A
    bin whose only effect is that things fall a bit slower is indistinguishable
    from a shutter-speed change, so all three are qualitative claims -- the body
    turns around and comes back -- and severity is how violently. When the
    geometry cannot host that, the *window* shortens rather than the reversal
    weakening (see `_fit_window_to_frame`).

    **It acts on exactly one body, never the whole scene.** That is the entire
    difference from `global_gravity`, and it is invisible unless something else
    in the frame is still falling normally -- which is why the compatibility
    matrix pairs the two with different scenarios, and why both only become
    fully legible at `--population multi`.
    """

    family = "antigravity"

    def _targets(self, spec):
        live = self._all_actors(spec)
        if not live:
            return []
        return [live[self._instance_rng(spec).randint(len(live))]]

    def _choose(self, spec, traj):
        """One actor, and one that is actually in the air.

        Bending gravity for a body resting on the floor produces almost nothing
        to measure: `free_fall` is gated on the frames where a body is
        unsupported, so a grain sitting in a pile has its residual zeroed for
        the whole window however hard gravity is reversed. That is how
        `granular_pour x antigravity` came out with a clean-looking plan and a
        severity of exactly zero -- it had picked a grain that had already
        landed.

        Preferring the longest airborne run also keeps the choice deterministic
        and stable across severity bins, since it depends only on the valid
        rollout. The instance rng breaks ties, so a scene where every actor is
        equally airborne still varies which one is bent.

        How *many* comes from `_group`, so a scene made of many interchangeable
        bodies can ask for a share of them. One floating grain out of forty is
        a few pixels nobody will find; a dozen rising together is the point.
        """
        live = self._all_actors(spec)
        if not live:
            return []
        want = len(self._group(spec))
        jitter = self._instance_rng(spec)
        scored = []
        for body in live:
            bi = traj.index_of(int(body.segmentation_id))
            run = _geom.longest_airborne_run(
                traj, bi, _geom.surface_top(spec, body))
            span = 0 if run is None else (run[1] - run[0] + 1)
            scored.append(((span, float(jitter.rand())), body))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:max(1, want)]]


class GlobalGravity(_GravityScale):
    """The same bend, applied to every moving body at once.

    The label a consumer sees is `spatial_extent: "global"`, and that is the
    whole difference: nothing in the frame contradicts anything else, so the
    only evidence is the viewer's prior about how fast things fall on Earth.
    That makes it the hardest family in the taxonomy and the one where a
    per-pixel mask is least informative -- the mask is every moving body.

    Weak and medium stay inside the "this is a different planet" regime, which
    is the signature of a *global* constant being wrong: everything falls in
    slow motion together, and no object contradicts another. Strong leaves that
    regime and reverses, because `strong` is the development default and a
    strongest bin nobody can see is not worth generating.

    **Needs at least two moving bodies, and `plan` returns None below that.**
    With one object on screen, scaling gravity for the scene and scaling it for
    that object produce identical pixels -- the clip would be an `antigravity`
    clip with a different label, and the two families would be
    indistinguishable everywhere the dataset used them.
    """

    family = "global_gravity"
    ALPHA_BY_BIN = {"weak": 0.45, "medium": 0.05, "strong": -1.2}
    MIN_BODIES = 2
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
        nominal = direction * jump_r * radius
        # A teleport big enough to leave the frame depicts an object vanishing,
        # which is `permanence`, not `continuity`. Shorten it until both lobes
        # of the two-lobed mask are actually in shot -- fitting on the longest
        # jump so all three bins shrink together and stay ordered.
        strongest = direction * self.JUMP_RADII["strong"] * radius
        scale, _ = self._fit_to_frame(
            spec, traj, [actor], t0, strongest,
            lambda k: self._teleport(traj, actor, t0, strongest * k))
        delta = nominal * scale
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0, windows=[(t0, t0)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "position_set", "delta_m": delta.tolist(),
                    "frame_fit_scale": scale},
            magnitude=float(np.linalg.norm(delta)),
            magnitude_unit="m_jump_distance",
            severity_bin=severity_bin,
            notes={"radius": radius, "jump_radii": jump_r,
                   "surface_top": _geom.surface_top(spec, actor)})

    def _teleport(self, traj, actor, t0: int, delta) -> Trajectory:
        out = self._clone(traj)
        bi = traj.index_of(int(actor.segmentation_id))
        # Horizontal only, so the teleport cannot smuggle in a floor violation.
        out.pos[t0:, bi, :] = traj.pos[t0:, bi, :] + np.asarray(delta, np.float32)
        return out

    def apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        out = self._teleport(traj, actor, plan.t_event,
                             np.asarray(plan.params["delta_m"], np.float32))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class NonParabolic(Injector):
    """Free flight that snakes: no parabola fits it, and nothing jumps.

    The point of the family is to be wrong in a way the other two kinematic
    families are not. `antigravity` gets the *rate* of fall wrong; `continuity`
    puts the body somewhere it could not have travelled to. This one keeps both
    right -- the mean acceleration over the flight is still g, and the path is
    continuous with a continuous derivative everywhere -- and gets the *shape*
    wrong. A model that only checks how fast things fall sees nothing.

    Earlier this was a single sine bump over five frames, and it read as a
    teleport: at 12 fps a one-cycle wobble large enough to notice is a body
    appearing a body-width away and coming back. The fix is more cycles over
    more frames at a smaller amplitude -- a visible snake rather than a
    displacement.

    **The wobble is laid out in the camera's image plane**, using the same basis
    the frustum test uses. Perturbing a world axis is a gamble on where the
    camera happens to be: the same offset that snakes across the frame from one
    viewpoint is pure depth from another, and depth on a uniformly lit primitive
    is nearly invisible.
    """

    family = "non_parabolic"
    #: Peak lateral excursion in body radii. Small on purpose -- legibility here
    #: comes from the number of cycles, not from the size of each one.
    AMPLITUDE_RADII = {"weak": 0.5, "medium": 0.9, "strong": 1.5}
    CYCLES = 2.5

    def strong_residual_reference(self, spec) -> float:
        # The least-squares parabola absorbs a little of a symmetric wobble, so
        # the peak measured deviation lands just under the amplitude asked for.
        return float(self.AMPLITUDE_RADII["strong"]) * 0.9

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        bi = traj.index_of(int(actor.segmentation_id))
        top = _geom.surface_top(spec, actor)
        run = _geom.longest_airborne_run(traj, bi, top)
        if run is None or run[1] - run[0] < 4:
            return None
        # The whole airborne stretch, not a slice of it: a serpentine needs room
        # for several cycles, and the residual is fitted over exactly these
        # frames so a parabola through two arcs never enters the comparison.
        t0, t1 = int(run[0]) + 1, int(run[1])
        if t1 - t0 < 3:
            return None

        radius = float(actor.bounding_radius)
        amp = self.AMPLITUDE_RADII[severity_bin] * radius
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "path_warp", "profile": "serpentine",
                    "amplitude_m": amp, "cycles": self.CYCLES,
                    "frames": int(t1 - t0 + 1)},
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
        cycles = float(plan.params["cycles"])
        top = float(plan.notes["surface_top"])
        radius = float(plan.notes["radius"])

        _, _, right, up = _geom.camera_basis(spec)
        # A Hann envelope holds the offset and its slope at zero on both ends,
        # so the body joins and leaves its lawful arc without a corner and the
        # frames outside the window stay untouched.
        u = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
        envelope = np.sin(np.pi * u) ** 2
        phase = 2.0 * np.pi * cycles * u
        offset = (amp * envelope * np.sin(phase))[:, None] * up[None, :] \
            + (0.6 * amp * envelope * np.cos(phase))[:, None] * right[None, :]

        pos = traj.pos[t0:t1 + 1, bi, :].astype(np.float64) + offset
        pos[:, 2] = np.maximum(pos[:, 2], top + radius)
        out.pos[t0:t1 + 1, bi, :] = pos.astype(np.float32)

        vel = out.lin_vel[t0:t1 + 1, bi, :].astype(np.float64)
        vel[0] = (pos[0] - traj.pos[t0 - 1, bi]) / traj.dt
        vel[1:] = (pos[1:] - pos[:-1]) / traj.dt
        out.lin_vel[t0:t1 + 1, bi, :] = vel.astype(np.float32)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Newton1Inertia(Injector):
    """A moving body stops dead with nothing to stop it, and stays stopped.

    Halting only. It used to do double duty -- halt a moving body *or* shove a
    resting one -- and the second half was `phantom_impulse` wearing a different
    label: an uncaused velocity change on a body with nothing touching it. Two
    families that overlap on half their cases cannot be scored independently,
    which is the whole point of keeping them apart, so the shove belongs to
    `phantom_impulse` and the halt belongs here.

    That also settles the family's `kind`. The two branches disagreed --
    stopping is a state that persists, shoving is an event -- so the taxonomy
    could only be right about one of them. Now the window runs to the end of the
    clip, because a body sitting motionless where it should still be sliding is
    violating Newton 1 for every frame it sits there.
    """

    family = "newton1_inertia"
    persistent = True
    #: How much of the body's speed is removed. Below 1.0 it merely slows,
    #: which reads as friction; at 1.0 it stops dead, which reads as wrong.
    HALT_BY_BIN = {"weak": 0.55, "medium": 0.85, "strong": 1.0}
    MOVING = 0.3                                                # m/s
    STRONG_REFERENCE = 3.0                                      # dv / (g*dt)

    def strong_residual_reference(self, spec) -> float:
        return self.STRONG_REFERENCE

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        bi = traj.index_of(int(actor.segmentation_id))
        speed = np.linalg.norm(traj.lin_vel[:, bi, :].astype(np.float64), axis=1)
        moving = np.flatnonzero(speed > self.MOVING)
        moving = moving[(moving >= 1) & (moving < T - 2)]
        if not moving.size:
            return None                 # nothing to halt: not a cell we can build
        t0 = int(max(moving[0], min(moving[-1], T // 3)))

        fraction = float(self.HALT_BY_BIN[severity_bin])
        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, T - 1)],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "velocity_damp", "removed_fraction": fraction},
            magnitude=float(speed[t0] * fraction / max(g_dt, 1e-9)),
            magnitude_unit="dv_over_g_dt", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "removed_fraction": fraction,
                   "speed_at_event": float(speed[t0])})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        keep = 1.0 - float(plan.notes["removed_fraction"])

        if keep <= 1e-6:
            # Frozen exactly where it was. Held rather than re-integrated: a
            # body that stops on a slope must stay on the slope, and gravity
            # would slide it back off.
            out.pos[t0:, bi, :] = traj.pos[t0 - 1, bi][None, :]
            out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
            out.lin_vel[t0:, bi, :] = 0.0
            out.ang_vel[t0:, bi, :] = 0.0
        else:
            # Partly damped: follow the body's own lawful path at a slower rate,
            # so it stays on whatever surface it was travelling along.
            n = traj.num_frames - t0
            u = float(t0 - 1) + np.cumsum(np.full((n,), keep))
            pos = _geom.path_sample(traj.pos[:, bi, :], u)
            out.pos[t0:, bi, :] = pos.astype(np.float32)
            out.quat[t0:, bi, :] = traj.quat[t0 - 1, bi][None, :]
            out.ang_vel[t0:, bi, :] = (traj.ang_vel[t0:, bi, :] * keep)
            self._sync_velocity(traj, out, bi, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(AntiGravity())
register(GlobalGravity())
register(Continuity())
register(NonParabolic())
register(Newton1Inertia())
