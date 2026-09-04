"""Identity-domain injectors: permanence, immutability, fission.

The three ways a body can stop being itself: it goes away, it changes size, or
it becomes two. All three are the reason the mask is a **union over both
twins** -- each one leaves the invalid render with the wrong number of pixels in
the wrong place, and a mask built from the invalid segmentation alone would be
empty or half-empty exactly while the violation is happening.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class Permanence(Injector):
    """Remove the actor from the scene, ideally while it is occluded.

    Instant: between one frame and the next the body is simply not there. The
    *gradual* version is `dissolve`, and keeping them apart is what lets
    `permanence` own a clean tripwire -- it moves `mass_continuity` and
    `object_count` and touches nothing else, where anything that fades has to
    change size on the way out. Severity is how long it stays gone: `weak`
    blinks out briefly, `strong` never returns. When the scenario provides an occlusion interval the removal
    fires inside it, so the violation is not directly observable and only its
    failure to re-emerge betrays it -- the observability-lag case of PLAN 1.1.
    """

    family = "permanence"
    GONE_FRACTION = {"weak": 0.25, "medium": 0.55, "strong": 1.0}

    def strong_residual_reference(self, spec) -> float:
        return 1.0                       # all of the actor's mass is missing

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._group(spec)
        if not targets:
            return None
        actor = targets[0]
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None

        frac = self.GONE_FRACTION[severity_bin]
        if frac >= 1.0:
            # `strong` means the body never comes back, so it outranks
            # `--window`. Letting the flag truncate it did two bad things at
            # once: the object reappeared, and all three bins collapsed to the
            # same short absence while `magnitude` still claimed 1.0.
            t1 = T - 1
        else:
            n_win = self._window_len(max(1, int(round(frac * (T - t0)))), t0, T)
            t1 = min(T - 1, t0 + n_win - 1)
        occ = spec.notes.get("occluded_frames") or []
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            intervention_windows=[(t0, t0)],
            consequence_windows=[(t0, t1)],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "remove_body",
                    "bodies": [int(b.segmentation_id) for b in targets],
                    "frames_absent": int(t1 - t0 + 1)},
            magnitude=1.0, magnitude_unit="mass_ratio_removed",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "occluded_at_event": bool(t0 in occ)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0, t1 = plan.windows[0]
        for bid in plan.causal_body_ids:
            out.present[t0:t1 + 1, traj.index_of(int(bid))] = False
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Immutability(Injector):
    """The body swells or shrinks over a few frames, and stays the wrong size.

    Sustained rather than instant, and the distinction is not pedantic: the
    resize *starts* at one instant, but the body being the wrong size is an
    ongoing contradiction of its own identity, not a downstream consequence of
    it. Compare `continuity`, where the teleport is the breach and the body
    being somewhere else afterwards is merely what follows.

    **Eased over several frames rather than switched.** A body that changes size
    between two consecutive frames reads as a cut between two different shots;
    one that visibly swells reads as physics going wrong, which is the thing
    being annotated. The scale ramps over `RAMP_FRAMES` and then holds, so the
    residual rises with it and the severity field has a shape.

    Direction is drawn per instance. Shrinking is the harder case -- the mask
    contracts toward nothing just as the violation peaks -- so it is worth
    having in the set rather than shipping a release of nothing but balloons,
    and the floor on `SHRINK_TO` keeps the body from becoming a few pixels.
    """

    family = "immutability"
    persistent = True
    #: Linear scale factor. Below 1.0 shrinks, above grows; the seed picks a
    #: direction so a release is not all balloons.
    SCALE_BY_BIN = {"weak": 1.30, "medium": 1.75, "strong": 2.30}
    #: How long the resize eases over, in SECONDS -- see `Injector._frames_for`.
    #: A body that snaps to a new size between two frames reads as a cut; one
    #: that swells reads as physics going wrong.
    RAMP_SECONDS = 0.4
    #: Floor on the shrink factor, so a shrinking body stays big enough to have
    #: a mask worth annotating at 128 squared.
    SHRINK_TO = 0.42
    #: How many times within a frame the collision shape is rebuilt during the
    #: ramp. See the note in `stage`.
    SWAPS_PER_FRAME = 4

    def _factor(self, severity_bin: str, grow: bool) -> float:
        k = self.SCALE_BY_BIN[severity_bin]
        return k if grow else max(self.SHRINK_TO, 1.0 / k)

    def strong_residual_reference(self, spec) -> float:
        # Direction-aware, because shrinking cannot reach the volume ratio that
        # growing can: |2.3^3 - 1| is 11.2 but the strongest shrink is 0.92.
        # Scoring a shrink against the grow reference reports a maximal
        # violation as severity 0.08.
        grow = bool(self._instance_rng(spec).rand() < 0.55)
        return abs(self._factor("strong", grow) ** 3 - 1.0)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        t0 = _geom.default_event_frame(spec, T)
        if t0 is None:
            return None
        # Direction is a property of the instance, not of the bin -- see
        # `_instance_rng`. Drawing it from `rng` would let weak grow while
        # strong shrinks, which makes the three bins three different violations.
        grow = bool(self._instance_rng(spec).rand() < 0.55)
        k = self._factor(severity_bin, grow)
        ramp = max(2, min(self._frames_for(spec, self.RAMP_SECONDS), T - t0))
        occ = spec.notes.get("occluded_frames") or []
        union, applied, after = self._split_windows(t0, T, ramp)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=union,
            intervention_windows=applied, consequence_windows=after,
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "scale_ramp", "scale_factor": k,
                    "volume_ratio": k ** 3, "ramp_frames": int(ramp),
                    "direction": "grow" if grow else "shrink"},
            # The knob is how far the volume ends up from where it started, so
            # shrinking to 1/k and growing to k report the same magnitude and
            # the bins stay ordered in both directions.
            magnitude=float(abs(k ** 3 - 1.0)), magnitude_unit="volume_ratio",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "scale_factor": k, "ramp_frames": int(ramp),
                   "r_strong": abs(self._factor("strong", grow) ** 3 - 1.0),
                   "occluded_at_event": bool(t0 in occ)})

    #: STAGED, through `stepper.ShapeSwap`. PyBullet cannot rescale a collision
    #: shape in place, which is why this family and `deformation` were the last
    #: two that could only edit a finished trajectory -- and it showed exactly
    #: where you would expect: the render knew the ball had swollen and the
    #: physics did not, so it grew half-way into the barrier beside it and a
    #: shrunken cube stopped touching the floor it stood on.
    #:
    #: What PyBullet *can* do is replace a body mid-run, carrying the pose and
    #: both velocities across. So the size change is real from `t_event` on: the
    #: actor rests at the height its new size dictates, and a wall stops it
    #: where its new surface meets the wall's.
    simulated = True

    def _profile(self, plan, n: int) -> np.ndarray:
        """Scale factor per frame from `t_event`: smoothstep to `k`, then hold.

        Shared by the staged hook and by `_apply`, so the collision shape the
        simulator is given and the scale the renderer draws are the same number
        rather than two implementations of the same intention.
        """
        k = float(plan.notes["scale_factor"])
        ramp = max(1, int(plan.notes["ramp_frames"]))
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        return 1.0 + (k - 1.0) * (u * u * (3.0 - 2.0 * u))

    def stage(self, spec, simulator, objs, plan):
        from ..render import stepper

        actor = next((b for b in spec.bodies
                      if int(b.segmentation_id) == int(plan.causal_body_ids[0])),
                     None)
        if actor is None:
            return ()
        self._swap = None
        swap = stepper.ShapeSwap(simulator, objs, spec, actor)
        if not swap.ok:
            return ()
        self._swap = swap
        t0 = plan.t_event
        profile = self._profile(plan, spec.tier.num_frames - t0)
        spf = stepper.substeps_of(simulator)
        every = max(1, spf // self.SWAPS_PER_FRAME)
        state = {"step": -1}

        def resize(_client, step, frame):
            # Several times a frame, not once. A body that grows in one step per
            # frame arrives inside whatever it is resting on or standing next
            # to, and the solver spends the next frame pushing it back out --
            # measured at the debug tier, a ball swelling to 2.3x buried itself
            # 0.11 m in the floor before recovering. Sub-frame steps make each
            # jump a quarter the size; every step would rebuild the body twenty
            # times a frame for no visible gain.
            if frame < t0 or step <= state["step"] or step % every:
                return
            state["step"] = step
            # `run_from` starts at `t_event`, so the substep counter IS the
            # elapsed time in frames -- no need to reconcile it against `frame`.
            k = min(step / float(spf), profile.shape[0] - 1.0)
            f = float(np.interp(k, np.arange(profile.shape[0]), profile))
            swap.set_scale((f, f, f))

        return (resize,)

    def unstage(self, spec, simulator, objs, plan) -> None:
        swap = getattr(self, "_swap", None)
        if swap is not None:
            swap.restore()
        self._swap = None

    def post_simulate(self, spec, traj_valid, traj_invalid, plan) -> Trajectory:
        """The visual half. PyBullet carries the size, Blender has to be told.

        `splice` copies every non-pose channel from the valid rollout, so
        without this the actor would collide at its new size and render at its
        old one.
        """
        bi = traj_valid.index_of(int(plan.causal_body_ids[0]))
        t0 = plan.t_event
        factor = self._profile(plan, traj_valid.num_frames - t0)
        traj_invalid.scale_mul = np.asarray(traj_invalid.scale_mul).copy()
        traj_invalid.scale_mul[t0:, bi, :] = factor[:, None].astype(np.float32)
        return super().post_simulate(spec, traj_valid, traj_invalid, plan)

    def _apply(self, spec, traj, plan) -> Trajectory:
        """The host-side approximation, for the mock rollout the tests run on.

        The container takes the staged path above; this one exists so every
        `plan()` and every downstream annotation can be exercised without a
        docker round trip. It reproduces the same geometry by hand: the body's
        clearance to its support is preserved rather than its centre height, and
        a body that grew is pushed out of whatever it now overlaps.
        """
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        factor = self._profile(plan, traj.num_frames - t0)
        out.scale_mul[t0:, bi, :] = factor[:, None].astype(np.float32)

        r0 = float(traj.radius[bi])
        _geom.reseat(spec, traj, out, actor, bi, t0, r0 * factor)
        _geom.push_out(spec, traj, out, actor, bi, t0, r0 * factor)
        self._sync_velocity(traj, out, bi, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Fission(Injector):
    """One body becomes two, each with half the volume, and they fly apart.

    Needs the scenario to declare a **dormant understudy** -- a duplicate body
    that is in the scene graph from frame 0 and contributes no pixels until this
    fires. Adding an object to a Kubric scene part-way through a render would
    perturb the render path, and prefix identity is the one thing that cannot be
    traded away.

    Volume is split rather than duplicated, so the violation is unambiguously
    "one thing became two" and not "mass appeared from nowhere" -- which is a
    different family. The residual is therefore the object *count*, and like
    `permanence` it is discrete: the bins vary how far the halves separate, not
    how true the residual is.
    """

    family = "fission"
    persistent = True
    #: Half the pair's separation speed, in m/s. Deliberately modest: you
    #: reported the split reading as "mechanical", two objects already far
    #: apart rather than one object coming apart, and 3.8 m/s at the strong bin
    #: was why -- at 30 fps that is 0.13 m between two consecutive frames, so
    #: the halves are a body-width apart on the frame after the split and the
    #: coming-apart is never on screen. Severity here is how far they end up,
    #: and there is a whole clip for them to get there.
    #: **Where the halves end up**, centre to centre, in radii -- so 2.0 is
    #: exactly touching and anything above it is a visible gap.
    #:
    #: A target, not a speed, and that is the whole design. Expressed as a
    #: separation *velocity* the outcome depends on everything the scenario
    #: happens to be: how much floor friction there is, whether the halves are
    #: airborne when they part, how long the clip has left. The same 3.4 radii/s
    #: that flung `drop`'s halves out of frame in the preview left them 0.50 m
    #: apart in the render, overlapping, because they landed and stopped. So the
    #: intervention was fitted against one number and rendered as another.
    #:
    #: Pushed apart to a target and then brought to relative rest, the answer is
    #: the same in every scene: this far apart, arrived at smoothly. It is also
    #: self-limiting -- neither half can travel more than half the target from
    #: where the body was -- so no frustum fit is needed to keep them in shot,
    #: which removes the machinery that was clamping the violation to a
    #: quarter of its nominal strength.
    SEPARATION_BY_BIN = {"weak": 2.6, "medium": 3.6, "strong": 5.0}
    #: How long the division takes. THE fix for "mechanical, not smooth": an
    #: instantaneous velocity change puts the halves a body-width apart on the
    #: frame after the split, so the coming apart is never on screen. Half this
    #: long accelerating and half decelerating, the division is the thing you
    #: watch, and it ends with the halves at rest relative to each other.
    SPLIT_SECONDS = 0.45
    #: Both halves keep their full size, in every bin.
    #:
    #: Shrinking them to conserve volume is the tempting choice and it costs
    #: more than it buys. Two 79%-scale halves sitting close together read as
    #: one blurry object at 128 px, so the count -- the actual claim -- becomes
    #: the hardest thing to see. Worse, it made the family score on *two*
    #: laws: `object_count` in every bin and `shape_continuity` in the weak and
    #: medium ones, so a clip's orthogonality depended on its severity. A
    #: family should break the same law at all three strengths and differ only
    #: in how hard, which here is how far apart the halves end up.
    SCALE_BY_BIN = {"weak": 1.0, "medium": 1.0, "strong": 1.0}

    def strong_residual_reference(self, spec) -> float:
        return 1.0                       # exactly one extra body exists

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        twin = next((b for b in spec.bodies if b.dormant), None)
        if actor is None or twin is None:
            return None
        T = traj.num_frames
        # Fire just BEFORE an occlusion, not inside it. Splitting while hidden
        # was the original choice -- it gives the halves the whole hidden
        # stretch to separate -- and you reported the cost: on `occluder_pass`
        # the one thing the family exists to show, an object coming apart,
        # happens where nobody can see it, and the clip reads as two objects
        # emerging from behind a screen one object went into. Which is
        # `permanence`'s picture, not this one's.
        occ_run = spec.notes.get("occluded_frames") or []
        if len(occ_run) >= 3:
            t0 = max(1, int(occ_run[0]) - self._frames_for(spec, 0.25))
        else:
            # And with room for the halves to actually part. Splitting on the
            # frame the body lands pins both halves under friction before they
            # have moved a body-width, which is how a split ends up looking
            # like one blurry object rather than two.
            t0 = _geom.acting_frame(spec, traj, int(actor.segmentation_id), T)
        if t0 is None or not (1 <= t0 < T - 1):
            return None

        # ACROSS THE SCREEN, and HORIZONTAL. Two failure modes bracket this
        # choice and the middle is narrow.
        #
        # A heading drawn uniformly on the horizontal circle -- the first
        # version -- comes out near the viewing axis often enough that the
        # halves separate in depth, one behind the other, and the clip shows a
        # single object that briefly looks lumpy. So the split must live in the
        # image plane.
        #
        # Perpendicular to the body's screen-space velocity -- the second
        # version -- was worse on exactly the scenarios that matter. A ball
        # rolling across frame has its screen velocity along the horizontal, so
        # the perpendicular is *vertical*: one half hops, the other is pressed
        # into the floor it is already resting on, and they end up in the same
        # place. Measured on `barrier_pass`, total separation over the whole
        # clip was 0.11 m against a body 0.50 m wide -- which is the "not
        # visible" you reported, and the "same line, occluding each other" on
        # `occluder_pass`.
        #
        # Horizontal and as close to the camera's right as the world allows is
        # the one direction that is always in the image plane and never fights
        # the floor. Where the body is also travelling that way the halves
        # string out fore and aft, which still reads as coming apart.
        _, _, right, _ = _geom.camera_basis(spec)
        flat = np.array([right[0], right[1], 0.0])
        n = float(np.linalg.norm(flat))
        unit = (flat / n if n > 1e-6 else np.array([1.0, 0.0, 0.0]))
        # Sign per instance -- a property of the SCENE, not of the bin, or weak,
        # medium and strong would be three different violations whose magnitudes
        # are not comparable.
        unit = unit * float(self._instance_rng(spec).choice([-1.0, 1.0]))
        twin_spec = [actor, twin]
        half_scale = self.SCALE_BY_BIN[severity_bin]
        radius = float(actor.bounding_radius)
        # No frustum fit. The target is bounded, so neither half can travel more
        # than half of it from where the body was -- there is nothing to keep in
        # shot that is not already in shot.
        target = float(self.SEPARATION_BY_BIN[severity_bin]) * radius
        push = (unit * target).tolist()
        occ = spec.notes.get("occluded_frames") or []
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, T - 1)],
            intervention_windows=[(t0, t0)],
            consequence_windows=[(t0, T - 1)],
            causal_body_ids=[int(actor.segmentation_id),
                             int(twin.segmentation_id)],
            params={"type": "split_body", "separation_m": float(target),
                    "separation_radii": float(
                        self.SEPARATION_BY_BIN[severity_bin]),
                    "split_seconds": float(self.SPLIT_SECONDS),
                    "push": push, "scale_factor": half_scale},
            magnitude=2.0, magnitude_unit="count_ratio",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "twin_id": int(twin.segmentation_id),
                   "sibling_ids": [int(actor.segmentation_id),
                                   int(twin.segmentation_id)],
                   "occluded_at_event": bool(t0 in occ)})

    def _split(self, spec, traj, actor, twin, t0: int, push,
               k: float) -> Trajectory:
        out = self._clone(traj)
        ai = traj.index_of(int(actor.segmentation_id))
        ti = traj.index_of(int(twin.segmentation_id))
        push = np.asarray(push, np.float64)

        out.present[t0:, ti] = True
        out.scale_mul[t0:, ai, :] = k
        out.scale_mul[t0:, ti, :] = k

        # The understudy starts exactly where the original was, so the split
        # looks like one body coming apart rather than a second one arriving.
        # Its own frames before t0 are left alone -- writing the original's pose
        # into frame t0-1 would break prefix identity on the frame before the
        # event, even though the body is invisible there.
        out.quat[t0:, ti, :] = traj.quat[t0:, ai, :]
        v0 = traj.lin_vel[t0 - 1, ai].astype(np.float64)
        # The same ACCELERATE-THEN-DECELERATE profile the staged hook applies,
        # expressed as an extra acceleration over the split window. Half the
        # window pushing apart and half braking, so the halves arrive `push`
        # apart and at rest relative to each other rather than still flying.
        # The two implementations must agree, because this one is what the mock
        # rollout in `tests/` exercises.
        n = traj.num_frames - t0
        g = np.tile(np.asarray(traj.gravity, np.float64)[None, :], (n, 1))
        bump = np.zeros_like(g)
        bump[:min(self._split_frames(traj.fps), n)] = self._split_accel(push)
        half = self._split_frames(traj.fps) // 2
        bump[half:min(self._split_frames(traj.fps), n)] *= -1.0
        self._rewrite_from(spec, traj, out, actor, t0, v0=v0,
                           g_per_frame=g + bump)
        self._rewrite_from(spec, traj, out, twin, t0, v0=v0,
                           p0=traj.pos[t0 - 1, ai], g_per_frame=g - bump)
        return out

    def _split_frames(self, fps: float) -> int:
        """Frames the division occupies -- always even, so the two halves of
        the accelerate/brake profile are the same length."""
        n = max(2, int(round(float(self.SPLIT_SECONDS) * float(fps))))
        return n + (n % 2)

    def _split_accel(self, separation) -> np.ndarray:
        """Acceleration each half needs to end `separation` apart, at rest.

        Each half covers half the gap. Accelerating for T/2 and braking for T/2
        covers `a*(T/2)^2`, so `a = 2*separation / T^2`.
        """
        sep = np.asarray(separation, np.float64)
        t = float(self.SPLIT_SECONDS)
        return 2.0 * sep / max(t * t, 1e-9)

    #: STAGED. Both halves are bodies the solver owns from `t_event` on, so
    #: whatever they hit, they hit for real. On `collision` the edited version
    #: could not do that: the striker split, both halves went elsewhere, and the
    #: bystander guard then had to hold the ball they no longer struck perfectly
    #: still -- which is what you saw, a target that never reacts to anything.
    #: Staged, a half that reaches the target moves it and a half that misses
    #: leaves it alone, and neither outcome has to be decided in advance.
    simulated = True

    def _twin_of(self, spec):
        return next((b for b in spec.bodies if b.dormant), None)

    def stage(self, spec, simulator, objs, plan):
        import pybullet as pb

        from ..render import stepper

        actor = next((b for b in spec.bodies
                      if int(b.segmentation_id) == int(plan.causal_body_ids[0])),
                     None)
        twin = self._twin_of(spec)
        if actor is None or twin is None:
            return ()
        # Cleared first, never assumed empty: an early return below would
        # otherwise leave the PREVIOUS variant's pair and proxy sitting on the
        # instance, and `unstage` would restore a body this plan never touched.
        self._pair = None
        self._swap = None
        idx = stepper.pybullet_index(simulator, objs, spec,
                                     int(actor.segmentation_id))
        if idx is None:
            return ()
        push = np.asarray(plan.params["push"], np.float64)
        pos, quat = pb.getBasePositionAndOrientation(idx)
        vel, ang = pb.getBaseVelocity(idx)
        v = np.asarray(vel, np.float64)

        # The understudy is declared `scripted`, so the simulator holds it at
        # mass 0 and no change of mass revives it -- see `ShapeSwap.dynamic`.
        swap = stepper.ShapeSwap(simulator, objs, spec, twin, dynamic=True)
        if not swap.ok:
            return ()
        # Both halves start on the ORIGINAL's velocity. The separation is not
        # handed to them here; it is pushed into them over `SPLIT_SECONDS` by
        # the hook below, which is what makes the split something you watch
        # rather than something you find already finished.
        swap.set_scale((1.0, 1.0, 1.0), pose=(pos, quat),
                       velocity=(v.tolist(), list(ang)))
        if swap.proxy is None:
            swap.restore()
            return ()
        # The two halves start in exactly the same place, so they must not see
        # each other: a pair overlapping completely resolves as an explosion,
        # which is a position jump large enough to read as a teleport. They are
        # two parts of one body that has just come apart; everything else in the
        # scene stays solid to both.
        self._pair = (idx, swap.proxy)
        pb.setCollisionFilterPair(idx, swap.proxy, -1, -1, 0)
        self._swap = swap

        # ...and solid to each other again the moment they have parted. Left
        # suppressed for the whole clip, two halves that come to rest near each
        # other simply share the space -- on `drop` they finished 0.37 m apart
        # with a body width of 0.78, which renders as one lumpy object and
        # undoes the only claim the family makes. Restored on separation rather
        # than on a timer, for the same reason `solidity` does it: how long two
        # bodies take to clear each other is a fact about the run.
        # The separation, as a FORCE over time: apart for half the window, then
        # braking for the other half, so the halves arrive `push` apart and at
        # rest relative to each other. Re-applied every substep because PyBullet
        # clears accumulated external forces after each step -- the same reason
        # `antigravity` has a hook at all.
        #
        # The same profile `_split` integrates by hand, so the clip that renders
        # is the one the plan describes.
        mass = float(getattr(actor, "mass", 1.0))
        n_split = self._split_frames(spec.tier.fps)
        force = self._split_accel(push) * mass
        # A generous cap, not a schedule. The gap decides when to stop pushing;
        # this only stops the hook running for the whole clip if the target
        # turns out to be unreachable.
        t_push_end = plan.t_event + 3 * n_split
        proxy = swap.proxy
        state = {"restored": False}

        target = float(np.linalg.norm(push))
        axis = push / max(target, 1e-9)

        def split(_client, _step, frame):
            if frame < t_push_end:
                pa, _ = pb.getBasePositionAndOrientation(idx)
                pb_pos, _ = pb.getBasePositionAndOrientation(proxy)
                gap = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb_pos)))
                rel = float((np.asarray(pb.getBaseVelocity(idx)[0])
                             - np.asarray(pb.getBaseVelocity(proxy)[0])) @ axis)
                # CLOSED LOOP, because the open-loop trapezoid overshot. The
                # halves are also being slowed by whatever they are resting on,
                # so by the time the brake half of the window arrives they are
                # separating more slowly than the profile assumed -- and the
                # full braking impulse then drove them back TOGETHER. Measured
                # on `drop`: the strong bin reached 1.97 m, its target, and
                # closed again to 0.79 m.
                #
                # Braking only while they are still separating, and never once
                # the target is reached, cannot converge them.
                # Driven by the GAP, not by a clock. A timed profile assumes
                # both halves are free to move, and on `barrier_pass` they are
                # not: the only horizontal direction that shows on screen is
                # the one the ball is already travelling, and there is a wall
                # across it. The forward half stopped against the wall, the
                # brake phase arrived on schedule and killed what separation
                # there was, and the halves finished 0.39 m apart inside a body
                # 0.50 m wide -- your "not visible".
                #
                # Pushing until the gap is actually reached lets the free half
                # take up whatever the blocked one cannot.
                if gap < target:
                    way = 1.0
                else:
                    way = -1.0 if rel > 1e-3 else 0.0
                if way == 0.0:
                    return
                for body_idx, sign in ((idx, 1.0), (proxy, -1.0)):
                    at, _ = pb.getBasePositionAndOrientation(body_idx)
                    pb.applyExternalForce(body_idx, -1,
                                          (force * sign * way).tolist(),
                                          list(at), pb.WORLD_FRAME)
                return
            # ...and solid to each other again the moment they have parted.
            # Left suppressed for the whole clip, two halves that come to rest
            # near each other simply share the space, which renders as one
            # lumpy object and undoes the only claim the family makes. Restored
            # on separation rather than on a timer, for the same reason
            # `solidity` does it: how long two bodies take to clear each other
            # is a fact about the run.
            if state["restored"]:
                return
            if pb.getClosestPoints(idx, proxy, distance=0.0):
                return
            pb.setCollisionFilterPair(idx, proxy, -1, -1, 1)
            state["restored"] = True

        return (split,)

    def unstage(self, spec, simulator, objs, plan) -> None:
        import pybullet as pb

        pair = getattr(self, "_pair", None)
        if pair is not None:
            try:
                pb.setCollisionFilterPair(pair[0], pair[1], -1, -1, 1)
            except Exception:                                 # noqa: BLE001
                pass
        self._pair = None
        swap = getattr(self, "_swap", None)
        if swap is not None:
            swap.restore()
        self._swap = None

    def post_simulate(self, spec, traj_valid, traj_invalid, plan) -> Trajectory:
        """Switch the understudy on. PyBullet moved it; nothing told the render.

        `present` is not a physical channel, so `splice` carries the valid
        rollout's -- in which the understudy does not exist -- straight through.
        """
        twin = self._twin_of(spec)
        if twin is not None:
            ti = traj_valid.index_of(int(twin.segmentation_id))
            traj_invalid.present = np.asarray(traj_invalid.present).copy()
            traj_invalid.present[plan.t_event:, ti] = True
        return super().post_simulate(spec, traj_valid, traj_invalid, plan)

    def _apply(self, spec, traj, plan) -> Trajectory:
        """The host-side approximation, for the mock rollout the tests run on."""
        actor = self._primary(spec)
        twin = self._twin_of(spec)
        out = self._split(spec, traj, actor, twin, plan.t_event,
                          np.asarray(plan.params["push"], np.float64),
                          float(plan.params["scale_factor"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Fusion(Injector):
    """Bodies touch and come out as one: the absorbed shrinks into its
    neighbour and is gone.

    The dual of `fission`, and the other half of what video generators do to
    object count.

    **Nothing changes size.** Swelling the survivor to conserve volume read as
    one body vanishing while an unrelated one grew -- `permanence` and
    `immutability` in the same frame, which is neither. Shrinking the absorbed
    one away was the second attempt and read as it evaporating rather than
    merging. What makes a merge legible is simply the *approach*: two identical
    balls slide together until one is inside the other, at which point they are
    visually one ball of the same size, and the absorbed one is removed. Every
    body keeps its own size throughout, which is also what keeps the family
    scoring on `object_count` and nothing else.

    Acts on as many pairs as the scene offers. One merge among forty grains is
    invisible; a third of a pour collapsing into its neighbours is the point.
    """

    family = "fusion"
    persistent = True
    #: How close two bodies must come to be merge candidates, in contact radii.
    MEET_RADII = {"weak": 3.5, "medium": 2.4, "strong": 1.6}
    #: Share of the clip the absorbed body takes to slide inside its neighbour.
    #: Long enough to read as travel rather than a cut, and it scales with the
    #: tier so a longer clip gets a slower, smoother merge.
    DRAW_IN_FRACTION = 0.18
    DRAW_IN_MIN = 3

    def strong_residual_reference(self, spec) -> float:
        return 0.5                      # two bodies become one

    # ------------------------------------------------------------------ #
    def _pairs(self, spec, traj, severity_bin):
        """(keeper, absorbed, frame) for every merge this scene can host.

        Greedy nearest-neighbour over the actors, each body used once. Pairing
        by proximity rather than by declaration order is what makes the merge
        look like a merge: two bodies that never come near each other cannot
        plausibly become one.
        """
        live = [b for b in self._all_actors(spec)]
        if len(live) < 2:
            return []
        want = max(1, len(self._group(spec)) // 2)
        reach_mult = self.MEET_RADII[severity_bin]

        cand = []
        for i, a in enumerate(live):
            ia = traj.index_of(int(a.segmentation_id))
            for b in live[i + 1:]:
                ib = traj.index_of(int(b.segmentation_id))
                gap = np.linalg.norm(
                    traj.pos[:, ia, :].astype(np.float64)
                    - traj.pos[:, ib, :].astype(np.float64), axis=1)
                reach = float(traj.radius[ia] + traj.radius[ib]) * reach_mult
                near = np.flatnonzero(gap <= reach)
                near = near[(near >= 1) & (near < traj.num_frames - 2)]
                if near.size:
                    cand.append((float(gap[near[0]]), int(near[0]), a, b))
        cand.sort(key=lambda x: (x[1], x[0]))

        used, pairs = set(), []
        for _, frame, a, b in cand:
            ka, kb = int(a.segmentation_id), int(b.segmentation_id)
            if ka in used or kb in used:
                continue
            used.update((ka, kb))
            pairs.append((a, b, frame))
            if len(pairs) >= want:
                break
        return pairs

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        pairs = self._pairs(spec, traj, severity_bin)
        if not pairs:
            return None
        t0 = min(f for _, _, f in pairs)
        keepers = [int(a.segmentation_id) for a, _, _ in pairs]
        absorbed = [int(b.segmentation_id) for _, b, _ in pairs]

        draw_in = max(self.DRAW_IN_MIN,
                      int(round(self.DRAW_IN_FRACTION * traj.num_frames)))
        # `draw_in + 1`, because the absorbed body is removed on the frame AFTER
        # the draw-in completes -- that removal is the moment the object count
        # actually changes, and it is what `object_count` measures. With the
        # window ending one frame earlier the residual rose to 0.5 exactly one
        # frame outside the window it was scored on, and every fusion cell
        # reported severity 0.000 with a perfectly correct residual sitting
        # beside it.
        union, applied, after = self._split_windows(
            t0, traj.num_frames, draw_in + 1)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=keepers + absorbed,
            params={"type": "merge_bodies", "pairs": len(pairs),
                    "meet_radii": self.MEET_RADII[severity_bin]},
            magnitude=float(len(pairs)) / max(len(keepers) + len(absorbed), 1),
            magnitude_unit="count_ratio", severity_bin=severity_bin,
            notes={"radius": float(pairs[0][0].bounding_radius),
                   "surface_top": _geom.surface_top(spec, pairs[0][0]),
                   "keepers": keepers, "absorbed": absorbed,
                   "merge_frames": [int(f) for _, _, f in pairs],
                   "sibling_ids": keepers + absorbed,
                   "draw_in": max(self.DRAW_IN_MIN,
                                  int(round(self.DRAW_IN_FRACTION
                                            * traj.num_frames)))})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        draw = int(plan.notes["draw_in"])
        T = traj.num_frames
        for keep_id, gone_id, frame in zip(plan.notes["keepers"],
                                           plan.notes["absorbed"],
                                           plan.notes["merge_frames"]):
            ki = traj.index_of(int(keep_id))
            gi = traj.index_of(int(gone_id))
            t = int(frame)
            n = min(draw, T - t)
            if n <= 0:
                continue
            u = (np.arange(n, dtype=np.float64) + 1.0) / n
            ease = u * u * (3.0 - 2.0 * u)
            # Drawn in along the line of centres, shrinking as it goes, so the
            # two are unmistakably overlapping before either one is gone.
            src = traj.pos[t:t + n, gi, :].astype(np.float64)
            dst = out.pos[t:t + n, ki, :].astype(np.float64)
            out.pos[t:t + n, gi, :] = (
                src + (dst - src) * ease[:, None]).astype(np.float32)
            self._sync_velocity(traj, out, gi, t)
            if t + n < T:
                out.present[t + n:, gi] = False

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Dissolve(Injector):
    """The body dwindles away to nothing and is gone.

    `permanence`'s gradual counterpart, and a separate family rather than a mode
    of it because the two make different claims and a benchmark should be able
    to ask them separately: one is "it was there and then it was not", the other
    is "it faded out of existence". They also differ in what a model has to
    notice -- an instant removal is a single-frame discontinuity, a dissolve is
    a trend across several.

    **Optical, not geometric**: the body keeps its size and shape and simply
    stops being opaque. The renderer mixes its shaded surface with a Transparent
    BSDF and keyframes the blend -- measured working in the pinned image, see
    `render/probe_opacity.py`, which also records that the Principled BSDF's own
    alpha input does nothing here and why it took three attempts to find that
    out.

    Fading alone is not enough to *remove* it, though, so the body is dropped
    from the scene once it is invisible. Cryptomatte tracks geometry, so a
    perfectly transparent body still reports every one of its pixels in
    `seg.npz`; leaving it there would mean the mask claimed an object that is no
    longer visible anywhere in the frame.

    Distinct from `immutability`, which settles at a new size and stays there,
    and from `permanence`, which is a single-frame cut. Fading changes neither
    shape nor position, so it moves no law but its own.
    """

    family = "dissolve"
    persistent = True
    #: Share of the clip the body takes to fade out. Slower reads more clearly
    #: as a dissolve and less as a cut, and it scales with the tier.
    FADE_BY_BIN = {"weak": 0.45, "medium": 0.28, "strong": 0.16}
    FADE_MIN = 3

    def strong_residual_reference(self, spec) -> float:
        return 1.0                       # all of it, gone

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        targets = self._group(spec)
        if not targets:
            return None
        T = traj.num_frames
        # IN PLAIN SIGHT, unlike `permanence` -- and actively so, not merely by
        # not seeking an occlusion. See `unoccluded_event_frame`.
        #
        # Placed against the SHORTEST fade, so all three bins share one
        # `t_event` and stay comparable; a longer weak fade may run on into the
        # occlusion, which is fine because its first and most legible frames are
        # the ones in the open.
        t0 = _geom.unoccluded_event_frame(
            spec, T, max(self.FADE_MIN,
                         int(round(self.FADE_BY_BIN["strong"] * T))))
        if t0 is None or not (1 <= t0 < T - 1):
            return None
        fade = max(self.FADE_MIN,
                   int(round(self.FADE_BY_BIN[severity_bin] * T)))
        fade = max(1, min(fade, T - t0))
        occ = spec.notes.get("occluded_frames") or []
        union, applied, after = self._split_windows(t0, T, fade)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=union, intervention_windows=applied,
            consequence_windows=after,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "dissolve_body", "fade_frames": int(fade)},
            magnitude=1.0, magnitude_unit="opacity_lost",
            severity_bin=severity_bin,
            notes={"radius": float(targets[0].bounding_radius),
                   "surface_top": _geom.surface_top(spec, targets[0]),
                   "fade_frames": int(fade),
                   "sibling_ids": [int(b.segmentation_id) for b in targets],
                   "occluded_at_event": bool(t0 in occ)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        t0 = plan.t_event
        T = traj.num_frames
        n = min(int(plan.notes["fade_frames"]), T - t0)
        u = (np.arange(n, dtype=np.float64) + 1.0) / n
        alpha = 1.0 - u * u * (3.0 - 2.0 * u)      # 1 -> 0, smooth at both ends

        for bid in plan.causal_body_ids:
            bi = traj.index_of(int(bid))
            out.opacity[t0:t0 + n, bi] = alpha.astype(np.float32)
            if t0 + n < T:
                out.opacity[t0 + n:, bi] = 0.0
                # Removed once it is invisible, so it leaves the segmentation
                # too. Transparency alone does not: cryptomatte tracks geometry,
                # so a fully faded body still reports every one of its pixels.
                out.present[t0 + n:, bi] = False

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Permanence())
register(Dissolve())
register(Fusion())
register(Immutability())
register(Fission())
