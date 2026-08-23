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

    Severity is how long it stays gone: `weak` blinks out briefly, `strong`
    never returns. When the scenario provides an occlusion interval the removal
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
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "remove_body",
                    "bodies": [int(b.segmentation_id) for b in targets],
                    "frames_absent": int(t1 - t0 + 1)},
            magnitude=1.0, magnitude_unit="mass_ratio_removed",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "occluded_at_event": bool(t0 in occ)})

    def apply(self, spec, traj, plan) -> Trajectory:
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
    #: Frames the resize eases over. A body that snaps to a new size between two
    #: frames reads as a cut; one that swells reads as physics going wrong.
    RAMP_FRAMES = 5
    #: Floor on the shrink factor, so a shrinking body stays big enough to have
    #: a mask worth annotating at 128 squared.
    SHRINK_TO = 0.42

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
        ramp = max(2, min(self.RAMP_FRAMES, T - t0))
        occ = spec.notes.get("occluded_frames") or []
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, T - 1)],
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

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        k = float(plan.notes["scale_factor"])
        ramp = int(plan.notes["ramp_frames"])
        T = traj.num_frames

        # Ease in over `ramp` frames, then hold. Smoothstep rather than linear
        # so the change has no visible corner at either end.
        n = T - t0
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        factor = 1.0 + (k - 1.0) * (u * u * (3.0 - 2.0 * u))
        out.scale_mul[t0:, bi, :] = factor[:, None].astype(np.float32)

        # A body that grows while resting on a surface would grow *into* it.
        # Lifting it keeps the clip to one violation: the size is wrong, the
        # floor is still solid. Per frame, because the size now changes per
        # frame -- lifting by the final radius would make it hover on the way.
        top = float(plan.notes["surface_top"])
        r_t = float(traj.radius[bi]) * factor
        out.pos[t0:, bi, 2] = np.maximum(out.pos[t0:, bi, 2], top + r_t)
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
    SEPARATION_BY_BIN = {"weak": 1.2, "medium": 2.4, "strong": 3.8}   # m/s
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
        # Fire at the *start* of an occlusion rather than its middle, so the
        # halves have the whole hidden stretch to separate and are unmistakably
        # two objects by the time they re-emerge. One frame in, because the
        # geometric occlusion test can be off by a frame at either edge.
        occ_run = spec.notes.get("occluded_frames") or []
        if len(occ_run) >= 3:
            t0 = int(occ_run[0]) + 1
        else:
            t0 = _geom.default_event_frame(spec, T)
        if t0 is None or not (1 <= t0 < T - 1):
            return None

        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        unit = np.array([np.cos(heading), np.sin(heading), 0.0])
        twin_spec = [actor, twin]
        half_scale = self.SCALE_BY_BIN[severity_bin]
        strongest = unit * self.SEPARATION_BY_BIN["strong"]
        scale, _ = self._fit_to_frame(
            spec, traj, twin_spec, t0, strongest,
            lambda k: self._split(spec, traj, actor, twin, t0, strongest * k,
                                  self.SCALE_BY_BIN["strong"]))
        push_v = unit * self.SEPARATION_BY_BIN[severity_bin] * scale
        speed = float(np.linalg.norm(push_v))
        push = push_v.tolist()
        occ = spec.notes.get("occluded_frames") or []
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, T - 1)],
            causal_body_ids=[int(actor.segmentation_id),
                             int(twin.segmentation_id)],
            params={"type": "split_body", "separation_speed": speed,
                    "push": push, "scale_factor": half_scale,
                    "frame_fit_scale": scale},
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
        self._rewrite_from(spec, traj, out, actor, t0, v0=v0 + push)
        self._rewrite_from(spec, traj, out, twin, t0, v0=v0 - push,
                           p0=traj.pos[t0 - 1, ai])
        return out

    def apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        twin = next(b for b in spec.bodies if b.dormant)
        out = self._split(spec, traj, actor, twin, plan.t_event,
                          np.asarray(plan.params["push"], np.float64),
                          float(plan.params["scale_factor"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class Fusion(Injector):
    """Two bodies merge into one, which then carries both their volumes.

    The dual of `fission`, and the other half of what video generators do to
    object count: two things that touch and come out as one. Needs two actors,
    so `plan` returns None below that rather than inventing a partner.

    Volume *is* conserved here, unlike at fission's strong bin, because the
    survivor swelling to hold both is what makes the merge readable -- two balls
    becoming one ball the same size as each would look like one of them simply
    vanished, which is `permanence`.
    """

    family = "fusion"
    persistent = True
    #: How close the two must come before merging is plausible, in radii.
    MEET_RADII = {"weak": 3.5, "medium": 2.2, "strong": 1.15}
    RAMP_FRAMES = 4

    def strong_residual_reference(self, spec) -> float:
        return 0.5                      # two bodies become one

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        live = self._all_actors(spec)
        if len(live) < 2:
            return None
        keeper = live[0]
        ki = traj.index_of(int(keeper.segmentation_id))

        # Merge with whichever body comes closest, on the frame it does. A merge
        # staged between two objects that never meet is a vanish and a swell in
        # different corners of the frame, not a fusion.
        best = None
        for other in live[1:]:
            oi = traj.index_of(int(other.segmentation_id))
            gap = np.linalg.norm(
                traj.pos[:, ki, :].astype(np.float64)
                - traj.pos[:, oi, :].astype(np.float64), axis=1)
            reach = float(traj.radius[ki] + traj.radius[oi])
            near = np.flatnonzero(
                gap <= reach * self.MEET_RADII[severity_bin])
            near = near[(near >= 1) & (near < traj.num_frames - 1)]
            if not near.size:
                continue
            t = int(near[0])
            if best is None or t < best[0]:
                best = (t, other, float(gap[t] / max(reach, 1e-9)))
        if best is None:
            return None
        t0, absorbed, meet = best

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0,
            windows=[(t0, traj.num_frames - 1)],
            causal_body_ids=[int(keeper.segmentation_id),
                             int(absorbed.segmentation_id)],
            params={"type": "merge_bodies", "meet_radii": meet,
                    "scale_factor": 2.0 ** (1.0 / 3.0)},
            magnitude=0.5, magnitude_unit="count_ratio",
            severity_bin=severity_bin,
            notes={"radius": float(keeper.bounding_radius),
                   "surface_top": _geom.surface_top(spec, keeper),
                   "keeper_id": int(keeper.segmentation_id),
                   "absorbed_id": int(absorbed.segmentation_id),
                   "sibling_ids": [int(keeper.segmentation_id),
                                   int(absorbed.segmentation_id)],
                   "ramp_frames": self.RAMP_FRAMES})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        ki = traj.index_of(int(plan.notes["keeper_id"]))
        ai = traj.index_of(int(plan.notes["absorbed_id"]))
        t0 = plan.t_event
        k = float(plan.params["scale_factor"])
        ramp = int(plan.notes["ramp_frames"])
        n = traj.num_frames - t0

        out.present[t0:, ai] = False
        u = np.clip((np.arange(n, dtype=np.float64) + 1.0) / ramp, 0.0, 1.0)
        factor = 1.0 + (k - 1.0) * (u * u * (3.0 - 2.0 * u))
        out.scale_mul[t0:, ki, :] = factor[:, None].astype(np.float32)

        top = float(plan.notes["surface_top"])
        r_t = float(traj.radius[ki]) * factor
        out.pos[t0:, ki, 2] = np.maximum(out.pos[t0:, ki, 2], top + r_t)
        self._sync_velocity(traj, out, ki, t0)

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Permanence())
register(Fusion())
register(Immutability())
register(Fission())
