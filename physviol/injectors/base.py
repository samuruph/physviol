"""Injectors -- docs/PLAN.md Part 2 / Part 4.

An injector edits a **trajectory**, never the simulator's rng. That is what makes
the valid prefix bit-identical by construction rather than by hope: the invalid
trajectory literally *is* the valid one up to `t_event`.

Two-phase API, because an injector generally has to look at the valid rollout
before it can choose a sensible `t_event` (you cannot break a contact before the
contact happens):

    plan(spec, valid_traj, rng, severity_bin) -> InterventionPlan | None
    apply(spec, valid_traj, plan)             -> invalid Trajectory

py3.9-compatible: runs inside the container.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom


@dataclass
class InterventionPlan:
    """What we are about to do, decided before we do it."""

    family: str
    kind: str                                   # instant | sustained | repeated
    t_event: int
    windows: List[Tuple[int, int]]              # inclusive [start, end] frame ranges
    causal_body_ids: List[int]
    params: Dict[str, Any]
    magnitude: float                            # the knob we turned -- exact
    magnitude_unit: str
    severity_bin: str
    spatial_extent: str = "local"               # local | global
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def t_end(self) -> int:
        return max(e for _, e in self.windows)

    def active_mask(self, num_frames: int) -> np.ndarray:
        """Rasterise `windows` into the per-frame boolean timeline (PLAN 3.2)."""
        a = np.zeros((num_frames,), dtype=bool)
        for s, e in self.windows:
            a[max(0, s):min(num_frames, e + 1)] = True
        return a

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family, "kind": self.kind,
            "t_event_frame": self.t_event, "t_end_frame": self.t_end,
            "violation_windows": [list(w) for w in self.windows],
            "causal_body_ids": list(self.causal_body_ids),
            "spatial_extent": self.spatial_extent,
            "intervention": {
                "type": self.params.get("type", self.family),
                "params": {k: v for k, v in self.params.items() if k != "type"},
                "magnitude": self.magnitude,
                "magnitude_unit": self.magnitude_unit,
                "severity_bin": self.severity_bin,
            },
            "notes": self.notes,
        }


class Injector:
    """Base class. `family` must be a key in taxonomy.FAMILIES."""

    family: str = "unnamed"

    #: Set by the caller to request a uniform violation duration, in frames.
    #: Honoured by `sustained` families; `instant` ones ignore it, because
    #: stretching a teleport's window would conflate the law breach with its
    #: consequences -- the object stays displaced afterwards, but the
    #: continuity law is broken only at the jump.
    window_frames: Optional[int] = None

    #: The violation is a *state*, not an event: once it starts it holds to the
    #: end of the clip and `--window` does not apply. A detached shadow does not
    #: reattach; a hovering body does not settle; a body that split stays split.
    #: Truncating the window on these families does not shorten the violation,
    #: it just stops annotating the tail of it -- which is how `shadow` came to
    #: ship a detached shadow across frames `active` said were lawful.
    persistent: bool = False

    def _window_len(self, default: int, t0: int, num_frames: int) -> int:
        if self.persistent:
            return max(1, num_frames - t0)
        n = self.window_frames or default
        return max(1, min(int(n), num_frames - t0))

    def plan(self, spec, traj: Trajectory, rng: np.random.RandomState,
             severity_bin: str) -> Optional[InterventionPlan]:
        raise NotImplementedError

    def apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        raise NotImplementedError

    def strong_residual_reference(self, spec) -> float:
        """The residual this family produces at its `strong` bin, in the units of
        its law. This is `r_strong` in the bounded score of PLAN 3.4 step 3, and
        it is what makes severity comparable across families whose physical
        units are not.

        Families whose knob is exact by construction -- a prescribed penetration
        depth, a gravity scale -- can answer from the spec alone. Families whose
        effect depends on the rollout (how fast was it going when it stopped?)
        cannot, so they measure it instead: `plan()` builds the strong-bin edit,
        runs the law over it, and records the answer as `notes["r_strong"]`,
        which the annotation pipeline prefers over this method. Measuring it
        even while planning a *weak* clip is the point -- the reference has to be
        the same for all three bins or the bins are not comparable.
        """
        raise NotImplementedError

    def _measure(self, traj: Trajectory, body_id: int, law_name: str,
                 ctx: Dict[str, Any]) -> float:
        """Peak residual of `law_name` on this trajectory. See above."""
        from ..residuals import laws
        r = laws.get(law_name)(traj, traj.index_of(int(body_id)), dict(ctx))
        return float(np.max(r)) if r.size else 0.0

    # -- helpers shared by subclasses -------------------------------------
    @staticmethod
    def _clone(traj: Trajectory) -> Trajectory:
        out = copy.copy(traj)
        out.pos = traj.pos.copy()
        out.quat = traj.quat.copy()
        out.lin_vel = traj.lin_vel.copy()
        out.ang_vel = traj.ang_vel.copy()
        out.present = traj.present.copy()
        out.scale_mul = traj.scale_mul.copy()
        out.meta = dict(traj.meta)
        return out

    # -- who the intervention acts on -------------------------------------
    def _instance_rng(self, spec) -> np.random.RandomState:
        """A generator keyed to the *scene*, not to the severity bin.

        `plan()` is handed an rng seeded per (family, severity) so that adding a
        family to a batched run cannot perturb the others. That is right for
        anything the bin should vary, and wrong for anything it must not: a
        choice like "does this body grow or shrink" has to come out the same for
        weak, medium and strong, or the three bins of one cell are three
        different violations and their magnitudes stop being comparable.
        """
        key = (int(spec.seed) * 2654435761
               + zlib.crc32(self.family.encode())) % (2 ** 31 - 1)
        return np.random.RandomState(key)

    def _primary(self, spec):
        """The one body most families act on. `None` if the scenario has no actor.

        A scenario may override the choice per family through
        `notes["family_targets"]`, which is how a scene says something about
        itself that the injector could not work out: `pyramid_impact` points
        `solidity` at a struck sphere rather than the falling cube, and
        `granular_pour` points every single-grain family at the grains that
        start highest, because any other grain spends the clip buried and its
        violation would be perfectly scored and completely invisible.

        Dormant understudies are skipped: they exist in the scene graph so
        `fission` has something to switch on, and until it does they are not
        part of the scene.
        """
        want = (spec.notes.get("family_targets") or {}).get(self.family)
        if want:
            by_id = {int(b.segmentation_id): b for b in spec.bodies}
            preferred = [by_id[int(i)] for i in want if int(i) in by_id]
            if preferred:
                return preferred[0]
        live = [b for b in _geom.actors(spec) if not b.dormant]
        return live[0] if live else None

    def _group(self, spec):
        """The bodies this family should act on: one, or several.

        A violation applied to one grain of forty is perfectly annotated and
        impossible to see -- the mask is a handful of pixels somewhere in a
        pile. Scenes that are made of many interchangeable bodies say so, and
        families that can act on a set do.

        Two ways a scenario can ask. `family_targets[family]` names the bodies
        outright, which is how `granular_pour` steers single-body families onto
        the grains that end up on top of the pile rather than buried in it.
        `group_fraction` asks for a share of the actors, chosen per scene so the
        three severity bins act on the same bodies.
        """
        live = self._all_actors(spec)
        if not live:
            return []
        named = (spec.notes.get("family_targets") or {}).get(self.family)
        if named:
            by_id = {int(b.segmentation_id): b for b in spec.bodies}
            chosen = [by_id[int(i)] for i in named if int(i) in by_id]
            if chosen:
                return chosen
        frac = float(spec.notes.get("group_fraction") or 0.0)
        if frac <= 0.0 or len(live) < 4:
            return live[:1]
        k = int(np.clip(round(frac * len(live)), 2, len(live)))
        picks = self._instance_rng(spec).choice(len(live), size=k, replace=False)
        return [live[i] for i in sorted(int(x) for x in picks)]

    @staticmethod
    def _all_actors(spec):
        """Every actor, for the families that act on a whole medium rather than
        on one culprit -- `antigravity` over a pour, `global_gravity` over a
        scene, an assembly whose parts must move together."""
        return [b for b in _geom.actors(spec) if not b.dormant]

    @staticmethod
    def _ballistic(p0: np.ndarray, v0: np.ndarray, g: np.ndarray,
                   dt: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Free-flight continuation: `n` frames of pos/vel starting one step after p0."""
        t = (np.arange(1, n + 1, dtype=np.float64) * dt)[:, None]
        pos = p0[None, :] + v0[None, :] * t + 0.5 * g[None, :] * t ** 2
        vel = v0[None, :] + g[None, :] * t
        return pos.astype(np.float32), vel.astype(np.float32)

    @staticmethod
    def _ballistic_with_floor(p0: np.ndarray, v0: np.ndarray, g: np.ndarray,
                              dt: float, n: int, floor_z, radius: float,
                              restitution: float, substeps: int = 24):
        """Free flight that still respects the ground.

        Needed after a violation window closes: the actor is legal again, so it
        must bounce rather than sink through on a later frame. Sub-steps so the
        bounce instant is not quantised to the frame rate.

        `floor_z` may be a callable (x, y) -> z. Scenarios with a raised surface
        need it: a mug kicked off a table must fall to the floor once it clears
        the edge, not glide along an infinite plane at table height, which is a
        solidity violation nobody asked for.
        """
        return Injector._integrate_profile(
            p0, v0, np.tile(np.asarray(g, np.float64)[None, :], (n, 1)),
            dt, floor_z, radius, restitution, substeps)

    @staticmethod
    def _integrate_profile(p0: np.ndarray, v0: np.ndarray, g_per_frame: np.ndarray,
                           dt: float, floor_z, radius: float,
                           restitution: float, substeps: int = 24,
                           obstacles=None, t_start: float = 0.0):
        """Integrate under a *time-varying* gravity, respecting a ground plane.

        Two things this buys over plain ballistics:

        * the floor stays solid. An injector that only meant to change gravity
          must not also drop the actor through the table -- that would put two
          violations in one clip and annotate only one of them.
        * gravity may differ per frame, so an intervention can ramp in and out
          instead of being a step that lasts to the end of the clip. That is
          what makes the residual -- and therefore the severity map -- vary over
          time rather than sitting at one value.
        """
        p = np.asarray(p0, np.float64).copy()
        v = np.asarray(v0, np.float64).copy()
        n = int(g_per_frame.shape[0])
        h = dt / float(substeps)
        ground = floor_z if callable(floor_z) else (lambda x, y: float(floor_z))
        pos = np.zeros((n, 3), np.float64)
        vel = np.zeros((n, 3), np.float64)
        for f in range(n):
            g = np.asarray(g_per_frame[f], np.float64)
            for k in range(substeps):
                v = v + g * h
                p = p + v * h
                fz = ground(p[0], p[1])
                if p[2] - radius <= fz and v[2] < 0.0:
                    p[2] = fz + radius
                    v[2] = -v[2] * restitution
                    if abs(v[2]) < 0.05:
                        v[2] = 0.0
                if obstacles is not None:
                    p, v = obstacles.resolve(
                        p, v, radius, restitution,
                        t_start + f + (k + 1) / float(substeps))
            pos[f] = p
            vel[f] = v
        return pos.astype(np.float32), vel.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _offscreen_frames(self, spec, traj: Trajectory, bodies,
                          from_frame: int, visible_fraction: float = 0.6) -> int:
        """How many frames after `from_frame` lose sight of the culprit.

        A frame counts as lost when fewer than `visible_fraction` of the culprit
        bodies are inside the frustum, so one grain of forty drifting out of a
        `granular_pour` is not treated the same as the whole pour leaving.
        """
        idx = [traj.index_of(int(b.segmentation_id)) for b in bodies]
        if not idx:
            return 0
        pts = traj.pos[from_frame:, idx, :]
        vis = _geom.in_frame(spec, pts)
        # Only bodies that are actually in the scene count. A dormant
        # understudy is parked far below the world until something summons it,
        # so including it made the *valid* baseline "half the culprits are off
        # camera on every frame" -- which set the budget so high that `fission`
        # passed any separation at all and flung its halves clean out of frame.
        here = np.asarray(traj.present[from_frame:, idx], bool)
        counted = here.sum(axis=1)
        seen = (vis & here).sum(axis=1)
        frac = np.where(counted > 0, seen / np.maximum(counted, 1), 1.0)
        return int((frac < visible_fraction).sum())

    def _fit_to_frame(self, spec, traj: Trajectory, bodies, t0: int, knob,
                      build, tolerance: int = 1,
                      ladder=(1.0, 0.72, 0.52, 0.36, 0.24)):
        """Weaken `knob` until the culprit stays on screen, and return what stuck.

        The severity bins are chosen for visual legibility on a typical clip,
        but "typical" is a per-seed claim: the same reversed gravity that makes
        a nicely rising ball in one sample sends it out of the top of the frame
        in another with a higher toss. Rather than tuning the constants down
        until nothing ever escapes -- which makes every clip weaker to fix a
        few -- each clip keeps the strongest setting its own geometry allows.

        `magnitude` then reports what was actually applied, so it stays exact,
        and the severity a clip earns honestly reflects the smaller
        intervention. `build(scale)` returns a candidate trajectory.

        Callers fit on the **strong** bin and apply the resulting scale to
        whichever bin they are planning. Fitting each bin separately clamps
        `strong` hardest -- it is the one that overshoots -- and can leave it
        weaker than `medium`, which would make the bin labels a lie. One scale
        per clip says something defensible instead: *this* geometry supports at
        most this fraction of the nominal intervention, and the three bins keep
        their spacing inside it.
        """
        # Measured against the valid twin, not against zero. If the scenario
        # already lets the actor drift out of shot, weakening the intervention
        # cannot fix that, and clamping to the floor of the ladder would turn a
        # framing problem into a violation nobody can see.
        budget = self._offscreen_frames(spec, traj, bodies, t0) + tolerance
        candidate = None
        for scale in ladder:
            candidate = build(scale)
            if self._offscreen_frames(spec, candidate, bodies, t0) <= budget:
                return scale, candidate
        return ladder[-1], candidate

    def _fit_window_to_frame(self, spec, traj: Trajectory, bodies, t0: int,
                             n_win: int, build, tolerance: int = 1,
                             floor: int = 3) -> int:
        """Shorten a window until the culprit stays in shot, keeping its strength.

        The counterpart to `_fit_to_frame`, and the right one whenever the bin
        is a *qualitative* claim. Weakening the knob to keep a body on screen
        turns "gravity reverses" into "gravity is slightly reduced" -- the clip
        no longer shows what its label says, which is the failure the user
        reported on `antigravity`. Shortening the window keeps the reversal and
        just gives it less time to carry the body out of frame.

        Never goes below `floor` frames, because `_pulse` needs a ramp on each
        side and a plateau between them: shorter than three and the profile
        degenerates into a step, losing the shape that makes `severity_map` a
        field rather than a flag.
        """
        budget = self._offscreen_frames(spec, traj, bodies, t0) + tolerance
        for n in range(int(n_win), floor - 1, -1):
            if self._offscreen_frames(spec, build(n), bodies, t0) <= budget:
                return n
        return floor

    # ------------------------------------------------------------------ #
    def _rewrite_from(self, spec, traj: Trajectory, out: Trajectory, body,
                      t0: int, v0=None, p0=None, g_per_frame=None,
                      restitution=None, obstacles=None, solid=True) -> None:
        """Re-integrate one body from frame `t0`, on the surface it belongs to.

        The workhorse behind most families: an intervention is usually "change
        the state at one instant, then let physics resume", and doing that by
        hand in each injector is how a clip ends up with two violations and one
        annotation.
        """
        bi = traj.index_of(int(body.segmentation_id))
        n = traj.num_frames - t0
        if n <= 0:
            return
        if g_per_frame is None:
            g_per_frame = np.tile(traj.gravity.astype(np.float64)[None, :], (n, 1))
        start_v = traj.lin_vel[t0 - 1, bi] if v0 is None else np.asarray(v0, np.float64)
        # `p0` exists for `fission`: the understudy has to start from the
        # *original's* last lawful pose, and writing that into frame t0-1 would
        # break prefix identity on the frame before the event.
        start_p = traj.pos[t0 - 1, bi] if p0 is None else np.asarray(p0, np.float64)
        pos, vel = self._integrate_profile(
            start_p, start_v, np.asarray(g_per_frame, np.float64),
            traj.dt, _geom.floor_fn(spec, body), float(traj.radius[bi]),
            float(body.restitution if restitution is None else restitution),
            obstacles=self._obstacles(spec, traj, body, obstacles, solid),
            t_start=float(t0 - 1))
        out.pos[t0:, bi, :] = pos
        out.lin_vel[t0:, bi, :] = vel

    @staticmethod
    def _obstacles(spec, traj, body, given, solid):
        """The scene as seen by a body being re-integrated.

        `solid=False` is for the one family that means to pass through things --
        `solidity` -- where an obstacle set would undo the entire intervention.
        Everywhere else the default is on, because a family that bends gravity
        should not also be a family that walks through walls.
        """
        if not solid:
            return None
        if given is not None:
            return given
        return _geom.Obstacles(spec, traj, exclude_ids=[body.segmentation_id])

    @staticmethod
    def _sync_velocity(traj: Trajectory, out: Trajectory, bi: int,
                       t0: int) -> None:
        """Make the recorded velocity agree with the positions actually written.

        Anything that edits `pos` without doing this leaves the two telling
        different stories, and `position_continuity` believes the velocity: it
        asks whether a step is larger than the speed in play could produce, so a
        body nudged upward at a stale velocity reads as having teleported.
        That is how the vertical clamp on `immutability` -- there only to stop a
        growing body sinking into the floor -- made half the appearance families
        trip the teleport detector.
        """
        dt = traj.dt
        pos = out.pos[t0 - 1:, bi, :].astype(np.float64)
        if pos.shape[0] < 2:
            return
        out.lin_vel[t0:, bi, :] = ((pos[1:] - pos[:-1]) / dt).astype(np.float32)

    def _rewrite_group(self, spec, traj: Trajectory, out: Trajectory, bodies,
                       t0: int, g_by_body=None, v0_by_body=None,
                       substeps: int = 24) -> None:
        """Re-integrate several bodies *together*, so they collide with each other.

        Stepping them one at a time and treating the others' lawful paths as
        obstacles is right for the bodies an injector leaves alone, and wrong
        for the ones it is moving: under `global_gravity` every sphere in a
        pyramid is being re-integrated at once, so each would dodge where its
        neighbours *would have been* rather than where they now are, and they
        end up sharing space.
        """
        n = traj.num_frames - t0
        if n <= 0 or not bodies:
            return
        ids = [int(b.segmentation_id) for b in bodies]
        obstacles = _geom.Obstacles(spec, traj, exclude_ids=ids)
        idx = [traj.index_of(i) for i in ids]
        rad = [float(traj.radius[j]) for j in idx]
        rest = [float(b.restitution) for b in bodies]
        ground = [_geom.floor_fn(spec, b) for b in bodies]
        g_def = np.tile(traj.gravity.astype(np.float64)[None, :], (n, 1))
        g_seq = [np.asarray((g_by_body or {}).get(i, g_def), np.float64)
                 for i in ids]
        p = [traj.pos[t0 - 1, j].astype(np.float64).copy() for j in idx]
        v = [np.asarray((v0_by_body or {}).get(i, traj.lin_vel[t0 - 1, j]),
                        np.float64).copy() for i, j in zip(ids, idx)]

        h = traj.dt / float(substeps)
        pos = np.zeros((len(idx), n, 3), np.float64)
        vel = np.zeros((len(idx), n, 3), np.float64)
        for f in range(n):
            for k in range(substeps):
                t = float(t0 - 1) + f + (k + 1) / float(substeps)
                for a in range(len(idx)):
                    v[a] = v[a] + g_seq[a][f] * h
                    p[a] = p[a] + v[a] * h
                    fz = ground[a](p[a][0], p[a][1])
                    if p[a][2] - rad[a] <= fz and v[a][2] < 0.0:
                        p[a][2] = fz + rad[a]
                        v[a][2] = -v[a][2] * rest[a]
                        if abs(v[a][2]) < 0.05:
                            v[a][2] = 0.0
                    p[a], v[a] = obstacles.resolve(p[a], v[a], rad[a], rest[a], t)
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        d = p[b] - p[a]
                        dist = float(np.linalg.norm(d))
                        reach = rad[a] + rad[b]
                        if dist >= reach or dist < 1e-9:
                            continue
                        nrm = d / dist
                        push = 0.5 * (reach - dist)
                        p[a] -= nrm * push
                        p[b] += nrm * push
                        rel = float(np.dot(v[b] - v[a], nrm))
                        if rel < 0.0:
                            e = 0.5 * (rest[a] + rest[b])
                            imp = -(1.0 + e) * rel * 0.5
                            v[a] -= nrm * imp
                            v[b] += nrm * imp
            for a in range(len(idx)):
                pos[a, f] = p[a]
                vel[a, f] = v[a]
        for a, j in enumerate(idx):
            out.pos[t0:, j, :] = pos[a].astype(np.float32)
            out.lin_vel[t0:, j, :] = vel[a].astype(np.float32)

    @staticmethod
    def _pulse(n: int, peak: float, base: float = 1.0,
               hold: float = 0.6) -> np.ndarray:
        """Trapezoid: ramp `base` -> `peak`, hold, ramp back to `base`.

        A violation with a shape, not a step: it comes on, sustains, releases,
        and the actor obeys real physics afterwards. The residual traces the
        same shape, which is what a per-frame severity field is for.

        Deliberately a trapezoid rather than a raised cosine. A cosine only
        touches its peak instantaneously, so the *mean* effect over the window
        is (base + peak) / 2 -- half the strength the bin advertises. On a body
        already moving fast that is not enough to visibly change its motion, and
        the strongest bin ends up looking like the weakest. Holding at the peak
        makes the advertised magnitude the one actually applied.
        """
        if n <= 1:
            return np.full((max(n, 1),), peak, np.float64)
        # The plateau is the load-bearing part and must never vanish: a profile
        # that only *passes through* its peak applies a mean effect of about
        # (base + peak) / 2, which is half what the bin advertises. The old
        # clamp kept a ramp frame on each side at all costs, so a 2-frame
        # window came out as two ramp values and never reached `peak` at all --
        # which only became reachable once the frustum fit started shortening
        # windows instead of weakening them.
        n_ramp = max(1, int(round((1.0 - hold) * n / 2.0)))
        while n_ramp > 0 and 2 * n_ramp >= n:
            n_ramp -= 1
        n_hold = n - 2 * n_ramp
        # Ramp values are strictly *between* base and peak: every frame inside
        # the window must actually be violating, or `active[t]` would mark
        # frames where alpha == 1 and nothing is wrong. The return to base
        # happens on the frame after the window, where normal physics resumes.
        up = np.linspace(base, peak, n_ramp + 2)[1:-1]
        down = up[::-1]
        return np.concatenate([up, np.full((n_hold,), peak, np.float64), down])

    @staticmethod
    def _first_contact(traj: Trajectory, a_id: int, b_id: int) -> Optional[int]:
        """First frame where bodies `a_id` and `b_id` are recorded in contact."""
        c = traj.contacts
        if len(c) == 0:
            return None
        m = (((c.body_a == a_id) & (c.body_b == b_id)) |
             ((c.body_a == b_id) & (c.body_b == a_id)))
        if not m.any():
            return None
        return int(c.frame[m].min())


_REGISTRY: Dict[str, Injector] = {}


def register(inj: Injector) -> Injector:
    _REGISTRY[inj.family] = inj
    return inj


def get(family: str) -> Injector:
    if family not in _REGISTRY:
        raise KeyError("no injector for family %r; have %s"
                       % (family, sorted(_REGISTRY)))
    return _REGISTRY[family]


def available() -> List[str]:
    return sorted(_REGISTRY)
