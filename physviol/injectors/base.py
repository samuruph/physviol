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

    def _window_len(self, default: int, t0: int, num_frames: int) -> int:
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
    @staticmethod
    def _primary(spec):
        """The one body most families act on. `None` if the scenario has no actor.

        Dormant understudies are skipped: they exist in the scene graph so
        `fission` has something to switch on, and until it does they are not
        part of the scene.
        """
        live = [b for b in _geom.actors(spec) if not b.dormant]
        return live[0] if live else None

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
                           restitution: float, substeps: int = 24):
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
            for _ in range(substeps):
                v = v + g * h
                p = p + v * h
                fz = ground(p[0], p[1])
                if p[2] - radius <= fz and v[2] < 0.0:
                    p[2] = fz + radius
                    v[2] = -v[2] * restitution
                    if abs(v[2]) < 0.05:
                        v[2] = 0.0
            pos[f] = p
            vel[f] = v
        return pos.astype(np.float32), vel.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _rewrite_from(self, spec, traj: Trajectory, out: Trajectory, body,
                      t0: int, v0=None, p0=None, g_per_frame=None,
                      restitution=None) -> None:
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
            float(body.restitution if restitution is None else restitution))
        out.pos[t0:, bi, :] = pos
        out.lin_vel[t0:, bi, :] = vel

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
        n_ramp = max(1, int(round((1.0 - hold) * n / 2.0)))
        if 2 * n_ramp >= n:
            n_ramp = max(1, (n - 1) // 2)
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
