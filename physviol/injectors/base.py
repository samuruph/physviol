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
        its law. This is `r_hard` in the bounded score of PLAN 3.4 step 3, and
        it is what makes severity comparable across families whose physical
        units are not."""
        raise NotImplementedError

    # -- helpers shared by subclasses -------------------------------------
    @staticmethod
    def _clone(traj: Trajectory) -> Trajectory:
        out = copy.copy(traj)
        out.pos = traj.pos.copy()
        out.quat = traj.quat.copy()
        out.lin_vel = traj.lin_vel.copy()
        out.ang_vel = traj.ang_vel.copy()
        out.meta = dict(traj.meta)
        return out

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
                              dt: float, n: int, floor_z: float, radius: float,
                              restitution: float, substeps: int = 24):
        """Free flight that still respects a horizontal plane.

        Needed after a violation window closes: the actor is legal again, so it
        must bounce rather than sink through on a later frame. Sub-steps so the
        bounce instant is not quantised to the frame rate.
        """
        p = p0.astype(np.float64).copy()
        v = v0.astype(np.float64).copy()
        h = dt / float(substeps)
        pos = np.zeros((n, 3), np.float64)
        vel = np.zeros((n, 3), np.float64)
        for f in range(n):
            for _ in range(substeps):
                v = v + g * h
                p = p + v * h
                if p[2] - radius <= floor_z and v[2] < 0.0:
                    p[2] = floor_z + radius
                    v[2] = -v[2] * restitution
                    if abs(v[2]) < 0.05:          # settle instead of jittering
                        v[2] = 0.0
            pos[f] = p
            vel[f] = v
        return pos.astype(np.float32), vel.astype(np.float32)

    @staticmethod
    def _integrate_profile(p0: np.ndarray, v0: np.ndarray, g_per_frame: np.ndarray,
                           dt: float, floor_z: float, radius: float,
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
        pos = np.zeros((n, 3), np.float64)
        vel = np.zeros((n, 3), np.float64)
        for f in range(n):
            g = g_per_frame[f].astype(np.float64)
            for _ in range(substeps):
                v = v + g * h
                p = p + v * h
                if p[2] - radius <= floor_z and v[2] < 0.0:
                    p[2] = floor_z + radius
                    v[2] = -v[2] * restitution
                    if abs(v[2]) < 0.05:
                        v[2] = 0.0
            pos[f] = p
            vel[f] = v
        return pos.astype(np.float32), vel.astype(np.float32)

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
