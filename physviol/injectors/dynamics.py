"""Dynamics-domain injectors: phantom_impulse, angular_momentum.

(`newton2_mass` lives with `newton3_reaction` in contact.py -- they are one
mechanism at two settings of the same dial and share their collision-finding.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class PhantomImpulse(Injector):
    """A body is shoved by nothing.

    The cleanest violation in the taxonomy and the one with the least ambiguity
    about *where*: the culprit is one body, the moment is one frame, and there
    is no second object anywhere near it. It fires only on frames with no
    recorded contact, because a shove during a collision is indistinguishable
    from the collision.
    """

    family = "phantom_impulse"
    DV_BY_BIN = {"weak": 1.0, "medium": 2.4, "strong": 4.5}      # m/s

    def strong_residual_reference(self, spec) -> float:
        g_dt = 9.81 / float(spec.tier.fps)
        return float(self.DV_BY_BIN["strong"] / g_dt)

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        free = _geom.contact_free_run(traj, int(actor.segmentation_id), min_len=2)
        if free is not None:
            t0 = max(free[0] + 1, min(free[1], T // 3))
        else:
            t0 = max(1, T // 3)
        if not (1 <= t0 < T - 1):
            return None

        dv = self.DV_BY_BIN[severity_bin]
        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        # Mostly sideways with a little lift: a purely horizontal shove on a
        # resting body is easy to mistake for a nudge from off screen, while a
        # visible hop is unmistakably uncaused.
        push = [dv * 0.85 * float(np.cos(heading)),
                dv * 0.85 * float(np.sin(heading)), dv * 0.5]
        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        n_win = self._window_len(2, t0, T)
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0,
            windows=[(t0, min(T - 1, t0 + n_win - 1))],
            causal_body_ids=[int(actor.segmentation_id)],
            params={"type": "impulse", "delta_v": push},
            magnitude=float(dv / max(g_dt, 1e-9)),
            magnitude_unit="impulse_over_m_vtyp", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "delta_v_ms": dv})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0 = plan.t_event
        v0 = (traj.lin_vel[t0 - 1, bi].astype(np.float64)
              + np.asarray(plan.params["delta_v"], np.float64))
        self._rewrite_from(spec, traj, out, actor, t0, v0=v0)
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class AngularMomentum(Injector):
    """Spin that reverses, or grows, with nothing to torque it.

    Two ways a body can carry angular momentum, and the injector dispatches on
    which one the *scenario declares* -- never on the scenario's name:

    * **Free body** (`spin_toss`, `rolling_ramp`): the body's own spin is
      rescaled and the orientation re-integrated from there. Position is
      untouched, so the clip contains exactly one anomaly.
    * **Constrained body** (`spec.notes["constraint"] == "pivot"`): the whole
      swing is re-solved from the event frame with its angular rate rescaled,
      by asking the scenario to do it. A pendulum turning around in the middle
      of its arc is the violation; a bob whose *spin* reverses would be
      invisible, since the bob is a sphere.

    Fires inside a contact-free stretch. A body in contact can change its spin
    lawfully, so an edit made during contact is not a violation the residual can
    prove -- which is exactly why `rolling_ramp` has a lip to fly off.
    """

    family = "angular_momentum"
    # New spin as a multiple of the old: nearly stopped, exactly reversed,
    # reversed and faster.
    SPIN_BY_BIN = {"weak": 0.2, "medium": -1.0, "strong": -1.9}
    # ...unless the body is barely spinning, in which case rescaling zero is
    # zero and the clip would carry no violation at all. Then a spin is
    # *imposed* instead, which the family's own description covers: "spin
    # reverses, or torque appears with no contact". rad/s.
    IMPOSED_BY_BIN = {"weak": 1.5, "medium": 4.0, "strong": 8.0}
    SPINNING = 0.5      # rad/s, above which there is a spin worth reversing

    def strong_residual_reference(self, spec) -> float:
        omega = float(spec.notes.get("omega_ref", 0.0))
        if omega <= 0.0:
            omega = max([float(np.linalg.norm(b.angular_velocity))
                         for b in spec.bodies if not b.static] or [0.0])
        omega = max(omega, 2.0)
        radius = max([float(b.bounding_radius)
                      for b in _geom.actors(spec)] or [0.3])
        g_dt = 9.81 / float(spec.tier.fps)
        return abs(self.SPIN_BY_BIN["strong"] - 1.0) * omega * radius / g_dt

    # ------------------------------------------------------------------ #
    def _pivot(self, spec) -> bool:
        return spec.notes.get("constraint") == "pivot"

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        T = traj.num_frames
        if self._pivot(spec):
            t0 = max(1, T // 2)
        else:
            free = _geom.contact_free_run(traj, int(actor.segmentation_id),
                                          min_len=3)
            if free is None:
                return None
            t0 = max(free[0] + 1, min(free[1] - 1, (free[0] + free[1]) // 2))
        if not (1 <= t0 < T - 1):
            return None

        bi = traj.index_of(int(actor.segmentation_id))
        omega0 = float(np.linalg.norm(traj.ang_vel[t0 - 1, bi]))
        targets = self._all_actors(spec) if self._pivot(spec) else [actor]
        spinning = omega0 > self.SPINNING or self._pivot(spec)

        k = self.SPIN_BY_BIN[severity_bin] if spinning else 0.0
        imposed = None if spinning else self._imposed(traj, bi, severity_bin)
        magnitude = (abs(k - 1.0) * omega0 if spinning
                     else float(np.linalg.norm(imposed)))

        strong_k = self.SPIN_BY_BIN["strong"] if spinning else 0.0
        strong_imposed = (None if spinning
                          else self._imposed(traj, bi, "strong"))
        strong = self._preview(spec, traj, t0, strong_k, targets, strong_imposed)
        r_strong = self._measure(strong, int(actor.segmentation_id),
                                 "angular_momentum", {})

        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0,
            windows=[(t0, min(T - 1, t0 + self._window_len(2, t0, T) - 1))],
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "spin_scale" if spinning else "spin_impose",
                    "omega_scale": k,
                    "omega_imposed": None if imposed is None
                                     else [float(x) for x in imposed],
                    "constraint": "pivot" if self._pivot(spec) else "free"},
            magnitude=float(magnitude),
            magnitude_unit="angular_momentum_defect", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": _geom.surface_top(spec, actor),
                   "omega_scale": k, "omega_at_event": omega0,
                   "omega_imposed": None if imposed is None
                                    else [float(x) for x in imposed],
                   "r_strong": float(r_strong)})

    def _imposed(self, traj, bi: int, severity_bin: str) -> np.ndarray:
        """A spin to give a body that has none, about the axis across its travel.

        Tumbling forward reads as physical; spinning about the direction of
        motion reads as a rendering artefact, so the axis is chosen rather than
        picked at random.
        """
        v = traj.lin_vel[:, bi, :].astype(np.float64)
        speed = np.linalg.norm(v, axis=1)
        f = int(np.argmax(speed))
        heading = v[f, :2]
        n = float(np.linalg.norm(heading))
        axis = (np.array([-heading[1], heading[0], 0.0]) / n if n > 1e-6
                else np.array([0.0, 1.0, 0.0]))
        return axis * self.IMPOSED_BY_BIN[severity_bin]

    # ------------------------------------------------------------------ #
    def _preview(self, spec, traj, t0: int, k: float, targets,
                 imposed=None) -> Trajectory:
        out = self._clone(traj)
        if self._pivot(spec):
            from .. import scenarios as scen_mod
            if scen_mod.get(spec.scenario).rescript(spec, out, t0, k):
                return out
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            omega = (traj.ang_vel[t0 - 1, bi].astype(np.float64) * k
                     if imposed is None else np.asarray(imposed, np.float64))
            n = traj.num_frames - t0
            out.ang_vel[t0:, bi, :] = omega.astype(np.float32)[None, :]
            out.quat[t0:, bi, :] = _geom.integrate_quaternion(
                traj.quat[t0 - 1, bi], omega, traj.dt, n)
        return out

    def apply(self, spec, traj, plan) -> Trajectory:
        targets = [b for b in spec.bodies
                   if int(b.segmentation_id) in set(plan.causal_body_ids)]
        out = self._preview(spec, traj, plan.t_event,
                            float(plan.notes["omega_scale"]), targets,
                            plan.notes.get("omega_imposed"))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(PhantomImpulse())
register(AngularMomentum())
