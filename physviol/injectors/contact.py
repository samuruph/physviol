"""Contact-domain injectors: solidity, superelastic, newton3_reaction.

Phase 0 implements `solidity`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from .base import Injector, InterventionPlan, register


class Solidity(Injector):
    """Suppress a collision pair so one body sinks into / through another.

    The knob is a **target penetration depth** in units of the actor's radius:

        easy   -> 0.30 r  (dips in, contact resumes, pushed back out)
        medium -> 0.80 r  (deeply swallowed, still recovered)
        hard   -> 2.50 r  (centre passes the surface: through, for good)

    The bins straddle 1.0 r deliberately. Below one radius the actor's centre is
    still above the surface, so restoring contact can push it back out and the
    violation is a *transient* sinking. Past one radius there is nothing left to
    push against. That qualitative jump is what `hard` encodes.

    **Why the descent is prescribed rather than ballistic.** Letting the actor
    free-fall through the surface sounds more principled, but at 12 fps a
    dropped ball covers ~0.6 m per frame at impact, so the reachable depths are
    quantised far more coarsely than a radius: asking for 0.8 r yields 1.8 r and
    the bins collapse into each other. Since the injector already owns the
    trajectory, it places the actor at *exactly* the requested depth instead.
    That is what makes `magnitude` exact by construction, and it means the
    measured penetration residual must reproduce it -- which is a real
    end-to-end check on the residual pipeline, not a tautology.
    """

    family = "solidity"

    DEPTH_BY_BIN = {"weak": 0.30, "medium": 0.80, "strong": 2.50}
    FRAMES_BY_BIN = {"weak": 2, "medium": 3, "strong": 4}

    # ------------------------------------------------------------------ #
    def plan(self, spec, traj: Trajectory, rng: np.random.RandomState,
             severity_bin: str) -> Optional[InterventionPlan]:
        actor = next((b for b in spec.bodies if b.role == "actor"), None)
        surface = next((b for b in spec.bodies
                        if b.role in ("floor", "prop") and b.static), None)
        if actor is None or surface is None:
            return None

        t_contact = self._first_contact(traj, actor.segmentation_id,
                                        surface.segmentation_id)
        # A contact at frame 0 is not an *event* -- the actor is already resting
        # or rolling on the surface, so there is no arrival to latch onto. Fall
        # through to the geometric helper, which breaks the contact mid-clip
        # instead ("it sank through the floor while rolling").
        if t_contact is None or t_contact < 1:
            t_contact = self._first_geometric_contact(spec, traj, actor)
        if t_contact is None or t_contact < 1 or t_contact >= traj.num_frames - 1:
            return None

        radius = actor.bounding_radius
        depth_r = self.DEPTH_BY_BIN[severity_bin]
        target_depth = depth_r * radius
        n_window = self._window_len(self.FRAMES_BY_BIN[severity_bin],
                                    t_contact, traj.num_frames)
        t_end = t_contact + n_window - 1

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t_contact,
            windows=[(t_contact, t_end)],
            causal_body_ids=[actor.segmentation_id, surface.segmentation_id],
            params={"type": "disable_collision_pair",
                    "pair": [int(actor.segmentation_id),
                             int(surface.segmentation_id)],
                    "frames_disabled": int(n_window),
                    "target_depth_radii": depth_r},
            magnitude=float(target_depth), magnitude_unit="m_penetration_depth",
            severity_bin=severity_bin,
            notes={"radius": float(radius), "surface_top": _surface_top(spec, surface),
                   "t_contact": int(t_contact),
                   "passes_through": bool(depth_r >= 1.0)},
        )

    # ------------------------------------------------------------------ #
    def apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        out = self._clone(traj)
        actor = next(b for b in spec.bodies if b.role == "actor")
        bi = traj.index_of(actor.segmentation_id)
        t0, t1 = plan.windows[0]
        n_win = t1 - t0 + 1

        g = traj.gravity.astype(np.float64)
        radius = float(plan.notes["radius"])
        top = float(plan.notes["surface_top"])
        target = float(plan.magnitude)

        p_prev = traj.pos[t0 - 1, bi].astype(np.float64)
        v_prev = traj.lin_vel[t0 - 1, bi].astype(np.float64)

        # Lateral motion carries on undisturbed; only the vertical axis is
        # prescribed, easing from "just touching" to exactly `target` deep.
        tt = (np.arange(1, n_win + 1, dtype=np.float64) * traj.dt)[:, None]
        xy = p_prev[None, :2] + v_prev[None, :2] * tt

        frac = np.linspace(1.0 / n_win, 1.0, n_win) ** 1.6   # ease-in
        depth = target * frac
        z = top + radius - depth

        out.pos[t0:t1 + 1, bi, 0:2] = xy.astype(np.float32)
        out.pos[t0:t1 + 1, bi, 2] = z.astype(np.float32)
        dz = np.diff(np.concatenate([[float(traj.pos[t0 - 1, bi, 2])], z]))
        out.lin_vel[t0:t1 + 1, bi, 0:2] = v_prev[None, :2].astype(np.float32)
        out.lin_vel[t0:t1 + 1, bi, 2] = (dz / traj.dt).astype(np.float32)
        out.ang_vel[t0:, bi, :] = traj.ang_vel[t0 - 1, bi][None, :]

        # ---- after the window, contact is restored -----------------------
        n_rest = traj.num_frames - (t1 + 1)
        if n_rest > 0:
            p_end = np.array([xy[-1, 0], xy[-1, 1], z[-1]], dtype=np.float64)
            v_end = np.array([v_prev[0], v_prev[1], float(dz[-1] / traj.dt)])
            if z[-1] > top:
                # Centre still above the surface: the surface shoves it back out.
                rest = float(actor.restitution)
                p_end[2] = top + radius
                v_end[2] = abs(v_end[2]) * rest
            # else: centre is past the surface, nothing to push against -- it
            # keeps falling, and the violation persists by consequence.
            if z[-1] > top:
                # Legal again: bounce properly, or a later frame would sink
                # back through the floor and fake a second violation.
                pos2, vel2 = self._ballistic_with_floor(
                    p_end, v_end, g, traj.dt, n_rest, top, radius,
                    float(actor.restitution))
            else:
                pos2, vel2 = self._ballistic(p_end, v_end, g, traj.dt, n_rest)
            out.pos[t1 + 1:, bi, :] = pos2
            out.lin_vel[t1 + 1:, bi, :] = vel2

        # Contacts for the suppressed pair no longer occur during the window.
        c = out.contacts
        pair = set(plan.params["pair"])
        if len(c):
            drop = np.array([(t0 <= int(f) <= t1) and ({int(a), int(b)} == pair)
                             for f, a, b in zip(c.frame, c.body_a, c.body_b)], bool)
            keep = ~drop
            out.contacts = type(c)(c.frame[keep], c.body_a[keep], c.body_b[keep],
                                   c.point[keep], c.normal[keep], c.impulse[keep],
                                   c.penetration[keep])

        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out

    def strong_residual_reference(self, spec) -> float:
        # The penetration law reports depth in radii, which is exactly the unit
        # DEPTH_BY_BIN is expressed in -- so the reference is the bin value.
        return float(self.DEPTH_BY_BIN["strong"])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_geometric_contact(spec, traj: Trajectory, actor) -> Optional[int]:
        """Fallback when the simulator reported no contact event.

        Two cases. If the actor falls onto the surface, use the first frame it
        arrives. If it is already resting at frame 0 -- a ball rolling along the
        ground, say -- there is no arrival to latch onto, so break the contact
        mid-clip instead: the violation is then "it sank through the floor while
        rolling", which is just as valid a solidity failure.
        """
        bi = traj.index_of(actor.segmentation_id)
        z = traj.pos[:, bi, 2] - actor.bounding_radius
        below = np.flatnonzero(z <= spec.floor_level + 1e-4)
        if not below.size:
            return None
        if int(below[0]) == 0:
            t = traj.num_frames // 3
            return int(t) if 1 <= t < traj.num_frames - 1 else None
        return int(below[0])


def _surface_top(spec, surface) -> float:
    """World z of the walkable top of a static surface body."""
    if surface.kind == "cube":
        return float(surface.position[2] + surface.scale[2])
    if surface.kind == "dome":
        # KuBasic's dome is a bowl whose interior floor sits at the origin.
        return float(spec.floor_level)
    return float(surface.position[2] + max(surface.scale))


register(Solidity())
