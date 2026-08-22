"""Contact-domain injectors: solidity, superelastic, newton3_reaction.

What the three have in common is that they need a *contact event* to hang the
violation on -- you cannot break a collision before the collision happens. That
is why the injector API is two-phase: `plan` gets to look at the finished valid
rollout and pick its moment, and only then does `apply` edit anything.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


def _top_of_partner(spec, traj, partner_id: int, frame: int) -> float:
    """World z of the top of whatever the actor just landed on.

    Taken at the contact frame and then held fixed. For a static floor that is
    exact; for the apex of a sphere pyramid it is an approximation good for the
    handful of frames a violation window lasts, and the plan records which case
    it was.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(partner_id)), None)
    if body is None:
        return float(spec.floor_level)
    if body.static:
        return _geom.top_of(spec, body)
    bi = traj.index_of(int(partner_id))
    f = min(max(int(frame), 0), traj.num_frames - 1)
    return float(traj.pos[f, bi, 2] + traj.radius[bi])


def _contact_event(spec, traj, actor) -> Optional[Tuple[int, int, bool]]:
    """(frame, partner_id, partner_is_static) for the actor's first collision.

    Falls back to geometry when the simulator reported nothing -- which happens
    when the actor is already resting or rolling at frame 0 and there is no
    *arrival* to latch onto. In that case the violation becomes "it sank through
    the floor while rolling", which is just as real a solidity failure.
    """
    dormant = tuple(int(b.segmentation_id) for b in spec.bodies if b.dormant)
    hit = _geom.first_contact_any(traj, int(actor.segmentation_id),
                                  exclude=dormant)
    if hit is not None and hit[0] >= 1:
        partner = next((b for b in spec.bodies
                        if int(b.segmentation_id) == hit[1]), None)
        return int(hit[0]), int(hit[1]), bool(partner is not None and partner.static)

    surface, _ = _geom.support_under(spec, actor)
    if surface is None:
        return None
    bi = traj.index_of(int(actor.segmentation_id))
    top = _geom.top_of(spec, surface)
    below = np.flatnonzero((traj.pos[:, bi, 2] - traj.radius[bi]) <= top + 1e-4)
    if not below.size:
        return None
    if int(below[0]) == 0:
        t = traj.num_frames // 3
        if not (1 <= t < traj.num_frames - 1):
            return None
        return int(t), int(surface.segmentation_id), True
    return int(below[0]), int(surface.segmentation_id), True


class Solidity(Injector):
    """Suppress a collision pair so one body sinks into / through another.

    The knob is a **target penetration depth** in units of the actor's radius:

        weak   -> 0.30 r  (dips in, contact resumes, pushed back out)
        medium -> 0.80 r  (deeply swallowed, still recovered)
        strong -> 2.50 r  (centre passes the surface: through, for good)

    The bins straddle 1.0 r deliberately. Below one radius the actor's centre is
    still above the surface, so restoring contact can push it back out and the
    violation is a *transient* sinking. Past one radius there is nothing left to
    push against. That qualitative jump is what `strong` encodes.

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

    def strong_residual_reference(self, spec) -> float:
        # The penetration law reports depth in radii, which is exactly the unit
        # DEPTH_BY_BIN is expressed in -- so the reference is the bin value.
        return float(self.DEPTH_BY_BIN["strong"])

    # ------------------------------------------------------------------ #
    def plan(self, spec, traj: Trajectory, rng: np.random.RandomState,
             severity_bin: str) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        event = _contact_event(spec, traj, actor)
        if event is None:
            return None
        t_contact, partner_id, partner_static = event
        if not (1 <= t_contact < traj.num_frames - 1):
            return None

        radius = actor.bounding_radius
        depth_r = self.DEPTH_BY_BIN[severity_bin]
        n_window = self._window_len(self.FRAMES_BY_BIN[severity_bin],
                                    t_contact, traj.num_frames)
        t_end = t_contact + n_window - 1

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t_contact,
            windows=[(t_contact, t_end)],
            causal_body_ids=[int(actor.segmentation_id), int(partner_id)],
            params={"type": "disable_collision_pair",
                    "pair": [int(actor.segmentation_id), int(partner_id)],
                    "frames_disabled": int(n_window),
                    "target_depth_radii": depth_r},
            magnitude=float(depth_r * radius),
            magnitude_unit="m_penetration_depth",
            severity_bin=severity_bin,
            notes={"radius": float(radius),
                   "surface_top": _top_of_partner(spec, traj, partner_id, t_contact),
                   "partner_id": int(partner_id),
                   "partner_static": bool(partner_static),
                   "t_contact": int(t_contact),
                   "passes_through": bool(depth_r >= 1.0)},
        )

    # ------------------------------------------------------------------ #
    def apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        out = self._clone(traj)
        actor = self._primary(spec)
        bi = traj.index_of(int(actor.segmentation_id))
        t0, t1 = plan.windows[0]
        n_win = t1 - t0 + 1

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
                # Centre still above the surface: the surface shoves it back
                # out, and it must bounce properly from there or a later frame
                # would sink through again and fake a second violation.
                p_end[2] = top + radius
                v_end[2] = abs(v_end[2]) * float(actor.restitution)
                pos2, vel2 = self._integrate_profile(
                    p_end, v_end,
                    np.tile(traj.gravity.astype(np.float64)[None, :], (n_rest, 1)),
                    traj.dt, top, radius, float(actor.restitution))
            else:
                # Centre is past the surface: nothing left to push against, so
                # it keeps falling and the violation persists by consequence.
                pos2, vel2 = self._ballistic(
                    p_end, v_end, traj.gravity.astype(np.float64), traj.dt, n_rest)
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


class SuperElastic(Injector):
    """Restitution greater than one: every bounce comes back faster than it went in.

    The taxonomy's only `repeated` family, and the reason `violation_windows` is
    a *list* rather than an onset. One boosted restitution produces a gain at
    every subsequent contact, so the clip has several disjoint violation
    intervals with lawful flight between them. A schema with a single `t_event`
    would have to pick one and silently discard the rest.

    Both bodies in a two-body collision are boosted. Rescaling only one would
    add a momentum violation on top of the energy one, and a clip carrying two
    violations while annotating one is worse than useless.
    """

    family = "superelastic"
    GAIN_BY_BIN = {"weak": 1.20, "medium": 1.55, "strong": 2.10}

    def strong_residual_reference(self, spec) -> float:
        # The law reports fractional kinetic-energy gain, and energy goes as
        # speed squared, so a speed gain of k shows up as k^2 - 1.
        return float(self.GAIN_BY_BIN["strong"] ** 2 - 1.0)

    def _boosted(self, spec, traj, targets, t0, gain, normal=None) -> Trajectory:
        """Reflect each body's *incoming* velocity with restitution `gain` > 1.

        Amplifying the outgoing velocity instead is the obvious implementation
        and it is wrong wherever the impact is heavily damped: a 2.2 kg cube
        landing on a pyramid of 0.8 kg spheres has essentially stopped by the
        next sampled frame, and 2.1x of nothing is nothing -- the clip shipped
        with a total-energy residual of exactly zero. Restitution is a statement
        about the ratio of outgoing to *incoming* speed, so that is what this
        computes, and the resulting energy gain is k^2 - 1 no matter how the
        simulator resolved the collision.
        """
        out = self._clone(traj)
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            v_in = traj.lin_vel[t0 - 1, bi].astype(np.float64)
            n = (np.asarray(normal, np.float64) if normal is not None
                 else self._contact_normal(traj, t0, int(body.segmentation_id),
                                           v_in))
            # Reflection is quadratic in the normal, so its sign does not
            # matter -- only that it is the *impact's* normal.
            v_out = v_in - (1.0 + gain) * float(np.dot(v_in, n)) * n
            out.lin_vel[t0, bi, :] = v_out.astype(np.float32)
            self._rewrite_from(spec, traj, out, body, t0 + 1, v0=v_out,
                               restitution=min(0.98, float(body.restitution) * gain))
        return out

    @staticmethod
    def _contact_normal(traj, frame: int, body_id: int, v_in) -> np.ndarray:
        """The normal of the frame's most head-on contact for this body.

        Taking the *first* matching contact is what made a ball-ball collision
        reflect about the floor: two balls rolling into each other are touching
        the ground on the same frame, and the ground's normal is perpendicular
        to their motion, so the reflection was the identity and the "bouncier"
        ball sailed straight through its partner. Scoring by how much of the
        body's velocity lies along each normal picks the collision that is
        actually happening.
        """
        best, best_score = np.array([0.0, 0.0, 1.0]), -1.0
        c = traj.contacts
        for k in range(len(c)):
            if int(c.frame[k]) != int(frame):
                continue
            if body_id not in (int(c.body_a[k]), int(c.body_b[k])):
                continue
            cand = np.asarray(c.normal[k], np.float64)
            mag = float(np.linalg.norm(cand))
            if mag < 1e-6:
                continue
            cand = cand / mag
            score = abs(float(np.dot(v_in, cand)))
            if score > best_score:
                best, best_score = cand, score
        return best

    def _targets(self, spec, actor, partner_id):
        out = [actor]
        partner = next((b for b in spec.bodies
                        if int(b.segmentation_id) == int(partner_id)), None)
        if partner is not None and not partner.static:
            out.append(partner)
        return out

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        if actor is None:
            return None
        # An impact, not merely a contact: reflecting the normal component of a
        # rolling ball's velocity along the floor changes nothing, and the clip
        # would ship claiming a super-elastic bounce that never happened.
        dormant = tuple(int(b.segmentation_id) for b in spec.bodies if b.dormant)
        impact = _geom.first_impact(traj, int(actor.segmentation_id),
                                    exclude=dormant)
        if impact is None:
            return None
        t0, partner_id, normal = impact
        if not (1 <= t0 < traj.num_frames - 2):
            return None

        gain = self.GAIN_BY_BIN[severity_bin]
        targets = self._targets(spec, actor, partner_id)
        # Roll the boosted trajectory forward here so the windows can be the
        # frames energy is *actually* gained on, rather than a guess. `apply`
        # recomputes the same thing deterministically.
        preview = self._boosted(spec, traj, targets, t0, gain, normal)
        windows = self._gain_windows(preview, targets, traj, t0)

        # The reference has to be measured, not declared. Boosting the speed by
        # k multiplies *kinetic* energy by k^2, but the law scores total
        # mechanical energy, so the achievable gain depends on how much of the
        # body's energy is potential when it bounces. A cube striking the apex
        # of a pyramid is most of a metre up and carries a large potential
        # term, so a declared reference of k^2 - 1 is unreachable there and
        # every clip in that scenario scores a fraction of what it deserves.
        top = _top_of_partner(spec, traj, partner_id, t0)
        strong_preview = self._boosted(spec, traj, targets, t0,
                                       self.GAIN_BY_BIN["strong"], normal)
        r_strong = self._measure(strong_preview, int(actor.segmentation_id),
                                 "energy_at_contact", {"surface_top": top})

        return InterventionPlan(
            family=self.family, kind="repeated", t_event=windows[0][0],
            windows=windows,
            causal_body_ids=[int(b.segmentation_id) for b in targets],
            params={"type": "restitution_gain", "speed_gain": gain,
                    "energy_gain_ratio": gain ** 2 - 1.0,
                    "first_contact_frame": int(t0)},
            magnitude=float(gain ** 2 - 1.0),
            magnitude_unit="energy_gain_ratio", severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "surface_top": top,
                   "speed_gain": gain, "n_bounces": len(windows),
                   "r_strong": float(r_strong),
                   "impact_normal": [float(x) for x in normal],
                   "targets": [int(b.segmentation_id) for b in targets]})

    @staticmethod
    def _gain_windows(preview, targets, traj, t0) -> List[Tuple[int, int]]:
        """Frames where the boosted rollout gains kinetic energy it should not."""
        T = traj.num_frames
        flag = np.zeros((T,), bool)
        for body in targets:
            bi = traj.index_of(int(body.segmentation_id))
            v = np.linalg.norm(preview.lin_vel[:, bi, :].astype(np.float64), axis=1)
            gained = np.zeros((T,), bool)
            gained[1:] = v[1:] > v[:-1] * 1.05 + 1e-3
            gained[:t0] = False
            flag |= gained
        # The first contact is a superelastic bounce by construction, whether
        # or not the speed test notices it on a slow one. It also has to be in
        # the list for `t_event` to equal the first frame this injector edits --
        # anything later would declare a prefix longer than the one it kept.
        out, f = [(t0, min(T - 1, t0 + 1))], 0
        while f < T:
            if flag[f]:
                end = min(T - 1, f + 1)
                out.append((f, end))
                f = end + 1
            else:
                f += 1
        # Merge anything that touches, so the windows stay sorted and disjoint
        # -- `physviol validate` rejects overlaps, and a bounce two frames after
        # another is one violation, not two.
        merged: List[Tuple[int, int]] = []
        for s, e in sorted(out):
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def apply(self, spec, traj, plan) -> Trajectory:
        actor = self._primary(spec)
        targets = self._targets(spec, actor, plan.notes["targets"][-1])
        out = self._boosted(spec, traj, targets,
                            int(plan.params["first_contact_frame"]),
                            float(plan.params["speed_gain"]),
                            plan.notes.get("impact_normal"))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


class _CollisionEdit(Injector):
    """Shared machinery for the two ways a collision can come out wrong.

    Both need the same thing -- the first contact between two bodies that both
    ought to respond -- and differ only in what outcome they substitute, so the
    collision-finding lives here and each subclass supplies `_outcome`.

    The normal is taken as the line of centres at the contact frame rather than
    from the recorded contact. That is exact for spheres, close enough for a
    cube landing on one, and it sidesteps the orientation question entirely:
    `Contacts.normal` points from `body_a` toward `body_b`, so using it
    correctly means knowing which of the two you are looking at, and getting
    that backwards silently mirrors the collision.
    """

    STRONG_REFERENCE = 3.0            # |dv| / (g*dt) that the pair fails to show

    def strong_residual_reference(self, spec) -> float:
        return self.STRONG_REFERENCE

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = self._primary(spec)
        actor_id = int(actor.segmentation_id) if actor is not None else None
        hit = _geom.first_dynamic_pair_contact(spec, traj, prefer=actor_id)
        if hit is None:
            return None
        t0, id_a, id_b = hit
        if not (1 <= t0 < traj.num_frames - 1):
            return None
        if actor_id is None or actor_id not in (id_a, id_b):
            actor_id = id_a
        other_id = id_b if actor_id == id_a else id_a

        ia, ib = traj.index_of(actor_id), traj.index_of(other_id)
        normal = traj.pos[t0, ib] - traj.pos[t0, ia]
        mag = float(np.linalg.norm(normal))
        if mag < 1e-6:
            return None
        normal = (normal / mag).astype(np.float64)

        ratio = float(self.RATIO_BY_BIN[severity_bin])
        preview = self._edited(spec, traj, actor_id, other_id, t0, normal, ratio)
        strong = self._edited(spec, traj, actor_id, other_id, t0, normal,
                              float(self.RATIO_BY_BIN["strong"]))
        r_strong = max(
            self._measure(strong, cid, "linear_momentum", {})
            for cid in (actor_id, other_id))
        del preview

        g_dt = float(np.linalg.norm(traj.gravity)) * traj.dt
        dv = float(np.linalg.norm(traj.lin_vel[t0, ib] - traj.lin_vel[t0 - 1, ib]))
        t1 = min(traj.num_frames - 1, t0 + 1)
        return InterventionPlan(
            family=self.family, kind="instant", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=self._causal_order(actor_id, other_id),
            params=dict(self._params(ratio), collision_frame=int(t0),
                        actor=actor_id, other=other_id),
            magnitude=float(dv * abs(1.0 - 1.0 / ratio) / max(g_dt, 1e-9)),
            magnitude_unit=self.magnitude_unit, severity_bin=severity_bin,
            notes={"radius": float(traj.radius[ia]),
                   "surface_top": _geom.surface_top(spec, actor),
                   "ratio": ratio, "actor_id": actor_id, "other_id": other_id,
                   "normal": [float(x) for x in normal],
                   "r_strong": float(r_strong), "lawful_dv": dv})

    def apply(self, spec, traj, plan) -> Trajectory:
        n = plan.notes
        out = self._edited(spec, traj, int(n["actor_id"]), int(n["other_id"]),
                           int(plan.params["collision_frame"]),
                           np.asarray(n["normal"], np.float64), float(n["ratio"]))
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out

    # ------------------------------------------------------------------ #
    def _edited(self, spec, traj, actor_id, other_id, t0, normal, ratio):
        out = self._clone(traj)
        for body_id, v_new in self._outcome(traj, actor_id, other_id, t0,
                                            normal, ratio).items():
            body = next(b for b in spec.bodies
                        if int(b.segmentation_id) == int(body_id))
            self._rewrite_from(spec, traj, out, body, t0, v0=v_new)
        return out

    @staticmethod
    def _split(v, normal):
        """Velocity as (normal component, tangential vector)."""
        vn = float(np.dot(v, normal))
        return vn, v - vn * normal


class Newton3Reaction(_CollisionEdit):
    """In a collision, only one body responds.

    The struck body carries straight on as though nothing hit it, while its
    partner rebounds normally -- so the pair's total momentum changes with no
    external force. **Momentum is not conserved, and that is the whole claim.**

    Implemented as a very large but finite suppression rather than a total one,
    so the victim still shows a sliver of response; a perfectly rigid
    non-response reads as a rendering glitch rather than as physics going wrong.
    """

    family = "newton3_reaction"
    magnitude_unit = "momentum_imbalance"
    RATIO_BY_BIN = {"weak": 4.0, "medium": 20.0, "strong": 200.0}

    @staticmethod
    def _params(ratio):
        return {"type": "suppress_reaction", "response_fraction": 1.0 / ratio}

    @staticmethod
    def _causal_order(actor_id, other_id):
        # The victim leads: it is the body whose behaviour is wrong, and
        # `causal_body_ids[0]` is what the residual is measured on.
        return [int(other_id), int(actor_id)]

    def _outcome(self, traj, actor_id, other_id, t0, normal, ratio):
        """The victim keeps almost all of its incoming velocity."""
        bi = traj.index_of(other_id)
        v_pre = traj.lin_vel[t0 - 1, bi].astype(np.float64)
        v_post = traj.lin_vel[t0, bi].astype(np.float64)
        return {other_id: v_pre + (v_post - v_pre) / ratio}


class Newton2Mass(_CollisionEdit):
    """Two identical-looking bodies collide as though their masses differed.

    The distinction from `newton3_reaction` is the one that makes both families
    worth having. **Both bodies respond, and the exchange is exactly what a
    lawful collision between masses `k*m` and `m` would produce** -- the
    velocity changes come out in the ratio `1:k`, the restitution is the one
    the valid twin used, and the whole outcome is internally consistent. The
    only thing wrong is that the two bodies are visually identical, so nothing
    in the image justifies the mass ratio the collision implies.

    Worth being precise about what is *not* claimed here, because the obvious
    formulation is self-contradicting: this does not conserve momentum for the
    bodies' true masses. It cannot. An outcome inconsistent with equal masses
    and an outcome that conserves equal-mass momentum are the same constraint,
    so "wrong mass ratio but momentum still conserved" describes nothing. What
    separates the two families is how many bodies react: here both do, in a
    fixed and self-consistent ratio; under `newton3_reaction` one simply does
    not react at all.

    That makes it the family most dependent on its valid twin -- "that ball
    barely moved" is not evidence of anything on its own -- and it is why
    `ball_collision` samples both balls with the same radius and the same
    colour. It is also why the family is not offered on `pyramid_impact`: a cube
    and a sphere are plainly different objects, so "the cube is heavy" is a
    lawful reading and there is no violation left to see.
    """

    family = "newton2_mass"
    magnitude_unit = "effective_mass_ratio"
    RATIO_BY_BIN = {"weak": 3.0, "medium": 8.0, "strong": 25.0}

    @staticmethod
    def _params(ratio):
        return {"type": "effective_mass_scale", "mass_ratio": ratio}

    @staticmethod
    def _causal_order(actor_id, other_id):
        return [int(actor_id), int(other_id)]

    def _outcome(self, traj, actor_id, other_id, t0, normal, ratio):
        """Re-solve the collision for effective masses (ratio*m, m).

        Standard one-dimensional restitution along the line of centres, with the
        tangential components untouched. The restitution is recovered from the
        lawful rollout so the invalid clip differs from its twin in the mass
        ratio and nothing else.
        """
        ia, ib = traj.index_of(actor_id), traj.index_of(other_id)
        ua, ta = self._split(traj.lin_vel[t0 - 1, ia].astype(np.float64), normal)
        ub, tb = self._split(traj.lin_vel[t0 - 1, ib].astype(np.float64), normal)
        va, _ = self._split(traj.lin_vel[t0, ia].astype(np.float64), normal)
        vb, _ = self._split(traj.lin_vel[t0, ib].astype(np.float64), normal)

        approach = ua - ub
        e = 0.5 if abs(approach) < 1e-6 else float(np.clip((vb - va) / approach,
                                                           0.0, 1.0))
        ma, mb = ratio * float(traj.mass[ia]), float(traj.mass[ib])
        total = ma + mb
        p = ma * ua + mb * ub
        va_new = (p + mb * e * (ub - ua)) / total
        vb_new = (p + ma * e * (ua - ub)) / total
        return {actor_id: ta + va_new * normal,
                other_id: tb + vb_new * normal}


register(Solidity())
register(SuperElastic())
register(Newton3Reaction())
register(Newton2Mass())
