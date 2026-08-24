"""Mechanical energy from the trajectory seam -- docs/energy.md.

Exact, not estimated: every term comes straight off `traj.npz` and the scene
spec. Runs host-side, needs no renderer and no simulator.

The point of the module is the two *anomaly* channels, which are what a
benchmark scores against. A passive rigid-body scene under gravity, contacts and
friction can only lose mechanical energy, and only at a contact -- so energy
that moves on a contact-free frame, or moves at a contact by more than
restitution and friction permit, is unphysical by construction rather than by a
tuned threshold. Measured floor on valid clips: 0.005% of E0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

#: Frames on either side of a reported contact that still count as "at contact".
#: A step's energy change is attributable to a contact at either of its ends, and
#: PyBullet reports the frame it *noticed* a contact, which can lag the touch.
CONTACT_SLACK = 1


@dataclass
class EnergyTrace:
    """Per-frame mechanical energy and its two anomaly channels."""

    total: np.ndarray                 # [T]  joules
    kinetic_translational: np.ndarray  # [T]
    kinetic_rotational: np.ndarray    # [T]
    potential: np.ndarray             # [T]
    by_body: np.ndarray               # [T, B]
    body_ids: np.ndarray              # [B]
    dissipated: np.ndarray            # [T]  cumulative E0 - E(t), >= 0 lawfully
    free_anomaly: np.ndarray          # [T]  |dE| on contact-free frames, / E0
    contact_anomaly: np.ndarray       # [T]  energy GAINED at a contact, / E0
    excess_loss: np.ndarray           # [T]  energy lost beyond the dissipation
                                      #      budget, / E0

    def to_npz(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, np.float32) if k != "body_ids"
                else np.asarray(v, np.int32)
                for k, v in self.__dict__.items()}

    def summary(self) -> Dict[str, float]:
        e0 = float(self.total[0])
        denom = max(abs(e0), 1e-9)
        return {
            "E0": e0, "E_end": float(self.total[-1]),
            "total_dissipated": float(self.dissipated[-1]),
            "total_dissipated_fraction": float(self.dissipated[-1] / denom),
            "peak_free_anomaly": float(self.free_anomaly.max()),
            "peak_contact_anomaly": float(self.contact_anomaly.max()),
            "peak_excess_loss": float(self.excess_loss.max()),
        }


def inertia_diag(kind: str, scale, mass: float) -> np.ndarray:
    """Body-frame principal moments for the primitives this project stages.

    `scale` is the half-extent triple Kubric uses, so a cube of `scale=(h,h,h)`
    has side `2h` and a sphere of `scale=(r,r,r)` has radius `r`.
    """
    s = np.asarray(scale, np.float64).reshape(3)
    if kind == "sphere":
        r = float(s.mean())
        return np.full(3, 0.4 * mass * r * r)
    if kind in ("cube", "box"):
        a, b, c = s
        return mass / 3.0 * np.array([b * b + c * c, a * a + c * c,
                                      a * a + b * b])
    if kind == "cylinder":                 # axis = z
        r, h = float(s[0]), float(s[2])
        return np.array([mass * (3 * r * r + 4 * h * h) / 12.0,
                         mass * (3 * r * r + 4 * h * h) / 12.0,
                         0.5 * mass * r * r])
    r = float(s.mean())                    # anything else: solid-sphere stand-in
    return np.full(3, 0.4 * mass * r * r)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """[T,4] (w,x,y,z), Kubric's order -> [T,3,3] rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < 1e-12, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def contact_frames(traj, slack: int = CONTACT_SLACK) -> np.ndarray:
    """[T] bool -- frames on or beside a reported contact."""
    T = traj.num_frames
    out = np.zeros(T, bool)
    c = traj.contacts
    if len(c):
        idx = np.clip(np.asarray(c.frame, int), 0, T - 1)
        out[idx] = True
    for k in range(1, slack + 1):
        out |= np.roll(out, k) | np.roll(out, -k)
    return out


def compute(traj, spec, floor_level: Optional[float] = None) -> EnergyTrace:
    """Mechanical energy of every dynamic body, frame by frame."""
    bodies = {b.name: b for b in spec.bodies}
    g = np.asarray(traj.gravity, np.float64)
    g_norm = float(np.linalg.norm(g))
    g_hat = g / max(g_norm, 1e-12)
    datum = float(getattr(spec, "floor_level", 0.0)
                  if floor_level is None else floor_level)

    T, B = traj.num_frames, traj.num_bodies
    ke_t = np.zeros(T); ke_r = np.zeros(T); pe = np.zeros(T)
    by_body = np.zeros((T, B))

    for j, name in enumerate(traj.body_names):
        body = bodies.get(name)
        if body is None or body.static:
            continue
        m0 = float(getattr(body, "mass", 1.0))
        # Mass follows volume -- see docs/energy.md. Volume-preserving squash
        # leaves this at exactly 1, uniform growth does not, which is the
        # taxonomy's mass-vs-shape split falling out of the arithmetic.
        vol = np.prod(np.asarray(traj.scale_mul[:, j, :], np.float64), axis=1)
        m = m0 * np.clip(vol, 1e-9, None)

        v = np.asarray(traj.lin_vel[:, j, :], np.float64)
        w = np.asarray(traj.ang_vel[:, j, :], np.float64)
        p = np.asarray(traj.pos[:, j, :], np.float64)
        here = np.asarray(traj.present[:, j], bool).astype(np.float64)

        # Inertia scales as m*r^2, so a resized body's tensor moves with the
        # cube of the linear factor times the mass factor.
        lin = np.asarray(traj.scale_mul[:, j, :], np.float64)
        I0 = inertia_diag(body.kind, getattr(body, "scale", (0.5,) * 3), 1.0)
        I_body = m[:, None] * I0[None, :] * lin ** 2
        R = _quat_matrix(np.asarray(traj.quat[:, j, :], np.float64))
        w_body = np.einsum("tji,tj->ti", R, w)      # world -> body frame

        e_kt = 0.5 * m * (v ** 2).sum(1)
        e_kr = 0.5 * (I_body * w_body ** 2).sum(1)
        e_pe = m * g_norm * (-(p @ g_hat) - datum)

        ke_t += e_kt * here
        ke_r += e_kr * here
        pe += e_pe * here
        by_body[:, j] = (e_kt + e_kr + e_pe) * here

    total = ke_t + ke_r + pe
    denom = max(abs(float(total[0])), 1e-9)
    dE = np.diff(total, prepend=total[0])
    free = ~contact_frames(traj)

    free_anom = np.where(free, np.abs(dE), 0.0) / denom
    # At a contact, a *gain* is unconditionally unphysical -- no budget can
    # justify one, because contacts and friction only ever remove mechanical
    # energy. So the gain side needs no model at all.
    contact_anom = np.where(~free, np.maximum(dE, 0.0), 0.0) / denom

    # The loss side does need a budget, and the honest one is simple: a contact
    # can dissipate at most the kinetic energy actually available that frame. A
    # perfectly inelastic stop takes all of it and no more. Potential energy
    # cannot be dissipated at all -- it can only be converted by the body
    # moving, which the balance already accounts for.
    #
    # This is deliberately *not* tuned to catch everything. `newton1_inertia`
    # halts a body on the ground, which costs exactly its kinetic energy and is
    # therefore within budget: a body stopping dead is energetically identical
    # to one that hit something. That family is a momentum violation and
    # `linear_momentum` is what scores it. What this catches is energy leaving
    # with no carrier at all -- `permanence` and `dissolve` taking a body's
    # potential energy with it.
    #
    # The budget is the kinetic energy available at the START of the step, not
    # at its end. Using the end reads 77% excess loss on a perfectly lawful
    # `drop`: at the impact frame the ball has already stopped, so its remaining
    # kinetic energy is nearly zero and every joule the floor legitimately
    # absorbed looks unexplained.
    kinetic = ke_t + ke_r
    kinetic_before = np.concatenate([kinetic[:1], kinetic[:-1]])
    excess = np.maximum(-dE - kinetic_before, 0.0) / denom

    return EnergyTrace(
        total=total, kinetic_translational=ke_t, kinetic_rotational=ke_r,
        potential=pe, by_body=by_body,
        body_ids=np.asarray(traj.body_ids, np.int32),
        dissipated=total[0] - total,
        free_anomaly=free_anom, contact_anomaly=contact_anom,
        excess_loss=excess)


def per_body_free_anomaly(traj, spec) -> np.ndarray:
    """[T, B] -- each body's own energy change on frames where it has no contact.

    The per-body counterpart of `free_anomaly`, and the only one of the three
    channels that is honestly attributable to a single body. A body's energy
    may legitimately jump at a contact, because a partner supplied it; with
    nothing touching it, only gravity acts and gravity is already in the
    balance. So "this body's energy moved while nothing was touching it" needs
    no budget and no partner bookkeeping.

    Narrower than the scene-level trace by design. `permanence` and `dissolve`
    take a resting body's potential energy with it, which is a scene-level
    excess loss and not a free-energy event; `energy.npz` carries those and this
    law does not claim them.
    """
    trace = compute(traj, spec)
    T, B = traj.num_frames, traj.num_bodies
    out = np.zeros((T, B))
    denom = max(abs(float(trace.total[0])), 1e-9)
    c = traj.contacts
    for j in range(B):
        touched = np.zeros(T, bool)
        if len(c):
            bid = int(traj.body_ids[j])
            m = (np.asarray(c.body_a, int) == bid) | (np.asarray(c.body_b, int) == bid)
            if m.any():
                touched[np.clip(np.asarray(c.frame, int)[m], 0, T - 1)] = True
        for k in range(1, CONTACT_SLACK + 1):
            touched |= np.roll(touched, k) | np.roll(touched, -k)
        d = np.diff(trace.by_body[:, j], prepend=trace.by_body[0, j])
        out[:, j] = np.where(~touched, np.abs(d), 0.0) / denom
    return out


def energy_map(trace: EnergyTrace, seg: np.ndarray) -> np.ndarray:
    """[T, H, W] -- each body's energy painted onto its own pixels.

    The same mechanism `severity_map` uses, and the same honest limit: energy is
    a per-body quantity, so it is constant inside a rigid body's silhouette.
    Resolving it further would invent structure the physics does not have.
    """
    T = trace.by_body.shape[0]
    out = np.zeros((T,) + seg.shape[1:], np.float32)
    denom = max(abs(float(trace.total[0])), 1e-9)
    for j, bid in enumerate(np.asarray(trace.body_ids, int)):
        e = trace.by_body[:, j] / denom
        for t in range(T):
            if e[t] != 0.0:
                out[t][seg[t] == bid] = np.float32(e[t])
    return out
