"""Physical-law residuals -- docs/PLAN.md 1.2 and 3.4 step 1.

Every residual is computed from `traj.npz` alone, never from pixels, and is
dimensionless so that families with different physical units stay comparable
after the noise-floor normalisation in 3.4 step 2.

Signature: law(traj, body_index, ctx) -> np.ndarray of shape [T], >= 0.
`ctx` carries scenario geometry the law needs (surface height, etc).
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from ..sim.trajectory import Trajectory

Ctx = Dict[str, Any]
_LAWS: Dict[str, Callable[[Trajectory, int, Ctx], np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        _LAWS[name] = fn
        return fn
    return deco


def get(name: str):
    if name not in _LAWS:
        raise KeyError("no residual law %r; have %s" % (name, sorted(_LAWS)))
    return _LAWS[name]


def available():
    return sorted(_LAWS)


# --------------------------------------------------------------------------
@register("penetration")
def penetration(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Solidity: how far the body's surface is inside another solid, in radii.

    Zero when clear of the surface. `ctx` needs `surface_top`; the body radius
    comes from the trajectory.
    """
    top = float(ctx["surface_top"])
    r = float(traj.radius[b])
    lowest = traj.pos[:, b, 2] - r
    depth = np.maximum(0.0, top - lowest)
    return (depth / max(r, 1e-9)).astype(np.float64)


@register("free_fall")
def free_fall(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Kinematics: |a - g| / |g| for a body that should be in free flight.

    Non-zero during legitimate contact too, so callers gate it on the frames
    where the body is actually unsupported (`ctx['unsupported']`, optional).
    """
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    r = np.linalg.norm(a - g[None, :], axis=1) / max(float(np.linalg.norm(g)), 1e-9)
    gate = ctx.get("unsupported")
    if gate is not None:
        r = r * np.asarray(gate, dtype=np.float64)
    return r


@register("mass_continuity")
def mass_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Permanence: the fraction of the body's mass missing from the scene.

    Discrete by nature -- 1.0 while absent, 0.0 while present. The IntPhys 2
    categories are all essentially discrete; that is exactly why PLAN Part 2
    adds continuously dialable families alongside them.
    """
    present = traj.present[:, b].astype(np.float64)
    return 1.0 - present


@register("position_continuity")
def position_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Continuity: displacement unexplained by the previous velocity, in radii."""
    dt = traj.dt
    p = traj.pos[:, b, :].astype(np.float64)
    v = traj.lin_vel[:, b, :].astype(np.float64)
    out = np.zeros((traj.num_frames,), np.float64)
    if traj.num_frames < 2:
        return out
    predicted = p[:-1] + v[:-1] * dt
    out[1:] = np.linalg.norm(p[1:] - predicted, axis=1) / max(float(traj.radius[b]), 1e-9)
    return out


@register("linear_momentum")
def linear_momentum(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Newton 1/2/3, phantom impulse: momentum change unaccounted for by forces.

    ||m*dv - (F_contact + m*g)*dt|| / (m*|g|*dt). With no contact recorded the
    only legal force is gravity, so this reduces to the free-fall residual.
    """
    g = traj.gravity.astype(np.float64)
    m = float(traj.mass[b])
    dt = traj.dt
    v = traj.lin_vel[:, b, :].astype(np.float64)
    dv = np.zeros_like(v)
    dv[1:] = v[1:] - v[:-1]

    f_contact = np.zeros_like(v)
    c = traj.contacts
    bid = int(traj.body_ids[b])
    if len(c):
        for k in range(len(c)):
            f = int(c.frame[k])
            if not (0 <= f < traj.num_frames):
                continue
            if int(c.body_a[k]) == bid:
                f_contact[f] += c.normal[k].astype(np.float64) * float(c.impulse[k])
            elif int(c.body_b[k]) == bid:
                f_contact[f] -= c.normal[k].astype(np.float64) * float(c.impulse[k])

    resid = m * dv - (f_contact + m * g[None, :]) * dt
    denom = max(m * float(np.linalg.norm(g)) * dt, 1e-9)
    return (np.linalg.norm(resid, axis=1) / denom).astype(np.float64)


@register("energy_at_contact")
def energy_at_contact(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Super-elastic: fractional kinetic-energy *gain* across a frame."""
    m = float(traj.mass[b])
    v = traj.lin_vel[:, b, :].astype(np.float64)
    e = 0.5 * m * np.sum(v * v, axis=1)
    out = np.zeros_like(e)
    prev = np.maximum(e[:-1], 1e-9)
    out[1:] = np.maximum(0.0, e[1:] - e[:-1]) / prev
    return out


@register("trajectory_shape")
def trajectory_shape(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Non-parabolic flight: per-frame deviation from the fitted g-parabola,
    in metres, normalised by the body radius."""
    g = traj.gravity.astype(np.float64)
    dt = traj.dt
    frames = np.asarray(ctx.get("flight_frames", np.arange(traj.num_frames)), int)
    out = np.zeros((traj.num_frames,), np.float64)
    if frames.size < 3:
        return out
    p = traj.pos[frames, b, :].astype(np.float64)
    t = (frames - frames[0])[:, None] * dt
    # Least squares for p0, v0 given known g.
    q = p - 0.5 * g[None, :] * t ** 2
    A = np.concatenate([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    fit = A @ coef + 0.5 * g[None, :] * t ** 2
    out[frames] = np.linalg.norm(p - fit, axis=1) / max(float(traj.radius[b]), 1e-9)
    return out


@register("shape_continuity")
def shape_continuity(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Immutability: how far the body's volume has drifted from its own frame 0.

    Reported as |V/V0 - 1|, so a body that doubles in volume scores 1.0 and one
    that halves scores 0.5. Asymmetric on purpose -- growing is unbounded,
    shrinking bottoms out at total disappearance, which is `permanence`'s law,
    not this one.
    """
    s = np.asarray(traj.scale_mul[:, b, :], np.float64)
    vol = np.prod(np.maximum(s, 1e-9), axis=1)
    return np.abs(vol / max(float(vol[0]), 1e-9) - 1.0)


@register("angular_momentum")
def angular_momentum(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Spin that changed with nothing to change it.

    ||dw|| * r / (|g| * dt): the frame-to-frame change in angular velocity,
    scaled to the linear units the other laws use so severities stay
    comparable. Gated to frames with no recorded contact for this body --
    a tumbling body swaps angular momentum with the ground legally, and only
    a change while free of contact is unexplained.
    """
    w = np.asarray(traj.ang_vel[:, b, :], np.float64)
    dw = np.zeros_like(w)
    dw[1:] = w[1:] - w[:-1]
    r = float(traj.radius[b])
    g = max(float(np.linalg.norm(traj.gravity)), 1e-9)
    out = np.linalg.norm(dw, axis=1) * r / (g * traj.dt)

    touching = np.zeros((traj.num_frames,), bool)
    c = traj.contacts
    bid = int(traj.body_ids[b])
    if len(c):
        for k in range(len(c)):
            f = int(c.frame[k])
            if 0 <= f < traj.num_frames and bid in (int(c.body_a[k]),
                                                    int(c.body_b[k])):
                touching[max(0, f - 1):f + 2] = True
    gate = ctx.get("contact_free")
    if gate is not None:
        touching |= ~np.asarray(gate, bool)
    return out * (~touching).astype(np.float64)


@register("support")
def support(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Equilibrium: a body held up by nothing.

    Clearance between the body's lowest point and the surface below it, in
    radii, counted only on frames where the body is *not* in free flight. Both
    halves are needed: a ball mid-arc has metres of clearance and is perfectly
    lawful, while a hovering one has the same clearance and is not. What
    separates them is whether it is accelerating at g.
    """
    top = float(ctx["surface_top"])
    r = max(float(traj.radius[b]), 1e-9)
    clearance = np.maximum(0.0, (traj.pos[:, b, 2] - r) - top) / r
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    falling = (np.linalg.norm(a - g[None, :], axis=1)
               / max(float(np.linalg.norm(g)), 1e-9)) < float(ctx.get("free_fall_tol", 0.35))
    return clearance * (~falling).astype(np.float64)


@register("friction")
def friction(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Along-track acceleration that gravity does not explain, over |g|.

    Signed component of (a - g) along the direction of travel, absolute-valued:
    it catches both "slid to a halt with nothing to stop it" and "sped up on
    its own". Legal sliding friction also registers here -- which is precisely
    what the noise floor measured on the valid twin subtracts out, so the score
    reflects only the part the intervention added.
    """
    g = traj.gravity.astype(np.float64)
    a = traj.acceleration(b).astype(np.float64)
    v = np.asarray(traj.lin_vel[:, b, :], np.float64)
    speed = np.linalg.norm(v, axis=1)
    moving = speed > float(ctx.get("min_speed", 0.05))
    vhat = v / np.maximum(speed, 1e-9)[:, None]
    along = np.sum((a - g[None, :]) * vhat, axis=1)
    return np.abs(along) / max(float(np.linalg.norm(g)), 1e-9) * moving.astype(np.float64)


@register("shadow_consistency")
def shadow_consistency(traj: Trajectory, b: int, ctx: Ctx) -> np.ndarray:
    """Optical: how far the cast shadow sits from where its caster puts it.

    `b` is the *shadow*, not the object. The expected position is the caster's
    centre projected along the light direction onto the ground plane; the
    residual is the miss distance in caster radii, so it is comparable with
    every other length-based law in the table.
    """
    ci = traj.index_of(int(ctx["caster_id"]))
    L = np.asarray(ctx["light_dir"], np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-9)
    top = float(ctx.get("surface_top", 0.0))
    p = np.asarray(traj.pos[:, ci, :], np.float64)
    denom = -L[2] if abs(L[2]) > 1e-6 else 1e-6
    t = (p[:, 2] - top) / denom
    expected = p[:, :2] + t[:, None] * L[None, :2]
    got = np.asarray(traj.pos[:, b, :2], np.float64)
    r = max(float(traj.radius[ci]), 1e-9)
    return np.linalg.norm(got - expected, axis=1) / r


# --------------------------------------------------------------------------
def compute_all(traj: Trajectory, ctx_for_body) -> Dict[str, np.ndarray]:
    """Every registered law for every body -> {law: [T, B]}."""
    T, B = traj.num_frames, traj.num_bodies
    out = {}
    for name, fn in _LAWS.items():
        arr = np.zeros((T, B), np.float64)
        for b in range(B):
            if bool(traj.is_static[b]):
                continue
            try:
                arr[:, b] = fn(traj, b, ctx_for_body(b))
            except Exception:                       # noqa: BLE001
                arr[:, b] = 0.0
        out[name] = arr
    return out
