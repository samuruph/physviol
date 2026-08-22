"""Geometry every injector needs, in one place.

Injectors compose with scenarios by asking questions about *state* -- "what is
under this body?", "when is it airborne?" -- never about scenario names. These
helpers are the vocabulary for those questions, and keeping them here is what
stops thirteen scenarios times sixteen families from becoming per-combination
code.

py3.9-compatible: runs inside the container.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------- bodies --
def actors(spec) -> List:
    """Every body a violation may act on. Several scenarios have more than one
    (`ball_collision`, `granular_pour`), which is why this is a list."""
    return [b for b in spec.bodies if b.role == "actor"]


def actor(spec):
    a = actors(spec)
    return a[0] if a else None


def dynamic_bodies(spec) -> List:
    return [b for b in spec.bodies if not b.static]


def top_of(spec, body) -> float:
    """World z of the walkable top face of `body`."""
    if body is None:
        return float(spec.floor_level)
    if body.kind == "cube":
        return float(body.position[2] + body.scale[2])
    if body.kind == "dome":
        # KuBasic's dome is a bowl whose interior floor sits at the origin.
        return float(spec.floor_level)
    return float(body.position[2] + max(body.scale))


def support_under(spec, body) -> Tuple[Optional[object], float]:
    """The static surface directly beneath `body`, and its top z.

    "Directly beneath" is decided by height, not by declaration order: a
    scenario with both a floor and a table has two static surfaces, and a mug
    on the table is supported by the higher one. Picking the first static body
    in the list -- which is what a naive helper does -- silently annotates the
    mug as floating a table's height above its support.
    """
    lowest = float(body.position[2]) - float(body.bounding_radius) + 1e-3
    best, best_top = None, None
    for other in spec.bodies:
        if other is body or not other.static:
            continue
        t = top_of(spec, other)
        if t <= lowest and (best_top is None or t > best_top):
            best, best_top = other, t
    if best_top is None:
        return None, float(spec.floor_level)
    return best, float(best_top)


def surface_top(spec, body=None) -> float:
    """Height of whatever `body` would land on. Defaults to the first actor."""
    body = body if body is not None else actor(spec)
    if body is None:
        return float(spec.floor_level)
    return support_under(spec, body)[1]


# ----------------------------------------------------------------- timing --
def occluded_midpoint(spec) -> Optional[int]:
    """The middle of the scenario's fully-occluded run, if it has one.

    Firing there is what makes `observability_lag` non-zero; the midpoint gives
    the widest margin against the geometric occlusion test being off by a frame
    at either edge.
    """
    occ = spec.notes.get("occluded_frames") or []
    return int(occ[len(occ) // 2]) if occ else None


def default_event_frame(spec, num_frames: int) -> Optional[int]:
    """When to fire, absent a physical cue: hidden if possible, else a third in.

    A third of the way in leaves the opening frames untouched (so the prefix is
    long enough to establish what lawful motion looks like) and still leaves
    two thirds for the consequences to play out.
    """
    t0 = occluded_midpoint(spec)
    if t0 is None:
        t0 = max(1, num_frames // 3)
    return int(t0) if 1 <= t0 < num_frames - 1 else None


def airborne(traj, bi: int, top: float, slack: float = 1e-3) -> np.ndarray:
    """[T] bool: frames where body `bi`'s lowest point clears the surface."""
    r = float(traj.radius[bi])
    return (traj.pos[:, bi, 2] - r) > (top + slack)


def first_contact_any(traj, body_id: int, exclude=()) -> Optional[Tuple[int, int]]:
    """(frame, other_body_id) of this body's earliest recorded contact.

    Static partners win ties -- a ball arriving at the floor is a cleaner event
    to hang a violation on than a graze against another moving body on the same
    frame.
    """
    c = traj.contacts
    if not len(c):
        return None
    best = None
    for k in range(len(c)):
        a, b = int(c.body_a[k]), int(c.body_b[k])
        if body_id not in (a, b):
            continue
        other = b if a == body_id else a
        if other in exclude:
            continue
        f = int(c.frame[k])
        if best is None or f < best[0]:
            best = (f, other)
    return best


def first_dynamic_pair_contact(spec, traj) -> Optional[Tuple[int, int, int]]:
    """(frame, id_a, id_b) of the first contact between two *dynamic* bodies.

    The event `newton2_mass` and `newton3_reaction` need: two things that both
    ought to respond, so that only one responding is visibly wrong.
    """
    dyn = {b.segmentation_id for b in spec.bodies
           if not b.static and not b.dormant}
    c = traj.contacts
    best = None
    for k in range(len(c)):
        a, b = int(c.body_a[k]), int(c.body_b[k])
        if a in dyn and b in dyn and a != b:
            f = int(c.frame[k])
            if f >= 1 and (best is None or f < best[0]):
                best = (f, a, b)
    return best


# ------------------------------------------------------------- quaternion --
def integrate_quaternion(q0: np.ndarray, omega: np.ndarray, dt: float,
                         n: int) -> np.ndarray:
    """[n,4] (w,x,y,z) from a constant body-frame angular velocity.

    Exact rather than the first-order `q + 0.5*w*q*dt` update: at 12 fps a
    tumbling body turns a large fraction of a radian per frame, and the
    first-order form both drifts off the unit sphere and visibly under-rotates.
    """
    w = np.asarray(omega, np.float64)
    speed = float(np.linalg.norm(w))
    out = np.zeros((n, 4), np.float32)
    q = np.asarray(q0, np.float64).copy()
    if speed < 1e-9:
        out[:] = q.astype(np.float32)
        return out
    axis = w / speed
    half = 0.5 * speed * dt
    dq = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    for f in range(n):
        q = _qmul(dq, q)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        out[f] = q.astype(np.float32)
    return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], np.float64)


def quat_about_y(angle: float) -> Tuple[float, float, float, float]:
    """(w,x,y,z) for a rotation of `angle` about +Y."""
    return (float(np.cos(angle / 2.0)), 0.0, float(np.sin(angle / 2.0)), 0.0)


def support_plane(spec, body):
    """(top, fallback, bounds) describing what `body` can land on.

    `bounds` is the supporting surface's (cx, cy, hx, hy) footprint, or None for
    an unbounded ground plane. `fallback` is what is underneath *that*.
    """
    surface, top = support_under(spec, body)
    if surface is None or surface.kind != "cube":
        return float(top), float(spec.floor_level), None
    cx, cy = float(surface.position[0]), float(surface.position[1])
    hx, hy = float(surface.scale[0]), float(surface.scale[1])
    return float(top), float(spec.floor_level), (cx, cy, hx, hy)


def floor_fn(spec, body):
    """A ground-height function (x, y) -> z for the integrators.

    A raised surface is finite. Without this a body pushed off a table keeps
    gliding along an invisible plane at table height, which renders as a
    hovering object -- a `support` violation smuggled into some other family's
    clip and annotated as neither.
    """
    top, fallback, bounds = support_plane(spec, body)
    if bounds is None or abs(top - fallback) < 1e-6:
        return lambda x, y: top
    cx, cy, hx, hy = bounds

    def ground(x, y):
        if abs(x - cx) <= hx and abs(y - cy) <= hy:
            return top
        return fallback
    return ground


def longest_airborne_run(traj, bi: int, top: float) -> Optional[Tuple[int, int]]:
    """The longest stretch of frames where body `bi` is off the surface.

    Families that bend free flight have to fire *inside* free flight. Picking a
    fixed fraction of the clip instead lands `ball_drop` mid-bounce as often as
    not, and a parabola fitted across a bounce is not a parabola.
    """
    air = airborne(traj, bi, top)
    best, run_start = None, None
    for f in range(int(air.shape[0])):
        if air[f] and run_start is None:
            run_start = f
        elif not air[f] and run_start is not None:
            if best is None or (f - run_start) > (best[1] - best[0] + 1):
                best = (run_start, f - 1)
            run_start = None
    if run_start is not None:
        end = int(air.shape[0]) - 1
        if best is None or (end - run_start + 1) > (best[1] - best[0] + 1):
            best = (run_start, end)
    return best


def contact_free_run(traj, body_id: int, min_len: int = 2) -> Optional[Tuple[int, int]]:
    """The longest stretch of frames on which nothing touches this body.

    What `angular_momentum` needs: a body in contact can change its spin
    lawfully, so a spin edit made during contact is not a violation of anything
    the residual can prove.
    """
    T = int(traj.num_frames)
    touched = np.zeros((T,), bool)
    c = traj.contacts
    for k in range(len(c)):
        if body_id in (int(c.body_a[k]), int(c.body_b[k])):
            f = int(c.frame[k])
            if 0 <= f < T:
                touched[max(0, f - 1):f + 2] = True
    best, start = None, None
    for f in range(T):
        if not touched[f] and start is None:
            start = f
        elif touched[f] and start is not None:
            if best is None or (f - start) > (best[1] - best[0] + 1):
                best = (start, f - 1)
            start = None
    if start is not None:
        if best is None or (T - start) > (best[1] - best[0] + 1):
            best = (start, T - 1)
    if best is None or (best[1] - best[0] + 1) < min_len:
        return None
    return best


def support_under_any(spec, body, traj=None, frame: int = 0):
    """Like `support_under`, but a *moving* body counts as support too.

    `support_under` deliberately ignores dynamic bodies because it feeds the
    integrators, and bouncing off something that is itself falling is not a
    plane. The `support` residual needs the opposite answer: the top block of a
    stack rests on the block below, and measuring its clearance against the
    floor two blocks down would report five radii of hover on a perfectly
    lawful clip and drown the real violation in the noise floor.
    """
    if traj is not None:
        bi = traj.index_of(int(body.segmentation_id))
        lowest = float(traj.pos[frame, bi, 2] - traj.radius[bi]) + 1e-3
    else:
        lowest = float(body.position[2]) - float(body.bounding_radius) + 1e-3
    best_top = None
    for other in spec.bodies:
        if other is body or other.dormant:
            continue
        if other.static:
            t = top_of(spec, other)
        elif traj is not None:
            oi = traj.index_of(int(other.segmentation_id))
            t = float(traj.pos[frame, oi, 2] + traj.radius[oi])
        else:
            t = float(other.position[2]) + float(other.bounding_radius)
        if t <= lowest and (best_top is None or t > best_top):
            best_top = t
    return float(spec.floor_level) if best_top is None else float(best_top)


def path_sample(pos: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Positions along a recorded path at fractional frame indices `u`.

    Lets an injector slow, stop or reverse a body *along its own lawful path*
    instead of re-integrating it. On a ramp that is the difference between a
    block that stops on the slope and a block that stops in mid-air next to it:
    the scenario's geometry is baked into the path already, so following it
    cannot produce a surface violation the annotation does not know about.
    """
    T = int(pos.shape[0])
    u = np.clip(np.asarray(u, np.float64), 0.0, T - 1.0)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (u - i0)[:, None]
    return pos[i0].astype(np.float64) * (1.0 - f) + pos[i1].astype(np.float64) * f
