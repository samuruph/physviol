"""A crude rigid-body rollout, so every cell can be smoke-tested on the host.

**Not a physics engine and not a substitute for one.** PyBullet lives in the
container and a real rollout costs a docker round trip per cell; at 48 build
cells that is most of an hour to discover that one injector has a typo. This
produces a trajectory of roughly the right shape -- things fall, land, collide
and are recorded as contacts -- which is enough to exercise every `plan()` and
`apply()` path end to end. Physical fidelity is the container's job.

Ramps are treated as axis-aligned boxes, so a block "slides" by dropping onto
the slab's bounding top. That is wrong, and deliberately so: the test asks
whether the code runs and the annotations are structurally sound, never whether
the numbers are right.
"""
from __future__ import annotations

import numpy as np

from physviol.sim.trajectory import Contacts, Trajectory

SUBSTEPS = 8


def _tops(spec):
    """Static surfaces as (top_z, cx, cy, hx, hy, seg_id), highest first."""
    out = []
    for b in spec.bodies:
        if not b.static:
            continue
        if b.kind == "cube":
            out.append((b.position[2] + b.scale[2], b.position[0], b.position[1],
                        b.scale[0], b.scale[1], int(b.segmentation_id)))
        else:
            out.append((spec.floor_level, 0.0, 0.0, 1e6, 1e6,
                        int(b.segmentation_id)))
    return sorted(out, key=lambda r: -r[0])


def _ground(tops, x, y, floor_level):
    for top, cx, cy, hx, hy, seg in tops:
        if abs(x - cx) <= hx and abs(y - cy) <= hy:
            return top, seg
    return floor_level, 0


def roll(spec, scenario=None) -> Trajectory:
    T = spec.tier.num_frames
    bodies = spec.bodies
    B = len(bodies)
    dt = 1.0 / float(spec.tier.fps)
    h = dt / SUBSTEPS
    g = np.asarray(spec.gravity, np.float64)
    tops = _tops(spec)

    pos = np.zeros((T, B, 3), np.float64)
    quat = np.zeros((T, B, 4), np.float64)
    lvel = np.zeros((T, B, 3), np.float64)
    avel = np.zeros((T, B, 3), np.float64)
    p = np.array([b.position for b in bodies], np.float64)
    v = np.array([b.velocity for b in bodies], np.float64)
    w = np.array([b.angular_velocity for b in bodies], np.float64)
    q = np.array([b.quaternion for b in bodies], np.float64)
    r = np.array([b.bounding_radius for b in bodies], np.float64)
    free = np.array([not b.sim_static for b in bodies], bool)
    seg = [int(b.segmentation_id) for b in bodies]

    cf, ca, cb, cn, cp = [], [], [], [], []

    def note(frame, a, b_, normal, point):
        cf.append(frame)
        ca.append(a)
        cb.append(b_)
        cn.append(normal)
        cp.append(point)

    for f in range(T):
        for _ in range(SUBSTEPS):
            v[free] += g * h
            p[free] += v[free] * h
            for i in range(B):
                if not free[i]:
                    continue
                top, sid = _ground(tops, p[i, 0], p[i, 1], spec.floor_level)
                if p[i, 2] - r[i] <= top and v[i, 2] < 0.0:
                    p[i, 2] = top + r[i]
                    v[i, 2] = -v[i, 2] * float(bodies[i].restitution)
                    if abs(v[i, 2]) < 0.06:
                        v[i, 2] = 0.0
                    note(f, seg[i], sid, [0.0, 0.0, 1.0], p[i].tolist())
            for i in range(B):
                for j in range(i + 1, B):
                    if not (free[i] and free[j]):
                        continue
                    d = p[j] - p[i]
                    dist = float(np.linalg.norm(d))
                    overlap = r[i] + r[j] - dist
                    if dist < 1e-6 or overlap <= 0.0:
                        continue
                    n = d / dist
                    p[i] -= n * overlap * 0.5
                    p[j] += n * overlap * 0.5
                    rel = float(np.dot(v[j] - v[i], n))
                    if rel < 0.0:
                        e = 0.5 * (bodies[i].restitution + bodies[j].restitution)
                        imp = -(1.0 + e) * rel * 0.5
                        v[i] -= n * imp
                        v[j] += n * imp
                        note(f, seg[i], seg[j], n.tolist(),
                             (p[i] + n * r[i]).tolist())
        pos[f], lvel[f], avel[f], quat[f] = p, v, w, q

    n = len(cf)
    contacts = Contacts(
        np.asarray(cf, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(ca, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cb, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cp, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.asarray(cn, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.ones((n,), np.float32), np.zeros((n,), np.float32))

    present = np.ones((T, B), bool)
    for j, b in enumerate(bodies):
        if b.dormant:
            present[:, j] = False

    traj = Trajectory(
        body_ids=np.asarray(seg, np.int32),
        body_names=[b.name for b in bodies],
        pos=pos.astype(np.float32), quat=quat.astype(np.float32),
        lin_vel=lvel.astype(np.float32), ang_vel=avel.astype(np.float32),
        present=present,
        mass=np.asarray([b.mass for b in bodies], np.float32),
        radius=r.astype(np.float32),
        is_static=np.asarray([b.static for b in bodies], bool),
        contacts=contacts, fps=float(spec.tier.fps),
        gravity=np.asarray(spec.gravity, np.float32),
        meta={"scenario": spec.scenario, "seed": spec.seed, "label": "valid",
              "spec": spec.to_dict()})
    if scenario is not None:
        scenario.script(spec, traj)
    return traj
