"""Step PyBullet forward from a state we choose -- the simulated-intervention seam.

Until now every injector edited a *finished* trajectory, and everything after
`t_event` was re-integrated by our own resolver against sphere and box
approximations taken from the scene spec. That resolver does not know about
`scale_mul`, does not know that a teleported body is already past the wall, and
does not produce real contacts. So a clip could show a ball bouncing off a
barrier it had been moved beyond, or two bodies "colliding" without touching.

Kubric's `simulator.run(frame_start, frame_end)` continues from whatever state
the physics client is currently in rather than resetting it, so the honest fix
was available all along: put the world back to the valid state at `t_event`,
make the intervention a change the *simulator* honours, and let PyBullet run.
Prefix identity is untouched, because frames before `t_event` are copied from
the valid rollout verbatim.

This module is the loop. It exists rather than calling `simulator.run` because
`run` has no per-step hook, and a per-body gravity change -- `antigravity`,
`support` -- is a counter-force that must be applied on every substep.

Container-side only: it imports nothing, but it is handed a live Kubric
simulator.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

#: Signature of a per-step hook: `hook(physics_client, step, frame)`.
Hook = Callable[[object, int, int], None]


def steps_per_frame(scene) -> int:
    return max(int(scene.step_rate) // max(int(scene.frame_rate), 1), 1)


def reset_to(spec, objs, traj, frame: int) -> None:
    """Put every dynamic body back to its state at `frame`.

    Kubric wires `position`, `quaternion`, `velocity` and `angular_velocity`
    through to `resetBasePositionAndOrientation` and `resetBaseVelocity`, and
    is careful to restore velocities after the pose reset zeroes them -- so
    assigning the four attributes is a complete state restore.

    Called before every variant, so one variant's staged intervention cannot
    leak into the next. That is the same discipline `worker._clear_animation`
    enforces on the render side, and for the same reason: this scene is reused.
    """
    for j, body in enumerate(spec.bodies):
        if body.static:
            continue
        obj = objs[body.name]
        obj.position = tuple(float(x) for x in traj.pos[frame, j])
        obj.quaternion = tuple(float(x) for x in traj.quat[frame, j])
        obj.velocity = tuple(float(x) for x in traj.lin_vel[frame, j])
        obj.angular_velocity = tuple(float(x) for x in traj.ang_vel[frame, j])


def _body_index_map(simulator, objs, order) -> Dict[int, str]:
    """PyBullet body id -> our body name.

    `asset.linked_objects[simulator]` is the direct route and is what the newer
    Kubric documents; the installed one does not always populate it, so fall
    back to walking the client's bodies through `_obj_idx_to_asset`.
    """
    out: Dict[int, str] = {}
    for name in order:
        obj = objs[name]
        idx = getattr(obj, "linked_objects", {}).get(simulator)
        if isinstance(idx, int):
            out[idx] = name
    if out:
        return out

    import pybullet as pc

    asset_to_name = {id(objs[n]): n for n in order}
    for i in range(pc.getNumBodies()):
        idx = pc.getBodyUniqueId(i)
        try:
            asset = simulator._obj_idx_to_asset(idx)
        except Exception:                                     # noqa: BLE001
            continue
        name = asset_to_name.get(id(asset))
        if name is not None:
            out[idx] = name
    return out


def pybullet_index(simulator, objs, spec, seg_id: int):
    """PyBullet body id for a declared `segmentation_id`, or None.

    The bridge a staged intervention needs: our ids are the ones the taxonomy
    and the masks speak, PyBullet's are whatever the client assigned.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(seg_id)), None)
    if body is None:
        return None
    order = [b.name for b in spec.bodies]
    for idx, name in _body_index_map(simulator, objs, order).items():
        if name == body.name:
            return idx
    return None


def run_from(simulator, scene, spec, objs, t0: int, t_end: int,
             hooks: Sequence[Hook] = ()) -> Dict[str, np.ndarray]:
    """Simulate frames `t0..t_end` inclusive from the client's current state.

    Returns per-frame poses and the contacts actually detected, with contact
    force **averaged over the substeps of its frame** rather than summed -- the
    same correction `worker.simulate` makes, and for the same reason: one row
    per substep collapsed onto one integer frame inflates a resting body's
    contact force twentyfold.
    """
    # The INSTALLED Kubric (2022.4.1) keeps a bare connection id in
    # `physics_client` and calls the `pybullet` module directly; the newer
    # reference checkout wraps it in a `_BulletClient` at `_physics_client`.
    # The image is the API authority, and with one DIRECT connection the module
    # functions address it, so go through the module and work on both.
    import pybullet as pc

    spf = steps_per_frame(scene)
    n_frames = t_end - t0 + 1
    order = [b.name for b in spec.bodies]
    idx_of = _body_index_map(simulator, objs, order)
    seg_of = {b.name: int(b.segmentation_id) for b in spec.bodies}

    pos = np.zeros((n_frames, len(order), 3), np.float32)
    quat = np.zeros((n_frames, len(order), 4), np.float32)
    lvel = np.zeros((n_frames, len(order), 3), np.float32)
    avel = np.zeros((n_frames, len(order), 3), np.float32)
    cf: List[int] = []
    ca: List[int] = []
    cb: List[int] = []
    cp: List[Sequence[float]] = []
    cn: List[Sequence[float]] = []
    ci: List[float] = []

    for step in range(n_frames * spf):
        frame = t0 + step // spf

        for hook in hooks:
            hook(pc, step, frame)

        for contact in pc.getContactPoints():
            (_flag, body_a, body_b, _la, _lb, _pa, position_b, normal_b,
             _dist, normal_force, *_rest) = contact
            if normal_force <= 1e-6:
                continue
            a_name = idx_of.get(body_a)
            b_name = idx_of.get(body_b)
            if a_name is None or b_name is None:
                continue
            cf.append(frame)
            # Kubric's own reader swaps the pair so the stored normal points
            # from body_a towards body_b; keep that convention so the residual
            # laws see the same thing whichever path produced the trajectory.
            ca.append(seg_of[b_name])
            cb.append(seg_of[a_name])
            cp.append(position_b)
            cn.append(normal_b)
            ci.append(float(normal_force) / spf)

        if step % spf == 0:
            f = step // spf
            for obj_idx, name in idx_of.items():
                j = order.index(name)
                p, q = simulator.get_position_and_rotation(obj_idx)
                v, w = simulator.get_velocities(obj_idx)
                pos[f, j] = p
                quat[f, j] = q
                lvel[f, j] = v
                avel[f, j] = w
        pc.stepSimulation()

    n = len(cf)
    return {
        "pos": pos, "quat": quat, "lin_vel": lvel, "ang_vel": avel,
        "contact_frame": np.asarray(cf, np.int32),
        "contact_a": np.asarray(ca, np.int32),
        "contact_b": np.asarray(cb, np.int32),
        "contact_point": (np.asarray(cp, np.float32).reshape(n, 3) if n
                          else np.zeros((0, 3), np.float32)),
        "contact_normal": (np.asarray(cn, np.float32).reshape(n, 3) if n
                           else np.zeros((0, 3), np.float32)),
        "contact_impulse": np.asarray(ci, np.float32),
    }


def splice(traj_valid, tail: Dict[str, np.ndarray], t0: int):
    """Valid prefix + simulated tail, as one trajectory.

    Frames before `t0` are copied verbatim, which is what makes prefix identity
    hold by construction rather than by luck. Contacts are taken from the tail
    for `>= t0` and from the valid rollout before it, so the contact list
    describes the trajectory it is attached to.
    """
    import copy as _copy

    from ..sim.trajectory import Contacts

    out = _copy.copy(traj_valid)
    for key in ("pos", "quat", "lin_vel", "ang_vel"):
        arr = np.asarray(getattr(traj_valid, key)).copy()
        arr[t0:] = tail[key][: arr.shape[0] - t0]
        setattr(out, key, arr)
    for key in ("present", "scale_mul", "colour", "opacity"):
        setattr(out, key, np.asarray(getattr(traj_valid, key)).copy())

    old = traj_valid.contacts
    keep = np.asarray(old.frame, int) < t0 if len(old) else np.zeros((0,), bool)
    out.contacts = Contacts(
        np.concatenate([np.asarray(old.frame, np.int32)[keep], tail["contact_frame"]]),
        np.concatenate([np.asarray(old.body_a, np.int32)[keep], tail["contact_a"]]),
        np.concatenate([np.asarray(old.body_b, np.int32)[keep], tail["contact_b"]]),
        np.concatenate([np.asarray(old.point, np.float32)[keep], tail["contact_point"]]),
        np.concatenate([np.asarray(old.normal, np.float32)[keep], tail["contact_normal"]]),
        np.concatenate([np.asarray(old.impulse, np.float32)[keep], tail["contact_impulse"]]),
        np.zeros((int(keep.sum()) + len(tail["contact_frame"]),), np.float32))
    out.meta = dict(traj_valid.meta)
    out.meta["contacts"] = "simulated"
    return out
