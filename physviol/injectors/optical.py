"""Optical-domain injector: shadow.

The only family whose culprit is not the object but its *shadow*. It relies on
`shadow_track` handing the cast shadow to a body of its own -- see that
scenario's docstring for why a real Blender shadow cannot carry a mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class Shadow(Injector):
    """The shadow detaches from what casts it.

    Displaced along the ground plane, perpendicular to the light's bearing, so
    the shadow is still plausibly *a* shadow -- right shape, right darkness,
    wrong place. Offsetting it along the light bearing instead would read as the
    object being at a different height, which is a lawful reading and therefore
    a much weaker violation.

    The culprit is the shadow body alone. Listing the caster too would put the
    object's own pixels in `violation_mask`, and the object is not what is
    wrong.
    """

    family = "shadow"
    # A shadow that has come loose stays loose: the window runs to the end of
    # the clip and `--window` is ignored. Truncating it left `apply` offsetting
    # the shadow for every frame after `t1` while `active` claimed those frames
    # were lawful -- a violation the clip showed and the annotation denied.
    persistent = True
    OFFSET_RADII = {"weak": 1.2, "medium": 3.0, "strong": 5.5}

    def strong_residual_reference(self, spec) -> float:
        return float(self.OFFSET_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        shade = next((b for b in spec.bodies if b.role == "shadow"), None)
        caster = self._primary(spec)
        if shade is None or caster is None:
            return None
        T = traj.num_frames
        t0 = max(1, T // 3)
        n_win = self._window_len(max(2, T - t0), t0, T)
        t1 = min(T - 1, t0 + n_win - 1)

        light = np.asarray(spec.notes["light_dir"], np.float64)
        bearing = light[:2]
        norm = float(np.linalg.norm(bearing))
        perp = (np.array([-bearing[1], bearing[0]]) / norm if norm > 1e-6
                else np.array([1.0, 0.0]))
        offset = (self.OFFSET_RADII[severity_bin]
                  * float(caster.bounding_radius)
                  * perp * float(rng.choice([-1.0, 1.0])))

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(shade.segmentation_id)],
            params={"type": "shadow_offset",
                    "offset_m": [float(offset[0]), float(offset[1])],
                    "offset_radii": self.OFFSET_RADII[severity_bin]},
            magnitude=float(np.linalg.norm(offset)),
            magnitude_unit="shadow_caster_offset_radii",
            severity_bin=severity_bin,
            notes={"radius": float(caster.bounding_radius),
                   "surface_top": float(spec.notes.get("surface_top", 0.0)),
                   "caster_id": int(caster.segmentation_id),
                   "light_dir": [float(x) for x in light],
                   "shadow_id": int(shade.segmentation_id)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        shade = next(b for b in spec.bodies if b.role == "shadow")
        bi = traj.index_of(int(shade.segmentation_id))
        t0, t1 = plan.windows[0]
        off = np.asarray(plan.params["offset_m"], np.float32)
        # Eased in over two frames: a shadow that jumps a body-width between
        # consecutive frames reads as a dropped frame rather than as physics.
        n = t1 - t0 + 1
        ramp = np.minimum(1.0, (np.arange(n, dtype=np.float32) + 1.0) / 2.0)
        out.pos[t0:t1 + 1, bi, 0] += off[0] * ramp
        out.pos[t0:t1 + 1, bi, 1] += off[1] * ramp
        if t1 + 1 < traj.num_frames:
            out.pos[t1 + 1:, bi, 0] += off[0]
            out.pos[t1 + 1:, bi, 1] += off[1]
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Shadow())
