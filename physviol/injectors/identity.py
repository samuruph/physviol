"""Identity-domain injectors: permanence.

`permanence` is the family that proves the union rule matters: the actor is
*gone* from the invalid render, so a mask built from the invalid segmentation
alone would be empty exactly when the violation is happening.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from .base import Injector, InterventionPlan, register


class Permanence(Injector):
    """Remove the actor from the scene, ideally while it is occluded.

    Severity is how long it stays gone: `easy` blinks out briefly, `hard` never
    returns. When the scenario provides an occlusion interval the removal fires
    inside it, so the violation is not directly observable and only its failure
    to re-emerge betrays it -- the observability-lag case of PLAN 1.1.
    """

    family = "permanence"
    GONE_FRACTION = {"weak": 0.25, "medium": 0.55, "strong": 1.0}

    def strong_residual_reference(self, spec) -> float:
        return 1.0                       # all of the actor's mass is missing

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        actor = next((b for b in spec.bodies if b.role == "actor"), None)
        if actor is None:
            return None
        T = traj.num_frames
        occ = spec.notes.get("occluded_frames") or []
        # Fire in the middle of the fully-occluded run: the widest margin
        # against the geometric test being off by a frame at either edge.
        t0 = int(occ[len(occ) // 2]) if occ else max(1, T // 3)
        if not (1 <= t0 < T - 1):
            return None

        frac = self.GONE_FRACTION[severity_bin]
        n_win = self._window_len(max(1, int(round(frac * (T - t0)))), t0, T)
        t1 = min(T - 1, t0 + n_win - 1)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[actor.segmentation_id],
            params={"type": "remove_body",
                    "body": int(actor.segmentation_id),
                    "frames_absent": int(t1 - t0 + 1)},
            magnitude=1.0, magnitude_unit="mass_ratio_removed",
            severity_bin=severity_bin,
            notes={"radius": float(actor.bounding_radius),
                   "occluded_at_event": bool(t0 in occ)})

    def apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        actor = next(b for b in spec.bodies if b.role == "actor")
        bi = traj.index_of(actor.segmentation_id)
        t0, t1 = plan.windows[0]
        out.present = traj.present.copy()
        out.present[t0:t1 + 1, bi] = False
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Permanence())
