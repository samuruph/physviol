"""Severity fields -- docs/PLAN.md 3.4.

Six steps from simulator state to a trainable [T,H,W] field:

  1. per-body, per-frame residual r(b,t)      (residuals/laws.py)
  2. calibrate the noise floor mu, sigma      (over the VALID arm)
  3. bound it:  s = clip((r - mu) / (r_strong - mu), 0, 1)
  4. paint into pixels via seg
  5. reduce to the severity_t curve
  6. special cases (global extent, shadow, two-body families)

Note on step 3: writing the bound as (r - mu)/(r_strong - mu) is algebraically the
z-score ratio z/z_ref with sigma cancelling out. That matters in practice --
geometric residuals like penetration have sigma == 0 on valid clips, so the
explicit z form divides by zero while this form stays well defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class NoiseFloor:
    """Per (scenario, law) residual distribution measured on valid clips."""

    mu: float
    sigma: float
    n_samples: int
    sigma_min: float = 1e-3

    @property
    def sigma_eff(self) -> float:
        return max(self.sigma, self.sigma_min)

    def z(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, np.float64) - self.mu) / self.sigma_eff

    def to_dict(self) -> Dict[str, float]:
        return {"mu": float(self.mu), "sigma": float(self.sigma),
                "sigma_eff": float(self.sigma_eff), "n_samples": int(self.n_samples)}

    @staticmethod
    def calibrate(valid_residuals: Sequence[np.ndarray]) -> "NoiseFloor":
        if not len(valid_residuals):
            return NoiseFloor(0.0, 0.0, 0)
        flat = np.concatenate([np.asarray(r, np.float64).ravel()
                               for r in valid_residuals])
        return NoiseFloor(float(flat.mean()), float(flat.std()), int(flat.size))


def bounded_score(r: np.ndarray, floor: NoiseFloor, r_strong: float) -> np.ndarray:
    """Step 3. `r_strong` is the family's residual at its `hard` bin."""
    denom = max(float(r_strong) - floor.mu, 1e-9)
    return np.clip((np.asarray(r, np.float64) - floor.mu) / denom, 0.0, 1.0)


def paint(seg: np.ndarray, score_by_body: Dict[int, np.ndarray],
          active: Optional[np.ndarray] = None,
          global_value: Optional[np.ndarray] = None) -> np.ndarray:
    """Step 4. Each pixel takes the score of the body occupying it.

    `seg` already resolves occlusion, so no depth reasoning is needed. Overlaps
    cannot occur (a pixel has one instance id), so `max` is implicit.
    `global_value` handles the no-culprit case (`global_gravity`) by filling the
    whole frame instead.
    """
    s = seg[..., 0] if seg.ndim == 4 else seg
    T = s.shape[0]
    out = np.zeros(s.shape, np.float32)

    if global_value is not None:
        out[:] = np.asarray(global_value, np.float32)[:, None, None]
    else:
        for body_id, score in score_by_body.items():
            m = s == body_id
            sc = np.asarray(score, np.float32)
            out = np.where(m, sc[:, None, None], out)

    if active is not None:
        out *= np.asarray(active, bool)[:, None, None]
    return out.astype(np.float16)


def temporal_profile(severity_map: np.ndarray) -> np.ndarray:
    """Step 5. severity_t[t] == severity_map[t].max() -- a schema guarantee."""
    T = severity_map.shape[0]
    return severity_map.reshape(T, -1).max(axis=1).astype(np.float32)


def peak(residual: np.ndarray, score: np.ndarray, floor: NoiseFloor,
         law: str) -> Dict[str, object]:
    """The `peak_residual` block of meta.json."""
    f = int(np.argmax(residual))
    return {"law": law, "value": float(residual[f]),
            "z_vs_valid": float(floor.z(np.asarray([residual[f]]))[0]),
            "score": float(score[f]), "frame": f}
