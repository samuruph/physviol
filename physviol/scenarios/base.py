"""Scenario scaffolding -- docs/PLAN.md Part 2 Level 3.

A scenario emits a declarative SceneSpec, never Kubric objects directly. That
keeps scenario sampling testable on the host (no Kubric import) and leaves the
container worker as the only place that touches `bpy`/`kb`.

py3.9-compatible: imported inside the container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Tiers -- docs/PLAN.md "The resolution ladder"
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    name: str
    resolution: int
    fps: int
    num_frames: int
    samples_per_pixel: int
    latent_frames: int      # (num_frames - 1) // 4 + 1, must be exact
    latent_hw: int
    publishable: bool

    def validate(self) -> None:
        assert (self.num_frames - 1) % 4 == 0, (
            "%s: num_frames must be 4k+1 for exact VAE latent alignment, got %d"
            % (self.name, self.num_frames))
        assert (self.num_frames - 1) // 4 + 1 == self.latent_frames, self.name


TIERS: Dict[str, Tier] = {
    #                      res  fps  frames  spp  F_lat  HW_lat  publishable
    "D": Tier("D", 128, 12, 13, 16, 4, 8, False),
    "A": Tier("A", 256, 12, 25, 64, 7, 16, True),
    "B": Tier("B", 512, 24, 97, 64, 25, 16, True),
}
DEFAULT_TIER = "D"

for _t in TIERS.values():
    _t.validate()


# --------------------------------------------------------------------------
# Complexity -- an augmentation axis orthogonal to severity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Complexity:
    """How realistic the *scene* is, independent of how wrong the physics is.

    Severity asks "how badly is the law broken"; complexity asks "how hard is
    the scene to parse". They are orthogonal, and reporting accuracy across
    both is what separates "the model understands physics" from "the model
    copes with clutter". Mirrors the MOVi ladder.

    Asset availability was measured against the pinned image, not assumed:
    GSO 1033 objects (~0.9 s/fetch), HDRI Haven 509 environments (~1.8 s), and
    KuBasic 15 primitives including the `dome` used as an HDRI backdrop.
    """

    name: str
    background: str        # "solid" | "hdri"
    actor_assets: str      # "primitive" | "gso"
    n_distractors: int
    camera_motion: str     # "static" | "orbit" | "linear"
    motion_blur: float
    movi_analogue: str
    implemented: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "background": self.background,
                "actor_assets": self.actor_assets,
                "n_distractors": self.n_distractors,
                "camera_motion": self.camera_motion,
                "motion_blur": self.motion_blur}


COMPLEXITY: Dict[str, Complexity] = {
    "L0": Complexity("L0", "solid", "primitive", 0, "static", 0.0, "MOVi-A", True),
    "L1": Complexity("L1", "hdri", "primitive", 0, "static", 0.0, "MOVi-B", True),
    "L2": Complexity("L2", "hdri", "gso", 6, "static", 0.0, "MOVi-C", False),
    "L3": Complexity("L3", "hdri", "gso", 12, "linear", 0.0, "MOVi-D/E", False),
    "L4": Complexity("L4", "hdri", "gso", 20, "linear", 0.5, "MOVi-F", False),
}
DEFAULT_COMPLEXITY = "L0"


def implemented_complexities() -> List[str]:
    return [k for k, v in COMPLEXITY.items() if v.implemented]


# --------------------------------------------------------------------------
# Declarative scene description
# --------------------------------------------------------------------------


@dataclass
class BodySpec:
    """One rigid body. `segmentation_id` is what shows up in seg.npz."""

    name: str
    kind: str                                  # "sphere" | "cube" | "dome"
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.5
    static: bool = False
    color: Tuple[float, float, float] = (0.7, 0.7, 0.75)
    segmentation_id: int = 0
    role: str = "prop"        # prop | floor | occluder | actor | shadow

    # Motion is written by the scenario rather than solved by the simulator --
    # a pendulum bob on a rigid rod, say, which PyBullet would need a joint for
    # and Kubric exposes no joints. The body is `static` to the simulator so it
    # neither falls nor generates contacts, but it counts as *dynamic*
    # everywhere downstream so residuals and masks still treat it as an actor.
    scripted: bool = False

    # Declared but absent: the body is in the scene graph from frame 0 and
    # parked off camera, contributing no pixels. `fission` needs a second body
    # to spawn, and a body cannot be added to a Kubric scene mid-render without
    # perturbing the very render path prefix identity depends on -- so the
    # understudy is always there, and the intervention only switches it on.
    dormant: bool = False

    # Cycles ray visibility. `shadow_track` uses these to hand the cast shadow
    # to a body of its own, which is the only way a shadow can carry a
    # segmentation id and therefore a mask.
    visible_camera: bool = True
    visible_shadow: bool = True

    @property
    def bounding_radius(self) -> float:
        return float(max(self.scale))

    @property
    def sim_static(self) -> bool:
        """What the simulator is told. Scripted bodies are pinned there and
        animated from the trajectory instead."""
        return bool(self.static or self.scripted)

    @property
    def dynamic(self) -> bool:
        """What annotation believes. A scripted actor is a moving body."""
        return not bool(self.static)


@dataclass
class LightSpec:
    name: str
    position: Tuple[float, float, float]
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    intensity: float = 1.5
    kind: str = "directional"


@dataclass
class SceneSpec:
    """Everything needed to build, simulate and render a clip -- deterministically."""

    scenario: str
    seed: int
    tier: Tier
    bodies: List[BodySpec]
    lights: List[LightSpec]
    camera_position: Tuple[float, float, float]
    camera_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.06)
    floor_level: float = 0.0
    physics_medium: str = "rigid"
    complexity: str = DEFAULT_COMPLEXITY
    hdri_id: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def actors(self) -> List[BodySpec]:
        """Every body the violation may act on, in declaration order."""
        return [b for b in self.bodies if b.role == "actor"]

    def body(self, name: str) -> BodySpec:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError("no body named %r in %s" % (name, self.scenario))

    def index_of(self, name: str) -> int:
        for i, b in enumerate(self.bodies):
            if b.name == name:
                return i
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario, "seed": self.seed, "tier": self.tier.name,
            "resolution": [self.tier.resolution] * 2, "fps": self.tier.fps,
            "num_frames": self.tier.num_frames, "gravity": list(self.gravity),
            "physics_medium": self.physics_medium,
            "complexity": COMPLEXITY[self.complexity].to_dict(),
            "hdri_id": self.hdri_id,
            "bodies": [{"name": b.name, "kind": b.kind, "role": b.role,
                        "segmentation_id": b.segmentation_id, "mass": b.mass,
                        "restitution": b.restitution, "friction": b.friction,
                        "static": b.static, "scripted": b.scripted,
                        "dormant": b.dormant,
                        "scale": list(b.scale)} for b in self.bodies],
            "notes": self.notes,
        }


class Scenario:
    """Base class. Subclasses implement `sample`."""

    name: str = "unnamed"

    def sample(self, seed: int, tier: Tier,
               complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        raise NotImplementedError

    def script(self, spec: "SceneSpec", traj) -> None:
        """Overwrite the simulated rollout for bodies the scenario drives itself.

        Called once, on the *valid* trajectory, right after simulation. Default
        is a no-op; only constrained scenarios (`pendulum_swing`) need it. It
        runs before any injector, so the seam's guarantee is untouched: an
        injector still edits a finished trajectory and the prefix stays
        bit-identical.
        """
        return None

    def rescript(self, spec: "SceneSpec", traj, t0: int,
                 omega_scale: float) -> bool:
        """Re-run the scenario's own constrained motion from `t0` with its
        angular rate multiplied by `omega_scale`. Return False if unsupported.

        The division of labour that keeps injectors scenario-agnostic: the
        *scenario* owns its constraint (a pendulum knows it swings about a
        pivot), the *injector* owns the intervention (angular momentum reverses).
        Without this an `angular_momentum` injector would have to know what a
        pendulum is, which is the per-combination code the whole design avoids.
        Injectors dispatch on `spec.notes["constraint"]`, never on the
        scenario's name.
        """
        return False

    @staticmethod
    def rng(seed: int) -> np.random.RandomState:
        # RandomState (not default_rng) so the stream is identical under
        # numpy 1.21 in the container and numpy 2.x on the host.
        return np.random.RandomState(seed)


_REGISTRY: Dict[str, Scenario] = {}


def register(scenario: Scenario) -> Scenario:
    _REGISTRY[scenario.name] = scenario
    return scenario


def get(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError("unknown scenario %r; have %s" % (name, sorted(_REGISTRY)))
    return _REGISTRY[name]


def available() -> Sequence[str]:
    return sorted(_REGISTRY)
