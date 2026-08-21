"""Injector invariants -- the ones whose violation is silent and expensive."""
import numpy as np
import pytest

from physviol import injectors, scenarios
from physviol.scenarios import TIERS
from physviol.sim.trajectory import Contacts, Trajectory, prefix_identical


def _valid_traj(spec, T=13, fps=12.0):
    """A hand-rolled free-fall + bounce rollout, so these tests need no render."""
    bodies = spec.bodies
    B = len(bodies)
    dt = 1.0 / fps
    pos = np.zeros((T, B, 3), np.float32)
    vel = np.zeros((T, B, 3), np.float32)
    g = np.array([0, 0, -9.81], np.float64)
    for j, b in enumerate(bodies):
        p = np.asarray(b.position, np.float64)
        v = np.asarray(b.velocity, np.float64)
        r = b.bounding_radius
        for t in range(T):
            pos[t, j] = p
            vel[t, j] = v
            if not b.static:
                v = v + g * dt
                p = p + v * dt
                if p[2] - r <= 0.0 and v[2] < 0:
                    p[2] = r
                    v[2] = -v[2] * b.restitution
    return Trajectory(
        body_ids=np.array([b.segmentation_id for b in bodies], np.int32),
        body_names=[b.name for b in bodies], pos=pos,
        quat=np.tile(np.array([1, 0, 0, 0], np.float32), (T, B, 1)),
        lin_vel=vel, ang_vel=np.zeros((T, B, 3), np.float32),
        mass=np.array([b.mass for b in bodies], np.float32),
        radius=np.array([b.bounding_radius for b in bodies], np.float32),
        is_static=np.array([b.static for b in bodies]),
        contacts=Contacts.empty(), fps=fps,
        gravity=np.array([0, 0, -9.81], np.float32), meta={})


@pytest.fixture
def spec():
    return scenarios.get("ball_drop").sample(555, TIERS["D"], "L0")


@pytest.mark.parametrize("family", ["antigravity", "continuity", "solidity",
                                    "permanence"])
def test_prefix_is_identical_by_construction(spec, family):
    """Every injector must leave frames before t_event untouched. This is what
    makes the pixel-level twin identity possible at all."""
    traj = _valid_traj(spec)
    inj = injectors.get(family)
    plan = inj.plan(spec, traj, np.random.RandomState(0), "medium")
    assert plan is not None
    out = inj.apply(spec, traj, plan)
    ok, why = prefix_identical(traj, out, plan.t_event, atol=0.0)
    assert ok, why


@pytest.mark.parametrize("severity", ["weak", "medium", "strong"])
def test_antigravity_does_not_smuggle_in_a_solidity_violation(spec, severity):
    """Bending gravity must not also drop the actor through the floor: that is a
    *different* family, and a clip carrying two violations while annotating one
    is worse than useless."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), severity)
    out = inj.apply(spec, traj, plan)
    bi = out.index_of(spec.body("ball").segmentation_id)
    lowest = (out.pos[:, bi, 2] - spec.body("ball").bounding_radius).min()
    assert lowest > -1e-3, "actor sank %.3f m below the floor" % -lowest


def test_antigravity_intervention_varies_over_time(spec):
    """The violation has a shape: alpha ramps up and releases, so the residual
    -- and with it severity_map -- is a curve rather than one value held flat."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    out = inj.apply(spec, traj, plan)
    alpha = out.meta["alpha_profile"]
    assert len(alpha) >= 3
    # Every frame in the window is genuinely violating -- alpha never sits at
    # 1.0, which would mark an active frame where nothing is wrong. The return
    # to normal physics happens on the frame after the window.
    assert all(abs(a - 1.0) > 1e-6 for a in alpha), alpha
    # ...but it has a shape: ramp in, hold, ramp out.
    assert len(set(round(a, 3) for a in alpha)) > 1, "profile must not be flat"
    assert alpha[0] != alpha[len(alpha) // 2], "must ramp, not step"
    assert min(alpha) < 0.0, "strong bin should reverse gravity at its peak"
    peak = min(alpha)
    assert sum(1 for a in alpha if abs(a - peak) < 1e-9) >= 2, "needs a plateau"


def test_severity_bins_are_visibly_different_not_just_numerically(spec):
    """The strong bin must actually reverse the actor's direction. A bin that is
    numerically larger but looks the same is useless as a difficulty axis."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    bi = traj.index_of(spec.body("ball").segmentation_id)
    rises = {}
    for sev in ("weak", "medium", "strong"):
        plan = inj.plan(spec, traj, np.random.RandomState(0), sev)
        out = inj.apply(spec, traj, plan)
        t0, t1 = plan.windows[0]
        z = out.pos[t0 - 1:t1 + 1, bi, 2]
        rises[sev] = float(z.max() - z[0])
    assert rises["strong"] > 0.25, (
        "strong antigravity should visibly lift the actor, got %.3f m"
        % rises["strong"])
    assert rises["strong"] > rises["medium"], rises


def test_severity_bins_are_ordered(spec):
    """weak < medium < strong, by intervention magnitude, for every family."""
    traj = _valid_traj(spec)
    for family in ("antigravity", "continuity", "solidity"):
        inj = injectors.get(family)
        mags = []
        for sev in ("weak", "medium", "strong"):
            plan = inj.plan(spec, traj, np.random.RandomState(0), sev)
            assert plan is not None, family
            mags.append(plan.magnitude)
        assert mags[0] < mags[1] < mags[2], "%s: %s" % (family, mags)


def test_antigravity_window_ends_before_the_clip_does(spec):
    """Real physics must resume, otherwise there is no 'after' to compare."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan.t_end < traj.num_frames - 1


def test_bounce_gate_does_not_eat_the_antigravity_signal():
    """A bounce is a velocity reversal *at the floor*.

    Gating the free-fall residual on velocity reversal alone is wrong in an
    instructive way: reversed gravity also flips the actor's vertical velocity,
    so the gate deletes exactly the frames where the violation peaks and reports
    zero severity at its strongest moment. Only proximity to the floor
    distinguishes "the ground pushed back" from "the law was broken".
    """
    dt = 1.0 / 12.0
    # A body high above the floor whose velocity reverses -- antigravity, not a
    # bounce: floor is at z=0 and the body never comes near it.
    z = np.array([3.0, 2.6, 2.3, 2.2, 2.4, 2.9], np.float64)
    vz = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], np.float64)
    radius, surface_top = 0.3, 0.0

    clear = np.maximum(0.05, np.abs(vz) * dt * 1.5)
    at_floor = (z - radius - surface_top) < clear
    bounce = np.zeros(len(z), bool)
    bounce[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                  & (at_floor[:-1] | at_floor[1:]))
    assert not bounce.any(), "a mid-air reversal must not be treated as a bounce"

    # The same reversal happening at the floor *is* a bounce.
    z_low = np.array([1.0, 0.5, 0.32, 0.32, 0.5, 1.0], np.float64)
    at_floor_low = (z_low - radius - surface_top) < clear
    bounce_low = np.zeros(len(z_low), bool)
    bounce_low[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                      & (at_floor_low[:-1] | at_floor_low[1:]))
    assert bounce_low.any(), "a reversal at the floor is a bounce"
