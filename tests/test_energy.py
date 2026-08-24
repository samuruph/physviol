"""Mechanical energy annotation -- docs/energy.md.

Host-side only: every number here comes off a trajectory, so none of it needs
docker.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physviol import scenarios
from physviol.injectors import get as get_injector
from physviol.residuals import energy as E
from physviol.scenarios import TIERS

SEED = 777


def _scene(name, seed=SEED):
    sc = scenarios.get(name)
    spec = sc.sample(seed, TIERS["debug"], "L0")
    return sc, spec, mockroll.roll(spec, sc)


def test_free_fall_trades_potential_for_kinetic():
    """No contact and no friction, so kinetic and potential trade one for one --
    to the integrator's order, which is the honest claim.

    A semi-implicit Euler step conserves discrete energy only to O(dt), and
    `mockroll` is exactly that, so the total drifts ~0.15% per frame here.
    Real PyBullet rollouts drift 0.005% (docs/energy.md), which is why the
    anomaly channels have the headroom they do. What must hold in both is the
    *direction*: free flight never gains energy.
    """
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    early = trace.total[:4]
    assert np.allclose(early, early[0], rtol=1e-2), early
    assert np.all(np.diff(early) <= 1e-9), "free flight gained energy"
    # The trade is real, not both terms sitting still.
    ke = trace.kinetic_translational[:4]
    pe = trace.potential[:4]
    assert ke[-1] > ke[0] and pe[-1] < pe[0]


@pytest.mark.parametrize("name", ["drop", "collision", "barrier_pass",
                                  "occluder_pass", "ramp_slide"])
def test_valid_clips_never_gain_energy(name):
    """A passive scene under gravity, contacts and friction can only lose
    mechanical energy. This is the invariant every anomaly channel is measured
    against, so if it does not hold the annotation means nothing."""
    sc, spec, traj = _scene(name)
    trace = E.compute(traj, spec)
    assert trace.contact_anomaly.max() < 0.05, (
        "%s valid clip gains %.1f%% of E0 at a contact"
        % (name, 100 * trace.contact_anomaly.max()))
    assert trace.free_anomaly.max() < 0.05, (
        "%s valid clip moves %.1f%% of E0 with nothing touching it"
        % (name, 100 * trace.free_anomaly.max()))


def test_superelastic_creates_energy_and_valid_twin_does_not():
    sc, spec, traj = _scene("drop")
    inj = get_injector("superelastic")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan is not None
    invalid = inj.apply(spec, traj, plan)
    v = E.compute(traj, spec)
    i = E.compute(invalid, spec)
    assert i.contact_anomaly.max() > 20 * max(v.contact_anomaly.max(), 1e-4)
    assert i.total[-1] > v.total[-1]


def test_deformation_is_energy_neutral_but_immutability_is_not():
    """The taxonomy's mass-vs-shape split, measured rather than asserted.

    `deformation` is volume-preserving, so mass follows volume to exactly 1 and
    the energy cannot move. `immutability` scales volume uniformly, so it does.
    If this ever fails, one of the two injectors has stopped meaning what
    docs/taxonomy_v2.md says it means.
    """
    sc, spec, traj = _scene("drop")
    base = E.compute(traj, spec)
    out = {}
    for family in ("deformation", "immutability"):
        inj = get_injector(family)
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        assert plan is not None, family
        out[family] = E.compute(inj.apply(spec, traj, plan), spec)

    floor = max(base.contact_anomaly.max(), base.free_anomaly.max(), 1e-4)
    assert out["deformation"].contact_anomaly.max() <= 2.0 * floor, (
        "volume-preserving squash moved the energy budget")
    assert out["immutability"].contact_anomaly.max() > 10 * floor, (
        "uniform resize left the energy budget alone")


def test_vanishing_body_shows_up_as_excess_loss():
    """`permanence` takes a body's potential energy with it, which no contact
    can account for."""
    sc, spec, traj = _scene("barrier_pass")
    inj = get_injector("permanence")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan is not None
    i = E.compute(inj.apply(spec, traj, plan), spec)
    v = E.compute(traj, spec)
    assert i.excess_loss.max() > 0.2
    assert v.excess_loss.max() < 0.05


def test_energy_map_paints_only_the_bodies_pixels():
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    T = traj.num_frames
    seg = np.zeros((T, 8, 8), np.uint16)
    bid = int(traj.body_ids[0])
    seg[:, 2:5, 2:5] = bid
    emap = E.energy_map(trace, seg)
    assert emap.shape == (T, 8, 8)
    for t in range(T):
        outside = emap[t][seg[t] != bid]
        assert np.all(outside == 0.0)


def test_inertia_matches_the_textbook_forms():
    sphere = E.inertia_diag("sphere", (0.5, 0.5, 0.5), 2.0)
    assert np.allclose(sphere, 0.4 * 2.0 * 0.25)
    cube = E.inertia_diag("cube", (0.5, 0.5, 0.5), 3.0)     # side 1.0
    assert np.allclose(cube, 3.0 / 6.0 * 1.0 ** 2)


def test_law_is_silent_without_a_spec_rather_than_wrong():
    from physviol.residuals import laws
    sc, spec, traj = _scene("drop")
    got = laws._LAWS["energy_balance"](traj, 0, {})
    assert got.shape == (traj.num_frames,)
    assert np.all(got == 0.0)


def test_body_state_reproduces_the_shipped_energy():
    """The published clip must be self-contained: a consumer with `bodies.npz`
    and nothing else must be able to recompute the energy trace and get the
    shipped one back."""
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    st = E.body_state(traj, spec)
    here = st["present"] & ~st["static"][None, :]
    recomputed = ((st["kinetic"] + st["potential"]) * here).sum(axis=1)
    assert np.allclose(recomputed, trace.total, rtol=1e-6, atol=1e-6)


def test_body_state_columns_are_physically_consistent():
    sc, spec, traj = _scene("collision")
    st = E.body_state(traj, spec)
    # momentum = m*v, and its magnitude agrees with the vector
    assert np.allclose(st["momentum"], st["mass"][..., None] * st["velocity"],
                       rtol=1e-5, atol=1e-6)
    assert np.allclose(st["momentum_magnitude"],
                       np.linalg.norm(st["momentum"], axis=-1), rtol=1e-5)
    assert np.allclose(st["speed"], np.linalg.norm(st["velocity"], axis=-1),
                       rtol=1e-5)
    # potential = m*g*h against the same datum the trace uses
    g = float(np.linalg.norm(st["gravity"]))
    assert np.allclose(st["potential"], st["mass"] * g * st["height"],
                       rtol=1e-5, atol=1e-6)
    # Dynamic bodies only: PyBullet spells "static" as mass 0, and the floor
    # keeps that convention all the way through the seam.
    assert np.all(st["mass"][:, ~st["static"]] > 0)


def test_mass_follows_volume_in_the_shipped_columns():
    """`bodies.npz` must tell the same story the energy does: uniform growth
    adds matter, volume-preserving squash does not."""
    sc, spec, traj = _scene("drop")
    base = E.body_state(traj, spec)["mass"]
    for family, grows in (("immutability", True), ("deformation", False)):
        inj = get_injector(family)
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        assert plan is not None, family
        m = E.body_state(inj.apply(spec, traj, plan), spec)["mass"]
        changed = bool(np.abs(m - base).max() > 1e-3 * base.max())
        assert changed is grows, "%s: mass changed=%s" % (family, changed)
