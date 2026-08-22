"""Every build cell plans, applies and keeps its prefix -- without docker.

The coverage check that runs in a second instead of an hour. It cannot say
whether a violation looks right; it says that no cell is dead, that no injector
crashes on a scenario it has never seen, and that the four structural
guarantees hold everywhere: a plan exists, its windows are legal, the invalid
trajectory is bit-identical before `t_event`, and something actually changed
after it.
"""
import numpy as np
import pytest

import mockroll
from physviol import injectors, scenarios
from physviol.scenarios import TIERS
from physviol.sim.trajectory import prefix_identical
from physviol.taxonomy import SEVERITY_BINS, build_cells

CELLS = [(s, f) for s, f in build_cells() if s in set(scenarios.available())]
SEED = 4242


def _prepare(scenario_name, family, severity="strong"):
    sc = scenarios.get(scenario_name)
    spec = sc.sample(SEED, TIERS["D"], "L0")
    traj = mockroll.roll(spec, sc)
    inj = injectors.get(family)
    inj.window_frames = None
    plan = inj.plan(spec, traj, np.random.RandomState(SEED + 7919), severity)
    return spec, traj, inj, plan


def test_every_family_has_an_injector():
    from physviol.taxonomy import FAMILIES
    assert set(FAMILIES) == set(injectors.available())


def test_all_build_scenarios_exist():
    missing = sorted({s for s, _ in build_cells()} - set(scenarios.available()))
    # clutter_toss is DEFER-only in the matrix, so it must not appear here.
    assert missing == [], "build cells reference unimplemented scenarios: %s" % missing


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_cell_plans_and_applies(scenario, family):
    spec, traj, inj, plan = _prepare(scenario, family)
    assert plan is not None, "%s x %s produced no plan" % (scenario, family)

    T = traj.num_frames
    assert 1 <= plan.t_event < T, "t_event %d out of range" % plan.t_event
    prev = -2
    for s, e in plan.windows:
        assert s <= e and 0 <= s < T and 0 <= e < T, (s, e, T)
        assert s > prev, "windows overlap or are unsorted at (%d,%d)" % (s, e)
        prev = e
    assert plan.causal_body_ids, "no causal bodies"
    for cid in plan.causal_body_ids:
        traj.index_of(int(cid))          # raises if the id is not in the scene

    invalid = inj.apply(spec, traj, plan)
    ok, why = prefix_identical(traj, invalid, plan.t_event)
    assert ok, "%s x %s: %s" % (scenario, family, why)

    changed = any(
        float(np.abs(getattr(invalid, a)[plan.t_event:]
                     - getattr(traj, a)[plan.t_event:]).max()) > 1e-6
        for a in ("pos", "quat", "scale_mul"))
    changed = changed or not np.array_equal(invalid.present, traj.present)
    assert changed, "%s x %s changed nothing after t_event" % (scenario, family)


@pytest.mark.parametrize("scenario,family", CELLS,
                         ids=["%s.%s" % (s, f) for s, f in CELLS])
def test_cell_magnitude_is_ordered(scenario, family):
    """weak <= medium <= strong, for every cell, on the family's own knob."""
    mags = []
    for sev in SEVERITY_BINS:
        _, _, _, plan = _prepare(scenario, family, sev)
        assert plan is not None, "%s x %s has no %s plan" % (scenario, family, sev)
        mags.append(abs(plan.magnitude))
    assert mags == sorted(mags), "%s x %s magnitudes not ordered: %s" % (
        scenario, family, mags)
