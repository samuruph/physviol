"""Interventions that are staged in the simulator produce REAL contacts.

The defect this exists to catch, reported cell by cell: a teleported ball
bouncing off a barrier it had already been moved past, and two balls exchanging
momentum without touching. Both came from the same cause -- the trajectory was
edited after the fact and re-integrated against the scene as *declared*, so the
resolver kept a wall in front of a body that was behind it, and wrote collision
outcomes for contacts that never occurred.

Only re-simulation can pass these. A hand-rolled resolver cannot, which is the
point of the test.

Needs worker output, so it skips without docker.
"""
import json
import os

import numpy as np
import pytest

from physviol.sim.trajectory import Trajectory

#: (scenario, family) -> must the culprit still touch what it touched lawfully?
SIMULATED = ("continuity", "phantom_impulse", "newton1_inertia", "solidity")


def _plans():
    """Every variant directory any worker run left under `out/`.

    Its own discovery rather than `conftest.find_workdir`, which returns a
    single variant directory; this test wants all of them across every scenario
    a run produced.
    """
    import glob

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return sorted(glob.glob(os.path.join(root, "out", "**", "variants", "*",
                                         "plan.json"), recursive=True))


def _pairs_after(traj, t_event):
    c = traj.contacts
    return {(int(a), int(b)) for f, a, b in
            zip(c.frame, c.body_a, c.body_b) if int(f) >= t_event}


@pytest.fixture(scope="module")
def work():
    plans = _plans()
    if not plans:
        pytest.skip("no worker output; run `python -m physviol.cli generate --debug`")
    return plans


def test_staged_families_carry_simulated_contacts(work):
    """`traj.meta['contacts']` says where the contact list came from.

    'simulated' means PyBullet detected them on the edited world; 'geometric'
    means we inferred them from positions afterwards. A staged family reporting
    'geometric' has silently fallen back to the trajectory-edit path.
    """
    seen = 0
    for pj in work:
        vdir = os.path.dirname(pj)
        family = json.load(open(pj))["plan"]["family"]
        if family not in SIMULATED:
            continue
        tp = os.path.join(vdir, "traj_invalid.npz")
        if not os.path.exists(tp):
            continue
        traj = Trajectory.load(tp)
        source = traj.meta.get("contacts")
        # `solidity` on granular scenes keeps the trajectory path on purpose --
        # removing the floor under forty grains is a scene edit, not one pair.
        if source == "geometric":
            continue
        assert source == "simulated", (
            "%s reports contacts=%r" % (family, source))
        seen += 1
    if not seen:
        pytest.skip("no staged families in this worker output")


def test_a_prevented_collision_leaves_no_contact(work):
    """The exact reported defect.

    When an intervention moves a body away from, or stops it short of, something
    it lawfully hit, that contact must be **absent** from the invalid clip. Not
    merely different -- absent, because the bodies never meet.
    """
    checked = 0
    for pj in work:
        vdir = os.path.dirname(pj)
        root = vdir.split(os.sep + "variants" + os.sep)[0]
        blob = json.load(open(pj))["plan"]
        if blob["family"] not in ("continuity", "solidity"):
            continue
        tv = os.path.join(root, "traj_valid.npz")
        ti = os.path.join(vdir, "traj_invalid.npz")
        if not (os.path.exists(tv) and os.path.exists(ti)):
            continue
        a, b = Trajectory.load(tv), Trajectory.load(ti)
        if b.meta.get("contacts") != "simulated":
            continue
        te = int(blob["t_event_frame"])
        lost = _pairs_after(a, te) - _pairs_after(b, te)
        assert lost, (
            "%s: the invalid clip kept every contact the valid one had -- the "
            "body it was moved past or through is still stopping it"
            % blob["family"])
        checked += 1
    if not checked:
        pytest.skip("no simulated continuity/solidity variants present")
