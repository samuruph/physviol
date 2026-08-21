"""`physviol validate` -- docs/schema.md "Cross-checks".

Structural checks that catch the failure modes the design actually has, not
just JSON well-formedness.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from ..annotate import windows as win
from ..taxonomy import FAMILIES, SCENARIOS, domain_of, is_compatible


def validate_clip(cdir: str) -> List[str]:
    errs: List[str] = []

    def bad(msg):
        errs.append("%s: %s" % (os.path.basename(cdir), msg))

    mp = os.path.join(cdir, "meta.json")
    if not os.path.exists(mp):
        return ["%s: no meta.json" % cdir]
    with open(mp) as fh:
        m = json.load(fh)

    fam = m.get("family")
    if fam not in FAMILIES:
        bad("unknown family %r" % fam)
        return errs
    if m.get("scenario") not in SCENARIOS:
        bad("unknown scenario %r" % m.get("scenario"))
    # 9. domain matches family; (scenario, family) is in the matrix
    if m.get("domain") != domain_of(fam):
        bad("domain %r != %r for family %s" % (m.get("domain"),
                                               domain_of(fam), fam))
    if not is_compatible(m.get("scenario"), fam):
        bad("(%s, %s) not in the compatibility matrix"
            % (m.get("scenario"), fam))
    # 9b. physics_medium
    pm = m.get("physics_medium")
    if pm == "fluid":
        bad("physics_medium 'fluid' is not permitted at schema v0")
    if (pm == "granular") != (m.get("scenario") == "granular_pour"):
        bad("physics_medium %r inconsistent with scenario %r"
            % (pm, m.get("scenario")))
    # 10. every asset carries a licence
    for asset in m.get("assets", []):
        if not asset.get("license"):
            bad("asset %r has no license" % asset.get("name"))
    # 11. prefix identity verified
    if not m.get("provenance", {}).get("prefix_identical_verified"):
        bad("prefix_identical_verified is not true")

    v = m.get("violation")
    # 12. violation is null iff label == valid
    if (v is None) != (m.get("label") == "valid"):
        bad("violation/label mismatch (label=%r)" % m.get("label"))
    if v is None:
        return errs

    T = int(m["num_frames"])
    te, to, tend = (int(v["t_event_frame"]), int(v["t_observable_frame"]),
                    int(v["t_end_frame"]))
    # 2. ordering
    if not (te <= to <= tend):
        bad("t_event=%d <= t_observable=%d <= t_end=%d violated" % (te, to, tend))
    # 3. windows sorted, disjoint, in range; endpoints agree
    wins = [tuple(w) for w in v["violation_windows"]]
    if not wins:
        bad("no violation_windows")
    prev_end = -2
    for s, e in wins:
        if s > e:
            bad("window (%d,%d) is inverted" % (s, e))
        if s <= prev_end:
            bad("windows overlap or are unsorted at (%d,%d)" % (s, e))
        if not (0 <= s < T and 0 <= e < T):
            bad("window (%d,%d) out of range T=%d" % (s, e, T))
        prev_end = e
    if wins and (min(s for s, _ in wins) != te or max(e for _, e in wins) != tend):
        bad("t_event/t_end do not match window extremes")
    # 8. spatial_extent
    if (v.get("spatial_extent") == "global") != (fam == "global_gravity"):
        bad("spatial_extent %r inconsistent with family %s"
            % (v.get("spatial_extent"), fam))
    # 7. newton3 needs both bodies
    if fam == "newton3_reaction" and len(v.get("causal_body_ids", [])) < 2:
        bad("newton3_reaction needs >= 2 causal_body_ids")

    tl_p = os.path.join(cdir, "timelines.npz")
    vm_p = os.path.join(cdir, "violation_mask.npz")
    sm_p = os.path.join(cdir, "severity_map.npz")
    if not (os.path.exists(tl_p) and os.path.exists(vm_p) and os.path.exists(sm_p)):
        bad("missing timelines/violation_mask/severity_map")
        return errs
    tl = np.load(tl_p)
    mask = np.load(vm_p)["mask"]
    smap = np.load(sm_p)["severity"].astype(np.float32)

    # 4. active is exactly the rasterisation of the windows
    if not np.array_equal(tl["active"], win.rasterise(wins, T)):
        bad("timelines.active != rasterise(violation_windows)")
    ow = [tuple(w) for w in v["observable_windows"]]
    if not np.array_equal(tl["observable"], win.rasterise(ow, T)):
        bad("timelines.observable != rasterise(observable_windows)")
    # 5. The mask must be non-empty on every frame that is active AND
    # observable -- that is the union-rule check.
    #
    # Deliberately NOT "every active frame". When the culprit is fully occluded
    # the violation is active but has no visible extent anywhere: it is absent
    # from the invalid render *and* hidden in the valid twin, so the union of
    # both footprints is legitimately empty. An empty mask is the truthful
    # annotation there, and it is exactly the case the observability lag exists
    # to describe. Requiring pixels would force us to invent them.
    active = np.asarray(tl["active"], bool)
    observable = np.asarray(tl["observable"], bool)
    per = mask.reshape(T, -1).any(axis=1)
    empty = np.flatnonzero(active & observable & ~per)
    if empty.size:
        bad("violation_mask empty on active+observable frames %s" % empty.tolist())
    hidden = np.flatnonzero(active & ~observable)
    if hidden.size and per[hidden].any():
        bad("violation_mask non-empty on frames %s where nothing is observable"
            % np.flatnonzero(active & ~observable & per).tolist())
    # 6. severity_t == severity_map.max()
    st = np.asarray(tl["severity_t"], np.float32)
    mx = smap.reshape(T, -1).max(axis=1)
    if not np.allclose(st, mx, atol=1e-3):
        bad("severity_t != severity_map.max() (max diff %.4g)"
            % float(np.abs(st - mx).max()))
    # 7. causal ids present in seg
    seg_p = os.path.join(cdir, "seg.npz")
    if os.path.exists(seg_p):
        seg = np.load(seg_p)["seg"]
        present = set(np.unique(seg).tolist())
        for cid in v.get("causal_body_ids", []):
            if int(cid) not in present:
                bad("causal_body_id %d absent from seg.npz" % cid)
    return errs


def validate_release(root: str) -> Dict[str, object]:
    metas = sorted(glob.glob(os.path.join(root, "clips", "**", "meta.json"),
                             recursive=True))
    errs: List[str] = []
    pairs: Dict[str, List[str]] = {}
    for mp in metas:
        cdir = os.path.dirname(mp)
        errs.extend(validate_clip(cdir))
        with open(mp) as fh:
            m = json.load(fh)
        pairs.setdefault(m.get("pair_uid", "?"), []).append(m.get("label"))
    # 13. every pair has exactly one valid twin and at least one invalid.
    # Several invalid variants legitimately share a single valid clip: they come
    # from the same scenario+seed, so the valid render is bit-identical and
    # storing it once is both correct and cheaper. IntPhys 2 and LikePhys ship
    # the same one-to-many shape.
    for uid, labels in pairs.items():
        n_valid = labels.count("valid")
        n_invalid = labels.count("invalid")
        if n_valid != 1 or n_invalid < 1:
            errs.append("%s: expected 1 valid and >=1 invalid, got %d/%d"
                        % (uid, n_valid, n_invalid))
    return {"ok": not errs, "clips": len(metas), "pairs": len(pairs),
            "errors": errs}
