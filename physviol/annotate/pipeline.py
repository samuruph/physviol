"""Annotation pipeline -- runs on the HOST side of the seam.

Reads what the container worker produced (traj_*.npz, passes_*.npz, plan.json)
and writes the released clip layout of docs/PLAN.md Part 4.

    conda activate physviol
    python -m physviol.cli annotate out/phase0/ball_drop/0173
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import injectors
from ..residuals import laws
from ..scenarios import TIERS
from ..sim.trajectory import Trajectory
from ..taxonomy import FAMILIES, SCENARIOS, domain_of
from . import grids as grids_mod
from . import masks as masks_mod
from . import severity as sev_mod
from . import windows as win_mod

SCHEMA_VERSION = 0
PASS_FILES = {"depth": "depth", "forward_flow": "flow_fwd",
              "backward_flow": "flow_bwd", "normal": "normals",
              "object_coordinates": "object_coords"}


def _seg(passes) -> np.ndarray:
    s = passes["segmentation"]
    return s[..., 0] if s.ndim == 4 else s


def annotate_work(workdir: str, outroot: str, release: str = "physviol_v0",
                  write_video: bool = True,
                  only: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """Annotate the variants produced by one batched worker run.

    The worker emits one valid rollout plus N invalid variants (one per
    severity). They share a bit-identical valid render, so it is annotated once
    and every variant is scored against it.

    Pass `only` -- the variant directories this run actually produced. Several
    families share a workdir (same scenario and seed), so without it every call
    re-annotates each earlier family's variants as well: wasted work, and log
    lines attributed to the wrong family.
    """
    vroot = os.path.join(workdir, "variants")
    if not os.path.isdir(vroot):
        raise FileNotFoundError("no variants/ in %s" % workdir)
    if only:
        dirs = [d for d in only if os.path.exists(os.path.join(d, "plan.json"))]
    else:
        dirs = [os.path.join(vroot, n) for n in sorted(os.listdir(vroot))
                if os.path.exists(os.path.join(vroot, n, "plan.json"))]
    return [annotate_pair(workdir, d, outroot, release, write_video)
            for d in dirs]


def annotate_pair(workdir: str, vdir: str, outroot: str,
                  release: str = "physviol_v0",
                  write_video: bool = True) -> Dict[str, object]:
    """Turn one worker variant into a released valid/invalid clip pair."""
    with open(os.path.join(vdir, "plan.json")) as fh:
        blob = json.load(fh)
    spec_d, plan_d = blob["spec"], blob["plan"]

    scenario = spec_d["scenario"]
    seed = int(spec_d["seed"])
    tier = TIERS[spec_d["tier"]]
    family = plan_d["family"]
    law_name = FAMILIES[family].law

    traj_v = Trajectory.load(os.path.join(workdir, "traj_valid.npz"))
    traj_i = Trajectory.load(os.path.join(vdir, "traj_invalid.npz"))
    pv = np.load(os.path.join(workdir, "passes_valid.npz"))
    pi = np.load(os.path.join(vdir, "passes_invalid.npz"))
    seg_v, seg_i = _seg(pv), _seg(pi)
    T = tier.num_frames

    # ---- rebuild the scene spec so the residual context is available -----
    from .. import scenarios as scen_mod
    spec = scen_mod.get(scenario).sample(
        seed, tier, spec_d.get("complexity", {}).get("name", "L0"))
    inj = injectors.get(family)

    causal_ids: List[int] = [int(i) for i in plan_d["causal_body_ids"]]
    # Order is the injector's, not the scene's. `causal_body_ids[0]` is the
    # culprit the plan is *about* -- the body that failed to react, the half
    # that split off, the actor whose bounce gained energy -- and the residual,
    # the noise floor and `r_strong` are all measured on it. Re-deriving the
    # order by walking `spec.bodies` silently picked whichever participant the
    # scenario happened to declare first: in `pyramid_impact` that is a pyramid
    # sphere rather than the falling cube, so `superelastic` scored the wrong
    # body against the right reference and read 0.18 on a maximal violation.
    dynamic = {int(b.segmentation_id) for b in spec.bodies if not b.static}
    dynamic_ids = [i for i in causal_ids if i in dynamic]
    static_ids = [i for i in causal_ids if i not in dynamic]
    primary_id = dynamic_ids[0] if dynamic_ids else causal_ids[0]

    # The plan's notes ARE the residual context. Each injector knows what its
    # law needs -- the surface a body sank through, the frames a parabola should
    # be fitted over, which bodies count as siblings, where the light is -- and
    # passing the whole block through means adding a law never means touching
    # this file.
    ctx = dict(plan_d.get("notes", {}))
    surface_top = ctx.get("surface_top", spec.floor_level)
    ctx["surface_top"] = surface_top

    # The free-fall law asks "is this body accelerating like gravity?", which is
    # only a fair question while nothing is holding it up. Without this gate a
    # perfectly legal bounce registers as a violation, and the noise floor
    # measured on the valid arm inherits the same spikes -- which then swamps
    # the real signal on the weak bin.
    #
    # Resting on the floor is not enough of a test. A bounce completes *between*
    # sampled frames, so the actor can be airborne on both neighbours while its
    # velocity reversed in between; the finite-difference acceleration then
    # straddles the contact and reports a huge bogus residual. So also exclude
    # frames where the vertical velocity flips upward, and their neighbours,
    # since the acceleration estimate is a central difference.
    def _unsupported(tr, body_index):
        r = float(tr.radius[body_index])
        z = tr.pos[:, body_index, 2] - r
        vz = tr.lin_vel[:, body_index, 2]
        ok = z > surface_top + 1e-3

        # A bounce is a velocity reversal *at the floor*. Testing the reversal
        # alone is wrong in an instructive way: an antigravity violation also
        # flips the actor's vertical velocity, so a reversal-only gate deletes
        # exactly the frames where the violation is strongest and reports zero
        # severity at the peak. The proximity term is what distinguishes "the
        # ground pushed back" from "the law was broken".
        #
        # The clearance allowance matters because the contact completes between
        # samples: the actor can be up to about |vz| * dt above the floor on the
        # neighbouring frames.
        clear = np.maximum(0.05, np.abs(vz) * tr.dt * 1.5)
        at_floor = (z - surface_top) < clear
        bounce = np.zeros_like(ok)
        bounce[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                      & (at_floor[:-1] | at_floor[1:]))
        near = np.zeros_like(ok)
        for shift in (-1, 0, 1):
            near |= np.roll(bounce, shift)
        near[0] = bool(bounce[0])
        return (ok & ~near).astype(np.float64)

    # ---- 3.4 steps 1-3: residual, noise floor, bounded score -------------
    bi_v = traj_v.index_of(primary_id)
    bi_i = traj_i.index_of(primary_id)
    law = laws.get(law_name)
    r_valid = law(traj_v, bi_v, dict(ctx, unsupported=_unsupported(traj_v, bi_v)))
    r_invalid = law(traj_i, bi_i, dict(ctx, unsupported=_unsupported(traj_i, bi_i)))

    floor = sev_mod.NoiseFloor.calibrate([r_valid])
    # Families whose effect depends on the rollout measure their own strong-bin
    # reference at plan time and record it; the rest can answer from the spec.
    # Either way it is the *strong* bin's value even on a weak clip, or the
    # three bins would not be comparable.
    r_strong = ctx.get("r_strong")
    if not r_strong:
        r_strong = inj.strong_residual_reference(spec)
    s_invalid = sev_mod.bounded_score(r_invalid, floor, r_strong)
    s_valid = sev_mod.bounded_score(r_valid, floor, r_strong)

    # ---- 3.2 windows and timelines ---------------------------------------
    plan_windows = [tuple(w) for w in plan_d["violation_windows"]]
    active = win_mod.rasterise(plan_windows, T)

    # Observability is measured on the dynamic causal bodies only -- see the
    # note in windows.observable_frames -- and it gates the *spatial*
    # annotations, which answer "where can this be seen" rather than "when is
    # it happening". `active` remains the ground-truth timeline.
    observable = win_mod.observable_frames(seg_v, seg_i, dynamic_ids or causal_ids)
    visible, s_visible = sev_mod.attribute_to_evidence(s_invalid, active, observable)

    # ---- 3.3 masks (the union rule) --------------------------------------
    vmask = masks_mod.violation_mask(seg_v, seg_i, dynamic_ids, visible)
    rmask = masks_mod.reference_mask(seg_v, dynamic_ids)
    cmask = masks_mod.causal_mask(seg_v, seg_i, dynamic_ids, static_ids, visible)
    dmap = masks_mod.divergence_map(pv["rgba"], pi["rgba"])

    # ---- 3.4 steps 4-5: paint, then the temporal profile ------------------
    # Every dynamic culprit is painted, not just the primary. `global_gravity`
    # acts on the whole scene and `fission` on both halves; painting one body
    # would leave the severity field describing a fraction of the violation.
    smap = sev_mod.paint(seg_i, {int(b): s_visible for b in dynamic_ids or [primary_id]},
                         active=visible)
    # The union rule can mark pixels the culprit does not occupy in the invalid
    # render (it vanished / moved). Those carry the same score.
    extra = vmask & (smap == 0)
    if extra.any():
        smap = smap.astype(np.float32)
        broadcast = np.broadcast_to(s_visible.astype(np.float32)[:, None, None],
                                    smap.shape)
        smap = np.where(extra, broadcast, smap).astype(np.float16)
    sev_t = sev_mod.temporal_profile(smap)

    tinfo = win_mod.build(plan_windows, T, seg_v, seg_i, dynamic_ids or causal_ids,
                          primary_id, severity_t=sev_t, observable=observable)
    arrays = tinfo.pop("_arrays")
    arrays["severity_t"] = sev_t

    # ---- 3.6 token grids --------------------------------------------------
    g = grids_mod.reduce_all(vmask, smap.astype(np.float32),
                             tier.latent_frames, tier.latent_hw)

    # ---- write both clips -------------------------------------------------
    sev_bin = plan_d["intervention"]["severity_bin"]
    pair_uid = "%s/%s/%04d" % (release, scenario, seed)
    written = {}
    for label in ("valid", "invalid"):
        uid = "%s/%s" % (pair_uid, "valid" if label == "valid"
                         else "invalid_%s_%s" % (family, sev_bin))
        cdir = os.path.join(outroot, "clips", uid)
        os.makedirs(cdir, exist_ok=True)
        p = pv if label == "valid" else pi
        traj = traj_v if label == "valid" else traj_i

        np.savez_compressed(os.path.join(cdir, "seg.npz"),
                            seg=(seg_v if label == "valid" else seg_i).astype(np.uint16))
        # Shipped on BOTH twins: the lawful footprint of the bodies the
        # violation acts on, so "where it should be" is always available
        # without having to load the other clip.
        np.savez_compressed(os.path.join(cdir, "reference_mask.npz"), mask=rmask)
        for src, dst in PASS_FILES.items():
            if src in p.files:
                np.savez_compressed(os.path.join(cdir, "%s.npz" % dst), **{dst: p[src]})
        traj.save(os.path.join(cdir, "traj.npz"))

        if label == "invalid":
            np.savez_compressed(os.path.join(cdir, "violation_mask.npz"), mask=vmask)
            np.savez_compressed(os.path.join(cdir, "causal_mask.npz"), mask=cmask)
            np.savez_compressed(os.path.join(cdir, "severity_map.npz"), severity=smap)
            np.savez_compressed(os.path.join(cdir, "divergence_map.npz"),
                                divergence=dmap)
            np.savez_compressed(os.path.join(cdir, "grids.npz"), **g)
            np.savez_compressed(os.path.join(cdir, "timelines.npz"), **arrays)
            np.savez_compressed(os.path.join(cdir, "residuals.npz"),
                                r=r_invalid.astype(np.float32),
                                z=floor.z(r_invalid).astype(np.float32),
                                s=s_invalid.astype(np.float32),
                                law=np.array(law_name))
        else:
            zeros_t = np.zeros((T,), np.float32)
            np.savez_compressed(
                os.path.join(cdir, "timelines.npz"),
                active=np.zeros((T,), bool), observable=np.zeros((T,), bool),
                occluded=win_mod.occluded_frames(seg_v, primary_id),
                severity_t=zeros_t)
            np.savez_compressed(os.path.join(cdir, "residuals.npz"),
                                r=r_valid.astype(np.float32),
                                z=floor.z(r_valid).astype(np.float32),
                                s=s_valid.astype(np.float32),
                                law=np.array(law_name))

        if write_video:
            _write_mp4(p["rgba"], os.path.join(cdir, "rgb.mp4"), tier.fps)

        meta = _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier,
                           tinfo, floor, law_name, r_invalid, s_invalid, family,
                           scenario, seed, primary_id, sev_bin)
        with open(os.path.join(cdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        written[label] = cdir

    return {"pair_uid": pair_uid, "family": family, "severity": sev_bin,
            "clips": written,
            "t_event": tinfo["t_event_frame"],
            "t_observable": tinfo["t_observable_frame"],
            "observability_lag": tinfo["observability_lag_frames"],
            "violation_windows": tinfo["violation_windows"],
            "observable_windows": tinfo["observable_windows"],
            "peak_residual": float(r_invalid.max()),
            # The raw bounded score, and the one that actually ships. They
            # differ whenever the residual peaks on a frame no camera can see,
            # and reporting only the raw one hid a clip whose severity field
            # was entirely zero behind a cheerful "peak_s=1.00".
            "peak_score": float(s_invalid.max()),
            "peak_severity": float(sev_t.max()),
            "mask": masks_mod.summarise(vmask),
            "noise_floor": floor.to_dict()}


def _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier, tinfo,
                floor, law_name, r_inv, s_inv, family, scenario, seed,
                primary_id, sev_bin) -> Dict[str, object]:
    fam = FAMILIES[family]
    twin = "%s/%s" % (pair_uid, "invalid_%s_%s" % (family, sev_bin)
                      if label == "valid" else "valid")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "clip_uid": uid, "pair_uid": pair_uid, "twin_uid": twin,
        "label": label, "tier": tier.name, "release": release,
        "domain": domain_of(family), "family": family,
        "scenario": scenario, "seed": seed,
        "physics_medium": SCENARIOS[scenario].physics_medium,
        "complexity": spec_d.get("complexity", {}),
        "hdri_id": spec_d.get("hdri_id"),
        "intphys2_category": fam.intphys2, "likephys_domain": fam.likephys,
        "fps": tier.fps, "num_frames": tier.num_frames,
        "resolution": [tier.resolution, tier.resolution],
        "camera": {"motion": "static", "intrinsics": [], "extrinsics_per_frame": []},
        "controls": {"is_surprising_but_valid": False, "is_artifact_probe": False},
        "assets": [{"name": b["name"], "source": "kubric_primitive",
                    "license": "Apache-2.0",
                    "segmentation_id": b["segmentation_id"]}
                   for b in spec_d["bodies"]],
        "provenance": {
            "generator_commit": os.environ.get("PHYSVIOL_COMMIT", "uncommitted"),
            "kubric_image_digest": _digest(), "blender_version": "2.93.4",
            "render_seed": seed, "prefix_identical_verified": True,
            "prefix_identical_upto_frame": tinfo["t_event_frame"],
        },
        "noise_floor": {law_name: floor.to_dict()},
        "real2sim": None,
    }
    if label == "invalid":
        meta["violation"] = {
            "kind": plan_d["kind"],
            "t_event_frame": tinfo["t_event_frame"],
            "t_observable_frame": tinfo["t_observable_frame"],
            "t_end_frame": tinfo["t_end_frame"],
            "observability_lag_frames": tinfo["observability_lag_frames"],
            "violation_windows": tinfo["violation_windows"],
            "observable_windows": tinfo["observable_windows"],
            "occluded_at_event": tinfo["occluded_at_event"],
            "causal_body_ids": plan_d["causal_body_ids"],
            "spatial_extent": plan_d["spatial_extent"],
            "intervention": plan_d["intervention"],
            "consequences": [],
            "peak_residual": sev_mod.peak(r_inv, s_inv, floor, law_name),
        }
    else:
        meta["violation"] = None
    return meta


def _digest() -> str:
    for p in ("docker/IMAGE_DIGEST", os.path.join(os.path.dirname(__file__),
                                                  "..", "..", "docker",
                                                  "IMAGE_DIGEST")):
        if os.path.exists(p):
            with open(p) as fh:
                return fh.read().strip().split("@")[-1]
    return "unknown"


def _write_mp4(rgba: np.ndarray, path: str, fps: int) -> Optional[str]:
    from ..viz import video
    return video.write(rgba, path, fps=fps)
