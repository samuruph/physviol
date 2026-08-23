# PhysViol — orientation

A spatio-temporally annotated physics-violation video dataset: every invalid clip ships
*where*, *when* and *how badly*, derived from the simulator rather than annotated by hand.

**Read [docs/PLAN.md](docs/PLAN.md) first — it is the single source of truth for this
project.** [docs/schema.md](docs/schema.md) holds the `meta.json` reference;
[docs/prior_art.md](docs/prior_art.md) holds the IntPhys 2 × LikePhys coverage matrix.

**Status: pre-Phase 0.** Environment verified, no clips generated yet, no package code
beyond `physviol/render/worker_smoke.py`.

## Locked decisions — don't relitigate without saying so

- **Kubric** (Blender + PyBullet in docker) for v0. MuJoCo replaces PyBullet later, behind
  the trajectory seam.
- **Three tiers.** **Tier D (debug, default)**: 128×128, 13 frames, 16 spp — **~11 s per
  valid/invalid pair**, never published; this is what you iterate on. Tier A
  (`physviol_v0`, build now): 256×256, 12 fps, 25 frames — MOVi defaults, ~44 s/clip.
  Tier B (`physviol_v1`, later): 512×512, 24 fps, 97 frames. Same generators, one config
  block apart. **Debug at Tier D; a bug found there is fixed for all three.**
- **Scenarios and injectors compose; never write per-combination code.** An injector edits
  `traj.npz`, which is scenario-agnostic, so 14 scenarios x 17 families needs 13 + 6 files,
  not 238. `taxonomy.COMPATIBILITY` selects the 48 meaningful cells.
- **No fluid in v0.** Tested: Blender 2.93.4 has Mantaflow but headless baking fails
  (`NameError: liquid_save_data_N` → `Manta::Error`), Kubric exposes no fluid objects, and a
  liquid does not fit the pose-based seam. `pour` (~200 rigid spheres) is the v0
  stand-in and is labelled `physics_medium: "granular"` — never call it fluid. True fluid and
  cloth are Phase 3.
- **CPU rendering, upstream image, unchanged.** GPU/OptiX caps at ~1.37× (only 26% of frame
  time is sampling); clip-level parallelism measures 1.92× for free. Do not build a GPU
  image without a new measurement.
- **Two different magnitudes, never conflated.** `intervention.magnitude` is the knob we
  turned (one scalar, exact, known *before* simulating, drives weak/medium/strong splits). The
  residual `r`/`z`/`s` is the measured effect (per body per frame, known *after*, becomes
  `severity_map`). There is no "severity mask": `violation_mask` is binary *where*,
  `severity_map` is continuous *how badly*. PLAN §3.4 works a full numeric example.
- **Taxonomy is four levels**: domain (7) → family (17) → scenario (14) → instance. Encoded
  as data in `physviol/taxonomy.py`, including the scenario × family compatibility matrix.
  13 scenarios are built; `clutter_toss` is DEFER-only.
- **`reference_mask` ships on both twins** -- the culprit's lawful footprint, taken from the
  valid render and ungated in time. It is the counterfactual "where it should be", and for a
  vanished body it is the only mask with any pixels. Visualisers draw it as a green outline
  *on top of* the red violation mask (drawing it under lets the fill hide it).
- **Debug generation defaults to one `strong` variant per cell**, with `--severity all` for
  the ladder and `--window N` for a uniform duration. Breadth of coverage is what you check
  first; three strengths of one cell is what you check second.
- **Violations are windows, not onsets.** `violation_windows` is a *list* of intervals —
  `superelastic` fires once per bounce, and observability can be interrupted by re-occlusion.
- **Decouple dynamics from rendering.** Simulate → write `traj.npz` → render by replaying
  keyframes. This is the seam; it is what makes valid/invalid twins bit-identical before
  `t_event`.

## Non-negotiables

1. **Prefix identity.** Valid and invalid renders must be bit-identical for every frame
   before `t_event`. If that assert fails, every downstream annotation is suspect. Anything
   that perturbs the render path — resolution, `samples_per_pixel`, denoising, motion blur,
   seeds — must be identical across a twin pair.
2. **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel
   space and it diverges everywhere downstream of the event. Ship it, label it, never train
   on it. The training targets are `violation_mask` and `severity_map`.
3. **The mask is the union over BOTH twins**:
   `violation_mask[t] = footprint(culprit, invalid, t) ∪ footprint(culprit, valid, t)`.
   Without this, vanish and teleport violations produce empty or half-empty masks — the body
   has no pixels in the invalid render precisely because it vanished. Guarded by
   `test_mask_union.py`.
4. **Injectors edit trajectories, never the sim rng.** Otherwise the twin diverges from
   frame 0 and (1) breaks.
5. **Look at the clips before scaling.** Overlay video with mask, severity and all three
   clocks burned in, every phase, before generating more.
6. **Generated output never enters git.** `data/ renders/ out/ clips/ *.npz *.mp4` are
   gitignored; keep it that way.
7. **Every asset carries a license string.** Enforced by `physviol validate`. Record it when
   you enable an asset source, not at release.

## Two environments, never mixed

- **container** (pinned image): Kubric 2022.4.1, Blender 2.93.4, PyBullet, Python 3.9 —
  scene sampling, simulation, rendering. Writes `traj.npz` + raw passes.
- **host** (`conda activate physviol`, Python 3.11): numpy/scipy/opencv/jsonschema —
  residuals, annotation, masks, severity, grids, validation, viz.

Never install Kubric, Blender or PyBullet on the host. The trajectory seam is the boundary.

## Two traps that fail silently

1. **Always run `kb.adjust_segmentation_idxs()` after `renderer.render()`.** Kubric numbers
   the raw segmentation by scene-asset order; declared `segmentation_id`s are honoured only
   by that post-process. Skipping it relabels instances whenever declaration order differs
   from insertion order and quietly corrupts every mask and residual. The worker asserts the
   rendered ids are a subset of the declared ones.
2. **"Occluded" means *fully* occluded.** A few visible actor pixels make a violation
   instantly observable and collapse the observability lag to zero. `occluder_pass` shrinks
   the occluder extents by the actor's projected silhouette radius and fires mid-run.

## Visualisation

`overlay.mp4` only -- **no image files anywhere**. The container has no ffmpeg, so it writes
arrays and every mp4 comes from `physviol/viz/video.py`. Five panels (RGB, +mask, severity,
causal, divergence), a red dot while the violation is active, and a timeline with both
window families and the three clocks.

When drawing text on frames, use `viz.overlay._text`: it draws a dark backing box rather
than a thick outline, because OpenCV's Hershey glyph advance grows with stroke thickness, so
an outline pass drawn at `thick+2` is wider than the fill and leaves dark ghost glyphs.

## Working with Kubric

Kubric is **not** a Python dependency — it lives in the docker image. Pattern is *your
script, their container*:

```bash
bash docker/kubric.sh physviol/render/worker_smoke.py --frames 4
```

`refs/kubric` is a **pinned, gitignored, read-only reference checkout** (`bash
scripts/fetch_refs.sh`). Read `refs/kubric/challenges/movi/movi_def_worker.py` — it is the
template this project adapts (PLAN Part 0.5). The clone is ~4 years newer than the image;
**the installed copy inside the container is the API authority.**

## Measured, not assumed

Blender 2.93.4 / Python 3.9.5 / kubric 2022.4.1 in the image. **1.75 s per 256² frame,
7.16 s per 512²**, linear in frames, all seven passes. Frame time fits
`T = 1.29 + 0.0074·spp` at 256² → only ~26% is sampling, so OptiX caps at ~1.37×. Clip-level
parallelism: 4 clips serial 46 s → parallel 24 s (**1.92×**). `sim_seconds` is 0.0 — physics
is free at v0 scale. See PLAN Part 0.
