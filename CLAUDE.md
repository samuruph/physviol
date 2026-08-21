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
- **Master clip: 512×512, 24 fps, 96 frames.** Derivative: 256×256, 81 frames.
- **Severity is both** the injected intervention magnitude *and* a measured per-frame law
  residual. Never a flag.
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
3. **Injectors edit trajectories, never the sim rng.** Otherwise the twin diverges from
   frame 0 and (1) breaks.
4. **Look at the clips before scaling.** Overlay video with mask, severity and all three
   clocks burned in, every phase, before generating more.
5. **Generated output never enters git.** `data/ renders/ out/ clips/ *.npz *.mp4` are
   gitignored; keep it that way.
6. **Every asset carries a license string.** Enforced by `physviol validate`. Record it when
   you enable an asset source, not at release.

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

Blender 2.93.4 / Python 3.9.5 / kubric 2022.4.1 in the image. **~7.2 s per 512² frame on 8
CPU cores**, linear in frames. Dropping `samples_per_pixel` 4× saves only 22% — the cost is
the auxiliary passes and I/O, not sampling. A GPU (OptiX) image is scheduled work before
Phase 2. See PLAN Part 0.
