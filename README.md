# PhysViol

**A spatio-temporally annotated physics-violation video dataset.** Every clip that breaks a
physical law ships with *where in the frame*, *exactly when*, and *how badly* — derived from
the simulator, not from human annotation.

> Status: **pre-Phase 0.** The design is settled and the render environment is verified; no
> clips exist yet. The design document is [docs/PLAN.md](docs/PLAN.md).

## Why

Existing intuitive-physics video benchmarks label **whole clips**. IntPhys 2 and LikePhys
ship valid/invalid pairs with one binary label per video. PhyCheck and PhyGround add
fine-grained *questions*, but their labels are human-written QA over real or AIGC video, not
simulator-derived geometry. None of them says which pixels are wrong, at which frame the law
actually broke, or by how much — so spatial-localization work on them is not merely
unsupervised, it is **unevaluable**.

PhysViol inverts the problem. We do not *recover* a violation from a rendered clip; we
*inject* it into a simulation, so onset, extent, culprit object and magnitude are known by
construction.

Three things make it more than "a synthetic benchmark with masks":

1. **Three clocks, not one onset.** `t_event` (the law breaks in simulator state),
   `t_observable` (the first frame with visual evidence — different by a second or more when
   the culprit is behind an occluder), and per-body `t_consequence`. That yields **detection
   latency** and **occlusion lag**, two metrics no current benchmark can compute.
2. **Violation as a law residual, not a difference from the valid rollout.** After the event
   a twin pair diverges *everywhere* downstream, so the pixels that differ are not the pixels
   where the violation is. Severity is computed per body per frame as a dimensionless
   residual against the specific law that was broken.
3. **Severity is a field, not a flag.** A continuous, dialable magnitude per clip and a
   per-frame `severity_map`, which is what makes principled easy/medium/hard splits possible.

## Locked decisions

| | |
|---|---|
| Stack | **Kubric** (Blender + PyBullet, docker) for v0, with an explicit trajectory seam so **MuJoCo** can replace PyBullet later |
| Master clip | **512×512, 24 fps, 96 frames (4 s)**; plus a 256×256 / 81-frame derivative |
| Severity | **both** the injected intervention magnitude and a measured per-frame law residual |
| v0 categories | Permanence, Solidity, Continuity, Dynamics — plus new families (parabolic-flight, Newton 1/2/3, support, friction, shadow, global-g) |
| Realism | procedural room + HDRI + orbit camera first; architecture must not block the cluttered-scene / moving-camera tier |
| Real→sim | schema fields reserved now, built in Phase 4 |

## Quickstart

Kubric is **not** a dependency of this package — it lives in the docker image. The pattern is
*your script, their container*.

```bash
# 1. reference checkouts (read-only, gitignored) -- optional but recommended
bash scripts/fetch_refs.sh

# 2. the image (digest pinned in docker/IMAGE_DIGEST)
docker pull kubricdockerhub/kubruntu

# 3. render something
bash docker/kubric.sh physviol/render/worker_smoke.py --frames 4
```

`docker/kubric.sh` mounts this repo at `/kubric`, runs as your uid so output is not
root-owned, and prefers the pinned digest over the floating tag.

### Verified environment

Blender 2.93.4 / Python 3.9.5 / kubric 2022.4.1 inside the image; **~7.2 s per 512² frame on
8 CPU cores** with all seven passes exported. Cutting `samples_per_pixel` 4× saves only 22%,
because the cost is the auxiliary passes and I/O, not sampling — so a GPU (OptiX) image is
scheduled work before Phase 2, not a contingency. Details in
[docs/PLAN.md](docs/PLAN.md) Part 0.

## Layout

```
docs/PLAN.md        the design document — start here
docs/schema.md      meta.json field reference
docs/prior_art.md   IntPhys 2 x LikePhys x PhysViol coverage matrix
docker/             kubric.sh wrapper + pinned image digest
scripts/            fetch_refs.sh
physviol/           the generator package (see PLAN.md Part 3)
refs/               gitignored reference checkouts
context/            source papers, gitignored (see context/README.md)
```

## Using the data

The published on-disk format is the entire contract — PhysViol has no privileged consumer,
and nothing here needs to be imported to read a clip. See
[docs/PLAN.md](docs/PLAN.md) Part 5 for the loader mapping and
[docs/schema.md](docs/schema.md) for the fields.

Two warnings worth reading before training on anything:

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space,
  shipped for analysis, and it diverges everywhere downstream of the event. Use
  `violation_mask` and `severity_map`.
- **Check `provenance.prefix_identical_verified`** rather than assuming twins are
  pixel-aligned. It records that the bit-identity test passed for that pair.
