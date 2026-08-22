# PhysViol

**A spatio-temporally annotated physics-violation video dataset.** Every clip that breaks a
physical law ships with *where in the frame*, *exactly when*, *for how long*, and *how badly*
— all derived from the simulator, not from human annotation.

> **Status: all 48 build cells runnable.** 13 scenarios × 17 violation families compose
> through the trajectory seam; `physviol generate` walks the whole matrix.
> Design doc: [docs/PLAN.md](docs/PLAN.md).

---

## 1. Setup (once)

```bash
# Host environment: annotation, severity, grids, validation, viz.
# Does NOT contain Kubric/Blender/PyBullet -- those live in the docker image.
conda env create -f environment.yml
conda activate physviol

# The render image (digest pinned in docker/IMAGE_DIGEST)
docker pull kubricdockerhub/kubruntu

# Optional: read-only Kubric source for reference (gitignored)
bash scripts/fetch_refs.sh
```

**Two environments, never mixed.** The pinned container holds Kubric 2022.4.1 / Blender
2.93.4 / PyBullet (Python 3.9) and does simulation + rendering. The `physviol` conda env
(Python 3.11) does everything else. They meet at the trajectory seam (`traj.npz`).

---

## 2. The three things you will actually run

Everything below is `python -m physviol.cli <command>`, with `conda activate physviol` first.

### a. Debug loop — one cell, fastest possible

```bash
python -m physviol.cli generate --debug --complexity L0 \
    --scenario occluder_pass --family permanence --seed 777
```

`--debug` is Tier D: 128², 13 frames, 16 spp. `--complexity L0` is a solid background with a
sun lamp, which needs no network and renders ~4.6× faster than the photographic L1. Roughly
**8 s per pair**. This is the loop to iterate in — a bug found at Tier D is fixed for all
three tiers.

### b. Small subset — every family, one scenario, or a slice of the matrix

```bash
# every family that ball_drop supports (9 cells) -- one container run, one valid render
python -m physviol.cli generate --debug --complexity L0 --scenario ball_drop --seed 777

# one family everywhere it is meaningful
python -m physviol.cli generate --debug --complexity L0 --family solidity --seed 777

# the whole matrix, but stop after 6 cells
python -m physviol.cli generate --debug --complexity L0 -n 6 --seed 777

# the whole matrix, one randomisation each: 48 cells, ~13 min at L0/Tier D
bash scripts/generate_sample.sh L0 D 777 strong 4
```

`--keep-going` carries on past a cell that fails and lists the failures at the end, which is
what you want for a sweep.

### c. The full dataset

```bash
# Tier A (256px, 25 frames), photographic backgrounds, 6 randomisations per cell.
# 48 cells x 6 = 288 invalid clips, sharing 78 valid twins (one per
# scenario+seed, since every family of a scenario renders in the same run).
python -m physviol.cli generate --tier A --complexity L1 \
    --variants 6 --seed 100000 --severity strong --keep-going \
    --outdir out/physviol_v0

# ...or the full severity ladder as well (weak/medium/strong -> 3x the invalid clips,
# but the valid twin and the scene build are shared, so it is far cheaper than 3x)
python -m physviol.cli generate --tier A --complexity L1 \
    --variants 6 --severity all --seed 100000 --keep-going \
    --outdir out/physviol_v0

python -m physviol.cli validate out/physviol_v0
```

One valid clip serves every family of the same scenario and seed: the twins are
bit-identical by construction, so rendering it once is both correct and cheaper.
`validate` expects exactly that shape — one `valid` and one or more `invalid` per `pair_uid`.

**`--variants N` is the randomisation knob.** Each variant is a fresh seed, and a seed drives
every free parameter a scenario has: sizes, speeds, drop heights, restitution, colours, the
HDRI environment at L1, the camera position and aim, the lamp direction, and — where it is
physically neutral — whether the actor is a sphere or a cube. So six variants of `ball_drop ×
solidity` are six visibly different clips of the same violation, not one clip with a
different random number in the filename.

Use `--variants 1` while debugging and `--variants 5`–`8` for a release.

**Running it in parallel.** Clip-level parallelism measured **1.92×** at width 4 on this box.
Split by scenario, since one worker run covers all families of one scenario anyway:

```bash
for s in ball_drop projectile_toss occluder_pass ball_collision; do
  python -m physviol.cli generate --tier A --complexity L1 --variants 6 \
      --scenario "$s" --seed 100000 --outdir out/physviol_v0 &
done; wait
```

---

## 3. Look at what came out

**Nothing writes image files.** The container has no ffmpeg, so it writes arrays and every
mp4 is encoded host-side by `physviol/viz/video.py`.

```bash
# five-panel annotated video for ONE clip
python -m physviol.cli overlay out/release/clips/physviol_v0/ball_drop/0777/invalid_solidity_strong

# the valid clip beside every severity bin of one family
python -m physviol.cli grid out/release/clips/physviol_v0/ball_drop/0777 --family solidity

# ONE video tiling every invalid clip in the release -- the coverage check
python -m physviol.cli coverage out/release
```

`coverage.mp4` is the one to open first after a sweep: every cell at once, each tile captioned
with its scenario, family and observability lag, so a broken cell is obvious at a glance.

### Reading the overlay

| element | meaning |
|---|---|
| **red dot**, top right | the violation is active on *this* frame |
| **RGB** | the clip as released |
| **MASK** | red = `violation_mask` (the annotation); **green outline** = `reference_mask`, where the object *should* be per the valid twin |
| **SEVERITY MAP** | the residual painted into the culprit, with a live 0..1 scale |
| **CAUSAL MASK** | red = primary culprit, blue = other participants |
| **DIVERGENCE** | `\|valid - invalid\|` — shipped for analysis, **not** ground truth |
| red bar, timeline | `violation_windows` — when the law is actually broken |
| amber bar, timeline | `observable_windows` — when there is visual evidence |
| `t_event` / `t_obs` / `t_end` | the three clocks; they merge into one label when they coincide |

If `t_obs > t_event` the violation happened while the culprit was hidden — that gap is the
occlusion lag, and `occluder_pass` is the scenario built to produce it.

---

## 4. Every command

```bash
# Taxonomy: 7 domains, 17 families, 14 scenarios, 48 build cells
python -m physviol.cli taxonomy
python -m physviol.cli taxonomy -v          # + every (scenario, family) cell

# Generate (see section 2 for the flags that matter)
python -m physviol.cli generate --debug --complexity L0 --scenario ball_drop
#   --tier D|A|B      --complexity L0|L1     --severity weak|medium|strong|all
#   --variants N      --window N             --scenario X   --family Y
#   -n / --limit N    --keep-going           --no-overlay   --outdir / --workdir

# Re-annotate an existing worker output without re-rendering
python -m physviol.cli annotate out/work/ball_drop/0777 --outdir out/release

# Rebuild one overlay / one grid / the coverage tiling
python -m physviol.cli overlay out/release/clips/.../invalid_solidity_strong
python -m physviol.cli grid    out/release/clips/physviol_v0/ball_drop/0777 --family solidity
python -m physviol.cli coverage out/release

# Schema + cross-checks over a whole release (13 structural checks)
python -m physviol.cli validate out/release

# Tests -- including one that plans and applies all 48 cells without docker
python -m pytest tests/ -q
python -m pytest tests/test_all_cells.py -q
```

### Running a worker in the container directly

```bash
# throughput probe (no assets downloaded, pure render cost)
bash docker/kubric.sh physviol/render/worker_smoke.py --resolution 256 --frames 8

# the real worker: simulate + inject + render both twins.
# --family takes a comma list; they share one scene build and one valid render.
bash docker/kubric.sh physviol/render/worker.py \
    --scenario ball_drop --seed 777 --tier D --complexity L0 \
    --family solidity,antigravity --severity strong --outdir out/work
```

`docker/kubric.sh` mounts the repo at `/kubric`, runs as your uid so output is not
root-owned, and prefers the pinned digest over `:latest`.

---

## 5. What a clip contains

Output lands in `out/release/clips/<release>/<scenario>/<seed>/{valid,invalid_<family>_<bin>}/`
(gitignored, so it never enters version control).

```
meta.json            labels, taxonomy, violation windows, severity, provenance
rgb.mp4              the video
overlay.mp4          annotated: mask + severity + clocks + window bar
timelines.npz        active[T], observable[T], occluded[T], severity_t[T]
violation_mask.npz   bool [T,H,W]  <- the primary annotation: where AND when
reference_mask.npz   bool [T,H,W]  where the culprit SHOULD be (valid twin); on both clips
causal_mask.npz      uint8 [T,H,W] 0=none, 1=culprit, 2+=participants
severity_map.npz     f16  [T,H,W]  how badly, localised in space and time
residuals.npz        r (physical units), z (vs noise floor), s (bounded [0,1])
divergence_map.npz   f16  [T,H,W]  NOT the violation region -- see below
seg.npz  depth.npz  flow_fwd.npz  flow_bwd.npz  normals.npz  object_coords.npz
grids.npz            masks + severity reduced to the latent token grid
traj.npz             the seam file: poses, velocities, contacts
```

Field reference: [docs/schema.md](docs/schema.md).

**Three things to know before training on any of it:**

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space
  and it diverges *everywhere* downstream of the event. Use `violation_mask` and
  `severity_map`.
- **`violation_mask` is gated on visibility, not just on the window.** It answers "where can
  this be seen", so it is empty on frames where the violation is active but the culprit is
  hidden — or where the intervention has not yet moved a pixel. `timelines.active` is the
  ground-truth timeline; the gap between them is the observability lag, and it is the point.
- **Check `provenance.prefix_identical_verified`** rather than assuming twins are
  pixel-aligned.

---

## 6. How it is organised

Four levels — `python -m physviol.cli taxonomy` prints the live version:

```
DOMAIN     7   which physical law is at stake      identity, kinematics, contact,
                                                   dynamics, equilibrium, optical, global
FAMILY    17   the specific way it breaks          solidity, fission, newton3_reaction, ...
SCENARIO  14   the staged scene                    ball_drop, occluder_pass, granular_pour, ...
               (13 built; clutter_toss is deferred)
INSTANCE       scenario x family x seed x severity -> one valid/invalid pair
```

| scenario | what it stages | why it is in the set |
|---|---|---|
| `ball_drop` | sphere or cube falls and bounces | simplest scenario with a real contact instant |
| `ball_collision` | two spheres roll together | two bodies that both *ought* to respond |
| `projectile_toss` | ballistic arc, no contact | the clean control: `t_event == t_observable` |
| `spin_toss` | cube tumbling in free flight | a sphere cannot show rotation |
| `occluder_pass` | body passes behind a screen | the only source of observability lag |
| `ramp_slide` | block slides down an incline | sustained contact + friction |
| `rolling_ramp` | cube tumbles off a raised ramp | contact, then a short free flight |
| `stack_topple` | marginally stable stack | *surprising but lawful* — the control for false positives |
| `pyramid_impact` | cube dropped on a sphere pyramid | multi-body contact chain |
| `pendulum_swing` | bob on a rigid rod | constrained periodic motion (scripted; Kubric has no joints) |
| `resting_table` | bodies at rest on a table | static equilibrium; any motion is the violation |
| `shadow_track` | object translating under a key light | the only violation whose mask is not on the object |
| `granular_pour` | 40 grains falling into an open box | **granular, never "fluid"** — see below |

Two orthogonal augmentation axes:

- **severity** — `weak` / `medium` / `strong`, set by the *intervention magnitude*, which is
  exact by construction. Named for how hard the law is bent, not for how hard the clip is to
  classify — those are different things, and conflating them is how difficulty splits go bad.
- **complexity** — `L0`..`L4`, mirroring the MOVi ladder. **L0 (solid background + sun) and
  L1 (photographic HDRI environment + dome ground) are built; L1 is the default.** L2–L4 add
  GSO objects, distractors and camera motion, and **raise if requested** rather than silently
  degrading. GSO (1033 objects) and HDRI Haven (509 environments) are verified reachable from
  the pinned container, so those levels need scenario code only.

**Why 19 files and not 238.** Scenarios and injectors are orthogonal and compose through the
trajectory seam — an injector edits `traj.npz`, per-body poses and velocities, which knows
nothing about the scene that produced it. So the project needs 13 scenario files + 6 injector
files (one per domain) + one shared geometry helper, not one file per scenario × family pair.
`antigravity` written once runs on every airborne scenario; adding a scenario makes every
compatible family available in it for free.

The corollary is that injectors must never branch on a scenario's *name*. They branch on
**state** — is this body moving, is it in contact, is it hidden — or on a **constraint the
scenario declares** in `spec.notes`. `newton1_inertia` halts a sliding block and shoves a
resting mug from one code path; `angular_momentum` asks the scenario to re-solve its own
pendulum arc rather than knowing what a pendulum is.

### No fluid at v0, and why

Tested rather than assumed: Blender 2.93.4 in the pinned image ships Mantaflow, but headless
scripted baking fails (`NameError: liquid_save_data_N` → `Manta::Error`), Kubric exposes no
fluid object, and a liquid's per-frame mesh state does not fit a pose-based trajectory seam.
`granular_pour` is the v0 stand-in — a few dozen rigid grains that stream, pile and break up —
and it is labelled `physics_medium: "granular"`. `physviol validate` rejects any clip
claiming `"fluid"`. Real fluid and cloth are **Phase 3**, behind a newer Blender.

---

## 7. Tiers

| | Tier D (debug) | Tier A (`physviol_v0`) | Tier B (`physviol_v1`) |
|---|---|---|---|
| resolution | 128² | 256² | 512² |
| frames @ fps | 13 @ 12 | 25 @ 12 | 97 @ 24 |
| latent grid | 4×8×8 | 7×16×16 | 25×16×16 |
| render cost | ~0.44 s/frame | ~1.75 s/frame | ~7.16 s/frame |
| published | never | yes | yes |

Frame counts are all `4k+1` so the token-grid reduction aligns exactly with a video-DiT VAE's
4× temporal binning. **Tier D is the default** — debug there.

**Rendering is CPU and stays that way.** Frame time fits `T = 1.29 + 0.0074·spp` at 256², so
only ~26% is sampling — the only part OptiX would accelerate, capping a GPU build at ~1.37×
by Amdahl. Clip-level parallelism measures **1.92×** at width 4, for free. Details in
[docs/PLAN.md](docs/PLAN.md) Part 0.

---

## 8. Repo layout

```
docs/PLAN.md          the design document -- start here
docs/schema.md        meta.json field reference
docs/prior_art.md     IntPhys 2 x LikePhys x PhysViol coverage matrix
environment.yml       host conda env
docker/               kubric.sh wrapper + pinned image digest
scripts/fetch_refs.sh pinned read-only Kubric checkout -> refs/
physviol/
  taxonomy.py         Part 2 as data: domains, families, scenarios, compatibility
  scenarios/          seeded scene samplers (declarative SceneSpec, no Kubric import)
                      _common.py ground/lights/ramps/understudies; _hdri.py environments
  sim/trajectory.py   THE SEAM -- container writes it, host reads it
  injectors/          trajectory-level interventions, one file per domain
                      _geom.py the shared vocabulary they ask questions in
  residuals/laws.py   physical-law residuals, one per family
  render/worker.py    container-side: simulate + inject + render every variant
  annotate/           windows, masks, severity, grids, meta -> the released layout
  viz/                overlay.mp4, grid.mp4, coverage.mp4, one shared encoder
  schema/validate.py  cross-checks
  cli.py
tests/                prefix identity, mask union, windows, grids, taxonomy,
                      mockroll.py + test_all_cells.py (all 48 cells, no docker)
out/                  all generated output (gitignored)
```

## 9. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). PDFs of the first two live in
`context/` (gitignored — see [context/README.md](context/README.md)).
