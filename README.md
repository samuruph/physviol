# PhysViol

**A spatio-temporally annotated physics-violation video dataset.** Every clip that breaks a
physical law ships with *where in the frame*, *exactly when*, *for how long*, and *how badly*
— all derived from the simulator, not from human annotation.

> **Status: Phase 0 built.** `ball_drop` × `solidity` runs end to end at all three severity
> levels, with prefix identity verified and 21 tests green. Design doc: [docs/PLAN.md](docs/PLAN.md).

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

## 2. Generate some clips

```bash
conda activate physviol

# One cell. Defaults to a single `strong` variant -- while checking coverage,
# breadth is what you want to look at, not three strengths of the same thing.
python -m physviol.cli generate --debug -n 1 \
    --scenario occluder_pass --family permanence --seed 777

# the whole severity ladder (weak/medium/strong) in ONE container run
python -m physviol.cli generate --debug -n 1 --severity all \
    --scenario ball_drop --family solidity

# uniform violation duration, so families are comparable side by side
python -m physviol.cli generate --debug -n 1 --window 4 \
    --scenario ball_drop --family antigravity

# every runnable cell, plus a comparison grid per pair
#   args: complexity tier seed severity window
bash scripts/generate_sample.sh L1 D 777 strong 4

# plain solid background (no network, faster)
python -m physviol.cli generate --debug -n 1 --complexity L0

# the publishable tier (256px, 25 frames)
python -m physviol.cli generate --tier A -n 1 --seed 91731
```

Implemented cells right now — any scenario × any compatible family, no
per-combination code:

| scenario | families available |
|---|---|
| `ball_drop` | `solidity`, `antigravity`, `continuity`, `permanence` |
| `projectile_toss` | `antigravity`, `continuity` |
| `occluder_pass` | `permanence`, `continuity`, `solidity` |

`occluder_pass` is the one that produces a **non-zero observability lag**: the violation
fires while the actor is hidden behind the screen, so `t_observable > t_event`.

Output lands in `out/release/clips/<release>/<scenario>/<seed>/{valid,invalid_<family>_a}/`
(gitignored, so it never enters version control).

**Look at the result** — `overlay.mp4` is a five-panel visualiser: RGB, RGB+mask, severity
heatmap with a live 0..1 scale, causal mask, and the divergence map (labelled as *not*
ground truth). Plus a red dot whenever the violation is active on that frame, and a timeline
showing the violation window (red), observable window (amber), the three clocks and a
playhead:

```bash
xdg-open out/release/clips/physviol_v0/occluder_pass/0777/invalid_permanence_a/overlay.mp4
```

---

## 3. The other commands

```bash
# Print the taxonomy: 7 domains, 16 families, 13 scenarios, 40 build cells
python -m physviol.cli taxonomy
python -m physviol.cli taxonomy -v          # + every (scenario, family) cell

# Re-annotate an existing worker output without re-rendering
python -m physviol.cli annotate out/work/ball_drop/4200 --outdir out/release

# Rebuild just the overlay video for one clip
python -m physviol.cli overlay out/release/clips/.../invalid_solidity_medium

# Comparison grid: the real clip beside every severity, annotated
python -m physviol.cli grid out/release/clips/physviol_v0/ball_drop/0777 \
    --family solidity

# Schema + cross-checks over a whole release
python -m physviol.cli validate out/release

# Tests
python -m pytest tests/ -q
```

### Running a worker in the container directly

```bash
# throughput probe (no assets downloaded, pure render cost)
bash docker/kubric.sh physviol/render/worker_smoke.py --resolution 256 --frames 8

# the real worker: simulate + inject + render both twins
bash docker/kubric.sh physviol/render/worker.py \
    --scenario ball_drop --seed 91731 --tier D \
    --family solidity --severity medium --outdir out/work
```

`docker/kubric.sh` mounts the repo at `/kubric`, runs as your uid so output is not
root-owned, and prefers the pinned digest over `:latest`.

---

### Two videos per result

- **`overlay.mp4`** (one per invalid clip) — five panels for *one* variant: RGB, +mask,
  severity, causal mask, divergence.
- **`grid_<family>.mp4`** (one per scenario/seed/family) — the **valid clip beside every
  severity bin**, three rows deep (RGB / +mask / severity), each column with its own
  magnitude, red ACTIVE dot and window bar. This is the one for checking that the severity
  ladder behaves.

### Reading the overlay

| element | meaning |
|---|---|
| **red dot**, top right | the violation is active on *this* frame |
| **RGB** | the clip as released |
| **MASK** | red = `violation_mask` (the annotation); **green outline** = `reference_mask`, where the object *should* be per the valid twin |
| **SEVERITY MAP** | the residual painted into the culprit, with a live 0..1 scale |
| **CAUSAL MASK** | red = primary culprit, blue = other participants |
| **DIVERGENCE** | `\|valid - invalid\|` -- shipped for analysis, **not** ground truth |
| red bar, timeline | `violation_windows` -- when the law is actually broken |
| amber bar, timeline | `observable_windows` -- when there is visual evidence |
| `t_event` / `t_obs` / `t_end` | the three clocks; they merge into one label when they coincide |

If `t_obs > t_event` the violation happened while the culprit was hidden -- that gap is the
occlusion lag, and `occluder_pass` is the scenario built to produce it.

## 4. What a clip contains

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

**Two warnings before training on anything:**

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space
  and it diverges *everywhere* downstream of the event. Use `violation_mask` and
  `severity_map`.
- **Check `provenance.prefix_identical_verified`** rather than assuming twins are
  pixel-aligned.

---

## 5. How it is organised

Four levels — `python -m physviol.cli taxonomy` prints the live version:

```
DOMAIN     7   which physical law is at stake      identity, kinematics, contact,
                                                   dynamics, equilibrium, optical, global
FAMILY    16   the specific way it breaks          solidity, non_parabolic, newton3_reaction, ...
SCENARIO  13   the staged scene                    ball_drop, occluder_pass, granular_pour, ...
               (3 built: ball_drop, projectile_toss, occluder_pass)
INSTANCE       scenario x family x seed x severity -> one valid/invalid pair
```

Two orthogonal augmentation axes:

- **severity** — `weak` / `medium` / `strong`, set by the *intervention magnitude*, which is
  exact by construction. Named for how hard the law is bent, not for how hard the clip is to
  classify -- those are different things and conflating them is how difficulty splits go bad.
- **complexity** — `L0`..`L4`, mirroring the MOVi ladder. **L0 (solid background) and L1
  (photographic HDRI environment + dome ground) are built; L1 is the default.** L2-L4 add
  GSO objects, distractors and camera motion, and **raise if requested** rather than
  silently degrading. GSO (1033 objects) and HDRI Haven (509 environments) are verified
  reachable from the pinned container, so those levels need scenario code only.

**All three severity bins are produced in one container run.** The valid rollout, scene build
and HDRI load are shared across the variants, which matters because an L1 render costs ~4.6x
an L0 one — see the timing table below.

**Why 20 files and not 208.** Scenarios and injectors compose through the trajectory seam —
an injector edits `traj.npz`, which knows nothing about the scene that produced it. So the
project needs 13 scenario files + 7 injector files (one per domain), not one per
scenario × family pair. `antigravity` written once runs on every airborne scenario.

---

## 6. Tiers

| | Tier D (debug) | Tier A (`physviol_v0`) | Tier B (`physviol_v1`) |
|---|---|---|---|
| resolution | 128² | 256² | 512² |
| frames @ fps | 13 @ 12 | 25 @ 12 | 97 @ 24 |
| latent grid | 4×8×8 | 7×16×16 | 25×16×16 |
| render cost | ~0.44 s/frame | ~1.75 s/frame | ~7.16 s/frame |
| published | never | yes | yes |

Frame counts are all `4k+1` so the token-grid reduction aligns exactly with a video-DiT
VAE's 4× temporal binning. **Tier D is the default** — debug there, since a bug found at
Tier D is fixed for all three.

**Rendering is CPU and stays that way.** Frame time fits `T = 1.29 + 0.0074·spp` at 256², so
only ~26% is sampling — the only part OptiX would accelerate, capping a GPU build at ~1.37×
by Amdahl. Clip-level parallelism measures **1.92×** at width 4, for free. Details in
[docs/PLAN.md](docs/PLAN.md) Part 0.

---

## 7. Repo layout

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
                      _common.py ground/lights/colour; _hdri.py environment ids
  sim/trajectory.py   THE SEAM -- container writes it, host reads it
  injectors/          trajectory-level interventions, one file per domain
  residuals/laws.py   physical-law residuals
  render/worker.py    container-side: simulate + inject + render both twins
  annotate/           windows, masks, severity, grids, meta -> the released layout
  viz/                overlay.mp4 and the shared video encoder
  schema/validate.py  cross-checks
  cli.py
tests/                prefix identity, mask union, windows, grids, taxonomy
out/                  all generated output (gitignored)
```

## 8. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). PDFs of the first two live in
`context/` (gitignored — see [context/README.md](context/README.md)).
