# PhysViol — a spatio-temporally annotated physics-violation dataset

> The design document for this repository. Field-level schema lives in
> [schema.md](schema.md); the prior-art coverage matrix lives in
> [prior_art.md](prior_art.md).

## Context

Every existing intuitive-physics video benchmark labels **whole clips**. IntPhys 2 and
LikePhys both ship valid/invalid pairs with one binary label per video; PhyCheck and
PhyGround add fine-grained *questions*, but their labels are human-written QA over
real/AIGC video, not simulator-derived geometry. Nothing provides **where in the frame**,
**exactly when**, **for how long**, or **how badly** the law broke.

That gap blocks concrete work, and we hit it directly. In a prior probing project of ours,
per-timestep violation probes could only be scored against a *pseudo-onset* recovered by
diffing a clip against its "twin" — and that labeller fails structurally, because LikePhys
valid/invalid clips are *separate renders*: only **3%** of pairs share a near-zero baseline.
An object-centric fallback covered **46%** of violated clips, and just **10 of 60** on the
`cloth_drape` scenario. Spatial localization there was not merely unsupervised — it was
**unevaluable**.

Generating the data inverts the problem: we do not *recover* a violation, we *inject* it,
so onset, duration, extent, culprit object and magnitude are known by construction.

**Scope.** PhysViol is a **general-purpose dataset in its own right**, usable the way
IntPhys 2 and LikePhys are used today by anyone. It has no privileged consumer; the
published on-disk format is the entire contract (Part 6).

**Decisions locked:**

| | |
|---|---|
| Location | this repository, standalone |
| Stack | **Kubric** (Blender + PyBullet, docker) for v0, with an explicit trajectory seam so **MuJoCo/MJX** can replace PyBullet later |
| Clip spec | **staged ladder** — see the two tiers below. Start at MOVi defaults, move up once the pipeline is proven. |
| Rendering | **CPU Cycles, upstream image, unchanged.** Measured: a GPU/OptiX image caps out at ~1.37× (Part 0). The real lever is clip-level parallelism. |
| Severity | **both** — the injected intervention magnitude *and* a measured per-frame physical-law residual field |
| Taxonomy | four levels: **domain → family → scenario → instance** (Part 2) |
| Annotations | clip labels, violation **windows**, per-frame timelines, spatiotemporal masks, severity fields, token grids (Part 3) |
| Environments | pinned docker image for sim+render; `physviol` conda env on the host for everything else |
| Real→sim | reserve the schema fields now, build in Phase 4 |

### The resolution ladder

The original target spec was 512²/24 fps/~96 frames from day one. That is the *destination*,
not the starting point: it is ~16× the render cost of MOVi's defaults, and paying it before
the annotation pipeline is proven means every bug costs 16× to re-render.

| | **Tier D — debug** | **Tier A — `physviol_v0`** | **Tier B — `physviol_v1`** |
|---|---|---|---|
| resolution | **128 × 128** | **256 × 256** (MOVi default) | **512 × 512** |
| frame rate | 12 fps | **12 fps** (MOVi default) | **24 fps** |
| frames | **13** (≈1.1 s) | **25** (≈2.1 s) | **97** (≈4.0 s) |
| samples/pixel | 16 | 64 | 64 |
| latent grid | 4 × 8 × 8 | 7 × 16 × 16 | 25 × 16 × 16 |
| measured cost | **~5.7 s/clip**, ~11 s/pair | **~44 s/clip**, ~7 h for 600 clips | ~11.5 min/clip, ~114 h for 600 clips |
| status | never released; the iteration loop | build now; **a real, shippable release** | supersedes Tier A once proven |

**Tier D exists so that "generate a few examples and look at them" is an 11-second
operation, not an 11-minute one.** It is the default for every `physviol generate` call
without an explicit `--tier`, and it is what Phase 0 and all injector/annotation debugging
run on. It is never published: 128² is too small for a real benchmark and 4×8×8 is a
degenerate token grid. Its job is to make bugs cheap.

```bash
physviol generate --debug -n 4          # 4 pairs, Tier D, ~45 s total
physviol viz overlay <clip> --open      # look at them before believing anything
```

The three tiers differ only in a config block — same scenarios, same injectors, same
annotation code — so a bug found at Tier D is a bug fixed for all three.

Tier A is not a throwaway. It gets the full schema, splits, validation and release
packaging, because that work is needed regardless and doing it at 1/16th the render cost is
strictly better. Tier B re-runs the *same generators* at the larger spec — the only changes
are three numbers in a config.

**Why 25 frames and not MOVi's 24.** The token-grid reduction (Part 3) needs a frame count of
the form `4k + 1` to align exactly with a video-DiT VAE's 4× temporal binning. 25 → 7 latent
frames exactly; 24 → 6.75, which forces a ragged edge. One extra frame buys an exact grid.
The rule generalises: **every tier's grid timeline is `4k + 1` frames.** Tier B is 97 frames rather than
96 for the same reason — 97 → 25 latent frames exactly, so no resampling is needed and the
grid is computed on the master itself.

**Can everything be scripted? Yes, entirely.** Blender runs headless (`blender -b -P`),
Kubric drives it from Python, PyBullet is headless by default, and Poly Haven / GSO assets
download through manifests. No GUI is required at any point. Diversity comes from seeded
procedural sampling, not from hand-authoring scenes.

---

## Part 0 — Environments, and how Kubric is actually used

### Two environments, and the seam between them

This project has **two** Python environments and they never mix. The trajectory seam
(Part 4) is exactly the boundary.

| | **container** (`kubricdockerhub/kubruntu`, digest-pinned) | **host** (`physviol` conda env) |
|---|---|---|
| Python | 3.9.5 (fixed by the image) | 3.11 |
| holds | Kubric 2022.4.1, Blender 2.93.4, PyBullet | numpy, scipy, opencv, imageio, jsonschema, pypdf, pytest |
| does | scene sampling, simulation, rendering | residuals, annotation, masks, severity, grids, validation, viz, evaluation |
| writes | `traj.npz` + raw Kubric passes | everything in Part 3 |

```bash
conda env create -f environment.yml     # once
conda activate physviol
```

**Kubric, Blender and PyBullet are never installed on the host.** They live in the image.
This is deliberate: it keeps the render path byte-reproducible behind a pinned digest, and it
means the host env can be upgraded freely without touching render determinism.

### Your script, their container

The published docker image already contains a complete Kubric installation and its
dependencies, Blender included. Mount this repo at `/kubric` and run your own worker file.

```bash
docker pull kubricdockerhub/kubruntu          # once; digest is pinned in docker/IMAGE_DIGEST

docker run --rm --interactive \
  --user $(id -u):$(id -g) \
  --volume "$PWD:/kubric" \
  kubricdockerhub/kubruntu \
  python3 physviol/render/worker_smoke.py
```

`--user` makes output files land owned by you rather than root; `--volume` is both how the
worker script gets *in* and how rendered frames get *out*. That is wrapped as
[`docker/kubric.sh`](../docker/kubric.sh), which prefers the pinned digest over `:latest`.

A worker is plain Python — `kb.Scene(...)`, add objects, `kb.simulator.PyBullet` to run
dynamics, `kb.renderer.Blender` to render, then `kb.write_image_dict(...)`. Fully headless,
fully scriptable, no GUI at any point.

### The reference checkout

**Do clone Kubric — as pinned, read-only, gitignored reference material.** The image gives
you a *runnable* Kubric; the clone gives you a *readable* one.

```bash
bash scripts/fetch_refs.sh     # -> refs/kubric at a pinned sha, gitignored
```

Start with `refs/kubric/challenges/movi/movi_def_worker.py` (Part 0.5), then
`refs/kubric/examples/keyframing.py` (the replay mechanism the seam depends on) and
`refs/kubric/kubric/renderer/blender.py` (what the exporters actually emit).

### Measured environment (recorded, not assumed)

| | |
|---|---|
| Host | 8 CPU cores, NVIDIA L40S 46 GB, 344 GB free |
| Docker | 25.0.14, `nvidia` runtime present |
| Image | `kubricdockerhub/kubruntu`, digest in `docker/IMAGE_DIGEST`, **built 2022** |
| Inside the image | Blender **2.93.4**, Python **3.9.5**, kubric **2022.4.1**, PyBullet 2020.10.06 |
| Reference clone | `google-research/kubric` @ `61f2422c`, carrying 2026 copyright headers |

**The image and the clone are ~4 years apart. This was checked rather than assumed:** every
API symbol the 2026 `movi_def_worker.py` uses — `kb.setup`, `AssetSource.from_manifest`,
`compute_visibility`, `adjust_segmentation_idxs`, `write_image_dict`, `process_collisions`,
`move_until_no_overlap`, `sample_point_in_half_sphere_shell`, `post_processing.compute_bboxes`
— **is present in the image's Kubric.** The clone is safe as a template source. The rule
stands regardless: **the installed copy in the container is the API authority.**

### Measured throughput, and why we are staying on CPU

Measured with [`physviol/render/worker_smoke.py`](../physviol/render/worker_smoke.py) — a
two-object scene, all seven passes exported, primitives only so nothing downloads.

**Cost scales with pixels.** 256² is 1.75 s/frame; 512² is 7.16 s/frame — almost exactly 4×,
as expected. Cost is also **linear in frame count** (4-frame and 16-frame runs agree within
2%), so per-clip extrapolation is sound.

**Cost is mostly *not* sampling.** Four points at 256²:

| samples/pixel | 8 | 32 | 64 | 128 |
|---|---|---|---|---|
| s / frame | 1.35 | 1.53 | 1.75 | 2.24 |

These fit `T = 1.29 + 0.0074 · spp` almost exactly. At MOVi's 64 spp that is **1.29 s fixed
(74%)** and **0.47 s sampling (26%)**. The fixed part is scene sync, the auxiliary passes,
denoising and file I/O — none of which scale with sample count, and all of which we need.

**Therefore GPU rendering is not worth building.** OptiX accelerates the *sampling* fraction.
By Amdahl, with sampling at 26% of frame time, **even an infinitely fast GPU yields at most
~1.37×.** Against that: the upstream image is Blender 2.93 on CPU by design, and a GPU image
means upgrading Blender under a Kubric release from 2022 that is pinned to 2.93's `bpy` API —
a real risk of breaking the renderer integration, for ~1.4×. **We stay on the upstream CPU
image.** Revisit only if Tier B measurements change the arithmetic, and then as a timeboxed
spike with a measured exit criterion, not as scheduled work.

**The real lever is clip-level parallelism, and it beats the GPU ceiling for free.** All
600 clips are independent, and 74% of per-frame cost is in phases that do not saturate 8
cores. Running N clips concurrently overlaps one clip's sync and I/O with another's sampling.
Measured at width 4 on this box: **4 clips serial 46 s → parallel 24 s, a 1.92× speedup** —
already more than the ~1.37× ceiling an infinitely fast GPU could offer, at zero engineering
risk. Phase 1 tunes the optimal N (likely 4-8) and the generator ships as a resumable,
per-clip-checkpointed queue.

**What this means for the schedule.** Tier A is ~44 s/clip → ~7 h for 600 clips serial,
**under 4 h at the measured 1.92×**. That is comfortable. Tier B is ~11.5 min/clip → ~114 h
serial, ~60 h parallel — which is when the queue stops being an optimisation and starts being
the plan.

**GPU is still reserved — for physics, not pixels.** `sim_seconds` has measured **0.0** in
every run: PyBullet rigid-body dynamics is free at v0 scale. That changes in Phase 3 with
deformables and many-body scenes, which is exactly when a GPU physics backend (MuJoCo MJX)
drops in behind the seam. The seam exists so that swap costs nothing downstream.

---

## Part 0.5 — From MOVi to PhysViol

MOVi is the right skeleton and the wrong dataset. `movi_def_worker.py` is "toss N random
objects and export everything"; we need "stage a *specific* physical situation, break one
law at a known instant, and export what proves it". All claims below were read out of the
pinned checkout, not recalled.

**Taken directly from the MOVi worker:**

- asset-source plumbing — KuBasic / GSO / HDRI Haven from `gs://kubric-public/assets/*.json`
  manifests, plus the dome-as-background idiom that textures a KuBasic `dome` with an HDRI
- the `kb.Scene` → `kb.simulator.PyBullet` → `kb.renderer.Blender` spine, and
  `kb.setup(FLAGS) -> scene, rng, output_dir, scratch_dir` for a single seeded rng
- `animation, collisions = simulator.run(...)` — it returns **both** per-object keyframes
  **and** collision events, and `kb.process_collisions(...)` serializes them. That is already
  most of our `contacts[T]` seam field, for free.
- the export path: `rgba`, `segmentation`, `depth`, `forward_flow`, `backward_flow`,
  `normal`, `object_coordinates` via `kb.write_image_dict`, with `kb.compute_visibility` and
  `kb.adjust_segmentation_idxs` to renumber instances by visibility
- `kb.move_until_no_overlap(obj, simulator, spawn_region=..., rng=rng)` for valid placement
- the settling idiom — `simulator.run(frame_start=-100, frame_end=0)`, then zero velocities —
  which is exactly what our support/stack scenarios need
- the camera modes `fixed_random` / `linear_movement` / `linear_movement_linear_lookat`

**What we replace, and why each one is load-bearing:**

1. **Split simulate from render.** MOVi renders straight out of `simulator.run()`. We write
   `traj.npz` in between and render by *replaying* keyframes. `simulator.run()` works by
   writing keyframes onto scene objects, and `examples/keyframing.py` shows those same
   objects keyframed by hand via `obj.keyframe_insert("position", frame)` — replay is a
   supported path, not a hack. This is the seam, and the single structural edit to the
   template.
2. **Scenario library instead of random tossing** (Part 2), including **occluders** — without
   them `t_observable` always equals `t_event` and the occlusion-lag contribution disappears.
3. **Injectors.** MOVi has no notion of intervention. Ours operate on the trajectory, never
   on the sim rng, so the twin's seed and render path are untouched.
4. **Residuals and annotation** (Part 3). Entirely new.

**Two facts from MOVi that shaped our numbers:**

- MOVi-def defaults to 256², 24 frames, 12 fps. **Tier A adopts these defaults** (with 25
  frames for exact latent alignment), which is why Tier A costs what MOVi costs.
- MOVi's asset mix is not uniformly CC0. The schema promises a license string per asset;
  record it as each source is enabled, not at release.

---

## Part 1 — The design that makes this a contribution

Three ideas carry the whole dataset. Everything else is plumbing.

### 1.1 Clocks and windows, not a single onset

Existing benchmarks have no onset at all; the naive version of this dataset would have one
number. A violation is really an **interval** — often several — observed through a second,
lagging interval.

**Three clocks:**

- **`t_event`** — the frame the law breaks *in simulator state*. Known exactly: it is when
  the injector fires.
- **`t_observable`** — the first frame carrying *visual evidence*. If the violation happens
  behind an occluder these differ by a second or more. Computed exactly, not estimated:
  because valid and invalid twins go through a bit-identical render path with the same seed,
  the pixel difference before `t_event` is **exactly zero**, so `t_observable` is the first
  frame where the causal bodies' rendered footprint differs at all.
- **`t_consequence[k]`** — when each *downstream* body first diverges, each with its own
  observability frame.

**Two families of windows.** A single onset cannot describe a super-elastic ball that gains
energy on bounce 1, 2 and 3, nor an object that becomes visible, hides behind the occluder
again, and re-emerges. So both are stored as **lists of intervals**:

- **`violation_windows`** — `[[start, end], …]` in simulator time. The frames during which
  the intervention is actively breaking the law.
  - `instant` (teleport, vanish) → one window of length 1
  - `sustained` (collision pair disabled for N frames, per-body `αg`) → one window of length N
  - `repeated` (restitution `e>1` firing on each of three bounces) → **three windows**
- **`observable_windows`** — `[[start, end], …]`, the frames during which visual evidence is
  actually present. Derived from pixel evidence, so it lags, and it can be interrupted by
  re-occlusion. Not simply a shifted copy of `violation_windows`.

The two are not interchangeable, and the difference is what the spatial annotations are gated
on. `active` is ground truth about the *world*: the frames on which the law is being broken.
`observable` is about the *image*: the frames on which the two renders differ at all. A
super-elastic bounce is unlawful the instant the contact resolves, while the body is still in
exactly the same place in both twins — active, not yet observable. `violation_mask` and
`severity_map` answer "where can this be seen" and so are gated on `active AND observable`;
`timelines.active` keeps the unhedged truth. Severity accumulated while nothing is observable
is carried to the next observable frame inside the same window, so a violation whose residual
is a single invisible spike is not annotated as harmless.

Both are also rasterised to per-frame boolean timelines (`active[T]`, `observable[T]`) so a
consumer never has to expand intervals itself.

This yields metrics no benchmark currently has: **detection latency**
(`predicted_onset − t_observable`), **occlusion lag** (`t_observable − t_event`), and
**duration IoU** (predicted vs true active interval). It also lets us build a split where the
violation is *never* directly observable and only its consequences are — the hardest honest
test in the space.

### 1.2 Violation is a law residual, never "difference from the valid rollout"

This is the trap to design around. After `t_event` a valid/invalid pair diverges
*everywhere* downstream — shadows, contact chains, secondary collisions — and in chaotic
scenes it diverges enormously. **The pixels that differ are not the pixels where the
violation is.** Worse, a merely different-but-valid rollout also diverges, so divergence
cannot define violation at all.

So severity is computed **per body per frame from simulator state**, as a dimensionless
residual against the law that was broken:

| law | residual `r(b,t)` |
|---|---|
| linear momentum | `‖m·Δv − (ΣF_contact + m·g)·Δt‖ / (m·‖g‖·Δt)` |
| angular momentum | same form with torques and `I·Δω` |
| energy at contact | `max(0, E_after − E_before) / E_before` |
| solidity | max signed penetration depth (m), normalised by body radius |
| free fall / support | `‖a − g‖ / ‖g‖` for an unsupported body |
| trajectory shape | RMS deviation (m) of flight path from the fitted `g`-parabola |
| identity | Δvolume ratio, ΔLab, or mass discontinuity |

Residuals are computed for **valid clips too**. They are not exactly zero there — solver
error is real — so a **noise-floor calibration pass** over valid clips is a required
deliverable (Part 3, step 2).

The pixel-space `|valid − invalid|` map is still exported, as `divergence_map`, with an
explicit schema note that it is **not** the violation region — so nobody trains on it by
accident.

### 1.3 Severity is a field, not a flag

**There are two different "magnitudes", and keeping them separate is the point.**

| | **intervention magnitude** | **measured residual** |
|---|---|---|
| what it is | the knob we turned | the effect that knob had |
| known | **before** simulating — we chose it | **after** simulating, from `traj.npz` |
| exact? | yes, by construction | up to the noise floor |
| shape | one scalar per clip | per body, per frame → `[T,H,W]` field |
| lives in | `meta.json:violation.intervention.magnitude` | `residuals.npz`, `severity_map.npz`, `severity_t` |
| used for | building weak/medium/strong splits | training targets, difficulty analysis |

They are **not** the same number and the mapping between them is not the identity — a large
teleport that happens behind an occluder produces a large intervention magnitude and a small
*observable* consequence. That gap is itself a research object, which is why both ship.

There is no "severity mask". There are two distinct arrays: **`violation_mask`** (binary —
*where*) and **`severity_map`** (continuous — *how badly*). Part 3.3 and 3.4 build them, and
§3.4 works a full numeric example end to end.

---

## Part 2 — Taxonomy: how the dataset is organised

Four levels. Every clip is uniquely addressed by the bottom one, and every level above is a
grouping key that a consumer can aggregate or split on.

```
Level 1  DOMAIN     ── which physical law is at stake        (7 domains)
Level 2  FAMILY     ── the specific way it breaks            (17 families)
Level 3  SCENARIO   ── the staged scene it happens in        (14 scenarios)
Level 4  INSTANCE   ── scenario × family × seed × severity   (the clip pair)
```

The top level is organised **by physical law**, not by cognitive principle, because severity
is a law residual (Part 1.2) — so the domain determines the residual, which determines the
severity unit. IntPhys 2 principles and LikePhys domains are retained as *cross-reference
fields* (`intphys2_category`, `likephys_domain`) for head-to-head comparability, mapped in
[prior_art.md](prior_art.md).

### Level 1 + 2 — Domains and families

| # | domain | the question it tests | families | residual |
|---|---|---|---|---|
| 1 | **`identity`** | does the object persist and stay itself? | `permanence`, `immutability`, `fission` | mass / volume / object-count discontinuity |
| 2 | **`kinematics`** | is unsupported motion consistent with `g`? | `continuity`, `non_parabolic`, `antigravity`, `newton1_inertia` | `‖a−g‖/‖g‖`, parabola RMS, jump distance |
| 3 | **`contact`** | do bodies interact legally when they touch? | `solidity`, `superelastic`, `newton3_reaction` | penetration depth, energy gain ratio, momentum imbalance |
| 4 | **`dynamics`** | do forces and masses behave? | `phantom_impulse`, `newton2_mass`, `angular_momentum` | `J/(m·v_typ)`, effective-mass ratio, `L` defect |
| 5 | **`equilibrium`** | do resting and supported bodies behave? | `support`, `friction` | clearance above the supporting surface, effective µ vs declared µ |
| 6 | **`optical`** | is light consistent with geometry? | `shadow` | shadow–caster geometric mismatch |
| 7 | **`global`** | are the scene's constants physical? | `global_gravity` | `\|α − 1\|` on scene gravity |

**The 17 families in full:**

| family | domain | injection | severity unit | kind |
|---|---|---|---|---|
| `permanence` | identity | remove / duplicate body, ideally while occluded | occlusion lag (frames), Δmass | instant |
| `immutability` | identity | the body grows or shrinks, ideally behind an occluder | volume ratio | sustained |
| `fission` | identity | one body becomes two, each half the volume | object-count ratio | sustained |
| `continuity` | kinematics | discontinuous position set | jump distance (m) | instant |
| `non_parabolic` | kinematics | replace free-flight arc with a non-`g` curve | RMS deviation from fitted parabola (m) | sustained |
| `antigravity` | kinematics | per-body gravity scale `g → αg`, `α ∈ [−1, 0.5]` | `‖a−g‖/‖g‖` | sustained |
| `newton1_inertia` | kinematics | resting body accelerates; sliding body stops on frictionless surface | `‖Δv‖/(‖g‖·Δt)` | instant / sustained |
| `solidity` | contact | disable collision pair for N frames | penetration depth (m) | sustained |
| `superelastic` | contact | restitution `e > 1` | energy gain ratio | **repeated** |
| `newton3_reaction` | contact | in a collision only one body responds | momentum imbalance | instant |
| `phantom_impulse` | dynamics | impulse with no contact | `J/(m·v_typ)` | instant |
| `newton2_mass` | dynamics | identical-looking bodies respond differently to equal impulse | effective-mass ratio | instant |
| `angular_momentum` | dynamics | spin reverses; torque with no contact | `L` defect | instant / sustained |
| `support` | equilibrium | unsupported body hovers; unstable stack does not topple | CoM-to-polygon distance (m) | sustained |
| `friction` | equilibrium | body accelerates against its motion, or slides forever | effective µ vs declared µ | sustained |
| `shadow` | optical | shadow detaches, inverts, vanishes, or mismatches its caster | shadow–caster offset (px or m) | sustained |
| `global_gravity` | global | whole scene runs at `αg` — internally consistent | `\|α − 1\|` | sustained (whole clip) |

Three families deserve their reason-for-existing stated, because they are what makes the
"where" annotation non-trivial:

- **`non_parabolic`** is the **clean control**. No occluder, so `t_event ≈ t_observable`,
  which is the baseline against which every occluded family's latency is measured. It is also
  the most smoothly dialable severity axis we have.
- **`shadow`** is the **shortcut probe**. The annotated region is **not on the object**. A
  model that scores well by running object detection and calling it localisation fails here.
- **`global_gravity`** is the **no-culprit case**. The mask is the entire frame. It tests
  whether a model can flag a violation when "where" is legitimately "everywhere".

### On fluid and deformables — measured, and deferred

LikePhys covers Fluid Mechanics (*Droplet Fall*, *Faucet Flow*, *River Flow*) and Continuum
Mechanics (*Cloth Drape*, *Cloth Waving*). We want both eventually. Here is exactly where
they stand, tested in the pinned image rather than assumed:

| finding | result |
|---|---|
| Blender 2.93.4 ships **Mantaflow** (FLIP liquid + gas) | ✅ `FluidModifier`, `FluidDomainSettings`, `FluidFlowSettings`, `bpy.ops.fluid.bake_*` all present |
| Headless scripted bake works | ❌ **fails.** `bpy.ops.fluid.bake_data()` → `NameError: liquid_save_data_N`, then a `Manta::Error` crash. The REPLAY-cache + `frame_set()` workaround fails identically. |
| Kubric exposes fluid / soft / particle objects | ❌ none — its asset types are rigid only (`Cube`, `Sphere`, `FileBasedObject`, …) |
| PyBullet has a fluid solver | ❌ only `loadSoftBody` / `createSoftBodyAnchor`; no SPH/FLIP |

The bake failure is the known Blender-in-background limitation: `bake_data` is a modal
operator that expects the job system, which `blender -b` does not provide. It is likely
solvable — by driving Mantaflow through its own Python bindings, or by a newer Blender — but
that is a project, not a config flag.

**Three deeper reasons it is Phase 3 and not v0**, independent of the bake bug:

1. **It breaks the seam.** `traj.npz` is per-body `pos[T,3]`, `quat[T,4]`. A liquid's state is
   a mesh or particle set per frame; it cannot be expressed as rigid poses, so fluid needs a
   *second* seam format (a mesh cache) and its own replay path.
2. **It breaks the residual.** Severity is a per-body law residual. A fluid needs field-level
   laws — volume/mass conservation, divergence — which is a different module, not a new row
   in the table in §1.2.
3. **Cost is unbounded at this stage.** Mantaflow bake time scales with domain resolution
   cubed and is unmeasured here precisely because the bake does not run. Committing v0 to it
   would put an unquantified number on the critical path.

**What we can do now instead: a granular proxy.** A stream of many small rigid spheres
reproduces pouring, splashing-ish spreading and stream break-up — using the rigid-body
machinery we already have, so it keeps the seam, keeps exact per-body residuals, and keeps
exact masks. It covers a real subset of LikePhys's fluid *violations* (mass created or
removed, antigravity flow, stream fragmenting into disconnected clumps) with none of the
solver risk. It ships as the `pour` scenario below and is **labelled honestly as
granular flow, not fluid** — the schema records `physics_medium: "granular"`, so nobody can
mistake it for an SPH benchmark.

True fluid and cloth arrive in **Phase 3**, alongside the MuJoCo/MJX backend swap, where a
mesh-cache seam and field-level residuals are being built anyway.

### Complexity — a second augmentation axis, measured

Severity asks *how badly is the law broken*; **complexity** asks *how hard is the scene to
parse*. They are orthogonal, and reporting accuracy across both is what separates "the model
understands physics" from "the model copes with clutter". The ladder mirrors MOVi's.

Asset availability was **tested against the pinned image, not assumed**:

| source | count | fetch | verdict |
|---|---|---|---|
| GSO objects | 1033 (930 train / 103 test) | ~0.9 s | ✅ downloads fine |
| HDRI Haven environments | 509 (458 / 51) | ~1.8 s | ✅ downloads fine |
| KuBasic primitives (incl. `dome`) | 15 | <1 s | ✅ |

So MOVi-C/D/E-style realism needs **no new image** — only scenario code.

| level | background | actors | distractors | camera | ~MOVi | status |
|---|---|---|---|---|---|---|
| **L0** | solid | primitives | 0 | static | MOVi-A | **built** |
| **L1** | HDRI dome | primitives | 0 | static | MOVi-B | **built** |
| L2 | HDRI dome | GSO | 6 | static | MOVi-C | scaffolded |
| L3 | HDRI dome | GSO | 12 | linear | MOVi-D/E | scaffolded |
| L4 | HDRI dome | GSO | 20 | linear + blur | MOVi-F | scaffolded |

Encoded in `physviol/scenarios/base.py:COMPLEXITY`, carried in `meta.json`, and **guarded**:
requesting an unimplemented level raises rather than silently producing an L0 scene labelled
as MOVi-C. **L1 is the default for generation** — a photographic environment costs one HDRI
fetch (~1.8 s) and makes the clips look real while keeping shapes simple enough that a
violation stays legible. L0 remains available and is faster, since it needs no network.

### Level 3 — Scenarios (the staged scenes)

A scenario is a seeded scene sampler: assets, camera, lighting, initial state, and a
guaranteed physical *event structure* (a flight, a collision, a rest). Thirteen for v0,
chosen so that between them they cover every family and every LikePhys rigid-body scenario,
plus a granular stand-in for the fluid domain.

| scenario | the scene | event structure it guarantees | occluder? | grounded in |
|---|---|---|---|---|
| `drop` | sphere falls to a floor and bounces | free fall → contact → rebound | optional | LikePhys *Ball Drop* |
| `collision` | two spheres roll toward each other | approach → collision → separation | no | LikePhys *Ball Collision* |
| `ramp_slide` | block slides down an incline | sustained contact + friction | no | LikePhys *Block Slide* |
| `toss` | body thrown on a ballistic arc | pure free flight, no contact | no | new — parabola control |
| `tumble` | cube tumbling in the air, thrown with heavy spin | free flight with visible rotation | no | new — a uniformly coloured sphere cannot show spin |
| `occluder_pass` | body travels behind a screen and re-emerges | occlusion interval of known length | **yes** | IntPhys 2 occlusion |
| `stack_topple` | stacked bodies, marginally stable | static equilibrium → topple | no | IntPhys 2 / LikePhys |
| `pyramid_impact` | cube dropped onto a sphere pyramid | multi-body contact chain | no | LikePhys *Pyramid Impact* |
| `pendulum_swing` | bob on a rigid rod | constrained periodic arc | optional | LikePhys *Pendulum* |
| `resting_table` | several bodies at rest on a surface | sustained static equilibrium | optional | IntPhys 2 permanence |
| `rolling_ramp` | cube tumbles down a raised ramp and off its lip | rolling contact, then a short free flight | no | new |
| `shadow_track` | object translates under a fixed light | a clean, trackable cast shadow | no | LikePhys *Moving Shadow* |
| `clutter_toss` | MOVi-style multi-object toss | dense collisions, heavy occlusion | implicit | MOVi baseline |
| `pour` | a loose column of grains falls into an open box (40 at Tier D, 96 above) | streaming flow, accumulation, break-up | no | LikePhys *Faucet Flow* (as **granular**, not fluid) |

### Which families can be injected into which scenarios

Not every combination is meaningful — `newton3_reaction` needs a collision, `shadow` needs a
clean cast shadow, `support` needs something at rest. This matrix is the generator's job
list; **`●` = build it in v0**, `○` = valid but deferred, blank = not meaningful.

| | drop | coll | ramp | toss | spin | occl | stack | pyr | pend | rest | roll | shad | clut | gran |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `permanence` | ● | ○ |   | ○ |   | ● | ○ |   | ○ | ● |   |   | ○ | ○ |
| `immutability` | ● | ○ |   | ● | ○ | ● |   |   |   | ● |   |   | ○ |   |
| `fission` | ● | ○ |   | ● | ○ | ● |   |   |   |   |   |   | ○ |   |
| `continuity` | ● | ○ | ○ | ● |   | ● |   |   | ○ | ○ | ○ |   | ○ | ○ |
| `non_parabolic` | ● |   |   | ● | ● | ○ |   |   |   |   |   |   |   |   |
| `antigravity` | ● |   | ○ | ● |   | ○ |   |   |   | ○ |   |   | ○ | ● |
| `newton1_inertia` |   | ○ | ● |   |   |   |   |   |   | ● | ● |   |   |   |
| `solidity` | ● | ● | ○ |   |   | ● | ○ | ● |   |   | ○ |   | ○ | ○ |
| `superelastic` | ● | ● |   |   |   |   |   | ● |   |   |   |   | ○ | ○ |
| `newton3_reaction` |   | ● |   |   |   |   |   | ● |   |   |   |   | ○ |   |
| `phantom_impulse` | ○ | ● | ○ | ● | ● | ○ | ○ |   | ○ | ● | ○ |   | ○ | ○ |
| `newton2_mass` | ○ | ● | ○ |   |   |   |   | ● |   |   |   |   |   |   |
| `angular_momentum` |   | ○ | ○ | ○ | ● |   |   |   | ● |   | ● |   |   |   |
| `support` |   |   |   |   |   |   | ● | ○ |   | ● |   |   |   |   |
| `friction` |   |   | ● |   |   |   |   |   |   | ○ | ● |   |   | ○ |
| `shadow` | ○ |   |   | ○ |   |   |   |   |   | ○ |   | ● |   |   |
| `global_gravity` | ● | ○ | ○ | ● |   | ○ | ○ | ○ | ○ | ○ | ○ |   | ○ | ● |

That is **48 `●` combinations** across 13 buildable scenarios (`clutter_toss` is deferred;
the authoritative list is `taxonomy.build_cells()`). At ~6 randomisations each — different
sizes, speeds, colours, camera, environment map and, where physically neutral, actor shape —
that is ~288 pairs / 576 clips, the Phase 2 target, arrived at from the matrix rather than
picked as a round number.
`pour` contributes the two cells (`antigravity`, `global_gravity`) that read most
like LikePhys's fluid violations while staying inside exact rigid-body physics.

### How much code this actually takes

A natural worry with 14 scenarios × 17 families is that it means 238 generators. It does
not. **Scenarios and injectors are orthogonal and compose through the seam**, because an
injector edits `traj.npz` — per-body poses and velocities — which knows nothing about which
scene produced it.

| | count | what each one is |
|---|---|---|
| scenario files | **13** | a staged scene: assets, camera, initial state (~60 lines; generic parts live in `scenarios/_common.py`) |
| injector files | **6** | one per *domain*, holding all its families, plus a shared `_geom.py` |
| per-combination files | **0** | the compatibility matrix selects the 48 meaningful cells |

**19 files, not 238.** `antigravity` written once runs on `drop`, `toss` and
`occluder_pass` unchanged; `permanence` written once works anywhere there is an actor. Adding
a scenario makes every compatible family available in it for free, and vice versa — which is
the practical payoff of decoupling dynamics from rendering, beyond the twin-identity property
it was chosen for.

### Level 4 — Instances, and the two control arms

An instance is `scenario × family × seed × severity_bin`, and it always produces a **pair**:
one `valid` clip and one `invalid` twin, rendered through a bit-identical path.

```
clip_uid  =  physviol_v0/<scenario>/<seed:04d>/<label>[_<family>][_<variant>]

  physviol_v0/collision/0173/valid
  physviol_v0/collision/0173/invalid_solidity_a
  ^^^^^^^^^^^ ^^^^^^^^^^^^^^ ^^^^ ^^^^^^^^^^^^^^^^^
   release      scenario      seed  label + family + variant
```

`pair_uid` is `physviol_v0/collision/0173` and is the **leakage-free grouping key** —
never split a pair, or a scenario, across train/val/test.

**Two control families, labelled `valid`, that are not violations:**

- **Surprising-but-valid.** Rare but legal: a bounce off an `e = 0.95` surface that goes
  improbably high, a domino chain, a body resting balanced on an edge, an object re-emerging
  from the *wrong* side of an occluder because it bounced offscreen. Without these, a model
  scores well by detecting *weirdness* rather than *illegality*. Cheap for us, because
  residuals stay at the noise floor by construction.
- **Artifact / shortcut probe.** Valid clips pushed through the exact encode path the invalid
  clips use, plus deliberate motion-blur toggles, compression sweeps and render seams.
  Measures whether a detector keys on our renderer instead of on physics. IntPhys 2 ships
  this idea; we copy it.

---

## Part 3 — Annotations: exactly what ships with every clip

This is the contribution. Grouped by the question each annotation answers.

| group | answers | files |
|---|---|---|
| 3.1 labels | *what* and *whether* | `meta.json` |
| 3.2 temporal | ***when***, and ***for how long*** | `meta.json`, `timelines.npz` |
| 3.3 spatiotemporal masks | ***where***, per frame | `violation_mask.npz`, `causal_mask.npz`, `seg.npz` |
| 3.4 severity fields | ***how badly***, localised in space and time | `severity_map.npz`, `severity_t`, `residuals.npz` |
| 3.5 geometry | scene structure, free from Kubric | `depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords` |
| 3.6 token grids | ready-to-train reductions | `grids.npz` |
| 3.7 raw physics + provenance | reproducibility | `traj.npz`, `meta.json:provenance` |

### 3.1 Clip-level labels

`label` (valid/invalid), `domain`, `family`, `scenario`, `seed`, `severity_bin`
(weak/medium/strong), the `intervention` block (type, params, `magnitude`, `magnitude_unit`),
`peak_residual` (law, value, frame, z-score), the `controls` flags, `assets` with licenses,
and the cross-reference fields `intphys2_category` / `likephys_domain`.

### 3.2 Temporal annotations — when, and for how long

The part that answers *"a window of when the physics is violated"*.

| annotation | type | meaning |
|---|---|---|
| `t_event_frame` | int | law breaks in simulator state — exact, by construction |
| `t_observable_frame` | int | first frame with visual evidence — exact, via prefix identity |
| `t_end_frame` | int | last frame the intervention is active |
| `observability_lag_frames` | int | `t_observable − t_event` — the occlusion-lag metric |
| **`violation_windows`** | `[[s,e], …]` | **every interval during which the law is actively broken** |
| **`observable_windows`** | `[[s,e], …]` | every interval during which visual evidence is present |
| `consequences[k]` | list | per downstream body: `t_diverge_frame`, `t_observable_frame`, `displacement_m`, `relation` |

Rasterised into **`timelines.npz`** so consumers never expand intervals themselves:

| array | shape | dtype | meaning |
|---|---|---|---|
| `active` | `[T]` | bool | is the violation active at frame `t` |
| `observable` | `[T]` | bool | is there visual evidence at frame `t` |
| `occluded` | `[T]` | bool | is the primary culprit hidden at frame `t` |
| `severity_t` | `[T]` | f32 | violation magnitude over time — `max_b s(b,t)` (§3.4) |

**Why lists of intervals and not a single `[start, end]`:**

- `superelastic` fires on *each* bounce → `violation_windows = [[12,13],[31,32],[47,48]]`
- an object behind an occluder can become visible, re-hide, and re-emerge →
  `observable_windows = [[19,26],[38,50]]`
- `instant` violations are the degenerate case, a single window of length 1

**Invariant, enforced by `physviol validate`:** `t_event ≤ t_observable` and `t_event ≤ t_end`;
`violation_windows` are sorted, non-overlapping, and within `[0, T)`; `active` is exactly the
rasterisation of `violation_windows`.

### 3.3 Spatiotemporal masks — where, per frame

**Yes — spatiotemporal localisation is the core annotation.** Every mask is `[T, H, W]`: a
value per pixel per frame, so "where" and "when" are answered by the same array.

| array | shape | dtype | meaning |
|---|---|---|---|
| **`violation_mask`** | `[T,H,W]` | bool | **the primary annotation.** True on the culprit body's pixels, on frames where the violation is active **and visible**. |
| **`reference_mask`** | `[T,H,W]` | bool | where the culprit *should* be — its footprint in the valid twin, ungated in time. Shipped on **both** clips. |
| `causal_mask` | `[T,H,W]` | uint8 | `0` = nothing, `1` = primary culprit, `k ≥ 2` = consequence body `k−1`. Separates cause from effect spatially. |
| `seg` | `[T,H,W]` | uint16 | instance ids — the substrate every other mask is painted into |
| `divergence_map` | `[T,H,W]` | f16 | `\|valid − invalid\|` in pixel space. **Shipped for analysis, NOT the violation region, never a training target.** |

**How `violation_mask` is built — and the subtlety that makes it correct.** The naive rule
("pixels of the culprit body in the invalid render") is wrong for half our families:

- `permanence` (vanish): the body has **no pixels** in the invalid render. The violation is
  precisely that it is absent.
- `permanence` (duplicate) / `continuity` (teleport): the body is in the **wrong place**, so
  both the place it should be and the place it is are relevant.

So the rule is:

> **`violation_mask[t] = footprint(culprit, invalid, t) ∪ footprint(culprit, valid, t)`**

The union over both twins. This is only well-defined *because* the twins are pixel-aligned
and share instance indexing — the prefix-identity property paying off directly. It handles
vanish (only the valid side contributes), spawn (only the invalid side), and teleport (both,
disjoint) with one rule and no special cases.

**Per-family exceptions, stated explicitly:**

| family | mask region | why |
|---|---|---|
| `newton3_reaction` | **both** bodies in the pair | the violation is in the interaction, not in one body |
| `shadow` | the **shadow** region, not the caster | isolated via a shadow-only render pass; this is the shortcut probe |
| `global_gravity` | the **entire frame**, with `spatial_extent: "global"` | there is no localised culprit; the flag lets consumers exclude these from localisation metrics |
| `solidity` | union of both bodies, restricted to the **overlap region** where available | the violation is the interpenetration itself |

### 3.4 Severity fields — how badly, localised in space and time

The design the rest of the dataset hangs on. Six steps, from simulator state to a trainable
`[T,H,W]` field.

**Step 1 — per-body, per-frame scalar residual.** For body `b` at frame `t`, compute
`r(b,t)` for the law the injector broke, using the table in §1.2. Dimensionless by
construction. Computed from `traj.npz`, never from pixels.

**Step 2 — calibrate the noise floor.** Solver error means `r > 0` even on valid clips. Over
the valid arm of each `(scenario, law)` pair, collect the residual distribution and record
`μ`, `σ`. This is a required deliverable, not an optimisation — without it "severity" is
uncalibrated. Then

```
z(b,t) = (r(b,t) − μ_scenario,law) / σ_scenario,law
```

**Step 3 — bounded, cross-family-comparable score.** Physical units are not comparable across
families (metres of penetration vs an energy ratio). So map to `[0,1]`:

```
s(b,t) = clip( z(b,t) / z_ref(family), 0, 1 )
```

where `z_ref(family)` is the z-score at that family's `strong` severity bin. **All three of
`r`, `z` and `s` are stored** — physical units for interpretability, z for calibration, `s`
for training.

**Step 4 — paint into pixels.** For each frame, each pixel takes the score of the body
occupying it:

```
severity_map[t, y, x] = s(b, t)   where  b = seg[t, y, x],  b ∈ causal_body_ids
                      = 0         otherwise
```

Occlusion is already resolved by `seg`, so no depth reasoning is needed. Where the mask rule
in §3.3 pulls in the valid twin's footprint (vanish, teleport), the score is painted there
too. Overlaps resolve by `max`. Every *dynamic* culprit is painted, not only the primary one:
`global_gravity` acts on the whole scene and `fission` on both halves, and painting one body
would describe a fraction of the violation.

Painting is gated on `active AND observable`, like the mask. Severity that accrues while
nothing is observable is carried forward to the next observable frame **within the same
window** — otherwise a violation whose residual is a single spike on an invisible frame
(a super-elastic bounce changes velocity while the pixels are still identical; a pendulum's
angular momentum reverses a frame before the arc turns) paints its whole magnitude where
nobody can see it and reads as `0.00` on every frame that shows anything. Where nothing is
hidden the carry is the identity, so a shaped intervention keeps its rise and fall rather
than becoming a running maximum.

**Step 5 — the temporal profile.** `severity_t[t] = max_b s(b,t)`, stored in
`timelines.npz`. This is the 1-D curve a temporal model regresses; `severity_map` is the 3-D
field a spatial model regresses. They are consistent by construction:
`severity_t[t] == severity_map[t].max()`.

**Step 6 — special cases.** `global_gravity` paints a *uniform* value over the whole frame.
`shadow` paints into the shadow region from the shadow-only pass. `newton3_reaction` paints
the same imbalance value into both bodies.

#### The intervention should have a *shape*

A violation implemented as a step — flip a parameter at `t_event` and leave it flipped —
produces a residual that is constant for the rest of the clip, so `severity_map` holds one
value everywhere and the "field" is a flag wearing a costume. It also means there is no
*after*: the clip never returns to legal physics, so nothing shows what recovery looks like.

So a sustained intervention ramps: `antigravity` drives its gravity scale from 1 up to the
bin's peak and back to 1 over the window (a raised cosine), after which the actor obeys real
physics again. The residual traces the same curve, and `severity_t` becomes a genuine
profile — e.g. `0.13 → 0.40 → 0.40 → 0.13 → 0` for the medium bin.

**Two traps this exposed, both of which produce confident nonsense:**

1. **An injector must not smuggle in a second violation.** Bending gravity with plain
   ballistic integration drops the actor straight through the floor — a *solidity* failure
   inside a clip labelled `antigravity`, with only one of the two annotated. Every injector
   that re-integrates motion does so against a solid ground plane
   (`Injector._integrate_profile`). Guarded by
   `test_antigravity_does_not_smuggle_in_a_solidity_violation`.
2. **An intervention must be sustained, not merely peaked.** A raised-cosine ramp touches
   its peak instantaneously, so the *mean* effect over the window is `(1 + peak) / 2` — half
   what the bin advertises. On a body already moving fast that is not enough to change its
   visible motion, and the strongest bin ends up looking like the weakest. The profile is a
   **trapezoid**: ramp in, hold at the peak, ramp out. Ramp values sit strictly between 1 and
   the peak so every frame inside the window is genuinely violating — an `active` frame where
   `alpha == 1` would be a frame marked wrong with nothing wrong in it.
3. **A bounce is a velocity reversal *at the floor*.** The free-fall law asks "is this
   The free-fall law asks "is this body accelerating like gravity?", which is only fair while
   nothing holds it up, so contact frames must be excluded. Two ways to get that gate wrong,
   and we hit both:

   * Gating on "is it resting on the floor" misses bounces, which complete *between* sampled
     frames — the actor is airborne on both neighbours while its velocity reversed in
     between, and the central-difference acceleration straddling that contact reported
     residuals of ~6 against a physical maximum of 2.6. Those spikes also contaminate the
     noise floor measured on the valid arm and swamp the weak bin entirely.
   * Gating on velocity reversal *alone* is worse, and fails silently: reversed gravity flips
     the actor's vertical velocity too, so the gate deletes exactly the frames where the
     violation peaks and reports **zero severity at its strongest moment**.

   The gate therefore requires both — a reversal **and** proximity to the surface, with the
   clearance allowance scaled by `|v_z| · dt` since the contact happens between samples.
   Guarded by `test_bounce_gate_does_not_eat_the_antigravity_signal`.

#### Worked example, end to end — `antigravity` on `drop`

Every number below is either chosen by us or computed from `traj.npz`. Nothing comes from
pixels.

| step | quantity | value | where it comes from |
|---|---|---|---|
| **choose** | gravity scale `α` | `0.3` | the sampler picks it from the `medium` bin |
| **choose** | `intervention.magnitude` | `\|1 − α\| = 0.70` | pure arithmetic on the knob; unit `gravity_scale_deviation` |
| **inject** | `violation_windows` | `[[9, 24]]` | frames the injector applies `g → αg` to body 3 |
| **simulate** | acceleration `a(3,t)` | `≈ 0.3·g` | second difference of `pos[t,3]` from `traj.npz` |
| **measure** | residual `r(3,t)` | `‖a − g‖/‖g‖ = 0.70` | the free-fall law from §1.2 |
| **calibrate** | noise floor `μ, σ` | `0.004, 0.002` | over the *valid* arm of `drop` |
| **normalise** | `z(3,t)` | `(0.70 − 0.004)/0.002 = 348` | z-score vs that floor |
| **bound** | `z_ref(antigravity)` | `498` | the z of the `strong` bin (`α = 0`, so `r = 1.0`) |
| **bound** | `s(3,t)` | `clip(348/498, 0, 1) = 0.70` | the trainable score |
| **paint** | `severity_map[t,y,x]` | `0.70` where `seg[t,y,x] == 3`, else `0` | §3.4 step 4 |
| **reduce** | `severity_t[t]` | `0.70` for `t ∈ [9,24]`, else `0` | `max_b s(b,t)` |

Read the last three rows together and the design becomes concrete: **the magnitude is a
number attached to a body, and the mask is what turns that number into a location.** The
body occupies certain pixels on certain frames (`seg`), so painting its score into those
pixels produces a quantity that is simultaneously *how badly* (`0.70`), *where* (the ball's
footprint) and *when* (frames 9-24).

Note that `r = 0.70` coincides with `magnitude = 0.70` here only because `antigravity` is a
family whose knob and whose law residual happen to share units. For `solidity` the knob is
"disable the contact pair for 15 frames" and the residual is a penetration depth in metres —
related, but not equal, and only knowable after simulating. That is exactly why both are
stored.

**`violation_mask` and `severity_map` are deliberately derived differently**, and a consumer
should know which they are using:

| | `violation_mask` | `severity_map` |
|---|---|---|
| source | the injector's ground truth | measured law residuals |
| value | binary | continuous `[0,1]` |
| exact? | **yes, by construction** | up to the noise floor |
| use for | localisation, detection | magnitude regression, difficulty |

They will *mostly* agree. Where they disagree — a sustained violation whose residual dips
mid-window — that disagreement is real physics, not a bug.

### 3.5 Geometry passes

`depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords`, all `[T,H,W,·]`, straight from
Kubric's exporters at no extra cost. Shipped because they make the dataset useful for work
that is not about violations at all.

### 3.6 Token grids

`grids.npz` pre-reduces masks and severity to the latent token grid so a consumer never
re-derives a VAE's binning. Tier A: `7×16×16`. Tier B: `21×16×16` on the 81-frame derivative.

- `mask_<F>x16x16` — bool, reduced by **max** (a violation in any contributing source frame
  marks the latent frame)
- `severity_max_<F>x16x16`, `severity_mean_<F>x16x16` — f16. Peak and average are different
  questions; both ship.
- a `T×32×32` reduction alongside, for consumers on a different tokenizer

**Ordering guarantee:** time-major, `[F_lat, H_lat, W_lat]`, latent frame slowest. This is a
schema guarantee, not an implementation detail — flattening must match transformer token
order.

### 3.7 Raw physics and provenance

`traj.npz` (the seam file: per-body pose, velocity, applied force, contacts, residuals,
events) and the `provenance` block (`generator_commit`, `kubric_image_digest`,
`blender_version`, `render_seed`, `prefix_identical_verified`,
`prefix_identical_upto_frame`). Shipping `traj.npz` means renders reproduce without
re-simulating.

---

## Part 4 — Architecture

The one architectural choice that matters: **decouple dynamics from rendering.** Simulate,
dump a trajectory file, render from the trajectory. Violation injection then becomes a
well-defined edit at trajectory level, and valid/invalid twins go through a *bit-identical*
render path — exactly the appearance-matching property LikePhys and IntPhys 2 rely on and
which twin-diff labellers on those datasets fail to get.

```
scenarios/*.py          seeded scene sampler (assets, camera, lighting, initial state)
        │               Level 3 of the taxonomy
        ▼
sim/                    dynamics backend — v0: PyBullet (via Kubric); Phase 3: MuJoCo/MJX
        │               injectors/ apply the intervention over violation_windows
        ▼
traj/<clip>.npz    ◄──  THE SEAM.  Swapping engines touches nothing downstream.
        │               per-body pos[T,3] quat[T,4] lin_vel ang_vel applied_force
        │               contacts[T]: (bodyA, bodyB, point, normal, impulse, penetration)
        │               residuals[T, n_bodies, n_laws]
        │               events: violation_windows, causal_body_ids, intervention params
        │
        │               ══ container above this line / host conda env below ══
   ┌────┴────┐
   ▼         ▼
render valid  render invalid       Blender. Same seed, same code path, same everything.
   └────┬────┘
        ▼
annotate/               observability + windows (3.2), masks (3.3), severity (3.4),
                        grids (3.6), meta.json
        ▼
clips/<uid>/            the released layout
```

**The single most valuable test in the repo:** assert that valid and invalid frames are
**bit-identical for every frame before `t_event`**. If that assert ever fails, the seed
plumbing or the render path has drifted and every downstream annotation is suspect.

**Two traps found while building this, both silent.**

1. **Kubric's raw segmentation is numbered by scene-asset order**, and each asset's declared
   `segmentation_id` is honoured only if you run `kb.adjust_segmentation_idxs()` afterwards
   (as `movi_def_worker.py` does). Skip it and instances are relabelled whenever declaration
   order differs from insertion order — which mislabels every mask, residual and severity
   value downstream while still looking entirely plausible. `drop` happened to be
   immune because its declaration order matched; `occluder_pass` swapped actor and occluder.
   Guarded by `test_segmentation_ids_match_what_the_scenario_declared`.
2. **"Occluded" must mean *fully* occluded.** A geometric occlusion test that allows a few
   actor pixels to still show makes a violation instantly observable and collapses the
   observability lag to zero — the very quantity `occluder_pass` exists to produce. The test
   shrinks the occluder's extents by the actor's *projected* silhouette radius, and the
   injector fires at the middle of the fully-hidden run.

**Measured, with one exception that is real physics rather than a bug.** On the Phase 0 pair,
six of the seven passes are identical to *exactly* 0.0 before `t_event`: `rgba`,
`segmentation`, `depth`, `normal`, `object_coordinates`, `backward_flow`. **`forward_flow`
diverges one frame earlier**, at `t_event − 1`, because forward flow at frame *t* encodes
motion *t → t+1* and frame `t_event` is where the twins part. `backward_flow` correspondingly
diverges exactly at `t_event`. So the assertion is per-pass with a one-frame lookahead
exemption for `forward_flow` only — encoded as `LOOKAHEAD` in
`tests/test_prefix_identity.py`, with a test that pins the exemption to exactly one frame so
nobody widens it to paper over a genuine drift.

### On-disk layout

```
physviol_v0/
  README.md  LICENSE  index.json          version, taxonomy, counts, split defs
  splits/{train,val,test}.txt             grouped by scenario, not by clip
  clips/<clip_uid>/
    meta.json              §3.1 + §3.2 scalars + provenance — the source of truth
    rgb.mp4                master (Tier A: 256²/25f; Tier B: 512²/96f), H.264 CRF 14
    rgb_derivative.mp4      Tier B only: 256²/81f for video-DiT consumers
    timelines.npz          §3.2  active, observable, occluded, severity_t   [T]
    seg.npz                §3.3  uint16 [T,H,W]
    violation_mask.npz     §3.3  bool   [T,H,W]   ← the primary annotation
    causal_mask.npz        §3.3  uint8  [T,H,W]
    severity_map.npz       §3.4  f16    [T,H,W]
    residuals.npz          §3.4  r, z, s per body per frame per law
    divergence_map.npz     §3.3  f16    [T,H,W]  — NOT the violation region
    depth.npz  flow_fwd.npz  flow_bwd.npz  normals.npz  object_coords.npz   §3.5
    grids.npz              §3.6  reduced masks + severity
    traj.npz               §3.7  the seam file
```

The `meta.json` field reference is maintained separately in [schema.md](schema.md).
`real2sim` is `null` in v0 and reserved for Phase 4; reserving it now means Phase 4 does not
force a regeneration.

### Repo skeleton

```
physviol/
  environment.yml  pyproject.toml
  docker/          kubric.sh  IMAGE_DIGEST
  scripts/         fetch_refs.sh
  refs/            gitignored reference checkouts (kubric)
  context/         source papers, gitignored; see context/README.md
  physviol/
    taxonomy.py    DOMAINS, FAMILIES, SCENARIOS, COMPATIBILITY — Part 2 as data
    scenarios/     drop.py  collision.py  ramp_slide.py  toss.py
                   occluder_pass.py  stack_topple.py  pyramid_impact.py  pendulum_swing.py
                   resting_table.py  rolling_ramp.py  shadow_track.py  clutter_toss.py
    sim/           backend.py (ABC)  pybullet_backend.py  [mjx_backend.py — Phase 3]
    injectors/     base.py  identity.py  kinematics.py  contact.py  dynamics.py
                   equilibrium.py  optical.py  global_scene.py
    residuals/     laws.py  noise_floor.py
    render/        worker_smoke.py  worker.py  passes.py  derivatives.py
    annotate/      windows.py  observability.py  masks.py  severity.py  grids.py  meta.py
    controls/      surprising_valid.py  artifact_probe.py
    schema/        meta.schema.json  validate.py
    viz/           overlay.py  contact_sheet.py        ← built in Phase 0, used forever
    queue/         runner.py      resumable per-clip parallel render queue
    cli.py         generate | render | annotate | validate | report
  tests/           test_prefix_identity.py  test_schema.py  test_residual_zero_on_valid.py
                   test_windows.py  test_mask_union.py
  docs/            PLAN.md  schema.md  prior_art.md
```

---

## Part 5 — Phasing

**Phase 0 — two clips (days 1-3). — BUILT.** `drop` × `solidity`, all three severity
bins, end to end: sample → simulate → inject → render both twins → annotate → overlay →
validate. ~16.7 s per pair at Tier D including annotation and video encoding. Prefix identity
holds exactly; `physviol validate` is clean; 21 tests pass. Remaining Phase 0 item: produce
the deliverable pair at **Tier A** and eyeball it. Kubric smoke render → full annotation export → an overlay video
with the mask, the severity heatmap and all three clocks burned into the frame. **Look at it
before anything scales.** Assert prefix identity is exactly zero. Nothing else starts until
these two clips are visually correct.

**Phase 1 — ~50 clips (weeks 1-2).** Tier A. Three scenarios, four families spanning at least
three domains, schema frozen and schema-validated. Measure the optimal parallel-render width
and ship the queue. Then **immediately** run 2-3 off-the-shelf models — a V-JEPA-2 surprise
baseline, a VLM prompt, a video-DiT probe. If they sit at 95% or at chance, difficulty
calibration is wrong and must be fixed *now*, not after 600 clips.

**Phase 2 — ~300 pairs / 600 clips (weeks 3-8).** Tier A, the full 41-cell compatibility
matrix, both control arms, severity-binned weak/medium/strong splits, scenario-grouped
train/val/test. **Release `physviol_v0`.** ~7 h of render serial, less parallel.

**Phase 3 — Tier B + fluid/cloth + realism + MJX.** Re-run the same generators at
512²/24 fps/97 frames. Swap the dynamics backend behind the seam for exact residuals and
deformables — this is where GPU physics earns its place. **This is also where true fluid and
cloth land**: a mesh-cache seam, field-level conservation residuals, and either a working
headless Mantaflow path or MJX deformables (see the fluid findings in Part 2). Upgrade to the cluttered-scene, moving-camera, dynamic-lighting
tier. **Release `physviol_v1`.**

**Phase 4 — real→sim.** Own props, photogrammetry, calibrated multi-view capture, system ID
(mass/friction/restitution by differentiable sim or CMA-ES), re-render at the true camera
pose, and report the **twin gap** (reprojection error, LPIPS) as a dataset property. A small
physically-staged *real-invalid* set (hidden magnet, thread, false bottom) makes the benchmark
much harder to dismiss as synthetic-only.

**Phase 5 — release hardening.** Human study (IntPhys 2's argument rests on humans near
ceiling while models sit at chance), baseline table, asset license audit, HF dataset card.

---

## Part 6 — Consumer contract

PhysViol has no privileged consumer. The published on-disk layout **is** the API. Nothing in
`physviol/` imports a consumer; no consumer needs to import `physviol`.

**What a consumer reads:** `index.json` for version, taxonomy and counts;
`splits/{train,val,test}.txt` for the scenario-grouped split; then per clip, `meta.json` plus
whichever arrays it needs. A clip-level classifier needs only `rgb.mp4` and `label`. A
temporal model adds `timelines.npz`. A spatial model adds `grids.npz` or the full-resolution
masks.

**A typical loader mapping.** `meta.json` is arranged to fill a standard record without
computation:

| record field | source |
|---|---|
| `uid` | `clip_uid` |
| `video_path` | `rgb.mp4` (or `rgb_derivative.mp4`) |
| `label` | `label` — valid / invalid |
| `task` | `family` (or `domain` for coarser grouping) |
| `scene_id` | `scenario` + `seed` — the **leakage-free grouping key** |
| `pair_id` | `pair_uid` |
| `variant` | `family` + intervention type |
| `twin_path` | the valid twin's video |
| extras | `t_event_s`, `t_observable_s`, `violation_windows`, severity, `grids_path` |

A consumer may set its equivalent of `pixel_aligned_twin = True` — and here that flag is
**earned**, not asserted: `provenance.prefix_identical_verified` records that the
prefix-identity test passed for that pair, with the frame it held to.

**Four evaluation protocols this dataset unlocks**, none possible on whole-clip-labelled
benchmarks:

1. **Onset-labeller calibration.** Twin-diff and object-centric pseudo-onset heuristics can
   finally be *scored*, against a known `t_observable`, on data where the twins really are
   pixel-aligned. That converts "this heuristic covers 46% of clips" from a caveat into a
   measured error bar.
2. **Duration and window prediction.** With `violation_windows` as ground truth, temporal IoU
   becomes a metric — not just onset error.
3. **Spatial supervision.** `grids.npz` is a drop-in target for a per-token spatial head: for
   hidden states `[B, S, D]` with `S = F_lat·H_lat·W_lat`, drop the spatial mean-pool and
   reshape to `[B, F_lat, H_lat, W_lat, D]`, which aligns with the stored grid given the
   time-major guarantee. Caveat: retaining full spatial features costs hundreds of MB per
   video per denoising step, so hold a *subset of layers*.
4. **Severity regression.** `severity_t` and `severity_map` make "how wrong is it" a
   regression target, not a class label.

---

## Part 7 — Risks

1. **Render throughput — measured, and mitigated by parallelism rather than hardware.**
   ~1.75 s/frame at 256², ~7.16 s at 512², linear in frames, 74% of it *not* sample-bound.
   The GPU/OptiX path caps at ~1.37× by Amdahl and risks breaking Kubric's 2.93-pinned `bpy`
   integration, so it is **not** scheduled work — clip-level parallelism already measures
   1.92× at width 4, for free. Tier A at ~4 h parallel is comfortable; Tier B at ~60 h is why
   the resumable queue is a Phase 1 deliverable.
2. **The L40S is shared** with other jobs on this box. The render queue is CPU-only, so it
   does not contend — but the Phase 1 baseline evaluations do. Keep them checkpointed.
3. **Determinism across versions.** PyBullet is deterministic for a fixed build and thread
   count, not across versions. Pin the docker digest, ship `traj.npz` so renders reproduce
   without re-simulating, and let the prefix-identity assert catch any drift. **Parallel
   rendering must not change results** — each clip is a separate container, so this holds,
   but the prefix-identity test is what proves it.
4. **Severity noise floor.** Residuals are nonzero on valid clips. Calibrate per
   `(scenario, law)`; report physical units, z-score and bounded score.
5. **Mask correctness for absence-type violations.** The `valid ∪ invalid` union rule (§3.3)
   is the whole reason `permanence` and `continuity` masks are meaningful. `test_mask_union.py`
   guards it; a regression here silently produces empty masks on a sixth of the dataset.
6. **Domain gap.** Randomise aggressively (shape, material, HDRI, camera, counts, initial
   velocities, occluder geometry, friction); report transfer onto IntPhys 2 / LikePhys as a
   headline number; Phase 3 realism tier.
7. **Shortcut leakage** — handled by the artifact/debug control arm and by the `shadow`
   family, whose mask is deliberately not on the object.
8. **Asset licensing.** MOVi's asset mix is not uniformly CC0, and the schema promises a
   license string per asset. Record it as each source is enabled, not at release time.
9. **Positioning.** PhyGround and PhyCheck are the nearest neighbours and are moving fast, but
   both are QA-style and human-annotated. Our differentiators — simulator-derived pixel-precise
   ground truth, continuous severity instead of a label, violation *windows* rather than a
   single onset, and the causal-onset vs observability distinction — should be written up as a
   positioning paragraph in **week one**, because that paragraph also tells us if the gap has
   closed.

---

## Verification

Ordered by when it runs; the first four catch real bugs.

- **`tests/test_prefix_identity.py`** — for every pair, `max|valid − invalid| == 0` for all
  frames `< t_event`. Fails loudly; blocks the release build.
- **`tests/test_residual_zero_on_valid.py`** — every law residual on valid clips stays below
  the calibrated noise floor. Catches injector leakage into the control arm.
- **`tests/test_windows.py`** — `violation_windows` are sorted, non-overlapping, in range;
  `active[T]` is exactly their rasterisation; `t_event ≤ t_observable`;
  `severity_t[t] == severity_map[t].max()`.
- **`tests/test_mask_union.py`** — for `permanence`-vanish and `continuity`-teleport clips,
  `violation_mask` is **non-empty** on every frame that is active **and** observable, and
  empty on active frames where nothing is observable. This is the test that would
  catch the naive invalid-only footprint rule.
- **Visual check, every phase, before scaling.** `physviol viz overlay <clip>` burns the
  violation mask, the severity heatmap, the window bar and all three clocks onto the RGB and
  writes an mp4; `contact_sheet` builds an HTML gallery. Never trust a metric built on an
  unviewed label.
- **`physviol validate`** — jsonschema over every `meta.json`, plus cross-checks: mask
  non-empty while active *and* observable, `causal_body_ids` present in `seg.npz`, every asset carries a
  license string, `prefix_identical_verified` true, `family`/`domain` consistent with
  `taxonomy.py`, and `(scenario, family)` present in the compatibility matrix.
- **Difficulty calibration (end of Phase 1)** — V-JEPA-2 surprise, a VLM prompt, and a
  video-DiT probe on the 50-clip set. Target: meaningfully above chance, well below ceiling.
  IntPhys 2's reference point is models at 55-59% against humans at 96%.
- **Round trip through an external consumer** — build the Part 6 loader mapping against the
  published format alone (no imports from this package), then report pseudo-onset error
  against known `t_observable` and train a spatial head against `grids.npz`.
- **Transfer** — a probe trained on PhysViol, evaluated on LikePhys / IntPhys 2. This is the
  test that it learned physics rather than our shader.

## Sources

- [IntPhys 2 (arXiv:2506.09849)](https://arxiv.org/abs/2506.09849) — four principles, debug split
- [LikePhys (arXiv:2510.11512)](https://arxiv.org/abs/2510.11512) — 12 scenarios, 4 domains
- [Kubric](https://github.com/google-research/kubric) — masks/depth/flow exporters, `kubricdockerhub/kubruntu`
- [PhyGround (arXiv:2605.10806)](https://arxiv.org/html/2605.10806v1) and [PhyCheck (arXiv:2608.02150)](https://arxiv.org/abs/2608.02150v1) — nearest prior art, both QA-style
- [Real2Sim / Gaussian-splat editing (arXiv:2605.13591)](https://arxiv.org/html/2605.13591), [Scalable Real2Sim](https://www.researchgate.net/publication/389547593_Scalable_Real2Sim_Physics-Aware_Asset_Generation_Via_Robotic_Pick-and-Place_Setups) — Phase 4 machinery

The first two PDFs are kept locally in `context/` (gitignored — see
[context/README.md](../context/README.md) for the ids to re-fetch them).
