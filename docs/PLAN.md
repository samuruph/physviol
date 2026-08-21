# PhysViol — a spatio-temporally annotated physics-violation dataset

> The design document for this repository. Schema details live in
> [schema.md](schema.md); the prior-art coverage matrix lives in
> [prior_art.md](prior_art.md).

## Context

Every existing intuitive-physics video benchmark labels **whole clips**. IntPhys 2 and
LikePhys both ship valid/invalid pairs with one binary label per video; PhyCheck and
PhyGround add fine-grained *questions*, but their labels are human-written QA over
real/AIGC video, not simulator-derived geometry. Nothing provides **where in the frame**
and **exactly when** the law broke, and nothing provides **how badly**.

That gap blocks concrete work, and we hit it directly. In a prior probing project of ours,
per-timestep violation probes could only be scored against a *pseudo-onset* recovered by
diffing a clip against its "twin" — and that labeller fails structurally, because LikePhys
valid/invalid clips are *separate renders*: only **3%** of pairs share a near-zero baseline.
An object-centric fallback covered **46%** of violated clips, and just **10 of 60** on the
`cloth_drape` scenario. Spatial localization there was not merely unsupervised — it was
**unevaluable**.

Generating the data inverts the problem: we do not *recover* a violation, we *inject* it,
so onset, extent, culprit object and magnitude are known by construction.

**Scope.** PhysViol is a **general-purpose dataset in its own right**, usable the way
IntPhys 2 and LikePhys are used today by anyone. It has no privileged consumer; the
published on-disk format is the entire contract (Part 5).

**Decisions locked:**

| | |
|---|---|
| Location | this repository, standalone |
| Stack | **Kubric** (Blender + PyBullet, docker) for v0, with an explicit trajectory seam so **MuJoCo** can replace PyBullet later |
| Master clip | **512×512, 24 fps, 96 frames (4 s)**; ship a 256×256/81-frame derivative |
| Severity | **both** — the injected intervention magnitude *and* a measured per-frame physical-law residual field |
| v0 categories | Permanence, Solidity, Continuity, Dynamics — **plus new families** (parabolic-flight, Newton 1/2/3, support, friction, shadow, global-g) |
| Realism | start procedural-room + HDRI + orbit camera; **architecture must not block** the cluttered-scene / moving-camera tier later |
| Real→sim | reserve the schema fields now, build in Phase 4 |

**Can everything be scripted? Yes, entirely.** Blender runs headless (`blender -b -P`),
Kubric drives it from Python, PyBullet is headless by default, and Poly Haven / GSO assets
download through manifests. No GUI is required at any point. Diversity comes from seeded
procedural sampling, not from hand-authoring scenes.

---

## Part 0 — How Kubric is actually used

**You do not need to clone Kubric as a dependency, and you must not vendor it into the
package.** The published docker image already contains a complete Kubric installation and
its dependencies, Blender included.

The pattern is: **your script, their container.** Mount this repo at `/kubric` inside the
container and run your own worker file; Kubric is already importable there.

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
[`docker/kubric.sh`](../docker/kubric.sh) so no one has to retype it, and it prefers the
pinned digest over the floating `:latest` tag.

A worker is plain Python — `kb.Scene(...)`, add objects, `kb.simulator.PyBullet` to run
dynamics, `kb.renderer.Blender` to render, then `kb.write_image_dict(...)`. Fully headless,
fully scriptable, no GUI at any point.

### The reference checkout

**Do clone Kubric — as pinned, read-only, gitignored reference material.** The image gives
you a *runnable* Kubric; the clone gives you a *readable* one. Phase 0 largely consists of
reading `challenges/movi/movi_def_worker.py` (Part 0.5) and grepping `write_image_dict`,
the `AssetSource` plumbing and the optical-flow sign conventions. Doing that through
`docker run … cat` is needlessly painful.

```bash
bash scripts/fetch_refs.sh     # -> refs/kubric at a pinned sha, gitignored
```

Start with `refs/kubric/challenges/movi/movi_def_worker.py`, then
`refs/kubric/examples/keyframing.py` (the replay mechanism the seam depends on) and
`refs/kubric/kubric/renderer/blender.py` (what the exporters actually emit).
`refs/kubric/docker/` also carries upstream's `Blender.Dockerfile`, which is the starting
point if we ever build our own image.

### Measured environment (recorded Phase 0, not assumed)

| | |
|---|---|
| Host | 8 CPU cores, NVIDIA L40S 46 GB, 344 GB free |
| Docker | 25.0.14, **`nvidia` runtime present** — a GPU image is buildable here |
| Image | `kubricdockerhub/kubruntu`, digest in `docker/IMAGE_DIGEST`, **built 2022** |
| Inside the image | Blender **2.93.4**, Python **3.9.5**, kubric **2022.4.1**, PyBullet 2020.10.06 |
| Reference clone | `google-research/kubric` @ `61f2422c`, carrying 2026 copyright headers |

**The image and the clone are ~4 years apart. This matters, and was checked rather than
assumed:** every API symbol the 2026 `movi_def_worker.py` uses — `kb.setup`,
`AssetSource.from_manifest`, `compute_visibility`, `adjust_segmentation_idxs`,
`write_image_dict`, `process_collisions`, `move_until_no_overlap`,
`sample_point_in_half_sphere_shell`, `post_processing.compute_bboxes` — **is present in the
image's Kubric.** So the clone is safe to use as a template source. The rule stands
regardless: **the installed copy in the container is the API authority**; if a call
signature surprises you, diff against `refs/kubric` before believing either one.

### Measured throughput, and what it means (Phase 0 result)

Measured with [`physviol/render/worker_smoke.py`](../physviol/render/worker_smoke.py) — a
two-object scene (floor + falling sphere, one light), all seven passes exported, primitives
only so nothing downloads:

| config | s / frame | projected for 600 clips × 96 frames |
|---|---|---|
| 512², 64 spp, 4 frames | 7.16 | 115 h |
| 512², 64 spp, 16 frames | 7.29 | 117 h |
| 512², **16 spp**, 3 frames | 5.61 | 90 h |

Three conclusions, and the second one was a surprise:

1. **Cost is linear in frames.** 4-frame and 16-frame runs agree to within 2%, so there is
   no meaningful fixed startup inside `renderer.render()` and the extrapolation is sound.
2. **Sampling is *not* the dominant knob.** Cutting `samples_per_pixel` 4× (64 → 16) bought
   only **22%**. The per-frame cost is dominated by the auxiliary passes (depth, both flows,
   normals, object coordinates, segmentation), scene sync and file I/O — none of which scale
   with sample count, and all of which we need. **Turning down spp will not save this
   project**, which is the opposite of what MOVi's config suggests.
3. **~5 days of continuous CPU rendering for the v0 target, and that is a floor.** This
   scene has two objects and one light. A realistic scenario — HDRI dome, 10-20 GSO
   distractors, an occluder, an orbiting camera — will be materially slower, and the L40S is
   shared. Phase 2 is not affordable on CPU Cycles at 512².

**Therefore:** the custom-image path in Part 6 is not a contingency, it is scheduled work.
Build from upstream's `refs/kubric/docker/Blender.Dockerfile` with a modern Blender LTS plus
OptiX; the docker `nvidia` runtime is confirmed present on this box. Do it before Phase 2,
not during. Phase 0 and Phase 1 (two clips, then ~50) are perfectly affordable on CPU as-is —
50 clips is roughly 10 hours — so this does not block starting.


---

## Part 0.5 — From MOVi to PhysViol

MOVi is the right skeleton and the wrong dataset. `movi_def_worker.py` is "toss N random
objects and export everything"; we need "stage a *specific* physical situation, break one
law at a known instant, and export what proves it". Writing down which half we keep stops
the build from either re-deriving Kubric plumbing or accidentally shipping MOVi with a
label column. All claims below were read out of the pinned checkout, not recalled.

**Taken directly from the MOVi worker:**

- asset-source plumbing — KuBasic / GSO / HDRI Haven, loaded from
  `gs://kubric-public/assets/*.json` manifests, plus the dome-as-background idiom that
  textures a KuBasic `dome` with an HDRI
- the `kb.Scene` → `kb.simulator.PyBullet` → `kb.renderer.Blender` spine, and
  `kb.setup(FLAGS) -> scene, rng, output_dir, scratch_dir` for a single seeded rng
- `animation, collisions = simulator.run(...)` — it returns **both** per-object keyframes
  **and** collision events, and `kb.process_collisions(...)` serializes them. That is
  already most of our `contacts[T]` seam field, for free.
- the export path: `rgba`, `segmentation`, `depth`, `forward_flow`, `backward_flow`,
  `normal`, `object_coordinates` via `kb.write_image_dict`, with `kb.compute_visibility`
  and `kb.adjust_segmentation_idxs` to renumber instances by visibility. **Five of the
  arrays in our on-disk layout come free from here.**
- `kb.move_until_no_overlap(obj, simulator, spawn_region=..., rng=rng)` for valid placement
- the settling idiom — `simulator.run(frame_start=-100, frame_end=0)`, then zero the
  velocities — which is exactly what our support/stack scenarios need
- the camera modes `fixed_random` / `linear_movement` / `linear_movement_linear_lookat`.
  The moving-camera tier the realism roadmap wants **already exists in the template**.

**What we replace, and why each one is load-bearing:**

1. **Split simulate from render.** MOVi renders straight out of `simulator.run()`. We write
   `traj.npz` in between and render by *replaying* keyframes. `simulator.run()` works by
   writing keyframes onto the scene objects, and `examples/keyframing.py` shows the same
   objects being keyframed by hand via `obj.keyframe_insert("position", frame)` — so replay
   is a supported path, not a hack. This is the seam in Part 3, the thing that makes
   valid/invalid twins bit-identical before `t_event`, and the single structural edit to
   the template.
2. **Scenario library instead of random tossing.** MOVi has one scene generator. We need a
   staged set grounded in the prior art, including **occluders** — without them
   `t_observable` always equals `t_event` and the occlusion-lag contribution disappears.
3. **Injectors.** MOVi has no notion of intervention. Ours operate on the trajectory, never
   on the sim rng, so the twin's seed and render path are untouched.
4. **Residuals and annotation.** Entirely new: `residuals/`, `annotate/`, and the four
   arrays MOVi cannot give us (`violation_mask`, `causal_mask`, `severity_map`,
   `divergence_map`) plus `grids.npz`.

**Three facts from MOVi that change our numbers:**

- MOVi-def defaults to **256², 24 frames, 12 fps**. Our master is **512², 96 frames, 24
  fps** — 4× the pixels and 4× the frames, so roughly **16× the render cost per clip**. No
  published MOVi timing is a proxy for ours.
- `samples_per_pixel=64` with `use_denoising=True` is MOVi's setting. It looks like the
  obvious throughput dial and **measurably is not** — see Part 0: 4× fewer samples bought
  22%. Whatever value we pick, it must be identical across a valid/invalid twin pair or
  prefix identity breaks.
- MOVi's asset mix is not uniformly CC0. Since the schema promises a license string per
  asset and Phase 5 promises an audit, record the license of each asset source as it is
  enabled rather than discovering the problem at release.

---

## Part 1 — The design that makes this a contribution

Three ideas carry the whole dataset. Everything else is plumbing.

### 1.1 Three clocks, not one onset

Existing benchmarks have no onset at all; the naive version of this dataset would have
one. There are really three distinct times, and the gaps between them are the point:

- **`t_event`** — the frame the law breaks *in simulator state*. Known exactly: it is when
  the injector fires.
- **`t_observable`** — the first frame carrying *visual evidence*. If the violation happens
  behind an occluder these differ by a second or more. Computed exactly, not estimated:
  because valid and invalid twins go through a bit-identical render path with the same
  seed, the pixel difference before `t_event` is **exactly zero**, so `t_observable` is the
  first frame where the causal bodies' rendered footprint differs at all.
- **`t_consequence[k]`** — when each *downstream* body first diverges (from trajectory
  data), each with its own observability frame.

This yields two metrics no benchmark currently has: **detection latency**
(`predicted_onset − t_observable`) and **occlusion lag** (`t_observable − t_event`). It also
lets us build a split where the violation is *never* directly observable and only its
consequences are — the hardest honest test in the space.

### 1.2 Violation is a law residual, never "difference from the valid rollout"

This is the trap to design around. After `t_event` a valid/invalid pair diverges
*everywhere* downstream — shadows, contact chains, secondary collisions — and in chaotic
scenes it diverges enormously. **The pixels that differ are not the pixels where the
violation is.** Worse, a merely different-but-valid rollout also diverges, so divergence
cannot define violation at all.

So severity is computed **per body per frame from simulator state**, as a dimensionless
residual against the law that was broken:

| law | residual |
|---|---|
| linear momentum | `‖m·Δv − (ΣF_contact + m·g)·Δt‖ / (m·‖g‖·Δt)` |
| angular momentum | same form with torques and `I·Δω` |
| energy at contact | `max(0, E_after − E_before) / E_before` |
| solidity | max signed penetration depth, in m, normalized by body radius |
| free fall / support | `‖a − g‖ / ‖g‖` for an unsupported body |
| trajectory shape | RMS deviation (m) of flight path from the fitted `g`-parabola |

Residuals are computed for **valid clips too**. They are not exactly zero there — solver
error is real — so a **noise-floor calibration pass** over valid clips is a required
deliverable, and severity is reported *both* in physical units *and* as a z-score above
that scenario's valid-clip residual distribution.

The pixel-space `|valid − invalid|` map is still exported, but under the name
`divergence_map` with an explicit note in the schema that it is **not** the violation
region — so nobody trains on it by accident.

### 1.3 Severity is a field, not a flag

Two numbers per clip, and one map per frame:

- **intervention magnitude** — the knob we turned, exact by construction (teleport distance
  in metres, gravity scale α, restitution `e`, impulse `J/(m·v_typ)`). This is what we sweep
  to build principled **easy / medium / hard** splits, instead of splitting by vibes.
- **peak residual** — the measured consequence of that knob.
- **`severity_map[T,H,W]`** — the residual `r(b,t)` painted into body *b*'s instance mask,
  zero elsewhere. A model trained on this learns *how wrong* and, through which law's
  residual is firing, *why*.

---

## Part 2 — Violation taxonomy

Grounded in IntPhys 2's four principles (Permanence, Immutability, Spatio-Temporal
Continuity, Solidity) and LikePhys's four domains (Rigid Body, Continuum, Fluid, Optical),
then extended. The cell-by-cell coverage matrix — which prior-art condition each of our
scenarios covers, what v0 deliberately skips, and which families are genuinely new — lives
in [prior_art.md](prior_art.md).

### v0 — rigid body (build these)

| family | injection | severity unit | kind |
|---|---|---|---|
| **Permanence** | remove / duplicate body, ideally while occluded | occlusion lag (frames) | instant, sustained |
| **Immutability** | swap shape, size, material or colour behind occluder | Δvolume ratio, ΔLab | instant |
| **Continuity** | discontinuous position set | jump distance (m) | instant |
| **Solidity** | disable collision pair for N frames | penetration depth (m) | sustained |
| **Anti-gravity** | per-body gravity scale `g → αg`, `α ∈ [−1, 0.5]` | `‖a−g‖/‖g‖` | sustained |
| **Phantom impulse** | impulse with no contact | `J/(m·v_typ)` | instant |
| **Super-elastic** | restitution `e > 1` | energy gain ratio | repeated |

### New families worth benchmarking

The four IntPhys 2 categories are all essentially *discrete* — something appears, vanishes,
teleports, or passes through. These add **continuously dialable** violations, which is what
a severity axis actually needs:

1. **Non-parabolic flight** — the single best category here. A projectile in free flight
   follows something that is not a `g`-parabola: constant-velocity straight line, circular
   arc, decelerating rise, or an apex that returns *upward*. Severity dials smoothly from
   "barely wrong" to "absurd" via RMS deviation from the fitted parabola. It has no
   occluder, so `t_event ≈ t_observable` — making it the **clean control** against which the
   occluded categories' latency is measured.
2. **Inertia / Newton-1** — a resting body spontaneously accelerates; a sliding body stops on
   a frictionless surface; velocity reverses with no contact.
3. **Mass–acceleration / Newton-2** — two visually identical bodies respond differently to
   the same impulse; a heavy body behaves as if massless. Severity = effective-mass ratio.
4. **Action–reaction / Newton-3** — in a collision only one body responds. Severity =
   momentum imbalance. Note the mask must cover *both* bodies: the violation is in the pair.
5. **Angular momentum** — spin spontaneously reverses; torque with no contact; a ball rolls
   uphill. Severity = angular-momentum defect.
6. **Support / static equilibrium** — an unsupported body hovers; a stack whose centre of
   mass is outside its support polygon does not topple. Severity = CoM-to-polygon distance.
7. **Friction inversion** — a body accelerates *against* its direction of motion along a
   surface, or slides forever on a high-friction one. Severity = effective µ vs declared µ.
8. **Shadow / optical inconsistency** — the shadow does not track the object, points the
   wrong way, or is missing. **The annotated region is not on the object.** This is the
   category that proves "where" annotations are being used rather than object detection
   being used as a shortcut. Implemented at render time via light linking / shadow catchers,
   which the decoupled architecture makes clean.
9. **Global gravity scale** — the *whole scene* runs at `αg`: internally consistent, but
   "toy-like" or "moon-like". The mask is the entire frame. Tests whether a model can flag a
   violation with **no localized culprit** — a case where "where" is legitimately "everywhere".

### Two control families (labelled valid; not violations)

- **Surprising-but-valid.** Rare but legal: a bounce off an `e = 0.95` surface that goes
  improbably high, a domino chain, a body coming to rest balanced on an edge, an object
  re-emerging from the *wrong* side of an occluder because it bounced offscreen. Without
  these, a model scores well by detecting *weirdness* rather than *illegality*. This is the
  control most benchmarks skip, and it is cheap for us because residuals stay ~0.
- **Artifact / shortcut probe split.** Valid clips pushed through the exact encode path the
  invalid clips use, plus deliberate motion-blur toggles, compression sweeps and render
  seams. Measures whether a detector keys on our renderer instead of on physics. IntPhys 2
  ships this idea; we copy it.

---

## Part 3 — Architecture

The one architectural choice that matters: **decouple dynamics from rendering.** Simulate,
dump a trajectory file, render from the trajectory. Violation injection then becomes a
well-defined edit at trajectory level, and valid/invalid twins go through a *bit-identical*
render path — which is exactly the appearance-matching property LikePhys and IntPhys 2 rely
on and which twin-diff labellers on those datasets fail to get.

```
scenarios/*.py          seeded scene sampler (assets, camera, lighting, initial state)
        │
        ▼
sim/                    dynamics backend — v0: PyBullet (via Kubric); later: MuJoCo
        │               injectors/ apply the intervention at t_event
        ▼
traj/<clip>.npz    ◄──  THE SEAM.  Swapping engines touches nothing downstream.
        │               per-body pos[T,3] quat[T,4] lin_vel ang_vel applied_force
        │               contacts[T]: (bodyA, bodyB, point, normal, impulse, penetration)
        │               residuals[T, n_bodies, n_laws]
        │               events: t_event, t_end, causal_body_ids, intervention params
        │
   ┌────┴────┐
   ▼         ▼
render valid  render invalid       Blender. Same seed, same code path, same everything.
   └────┬────┘
        ▼
annotate/               derive t_observable + consequence onsets, build masks and
                        severity maps, reduce to token grids, write meta.json
        ▼
clips/<uid>/            the released layout
```

**The single most valuable test in the repo:** assert that valid and invalid frames are
**bit-identical for every frame before `t_event`**. If that assert ever fails, the seed
plumbing or the render path has drifted and every downstream annotation is suspect.

### On-disk layout (this is what makes it generally usable)

```
physviol_v0/
  README.md  LICENSE  index.json          version, category list, counts, split defs
  splits/{train,val,test}.txt             grouped by scenario, not by clip
  clips/<clip_uid>/
    meta.json              everything below, human-readable — the single source of truth
    rgb_512.mp4            master, H.264 CRF 14
    rgb_256x81.mp4         derivative sized for video-DiT consumers (81 frames @ 256²)
    seg.npz                uint16 [T,H,W] instance ids
    depth.npz  flow_fwd.npz  flow_bwd.npz  normals.npz  object_coords.npz   (free from Kubric)
    violation_mask.npz     bool  [T,H,W]  — causal bodies while the violation is active
    causal_mask.npz        uint8 [T,H,W]  — 0 = none, k = consequence index k
    severity_map.npz       f16   [T,H,W]  — residual painted into the culprit's mask
    divergence_map.npz     f16   [T,H,W]  — NOT the violation region; labelled as such
    grids.npz              masks + severity reduced to 21×16×16 and T×32×32
    traj.npz  residuals.npz   the seam file and the raw law residuals, shipped
```

The `meta.json` field reference is maintained separately in [schema.md](schema.md) so it can
be versioned as the schema evolves. `real2sim` is `null` in v0 and reserved for Phase 4;
reserving it now means Phase 4 does not force a regeneration.

**Token-grid reduction.** `grids.npz` pre-reduces masks and severity to **21×16×16** — the
latent token grid a WAN-2.1-class video DiT yields for an 81-frame 256² clip, where the VAE
bins time 4× (81 → 21) and space 16× (256 → 16). Pre-reducing means a consumer never has to
re-derive the binning. The layout is **time-major** (`[F_lat, H_lat, W_lat]`, latent frame
slowest), which is a schema guarantee, not an implementation detail. Masks reduce by **max**
(a violation in any source frame marks the latent frame); severity is stored **both** max
and mean, since peak and average are different questions. A `T×32×32` reduction ships
alongside for consumers on a different tokenizer.

### Repo skeleton

```
physviol/
  pyproject.toml   requirements.txt
  docker/          kubric.sh  IMAGE_DIGEST   [Dockerfile — only if we build our own image]
  scripts/         fetch_refs.sh
  refs/            gitignored reference checkouts (kubric)
  context/         source papers, gitignored; see context/README.md
  physviol/
    scenarios/     __init__.py  rigid_ramp.py  occluder_pass.py  toss.py  stack.py  …
    sim/           backend.py (ABC)  pybullet_backend.py  [mujoco_backend.py — Phase 3]
    injectors/     base.py  permanence.py  solidity.py  continuity.py  dynamics.py
                   parabola.py  newton.py  support.py  friction.py  shadow.py  global_g.py
    residuals/     laws.py  noise_floor.py
    render/        worker_smoke.py  blender_render.py  passes.py  derivatives.py
    annotate/      observability.py  masks.py  severity.py  grids.py  meta.py
    controls/      surprising_valid.py  artifact_probe.py
    schema/        meta.schema.json  validate.py
    viz/           overlay.py  contact_sheet.py        ← built in Phase 0, used forever
    cli.py         generate | render | annotate | validate | report
  tests/           test_prefix_identity.py  test_schema.py  test_residual_zero_on_valid.py
  docs/            PLAN.md  schema.md  prior_art.md
```

---

## Part 4 — Phasing

**Phase 0 — two clips (days 1-3).** One scenario, one violation, one valid twin. Kubric
docker smoke render → full annotation export → an overlay video with the mask, the severity
map and all three clocks burned into the frame. **Look at it before anything scales.**
Assert prefix identity is exactly zero. Measure render throughput and settle the image
question. Write [prior_art.md](prior_art.md) so Part 2's grounding claim is auditable.
Nothing else starts until these two clips are visually correct.

**Phase 1 — ~50 clips (weeks 1-2).** Three scenarios, four v0 families, schema frozen and
schema-validated. Then **immediately** run 2-3 off-the-shelf models — a V-JEPA-2 surprise
baseline, a VLM prompt, and a video-DiT probe. If they sit at 95% or at chance, the
difficulty calibration is wrong and it must be fixed *now*, not after 600 clips.

**Phase 2 — ~300 pairs / 600 clips (weeks 3-8).** 10-14 scenarios, the full v0 taxonomy
including the new families, both control families, severity-binned easy/medium/hard splits,
scenario-grouped train/val/test.

**Phase 3 — realism + MuJoCo.** Swap the dynamics backend behind the seam for exact
residuals and deformables; upgrade to the cluttered-scene, moving-camera, dynamic-lighting
tier as the end state. Optionally an Isaac Sim / UE photoreal subset.

**Phase 4 — real→sim.** Own props, photogrammetry, calibrated multi-view capture, system ID
(mass/friction/restitution by differentiable sim or CMA-ES), re-render at the true camera
pose, and report the **twin gap** (reprojection error, LPIPS) as a dataset property. A small
physically-staged *real-invalid* set (hidden magnet, thread, false bottom) makes the
benchmark much harder to dismiss as synthetic-only.

**Phase 5 — release.** Human study (IntPhys 2's argument rests on humans near ceiling while
models sit at chance), baseline table, asset license audit, HF dataset card.

---

## Part 5 — Consumer contract

PhysViol has no privileged consumer. The published on-disk layout **is** the API, and this
section is what a downstream project needs in order to use it. Nothing in `physviol/`
imports a consumer; no consumer needs to import `physviol`.

**What a consumer reads:** `index.json` for version, category list and counts;
`splits/{train,val,test}.txt` for the scenario-grouped split; then per clip, `meta.json`
plus whichever arrays it needs. A clip-level classifier needs only the mp4 and
`meta.json:label`. A temporal model adds `violation.t_observable_frame`. A spatial model
adds `grids.npz` or the full-resolution masks.

**A typical loader mapping.** Most video-benchmark loaders converge on the same record, and
`meta.json` is arranged to fill one without computation:

| record field | source |
|---|---|
| `uid` | `clip_uid` |
| `video_path` | `rgb_256x81.mp4` or `rgb_512.mp4` |
| `label` | `label` — valid / invalid |
| `task` | `category` |
| `scene_id` | `scenario` + `seed` — the **leakage-free grouping key**; never split within a pair |
| `pair_id` | `pair_uid` |
| `variant` | `category` + intervention type |
| `twin_path` | the valid twin's video |
| extras | `t_event_s`, `t_observable_s`, severity, path to `grids.npz` |

A consumer may set its equivalent of `pixel_aligned_twin = True` — and here that flag is
**earned**, not asserted: `provenance.prefix_identical_verified` records that the
prefix-identity test passed for that pair, with the frame it held to.

**Two evaluation protocols this dataset unlocks**, neither of which is possible on
whole-clip-labelled benchmarks:

1. **Onset-labeller calibration.** Twin-diff and object-centric pseudo-onset heuristics can
   finally be *scored*, against a known `t_observable`, on data where the twins really are
   pixel-aligned. That converts "this heuristic covers 46% of clips" from a caveat into a
   measured error bar, and it is the direct answer to the failure described in Context.
2. **Spatial supervision.** `grids.npz` is a drop-in target for a per-token spatial head:
   for a model exposing hidden states as `[B, S, D]` with `S = F_lat · H_lat · W_lat`,
   dropping the usual spatial mean-pool and reshaping to `[B, F_lat, H_lat, W_lat, D]`
   aligns directly with the stored grid, given the time-major guarantee in Part 3. The
   honest caveat: retaining full spatial features costs on the order of **hundreds of MB per
   video per denoising step**, so a consumer must hold a *subset of layers*, not all of them.

---

## Part 6 — Risks

1. **Render throughput is the schedule risk, not simulation — now measured, not feared.**
   The published image bundles Blender 2.93 and renders Cycles on **CPU**; on this box's 8
   cores that is **~7.2 s/frame at 512²**, i.e. ~116 h for 600 clips × 96 frames, on a
   two-object scene that is far simpler than anything we will ship (Part 0). The cheap dial
   does not work: `samples_per_pixel` 64 → 16 saves only 22%, because the cost is in the
   auxiliary passes and I/O. So the mitigation is the GPU image — build from upstream's
   `refs/kubric/docker/Blender.Dockerfile` with a modern Blender LTS plus OptiX, keeping
   Kubric's exporter logic; the docker `nvidia` runtime is confirmed present. **Schedule it
   before Phase 2.** Phases 0-1 are affordable on CPU as-is.
2. **The L40S is shared** with other jobs on this box. Render in a resumable,
   per-clip-checkpointed background queue; never hold the GPU in one long job.
3. **Determinism across versions.** PyBullet is deterministic for a fixed build and thread
   count, not across versions. Pin the docker digest, ship `traj.npz` so renders reproduce
   without re-simulating, and let the prefix-identity assert catch any drift.
4. **Severity noise floor.** Residuals are nonzero on valid clips. Calibrate per scenario;
   report both physical units and z-score.
5. **Domain gap.** Randomize aggressively (shape, material, HDRI, camera, counts, initial
   velocities, occluder geometry, friction); report transfer onto IntPhys 2 / LikePhys as a
   headline number; Phase 3 realism tier.
6. **Shortcut leakage** — handled by the artifact/debug split.
7. **Asset licensing.** The schema promises a license string per asset and Phase 5 promises
   an audit; MOVi's asset mix is not uniformly CC0. Record the license as each source is
   enabled, not at release time.
8. **Positioning.** PhyGround and PhyCheck are the nearest neighbours and are moving fast,
   but both are QA-style and human-annotated. Our differentiators — simulator-derived
   pixel-precise ground truth, a continuous severity residual instead of a label, and the
   causal-onset vs observability distinction — should be written up as a positioning
   paragraph in **week one**, because that paragraph also tells us if the gap has closed.

---

## Verification

Ordered by when it runs; the first three catch real bugs.

- **`tests/test_prefix_identity.py`** — for every pair, `max|valid − invalid| == 0` for all
  frames `< t_event`. Fails loudly; blocks the release build.
- **`tests/test_residual_zero_on_valid.py`** — every law residual on valid clips stays below
  the calibrated noise floor. Catches injector leakage into the control arm.
- **Visual check, every phase, before scaling.** `physviol viz overlay <clip>` burns the
  violation mask, the severity heatmap and the three clocks onto the RGB and writes an mp4;
  `contact_sheet` builds an HTML gallery. Never trust a metric built on an unviewed label.
- **`physviol validate`** — jsonschema over every `meta.json`, plus cross-checks:
  `t_event ≤ t_observable ≤ t_end`, mask non-empty while active, `causal_body_ids` present in
  `seg.npz`, every asset carries a license string, `prefix_identical_verified` is true.
- **Difficulty calibration (end of Phase 1)** — V-JEPA-2 surprise, a VLM prompt, and a
  video-DiT probe on the 50-clip set. Target: meaningfully above chance, well below ceiling.
- **Round trip through an external consumer** — build the Part 5 loader mapping against the
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
