# Evaluating on PhysViol

*What to measure, on which axes, and what the numbers do and do not support.*

The dataset is built so that a failure can be attributed. That is a stronger claim than
"it contains physics violations", and it is what decides how evaluation should be
structured: **one axis says what broke, and every other axis is orthogonal to it.**

---

## 1. The primary axis — what law broke

**Report per family (20), aggregate to domain (8).** This is the axis the dataset exists for.

| domain | the question it tests | families |
|---|---|---|
| `identity` | does the object persist and stay itself? | `permanence`, `immutability`, `fission`, `fusion` |
| `kinematics` | is unsupported motion consistent with `g`? | `continuity`, `non_parabolic`, `antigravity`, `newton1_inertia` |
| `contact` | do bodies interact legally when they touch? | `solidity`, `superelastic`, `newton3_reaction` |
| `dynamics` | do forces and masses behave? | `phantom_impulse`, `newton2_mass`, `angular_momentum` |
| `equilibrium` | do resting and supported bodies behave? | `support`, `friction` |
| `optical` | is light consistent with geometry? | `shadow`, `shadow_shape` |
| `appearance` | does it look like itself from frame to frame? | `deformation` |
| `global` | are the scene's constants physical? | `global_gravity` |

**Do not pool across families and quote one number.** Cell counts are uneven by an order of
magnitude — `identity` has 21 build cells, `optical` has 2 — so a pooled score is mostly a
measurement of `identity`. Report the per-family vector; aggregate to domain if you need one
figure per axis.

### What "orthogonal" is guaranteed to mean here

Eight families have an **exclusive tripwire**: a residual with a clean zero baseline on a
lawful clip that only they are permitted to move. `taxonomy.EXCLUSIVE_LAWS` is the contract
and `tests/test_orthogonality.py` enforces it in both directions — no other family may move
the law, and the owner must actually move it.

| tripwire | owned by |
|---|---|
| `penetration` | `solidity` |
| `position_continuity` | `continuity` |
| `mass_continuity` | `permanence` |
| `object_count` | `fission`, `fusion`, `permanence` |
| `shape_continuity` | `immutability` |
| `shape_anisotropy` | `deformation`, `shadow_shape` |

For those eight, a clip labelled X provably contains X and not the others. **The remaining
twelve are separated by staging, not by residual**, and the distinction is worth
understanding before you draw conclusions from them. `antigravity`, `phantom_impulse`,
`newton1_inertia` and `newton3_reaction` all move `linear_momentum` or `free_fall`, because
they must: bend a body's gravity and its momentum residual moves too. Physics is not
separable there, and demanding that it be would be demanding the wrong thing. What separates
them is the situation — is anything touching the body, is there one body or two, was it
moving or at rest — which a model must read from the image.

So: **a confusion matrix over families is a meaningful thing to compute, and confusions
*within* those four are not evidence of a broken model.** Expect and report them.

---

## 2. Secondary axes — each independent of the family

Every one of these can be crossed with the family axis, which is the point of them.

| axis | values | what it isolates |
|---|---|---|
| **severity** | `weak` / `medium` / `strong` | How hard the law is bent. Exact by construction — it is the knob, known before simulating, not a measurement. Use it as a difficulty curve: accuracy should rise monotonically, and a model that is flat across it is not detecting the physics. |
| **complexity** | `L0` / `L1` | How hard the *scene* is to parse, with the physics held fixed. L0 is a solid background, L1 a photographic HDRI environment. Separates "understands physics" from "copes with clutter". |
| **observability lag** | 0, or 1–2 frames | `t_observable − t_event`: how long the violation is real before it is visible. Non-zero only where something occludes the culprit. This is the axis for "can the model reason about what it cannot currently see". |
| **population** | `single` / `multi` | How many independent movers. *Not yet built* — see the roadmap. |
| **physics medium** | `rigid` / `granular` | `pour` only. Never `fluid` at schema v0. |

**Split by `scene_id`, never by clip.** A scenario+seed produces one valid render and several
invalid twins that share it frame-for-frame before `t_event`. Putting one twin in train and
another in test leaks almost the entire clip.

---

## 3. The four tasks, and what each needs

### a. Detection — is this clip lawful?

Paired: every invalid clip ships with its valid twin, bit-identical before `t_event`. So
**chance is 50%, and the right metric is paired accuracy** — the model sees both and says
which is wrong. Reporting unpaired accuracy on a set that is mostly invalid clips overstates
performance and is not comparable to IntPhys 2 or LikePhys, which both pair.

### b. Temporal localisation — when?

Ground truth is `violation_windows` (a *list* of intervals; `superelastic` fires once per
bounce). Metrics: onset error `|t̂_event − t_event|` in frames, and window IoU against the
rasterised `active[T]`.

Score against **`t_event`**, not `t_observable`. The gap between them is the thing being
measured, and a model that reports the frame the evidence appears is doing the right thing
under a lag — which the two clocks let you say explicitly rather than penalising silently.

### c. Spatial localisation — where?

Ground truth is `violation_mask[T,H,W]`. Metric: per-frame IoU, averaged over **frames where
the mask is non-empty**. The mask is gated on `active AND observable`, so frames where the
culprit is hidden are legitimately empty and averaging over all frames rewards predicting
nothing.

`grids.npz` is the same thing reduced to a video-DiT token grid, if you are attaching a head
to latents rather than pixels.

### d. Severity regression — how badly?

Ground truth is `severity_t[T]` (a curve) or `severity_map[T,H,W]` (a field). Both are
bounded `[0,1]` and comparable across families by construction — that is what the noise floor
and the per-family `r_strong` reference are for. Spearman correlation against `severity_t` is
the least assumption-laden metric; MAE is fine if you also report the family breakdown, since
the bins are not equally spaced in every family's physical units.

---

## 4. Five things that will bite you

1. **Never train on `divergence_map`.** It is `|valid − invalid|` in pixel space and it
   diverges *everywhere* downstream of the event, including on shadows and secondary
   collisions. It is shipped for analysis and labelled as not-ground-truth. The targets are
   `violation_mask` and `severity_map`.
2. **`violation_mask` is not `timelines.active`.** The mask answers *where can this be seen*
   and is empty while the culprit is hidden; `active` is the unhedged truth about when the
   law is broken. Use `active` for temporal metrics and the mask for spatial ones.
3. **Under-powered families.** `shadow`, `shadow_shape`, `newton2_mass` and `newton3_reaction`
   have one build cell each, because each needs staging almost nothing else provides. Report
   them; do not read much into a single cell's score.
4. **`stack_topple` is a control, not just a scenario.** Its *valid* clip contains a tower
   falling over. A model that flags every clip where something dramatic happens will score
   well everywhere else and fail here, which is the point of including it.
5. **Check `provenance.prefix_identical_verified`** rather than assuming twins are
   pixel-aligned. It is asserted at generation time, and a clip that failed would be a clip
   whose annotations are all suspect.

---

## 5. A reporting template

```
per family (20 rows):     detection paired-acc | onset MAE | mask IoU | severity rho
per domain (8 rows):      the same, aggregated
severity curve:           accuracy at weak / medium / strong, per domain
complexity delta:         L1 accuracy minus L0 accuracy, per domain
lag breakdown:            accuracy at lag 0 vs lag > 0
confusion matrix:         20 x 20 over families, with the four momentum-sharing
                          families read together rather than as separate errors
```

The two numbers worth leading with are **the per-domain vector** and **the severity curve**.
One says which laws a model does not know; the other says whether it knows them at all or is
picking up on how big the pixel difference is.
