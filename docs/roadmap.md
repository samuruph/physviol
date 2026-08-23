# Roadmap — macro-categories, perception, and the realistic twin

Three questions came out of reviewing the v0 clips, and they have different answers.

---

## 1. Macro-categories like LikePhys — yes, and they cost almost nothing

LikePhys groups by **medium**: Rigid Body, Optical, Fluid, Continuum. PhysViol groups by
**law**: identity, kinematics, contact, dynamics, equilibrium, optical, appearance, global.

These are not competing schemes. They are two independent groupings of the same cells, and
that is exactly what makes them worth having both of:

- **medium** answers *what kind of matter is misbehaving* — the axis that maps onto prior art
  and onto a capability question ("does this model handle deformables at all?")
- **domain** answers *which principle is broken* — the axis severity is measured against,
  since the domain determines the residual

So the taxonomy grows a level at the top rather than being reorganised:

```
Level 0  MEDIUM    what kind of matter        (5)   <- new, maps to LikePhys
Level 1  DOMAIN    which law                  (8)
Level 2  FAMILY    how it breaks             (20)
Level 3  SCENARIO  the staged scene          (15)
Level 4  INSTANCE  scenario x family x seed x severity
```

| medium | scenarios | status |
|---|---|---|
| `rigid` | 12 of the 15 | built |
| `granular` | `granular_pour` | built — the fluid stand-in, never labelled fluid |
| `optical` | `shadow_track` | built |
| `fluid` | — | **Phase 3**: needs a Blender with working headless Mantaflow |
| `continuum` | — | **Phase 3**: cloth and soft bodies, needs MuJoCo/MJX behind the seam |

`physics_medium` already exists on every scenario and is already cross-checked by
`physviol validate`; today it only takes `rigid` and `granular`. Promoting it means widening
the enum, tagging `shadow_track` as `optical`, and reporting on it. **Half a day**, and it
makes head-to-head comparison with LikePhys a table rather than an argument.

The two empty rows are the honest part: declaring `fluid` and `continuum` as known-empty is
more useful than omitting them, because it says what the dataset does *not* cover without a
reader having to infer it from absence.

---

## 2. Perceptual violations — the half of v0 that is still missing

`deformation` and `shadow_shape` landed. Two more belong in v0, and both are mechanically
cheap because the render path already animates what they need.

### `colour_shift` — the object changes colour

**Verified working in the pinned image.** `material.keyframe_insert("color", frame)` creates
one fcurve per RGBA channel on the Principled BSDF's Base Colour, and the rendered pixels
follow. Needs a `Trajectory.colour[T,B,3]` channel, keyframed in `replay` exactly the way
`scale_mul` already is.

Bins by perceptual distance: a hue nudge, a clear shift, a jump to the complementary colour.
Law: `colour_continuity`, the CIE-Lab distance from the body's own frame 0 — Lab rather than
RGB so "how different does this look" is measured the way a viewer would rank it.

Ramped, not switched, for the same reason `immutability` is: a colour that changes between
two frames is a cut, one that visibly shifts is a violation.

### `illumination_shift` — the light changes with nothing to change it

Light `color` and `intensity` are both keyframable traits. The shadows and shading follow
automatically, which is what makes it a *scene-wide* perceptual violation rather than an
object one — the counterpart to `global_gravity` in the appearance domain.

Needs a `LightSpec` animation channel. Note it necessarily co-moves with `shadow`, since
moving the light moves the shadow, so the two must not be built on the same scenario.

### Deliberately not in v0

| | why |
|---|---|
| `texture_swap` | needs GSO assets, which arrive with the complexity ladder in v1 |
| `reflection` | no scenario has a reflective surface, and adding one is a lighting project |
| `transparency` | **measured and rejected** — see `render/probe_opacity.py`. Neither material transmission nor the Principled BSDF's alpha moves the rendered pixels, and the segmentation pass reports the body at full size at alpha 0 regardless, so no mask could see it |

---

## 3. v1 — the realistic twin, and randomisation

The v0 goal is *coverage*: every meaningful law × scene combination, staged as plainly as
possible. v1's goal is *variety*, and it is three separate pieces of work.

### 3a. Population — single vs multi

The remaining structural piece from the original plan. A `--population` axis putting 3–6
independent movers in the existing scenarios, and multi-culprit annotation with per-body
windows so a violation can hit a random subset at staggered times.

The design is written up already (`Culprit` records, an additive `[T,N]` body axis beside the
existing arrays, three new validator checks, a five-step migration that is green at every
step). **The largest single item on this list**, and the one that makes `antigravity` and
`global_gravity` differ in *every* scenario rather than only where several bodies already
move.

### 3b. The realistic twin

Not new scenarios — **the same 15 scenarios re-rendered at higher complexity**, so every clip
has a plain counterpart and a realistic one. The complexity ladder (`L0`..`L4`, mirroring
MOVi-A..F) is already scaffolded; `L2`–`L4` add GSO objects, distractors and camera motion,
and currently raise rather than silently degrade.

That pairing is the point: it isolates "understands physics" from "copes with clutter",
because the physics is bit-identical across the pair and only the scene changed.

**The cost is the thing to decide, not the code.** L1 already costs ~5.5× L0 per render, and
L2–L4 add GSO fetches and more geometry on top. A full-ladder release at L1 is a week of
rendering where the same release at L0 is an overnight job. The recommendation is to build
the code, then choose the tier and complexity per release rather than committing now.

### 3c. Randomisation depth

Today one seed varies sizes, speeds, restitution, colours, camera, HDRI, and — where
physically neutral — actor shape. v1 should add:

- **object identity** — draw actors from GSO's 1033 scanned objects rather than five
  primitives (needs the `kubasic:`/GSO kind support, half-built already)
- **backgrounds** — 509 HDRI environments are reachable; currently one is picked per clip,
  and the set is curated down to 44 for determinism
- **layout** — spawn positions, camera framing and lighting direction sampled more widely,
  with the frustum fit already in place to keep the culprit in shot

All three are per-seed, so `--variants N` picks them up without further work.

---

## Suggested order

1. **`colour_shift`** — the largest gap in v0's coverage, and the render path is proven
2. **Medium as Level 0** — half a day, and it settles the prior-art comparison
3. **`illumination_shift`** — completes the appearance domain
4. **Population + multi-culprit** — the big structural piece
5. **Complexity ladder L2–L4** — the realistic twin
6. **Randomisation depth** — mostly falls out of 5

1–3 are v0 completion. 4–6 are v1.
