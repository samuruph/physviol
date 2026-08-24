# Settle before the full v0 generation run

Things that change the **published format** and are therefore cheap now and expensive
later. None of them require re-rendering — `physviol annotate` re-labels an existing worker
dir host-side — but all of them change what a consumer of the release reads.

## 1. Split `violation_mask` into valid and invalid halves

**Raised by the user, 2026-08-24, after reviewing the coverage video.**

Today `violation_mask[t]` is the **union over both twins**:

```
violation_mask[t] = footprint(culprit, invalid, t) ∪ footprint(culprit, valid, t)
```

That rule is a CLAUDE.md non-negotiable and it exists for a real reason: without it, `vanish`
and `teleport` violations produce empty or half-empty masks, because the body has no pixels
in the invalid render *precisely because it vanished*. `tests/test_mask_union.py` guards it.

The user's objection is also real: when you are evaluating **what is wrong**, the union
includes pixels where the object lawfully *is*, and a model should not be scored for
attending to those. "We should just focus on that area."

Both are right, and the resolution is probably that the union was doing two jobs. The likely
shape of the fix — to be agreed, not assumed:

| array | meaning |
|---|---|
| `violation_mask` | unchanged — the union. Stays the documented training target, stays a non-negotiable |
| `mask_invalid` | the culprit's footprint in the **invalid** render only — where the wrong thing is |
| `mask_valid` | the culprit's footprint in the **valid** render only — already shipped as `reference_mask` |

Note `reference_mask` already *is* `mask_valid`. So this may be a rename plus one new array
plus a doc paragraph, rather than a change of semantics. Check that before designing
anything.

**Decide before the release run**, because the answer changes `docs/schema.md` and every
loader written against it.

## 2. Does `causal_mask` earn its place?

**Raised by the user in the same message; explicitly deferred by them.**

Open question, not yet analysed: what `causal_mask` is for, whether consumers can use it, and
whether it is distinct enough from `violation_mask` and `reference_mask` to be worth the
array. Review it against real clips from the review sweep rather than in the abstract.

## 3. Scenarios go static long before the clip ends

**Found 2026-08-24 while building `time_slip`, which could not find a window to stall in.**

The tier lengthening (13 -> 25 frames at the debug tier, 25 -> 49 at tier v0) broke time-scaling the
same way it broke framing, and only the framing half has been fixed. Fraction of the clip
with any actor in motion, worst of five seeds (host-side mock, so indicative not exact):

| scenario | D | A | B |
|---|---|---|---|
| ramp_slide | 24% | 12% | 12% |
| barrier_pass | 100% | 24% | 25% |
| pour | 48% | 27% | 31% |
| stack_topple | 52% | 27% | 28% |
| collision | 52% | 33% | 34% |
| rolling_ramp | 68% | 35% | 37% |
| pyramid_impact | 80% | 41% | 46% |
| drop | 92% | 57% | 61% |
| *(occluder_pass, shadow_track, toss, tumble, pendulum_swing)* | 100% | 92-100% | 92-100% |

`resting_table` is 0% by design and is not in the table.

Two different causes, needing two different fixes:

- **Traversal scenarios** (`collision`, `barrier_pass`, `occluder_pass`). Deriving the launch
  speed from the clip length made the bodies slower, and with friction unchanged a slower
  body stops sooner in *both* time and distance. Friction has to scale with the scene too:
  `mu_eff = (1 - remaining) * v0 / (g * T)`. In `collision` at tier v0 the striker now stops
  before it reaches the target, so the staged collision does not happen at all.
- **Fixed-duration events** (`ramp_slide`, `rolling_ramp`, `stack_topple`, `pyramid_impact`,
  `drop`). The slab length and the drop height are fixed, so the event takes the same ~1 s
  whatever the clip length. Either the scene scales with duration the way
  `camera.frame_flight` does for `toss`, or the event has to be slowed by moving friction
  closer to `tan(tilt)` -- which is a marginal balance and fragile in the simulator. Probably
  the former.

**This is the direct blocker on three `time_slip` cells** (`collision`, `ramp_slide`,
`barrier_pass` at the longer tiers): the injector declines to stage a stall when the body has
no motion left to resume into, which is correct behaviour and a symptom, not a bug in it.

Not fixed yet, deliberately: `tests/mockroll.py`'s friction is a recent local approximation,
and tuning seven scenarios against it before checking against the real simulator would be
tuning against the wrong numbers. The review sweep at the debug tier -- where most scenarios are
above 50% -- is the run that produces those numbers.

### Known-tight cell: `occluder_pass x time_slip` at the debug tier, strong

The stall is capped so the body re-emerges before the clip ends, using the scenario's
declared `occluded_frames`. That test is for *full* occlusion, so partial visibility starts a
frame or two later than it predicts, and at the debug tier the strong bin re-emerges only in the last
frame or two -- close to reading as `permanence`. Comfortable at tier v0 (4 visible frames
after emerging) and tier v1 (11), which are the tiers that ship. Fixing it properly means
either estimating emergence from the screen geometry rather than from `occluded_frames`, or
giving `occluder_pass` more runway to the right of its screen.

## 4. Release configuration

Severity bins and variant count per cell. `configs/v0_release.yaml` currently proposes tier v0
/ L0 / all three bins / 3 variants; `physviol taxonomy --config v0_release` prices it. The
user asked to settle this together before the run.

## 5. Taxonomy v2

`docs/taxonomy_v2.md` — replacing `domain` with a derived `principle`. Agreed in outline,
not implemented. Re-labelling only, no re-rendering.
