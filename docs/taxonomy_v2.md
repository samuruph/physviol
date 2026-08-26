# Reorganising the taxonomy around physical laws

**Status: proposal, not built.** Nothing here is implemented. It exists to be argued
with. `physviol/taxonomy.py` is unchanged and still ships eight `DOMAINS`.

## The complaint, and why it is right

Every clip currently carries five different labels for "what kind of violation is this":

| label | values | assigned by |
|---|---|---|
| `medium` | 5 (rigid, granular, optical, fluid, continuum) | the scenario |
| `domain` | 8 (identity, kinematics, contact, dynamics, equilibrium, optical, appearance, global) | by hand, on the family |
| `family` | 23 | the injector |
| `law` | 16 | by hand, on the family — picks the residual |
| `scenario` | 15 | the staging |

Three of those are load-bearing. `domain` is not, and it is actively misleading, because it
is **not a law axis** — it cuts across the laws in three separate places:

- `linear_momentum` is the law for **four** families spread over **three** domains:
  `newton1_inertia` (kinematics), `newton3_reaction` (contact), `newton2_mass` and
  `phantom_impulse` (dynamics). Same residual, same principle, three different labels.
- `free_fall` is the law for `antigravity` (kinematics) *and* `global_gravity` (global).
  Same residual. The only real difference between those two families is **how much of the
  scene is affected** — which is not what "domain" is supposed to mean, and which is
  already recorded separately as `violation.spatial_extent`.
- `shape_anisotropy` is the law for `deformation` (appearance) *and* `shadow_shape`
  (optical).

So `domain` encodes a blend of three things — which law, how far it reaches, and whether the
culprit is matter or light — and it is hand-typed into every `Family`, which means it can
silently disagree with the residual the family actually uses. That is the "too many
categories" problem: not that there are five labels, but that one of them is a muddle.

## What LikePhys does, and where we already differ

LikePhys Table 5 groups its scenarios under seven principles: Temporal Continuity, Spatial
Continuity, Conservation of Energy, Conservation of Mass, Geometric Invariance, Optical
Consistency, Material Response.

The important difference is that for LikePhys the principle is **the only** structure — each
cell is (principle, scenario, variation key) and the variation key is a name, not a number.
We already have a genuine law axis underneath: `Family.law` names a residual in
`residuals/laws.py` that returns a real per-body per-frame number. So we do not need to adopt
their scheme; we need to *expose the one we already have* and throw away the layer that is
duplicating it badly.

## Proposal: three axes, one of them derived

**Drop `DOMAINS` entirely.** Replace it with:

### Axis 1 — PRINCIPLE (the new Level 1)

Which conservation law or invariance is broken. **Derived from the law, never hand-assigned**:
a single `LAW_PRINCIPLE: Dict[str, str]` maps each residual to exactly one principle, and
`principle_of(family) == LAW_PRINCIPLE[FAMILIES[family].law]`. Nothing can drift, because
there is nothing to type twice.

| principle | families | cells | LikePhys row |
|---|---|---|---|
| **Spatio-temporal continuity** | continuity, non_parabolic | 21 | Temporal + Spatial Continuity |
| **Conservation of mass** | permanence, dissolve, fission, fusion, immutability | 52 | Conservation of Mass |
| **Conservation of momentum and energy** | phantom_impulse, newton1_inertia, newton2_mass, angular_momentum, superelastic | 35 | Conservation of Energy |
| **Gravitation** | antigravity, global_gravity | 13 | *(none — they file hovering under Material Response)* |
| **Solidity and support** | solidity, support | 19 | Spatial Continuity "invisible wall" |
| **Material response** | deformation, friction | 20 | Material Response |
| **Optical consistency** | shadow, shadow_inverted, shadow_shape, colour_shift | 17 | Optical Consistency |
| | **23 families** | **177** | |

Seven principles, matching LikePhys's seven. Every family lands in exactly one, every law
lands in exactly one principle, and the cells sum to the current 177 — this is a
*re-labelling*, not a change to what gets generated.

#### Why `immutability` is a mass violation, not a geometric one

Because the two families that change a body's shape change **different quantities**, and the
code already enforces the difference:

- `immutability` scales a body **uniformly**. Its `magnitude_unit` is literally
  `volume_ratio`, and volume at constant density *is* mass — the same quantity `permanence`
  reports as `mass_ratio`. A body that doubles in size has twice the matter in it.
- `deformation` scales it **non-uniformly and volume-preservingly**: one axis by `k`, the
  other two by `1/sqrt(k)` (`injectors/appearance.py`). No matter is created or destroyed.
  What is violated is that a *rigid* body held its proportions — a material property.

So the split is not a naming preference, it is a measurable one: does the volume change, or
only the proportions? `EXCLUSIVE_LAWS` already asserts exactly this — `shape_continuity`
(volume) belongs to `immutability` alone and `deformation` must leave it at zero, which
`tests/test_orthogonality.py` checks on every cell. The principle assignment is just reading
back what the residuals already say.

#### Why momentum and energy merge, and what is kept underneath

They are genuinely different conserved quantities — momentum is conserved in *every*
collision, kinetic energy only in elastic ones, and our families do dissociate them
(`superelastic` conserves momentum and creates energy; `newton1_inertia` destroys momentum).
But that distinction survives one level down, where reporting already happens: the laws
`linear_momentum`, `angular_momentum` and `energy_at_contact` stay separate, and per-family
tables still separate them. Merging costs nothing and it removes a headline split that reads
as two names for the same idea.

`friction` moves **out** of energy and into Material response, where LikePhys puts Block Slide
too. Its residual is `effective_mu_ratio` — a surface property mismatch, not an energy budget.
Half its severity range ("slides forever", µ = 0) creates no energy at all.

#### Why `solidity` pairs with `support`

The two are exact duals, which is why neither needs a principle to itself:

- `solidity` — the contact constraint is **absent where it should be present**: two bodies
  share space.
- `support` — the contact constraint is **present where it should be absent**: a body is held
  up by nothing.

Both are the statement "surfaces interact when and only when they meet". Filing `solidity`
under continuity instead (LikePhys's choice for "invisible wall") would be wrong for us: a
ball passing through a wall traces a perfectly continuous path, so `position_continuity`
reads zero and the principle would not describe the residual.

That also leaves **Gravitation** as exactly what it says — unsupported matter accelerates at
g — with `antigravity` and `global_gravity` as the same law at two extents (see Axis 2).

### Axis 2 — EXTENT

How much of the scene is wrong: `local` (one body) / `group` (a medium, or a declared set) /
`global` (every dynamic body). Already half-present as `violation.spatial_extent`, currently
only ever `"global"` for `global_gravity` and `"local"` otherwise.

Promoting it is what makes `antigravity` and `global_gravity` describable as *the same
principle at two extents*, which is the honest description and the structural answer to the
earlier complaint that they were indistinguishable. It also gives the granular families a
place to say "this acted on the whole medium, not one grain".

### Axis 3 — MEDIUM

Unchanged: rigid / granular / optical / (fluid, continuum — declared empty). This is the
axis that lines up with LikePhys's four macro-categories and answers a capability question
("does the model handle deformables at all?") rather than a law question.

### What a clip is labelled with afterwards

`medium × principle × extent × family × scenario × severity × complexity`, where **principle
is derived from law and law is derived from family**, so there are three hand-assigned
strings on the violation side instead of four, and the redundant one is gone.

## One residual has to change

`shape_anisotropy` is currently the law for both `deformation` and `shadow_shape`, so under
any law→principle map those two are forced into the same principle, which is wrong.

The fix is worth making on its own merits: `shadow_shape` should be scored **against its
caster**, not in absolute terms. Today the residual measures the shadow's own aspect ratio,
so a scene whose lawful shadow is legitimately elongated (a low sun) starts with a non-zero
baseline. A new `shadow_shape_match` law — the shadow's aspect ratio divided by the caster's
projected aspect ratio, minus one — is zero on every lawful clip regardless of the light, and
puts `shadow_shape` cleanly under Optical consistency.

## What changes, and what it costs

| file | change |
|---|---|
| `physviol/taxonomy.py` | `PRINCIPLES` + `LAW_PRINCIPLE`; `Family` drops `domain`, gains `extent`; `domain_of()` → `principle_of()`; `validate_taxonomy` checks the map is total and single-valued |
| `physviol/residuals/laws.py` | new `shadow_shape_match` |
| `physviol/injectors/appearance.py` | `ShadowShape.law` points at it |
| `physviol/annotate/pipeline.py` | `meta["violation"]["domain"]` → `principle`; `spatial_extent` reads the family's declared extent |
| `docs/schema.md` | the two renamed keys |
| `docs/prior_art.md` | rewrite the coverage table principle-by-principle against LikePhys Table 5 — this becomes a like-for-like comparison rather than a family-by-family one |
| `docs/evaluation.md` | per-principle reporting as the headline table, per-family underneath |
| `physviol/cli.py`, `physviol/viz/grid.py` | group by principle |
| `tests/test_taxonomy.py` | assert `LAW_PRINCIPLE` is total over the laws families use, and that each principle is non-empty |

**Cost: no re-rendering.** `physviol annotate` runs host-side over an existing worker dir, so
an already-generated release is re-labelled by re-running annotation, not by re-simulating.
The renders are the expensive part and they do not change at all.

## Open questions — what is left to decide

1. **Is `Gravitation` too small at 2 families / 13 cells?** It could fold into Solidity and
   support ("what holds matter up, and what lets it through"). Against: free fall is the most
   tested intuition in the infant-cognition literature and the only principle with a
   *continuous* dial that is a physical constant.
2. **Does `colour_shift` belong under Optical consistency?** It is not about light at all —
   the object simply stops being the colour it was. The alternative is an eighth principle,
   *Appearance / identity of surface*, which would also be where texture and material-swap
   families land later. Optical consistency is the pragmatic home for now.
3. **Is `extent` worth promoting to a top-level axis**, or should it stay a field on the
   violation record? Only two of its three values are populated today.
4. **Keep `domain` as a deprecated alias for one release**, or delete it? Nothing has been
   published, so deleting outright is defensible and much cleaner.
