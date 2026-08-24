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
a single `LAW_PRINCIPLE: Dict[str, str]` maps each of the 16 residuals to exactly one
principle, and `principle_of(family) == LAW_PRINCIPLE[FAMILIES[family].law]`. Nothing to
drift, because there is nothing to type twice.

| principle | families | build cells | LikePhys row |
|---|---|---|---|
| **Spatio-temporal continuity** | continuity, non_parabolic | 21 | Temporal + Spatial Continuity |
| **Conservation of mass** | permanence, dissolve, fission, fusion | 38 | Conservation of Mass |
| **Conservation of momentum** | phantom_impulse, newton1_inertia, newton2_mass, newton3_reaction, angular_momentum | 29 | *(they fold this into Energy)* |
| **Conservation of energy** | superelastic, friction | 12 | Conservation of Energy |
| **Gravitation and support** | antigravity, global_gravity, support | 23 | *(only as Block Slide "hovering")* |
| **Solidity** | solidity | 9 | Spatial Continuity "invisible wall" |
| **Geometric invariance** | immutability, deformation | 28 | Geometric Invariance |
| **Optical consistency** | shadow, shadow_inverted, shadow_shape, colour_shift | 17 | Optical Consistency |
| | **23 families** | **177** | |

Every family lands in exactly one principle and the cells sum to the current 177 — this is a
*re-labelling*, not a change to what gets generated.

Two deliberate departures from LikePhys:

- **Momentum is split out from energy.** They put "momentum amplification" and "phantom
  force" under Conservation of Energy. Those are momentum violations; an elastic collision
  conserves both, and the two come apart precisely in the cases we stage (`newton2_mass`
  conserves momentum with a lying mass ratio, `superelastic` conserves momentum and creates
  energy). Merging them would make the single most useful distinction in the dataset
  invisible.
- **Gravitation is its own principle.** LikePhys has nowhere to put "the ball falls at 0.3 g",
  so hovering ends up under Material Response. Free fall is the single most-tested intuition
  in the infant-cognition literature and it deserves a row.

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

## Open questions — these are the decisions

1. **Eight principles, or fewer?** Folding `Solidity` (1 family, 9 cells) into
   Spatio-temporal continuity would match LikePhys, which files "invisible wall" under
   Spatial Continuity. Against: solidity is one of IntPhys 2's four core principles and one
   of the four things infants are tested on. Keeping it separate costs nothing except a row
   with one family in it.
2. **Does `friction` belong under energy?** Both `superelastic` and `friction` are staged so
   that energy *appears* — a bounce that returns more than it took, a body that accelerates
   against its own friction. That reading is consistent. The alternative is a `Material
   response` principle matching LikePhys, which would hold those two and would be where cloth
   and soft bodies land in Phase 3.
3. **Is `extent` worth promoting to a top-level axis**, or should it stay a field on the
   violation record? It only has two populated values today (`local`, `global`) and `group`
   only becomes real for the granular families.
4. **Keep `domain` as a deprecated alias for one release**, or delete it outright? Nothing
   has been published, so outright is defensible and much cleaner.
