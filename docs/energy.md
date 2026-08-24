# Energy annotation

**Status: designed and measured, implementation in progress.** Numbers here are from real
PyBullet rollouts at tier `debug`, not estimates.

## Why

The taxonomy's third principle is *conservation of momentum and energy*, and until now the
only thing scoring it was `energy_at_contact`, which fires on one family. Energy is
computable **exactly** from the trajectory seam, for both twins, at negligible cost -- so it
can be both a descriptive channel a benchmark consumer reads and a second, independent
scorer for that principle.

It also cross-checks the taxonomy. `deformation` is volume-preserving and reads **0.00%**;
`immutability` scales volume and reads **+34%**. That is the mass-vs-shape split of
[taxonomy_v2.md](taxonomy_v2.md) showing up in an independent measurement.

## What energy

Total **mechanical** energy of the dynamic bodies:

```
E(t) = Σ_b  ½·m_b(t)·|v_b(t)|²  +  ½·ω_b(t)ᵀ·I_b(t)·ω_b(t)  +  m_b(t)·g·h_b(t)
```

- `v`, `ω`, `pos`, `quat` come straight off the trajectory. No estimation, no fitting.
- `h` is height along `-ĝ` from the scene's `floor_level`, so the datum is fixed per scene
  and absolute values are comparable between twins.
- `I` is the analytic inertia tensor of the primitive -- `⅖mr²` for a sphere,
  `⅓m(b²+c²)` per axis for a box of half-extents `(a,b,c)` -- rotated into world frame by
  the body's quaternion.
- Bodies with `present == False` contribute nothing. That is what makes `permanence` read
  −100%.

**Mass follows volume**: `m_b(t) = m_b(0) · Π scale_mul[t,b]`. This is a deliberate choice
and it is the one that keeps the annotation consistent with the taxonomy. A body that
doubles in size is depicting twice the matter, so it must carry twice the energy;
`deformation`'s squash is volume-preserving, so its product is exactly 1 and its energy does
not move. Every other family leaves `scale_mul` at 1 and is unaffected either way.

## The two channels

A passive rigid-body scene under gravity, contacts and friction can only **lose** mechanical
energy, and only **at a contact**. That gives two independent anomaly channels, and the
measurement shows they separate different families rather than restating each other.

### 1. Free-energy anomaly -- ΔE on frames with no contact

Energy that changes with nothing touching. No budget model needed; the lawful value is zero.

| | free-loss | free-gain |
|---|---|---|
| **valid twins** | **0.00%** | **0.00%** |
| solidity | 146% | 61% |
| non_parabolic | 48% | 32% |
| support | 47% | — |
| antigravity | 25% | 13% |
| angular_momentum | — | 12% |
| immutability | — | 9% |
| phantom_impulse | 9% | — |

### 2. Contact-budget anomaly -- ΔE at a contact beyond what is allowed

Energy may be lost at a contact, but only within a budget:

- **Restitution.** Normal-direction kinetic energy after a bounce is at most `e²` times
  before it. Gaining any is impossible. `superelastic` reads **−190%** here (a *gain*) while
  sitting at 0.06% in the free channel -- the exact inverse of `support`.
- **Friction.** Frictional dissipation over a step is bounded by `µ·m·g·Δs` for a body on a
  surface -- textbook, and it needs only µ, m, g and path length, all of which the seam
  carries exactly.

Contact *count* is not a signal on its own: valid clips dissipate 27% of `E₀` at contacts,
which is simply what inelastic collisions do. Only the excess over budget means anything.

## Noise floor

The largest frame-to-frame energy **increase** on a valid clip is **0.005% of E₀** -- that is
integrator drift and nothing else. Every violating family above is two to five orders of
magnitude clear of it, so this scores without a tuned threshold. It still goes through
`bounded_score(baseline=r_valid)` like every other law, so a scenario with unusual drift is
scored against its own twin rather than against a global constant.

## What ships

| file | contents |
|---|---|
| `energy.npz` | `total[T]`, `kinetic_translational[T]`, `kinetic_rotational[T]`, `potential[T]`, `by_body[T,B]`, `body_ids[B]`, `dissipated[T]`, `free_anomaly[T]`, `contact_anomaly[T]` |
| `energy_map.npz` | `energy[T,H,W]` -- each body's energy painted onto its pixels through the segmentation pass, the same mechanism `severity_map` already uses |
| `meta.json` | an `energy` block: `E0`, `E_end`, `peak_free_anomaly`, `peak_contact_anomaly`, `total_dissipated`, all normalised by `E₀` as well as in joules |

`energy_map` is per-body constant within a body's silhouette. For a rigid body that is the
honest spatial resolution -- energy is not a field inside a rigid body, and pretending
otherwise would invent structure the physics does not have.

## Shipped for every family, including the ones that should read zero

A flat trace on `colour_shift` is a **positive, falsifiable statement**: this violation
provably does not touch energy. Omitting it would turn a measured fact into an untested
assumption, and it is what lets the orthogonality test check energy bidirectionally the way
it already checks the exclusive laws.

## Known limits

- **Scripted scenarios.** `pendulum_swing` and `shadow_track` drive a body kinematically, so
  the conservation invariant does not apply to those bodies and the clip-level claim must
  exclude them. Handled by flagging, not by hiding the numbers.
- **PyBullet's `impulse` field may be a force.** Noted in CLAUDE.md and the cause of an
  earlier residual bimodality. The free-energy channel deliberately needs no force at all,
  and the restitution budget uses velocities rather than the reported impulse, so neither
  depends on resolving this.
