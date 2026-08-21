# PhysViol `meta.json` schema

Versioned separately from [PLAN.md](PLAN.md) because the schema will evolve while the design
does not. The machine-readable copy lives at `physviol/schema/meta.schema.json`; this file is
the human reference. `physviol validate` enforces both this schema and the cross-checks at
the bottom.

**Schema version: 0 (draft — frozen at the end of Phase 1).**

## Example

```json
{
  "clip_uid": "physviol_v0/rigid_ramp/0173/invalid_solidity_a",
  "pair_uid": "physviol_v0/rigid_ramp/0173",  "twin_uid": "physviol_v0/rigid_ramp/0173/valid",
  "label": "invalid",
  "category": "solidity", "intphys2_category": "solidity", "likephys_domain": "rigid_body",
  "scenario": "rigid_ramp", "seed": 91731,
  "fps": 24, "num_frames": 96, "resolution": [512, 512],
  "camera": {"intrinsics": [], "extrinsics_per_frame": [], "motion": "orbit"},
  "violation": {
    "kind": "sustained",
    "t_event_frame": 38, "t_observable_frame": 51, "observability_lag_frames": 13,
    "occluded_at_event": true, "t_end_frame": 62,
    "causal_body_ids": [3, 5],
    "intervention": {"type": "disable_collision_pair", "params": {"pair": [3, 5]},
                     "magnitude": 0.043, "magnitude_unit": "m_penetration_depth",
                     "severity_bin": "medium"},
    "consequences": [{"body_id": 7, "t_diverge_frame": 55, "t_observable_frame": 57,
                      "displacement_m": 0.21, "relation": "struck_by_causal_body"}],
    "peak_residual": {"law": "penetration", "value": 0.043, "z_vs_valid": 41.2, "frame": 49}
  },
  "controls": {"is_surprising_but_valid": false, "is_artifact_probe": false},
  "assets": [{"name": "...", "source": "polyhaven", "license": "CC0"}],
  "provenance": {"generator_commit": "…", "kubric_image_digest": "sha256:…",
                 "blender_version": "…", "render_seed": 91731,
                 "prefix_identical_verified": true, "prefix_identical_upto_frame": 38},
  "real2sim": null
}
```

## Fields

### Identity

| field | type | notes |
|---|---|---|
| `clip_uid` | str | globally unique; also the directory name |
| `pair_uid` | str | shared by a valid/invalid twin pair |
| `twin_uid` | str | the counterpart clip. **Never split a pair across train/val/test.** |
| `label` | `"valid"` \| `"invalid"` | clip-level ground truth |
| `scenario` | str | scene generator name, e.g. `rigid_ramp` |
| `seed` | int | the seed for the whole sampling + render path |

### Categorization

| field | type | notes |
|---|---|---|
| `category` | str | our family, e.g. `solidity`, `non_parabolic_flight`, `newton3` |
| `intphys2_category` | str \| null | nearest IntPhys 2 principle, or null if genuinely new |
| `likephys_domain` | str \| null | nearest LikePhys domain |

Mapping rationale is tabulated in [prior_art.md](prior_art.md). A null is a claim of novelty
and must be justified there.

### Clip properties

`fps` (int), `num_frames` (int), `resolution` (`[H, W]`), and `camera` with `intrinsics`,
`extrinsics_per_frame` (length `num_frames`) and `motion` (`static` | `orbit` | `linear`).

### `violation` — null on valid clips

| field | type | notes |
|---|---|---|
| `kind` | `instant` \| `sustained` \| `repeated` | shape of the intervention in time |
| `t_event_frame` | int | when the law breaks **in simulator state**. Exact by construction. |
| `t_observable_frame` | int | first frame with **visual evidence**. Exact, via prefix identity. |
| `observability_lag_frames` | int | `t_observable − t_event`. The occlusion-lag metric. |
| `occluded_at_event` | bool | whether the culprit was hidden when the law broke |
| `t_end_frame` | int | last frame the violation is active |
| `causal_body_ids` | int[] | instance ids in `seg.npz`. **Newton-3 requires both bodies.** |
| `intervention` | object | `type`, `params`, `magnitude`, `magnitude_unit`, `severity_bin` |
| `consequences` | object[] | per downstream body: `body_id`, `t_diverge_frame`, `t_observable_frame`, `displacement_m`, `relation` |
| `peak_residual` | object | `law`, `value` (physical units), `z_vs_valid`, `frame` |

**Invariant:** `t_event_frame ≤ t_observable_frame ≤ t_end_frame`.

`magnitude` is the knob we turned (exact); `peak_residual.value` is the measured consequence.
`severity_bin` (`easy`/`medium`/`hard`) is derived from `magnitude`, not from model
performance — splits are principled, not tuned.

### `controls`

`is_surprising_but_valid` and `is_artifact_probe`. Both are `label: "valid"`; both exist so a
detector cannot score by flagging weirdness or by keying on the renderer.

### `assets`

Array of `{name, source, license}`. **`license` is mandatory and non-empty** — `physviol
validate` fails otherwise. This is what makes the Phase 5 audit a check rather than an
archaeology project.

### `provenance`

`generator_commit`, `kubric_image_digest`, `blender_version`, `render_seed`,
`prefix_identical_verified` (bool), `prefix_identical_upto_frame` (int).

`prefix_identical_verified` is the dataset's central integrity claim: valid and invalid
renders are **bit-identical** before `t_event`. Consumers relying on pixel-aligned twins
should check this flag rather than assuming it.

### `real2sim`

`null` in v0. Reserved for Phase 4:
`{real_twin_uid, capture_rig, sysid_params, twin_gap_lpips, twin_gap_reproj_px}`. Reserving
the key now means Phase 4 does not force a regeneration.

## Array files

| file | dtype / shape | notes |
|---|---|---|
| `seg.npz` | uint16 `[T,H,W]` | instance ids; `causal_body_ids` index into these |
| `depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords` | `[T,H,W,·]` | straight from Kubric's exporters |
| `violation_mask.npz` | bool `[T,H,W]` | causal bodies **while the violation is active** |
| `causal_mask.npz` | uint8 `[T,H,W]` | 0 = none, k = consequence index k |
| `severity_map.npz` | f16 `[T,H,W]` | residual painted into the culprit's instance mask |
| `divergence_map.npz` | f16 `[T,H,W]` | **NOT the violation region** — see below |
| `grids.npz` | see below | pre-reduced token grids |
| `traj.npz`, `residuals.npz` | — | the seam file and raw per-law residuals, shipped |

> **`divergence_map` warning.** This is `|valid − invalid|` in pixel space. After `t_event` a
> twin pair diverges *everywhere* downstream — shadows, contact chains, secondary collisions
> — so the pixels that differ are **not** the pixels where the violation is. A
> different-but-valid rollout diverges too. It is shipped for analysis and it is **not a
> training target**. Use `violation_mask` / `severity_map`.

### `grids.npz`

Masks and severity pre-reduced to the latent token grid, so a consumer never re-derives the
tokenizer's binning.

- `mask_21x16x16` — bool `[21,16,16]`, the grid a WAN-2.1-class video DiT yields for an
  81-frame 256² clip (time binned 4×: 81 → 21; space 16×: 256 → 16)
- `severity_max_21x16x16`, `severity_mean_21x16x16` — f16. Peak and average are different
  questions; both are stored.
- `mask_Tx32x32`, `severity_{max,mean}_Tx32x32` — for consumers on a different tokenizer

**Ordering guarantee:** time-major, `[F_lat, H_lat, W_lat]` with the latent frame slowest.
This is a schema guarantee, not an implementation detail — flattening to `[F_lat·H_lat·W_lat]`
must match a transformer's token order.

**Reduction rule:** masks reduce by **max** (a violation in any contributing source frame
marks the latent frame); severity by both max and mean.

## Cross-checks enforced by `physviol validate`

1. jsonschema conformance against `meta.schema.json`
2. `t_event_frame ≤ t_observable_frame ≤ t_end_frame`
3. `violation_mask` is non-empty on every frame in `[t_event, t_end]`
4. every id in `causal_body_ids` actually appears in `seg.npz`
5. every entry in `assets` carries a non-empty `license`
6. `provenance.prefix_identical_verified` is true
7. `violation` is null iff `label == "valid"`
8. `twin_uid` resolves to an existing clip whose `pair_uid` matches
