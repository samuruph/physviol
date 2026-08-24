# Prior-art coverage matrix

Part 2 of [PLAN.md](PLAN.md) claims our taxonomy is "grounded in IntPhys 2 and LikePhys,
then extended". This file is what makes that claim auditable rather than rhetorical: which
prior-art cell each of our families covers, what v0 deliberately skips, and which families
are genuinely new.

Extracted from the papers in [`../context/`](../context/README.md) — arXiv:2506.09849
(IntPhys 2) and arXiv:2510.11512v3 (LikePhys). Scenario names and descriptions below are
quoted/paraphrased from those PDFs, not recalled.

---

## IntPhys 2

**Four core principles** for macroscopic objects: **Permanence**, **Immutability**,
**Spatio-Temporal Continuity**, **Solidity**. Violation-of-expectation framing, photorealistic
synthetic environments.

Axes it splits on, both of which we adopt:

| axis | values | our position |
|---|---|---|
| difficulty | Easy / Medium / Hard, plus a **Held Out** set | we use **weak / medium / strong** -- named for the *intervention*, not the task, because ours is derived from intervention magnitude rather than assigned by hand |
| camera | **Fixed** vs **Moving** | v0 ships orbit; MOVi's `linear_movement` camera mode gives us the moving tier cheaply (PLAN Part 0.5) |

**The numbers that set our difficulty target** (their Table 2/3, best run per model):

| | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| best MLLM (Gemini-2.5 Flash) | 64.4 | 56.8 | 54.5 | 55.6 |
| best predictive (V-JEPA 2-h) | 54.0 | 58.5 | 59.4 | 57.5 |
| **human** | 96.2 | 97.8 | 95.5 | **96.4** |

Models sit near chance while humans are near ceiling. That gap **is** the benchmark's
argument, and it is the calibration target for our Phase 1 check: if off-the-shelf models
score 95% on PhysViol, our violations are too coarse; at exactly 50% with no signal, they may
be unobservable rather than hard — which is precisely what `t_observable` lets us tell apart,
and which IntPhys 2 cannot distinguish.

> **Gap:** the per-principle *scene condition* lists (how many distinct setups per principle,
> and their names) were not cleanly recoverable from the PDF text layer. Not needed for v0
> mapping; fill in if we ever claim per-condition parity.

## LikePhys

**Twelve scenarios across four domains**: Rigid Body Mechanics, Continuum Mechanics, Fluid
Mechanics, Optical Effects. Each scenario ships a valid clip plus several invalid variants.
The paper also organizes violations under seven **physics laws**: Temporal Continuity,
Spatial Continuity, Conservation of Energy, Conservation of Mass, Geometric Invariance,
Optical Consistency, Material Response.

| # | scenario | domain | invalid variants (theirs) |
|---|---|---|---|
| 1 | **Ball Collision** | Rigid Body | altered restitution (inelastic sticking / super-elastic amplification), inter-penetration, phantom forces, radius change mid-impact, teleport-through, temporal disorder |
| 2 | **Ball Drop** | Rigid Body | colour change mid-flight, dynamic rescaling in free fall, over-bounce, floor penetration, teleport to another height, temporal disorder |
| 3 | **Block Slide** | Rigid Body | hovering above the plane, non-Newtonian erratic motion, spurious jitter, dimension change while sliding, teleport down-slope, temporal disorder |
| 4 | **Pyramid Impact** | Rigid Body | amplified/damped collision energy, holes in the pyramid (mass continuity), teleport-through, negated gravity |
| 5 | **Pendulum Oscillation** | Rigid Body | rod breaks mid-swing, bob disappears, trajectory leaves the circular path, time freezes, length/frequency varies |
| 6 | **Cloth Drape** | Continuum | colour change, cylinder/ground penetration, impossible folds, rigid-sheet behaviour with no flutter, temporal disorder |
| 7 | **Cloth Waving** | Continuum | sections freeze, fragments shatter, parts teleport, impossible 180° twist, outward explosion, jump cuts |
| 8 | **Droplet Fall** | Fluid | antigravity rise, stream fragments into blobs, mass created/removed, negative or oscillating viscosity, particle self-attraction, temporal disorder |
| 9 | **Faucet Flow** | Fluid | colour shift, non-coalescing fracture, negative viscosity, arbitrary mass injection/removal, instant phase shifts, self-attraction, droplet teleport, temporal disorder |
| 10 | **River Flow** | Fluid | fracture into isolated droplets, invisible barrier, mass vanishing, liquid→solid→liquid phase shifts, timestamp jumps |
| 11 | **Moving Shadow** | Optical | shadow inverts onto the ceiling, vanishes, appears without a caster, shape mismatches the object, teleports, temporal disorder |
| 12 | **Orbit Shadow** | Optical | inverts direction/plane, vanishes mid-orbit, detaches from caster, geometry distorts, teleports along path, temporal disorder |

---

## Coverage matrix — PhysViol v0 against both

| PhysViol family (PLAN Part 2) | IntPhys 2 principle | LikePhys scenario / law | status |
|---|---|---|---|
| Permanence | Permanence | Pendulum (bob disappears), Pyramid (holes) / Conservation of Mass | **covered** |
| Immutability | Immutability | Ball Drop (colour, rescale), Block Slide (dimensions) / Geometric Invariance | **covered** |
| Colour shift | Immutability | Ball Drop "colour change mid-flight", Cloth Drape "colour change" | **covered**, and continuous where theirs is a swap: severity is CIE-Lab distance, so a hue nudge and a jump to the opposite side of the wheel are on one scale |
| Dissolve (fades to nothing) | Permanence (the disappearance half) | none — every prior benchmark's disappearance is a switch | **covered, but continuous where theirs is discrete**: the same end state reached as a trend rather than a single-frame cut |
| Fusion (two become one) | Permanence (the disappearance half) | Cloth/fluid coalescence — both continuum | **covered for rigid bodies** |
| Fission (one body becomes two) | Permanence (the appearance half) | Cloth Waving "fragments shatter", Faucet "non-coalescing fracture" — both continuum | **covered for rigid bodies**, where neither benchmark stages it |
| Continuity | Spatio-Temporal Continuity | teleport variants across nearly every scenario / Spatial Continuity | **covered** |
| Solidity | Solidity | Ball Collision & Ball Drop penetration / Spatial Continuity | **covered** |
| Anti-gravity (per-body `αg`) | — | Pyramid (negated gravity), Droplet Fall (antigravity) | covered, but **we make it continuous** in α |
| Phantom impulse | — | Ball Collision (phantom forces) | covered, **continuous** in `J/(m·v_typ)` |
| Super-elastic (`e>1`) | — | Ball Drop over-bounce, Ball Collision / Conservation of Energy | covered, **continuous** in `e` |
| **Non-parabolic flight** | — | nearest is Block Slide "erratic motion" (discrete) | **new** — continuously dialable by RMS deviation from the fitted parabola; our clean control (`t_event ≈ t_observable`) |
| **Newton-1 / inertia** | — | partially Block Slide (hovering) | **new** as an explicit family |
| **Newton-2 / mass–acceleration** | — | none | **new** — no prior benchmark dials effective mass |
| **Newton-3 / action–reaction** | — | none | **new** — and the only family whose mask must span *two* bodies |
| **Angular momentum** | — | Pendulum (frequency variation) is adjacent | **new** as a conservation-law family |
| **Support / static equilibrium** | — | none | **new** — severity = clearance above the surface actually beneath the body |
| **Friction inversion** | — | Block Slide is the setting, not the violation | **new** — severity = effective µ vs declared µ |
| **Shadow / optical** (position) | — | Moving Shadow, Orbit Shadow / Optical Consistency | covered as a category, but **ours is the only one with a mask that is not on the object** |
| **Shadow inverted** | — | Moving Shadow "shadow inverts onto the ceiling" | covered, and the **only family in v0 that is wrong from frame 0** — nothing changes mid-clip, so there is no moment to catch and a model has to compare the shadow's bearing against the key light |
| **Shadow shape** | — | Moving Shadow "shape mismatches the object" | covered, and **split from shadow position on purpose** — a benchmark asking whether a model tracks where a shadow goes should not be scored on clips where it is also the wrong shape |
| **Deformation** | — | nearest is Cloth "impossible folds", which is continuum | **new** for rigid bodies — the path stays lawful and only the proportions change |
| **Global gravity scale** | — | none | **new** — internally consistent, mask is the whole frame, no localized culprit |
| Surprising-but-valid control | — | none | **new** — separates "weird" from "illegal" |
| Artifact / shortcut probe | debug split | — | adopted from IntPhys 2 |

### What v0 deliberately skips

| skipped | why |
|---|---|
| **Continuum** (Cloth Drape, Cloth Waving) | PyBullet exposes only `loadSoftBody`/`createSoftBodyAnchor` and Kubric wraps neither; no trustworthy per-element residual at v0. Deferred to **Phase 3** with the MuJoCo/MJX backend. |
| **Fluid** (Droplet Fall, Faucet Flow, River Flow) | **Tested, not assumed.** Blender 2.93.4 in our image *does* ship Mantaflow, but headless scripted baking fails (`NameError: liquid_save_data_N` → `Manta::Error`), Kubric exposes no fluid objects, and a liquid's per-frame mesh state does not fit the pose-based seam. Partially substituted at v0 by the **`pour`** scenario — a few dozen rigid grains, honestly labelled `physics_medium: "granular"`. True fluid is **Phase 3**. See PLAN Part 2. |
| **Temporal disorder** (frame shuffling / jump cuts) | Used by nearly every LikePhys scenario, and we intentionally omit it: it is an *encoding* artifact, not a physics violation. It has no law residual, and including it would reward exactly the shortcut detection our artifact-probe control exists to catch. |

### What this table is claiming

Of the 23 violation families in v0, **11 map cleanly onto prior art** (Permanence, Dissolve,
Immutability, Fission, Fusion, Continuity, Solidity, Colour shift, Shadow, Shadow shape,
Shadow inverted), **3 exist there only as discrete flags** where we make them continuous
(anti-gravity, phantom impulse, super-elastic), and **9 are new**. The novelty is not
primarily in the taxonomy anyway — it is that every one of these ships with a mask, three
clocks and a residual. `tests/test_taxonomy.py` asserts these three counts, so the table
cannot drift from the code without something going red.

`fission` is counted as covered rather than new because it is the appearance half of IntPhys
2's permanence principle — an object arriving where there was none. What is new is staging it
on a *rigid* body: both benchmarks only break object count in continuum media, where nothing
they ship can localise which pixels are wrong.
